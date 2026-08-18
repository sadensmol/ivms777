import sqlite3
from collections.abc import Callable

from embedding.caption_text import embed_caption_texts
from embedding.store import write_caption_vector
from embedding.vectors import l2_normalize
from inference.client import InferenceClient
from inference.models_client import ModelsClient
from ingest.jobs import complete, enqueue, record_done_at, reprocess_one
from ingest.taxonomy import reindex_fts
from ingest.thumbs import thumb_key
from ingest.worker import StageHandler
from storage.base import Storage


def caption_handler(
    derived: Storage,
    models_client: ModelsClient,
    detail_px: int,
    should_preempt: Callable[[], bool] = lambda: False,
) -> StageHandler:
    """The caption stage: one HTTP call per photo to the `models` service's
    `/caption` (design §5.1, plan 15 task 3) -> caption + AI title/description,
    written straight to the DB. It writes NO tags — the caption model no longer
    picks from the vocabulary; tags are owned by the SigLIP taxonomy stage (§7).
    It does NOT embed the caption: the caption-vector (§9) is its OWN stage,
    `caption_embed`, drained in one batch right after this one (pipeline group 2c,
    `backfill_caption_vectors`) with the dedicated text embedder (`nomic-embed-text`,
    in-process in the `models` service) — a different model, so it is not
    interleaved with the caption call per photo. Drains last (§8). A
    `models_client.caption` error raises so the queue retries the job rather than
    writing half a row.

    `should_preempt` is still threaded through for the stage's own between-photo
    yield point (`ingest.worker.drain`). Since plan 16 captioning is a remote
    OpenAI call to llama-server (not an in-process GPU model), there is nothing to
    preempt mid-call — SigLIP is the only heavy in-process model left (§8.1).
    """

    def handle(conn: sqlite3.Connection, photo_id: int) -> None:
        row = conn.execute(
            "SELECT content_hash, thumb_key FROM photos WHERE id = ?", (photo_id,)
        ).fetchone()
        if row["thumb_key"] is None:
            return  # no thumbnail yet -> nothing to caption; skip cleanly, don't crash
        detail_key = thumb_key(row["content_hash"], detail_px)
        # Prefer the detail thumbnail; fall back to the grid one the photo already has.
        image_key = detail_key if derived.exists(detail_key) else row["thumb_key"]
        image = derived.read(image_key)
        result = models_client.caption(image)

        conn.execute(
            "UPDATE photos SET caption=?, caption_model=?, ai_title=?, ai_description=? WHERE id=?",
            (result["caption"], result["model"], result["title"], result["description"], photo_id),
        )
        reindex_fts(conn, photo_id)
        # A NEW caption needs a NEW caption vector: queue the `caption_embed` stage
        # (idempotent — the row usually exists from upload) and reset it, so a
        # re-caption cannot leave the old text's vector behind.
        reprocess_one(conn, photo_id, "caption_embed")

    return handle


def backfill_caption_vectors(
    conn: sqlite3.Connection, client: InferenceClient, model: str, limit: int = 50
) -> int:
    """Drain the `caption_embed` stage: embed the pending photos' captions (§9) with
    the dedicated text embedder `model` (`nomic-embed-text`, design §4) — the ONLY
    place caption text is embedded (the caption stage just writes the text).

    **The queue says what to embed, the vector column does not.** Driving it off
    `caption_vec IS NULL` meant a re-caption silently kept the old text's vector, and
    the work was invisible to the UI. Now it claims pending `caption_embed` jobs, so
    the stage shows its own done/pending/throughput next to the others and
    Reprocess re-embeds exactly like every other stage.

    Still ONE batched call so the embedder loads once, a few per drain, so a freshly
    captioned or pre-existing library gains semantic similarity without a re-caption.
    Best-effort: an embedder that is down must not fail the pass — the jobs stay
    pending and retry next drain. Returns how many were embedded."""
    rows = conn.execute(
        "SELECT p.id, p.caption FROM photos p JOIN jobs j ON j.photo_id = p.id"
        " WHERE j.stage = 'caption_embed' AND j.status = 'pending' AND p.caption IS NOT NULL"
        " ORDER BY p.id LIMIT ?",
        (limit,),
    ).fetchall()
    rows = [r for r in rows if r["caption"]]
    if not rows:
        return 0
    try:
        vectors = embed_caption_texts(
            client, model, [r["caption"] for r in rows], is_query=False
        )
    except Exception:  # noqa: BLE001 - embeddings are an enhancement, not required
        return 0
    for row, vector in zip(rows, vectors):
        write_caption_vector(conn, row["id"], l2_normalize(vector))
        complete(conn, row["id"], "caption_embed")
    return len(rows)


def backfill_caption_embeds(conn: sqlite3.Connection) -> int:
    """Give every photo predating the `caption_embed` stage its job row (self-healing,
    the mirror of `backfill_captions`). Returns how many were queued to embed.

    A photo that ALREADY has a vector gets a row marked `done`, not just skipped:
    otherwise a library embedded before this stage existed showed "0 done · 0
    pending" beside four rows reading "206 done", which looks like a broken stage
    rather than a finished one.
    """
    missing = conn.execute(
        "SELECT p.id, p.caption_vec IS NOT NULL AS embedded,"
        "       COALESCE((SELECT j.updated_at FROM jobs j"
        "                 WHERE j.photo_id = p.id AND j.stage = 'caption'), p.updated_at) AS done_at"
        " FROM photos p"
        " WHERE (p.caption IS NOT NULL OR p.caption_vec IS NOT NULL) AND NOT EXISTS ("
        "   SELECT 1 FROM jobs j WHERE j.photo_id = p.id AND j.stage = 'caption_embed')"
    ).fetchall()
    queued = 0
    for row in missing:
        if row["embedded"]:
            # Dated when the caption was written — that is when the vector followed.
            record_done_at(conn, row["id"], "caption_embed", row["done_at"])
        else:
            enqueue(conn, row["id"], "caption_embed")
            queued += 1
    return queued


def backfill_captions(conn: sqlite3.Connection) -> int:
    """Queue a caption job for every photo that has none yet (self-healing)."""
    rows = conn.execute(
        "SELECT id FROM photos WHERE thumb_key IS NOT NULL AND caption IS NULL"
        " AND NOT EXISTS (SELECT 1 FROM jobs j WHERE j.photo_id = photos.id AND j.stage = 'caption')"
    ).fetchall()
    for row in rows:
        enqueue(conn, row["id"], "caption")
    return len(rows)
