import sqlite3
from collections.abc import Callable

from embedding.caption_text import embed_caption_texts
from embedding.store import write_caption_vector
from embedding.vectors import l2_normalize
from inference.client import InferenceClient
from inference.models_client import ModelsClient
from ingest.jobs import enqueue
from ingest.taxonomy import reindex_fts
from ingest.thumbs import thumb_key
from ingest.vocab import tag_id_map
from ingest.worker import StageHandler
from storage.base import Storage


def caption_handler(
    derived: Storage,
    models_client: ModelsClient,
    dimensions: list[str],
    detail_px: int,
    should_preempt: Callable[[], bool] = lambda: False,
) -> StageHandler:
    """The caption stage: one HTTP call per photo to the `models` service's
    `/caption` (design §5.1, plan 15 task 3) -> caption, AI title/description, and
    vlm tags, written straight to the DB. It does NOT embed the caption: the
    caption-vector (§9) is a SEPARATE, batched step (`backfill_caption_vectors`,
    pipeline group 2c) that embeds captions with the dedicated text embedder
    (`nomic-embed-text`, in-process in the `models` service) — a different model,
    so it is not interleaved with the caption call per photo. Drains last (§8). A
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
        result = models_client.caption(image, dimensions)

        conn.execute(
            "UPDATE photos SET caption=?, caption_model=?, ai_title=?, ai_description=? WHERE id=?",
            (result["caption"], result["model"], result["title"], result["description"], photo_id),
        )
        _write_vlm_tags(conn, photo_id, result["tags"])
        reindex_fts(conn, photo_id)

    return handle


def backfill_caption_vectors(
    conn: sqlite3.Connection, client: InferenceClient, model: str, limit: int = 50
) -> int:
    """Embed captions that have no caption-vector yet (§9) with the dedicated text
    embedder `model` (`nomic-embed-text`, design §4) — this is the ONLY place caption
    text is embedded (the caption stage just writes the text). Embedded in ONE batch
    so the embedder loads once, a few per drain, so a freshly-captioned or pre-existing
    library gains semantic similarity without a re-caption. Best-effort: an embedder
    that is down must not fail the pass — the vectors retry next drain. Returns how
    many were embedded."""
    rows = conn.execute(
        "SELECT id, caption FROM photos WHERE caption IS NOT NULL AND caption_vec IS NULL"
        " LIMIT ?",
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
    return len(rows)


def _write_vlm_tags(conn: sqlite3.Connection, photo_id: int, tags: dict) -> None:
    ids = tag_id_map(conn)
    conn.execute(
        "DELETE FROM photo_tags WHERE photo_id = ? AND source = 'vlm'", (photo_id,)
    )
    rows = [
        (photo_id, ids[(dimension, label)], 1.0, "vlm")
        for dimension, labels in tags.items()
        for label in labels
        if (dimension, label) in ids  # only labels that are in the vocabulary
    ]
    conn.executemany(
        "INSERT INTO photo_tags(photo_id, tag_id, score, source) VALUES (?, ?, ?, ?)"
        " ON CONFLICT(photo_id, tag_id, source) DO UPDATE SET score = excluded.score",
        rows,
    )


def backfill_captions(conn: sqlite3.Connection) -> int:
    """Queue a caption job for every photo that has none yet (self-healing)."""
    rows = conn.execute(
        "SELECT id FROM photos WHERE thumb_key IS NOT NULL AND caption IS NULL"
        " AND NOT EXISTS (SELECT 1 FROM jobs j WHERE j.photo_id = photos.id AND j.stage = 'caption')"
    ).fetchall()
    for row in rows:
        enqueue(conn, row["id"], "caption")
    return len(rows)
