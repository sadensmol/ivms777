import sqlite3
from collections.abc import Callable

from captioning.base import Captioner
from embedding.store import write_caption_vector
from embedding.vectors import l2_normalize
from inference.client import InferenceCancelled, InferenceClient
from ingest.jobs import enqueue
from ingest.taxonomy import reindex_fts
from ingest.thumbs import thumb_key
from ingest.vocab import tag_id_map
from ingest.worker import Preempted, StageHandler
from storage.base import Storage


def _embed_caption(
    conn: sqlite3.Connection, client: InferenceClient, embed_model: str, photo_id: int, caption: str
) -> None:
    """Embed the caption text and store it (§9). Best-effort: an embeddings backend
    that is down must not fail the caption itself — the vector backfills later."""
    if not caption:
        return
    try:
        vector = client.embed(embed_model, [caption])[0]
    except Exception:  # noqa: BLE001 - embeddings are an enhancement, not required
        return
    write_caption_vector(conn, photo_id, l2_normalize(vector))


def caption_handler(
    derived: Storage,
    captioner: Captioner,
    embed_client: InferenceClient,
    embed_model: str,
    dimensions: list[str],
    detail_px: int,
    should_preempt: Callable[[], bool] = lambda: False,
) -> StageHandler:
    """The caption stage: one call per photo through the `Captioner` adapter
    (design §4) -> caption, AI title/description, and vlm tags — and the
    caption's own text embedding for semantic similarity (§9), computed here
    while the caption is fresh, over `embed_client` (Ollama, unchanged even on
    jetson). Drains last (§8). A captioner error raises so the queue retries the
    job rather than writing half a row.

    The captioner call is cancellable: `should_preempt` is threaded through, so
    an interactive request (chat, memory rebuild) aborts an in-flight caption
    within a chunk instead of waiting out the whole call (§8.1). An abort raises
    `InferenceCancelled`, mapped here to `Preempted`, which leaves the photo's
    job PENDING for the next pass — never a burnt retry.
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
        try:
            result = captioner.caption(image, dimensions, should_preempt=should_preempt)
        except InferenceCancelled as cancelled:
            raise Preempted() from cancelled   # yield the models; photo stays pending

        conn.execute(
            "UPDATE photos SET caption=?, caption_model=?, ai_title=?, ai_description=? WHERE id=?",
            (result.caption, captioner.caption_model, result.title, result.description, photo_id),
        )
        _write_vlm_tags(conn, photo_id, result.tags)
        _embed_caption(conn, embed_client, embed_model, photo_id, result.caption)
        reindex_fts(conn, photo_id)

    return handle


def backfill_caption_vectors(
    conn: sqlite3.Connection, client: InferenceClient, embed_model: str, limit: int = 50
) -> int:
    """Embed captions that predate the caption-vector column (§9), a few per drain
    so an already-captioned library gains semantic similarity without a full
    re-caption. Returns how many were embedded."""
    rows = conn.execute(
        "SELECT id, caption FROM photos WHERE caption IS NOT NULL AND caption_vec IS NULL"
        " LIMIT ?",
        (limit,),
    ).fetchall()
    for row in rows:
        _embed_caption(conn, client, embed_model, row["id"], row["caption"])
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
