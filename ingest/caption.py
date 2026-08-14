import json
import sqlite3

from embedding.store import write_caption_vector
from embedding.vectors import l2_normalize
from inference.client import InferenceClient, encode_image
from inference.prompts import CAPTION_SCHEMA, caption_messages
from ingest.jobs import enqueue
from ingest.taxonomy import reindex_fts
from ingest.thumbs import thumb_key
from ingest.vocab import tag_id_map
from ingest.worker import StageHandler
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
    client: InferenceClient,
    model: str,
    embed_model: str,
    dimensions: list[str],
    detail_px: int,
) -> StageHandler:
    """The caption stage: one VLM call per photo -> caption, AI title/description,
    and vlm tags — and the caption's own text embedding for semantic similarity
    (§9), computed here while the caption is fresh. Drains last (§8). Invalid JSON
    raises so the queue retries the job rather than writing half a row.
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
        messages = caption_messages(model, encode_image(image), dimensions)
        obj = json.loads(client.complete(model, messages, json_schema=CAPTION_SCHEMA))

        conn.execute(
            "UPDATE photos SET caption = ?, caption_model = ?, ai_title = ?, ai_description = ?"
            " WHERE id = ?",
            (obj["caption"], model, obj["title"], obj["description"], photo_id),
        )
        _write_vlm_tags(conn, photo_id, obj.get("tags") or {})
        _embed_caption(conn, client, embed_model, photo_id, obj["caption"])
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
