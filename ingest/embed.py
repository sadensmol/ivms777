import sqlite3
from pathlib import Path

from PIL import Image, ImageOps

from embedding.base import Embedder
from embedding.store import write_vector
from embedding.vectors import l2_normalize
from ingest.jobs import enqueue
from ingest.worker import StageHandler
from storage.base import Storage


def backfill_embeds(conn: sqlite3.Connection) -> int:
    """Queue an embed job for every photo that has no vector yet.

    Photos ingested before the embed stage existed carry no `embedding_model`.
    This makes the pipeline self-healing: on the next drain they get embedded and
    become searchable, with no manual migration. Idempotent — `enqueue` skips
    stages already queued.
    """
    rows = conn.execute(
        "SELECT id FROM photos WHERE embedding_model IS NULL"
    ).fetchall()
    for row in rows:
        enqueue(conn, row["id"], "embed")
    return len(rows)


def embed_handler(originals: Storage, embedder: Embedder, model_name: str) -> StageHandler:
    def handle(conn: sqlite3.Connection, photo_id: int) -> None:
        row = conn.execute(
            "SELECT storage_key FROM photos WHERE id = ?", (photo_id,)
        ).fetchone()
        source: Path | None = originals.local_path(row["storage_key"])
        if source is None or not source.is_file():
            raise FileNotFoundError(row["storage_key"])
        with Image.open(source) as image:
            image.load()
            upright = ImageOps.exif_transpose(image).convert("RGB")
        vector = l2_normalize(embedder.embed_images([upright])[0])
        write_vector(conn, photo_id, vector)
        conn.execute(
            "UPDATE photos SET embedding_model = ? WHERE id = ?", (model_name, photo_id)
        )

    return handle
