import sqlite3
from pathlib import Path

import pillow_heif
from PIL import Image, ImageOps

# Apple HEIC is a first-class source format, and this stage opens the ORIGINAL.
# Registering here rather than relying on `ingest.exif` / `ingest.thumbs` having
# been imported first: the opener is global and registration is idempotent, so
# the cost is nothing and the stage stops depending on import order.
pillow_heif.register_heif_opener()

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


def relabel_legacy_embeds(conn: sqlite3.Connection, model_name: str) -> int:
    """Repair `embedding_model` stamps that name no model (design §4.1).

    The stage used to stamp `settings.embed_model_name` — an HF repo id, not a
    catalog key — so every photo claimed `siglip2-so400m-patch14-384` on the photo
    page whatever the `image_embed` slot held. The VECTORS were always the slot's
    (the `models` service picks the model), so only the label is wrong, and a
    re-embed of the whole library would be a pointless way to fix a string.

    A label that is a catalog key is left alone: it is a real answer, and a photo
    still carrying an older key is one the current slot has not re-embedded yet
    (`slots.switch` requeues it). Photos with no vector keep their NULL. Idempotent.
    """
    from models import catalog

    # ONE statement, not a scan in Python: this runs on every worker poll (10 s),
    # and on a converged library it must match no rows and be over.
    honest = {e.key for e in catalog.CATALOG if e.slot == "image_embed"} | {model_name}
    placeholders = ",".join("?" * len(honest))
    cursor = conn.execute(
        "UPDATE photos SET embedding_model = ?"
        f" WHERE embedding_model IS NOT NULL AND embedding_model NOT IN ({placeholders})",
        (model_name, *honest),
    )
    return cursor.rowcount


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
