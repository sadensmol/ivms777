import io
import sqlite3
from pathlib import Path

import pillow_heif
from PIL import Image, ImageOps

from ingest.jobs import enqueue
from storage.base import Storage

pillow_heif.register_heif_opener()


def thumb_key(content_hash: str, size: int) -> str:
    return f"{content_hash[:2]}/{content_hash}_{size}.jpg"


def backfill_thumbnails(conn: sqlite3.Connection) -> int:
    """Queue a thumbnail job for every photo that still has none.

    Photos whose thumbnail never succeeded (an early failure, or ingest before
    the stage existed) are otherwise stuck: invisible in the grid and unable to be
    captioned. This re-attempts them on the next drain. Idempotent — `enqueue`
    skips a stage that already has a job, so a genuinely broken image fails once
    and stays failed rather than retrying forever.
    """
    rows = conn.execute("SELECT id FROM photos WHERE thumb_key IS NULL").fetchall()
    for row in rows:
        enqueue(conn, row["id"], "thumbnail")
    return len(rows)


def _render(image: Image.Image, box: int) -> bytes:
    copy = image.copy()
    copy.thumbnail((box, box), Image.LANCZOS)
    if copy.mode != "RGB":
        copy = copy.convert("RGB")
    buffer = io.BytesIO()
    copy.save(buffer, format="JPEG", quality=85, optimize=True)
    return buffer.getvalue()


def make_thumbnails(
    source: Path,
    content_hash: str,
    derived: Storage,
    grid_px: int,
    detail_px: int,
) -> str:
    with Image.open(source) as image:
        image.load()
        upright = ImageOps.exif_transpose(image)
        for box in (grid_px, detail_px):
            derived.write(thumb_key(content_hash, box), _render(upright, box))
    return thumb_key(content_hash, grid_px)
