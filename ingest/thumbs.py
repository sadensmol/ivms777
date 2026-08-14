import io
from pathlib import Path

import pillow_heif
from PIL import Image, ImageOps

from storage.base import Storage

pillow_heif.register_heif_opener()


def thumb_key(content_hash: str, size: int) -> str:
    return f"{content_hash[:2]}/{content_hash}_{size}.jpg"


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
