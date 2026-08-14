import hashlib
import io
from pathlib import Path

from PIL import Image


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def jpeg_bytes(size: tuple[int, int] = (64, 48), color: str = "red") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="JPEG")
    return buffer.getvalue()


def jpeg_bytes_with_exif(
    when: str = "2025:07:12 14:30:00", model: str = "TestCam"
) -> bytes:
    image = Image.new("RGB", (64, 48), "blue")
    exif = image.getexif()
    exif[0x0110] = model          # Model
    exif[0x9003] = when           # DateTimeOriginal
    exif[0x010F] = "TestMake"     # Make
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", exif=exif)
    return buffer.getvalue()


def make_jpeg(path: Path, size: tuple[int, int] = (64, 48), color: str = "red") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path, format="JPEG")
    return path


def make_jpeg_with_exif(
    path: Path, when: str = "2025:07:12 14:30:00", model: str = "TestCam"
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (64, 48), "blue")
    exif = image.getexif()
    exif[0x0110] = model          # Model
    exif[0x9003] = when           # DateTimeOriginal
    exif[0x010F] = "TestMake"     # Make
    image.save(path, format="JPEG", exif=exif)
    return path
