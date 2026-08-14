from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import pillow_heif
from PIL import ExifTags, Image

pillow_heif.register_heif_opener()

_DATETIME_ORIGINAL = 0x9003
_MODEL = 0x0110
_LENS_MODEL = 0xA434
_GPS_IFD = 0x8825
_EXIF_IFD = 0x8769


@dataclass(frozen=True)
class ExifFacts:
    shot_at: str | None = None
    camera: str | None = None
    lens: str | None = None
    gps_lat: float | None = None
    gps_lon: float | None = None
    width: int | None = None
    height: int | None = None
    raw: dict[str, object] = field(default_factory=dict)


def _jsonable(value: object) -> object:
    """EXIF values include IFDRational, bytes, and tuples. Make them JSON-safe."""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")[:200]
    if isinstance(value, Fraction) or hasattr(value, "numerator"):
        try:
            return float(value)  # type: ignore[arg-type]
        except (ZeroDivisionError, TypeError, ValueError):
            return None
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)[:200]


def _collect_raw(exif: Image.Exif) -> dict[str, object]:
    collected: dict[str, object] = {}
    for tag_id, value in exif.items():
        name = ExifTags.TAGS.get(tag_id, str(tag_id))
        collected[name] = _jsonable(value)
    try:
        for tag_id, value in exif.get_ifd(_EXIF_IFD).items():
            name = ExifTags.TAGS.get(tag_id, str(tag_id))
            collected[name] = _jsonable(value)
    except Exception:  # noqa: BLE001, S110 - a malformed sub-IFD must not fail ingest
        pass
    collected.pop("ExifOffset", None)
    collected.pop("GPSInfo", None)
    return collected


def _parse_datetime(raw: object) -> str | None:
    if not isinstance(raw, str) or len(raw) < 19:
        return None
    date, _, time = raw.partition(" ")
    return f"{date.replace(':', '-')}T{time}"


def _to_degrees(value: object) -> float | None:
    try:
        degrees, minutes, seconds = (float(part) for part in value)  # type: ignore[misc]
    except (TypeError, ValueError):
        return None
    return degrees + minutes / 60 + seconds / 3600


def _read_gps(exif: Image.Exif) -> tuple[float | None, float | None]:
    try:
        gps = exif.get_ifd(_GPS_IFD)
    except Exception:  # noqa: BLE001 - unparseable GPS block means "no GPS", not a crash
        return None, None
    if not gps:
        return None, None
    lat = _to_degrees(gps.get(ExifTags.GPS.GPSLatitude))
    lon = _to_degrees(gps.get(ExifTags.GPS.GPSLongitude))
    if lat is not None and gps.get(ExifTags.GPS.GPSLatitudeRef) == "S":
        lat = -lat
    if lon is not None and gps.get(ExifTags.GPS.GPSLongitudeRef) == "W":
        lon = -lon
    return lat, lon


def read_exif(path: Path) -> ExifFacts:
    try:
        with Image.open(path) as image:
            width, height = image.size
            exif = image.getexif()
    except Exception:  # noqa: BLE001 - any unreadable image yields empty facts
        return ExifFacts()

    raw = _collect_raw(exif)
    lat, lon = _read_gps(exif)
    return ExifFacts(
        shot_at=_parse_datetime(raw.get("DateTimeOriginal") or exif.get(_DATETIME_ORIGINAL)),
        camera=exif.get(_MODEL) or None,
        lens=raw.get("LensModel") or exif.get(_LENS_MODEL) or None,
        gps_lat=lat,
        gps_lon=lon,
        width=width,
        height=height,
        raw=raw,
    )
