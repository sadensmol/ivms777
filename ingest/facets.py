import sqlite3
from dataclasses import dataclass
from datetime import datetime

from ingest.exif import ExifFacts
from ingest.geocode import reverse, reverse_many

FACET_KEYS: tuple[str, ...] = (
    "camera_make", "camera_model", "lens", "software",
    "iso", "aperture", "shutter_speed", "focal_length", "exposure_bias",
    "flash", "exposure_program", "metering_mode", "white_balance",
    "year", "month", "weekday", "hour", "time_of_day", "is_weekend",
    "has_gps", "gps_lat", "gps_lon", "place_city", "place_country",
    "megapixels", "orientation", "aspect",
)

WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")

EXPOSURE_PROGRAMS = {
    0: "not defined", 1: "manual", 2: "normal", 3: "aperture priority",
    4: "shutter priority", 5: "creative", 6: "action", 7: "portrait", 8: "landscape",
}
METERING_MODES = {
    0: "unknown", 1: "average", 2: "center weighted", 3: "spot",
    4: "multi spot", 5: "pattern", 6: "partial",
}
WHITE_BALANCES = {0: "auto", 1: "manual"}

NUMERIC_EXIF = {
    "aperture": ("FNumber", "ApertureValue"),
    "shutter_speed": ("ExposureTime",),
    "focal_length": ("FocalLength",),
    "exposure_bias": ("ExposureBiasValue",),
    "iso": ("ISOSpeedRatings", "PhotographicSensitivity", "ISO"),
}


@dataclass(frozen=True)
class Facet:
    key: str
    value_text: str | None = None
    value_num: float | None = None


def _time_of_day(hour: int) -> str:
    if hour < 5:
        return "night"
    if hour < 8:
        return "dawn"
    if hour < 12:
        return "morning"
    if hour < 17:
        return "afternoon"
    if hour < 21:
        return "evening"
    return "night"


def _first_number(raw: dict[str, object], names: tuple[str, ...]) -> float | None:
    for name in names:
        value = raw.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def derive_facets(facts: ExifFacts, width: int | None, height: int | None) -> list[Facet]:
    raw = facts.raw or {}
    out: list[Facet] = []

    def add_text(key: str, value: object) -> None:
        text = _text(value)
        if text is not None:
            out.append(Facet(key, value_text=text))

    def add_num(key: str, value: float | None) -> None:
        if value is not None:
            out.append(Facet(key, value_num=float(value)))

    add_text("camera_make", raw.get("Make"))
    add_text("camera_model", facts.camera)
    add_text("lens", facts.lens)
    add_text("software", raw.get("Software"))

    for key, names in NUMERIC_EXIF.items():
        add_num(key, _first_number(raw, names))

    flash = raw.get("Flash")
    if isinstance(flash, int) and not isinstance(flash, bool):
        add_text("flash", "fired" if flash & 1 else "did not fire")
    program = raw.get("ExposureProgram")
    if isinstance(program, int):
        add_text("exposure_program", EXPOSURE_PROGRAMS.get(program))
    metering = raw.get("MeteringMode")
    if isinstance(metering, int):
        add_text("metering_mode", METERING_MODES.get(metering))
    balance = raw.get("WhiteBalance")
    if isinstance(balance, int):
        add_text("white_balance", WHITE_BALANCES.get(balance))

    if facts.shot_at:
        try:
            when = datetime.fromisoformat(facts.shot_at)
        except ValueError:
            when = None
        if when is not None:
            add_num("year", when.year)
            add_num("month", when.month)
            add_num("hour", when.hour)
            add_text("weekday", WEEKDAYS[when.weekday()])
            add_text("time_of_day", _time_of_day(when.hour))
            add_text("is_weekend", "yes" if when.weekday() >= 5 else "no")

    has_gps = facts.gps_lat is not None and facts.gps_lon is not None
    add_text("has_gps", "yes" if has_gps else "no")
    if has_gps:
        add_num("gps_lat", facts.gps_lat)
        add_num("gps_lon", facts.gps_lon)
        # A real place name for filtering — coordinates stay technical (§11).
        place = reverse(facts.gps_lat, facts.gps_lon)
        if place is not None:
            add_text("place_city", place.city)
            add_text("place_country", place.country)

    orientation = raw.get("Orientation")
    if isinstance(orientation, int):
        add_num("orientation", orientation)
    if width and height:
        add_num("megapixels", round(width * height / 1_000_000, 1))
        if width > height:
            add_text("aspect", "landscape")
        elif width < height:
            add_text("aspect", "portrait")
        else:
            add_text("aspect", "square")

    return out


def store_facets(conn: sqlite3.Connection, photo_id: int, facets: list[Facet]) -> None:
    conn.execute("DELETE FROM photo_facets WHERE photo_id = ?", (photo_id,))
    conn.executemany(
        "INSERT INTO photo_facets(photo_id, key, value_text, value_num) VALUES (?, ?, ?, ?)",
        [(photo_id, f.key, f.value_text, f.value_num) for f in facets],
    )


def backfill_place_facets(conn: sqlite3.Connection) -> int:
    """Add place_city/place_country facets to every GPS photo that lacks them.

    Photos ingested before place facets existed carry GPS but no place name; this
    geocodes them in one batch pass on the next drain. Idempotent — a photo that
    already has a place_city facet is skipped.
    """
    rows = conn.execute(
        "SELECT id, gps_lat, gps_lon FROM photos p"
        " WHERE gps_lat IS NOT NULL AND gps_lon IS NOT NULL"
        " AND NOT EXISTS (SELECT 1 FROM photo_facets f"
        "                 WHERE f.photo_id = p.id AND f.key = 'place_city')"
    ).fetchall()
    if not rows:
        return 0
    places = reverse_many([(row["gps_lat"], row["gps_lon"]) for row in rows])
    for row, place in zip(rows, places):
        for key, value in (("place_city", place.city), ("place_country", place.country)):
            if value:
                conn.execute(
                    "INSERT INTO photo_facets(photo_id, key, value_text) VALUES (?, ?, ?)",
                    (row["id"], key, value),
                )
    return len(rows)
