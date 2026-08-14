import sqlite3
from datetime import datetime

from albums.base import Album

# The "By date" grains: one axis (the calendar) at three zooms. `month` is the
# default — the most useful bucket for browsing a phone dump. Order here is the
# sub-selector order. There is deliberately no "events" grain: time-gap
# clustering is a different principle, used only for Memories seeding (see
# docs/design.md §11), never a zoom level of the calendar.
GRAIN_LABELS = [("day", "Day"), ("month", "Month"), ("year", "Year")]
GRAINS = tuple(value for value, _ in GRAIN_LABELS)
DEFAULT_GRAIN = "month"


def _parse(shot_at: str) -> datetime | None:
    try:
        return datetime.fromisoformat(shot_at)
    except (ValueError, TypeError):
        return None


def _bucket(dt: datetime, grain: str) -> tuple[str, str, object]:
    """Return (url-safe key, display title, sort key) for a calendar grain."""
    if grain == "day":
        return f"day-{dt.date().isoformat()}", dt.strftime("%-d %b %Y"), dt.date()
    if grain == "year":
        return f"year-{dt.year}", str(dt.year), dt.year
    return f"month-{dt.year:04d}-{dt.month:02d}", dt.strftime("%B %Y"), (dt.year, dt.month)


def _describe(count: int, cameras: list[str | None]) -> str:
    desc = f"{count} photo{'s' if count != 1 else ''}"
    camera = _dominant(cameras)
    return desc + (f", mostly {camera}." if camera else ".")


class ByDateOrganizer:
    name = "date"
    label = "By date"
    grains = GRAINS

    def organize(
        self, conn: sqlite3.Connection, owner_id: int, grain: str | None = None
    ) -> list[Album]:
        grain = grain if grain in GRAINS else DEFAULT_GRAIN
        rows = conn.execute(
            "SELECT id, shot_at, camera FROM photos"
            " WHERE owner_id = ? AND thumb_key IS NOT NULL AND shot_at IS NOT NULL"
            " ORDER BY shot_at",
            (owner_id,),
        ).fetchall()

        buckets: dict[str, dict] = {}
        for row in rows:
            when = _parse(row["shot_at"])
            if when is None:
                continue
            key, title, sort = _bucket(when, grain)
            bucket = buckets.setdefault(
                key, {"title": title, "sort": sort, "ids": [], "cameras": []}
            )
            bucket["ids"].append(row["id"])
            bucket["cameras"].append(row["camera"])

        ordered = sorted(buckets.items(), key=lambda kv: kv[1]["sort"], reverse=True)
        return [
            Album(
                key=key,
                title=bucket["title"],
                description=_describe(len(bucket["ids"]), bucket["cameras"]),
                photo_ids=bucket["ids"],
                cover_id=bucket["ids"][0],
                meta={"kind": "date", "grain": grain},
            )
            for key, bucket in ordered
        ]


def _dominant(values: list[str | None]) -> str | None:
    counts: dict[str, int] = {}
    for value in values:
        if value:
            counts[value] = counts.get(value, 0) + 1
    if not counts:
        return None
    return max(counts, key=counts.get)
