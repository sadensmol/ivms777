import sqlite3
from datetime import datetime

from albums.base import Album

# A gap longer than this between consecutive shots starts a new event. Six hours
# separates "morning at the beach" from "dinner that night" without splitting a
# single afternoon into fragments.
EVENT_GAP_HOURS = 6.0


def _parse(shot_at: str) -> datetime | None:
    try:
        return datetime.fromisoformat(shot_at)
    except (ValueError, TypeError):
        return None


def _format_range(start: datetime, end: datetime) -> str:
    if start.date() == end.date():
        return start.strftime("%-d %b %Y")
    if (start.year, start.month) == (end.year, end.month):
        return f"{start.strftime('%-d')}–{end.strftime('%-d %b %Y')}"
    return f"{start.strftime('%-d %b')} – {end.strftime('%-d %b %Y')}"


class ByDateOrganizer:
    name = "date"
    label = "By date (events)"

    def organize(self, conn: sqlite3.Connection, owner_id: int) -> list[Album]:
        rows = conn.execute(
            "SELECT id, shot_at, camera FROM photos"
            " WHERE owner_id = ? AND thumb_key IS NOT NULL AND shot_at IS NOT NULL"
            " ORDER BY shot_at",
            (owner_id,),
        ).fetchall()

        events: list[list[sqlite3.Row]] = []
        last: datetime | None = None
        for row in rows:
            when = _parse(row["shot_at"])
            if when is None:
                continue
            if last is None or (when - last).total_seconds() > EVENT_GAP_HOURS * 3600:
                events.append([])
            events[-1].append(row)
            last = when

        albums: list[Album] = []
        for index, group in enumerate(events):
            times = [_parse(r["shot_at"]) for r in group]
            times = [t for t in times if t is not None]
            start, end = times[0], times[-1]
            cameras = _dominant([r["camera"] for r in group])
            span_days = (end.date() - start.date()).days + 1
            desc = f"{len(group)} photos over {span_days} day{'s' if span_days > 1 else ''}"
            if cameras:
                desc += f", mostly {cameras}"
            desc += "."
            albums.append(
                Album(
                    key=f"event-{index}",
                    title=_format_range(start, end),
                    description=desc,
                    photo_ids=[r["id"] for r in group],
                    cover_id=group[0]["id"],
                    meta={"kind": "date"},
                )
            )
        albums.sort(key=lambda a: a.photo_ids[0], reverse=True)
        return albums


def _dominant(values: list[str | None]) -> str | None:
    counts: dict[str, int] = {}
    for value in values:
        if value:
            counts[value] = counts.get(value, 0) + 1
    if not counts:
        return None
    return max(counts, key=counts.get)
