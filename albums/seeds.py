import sqlite3
from dataclasses import dataclass
from datetime import datetime

from albums.by_place import _cell

# A gap longer than this starts a new run. Owned here, not by the date organizer:
# time-gap clustering is a Memories seeding step, not a user-facing date view.
EVENT_GAP_HOURS = 6.0


@dataclass(frozen=True)
class Candidate:
    photo_ids: list[int]


def _parse(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None


def seed_candidates(
    conn: sqlite3.Connection, owner_id: int, *, min_size: int = 3
) -> list[Candidate]:
    """Time-and-place-contiguous runs of captioned photos, newest first.

    A run breaks on a > 6 h capture gap, or when it crosses into a different
    ~1 km GPS cell. Photos with no GPS never force a split — they extend the
    current run. Runs shorter than `min_size` are dropped.
    """
    rows = conn.execute(
        "SELECT id, shot_at, gps_lat, gps_lon FROM photos"
        " WHERE owner_id = ? AND thumb_key IS NOT NULL"
        " AND shot_at IS NOT NULL AND caption IS NOT NULL"
        " ORDER BY shot_at",
        (owner_id,),
    ).fetchall()

    runs: list[list[int]] = []
    last_time: datetime | None = None
    last_cell = None
    for row in rows:
        when = _parse(row["shot_at"])
        if when is None:
            continue
        has_gps = row["gps_lat"] is not None and row["gps_lon"] is not None
        cell = _cell(row["gps_lat"], row["gps_lon"]) if has_gps else None
        gap = last_time is None or (when - last_time).total_seconds() > EVENT_GAP_HOURS * 3600
        moved = cell is not None and last_cell is not None and cell != last_cell
        if gap or moved:
            runs.append([])
        runs[-1].append(row["id"])
        last_time = when
        if cell is not None:
            last_cell = cell

    candidates = [Candidate(ids) for ids in runs if len(ids) >= min_size]
    candidates.sort(key=lambda c: c.photo_ids[0], reverse=True)  # newest first
    return candidates
