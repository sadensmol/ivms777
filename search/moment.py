"""The `moment` signal (§9) — photos from the same stretch of time in the same place.

A moment is the one similarity signal that does not look at the picture at all. Two
photos can share no subject, no palette and no visual resemblance and still belong
together because they were taken minutes apart in the same spot: the cake and the
faces around it, the trailhead sign and the summit. That is what a photo library is
for, and neither SigLIP nor the caption can see it.

The strength itself is `signals.moment_strength`; this module is only the data side —
reading EXIF time and GPS, measuring the gap, and doing it with a bounded scan.
"""

import math
import sqlite3
from datetime import datetime

from search.signals import MOMENT_GATE, MOMENT_NO_GPS, MOMENT_TAU_HOURS, moment_strength

# Widest time gap that can still clear the gate, so SQL never scans further. With a
# PERFECT place match the strength is `exp(-Δt/τ)`, so the gate is reached at
# `τ·ln(1/gate)` hours; anything beyond that cannot qualify however close the places.
_MAX_GAP_HOURS = MOMENT_TAU_HOURS * math.log(1.0 / MOMENT_GATE)

_EARTH_RADIUS_M = 6_371_000.0


def parse_shot_at(value: str | None) -> float | None:
    """EXIF `shot_at` as epoch seconds, or None when it is missing or unparseable.

    A photo with no usable timestamp simply has no `moment` signal — it is absent,
    never zero (§9.2), so the candidate still competes on every other signal.
    """
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.strip()).timestamp()
    except ValueError:
        return None


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    d_phi = p2 - p1
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(d_lambda / 2) ** 2
    return 2 * _EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))


def gap(
    seed: tuple[float, float | None, float | None],
    other: tuple[float, float | None, float | None],
) -> tuple[float, float | None]:
    """`(hours_apart, metres_apart)` between two `(epoch, lat, lon)` triples.

    `metres_apart` is None when either photo lacks GPS — unknown, not zero, so
    `moment_strength` falls back to time alone instead of pretending they were in the
    same place.
    """
    hours = abs(seed[0] - other[0]) / 3600.0
    if seed[1] is None or seed[2] is None or other[1] is None or other[2] is None:
        return hours, None
    return hours, haversine_m(seed[1], seed[2], other[1], other[2])


def read_moment(
    conn: sqlite3.Connection, photo_id: int
) -> tuple[float, float | None, float | None] | None:
    """A photo's `(epoch, lat, lon)`, or None when it carries no timestamp."""
    row = conn.execute(
        "SELECT shot_at, gps_lat, gps_lon FROM photos WHERE id = ?", (photo_id,)
    ).fetchone()
    if row is None:
        return None
    when = parse_shot_at(row["shot_at"])
    if when is None:
        return None
    return when, row["gps_lat"], row["gps_lon"]


def moment_similarity(
    conn: sqlite3.Connection, owner_id: int, seed_id: int, gate: float = MOMENT_GATE
) -> dict[int, float]:
    """`{photo_id: strength}` for every photo sharing the seed's moment (§9).

    Scans the owner's timestamped photos and cuts by `_MAX_GAP_HOURS` in Python
    rather than by a SQL `BETWEEN`. `shot_at` is free-form ISO text, so a string
    range would silently drop rows written with a space separator instead of `T`;
    correctness wins over an index at this size, and the row is three columns wide.
    """
    if gate >= 1.0:
        return {}  # ablation: the signal is switched off
    seed = read_moment(conn, seed_id)
    if seed is None:
        return {}
    rows = conn.execute(
        "SELECT id, shot_at, gps_lat, gps_lon FROM photos"
        " WHERE owner_id = ? AND id != ? AND shot_at IS NOT NULL",
        (owner_id, seed_id),
    )
    hits: dict[int, float] = {}
    for row in rows:
        when = parse_shot_at(row["shot_at"])
        if when is None:
            continue
        hours = abs(seed[0] - when) / 3600.0
        if hours > _MAX_GAP_HOURS:
            continue
        hours, metres = gap(seed, (when, row["gps_lat"], row["gps_lon"]))
        strength = moment_strength(hours, metres)
        if strength >= gate:
            hits[row["id"]] = strength
    return hits


__all__ = [
    "MOMENT_NO_GPS",
    "gap",
    "haversine_m",
    "moment_similarity",
    "parse_shot_at",
    "read_moment",
]
