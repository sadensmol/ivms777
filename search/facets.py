import sqlite3
from dataclasses import dataclass

from ingest.facets import FACET_KEYS

SORTABLE: dict[str, str] = {
    "year_desc": "year", "year_asc": "year",
    "iso_desc": "iso", "iso_asc": "iso",
    "aperture_desc": "aperture", "aperture_asc": "aperture",
    "focal_length_desc": "focal_length", "focal_length_asc": "focal_length",
    "megapixels_desc": "megapixels", "megapixels_asc": "megapixels",
}

SIDEBAR_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Camera", ("camera_make", "camera_model", "lens")),
    ("Place", ("place_city", "place_country")),
    ("Time", ("time_of_day", "weekday", "is_weekend")),
    ("Image", ("aspect", "has_gps", "flash")),
)


@dataclass(frozen=True)
class FacetFilter:
    key: str
    values: list[str] | None = None
    gte: float | None = None
    lte: float | None = None


def _number(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def parse_filters(params: dict[str, str]) -> list[FacetFilter]:
    filters: list[FacetFilter] = []
    for name, raw in params.items():
        if name.startswith("f_"):
            key = name[2:]
            values = [part for part in raw.split(",") if part]
            if key in FACET_KEYS and values:
                filters.append(FacetFilter(key, values=values))
        elif name.startswith("n_"):
            key = name[2:]
            if key not in FACET_KEYS or ":" not in raw:
                continue
            low, _, high = raw.partition(":")
            gte, lte = _number(low) if low else None, _number(high) if high else None
            if gte is not None or lte is not None:
                filters.append(FacetFilter(key, gte=gte, lte=lte))
    return filters


def build_where(filters: list[FacetFilter]) -> tuple[str, list]:
    fragments: list[str] = []
    params: list = []
    for item in filters:
        conditions = ["f.photo_id = p.id", "f.key = ?"]
        params.append(item.key)
        if item.values:
            placeholders = ", ".join("?" for _ in item.values)
            conditions.append(f"f.value_text IN ({placeholders})")
            params.extend(item.values)
        if item.gte is not None:
            conditions.append("f.value_num >= ?")
            params.append(item.gte)
        if item.lte is not None:
            conditions.append("f.value_num <= ?")
            params.append(item.lte)
        fragments.append(
            " AND EXISTS (SELECT 1 FROM photo_facets f WHERE " + " AND ".join(conditions) + ")"
        )
    return "".join(fragments), params


def facet_counts(
    conn: sqlite3.Connection, owner_id: int, key: str, limit: int = 12
) -> list[tuple[str, int]]:
    rows = conn.execute(
        "SELECT f.value_text AS value, count(*) AS n FROM photo_facets f"
        " JOIN photos p ON p.id = f.photo_id"
        " WHERE f.key = ? AND p.owner_id = ? AND f.value_text IS NOT NULL"
        " GROUP BY f.value_text ORDER BY n DESC, f.value_text LIMIT ?",
        (key, owner_id, limit),
    ).fetchall()
    return [(row["value"], row["n"]) for row in rows]
