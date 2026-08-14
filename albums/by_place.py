import sqlite3

from albums.base import Album
from ingest.geocode import Place, reverse_many

# Round GPS to ~1 km. Used by the Memories seeder (plan 07) to split an event by
# location; the place organizer itself now groups by real place name, not cells.
GRID = 0.01


def _cell(lat: float, lon: float) -> tuple[float, float]:
    return (round(lat / GRID) * GRID, round(lon / GRID) * GRID)


class ByPlaceOrganizer:
    name = "place"
    label = "By place"

    def organize(
        self, conn: sqlite3.Connection, owner_id: int, grain: str | None = None
    ) -> list[Album]:
        rows = conn.execute(
            "SELECT id, gps_lat, gps_lon FROM photos"
            " WHERE owner_id = ? AND thumb_key IS NOT NULL"
            " AND gps_lat IS NOT NULL AND gps_lon IS NOT NULL"
            " ORDER BY COALESCE(shot_at, created_at) DESC, id DESC",
            (owner_id,),
        ).fetchall()
        if not rows:
            return []

        # One offline geocode pass over every point, then group by place name — so
        # a whole city is one album, never a coordinate string (§11).
        places = reverse_many([(row["gps_lat"], row["gps_lon"]) for row in rows])
        groups: dict[str, dict] = {}
        for row, place in zip(rows, places):
            label = place.label
            bucket = groups.setdefault(label, {"ids": [], "place": place})
            bucket["ids"].append(row["id"])

        albums = []
        for index, (label, bucket) in enumerate(
            sorted(groups.items(), key=lambda kv: len(kv[1]["ids"]), reverse=True)
        ):
            ids = bucket["ids"]
            place: Place = bucket["place"]
            count = len(ids)
            albums.append(
                Album(
                    key=f"place-{index}",
                    title=place.city or label,
                    description=f"{count} photo{'s' if count != 1 else ''} in {label}.",
                    photo_ids=ids,
                    cover_id=ids[0],
                    meta={"kind": "place"},
                )
            )
        return albums
