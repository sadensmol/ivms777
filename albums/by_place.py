import sqlite3

from albums.base import Album

# Round GPS to ~1 km so shots from the same spot land together. 0.01° of latitude
# is about 1.1 km; longitude shrinks with latitude but this is close enough for a
# "same place" bucket. Real place names arrive with offline reverse geocoding.
GRID = 0.01


def _cell(lat: float, lon: float) -> tuple[float, float]:
    return (round(lat / GRID) * GRID, round(lon / GRID) * GRID)


class ByPlaceOrganizer:
    name = "place"
    label = "By place (GPS)"

    def organize(self, conn: sqlite3.Connection, owner_id: int) -> list[Album]:
        rows = conn.execute(
            "SELECT id, gps_lat, gps_lon FROM photos"
            " WHERE owner_id = ? AND thumb_key IS NOT NULL"
            " AND gps_lat IS NOT NULL AND gps_lon IS NOT NULL"
            " ORDER BY COALESCE(shot_at, created_at) DESC, id DESC",
            (owner_id,),
        ).fetchall()

        groups: dict[tuple[float, float], list[int]] = {}
        for row in rows:
            groups.setdefault(_cell(row["gps_lat"], row["gps_lon"]), []).append(row["id"])

        albums = []
        for index, (cell, ids) in enumerate(
            sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True)
        ):
            lat, lon = cell
            title = f"{abs(lat):.2f}°{'N' if lat >= 0 else 'S'}, {abs(lon):.2f}°{'E' if lon >= 0 else 'W'}"
            albums.append(
                Album(
                    key=f"place-{index}",
                    title=title,
                    description=f"{len(ids)} photos taken near {title}.",
                    photo_ids=ids,
                    cover_id=ids[0],
                    meta={"kind": "place"},
                )
            )
        return albums
