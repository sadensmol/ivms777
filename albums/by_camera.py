import sqlite3

from albums.base import Album


class ByCameraOrganizer:
    name = "camera"
    label = "By camera / device"

    def organize(self, conn: sqlite3.Connection, owner_id: int) -> list[Album]:
        rows = conn.execute(
            "SELECT id, camera FROM photos"
            " WHERE owner_id = ? AND thumb_key IS NOT NULL"
            " ORDER BY COALESCE(shot_at, created_at) DESC, id DESC",
            (owner_id,),
        ).fetchall()

        groups: dict[str, list[int]] = {}
        for row in rows:
            camera = row["camera"] or "Unknown camera"
            groups.setdefault(camera, []).append(row["id"])

        albums = [
            Album(
                key=f"cam-{index}",
                title=camera,
                description=f"{len(ids)} photos taken with {camera}.",
                photo_ids=ids,
                cover_id=ids[0],
                meta={"kind": "camera"},
            )
            for index, (camera, ids) in enumerate(
                sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True)
            )
        ]
        return albums
