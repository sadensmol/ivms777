import sqlite3

from albums.base import Album
from albums.memory_store import read_memories


class MemoriesOrganizer:
    name = "memories"
    label = "Memories"

    def organize(
        self, conn: sqlite3.Connection, owner_id: int, grain: str | None = None
    ) -> list[Album]:
        # Reads stored memories and renders instantly — no model on the page load
        # (§11). The agentic build runs offline; `grain` is ignored.
        albums: list[Album] = []
        for index, memory in enumerate(read_memories(conn, owner_id)):
            albums.append(
                Album(
                    key=f"memory-{index}",
                    title=memory.title,
                    description=memory.description,
                    photo_ids=memory.photo_ids,
                    cover_id=memory.photo_ids[0],
                    meta={"kind": "memory"},
                )
            )
        return albums
