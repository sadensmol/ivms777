import json
import sqlite3
from dataclasses import dataclass

NOW_SQL = "strftime('%Y-%m-%dT%H:%M:%SZ','now')"


@dataclass(frozen=True)
class Memory:
    title: str
    description: str
    photo_ids: list[int]   # cover first
    signature: str
    # event = a time/place-bounded happening, theme = a thread across time (§11).
    # Stage 2 gives the file to the event and makes a theme a description-only
    # folder (§12), so the kind has to survive the round-trip through `params`.
    kind: str = "event"


def current_signature(conn: sqlite3.Connection, owner_id: int) -> str:
    """A cheap fingerprint of the owner's captioned library — count + newest
    updated_at. A rebuild is skipped when this still matches the stored one."""
    row = conn.execute(
        "SELECT count(*) AS n, COALESCE(max(updated_at), '') AS m"
        " FROM photos WHERE owner_id = ? AND caption IS NOT NULL",
        (owner_id,),
    ).fetchone()
    return f"{row['n']}:{row['m']}"


def stored_signature(conn: sqlite3.Connection, owner_id: int) -> str | None:
    row = conn.execute(
        "SELECT params FROM groups WHERE owner_id = ? AND kind = 'memory'"
        " ORDER BY id LIMIT 1",
        (owner_id,),
    ).fetchone()
    if row is None:
        return None
    return json.loads(row["params"] or "{}").get("signature")


def replace_memories(conn: sqlite3.Connection, owner_id: int, memories: list[Memory]) -> None:
    """Atomically swap the owner's memories, so the tab never shows a half build.

    Each memory's name + description is also (re)indexed into `memory_fts` so chat
    can find a memory by place/occasion/name (§10, §11). The FTS rows are keyed by
    the group id, so they are cleared before the old groups are deleted and
    re-inserted alongside the new groups — index and memories rebuild in lockstep.
    """
    conn.execute("BEGIN")
    try:
        conn.execute(
            "DELETE FROM memory_fts WHERE rowid IN"
            " (SELECT id FROM groups WHERE owner_id = ? AND kind = 'memory')",
            (owner_id,),
        )
        conn.execute(
            "DELETE FROM groups WHERE owner_id = ? AND kind = 'memory'", (owner_id,)
        )
        for memory in memories:
            cursor = conn.execute(
                "INSERT INTO groups(owner_id, kind, name, description, params, status, created_at)"
                f" VALUES (?, 'memory', ?, ?, ?, 'accepted', {NOW_SQL})",
                (owner_id, memory.title, memory.description,
                 json.dumps({"signature": memory.signature, "kind": memory.kind})),
            )
            group_id = int(cursor.lastrowid)
            conn.execute(
                "INSERT INTO memory_fts(rowid, name, description) VALUES (?, ?, ?)",
                (group_id, memory.title, memory.description),
            )
            conn.executemany(
                "INSERT INTO group_photos(group_id, photo_id, rank) VALUES (?, ?, ?)",
                [(group_id, pid, rank) for rank, pid in enumerate(memory.photo_ids)],
            )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def rewrite_memory(conn: sqlite3.Connection, group_id: int, memory: Memory) -> None:
    """Replace ONE memory's title, description and photo order, in place.

    In place matters. `replace_memories` drops every memory and inserts fresh rows, so
    it takes the group ids with it — and `memory_notes` cascades on those (§11.3). The
    viewer's own words about a memory must survive it being retold; they are the input
    to the next telling.

    What does NOT survive is the film: the memory it narrated no longer exists in that
    shape, so every stored presentation for this group is dropped and must be prepared
    again (§11.1). Its edited narration goes too, in every language: its photographs,
    cards and score plan were made for the old telling.
    """
    conn.execute(
        "UPDATE groups SET name = ?, description = ? WHERE id = ? AND kind = 'memory'",
        (memory.title, memory.description, group_id),
    )
    conn.execute("DELETE FROM group_photos WHERE group_id = ?", (group_id,))
    conn.executemany(
        "INSERT INTO group_photos(group_id, photo_id, rank) VALUES (?, ?, ?)",
        [(group_id, pid, rank) for rank, pid in enumerate(memory.photo_ids)],
    )
    conn.execute("DELETE FROM memory_presentations WHERE group_id = ?", (group_id,))
    conn.execute("DELETE FROM narration_drafts WHERE group_id = ?", (group_id,))
    _reindex_memory(conn, group_id, memory)


def _reindex_memory(conn: sqlite3.Connection, group_id: int, memory: Memory) -> None:
    """Keep the FTS row in step with the retold memory, so chat can still find it."""
    conn.execute("DELETE FROM memory_fts WHERE rowid = ?", (group_id,))
    conn.execute(
        "INSERT INTO memory_fts(rowid, name, description) VALUES (?, ?, ?)",
        (group_id, memory.title, memory.description),
    )


def photo_ids_for_group(conn: sqlite3.Connection, group_id: int) -> list[int]:
    """This memory's photographs, in its own stored order."""
    return [
        int(row["photo_id"])
        for row in conn.execute(
            "SELECT photo_id FROM group_photos WHERE group_id = ? ORDER BY rank",
            (group_id,),
        )
    ]


def album_key_for_group(
    conn: sqlite3.Connection, owner_id: int, group_id: int
) -> str | None:
    """The Organize album key (`memory-<index>`) for a memory `group_id` — its
    position in `read_memories` order (size desc, then id). Chat uses it to link
    into the very same memory grid the Organize tab shows, so a memory drilled into
    from chat pages exactly like one drilled into from Organize (§10, §13.1). The
    ordering MUST match `read_memories`/`MemoriesOrganizer`; both live here.

    None when the group is not this owner's memory."""
    rows = conn.execute(
        "SELECT g.id,"
        " (SELECT count(*) FROM group_photos gp WHERE gp.group_id = g.id) AS n"
        " FROM groups g WHERE g.owner_id = ? AND g.kind = 'memory'"
        " ORDER BY n DESC, g.id",
        (owner_id,),
    ).fetchall()
    for index, row in enumerate(rows):
        if row["id"] == group_id:
            return f"memory-{index}"
    return None


def read_memories(conn: sqlite3.Connection, owner_id: int) -> list[Memory]:
    rows = conn.execute(
        "SELECT g.id, g.name, g.description, g.params,"
        " (SELECT count(*) FROM group_photos gp WHERE gp.group_id = g.id) AS n"
        " FROM groups g WHERE g.owner_id = ? AND g.kind = 'memory'"
        " ORDER BY n DESC, g.id",
        (owner_id,),
    ).fetchall()
    memories: list[Memory] = []
    for row in rows:
        photo_ids = [
            r["photo_id"] for r in conn.execute(
                "SELECT photo_id FROM group_photos WHERE group_id = ? ORDER BY rank",
                (row["id"],),
            )
        ]
        params = json.loads(row["params"] or "{}")
        memories.append(Memory(row["name"], row["description"], photo_ids,
                               params.get("signature", ""),
                               params.get("kind", "event")))
    return memories


def group_id_for_album_key(
    conn: sqlite3.Connection, owner_id: int, key: str
) -> int | None:
    """The memory `group_id` behind an Organize album key (`memory-<index>`) — the
    inverse of `album_key_for_group`, and it lives beside it for the same reason: the
    ordering (size desc, then id) MUST match `read_memories`, and ONE file owns it.
    The feed resolves `/feed/memory-3` to a row through this (§11.1).

    None when the key names no memory of this owner."""
    rows = conn.execute(
        "SELECT g.id,"
        " (SELECT count(*) FROM group_photos gp WHERE gp.group_id = g.id) AS n"
        " FROM groups g WHERE g.owner_id = ? AND g.kind = 'memory'"
        " ORDER BY n DESC, g.id",
        (owner_id,),
    ).fetchall()
    for index, row in enumerate(rows):
        if f"memory-{index}" == key:
            return int(row["id"])
    return None
