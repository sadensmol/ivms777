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
    """Atomically swap the owner's memories, so the tab never shows a half build."""
    conn.execute("BEGIN")
    try:
        conn.execute(
            "DELETE FROM groups WHERE owner_id = ? AND kind = 'memory'", (owner_id,)
        )
        for memory in memories:
            cursor = conn.execute(
                "INSERT INTO groups(owner_id, kind, name, description, params, status, created_at)"
                f" VALUES (?, 'memory', ?, ?, ?, 'accepted', {NOW_SQL})",
                (owner_id, memory.title, memory.description,
                 json.dumps({"signature": memory.signature})),
            )
            group_id = int(cursor.lastrowid)
            conn.executemany(
                "INSERT INTO group_photos(group_id, photo_id, rank) VALUES (?, ?, ?)",
                [(group_id, pid, rank) for rank, pid in enumerate(memory.photo_ids)],
            )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


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
        signature = json.loads(row["params"] or "{}").get("signature", "")
        memories.append(Memory(row["name"], row["description"], photo_ids, signature))
    return memories
