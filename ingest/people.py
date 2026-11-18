"""People and their face groups (design §6, §13): a person is a saved voice
(`saved_voices`), and a face group is attached to one by the owner.

Every query is scoped to the owner. A group or face of another owner reads as missing.
"""

import json
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

DEFAULT_MIN_FACES = 2


@dataclass(frozen=True, slots=True, kw_only=True)
class GroupFace:
    id: int
    photo_id: int
    pinned: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class FaceGroup:
    id: int
    person_id: int | None
    person_name: str | None
    face_count: int
    faces: tuple[GroupFace, ...]


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _groups(conn: sqlite3.Connection, rows: Sequence[sqlite3.Row]) -> list[FaceGroup]:
    """Each group row (`id`, `person_id`, `person_name`) with its faces, in row order."""
    if not rows:
        return []
    faces: dict[int, list[GroupFace]] = {row["id"]: [] for row in rows}
    for face in conn.execute(
        "SELECT id, photo_id, pinned, group_id FROM faces"
        " WHERE group_id IN (SELECT value FROM json_each(?)) ORDER BY id",
        (json.dumps(list(faces)),),
    ):
        faces[face["group_id"]].append(
            GroupFace(id=face["id"], photo_id=face["photo_id"], pinned=bool(face["pinned"]))
        )
    return [
        FaceGroup(
            id=row["id"],
            person_id=row["person_id"],
            person_name=row["person_name"],
            face_count=len(faces[row["id"]]),
            faces=tuple(faces[row["id"]]),
        )
        for row in rows
    ]


_GROUP_SELECT = (
    "SELECT g.id, g.person_id, v.name AS person_name FROM face_groups g"
    " LEFT JOIN saved_voices v ON v.id = g.person_id"
)


def groups_to_name(
    conn: sqlite3.Connection, owner_id: int, *, min_faces: int = DEFAULT_MIN_FACES
) -> list[FaceGroup]:
    """Groups with no person and at least `min_faces` faces ("Faces to name"), the
    largest first."""
    rows = conn.execute(
        "SELECT g.id, g.person_id, NULL AS person_name FROM face_groups g"
        " JOIN faces f ON f.group_id = g.id"
        " WHERE g.owner_id = ? AND g.person_id IS NULL"
        " GROUP BY g.id HAVING COUNT(*) >= ? ORDER BY COUNT(*) DESC, g.id",
        (owner_id, min_faces),
    ).fetchall()
    return _groups(conn, rows)


def group_detail(conn: sqlite3.Connection, owner_id: int, group_id: int) -> FaceGroup | None:
    row = conn.execute(
        f"{_GROUP_SELECT} WHERE g.id = ? AND g.owner_id = ?", (group_id, owner_id)
    ).fetchone()
    if row is None:
        return None
    return _groups(conn, [row])[0]


def _set_person(
    conn: sqlite3.Connection, owner_id: int, group_id: int, person_id: int | None
) -> FaceGroup | None:
    conn.execute(
        "UPDATE face_groups SET person_id = ?, updated_at = ? WHERE id = ? AND owner_id = ?",
        (person_id, _now(), group_id, owner_id),
    )
    return group_detail(conn, owner_id, group_id)


def assign_group(
    conn: sqlite3.Connection, owner_id: int, group_id: int, person_id: int
) -> FaceGroup | None:
    """Attach the group to the owner's person. None when the group or person is missing;
    a person being deleted raises `VoiceDeleting`."""
    # feed.saved_voices reaches feed.signals, which imports this module: a cycle at import time.
    from feed import saved_voices

    if group_detail(conn, owner_id, group_id) is None:
        return None
    person = saved_voices.get_voice(conn, owner_id, person_id)
    if person is None:
        return None
    if person.deleting:
        raise saved_voices.VoiceDeleting("This voice is being deleted.")
    return _set_person(conn, owner_id, group_id, person.id)


def create_person_for_group(
    conn: sqlite3.Connection, owner_id: int, group_id: int, name: str
) -> FaceGroup | None:
    """Create a person (a saved voice) named `name` and attach the group to it. A taken or
    empty name raises the saved-voice error. None when the group is missing."""
    # feed.saved_voices reaches feed.signals, which imports this module: a cycle at import time.
    from feed import saved_voices

    if group_detail(conn, owner_id, group_id) is None:
        return None
    person = saved_voices.create_voice(conn, owner_id, name)
    return _set_person(conn, owner_id, group_id, person.id)


def release_face(conn: sqlite3.Connection, owner_id: int, face_id: int) -> FaceGroup | None:
    """Not this person: move the face into a new pinned group of its own. Returns that
    group; None when the face is missing. The old group is deleted when it is left empty."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        face = conn.execute(
            "SELECT f.group_id FROM faces f JOIN photos p ON p.id = f.photo_id"
            " WHERE f.id = ? AND p.owner_id = ?",
            (face_id, owner_id),
        ).fetchone()
        if face is None:
            conn.execute("ROLLBACK")
            return None
        new_group = conn.execute(
            "INSERT INTO face_groups(owner_id, person_id, created_at) VALUES (?, NULL, ?)",
            (owner_id, _now()),
        ).lastrowid
        conn.execute("UPDATE faces SET group_id = ?, pinned = 1 WHERE id = ?", (new_group, face_id))
        if face["group_id"] is not None:
            conn.execute(
                "DELETE FROM face_groups WHERE id = ?"
                " AND NOT EXISTS (SELECT 1 FROM faces WHERE group_id = face_groups.id)",
                (face["group_id"],),
            )
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")
    return group_detail(conn, owner_id, int(new_group))


def unassign_group(conn: sqlite3.Connection, owner_id: int, group_id: int) -> FaceGroup | None:
    """Detach the group from its person, back to "Faces to name"."""
    if group_detail(conn, owner_id, group_id) is None:
        return None
    return _set_person(conn, owner_id, group_id, None)


def person_groups(conn: sqlite3.Connection, owner_id: int, person_id: int) -> list[FaceGroup]:
    rows = conn.execute(
        f"{_GROUP_SELECT} WHERE g.owner_id = ? AND g.person_id = ? ORDER BY g.id",
        (owner_id, person_id),
    ).fetchall()
    return _groups(conn, rows)


def people_in_photos(
    conn: sqlite3.Connection, owner_id: int, photo_ids: Sequence[int]
) -> dict[int, list[str]]:
    """Per photo, the names of the people recognised in it, each name once, by name.
    A photo with no named face is not a key."""
    found: dict[int, list[str]] = {}
    for row in conn.execute(
        "SELECT DISTINCT f.photo_id, v.name FROM faces f"
        " JOIN face_groups g ON g.id = f.group_id"
        " JOIN saved_voices v ON v.id = g.person_id"
        " WHERE g.owner_id = ? AND f.photo_id IN (SELECT value FROM json_each(?))"
        " ORDER BY f.photo_id, v.name COLLATE NOCASE",
        (owner_id, json.dumps(list(photo_ids))),
    ):
        found.setdefault(row["photo_id"], []).append(row["name"])
    return found


def prune_empty_groups(conn: sqlite3.Connection, owner_id: int) -> int:
    """Delete the owner's groups that hold no face. Returns the number deleted."""
    cursor = conn.execute(
        "DELETE FROM face_groups WHERE owner_id = ?"
        " AND NOT EXISTS (SELECT 1 FROM faces WHERE group_id = face_groups.id)",
        (owner_id,),
    )
    return cursor.rowcount
