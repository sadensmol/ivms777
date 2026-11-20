"""People and their face groups (design §6, §13): a person is a `people` row of the owner.
A saved voice and a face group each belong to one person.

Every query is scoped to the owner. A person, group or face of another owner reads as missing.
"""

import json
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from feed.saved_voices import SavedVoice

DEFAULT_MIN_FACES = 2
NAME_MAX = 80


class PersonError(ValueError):
    """A message shown to the user as it is."""


class PersonNameTaken(PersonError):
    """Another person of the owner has this name; the route answers 409."""


class PersonHasVoices(PersonError):
    """The person still owns a saved voice, one being deleted included; the route answers 409."""


class PersonNotFound(PersonError):
    """No such person for this owner; the route answers 404."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Person:
    id: int
    name: str
    voice_names: tuple[str, ...]
    face_count: int


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


def _rollback(conn: sqlite3.Connection) -> None:
    if conn.in_transaction:
        conn.execute("ROLLBACK")


def _people(conn: sqlite3.Connection, owner_id: int, person_id: int | None = None) -> list[Person]:
    where, args = "pe.owner_id = ?", [owner_id]
    if person_id is not None:
        where, args = f"{where} AND pe.id = ?", [owner_id, person_id]
    rows = conn.execute(
        "SELECT pe.id, pe.name,"
        " (SELECT COUNT(*) FROM faces f JOIN face_groups g ON g.id = f.group_id"
        "  WHERE g.person_id = pe.id AND g.owner_id = pe.owner_id) AS face_count"
        f" FROM people pe WHERE {where}",
        args,
    ).fetchall()
    voices: dict[int, list[str]] = {}
    for row in conn.execute(
        "SELECT v.person_id, v.name FROM saved_voices v JOIN people pe ON pe.id = v.person_id"
        f" WHERE {where} AND v.owner_id = pe.owner_id",
        args,
    ):
        voices.setdefault(row["person_id"], []).append(row["name"])
    people = [
        Person(
            id=row["id"],
            name=row["name"],
            voice_names=tuple(sorted(voices.get(row["id"], []), key=str.casefold)),
            face_count=row["face_count"],
        )
        for row in rows
    ]
    # Python, not COLLATE NOCASE: SQLite folds ASCII only, and names are often Cyrillic.
    return sorted(people, key=lambda person: (person.name.casefold(), person.id))


def list_people(conn: sqlite3.Connection, owner_id: int) -> list[Person]:
    """Every person of the owner, by name."""
    return _people(conn, owner_id)


def get_person(conn: sqlite3.Connection, owner_id: int, person_id: int) -> Person | None:
    found = _people(conn, owner_id, person_id)
    return found[0] if found else None


def _clean_name(name: str) -> str:
    clean = name.strip()
    if not clean:
        raise PersonError("Give the person a name.")
    if len(clean) > NAME_MAX:
        raise PersonError(f"A person's name can have at most {NAME_MAX} characters.")
    return clean


def _check_name_free(
    conn: sqlite3.Connection, owner_id: int, name: str, own_id: int | None = None
) -> None:
    """Unique per owner by `casefold()`: the table's UNIQUE is case-sensitive, so the
    real check is here, under the caller's write lock."""
    folded = name.casefold()
    for row in conn.execute("SELECT id, name FROM people WHERE owner_id = ?", (owner_id,)):
        if row["id"] != own_id and row["name"].casefold() == folded:
            raise PersonNameTaken(f"A person named «{row['name']}» already exists.")


def _person_exists(conn: sqlite3.Connection, owner_id: int, person_id: int) -> bool:
    found = conn.execute(
        "SELECT 1 FROM people WHERE id = ? AND owner_id = ?", (person_id, owner_id)
    ).fetchone()
    return found is not None


def insert_person(conn: sqlite3.Connection, owner_id: int, name: str) -> int:
    """Add a person inside the caller's write transaction (`BEGIN IMMEDIATE`) and return its
    id. An empty, too long or taken name raises `PersonError`."""
    clean = _clean_name(name)
    _check_name_free(conn, owner_id, clean)
    cursor = conn.execute(
        "INSERT INTO people(owner_id, name, created_at) VALUES (?, ?, ?)",
        (owner_id, clean, _now()),
    )
    return int(cursor.lastrowid or 0)


def check_person(conn: sqlite3.Connection, owner_id: int, person_id: int) -> None:
    """Inside the caller's write transaction: raise `PersonNotFound` unless the owner has
    this person."""
    if not _person_exists(conn, owner_id, person_id):
        raise PersonNotFound("This person does not exist.")


def create_person(conn: sqlite3.Connection, owner_id: int, name: str) -> Person:
    """An empty, too long or taken name raises `PersonError`, and nothing is written."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        person_id = insert_person(conn, owner_id, name)
    except BaseException:
        _rollback(conn)
        raise
    conn.execute("COMMIT")
    return Person(id=person_id, name=name.strip(), voice_names=(), face_count=0)


def rename_person(conn: sqlite3.Connection, owner_id: int, person_id: int, name: str) -> Person:
    """A missing person raises `PersonNotFound`; an empty, too long or taken name `PersonError`."""
    clean = _clean_name(name)
    conn.execute("BEGIN IMMEDIATE")
    try:
        check_person(conn, owner_id, person_id)
        _check_name_free(conn, owner_id, clean, own_id=person_id)
        conn.execute(
            "UPDATE people SET name = ?, updated_at = ? WHERE id = ?", (clean, _now(), person_id)
        )
    except BaseException:
        _rollback(conn)
        raise
    conn.execute("COMMIT")
    person = get_person(conn, owner_id, person_id)
    assert person is not None  # renamed under the lock a moment ago; only delete_person deletes
    return person


def delete_person(conn: sqlite3.Connection, owner_id: int, person_id: int) -> None:
    """Delete the person; the foreign key returns their face groups to "Faces to name". No
    photo, face or voice is deleted. Refused with `PersonHasVoices` while a voice row names
    them, one being deleted included. A missing person raises `PersonNotFound`."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        check_person(conn, owner_id, person_id)
        voice = conn.execute(
            "SELECT name FROM saved_voices WHERE person_id = ? ORDER BY id LIMIT 1", (person_id,)
        ).fetchone()
        if voice is not None:
            raise PersonHasVoices(
                f"This person still has the voice «{voice['name']}». Move or delete it first."
            )
        conn.execute("DELETE FROM people WHERE id = ? AND owner_id = ?", (person_id, owner_id))
    except BaseException:
        _rollback(conn)
        raise
    conn.execute("COMMIT")


def person_voices(conn: sqlite3.Connection, owner_id: int, person_id: int) -> list["SavedVoice"]:
    """The person's saved voices, by name; pending ones too, with `deleting=True`."""
    # feed.saved_voices reaches feed.signals, which imports this module: a cycle at import time.
    from feed import saved_voices

    return [v for v in saved_voices.list_voices(conn, owner_id) if v.person_id == person_id]


def person_photos(conn: sqlite3.Connection, owner_id: int, person_id: int) -> list[int]:
    """The ids of the owner's photos that hold a face of the person, each once, the oldest
    shot first; a photo with no shot time comes last."""
    return [
        row["id"]
        for row in conn.execute(
            "SELECT DISTINCT p.id, p.shot_at FROM photos p"
            " JOIN faces f ON f.photo_id = p.id JOIN face_groups g ON g.id = f.group_id"
            " WHERE p.owner_id = ? AND g.owner_id = ? AND g.person_id = ?"
            " ORDER BY p.shot_at IS NULL, p.shot_at, p.id",
            (owner_id, owner_id, person_id),
        )
    ]


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
    "SELECT g.id, g.person_id, pe.name AS person_name FROM face_groups g"
    " LEFT JOIN people pe ON pe.id = g.person_id"
)


def groups_to_name(
    conn: sqlite3.Connection, owner_id: int, *, min_faces: int = DEFAULT_MIN_FACES
) -> list[FaceGroup]:
    """Groups with no person ("Faces to name"), the largest first: each one with at least
    `min_faces` faces, and each one holding a face the owner released ("Not this person")
    whatever its size, so a released face can still be named."""
    rows = conn.execute(
        "SELECT g.id, g.person_id, NULL AS person_name FROM face_groups g"
        " JOIN faces f ON f.group_id = g.id"
        " WHERE g.owner_id = ? AND g.person_id IS NULL"
        " GROUP BY g.id HAVING COUNT(*) >= ? OR MAX(f.pinned) = 1"
        " ORDER BY COUNT(*) DESC, g.id",
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
) -> None:
    conn.execute(
        "UPDATE face_groups SET person_id = ?, updated_at = ? WHERE id = ? AND owner_id = ?",
        (person_id, _now(), group_id, owner_id),
    )


def _group_exists(conn: sqlite3.Connection, owner_id: int, group_id: int) -> bool:
    found = conn.execute(
        "SELECT 1 FROM face_groups WHERE id = ? AND owner_id = ?", (group_id, owner_id)
    ).fetchone()
    return found is not None


def assign_group(
    conn: sqlite3.Connection, owner_id: int, group_id: int, person_id: int
) -> FaceGroup | None:
    """Attach the group to the owner's person. None when the group or the person is missing."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        if not (
            _group_exists(conn, owner_id, group_id) and _person_exists(conn, owner_id, person_id)
        ):
            conn.execute("ROLLBACK")
            return None
        _set_person(conn, owner_id, group_id, person_id)
    except BaseException:
        _rollback(conn)
        raise
    conn.execute("COMMIT")
    return group_detail(conn, owner_id, group_id)


def create_person_for_group(
    conn: sqlite3.Connection, owner_id: int, group_id: int, name: str
) -> FaceGroup | None:
    """Create a person named `name` (a `people` row only, never a voice) and attach the group
    to them. An empty, too long or taken name raises `PersonError`, and nothing is written.
    None when the group is missing."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        if not _group_exists(conn, owner_id, group_id):
            conn.execute("ROLLBACK")
            return None
        _set_person(conn, owner_id, group_id, insert_person(conn, owner_id, name))
    except BaseException:
        _rollback(conn)
        raise
    conn.execute("COMMIT")
    return group_detail(conn, owner_id, group_id)


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
        _rollback(conn)
        raise
    conn.execute("COMMIT")
    return group_detail(conn, owner_id, int(new_group))


def unassign_group(conn: sqlite3.Connection, owner_id: int, group_id: int) -> FaceGroup | None:
    """Detach the group from its person, back to "Faces to name"."""
    if group_detail(conn, owner_id, group_id) is None:
        return None
    _set_person(conn, owner_id, group_id, None)
    return group_detail(conn, owner_id, group_id)


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
        "SELECT DISTINCT f.photo_id, pe.name FROM faces f"
        " JOIN face_groups g ON g.id = f.group_id"
        " JOIN people pe ON pe.id = g.person_id"
        " WHERE g.owner_id = ? AND f.photo_id IN (SELECT value FROM json_each(?))"
        " ORDER BY f.photo_id, pe.name COLLATE NOCASE",
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
