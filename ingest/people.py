"""People and their face groups (design §6, §13): a person is a saved voice
(`saved_voices`), and a face group is attached to one by the owner.

Every query is scoped to the owner. A group or face of another owner reads as missing.
"""

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass


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


def groups_to_name(conn: sqlite3.Connection, owner_id: int) -> list[FaceGroup]:
    """Groups with no person and at least the minimum face count ("Faces to name")."""
    return []


def group_detail(conn: sqlite3.Connection, owner_id: int, group_id: int) -> FaceGroup | None:
    return None


def assign_group(
    conn: sqlite3.Connection, owner_id: int, group_id: int, person_id: int
) -> FaceGroup | None:
    """Attach the group to the owner's person. None when the group or person is missing."""
    return None


def create_person_for_group(
    conn: sqlite3.Connection, owner_id: int, group_id: int, name: str
) -> FaceGroup | None:
    """Create a person (a saved voice) named `name` and attach the group to it. A taken or
    empty name raises the saved-voice error. None when the group is missing."""
    return None


def release_face(conn: sqlite3.Connection, owner_id: int, face_id: int) -> FaceGroup | None:
    """Not this person: move the face into a new pinned group of its own. Returns that
    group; None when the face is missing."""
    return None


def unassign_group(conn: sqlite3.Connection, owner_id: int, group_id: int) -> FaceGroup | None:
    """Detach the group from its person, back to "Faces to name"."""
    return None


def person_groups(conn: sqlite3.Connection, owner_id: int, person_id: int) -> list[FaceGroup]:
    return []


def people_in_photos(
    conn: sqlite3.Connection, owner_id: int, photo_ids: Sequence[int]
) -> dict[int, list[str]]:
    """Per photo, the names of the people recognised in it, each name once."""
    return {}


def prune_empty_groups(conn: sqlite3.Connection, owner_id: int) -> int:
    """Delete the owner's groups that hold no face. Returns the number deleted."""
    return 0
