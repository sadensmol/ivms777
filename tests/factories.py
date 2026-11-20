"""Row builders for tests. Keep every INSERT INTO photos in here."""

import sqlite3
import struct
from collections.abc import Sequence

from storage.keys import content_key

NOW = "2026-01-01T00:00:00"


def add_upload(
    conn: sqlite3.Connection, *, owner_id: int = 1, root_label: str = "Pictures"
) -> int:
    cursor = conn.execute(
        "INSERT INTO uploads(owner_id, root_label, started_at) VALUES (?, ?, ?)",
        (owner_id, root_label, NOW),
    )
    return int(cursor.lastrowid)


def add_photo(
    conn: sqlite3.Connection,
    *,
    photo_id: int | None = None,
    owner_id: int = 1,
    content_hash: str,
    suffix: str = ".jpg",
    sources: Sequence[str] = (),
    upload_id: int | None = None,
    **columns: object,
) -> int:
    """Insert one photo and its source paths. `columns` sets any other photo column."""
    fields: dict[str, object] = {
        "owner_id": owner_id,
        "content_hash": content_hash,
        "storage_key": content_key(content_hash, suffix),
        "created_at": NOW,
        "updated_at": NOW,
        **columns,
    }
    if photo_id is not None:
        fields["id"] = photo_id
    names = ", ".join(fields)
    placeholders = ", ".join("?" for _ in fields)
    cursor = conn.execute(
        f"INSERT INTO photos({names}) VALUES ({placeholders})", tuple(fields.values())
    )
    new_id = int(cursor.lastrowid)
    if sources:
        if upload_id is None:
            upload_id = add_upload(conn, owner_id=owner_id)
        for rel_path in sources:
            conn.execute(
                "INSERT INTO photo_sources(photo_id, upload_id, rel_path, filename)"
                " VALUES (?, ?, ?, ?)",
                (new_id, upload_id, rel_path, rel_path.rsplit("/", 1)[-1]),
            )
    return new_id


def add_person(conn: sqlite3.Connection, *, owner_id: int = 1, name: str) -> int:
    """Insert one person (design §6: a person is its own record, not a saved voice)."""
    has_people = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'people'"
    ).fetchone()
    assert has_people is not None, "the schema has no people table"
    cursor = conn.execute(
        "INSERT INTO people(owner_id, name, created_at) VALUES (?, ?, ?)", (owner_id, name, NOW)
    )
    return int(cursor.lastrowid)


def add_face_group(
    conn: sqlite3.Connection, *, owner_id: int = 1, person_id: int | None = None
) -> int:
    cursor = conn.execute(
        "INSERT INTO face_groups(owner_id, person_id, created_at) VALUES (?, ?, ?)",
        (owner_id, person_id, NOW),
    )
    return int(cursor.lastrowid)


def add_face(
    conn: sqlite3.Connection,
    *,
    photo_id: int,
    group_id: int | None,
    vector: Sequence[float],
    box: tuple[float, float, float, float] = (0.25, 0.25, 0.5, 0.5),
    pinned: bool = False,
) -> int:
    """Insert one face; `vector` is stored as float32 bytes, the `faces.vector` format."""
    cursor = conn.execute(
        "INSERT INTO faces(photo_id, group_id, box_x, box_y, box_w, box_h, score, vector,"
        " model, pinned, created_at) VALUES (?, ?, ?, ?, ?, ?, 0.9, ?, 'buffalo-l@test', ?, ?)",
        (photo_id, group_id, *box, struct.pack(f"<{len(vector)}f", *vector), int(pinned), NOW),
    )
    return int(cursor.lastrowid)
