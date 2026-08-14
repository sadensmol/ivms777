"""Row builders for tests. Keep every INSERT INTO photos in here."""

import sqlite3
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
