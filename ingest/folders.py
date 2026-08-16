"""Upload folders as a first-class list, with outbox-driven deletion (§3.2c).

A "folder" is a distinct `uploads.root_label` for the owner. Deleting one removes
its library photos and ALL their metadata — never the source folder on disk. The
request only enqueues the intent (`folder_deletions`); a worker drains it, so the
cascade is reliable across restarts and never blocks the UI.
"""

import sqlite3
from datetime import datetime, timezone

from ingest.thumbs import thumb_key
from storage.base import Storage


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def list_folders(conn: sqlite3.Connection, owner_id: int) -> list[dict]:
    """Every folder the owner has uploaded, with its live photo count and whether
    a deletion is in progress — the persistent list shown on `/upload`."""
    pending = {
        row["root_label"]
        for row in conn.execute(
            "SELECT root_label FROM folder_deletions WHERE owner_id = ?", (owner_id,)
        )
    }
    rows = conn.execute(
        "SELECT u.root_label AS folder, COUNT(DISTINCT ps.photo_id) AS photos"
        " FROM uploads u LEFT JOIN photo_sources ps ON ps.upload_id = u.id"
        " WHERE u.owner_id = ? GROUP BY u.root_label ORDER BY u.root_label",
        (owner_id,),
    ).fetchall()
    return [
        {"name": r["folder"], "photos": r["photos"], "deleting": r["folder"] in pending}
        for r in rows
    ]


def enqueue_folder_deletion(conn: sqlite3.Connection, owner_id: int, root_label: str) -> None:
    """Record intent to delete a folder from the library (§3.2c). Idempotent."""
    conn.execute(
        "INSERT INTO folder_deletions(owner_id, root_label, requested_at) VALUES (?, ?, ?)"
        " ON CONFLICT(owner_id, root_label) DO NOTHING",
        (owner_id, root_label, _now()),
    )


def process_folder_deletions(
    conn: sqlite3.Connection,
    originals: Storage,
    derived: Storage,
    owner_id: int,
    grid_px: int,
    detail_px: int,
) -> int:
    """Drain the deletion outbox: for each pending folder, remove its library
    photos (and every trace of them) then clear the outbox row. Returns count."""
    pending = conn.execute(
        "SELECT id, root_label FROM folder_deletions WHERE owner_id = ?", (owner_id,)
    ).fetchall()
    for row in pending:
        _delete_folder(conn, originals, derived, owner_id, row["root_label"], grid_px, detail_px)
        conn.execute("DELETE FROM folder_deletions WHERE id = ?", (row["id"],))
    return len(pending)


def _delete_folder(
    conn: sqlite3.Connection,
    originals: Storage,
    derived: Storage,
    owner_id: int,
    root_label: str,
    grid_px: int,
    detail_px: int,
) -> None:
    upload_ids = [
        r["id"] for r in conn.execute(
            "SELECT id FROM uploads WHERE owner_id = ? AND root_label = ?", (owner_id, root_label)
        )
    ]
    if not upload_ids:
        return
    placeholders = ", ".join("?" for _ in upload_ids)
    affected = [
        r["photo_id"] for r in conn.execute(
            f"SELECT DISTINCT photo_id FROM photo_sources WHERE upload_id IN ({placeholders})",
            upload_ids,
        )
    ]
    # Drop the folder's uploads — cascades away its photo_sources rows.
    conn.execute(f"DELETE FROM uploads WHERE id IN ({placeholders})", upload_ids)
    # A photo with no source left belonged only to this folder -> delete it wholly.
    # One still sourced from another folder is kept (only this path was dropped).
    for photo_id in affected:
        if conn.execute(
            "SELECT 1 FROM photo_sources WHERE photo_id = ? LIMIT 1", (photo_id,)
        ).fetchone() is None:
            _delete_photo(conn, originals, derived, photo_id, grid_px, detail_px)


def _delete_photo(
    conn: sqlite3.Connection,
    originals: Storage,
    derived: Storage,
    photo_id: int,
    grid_px: int,
    detail_px: int,
) -> None:
    row = conn.execute(
        "SELECT storage_key, content_hash FROM photos WHERE id = ?", (photo_id,)
    ).fetchone()
    if row is None:
        return
    originals.delete(row["storage_key"])
    for px in (grid_px, detail_px):
        derived.delete(thumb_key(row["content_hash"], px))
    # Virtual tables carry no foreign key, so clear them by rowid explicitly.
    conn.execute("DELETE FROM photo_vec WHERE rowid = ?", (photo_id,))
    conn.execute("DELETE FROM photo_fts WHERE rowid = ?", (photo_id,))
    # The row itself cascades photo_tags, photo_facets, jobs, group_photos, sources.
    conn.execute("DELETE FROM photos WHERE id = ?", (photo_id,))
