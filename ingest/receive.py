import hashlib
import io
import json
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import PurePosixPath

from PIL import Image, UnidentifiedImageError

from ingest.exif import read_exif
from ingest.facets import derive_facets, store_facets
from ingest.jobs import enqueue
from storage.base import Storage
from storage.keys import content_key, suffix_of


class HashMismatchError(ValueError):
    """The bytes received do not hash to what the client declared."""


class UnreadableImageError(ValueError):
    """The bytes are not an image this build can open."""


@dataclass(frozen=True)
class ReceiveResult:
    photo_id: int
    content_hash: str
    created: bool
    source_added: bool


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def known_hashes(conn: sqlite3.Connection, owner_id: int, hashes: Iterable[str]) -> set[str]:
    """Which of these hashes this owner already has. Drives the upload probe."""
    wanted = list(dict.fromkeys(hashes))
    found: set[str] = set()
    # SQLite's parameter limit is 999 by default, so ask in chunks.
    for start in range(0, len(wanted), 500):
        chunk = wanted[start : start + 500]
        placeholders = ", ".join("?" for _ in chunk)
        rows = conn.execute(
            f"SELECT content_hash FROM photos WHERE owner_id = ?"
            f" AND content_hash IN ({placeholders})",
            (owner_id, *chunk),
        )
        found.update(row["content_hash"] for row in rows)
    return found


def _add_source(
    conn: sqlite3.Connection,
    photo_id: int,
    upload_id: int,
    rel_path: str,
    mtime: float | None,
) -> bool:
    cursor = conn.execute(
        "INSERT INTO photo_sources(photo_id, upload_id, rel_path, filename, mtime)"
        " VALUES (?, ?, ?, ?, ?) ON CONFLICT(photo_id, rel_path) DO NOTHING",
        (photo_id, upload_id, rel_path, PurePosixPath(rel_path).name, mtime),
    )
    return cursor.rowcount == 1


def link_existing(
    conn: sqlite3.Connection,
    *,
    owner_id: int,
    upload_id: int,
    rel_path: str,
    content_hash: str,
    mtime: float | None = None,
) -> int | None:
    """Record another local path for content already held. Returns None if unknown.

    This is what keeps duplicate detection honest when the probe tells the client
    not to send bytes it already has — the path still has to be recorded.
    """
    row = conn.execute(
        "SELECT id FROM photos WHERE owner_id = ? AND content_hash = ?",
        (owner_id, content_hash),
    ).fetchone()
    if row is None:
        return None
    _add_source(conn, int(row["id"]), upload_id, rel_path, mtime)
    return int(row["id"])


def receive(
    conn: sqlite3.Connection,
    originals: Storage,
    *,
    owner_id: int,
    upload_id: int,
    rel_path: str,
    declared_hash: str,
    data: bytes,
    mtime: float | None = None,
) -> ReceiveResult:
    """Turn uploaded bytes into a photo, or another source for one already held."""
    digest = hashlib.sha256(data).hexdigest()
    if digest != declared_hash.lower():
        raise HashMismatchError(f"declared {declared_hash!r}, received {digest!r}")

    existing = conn.execute(
        "SELECT id FROM photos WHERE owner_id = ? AND content_hash = ?", (owner_id, digest)
    ).fetchone()
    if existing is not None:
        photo_id = int(existing["id"])
        return ReceiveResult(
            photo_id=photo_id,
            content_hash=digest,
            created=False,
            source_added=_add_source(conn, photo_id, upload_id, rel_path, mtime),
        )

    # Open before storing, so a non-image never leaves a file or a row behind.
    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise UnreadableImageError(f"{rel_path}: {error}") from error

    key = content_key(digest, suffix_of(rel_path))
    originals.write(key, data)

    local = originals.local_path(key)
    facts = read_exif(local) if local is not None else None
    now = _now()
    cursor = conn.execute(
        "INSERT INTO photos(owner_id, content_hash, storage_key, bytes, width, height,"
        " shot_at, camera, lens, gps_lat, gps_lon, exif_json, created_at, updated_at)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            owner_id, digest, key, len(data),
            facts.width if facts else None,
            facts.height if facts else None,
            facts.shot_at if facts else None,
            facts.camera if facts else None,
            facts.lens if facts else None,
            facts.gps_lat if facts else None,
            facts.gps_lon if facts else None,
            json.dumps(facts.raw) if facts else None,
            now, now,
        ),
    )
    photo_id = int(cursor.lastrowid)
    if facts is not None:
        store_facets(
            conn, photo_id, derive_facets(facts, width=facts.width, height=facts.height)
        )
    added = _add_source(conn, photo_id, upload_id, rel_path, mtime)
    enqueue(conn, photo_id, "thumbnail")
    enqueue(conn, photo_id, "embed")
    enqueue(conn, photo_id, "taxonomy")
    enqueue(conn, photo_id, "caption")
    return ReceiveResult(
        photo_id=photo_id, content_hash=digest, created=True, source_added=added
    )
