import sqlite3
from pathlib import Path

import sqlite_vec

SCHEMA_PATH = Path(__file__).parent / "schema.sql"
SCHEMA_VERSION = 2


class SchemaTooOldError(RuntimeError):
    """The database predates the upload schema and cannot be migrated in place."""


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # FastAPI runs sync route handlers in a threadpool, so the connection is used
    # from threads other than the one that created it. Python's sqlite3 serializes
    # calls internally (default SQLITE_THREADSAFE=1), and every statement here is
    # autocommitted, so sharing one connection is safe at this scale.
    conn = sqlite3.connect(db_path, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _has_photos_table(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'photos'"
    ).fetchone()
    return row is not None


def migrate(conn: sqlite3.Connection) -> None:
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version == SCHEMA_VERSION:
        return
    if version == 0 and _has_photos_table(conn):
        raise SchemaTooOldError(
            "This database was built by the folder-scan ingest and has no upgrade "
            "path — photos are now keyed by content hash. Delete the database file "
            "and re-upload."
        )
    conn.executescript(SCHEMA_PATH.read_text())
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
