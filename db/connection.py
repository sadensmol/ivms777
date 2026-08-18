import sqlite3
from pathlib import Path

import sqlite_vec

SCHEMA_PATH = Path(__file__).parent / "schema.sql"
SCHEMA_VERSION = 10


class SchemaTooOldError(RuntimeError):
    """The database predates the upload schema and cannot be migrated in place."""


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # One connection PER THREAD (see AppContext): sharing a single sqlite3
    # connection across FastAPI's threadpool corrupts it, because the Python
    # connection/cursor objects are not safe for concurrent use even though the C
    # library is. `check_same_thread=False` only silences the guard — the worker
    # loop and each request thread each own their connection. Statements are
    # autocommitted (isolation_level=None); WAL + the busy timeout serialise
    # writers across the several connections that open the same file.
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


# Columns added to a table after its first shipped version. Ensured on every
# migrate (not gated on user_version), so a DB that reached the current version
# WITHOUT a column — e.g. a schema edit that landed during an app --reload,
# between the version bump and the column's ALTER — still self-heals instead of
# getting stuck. `CREATE TABLE IF NOT EXISTS` never adds a column to an existing
# table, so column drift has to be repaired explicitly.
_ADDED_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("photos", "ai_title", "TEXT"),
    ("photos", "ai_description", "TEXT"),
    ("photos", "caption_vec", "BLOB"),
    ("groups", "description", "TEXT"),
    ("chat_messages", "elapsed_ms", "INTEGER"),
)


def migrate(conn: sqlite3.Connection) -> None:
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version == 0 and _has_photos_table(conn):
        raise SchemaTooOldError(
            "This database was built by the folder-scan ingest and has no upgrade "
            "path — photos are now keyed by content hash. Delete the database file "
            "and re-upload."
        )
    # Both steps are idempotent, so migrate always converges the schema regardless
    # of the stamped version: create any missing tables, add any missing columns.
    conn.executescript(SCHEMA_PATH.read_text())
    _ensure_columns(conn)
    _backfill_memory_fts(conn)
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")


def _backfill_memory_fts(conn: sqlite3.Connection) -> None:
    """Index any memory that predates `memory_fts` (built before this column
    existed), so chat's `find_memory` works without a forced rebuild. Idempotent:
    only memories missing from the index are inserted."""
    conn.execute(
        "INSERT INTO memory_fts(rowid, name, description)"
        " SELECT g.id, g.name, COALESCE(g.description, '') FROM groups g"
        " WHERE g.kind = 'memory'"
        "   AND NOT EXISTS (SELECT 1 FROM memory_fts m WHERE m.rowid = g.id)"
    )


def _ensure_columns(conn: sqlite3.Connection) -> None:
    tables = _table_names(conn)
    for table, column, coltype in _ADDED_COLUMNS:
        if table not in tables:
            continue
        cols = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")


def _table_names(conn: sqlite3.Connection) -> set[str]:
    return {r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
