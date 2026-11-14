import sqlite3
import time
from pathlib import Path

import sqlite_vec

SCHEMA_PATH = Path(__file__).parent / "schema.sql"
SCHEMA_VERSION = 14


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
    conn.execute("PRAGMA busy_timeout=5000")
    _enable_wal(conn)
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _enable_wal(conn: sqlite3.Connection) -> None:
    """Switch the file to WAL (persistent, so a no-op once done). Two connections
    switching a FRESH file at once deadlock on the lock upgrade, and SQLite fails one
    immediately instead of honouring the busy timeout — so retry briefly."""
    for attempt in range(50):
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            return
        except sqlite3.OperationalError as error:
            if "locked" not in str(error) or attempt == 49:
                raise
            time.sleep(0.1)


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
    ("memory_presentations", "decisions_json", "TEXT"),
    ("memory_narrators", "preset_speaker", "TEXT"),
    ("voice_examples", "transcript_confirmed_at", "TEXT"),
    ("voice_examples", "kit_id", "TEXT"),
    ("voice_examples", "transcript_model", "TEXT"),
    ("voice_examples", "transcript_error", "TEXT"),
    ("voice_tests", "likeness_json", "TEXT"),
    ("voice_tests", "likeness_error", "TEXT"),
    ("voice_tests", "prompt_example_id", "INTEGER REFERENCES voice_examples(id) ON DELETE SET NULL"),
)


def migrate(conn: sqlite3.Connection) -> None:
    # app and worker start together and both migrate the same file. Without one
    # write lock around the whole migration, the second sees the first's half-built
    # schema (photos created, user_version still 0) and refuses it as folder-scan.
    conn.execute("BEGIN IMMEDIATE")
    try:
        _migrate_locked(conn)
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


def _migrate_locked(conn: sqlite3.Connection) -> None:
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version == 0 and _has_photos_table(conn):
        raise SchemaTooOldError(
            "This database was built by the folder-scan ingest and has no upgrade "
            "path — photos are now keyed by content hash. Delete the database file "
            "and re-upload."
        )
    # Both steps are idempotent, so migrate always converges the schema regardless
    # of the stamped version: create any missing tables, add any missing columns.
    _rebuild_presentations_for_language(conn)
    # Not executescript: it COMMITs first, which would drop the lock taken above.
    for statement in _statements(SCHEMA_PATH.read_text()):
        conn.execute(statement)
    _ensure_columns(conn)
    _backfill_memory_fts(conn)
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")


def _statements(script: str) -> list[str]:
    """Split a SQL script into statements; `complete_statement` ignores the `;` that
    appear inside comments and string literals."""
    statements, pending = [], ""
    for line in script.splitlines(keepends=True):
        pending += line
        if sqlite3.complete_statement(pending):
            statements.append(pending)
            pending = ""
    return statements


def _rebuild_presentations_for_language(conn: sqlite3.Connection) -> None:
    """Plan 24 keys presentations by `(group_id, language)`; plan 23 keyed them by
    `group_id` alone. `CREATE TABLE IF NOT EXISTS` cannot change a primary key, so the
    old shape is DROPPED rather than migrated — a presentation is derived data and is
    cheaply regenerated, unlike the memories it hangs off (which are untouched)."""
    columns = {r["name"] for r in conn.execute("PRAGMA table_info(memory_presentations)")}
    if columns and "language" not in columns:
        conn.execute("DROP TABLE memory_presentations")


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
