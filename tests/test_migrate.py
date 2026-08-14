import pytest

from db.connection import SCHEMA_VERSION, SchemaTooOldError, connect, migrate


def test_migrate_self_heals_a_missing_column_at_current_version(tmp_path):
    # The reload-race state: a DB stamped at the current version whose `groups`
    # table predates the `description` column. migrate must repair it, not skip.
    conn = connect(tmp_path / "t.db")
    conn.execute(
        "CREATE TABLE groups (id INTEGER PRIMARY KEY, owner_id INTEGER, kind TEXT,"
        " name TEXT, params TEXT, status TEXT, created_at TEXT)"
    )
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    migrate(conn)
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(groups)")}
    assert "description" in cols


def test_folder_scan_schema_is_still_rejected(tmp_path):
    conn = connect(tmp_path / "old.db")
    conn.execute("CREATE TABLE photos (id INTEGER PRIMARY KEY)")  # user_version stays 0
    with pytest.raises(SchemaTooOldError):
        migrate(conn)
