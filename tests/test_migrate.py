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


def test_concurrent_first_migrations_do_not_mistake_each_other_for_folder_scan(tmp_path):
    # `make up` starts app and worker together; both migrate the same fresh file. The
    # one that loses the race must wait for the other, not see its half-built schema
    # (photos created, user_version still 0) and refuse it as a folder-scan database.
    import threading

    for attempt in range(20):
        db_path = tmp_path / f"fresh{attempt}.db"
        barrier = threading.Barrier(2)
        errors: list[BaseException] = []

        def boot(db_path=db_path, barrier=barrier, errors=errors):
            barrier.wait(timeout=10)
            try:
                conn = connect(db_path)
                try:
                    migrate(conn)
                finally:
                    conn.close()
            except BaseException as exc:  # noqa: BLE001 — collected for the assert
                errors.append(exc)

        threads = [threading.Thread(target=boot) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
