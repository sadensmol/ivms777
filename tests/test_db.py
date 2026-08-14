import sqlite3

import pytest

from db.connection import SchemaTooOldError, connect, migrate
from tests.factories import add_photo


def test_wal_and_busy_timeout_are_set(conn):
    assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000


def test_sqlite_vec_extension_is_loaded(conn):
    version = conn.execute("SELECT vec_version()").fetchone()[0]
    assert isinstance(version, str)


def test_expected_tables_exist(conn):
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view')").fetchall()
    names = {r[0] for r in rows}
    for expected in [
        "photos", "photo_sources", "uploads", "tags", "photo_tags",
        "photo_facets", "jobs", "groups", "group_photos",
    ]:
        assert expected in names
    assert "scans" not in names
    assert "app_settings" not in names


def test_vector_table_accepts_and_returns_a_row(conn):
    conn.execute("INSERT INTO photo_vec(rowid, embedding) VALUES (1, ?)", (b"\x00" * (1152 * 4),))
    count = conn.execute("SELECT count(*) FROM photo_vec").fetchone()[0]
    assert count == 1


def test_migrate_is_idempotent(settings):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    c = connect(settings.db_path)
    migrate(c)
    migrate(c)
    assert c.execute("SELECT count(*) FROM photos").fetchone()[0] == 0
    c.close()


def test_photos_are_unique_per_owner_and_hash(conn):
    add_photo(conn, content_hash="aa" * 32)
    with pytest.raises(sqlite3.IntegrityError):
        add_photo(conn, content_hash="aa" * 32)


def test_a_photo_can_have_many_sources_but_each_path_once(conn):
    photo_id = add_photo(conn, content_hash="bb" * 32, sources=("a/one.jpg", "b/two.jpg"))
    assert conn.execute(
        "SELECT count(*) FROM photo_sources WHERE photo_id = ?", (photo_id,)
    ).fetchone()[0] == 2
    upload_id = conn.execute("SELECT id FROM uploads LIMIT 1").fetchone()["id"]
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO photo_sources(photo_id, upload_id, rel_path, filename)"
            " VALUES (?, ?, 'a/one.jpg', 'one.jpg')",
            (photo_id, upload_id),
        )


def test_sources_vanish_with_their_photo(conn):
    photo_id = add_photo(conn, content_hash="cc" * 32, sources=("x/one.jpg",))
    conn.execute("DELETE FROM photos WHERE id = ?", (photo_id,))
    assert conn.execute("SELECT count(*) FROM photo_sources").fetchone()[0] == 0


def test_an_old_database_is_refused_rather_than_silently_wrong(settings):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    stale = connect(settings.db_path)
    stale.execute("CREATE TABLE photos (id INTEGER PRIMARY KEY, path TEXT)")
    with pytest.raises(SchemaTooOldError):
        migrate(stale)
    stale.close()


def test_deleting_a_photo_cascades_to_jobs(conn):
    add_photo(conn, photo_id=7, content_hash="h")
    conn.execute(
        "INSERT INTO jobs(photo_id, stage, status, updated_at)"
        " VALUES (7, 'thumbnail', 'pending', '2026-01-01')"
    )
    conn.execute("DELETE FROM photos WHERE id = 7")
    assert conn.execute("SELECT count(*) FROM jobs").fetchone()[0] == 0
