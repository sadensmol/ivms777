import sqlite3

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


# Schema 16 (design §6): a person is its own record. These build a version 15
# database by hand: the saved voice is the person, and a face group links to a voice.
NOW = "2026-01-01T00:00:00"
V15_SAVED_VOICES = """CREATE TABLE saved_voices (
  id INTEGER PRIMARY KEY AUTOINCREMENT, owner_id INTEGER NOT NULL, name TEXT NOT NULL,
  created_at TEXT NOT NULL, UNIQUE(owner_id, name))"""
V15_FACE_GROUPS = """CREATE TABLE face_groups (
  id INTEGER PRIMARY KEY AUTOINCREMENT, owner_id INTEGER NOT NULL,
  person_id INTEGER REFERENCES saved_voices(id) ON DELETE SET NULL,
  created_at TEXT NOT NULL, updated_at TEXT)"""
V15_FACES = """CREATE TABLE faces (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  photo_id INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
  group_id INTEGER REFERENCES face_groups(id) ON DELETE SET NULL,
  box_x REAL NOT NULL, box_y REAL NOT NULL, box_w REAL NOT NULL, box_h REAL NOT NULL,
  score REAL NOT NULL, vector BLOB NOT NULL, model TEXT NOT NULL,
  pinned INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL)"""
# id, owner, name; voice 9 is being deleted when the migration runs.
V15_VOICES = ((7, 1, "Grandma"), (9, 1, "Бабушка"), (12, 2, "Grandma"))
# id, owner, voice it is attached to (None = waits in "Faces to name").
V15_GROUPS = ((5, 1, 7), (6, 1, None), (8, 2, 12))
# id, group
V15_FACES_ROWS = ((11, 5), (13, 5), (14, 6), (17, 8))


def _v15_db(tmp_path, *, with_faces: bool = True):
    from tests.factories import add_photo

    assert SCHEMA_VERSION == 16
    conn = connect(tmp_path / "v15.db")
    migrate(conn)
    conn.execute("PRAGMA foreign_keys=OFF")
    for table in ("faces", "face_groups", "people"):
        conn.execute(f"DROP TABLE {table}")
    conn.execute("DROP TABLE saved_voices")
    conn.execute(V15_SAVED_VOICES)
    if with_faces:
        conn.execute(V15_FACE_GROUPS)
        conn.execute(V15_FACES)
    conn.execute("PRAGMA foreign_keys=ON")
    for voice_id, owner_id, name in V15_VOICES:
        conn.execute("INSERT INTO saved_voices(id, owner_id, name, created_at) VALUES (?, ?, ?, ?)",
                     (voice_id, owner_id, name, NOW))
    conn.execute(
        "INSERT INTO voice_examples(id, voice_id, language, source, storage_key, content_hash,"
        " seconds, sample_rate, consent_at, created_at)"
        " VALUES (3, 7, 'en', 'recorded', 'k', 'h', 4.0, 48000, ?, ?)", (NOW, NOW))
    conn.execute("INSERT INTO voice_deletions(owner_id, kind, target_id, requested_at)"
                 " VALUES (1, 'voice', 9, ?)", (NOW,))
    if with_faces:
        photos = {owner: add_photo(conn, owner_id=owner, content_hash=f"{owner:064x}")
                  for owner in (1, 2)}
        for group_id, owner_id, voice_id in V15_GROUPS:
            conn.execute("INSERT INTO face_groups(id, owner_id, person_id, created_at)"
                         " VALUES (?, ?, ?, ?)", (group_id, owner_id, voice_id, NOW))
        owners = {group_id: owner_id for group_id, owner_id, _ in V15_GROUPS}
        for face_id, group_id in V15_FACES_ROWS:
            conn.execute(
                "INSERT INTO faces(id, photo_id, group_id, box_x, box_y, box_w, box_h, score,"
                " vector, model, created_at) VALUES (?, ?, ?, 0.1, 0.1, 0.2, 0.2, 0.9, ?, 'm', ?)",
                (face_id, photos[owners[group_id]], group_id, b"\0" * 8, NOW))
        # A deleted group: its id stays used, so the counter is at 20.
        conn.execute("INSERT INTO face_groups(id, owner_id, created_at) VALUES (20, 1, ?)", (NOW,))
        conn.execute("DELETE FROM face_groups WHERE id = 20")
    conn.execute("PRAGMA user_version = 15")
    return conn


def _snapshot(conn) -> dict[str, list[tuple]]:
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
        " AND name IN ('people', 'saved_voices', 'voice_examples', 'face_groups', 'faces')")]
    return {table: [tuple(r) for r in conn.execute(f"SELECT * FROM {table} ORDER BY 1")]
            for table in sorted(tables)}


def _parents(conn, table: str, column: str) -> set[str]:
    return {r["table"] for r in conn.execute(f"PRAGMA foreign_key_list({table})")
            if r["from"] == column}


def test_schema_16_has_people_and_links_voices_and_face_groups_to_them(tmp_path):
    assert SCHEMA_VERSION == 16
    conn = connect(tmp_path / "fresh.db")
    migrate(conn)

    people = {r["name"] for r in conn.execute("PRAGMA table_info(people)")}

    assert conn.execute("PRAGMA user_version").fetchone()[0] == 16
    assert {"id", "owner_id", "name", "created_at", "updated_at"} <= people
    assert _parents(conn, "saved_voices", "person_id") == {"people"}
    assert _parents(conn, "face_groups", "person_id") == {"people"}


def test_a_person_name_is_unique_per_owner(tmp_path):
    assert SCHEMA_VERSION == 16
    conn = connect(tmp_path / "fresh.db")
    migrate(conn)
    insert = "INSERT INTO people(owner_id, name, created_at) VALUES (?, ?, ?)"
    conn.execute(insert, (1, "Ann", NOW))
    conn.execute(insert, (2, "Ann", NOW))

    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(insert, (1, "Ann", NOW))


def test_migrating_v15_gives_each_voice_a_person_of_its_name_and_owner(tmp_path):
    conn = _v15_db(tmp_path)

    migrate(conn)

    voices = conn.execute(
        "SELECT v.id, v.owner_id, v.name, pe.owner_id AS person_owner, pe.name AS person_name,"
        " v.person_id FROM saved_voices v JOIN people pe ON pe.id = v.person_id ORDER BY v.id"
    ).fetchall()
    assert [(v["id"], v["owner_id"], v["name"]) for v in voices] == list(V15_VOICES)
    assert all((v["person_owner"], v["person_name"]) == (v["owner_id"], v["name"])
               for v in voices)
    assert len({v["person_id"] for v in voices}) == len(V15_VOICES)
    assert conn.execute("SELECT COUNT(*) FROM people").fetchone()[0] == len(V15_VOICES)


def test_migrating_v15_attaches_each_face_group_to_its_voices_person_and_keeps_every_id(tmp_path):
    conn = _v15_db(tmp_path)

    migrate(conn)

    person_of_voice = dict(conn.execute("SELECT id, person_id FROM saved_voices").fetchall())
    groups = [tuple(r) for r in conn.execute(
        "SELECT id, owner_id, person_id FROM face_groups ORDER BY id")]
    faces = [tuple(r) for r in conn.execute("SELECT id, group_id FROM faces ORDER BY id")]
    examples = [tuple(r) for r in conn.execute("SELECT id, voice_id FROM voice_examples")]
    assert groups == [(g, o, person_of_voice[v] if v else None) for g, o, v in V15_GROUPS]
    assert faces == list(V15_FACES_ROWS)
    assert examples == [(3, 7)]
    assert _parents(conn, "face_groups", "person_id") == {"people"}
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_a_group_made_after_the_migration_never_reuses_a_deleted_group_id(tmp_path):
    conn = _v15_db(tmp_path)
    migrate(conn)

    new = conn.execute(
        "INSERT INTO face_groups(owner_id, created_at) VALUES (1, ?)", (NOW,)).lastrowid

    assert new == 21


def test_a_second_migrate_of_a_v15_database_changes_nothing(tmp_path):
    conn = _v15_db(tmp_path)
    migrate(conn)
    first = _snapshot(conn)

    migrate(conn)

    assert first["people"] != []
    assert _snapshot(conn) == first


def test_a_v15_database_with_no_face_tables_migrates(tmp_path):
    conn = _v15_db(tmp_path, with_faces=False)

    migrate(conn)
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    people = conn.execute("SELECT COUNT(*) FROM saved_voices WHERE person_id IS NOT NULL").fetchone()

    assert {"people", "face_groups", "faces"} <= names
    assert people[0] == len(V15_VOICES)
    assert _parents(conn, "face_groups", "person_id") == {"people"}
