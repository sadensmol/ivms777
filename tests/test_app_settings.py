from db.connection import connect, migrate
from db.settings import all_settings, get_setting, set_setting


def _conn(tmp_path):
    conn = connect(tmp_path / "t.db")
    migrate(conn)
    return conn


def test_migrate_creates_app_settings_table(tmp_path):
    conn = _conn(tmp_path)
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(app_settings)")}
    assert cols == {"owner_id", "key", "value", "updated_at"}


def test_missing_setting_is_none(tmp_path):
    conn = _conn(tmp_path)
    assert get_setting(conn, 1, "model_slot.caption") is None
    assert all_settings(conn, 1) == {}


def test_set_then_get_round_trips(tmp_path):
    conn = _conn(tmp_path)
    set_setting(conn, 1, "model_slot.caption", "qwen3-vl-4b")
    assert get_setting(conn, 1, "model_slot.caption") == "qwen3-vl-4b"


def test_set_overwrites_and_refreshes_updated_at(tmp_path):
    conn = _conn(tmp_path)
    set_setting(conn, 1, "model_slot.caption", "a")
    first = conn.execute("SELECT updated_at FROM app_settings").fetchone()["updated_at"]
    set_setting(conn, 1, "model_slot.caption", "b")
    rows = conn.execute("SELECT value, updated_at FROM app_settings").fetchall()
    assert len(rows) == 1  # upsert, not a second row
    assert rows[0]["value"] == "b"
    assert rows[0]["updated_at"] >= first


def test_settings_are_owner_scoped(tmp_path):
    conn = _conn(tmp_path)
    set_setting(conn, 1, "model_slot.caption", "one")
    set_setting(conn, 2, "model_slot.caption", "two")
    assert get_setting(conn, 1, "model_slot.caption") == "one"
    assert get_setting(conn, 2, "model_slot.caption") == "two"
    assert all_settings(conn, 2) == {"model_slot.caption": "two"}
