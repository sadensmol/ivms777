from chat.prefs import ChatPrefs, get_prefs, set_prefs
from db.connection import connect, migrate


def _conn(tmp_path):
    conn = connect(tmp_path / "t.db")
    migrate(conn)
    return conn


def test_migrate_creates_chat_prefs_table(tmp_path):
    conn = _conn(tmp_path)
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(chat_prefs)")}
    assert cols == {"owner_id", "guardrails", "direct_answers"}


def test_defaults_when_no_row(tmp_path):
    conn = _conn(tmp_path)
    assert get_prefs(conn, 1) == ChatPrefs(guardrails=False, direct_answers=True)


def test_set_then_get_round_trips(tmp_path):
    conn = _conn(tmp_path)
    set_prefs(conn, 1, guardrails=True, direct_answers=False)
    assert get_prefs(conn, 1) == ChatPrefs(guardrails=True, direct_answers=False)


def test_set_upserts_second_write(tmp_path):
    conn = _conn(tmp_path)
    set_prefs(conn, 1, guardrails=True, direct_answers=True)
    set_prefs(conn, 1, guardrails=False, direct_answers=False)
    assert get_prefs(conn, 1) == ChatPrefs(guardrails=False, direct_answers=False)
