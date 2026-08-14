from chat.history import (
    add_message,
    answer_html,
    current_session,
    new_session,
    session_messages,
)


def test_current_session_creates_one_when_none(conn):
    sid = current_session(conn, owner_id=1)
    assert isinstance(sid, int)
    assert current_session(conn, owner_id=1) == sid  # stable: returns the latest


def test_new_session_starts_empty_and_becomes_current(conn):
    s1 = current_session(conn, 1)
    add_message(conn, s1, "q", "a", [1])
    s2 = new_session(conn, 1)
    assert s2 != s1
    assert current_session(conn, 1) == s2
    assert session_messages(conn, s2) == []


def test_messages_persist_with_sources_and_render(conn):
    sid = current_session(conn, 1)
    add_message(conn, sid, "what beaches?", "Here [photo:7].", [7, 9])
    msgs = session_messages(conn, sid)
    assert len(msgs) == 1
    assert msgs[0]["question"] == "what beaches?"
    assert msgs[0]["sources"] == [7, 9]
    assert "/thumb/7" in msgs[0]["answer_html"]       # citation rendered
    assert "[photo:7]" not in msgs[0]["answer_html"]


def test_answer_html_escapes_and_cites():
    out = answer_html("<b>hi</b> [photo:3]")
    assert "&lt;b&gt;" in out            # escaped, not live HTML
    assert "/thumb/3" in out
