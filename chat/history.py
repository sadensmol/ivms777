import html
import json
import re
import sqlite3
from datetime import UTC, datetime

_CITE = re.compile(r"\[photo:(\d+)\]")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def cited_ids(answer: str) -> list[int]:
    """Photo ids the answer actually cites as [photo:ID], first-appearance order.

    These are the evidence the model stood behind — not the loosely-related
    candidates retrieval handed it. They are what the UI shows and what §6 stores
    in `chat_messages.sources`.
    """
    seen: dict[int, None] = {}
    for match in _CITE.finditer(answer):
        seen.setdefault(int(match.group(1)), None)
    return list(seen)


def new_session(conn: sqlite3.Connection, owner_id: int) -> int:
    cursor = conn.execute(
        "INSERT INTO chat_sessions(owner_id, created_at) VALUES (?, ?)",
        (owner_id, _now()),
    )
    return int(cursor.lastrowid)


def current_session(conn: sqlite3.Connection, owner_id: int) -> int:
    """The owner's latest chat session, creating a first one if none exists."""
    row = conn.execute(
        "SELECT id FROM chat_sessions WHERE owner_id = ? ORDER BY id DESC LIMIT 1",
        (owner_id,),
    ).fetchone()
    return int(row["id"]) if row else new_session(conn, owner_id)


def add_message(
    conn: sqlite3.Connection, session_id: int, question: str, answer: str, sources: list[int]
) -> int:
    cursor = conn.execute(
        "INSERT INTO chat_messages(session_id, question, answer, sources, created_at)"
        " VALUES (?, ?, ?, ?, ?)",
        (session_id, question, answer, json.dumps(sources), _now()),
    )
    return int(cursor.lastrowid)


def session_messages(conn: sqlite3.Connection, session_id: int) -> list[dict]:
    """A session's turns, oldest first, with sources parsed and answer rendered."""
    rows = conn.execute(
        "SELECT question, answer, sources FROM chat_messages"
        " WHERE session_id = ? ORDER BY id",
        (session_id,),
    ).fetchall()
    return [
        {
            "question": row["question"],
            "answer_html": answer_html(row["answer"]),
            "sources": json.loads(row["sources"] or "[]"),
        }
        for row in rows
    ]


def answer_html(text: str) -> str:
    """Escaped answer with [photo:ID] turned into a clickable thumbnail — the
    server-render of the same substitution chat.js does live (so history is
    static HTML, not re-streamed). html.escape leaves [ ] intact, so the regex
    still matches after escaping."""
    return _CITE.sub(
        lambda m: f'<a href="/photo/{m.group(1)}?ctx=chat">'
        f'<img class="cite" src="/thumb/{m.group(1)}" alt="photo {m.group(1)}"></a>',
        html.escape(text),
    )
