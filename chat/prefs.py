import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True)
class ChatPrefs:
    """Global per-owner chat toggles (§10). Defaults reproduce today's pipeline:
    guardrails off (chat is a general assistant), direct_answers on (deterministic
    direct-DB step runs)."""

    guardrails: bool = False
    direct_answers: bool = True


def get_prefs(conn: sqlite3.Connection, owner_id: int) -> ChatPrefs:
    """This owner's chat toggles, or the defaults when no row exists yet (§10)."""
    row = conn.execute(
        "SELECT guardrails, direct_answers FROM chat_prefs WHERE owner_id = ?",
        (owner_id,),
    ).fetchone()
    if row is None:
        return ChatPrefs()
    return ChatPrefs(bool(row["guardrails"]), bool(row["direct_answers"]))


def set_prefs(
    conn: sqlite3.Connection, owner_id: int, *, guardrails: bool, direct_answers: bool
) -> None:
    """Persist this owner's chat toggles (upsert), applied to every session (§10)."""
    conn.execute(
        "INSERT INTO chat_prefs(owner_id, guardrails, direct_answers) VALUES (?, ?, ?)"
        " ON CONFLICT(owner_id) DO UPDATE SET"
        " guardrails = excluded.guardrails, direct_answers = excluded.direct_answers",
        (owner_id, int(guardrails), int(direct_answers)),
    )
