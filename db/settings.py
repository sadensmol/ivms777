"""`app_settings` — owner-level settings the UI can change (design §6).

Today it holds the four model-slot choices (`models/slots.py`). A MISSING row means
"fall back to the env override, else the profile default" — a default is never
written out, so an untouched install and a fresh library resolve identically.
"""

import sqlite3
from datetime import UTC, datetime


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def get_setting(conn: sqlite3.Connection, owner_id: int, key: str) -> str | None:
    row = conn.execute(
        "SELECT value FROM app_settings WHERE owner_id = ? AND key = ?", (owner_id, key)
    ).fetchone()
    return row["value"] if row is not None else None


def set_setting(conn: sqlite3.Connection, owner_id: int, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO app_settings(owner_id, key, value, updated_at) VALUES (?, ?, ?, ?)"
        " ON CONFLICT(owner_id, key) DO UPDATE SET value = excluded.value,"
        " updated_at = excluded.updated_at",
        (owner_id, key, value, _now()),
    )


def all_settings(conn: sqlite3.Connection, owner_id: int) -> dict[str, str]:
    return {
        row["key"]: row["value"]
        for row in conn.execute(
            "SELECT key, value FROM app_settings WHERE owner_id = ?", (owner_id,)
        )
    }
