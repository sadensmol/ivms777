# models/lease_store.py
"""The cross-process model lease (design §8.1). A single row (id=1) means held;
no row means idle. app and worker are separate processes; this shared-DB row is
their only coordination channel."""
import sqlite3
from typing import Literal, TypedDict

WorkloadName = Literal["CHAT", "MEMORY_REBUILD", "INGEST_EMBED", "INGEST_CAPTION", "SEARCH"]
INTERACTIVE: frozenset[WorkloadName] = frozenset({"CHAT", "MEMORY_REBUILD", "SEARCH"})


class Lease(TypedDict):
    id: int
    holder: str
    workload: str
    priority: int
    heartbeat: str
    preempt_requested: int


def read_lease(conn: sqlite3.Connection) -> Lease | None:
    row = conn.execute("SELECT * FROM model_lease WHERE id = 1").fetchone()
    return dict(row) if row is not None else None  # type: ignore[return-value]


def try_acquire(conn: sqlite3.Connection, holder: str, workload: WorkloadName, priority: int) -> bool:
    """Insert the single lease row iff none is held. Returns whether we now hold it.
    Atomic because it is exactly one statement under autocommit (isolation_level=None):
    INSERT OR IGNORE on the id=1 PK."""
    cur = conn.execute(
        "INSERT OR IGNORE INTO model_lease (id, holder, workload, priority, preempt_requested)"
        " VALUES (1, ?, ?, ?, 0)",
        (holder, workload, priority),
    )
    return cur.rowcount == 1


def release(conn: sqlite3.Connection, holder: str) -> None:
    conn.execute("DELETE FROM model_lease WHERE id = 1 AND holder = ?", (holder,))


def request_preempt(conn: sqlite3.Connection) -> None:
    conn.execute("UPDATE model_lease SET preempt_requested = 1 WHERE id = 1")


def preempt_requested(conn: sqlite3.Connection) -> bool:
    row = conn.execute("SELECT preempt_requested FROM model_lease WHERE id = 1").fetchone()
    return bool(row["preempt_requested"]) if row is not None else False


def heartbeat(conn: sqlite3.Connection, holder: str) -> None:
    conn.execute(
        "UPDATE model_lease SET heartbeat = CURRENT_TIMESTAMP WHERE id = 1 AND holder = ?",
        (holder,),
    )


def reclaim_stale(conn: sqlite3.Connection, max_age_s: float) -> bool:
    """Delete the lease row iff its holder has gone SILENT — its heartbeat has not
    been bumped within `max_age_s` seconds (design §8.1). A live holder runs a
    heartbeat thread that keeps the row fresh, so this only ever fires on a holder
    that crashed, was killed, or wedged; without it a process that dies holding the
    lease wedges every future acquire and chat reports "busy" forever. Returns
    whether a row was reclaimed. (`CURRENT_TIMESTAMP` and `datetime('now')` are both
    UTC, so the comparison is timezone-safe.)"""
    cur = conn.execute(
        "DELETE FROM model_lease WHERE id = 1 AND heartbeat <= datetime('now', ?)",
        (f"-{max_age_s} seconds",),
    )
    return cur.rowcount == 1
