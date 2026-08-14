import sqlite3
from datetime import UTC, datetime

STAGES: tuple[str, ...] = ("thumbnail", "embed", "taxonomy", "caption")
MAX_ATTEMPTS = 3
STATUSES = ("pending", "running", "done", "failed")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def enqueue(conn: sqlite3.Connection, photo_id: int, stage: str) -> None:
    conn.execute(
        "INSERT INTO jobs(photo_id, stage, status, updated_at) VALUES (?, ?, 'pending', ?)"
        " ON CONFLICT(photo_id, stage) DO NOTHING",
        (photo_id, stage, _now()),
    )


def claim_next(
    conn: sqlite3.Connection, stage: str, exclude: set[int] | None = None
) -> int | None:
    """Claim the lowest-id pending photo for `stage`.

    `exclude` skips photos already attempted in this pass. A failed job returns to
    'pending' while attempts remain, so without it a single failing photo would be
    re-claimed forever and starve the rest of the queue.
    """
    if exclude:
        placeholders = ", ".join("?" for _ in exclude)
        row = conn.execute(
            "SELECT photo_id FROM jobs WHERE stage = ? AND status = 'pending'"
            f" AND photo_id NOT IN ({placeholders}) ORDER BY photo_id LIMIT 1",
            (stage, *exclude),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT photo_id FROM jobs WHERE stage = ? AND status = 'pending'"
            " ORDER BY photo_id LIMIT 1",
            (stage,),
        ).fetchone()
    if row is None:
        return None
    conn.execute(
        "UPDATE jobs SET status = 'running', updated_at = ? WHERE photo_id = ? AND stage = ?",
        (_now(), row["photo_id"], stage),
    )
    return int(row["photo_id"])


def complete(conn: sqlite3.Connection, photo_id: int, stage: str) -> None:
    conn.execute(
        "UPDATE jobs SET status = 'done', error = NULL, updated_at = ?"
        " WHERE photo_id = ? AND stage = ?",
        (_now(), photo_id, stage),
    )


def fail(conn: sqlite3.Connection, photo_id: int, stage: str, error: str) -> None:
    conn.execute(
        "UPDATE jobs SET attempts = attempts + 1,"
        " status = CASE WHEN attempts + 1 >= ? THEN 'failed' ELSE 'pending' END,"
        " error = ?, updated_at = ? WHERE photo_id = ? AND stage = ?",
        (MAX_ATTEMPTS, error, _now(), photo_id, stage),
    )


def stage_counts(conn: sqlite3.Connection, stage: str) -> dict[str, int]:
    counts = dict.fromkeys(STATUSES, 0)
    for row in conn.execute(
        "SELECT status, count(*) AS n FROM jobs WHERE stage = ? GROUP BY status", (stage,)
    ):
        counts[row["status"]] = row["n"]
    return counts
