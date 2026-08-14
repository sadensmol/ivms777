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


def reprocess(
    conn: sqlite3.Connection, owner_id: int, from_stage: str, to_stage: str | None = None
) -> int:
    """Reset a range of stages to 'pending' for the owner's photos.

    Resets `from_stage` through `to_stage` inclusive (default: through the last
    stage). Downstream stages in the range are reset too because their output
    depends on the one being rerun (re-embedding changes the vectors the taxonomy
    reads). Bounding with `to_stage` lets the UI re-run the cheap stages
    (thumbnails → tags) without re-running the slow, unnecessary captioning of
    already-captioned photos (images are static). Handlers are idempotent, so the
    worker re-runs them on its next drain. Returns the number of photos queued.
    """
    if from_stage not in STAGES:
        raise ValueError(f"unknown stage: {from_stage}")
    end = STAGES.index(to_stage) + 1 if to_stage in STAGES else len(STAGES)
    stages = STAGES[STAGES.index(from_stage):end]
    photo_ids = [
        row["id"] for row in conn.execute(
            "SELECT id FROM photos WHERE owner_id = ?", (owner_id,)
        )
    ]
    for photo_id in photo_ids:
        for stage in stages:
            enqueue(conn, photo_id, stage)  # create a job row where one is missing
    if photo_ids:
        placeholders = ", ".join("?" for _ in photo_ids)
        for stage in stages:
            conn.execute(
                "UPDATE jobs SET status = 'pending', attempts = 0, error = NULL, updated_at = ?"
                f" WHERE stage = ? AND photo_id IN ({placeholders})",
                (_now(), stage, *photo_ids),
            )
    return len(photo_ids)


def stage_counts(conn: sqlite3.Connection, stage: str) -> dict[str, int]:
    counts = dict.fromkeys(STATUSES, 0)
    for row in conn.execute(
        "SELECT status, count(*) AS n FROM jobs WHERE stage = ? GROUP BY status", (stage,)
    ):
        counts[row["status"]] = row["n"]
    return counts
