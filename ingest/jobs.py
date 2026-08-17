import sqlite3
from datetime import datetime, timezone

STAGES: tuple[str, ...] = ("thumbnail", "embed", "taxonomy", "caption")
MAX_ATTEMPTS = 3
STATUSES = ("pending", "running", "done", "failed")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def reprocess_one(conn: sqlite3.Connection, photo_id: int, stage: str) -> None:
    """Reset one photo's one stage to 'pending' so the worker re-runs just it.

    For the per-photo re-tag / re-caption on `/photo` — the model-derived stages
    change (new vocab, new caption model) while thumbnails/embeddings are static.
    Idempotent; creates the job row if the photo predates the stage.
    """
    if stage not in STAGES:
        raise ValueError(f"unknown stage: {stage}")
    enqueue(conn, photo_id, stage)
    conn.execute(
        "UPDATE jobs SET status = 'pending', attempts = 0, error = NULL, updated_at = ?"
        " WHERE photo_id = ? AND stage = ?",
        (_now(), photo_id, stage),
    )


def requeue_running(conn: sqlite3.Connection, photo_id: int, stage: str) -> None:
    """Return one in-flight job to 'pending' WITHOUT counting an attempt — for a
    caption aborted mid-flight by an interactive preempt (§8.1). The photo was
    claimed ('running'); the abort is not a failure, so it re-runs on the next pass
    instead of being stranded 'running' until a worker restart."""
    conn.execute(
        "UPDATE jobs SET status = 'pending', updated_at = ? WHERE photo_id = ? AND stage = ?",
        (_now(), photo_id, stage),
    )


def requeue_stalled(conn: sqlite3.Connection) -> int:
    """Return every 'running' job to 'pending' — call once at worker startup.

    A job is 'running' only while a worker holds it; a killed or restarted worker
    (a crash, or `watchfiles` reloading it on a code change) leaves its in-flight
    jobs stranded in 'running', and `claim_next` only picks 'pending', so they
    would never run again. At startup no job is legitimately running yet, so any
    'running' row is an orphan to reclaim — this is what makes §8's "resume exactly
    where it stopped" true across restarts. Single-owner (§3.2), so there is no
    live worker whose job this could steal. Returns how many were requeued.
    """
    cur = conn.execute(
        "UPDATE jobs SET status = 'pending', updated_at = ? WHERE status = 'running'",
        (_now(),),
    )
    return cur.rowcount


def format_speed(per_sec: float | None) -> str | None:
    """Human-readable throughput label (§13), ALWAYS per second — never per minute
    or hour. A slow stage like captioning (~30 s each) keeps extra decimals so it
    still reads a real rate (e.g. `0.033/s`) instead of rounding to `0.0/s`."""
    if not per_sec:
        return None
    if per_sec >= 1.0:
        return f"{per_sec:.1f}/s"
    if per_sec >= 0.1:
        return f"{per_sec:.2f}/s"
    return f"{per_sec:.3f}/s"


def stage_speed(conn: sqlite3.Connection, stage: str, window: int = 50) -> float | None:
    """Recent throughput for a stage — completed jobs per second (§13).

    Measured over the most recent `window` completions from `jobs.updated_at`
    (set by `complete`). Because that history lives in the database, the last
    processing speed **survives a restart**: a freshly-started process reads the
    same rows and reports the last session's rate with no in-memory state. Returns
    None until there are two timed completions to measure a span across, or when
    all of them share one instant (a fully-batched stage in under a second).
    """
    rows = conn.execute(
        "SELECT updated_at FROM jobs WHERE stage = ? AND status = 'done'"
        " AND updated_at IS NOT NULL ORDER BY updated_at DESC LIMIT ?",
        (stage, window),
    ).fetchall()
    if len(rows) < 2:
        return None
    newest = datetime.fromisoformat(rows[0]["updated_at"])
    oldest = datetime.fromisoformat(rows[-1]["updated_at"])
    span = (newest - oldest).total_seconds()
    if span <= 0:
        return None
    return (len(rows) - 1) / span  # N completions span N-1 intervals over `span`


def stage_counts(conn: sqlite3.Connection, stage: str) -> dict[str, int]:
    counts = dict.fromkeys(STATUSES, 0)
    for row in conn.execute(
        "SELECT status, count(*) AS n FROM jobs WHERE stage = ? GROUP BY status", (stage,)
    ):
        counts[row["status"]] = row["n"]
    return counts
