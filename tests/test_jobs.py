from datetime import datetime, timedelta, timezone

from ingest.jobs import (
    MAX_ATTEMPTS,
    claim_next,
    complete,
    enqueue,
    fail,
    format_speed,
    reprocess,
    requeue_stalled,
    stage_counts,
    stage_speed,
)
from tests.factories import add_photo


def test_requeue_stalled_recovers_orphaned_running_jobs(conn):
    # A job left 'running' by a killed/restarted worker (claim_next set it running,
    # complete never came) must return to 'pending' so a fresh worker re-runs it —
    # otherwise claim_next (which only picks 'pending') strands it forever (§8).
    add_photo(conn, photo_id=1, content_hash="a", thumb_key="1.jpg")
    enqueue(conn, 1, "caption")
    claim_next(conn, "caption")
    assert stage_counts(conn, "caption")["running"] == 1
    assert requeue_stalled(conn) == 1
    assert stage_counts(conn, "caption")["running"] == 0
    assert stage_counts(conn, "caption")["pending"] == 1


def test_format_speed_stays_readable_for_slow_stages():
    assert format_speed(None) is None
    assert format_speed(2.0) == "2.0/s"          # fast stages: per second
    assert format_speed(0.0333333) == "2.0/min"  # ~30 s/caption reads as 2.0/min, not "0.0/s"


def _done_at(conn, photo_id, stage, when):
    add_photo(conn, photo_id=photo_id, content_hash=f"h{photo_id}", thumb_key=f"{photo_id}.jpg")
    conn.execute(
        "INSERT INTO jobs(photo_id, stage, status, updated_at) VALUES (?, ?, 'done', ?)",
        (photo_id, stage, when.isoformat()),
    )


def test_stage_speed_measures_recent_throughput(conn):
    # 5 captions finished 2s apart -> 4 intervals over 8s -> 0.5/s.
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i in range(5):
        _done_at(conn, i + 1, "caption", base + timedelta(seconds=2 * i))
    assert abs(stage_speed(conn, "caption") - 0.5) < 1e-9


def test_stage_speed_none_without_two_completions(conn):
    assert stage_speed(conn, "caption") is None
    _done_at(conn, 1, "caption", datetime(2026, 1, 1, tzinfo=timezone.utc))
    assert stage_speed(conn, "caption") is None  # one completion can't span a rate


def test_stage_speed_survives_restart_via_persisted_job_history(conn):
    # The rate is read from jobs.updated_at, which persists — no in-memory state,
    # so a fresh process (this fresh conn) still reports the last speed.
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i in range(3):
        _done_at(conn, i + 1, "embed", base + timedelta(seconds=i))  # 2 intervals / 2s = 1/s
    assert abs(stage_speed(conn, "embed") - 1.0) < 1e-9


def test_reprocess_resets_the_stage_and_everything_downstream(conn):
    add_photo(conn, photo_id=1, content_hash="a", thumb_key="1.jpg")
    for stage in ("embed", "taxonomy"):
        enqueue(conn, 1, stage)
        complete(conn, 1, stage)
    assert reprocess(conn, owner_id=1, from_stage="embed") == 1
    assert stage_counts(conn, "embed")["pending"] == 1
    assert stage_counts(conn, "taxonomy")["pending"] == 1  # downstream reset too


def test_reprocess_range_stops_at_to_stage(conn):
    add_photo(conn, photo_id=1, content_hash="a", thumb_key="1.jpg")
    for stage in ("thumbnail", "embed", "taxonomy", "caption"):
        enqueue(conn, 1, stage)
        complete(conn, 1, stage)
    reprocess(conn, owner_id=1, from_stage="thumbnail", to_stage="taxonomy")
    for stage in ("thumbnail", "embed", "taxonomy"):
        assert stage_counts(conn, stage)["pending"] == 1
    assert stage_counts(conn, "caption")["done"] == 1   # caption left alone


def test_reprocess_taxonomy_leaves_embed_alone(conn):
    add_photo(conn, photo_id=1, content_hash="a", thumb_key="1.jpg")
    enqueue(conn, 1, "embed")
    complete(conn, 1, "embed")
    reprocess(conn, owner_id=1, from_stage="taxonomy")
    assert stage_counts(conn, "embed")["done"] == 1        # upstream untouched
    assert stage_counts(conn, "taxonomy")["pending"] == 1  # created and pending


def test_reprocess_is_owner_scoped(conn):
    add_photo(conn, photo_id=1, owner_id=1, content_hash="a", thumb_key="1.jpg")
    add_photo(conn, photo_id=2, owner_id=2, content_hash="b", thumb_key="2.jpg")
    enqueue(conn, 2, "taxonomy")
    complete(conn, 2, "taxonomy")
    reprocess(conn, owner_id=1, from_stage="taxonomy")
    counts = stage_counts(conn, "taxonomy")
    assert counts["done"] == 1     # owner 2 untouched
    assert counts["pending"] == 1  # owner 1's reset job


def insert_photo(conn, digest="h1"):
    return add_photo(conn, content_hash=digest)


def test_claim_returns_pending_photo_and_marks_it_running(conn):
    photo_id = insert_photo(conn)
    enqueue(conn, photo_id, "thumbnail")

    assert claim_next(conn, "thumbnail") == photo_id
    status = conn.execute(
        "SELECT status FROM jobs WHERE photo_id=? AND stage='thumbnail'", (photo_id,)
    ).fetchone()["status"]
    assert status == "running"


def test_claim_returns_none_when_nothing_pending(conn):
    assert claim_next(conn, "thumbnail") is None


def test_claimed_job_is_not_returned_twice(conn):
    photo_id = insert_photo(conn)
    enqueue(conn, photo_id, "thumbnail")
    claim_next(conn, "thumbnail")
    assert claim_next(conn, "thumbnail") is None


def test_complete_marks_done(conn):
    photo_id = insert_photo(conn)
    enqueue(conn, photo_id, "thumbnail")
    claim_next(conn, "thumbnail")
    complete(conn, photo_id, "thumbnail")
    assert stage_counts(conn, "thumbnail")["done"] == 1


def test_failure_retries_then_sticks_at_failed(conn):
    photo_id = insert_photo(conn)
    enqueue(conn, photo_id, "thumbnail")

    for _ in range(MAX_ATTEMPTS):
        assert claim_next(conn, "thumbnail") == photo_id
        fail(conn, photo_id, "thumbnail", "boom")

    assert claim_next(conn, "thumbnail") is None
    counts = stage_counts(conn, "thumbnail")
    assert counts["failed"] == 1
    row = conn.execute(
        "SELECT attempts, error FROM jobs WHERE photo_id=? AND stage='thumbnail'", (photo_id,)
    ).fetchone()
    assert row["attempts"] == MAX_ATTEMPTS
    assert row["error"] == "boom"


def test_enqueue_is_idempotent(conn):
    photo_id = insert_photo(conn)
    enqueue(conn, photo_id, "thumbnail")
    enqueue(conn, photo_id, "thumbnail")
    assert stage_counts(conn, "thumbnail")["pending"] == 1


def test_stage_counts_reports_every_status_key(conn):
    counts = stage_counts(conn, "caption")
    assert counts == {"pending": 0, "running": 0, "done": 0, "failed": 0}
