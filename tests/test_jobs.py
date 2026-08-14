from ingest.jobs import (
    MAX_ATTEMPTS,
    claim_next,
    complete,
    enqueue,
    fail,
    stage_counts,
)
from tests.factories import add_photo


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
