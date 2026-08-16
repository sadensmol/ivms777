# tests/test_lease_store.py  — uses the `conn` fixture from tests/conftest.py
from models import lease_store as ls


def test_single_holder_excludes_others(conn):
    assert ls.try_acquire(conn, holder="app", workload="CHAT", priority=10) is True
    # a second holder cannot acquire while one is held
    assert ls.try_acquire(conn, holder="worker", workload="INGEST_EMBED", priority=1) is False
    lease = ls.read_lease(conn)
    assert lease["holder"] == "app" and lease["workload"] == "CHAT"


def test_release_frees_the_lease(conn):
    ls.try_acquire(conn, holder="worker", workload="INGEST_CAPTION", priority=1)
    ls.release(conn, holder="worker")
    assert ls.read_lease(conn) is None
    assert ls.try_acquire(conn, holder="app", workload="CHAT", priority=10) is True


def test_preempt_flag_roundtrip(conn):
    ls.try_acquire(conn, holder="worker", workload="INGEST_EMBED", priority=1)
    assert ls.preempt_requested(conn) is False
    ls.request_preempt(conn)
    assert ls.preempt_requested(conn) is True
    # releasing clears the row (and its flag) so the next holder starts clean
    ls.release(conn, holder="worker")
    assert ls.preempt_requested(conn) is False


def test_reclaim_stale_only_removes_a_silent_holder(conn):
    # A fresh lease (heartbeat just now) is NOT reclaimed — its holder is alive.
    ls.try_acquire(conn, holder="worker", workload="INGEST_CAPTION", priority=1)
    assert ls.reclaim_stale(conn, max_age_s=20) is False
    assert ls.read_lease(conn) is not None
    # Age its heartbeat past the threshold (a dead/wedged holder stops bumping it).
    conn.execute("UPDATE model_lease SET heartbeat = datetime('now', '-60 seconds') WHERE id = 1")
    assert ls.reclaim_stale(conn, max_age_s=20) is True
    assert ls.read_lease(conn) is None
    # Nothing held → nothing to reclaim.
    assert ls.reclaim_stale(conn, max_age_s=20) is False
