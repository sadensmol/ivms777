from albums.seeds import seed_candidates
from tests.factories import add_photo


def _p(conn, pid, shot_at, **cols):
    return add_photo(conn, photo_id=pid, content_hash=f"h{pid}", thumb_key=f"{pid}.jpg",
                     caption="x", shot_at=shot_at, **cols)


def test_a_contiguous_run_becomes_one_candidate(conn):
    for pid in range(1, 5):
        _p(conn, pid, f"2025-07-12T1{pid}:00:00")
    cands = seed_candidates(conn, owner_id=1, min_size=3)
    assert len(cands) == 1
    assert sorted(cands[0].photo_ids) == [1, 2, 3, 4]


def test_a_six_hour_gap_splits_candidates(conn):
    for pid in (1, 2, 3):
        _p(conn, pid, f"2025-07-12T09:0{pid}:00")
    for pid in (4, 5, 6):
        _p(conn, pid, f"2025-07-13T20:0{pid}:00")
    assert len(seed_candidates(conn, owner_id=1, min_size=3)) == 2


def test_one_event_across_two_places_splits_by_gps(conn):
    # same afternoon, two ~1 km-apart locations -> two candidates
    for pid in (1, 2, 3):
        _p(conn, pid, f"2025-07-12T12:0{pid}:00", gps_lat=51.50, gps_lon=-0.12)
    for pid in (4, 5, 6):
        _p(conn, pid, f"2025-07-12T12:1{pid}:00", gps_lat=48.85, gps_lon=2.35)
    assert len(seed_candidates(conn, owner_id=1, min_size=3)) == 2


def test_runs_below_min_size_are_dropped(conn):
    _p(conn, 1, "2025-07-12T09:00:00")
    _p(conn, 2, "2025-07-12T09:05:00")  # only two -> not a memory
    assert seed_candidates(conn, owner_id=1, min_size=3) == []


def test_uncaptioned_photos_are_ignored(conn):
    for pid in (1, 2, 3):
        _p(conn, pid, f"2025-07-12T09:0{pid}:00")
    add_photo(conn, photo_id=4, content_hash="h4", thumb_key="4.jpg",
              shot_at="2025-07-12T09:04:00")  # no caption
    cands = seed_candidates(conn, owner_id=1, min_size=3)
    assert cands[0].photo_ids == [1, 2, 3]
