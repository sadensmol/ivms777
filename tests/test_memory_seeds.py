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


def test_same_afternoon_across_different_regions_splits(conn):
    # London then Paris the same afternoon -> genuinely different regions -> two.
    for pid in (1, 2, 3):
        _p(conn, pid, f"2025-07-12T12:0{pid}:00", gps_lat=51.50, gps_lon=-0.12)
    for pid in (4, 5, 6):
        _p(conn, pid, f"2025-07-12T12:1{pid}:00", gps_lat=48.85, gps_lon=2.35)
    assert len(seed_candidates(conn, owner_id=1, min_size=3)) == 2


def test_nearby_spots_in_one_town_stay_one_memory(conn):
    # A day at Borjomi: the spring, then the park/waterfall ~1-2 km away, same
    # afternoon. Sub-city movement must NOT split — it is ONE memory (§11).
    for pid in (1, 2, 3):
        _p(conn, pid, f"2025-07-12T14:0{pid}:00", gps_lat=41.836, gps_lon=43.392)
    for pid in (4, 5, 6):
        _p(conn, pid, f"2025-07-12T14:1{pid}:00", gps_lat=41.845, gps_lon=43.389)
    cands = seed_candidates(conn, owner_id=1, min_size=3)
    assert len(cands) == 1 and len(cands[0].photo_ids) == 6


def test_a_multi_day_stay_away_from_home_is_ONE_memory(conn):
    # A week in Batumi is "a week in Batumi", not seven day-fragments: people sleep,
    # and a flat 6 h gap cut the trip at every night.
    for pid in range(1, 10):  # home — the most photographed region
        _p(conn, pid, f"2025-06-0{pid}T12:00:00", gps_lat=41.70, gps_lon=44.80)
    for day, pid in enumerate((20, 21, 22, 23), start=12):
        _p(conn, pid, f"2025-07-{day}T13:00:00", gps_lat=41.65, gps_lon=41.64)
    trips = [c for c in seed_candidates(conn, owner_id=1, min_size=3)
             if set(c.photo_ids) <= {20, 21, 22, 23}]
    assert len(trips) == 1
    assert sorted(trips[0].photo_ids) == [20, 21, 22, 23]


def test_days_at_home_stay_separate_memories(conn):
    # The wider trip gap must NOT weld ordinary days at home into one endless
    # memory — home is where most photos are, and there the 6 h gap still rules.
    for day in (12, 13, 14):
        for n in (1, 2, 3):
            _p(conn, day * 10 + n, f"2025-07-{day}T1{n}:00:00", gps_lat=41.70, gps_lon=44.80)
    assert len(seed_candidates(conn, owner_id=1, min_size=3)) == 3


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
