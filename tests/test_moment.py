"""The `moment` signal (§9) — same stretch of time, same place.

The point of this signal is the case no visual model can reach: two photos of
completely different things that belong to one experience.
"""

import pytest

from embedding.fakes import FakeEmbedder
from embedding.store import write_vector
from search import signals
from search.moment import gap, haversine_m, moment_similarity, parse_shot_at, read_moment
from search.retriever import Query, candidates
from search.semantic import similar_photos, similarity_breakdown
from tests.factories import add_photo


def _photo(conn, pid, *, shot_at=None, lat=None, lon=None, caption="a photo", vector=True):
    add_photo(conn, photo_id=pid, content_hash=f"h{pid}", thumb_key=f"{pid}.jpg",
              caption=caption, shot_at=shot_at, gps_lat=lat, gps_lon=lon)
    if vector:
        write_vector(conn, pid, FakeEmbedder().embed_texts([caption])[0])
    return pid


# --- reading the data ----------------------------------------------------------

def test_a_photo_without_exif_time_has_no_moment(conn):
    _photo(conn, 1)
    assert read_moment(conn, 1) is None


def test_unparseable_timestamps_are_absent_not_a_crash(conn):
    assert parse_shot_at(None) is None
    assert parse_shot_at("") is None
    assert parse_shot_at("not a date") is None
    assert parse_shot_at("2023-01-01T12:00:00") is not None


def test_haversine_measures_real_ground_distance():
    # ~111 m per 0.001 degree of latitude.
    assert haversine_m(42.0, 42.0, 42.001, 42.0) == pytest.approx(111.0, abs=2.0)


def test_gap_is_unknown_not_zero_when_gps_is_missing():
    seed = (0.0, 42.0, 42.0)
    _, metres = gap(seed, (3600.0, None, None))
    assert metres is None


# --- the signal ----------------------------------------------------------------

def test_photos_minutes_apart_in_one_spot_share_a_moment(conn):
    _photo(conn, 1, shot_at="2023-12-01T14:00:00", lat=42.0, lon=42.0)
    _photo(conn, 2, shot_at="2023-12-01T14:10:00", lat=42.0, lon=42.0)
    hits = moment_similarity(conn, owner_id=1, seed_id=1)
    assert hits[2] > 0.9


def test_a_photo_from_another_day_shares_no_moment(conn):
    _photo(conn, 1, shot_at="2023-12-01T14:00:00", lat=42.0, lon=42.0)
    _photo(conn, 2, shot_at="2023-12-05T14:00:00", lat=42.0, lon=42.0)
    assert moment_similarity(conn, owner_id=1, seed_id=1) == {}


def test_the_same_minute_across_town_shares_no_moment(conn):
    # Time alone is not a moment — a stranger's photo taken at the same instant 40 km
    # away is not part of the same experience.
    _photo(conn, 1, shot_at="2023-12-01T14:00:00", lat=42.0, lon=42.0)
    _photo(conn, 2, shot_at="2023-12-01T14:01:00", lat=42.4, lon=42.0)
    assert moment_similarity(conn, owner_id=1, seed_id=1) == {}


def test_without_gps_the_moment_falls_back_to_time_alone(conn):
    _photo(conn, 1, shot_at="2023-12-01T14:00:00")
    _photo(conn, 2, shot_at="2023-12-01T14:10:00")
    hits = moment_similarity(conn, owner_id=1, seed_id=1)
    assert hits[2] == pytest.approx(signals.MOMENT_NO_GPS, abs=0.02)


def test_the_signal_can_be_switched_off(conn):
    _photo(conn, 1, shot_at="2023-12-01T14:00:00", lat=42.0, lon=42.0)
    _photo(conn, 2, shot_at="2023-12-01T14:05:00", lat=42.0, lon=42.0)
    assert moment_similarity(conn, owner_id=1, seed_id=1, gate=signals.SIGNAL_OFF) == {}


# --- end to end through the core ------------------------------------------------

def test_visually_different_photos_from_one_moment_are_similar(conn):
    """The whole reason this signal exists: the cake and the faces around it share no
    subject, no palette and no visual resemblance, but they are one memory."""
    _photo(conn, 1, shot_at="2023-12-01T14:00:00", lat=42.0, lon=42.0,
           caption="a birthday cake with candles")
    _photo(conn, 2, shot_at="2023-12-01T14:03:00", lat=42.0, lon=42.0,
           caption="a crowd of people singing")
    # Moment-only, so it does not make the STRICT strip — it is what "Show more"
    # reveals (§9): same outing, but the photos share nothing else.
    assert similar_photos(conn, owner_id=1, photo_id=1, k=5) == []
    results = similar_photos(conn, owner_id=1, photo_id=1, k=5, loose=True)
    assert [r["id"] for r in results] == [2]
    assert results[0]["reasons"][0]["text"] == "same time & place"


def test_a_moment_needs_no_embedding_at_all(conn):
    # EXIF is written at ingest, long before the photo is embedded — so a just-uploaded
    # photo already has similar photos (§9.2 graceful degradation, taken to its limit).
    _photo(conn, 1, shot_at="2023-12-01T14:00:00", lat=42.0, lon=42.0, vector=False)
    _photo(conn, 2, shot_at="2023-12-01T14:05:00", lat=42.0, lon=42.0)
    assert candidates(conn, None, 1, Query(seed_photo_id=1, k=5)) == [2]


def test_a_distant_photo_is_not_pulled_in_by_the_moment(conn):
    _photo(conn, 1, shot_at="2023-12-01T14:00:00", lat=42.0, lon=42.0,
           caption="a birthday cake with candles")
    _photo(conn, 2, shot_at="2024-06-01T09:00:00", lat=10.0, lon=10.0,
           caption="a mountain ridge at dawn")
    assert similar_photos(conn, owner_id=1, photo_id=1, k=5) == []
    # Not even the loose pass reaches it — a different day and 5 km away is not a
    # weaker match, it is no match.
    assert similar_photos(conn, owner_id=1, photo_id=1, k=5, loose=True) == []


def test_the_panel_reports_the_actual_gap(conn):
    _photo(conn, 1, shot_at="2023-12-01T14:00:00", lat=42.0, lon=42.0)
    _photo(conn, 2, shot_at="2023-12-01T14:30:00", lat=42.0, lon=42.0)
    rows = similarity_breakdown(conn, owner_id=1, origin_id=1, current_id=2)
    [row] = [r for r in rows if r["param"] == "same time & place"]
    assert row["origin"] == "30 min apart"
    assert row["current"] == "0 m apart"


def test_the_panel_says_no_gps_when_coordinates_are_missing(conn):
    _photo(conn, 1, shot_at="2023-12-01T14:00:00")
    _photo(conn, 2, shot_at="2023-12-01T14:30:00")
    rows = similarity_breakdown(conn, owner_id=1, origin_id=1, current_id=2)
    [row] = [r for r in rows if r["param"] == "same time & place"]
    assert row["current"] == "no GPS"
