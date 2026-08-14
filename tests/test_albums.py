from albums.by_camera import ByCameraOrganizer
from albums.by_date import ByDateOrganizer
from albums.by_place import ByPlaceOrganizer
from albums.registry import ORGANIZERS, get_organizer
from tests.factories import add_photo


def _photo(conn, pid, **cols):
    return add_photo(conn, photo_id=pid, content_hash=f"h{pid}", thumb_key=f"{pid}.jpg", **cols)


def test_date_defaults_to_the_month_grain(conn):
    _photo(conn, 1, shot_at="2025-07-12T09:00:00", camera="X-T5")
    _photo(conn, 2, shot_at="2025-07-30T22:00:00", camera="X-T5")  # same month

    albums = ByDateOrganizer().organize(conn, owner_id=1)  # no grain -> month
    assert [(a.title, a.size) for a in albums] == [("July 2025", 2)]
    assert "2 photos" in albums[0].description
    assert "X-T5" in albums[0].description


def test_date_organizer_ignores_photos_without_a_date(conn):
    _photo(conn, 1, shot_at=None)
    assert ByDateOrganizer().organize(conn, owner_id=1) == []


def test_date_day_grain_buckets_by_calendar_day(conn):
    _photo(conn, 1, shot_at="2025-07-12T09:00:00", camera="X-T5")
    _photo(conn, 2, shot_at="2025-07-12T22:00:00", camera="X-T5")  # same day
    _photo(conn, 3, shot_at="2025-07-13T10:00:00", camera="X-T5")

    albums = ByDateOrganizer().organize(conn, owner_id=1, grain="day")
    assert [a.title for a in albums] == ["13 Jul 2025", "12 Jul 2025"]  # newest first
    assert [a.size for a in albums] == [1, 2]
    assert "2 photos" in albums[1].description
    assert "X-T5" in albums[1].description


def test_date_month_grain_buckets_by_calendar_month(conn):
    _photo(conn, 1, shot_at="2025-06-30T09:00:00")
    _photo(conn, 2, shot_at="2025-07-01T09:00:00")
    _photo(conn, 3, shot_at="2025-07-20T09:00:00")

    albums = ByDateOrganizer().organize(conn, owner_id=1, grain="month")
    assert [(a.title, a.size) for a in albums] == [("July 2025", 2), ("June 2025", 1)]


def test_date_year_grain_buckets_by_calendar_year(conn):
    _photo(conn, 1, shot_at="2024-12-31T23:00:00")
    _photo(conn, 2, shot_at="2025-01-01T01:00:00")
    _photo(conn, 3, shot_at="2025-08-14T12:00:00")

    albums = ByDateOrganizer().organize(conn, owner_id=1, grain="year")
    assert [(a.title, a.size) for a in albums] == [("2025", 2), ("2024", 1)]


def test_date_unknown_grain_falls_back_to_month(conn):
    _photo(conn, 1, shot_at="2025-07-12T09:00:00")
    _photo(conn, 2, shot_at="2025-07-13T10:00:00")  # same month

    albums = ByDateOrganizer().organize(conn, owner_id=1, grain="bogus")
    assert [a.title for a in albums] == ["July 2025"]  # month, the default


def test_camera_organizer_groups_by_device(conn):
    _photo(conn, 1, camera="X-T5")
    _photo(conn, 2, camera="X-T5")
    _photo(conn, 3, camera="iPhone")
    _photo(conn, 4, camera=None)

    albums = ByCameraOrganizer().organize(conn, owner_id=1)
    titles = {a.title: a.size for a in albums}
    assert titles["X-T5"] == 2
    assert titles["iPhone"] == 1
    assert titles["Unknown camera"] == 1


def test_place_organizer_groups_by_real_place_name(conn):
    _photo(conn, 1, gps_lat=51.5001, gps_lon=-0.1201)  # London
    _photo(conn, 2, gps_lat=51.5002, gps_lon=-0.1202)  # London, ~same spot
    _photo(conn, 3, gps_lat=48.8566, gps_lon=2.3522)   # Paris
    _photo(conn, 4)  # no GPS -> excluded

    albums = ByPlaceOrganizer().organize(conn, owner_id=1)
    assert sorted(a.size for a in albums) == [1, 2]           # London×2, Paris×1
    assert all("°" not in a.title for a in albums)            # never coordinates
    titles = " ".join(a.title for a in albums)
    assert "Paris" in titles or "London" in titles           # real names


def test_registry_exposes_the_live_organizers_and_defaults_to_date(conn):
    assert set(ORGANIZERS) == {"date", "memories", "camera", "place"}
    assert get_organizer(None).name == "date"
    assert get_organizer("place").name == "place"
    assert get_organizer("nonsense").name == "date"
