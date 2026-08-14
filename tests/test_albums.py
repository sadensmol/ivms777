from albums.by_camera import ByCameraOrganizer
from albums.by_date import ByDateOrganizer
from albums.by_place import ByPlaceOrganizer
from albums.by_similarity import BySimilarityOrganizer
from albums.registry import ORGANIZERS, get_organizer
from embedding.fakes import FakeEmbedder
from embedding.store import write_vector
from tests.factories import add_photo


def _photo(conn, pid, **cols):
    return add_photo(conn, photo_id=pid, content_hash=f"h{pid}", thumb_key=f"{pid}.jpg", **cols)


def test_date_organizer_splits_on_a_six_hour_gap(conn):
    # two shots minutes apart, then one the next morning -> two events
    _photo(conn, 1, shot_at="2025-07-12T09:00:00", camera="X-T5")
    _photo(conn, 2, shot_at="2025-07-12T09:20:00", camera="X-T5")
    _photo(conn, 3, shot_at="2025-07-13T10:00:00", camera="X-T5")

    albums = ByDateOrganizer().organize(conn, owner_id=1)
    assert len(albums) == 2
    sizes = sorted(a.size for a in albums)
    assert sizes == [1, 2]
    assert all(a.description for a in albums)  # every album is described
    assert any("X-T5" in a.description for a in albums)


def test_date_organizer_ignores_photos_without_a_date(conn):
    _photo(conn, 1, shot_at=None)
    assert ByDateOrganizer().organize(conn, owner_id=1) == []


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


def test_place_organizer_buckets_nearby_coordinates(conn):
    _photo(conn, 1, gps_lat=51.5001, gps_lon=-0.1201)
    _photo(conn, 2, gps_lat=51.5002, gps_lon=-0.1202)  # ~same spot
    _photo(conn, 3, gps_lat=48.8566, gps_lon=2.3522)   # Paris
    _photo(conn, 4)  # no GPS -> excluded

    albums = ByPlaceOrganizer().organize(conn, owner_id=1)
    assert sorted(a.size for a in albums) == [1, 2]


def test_similarity_organizer_groups_identical_vectors(conn):
    fake = FakeEmbedder()
    # 1 and 2 share a vector (identical -> cosine 1.0); 3 is on its own
    for pid, word in ((1, "beach"), (2, "beach"), (3, "keyboard")):
        _photo(conn, pid)
        write_vector(conn, pid, fake.embed_texts([word])[0])

    albums = BySimilarityOrganizer().organize(conn, owner_id=1)
    assert len(albums) == 1
    assert sorted(albums[0].photo_ids) == [1, 2]
    assert "2 visually similar" in albums[0].description


def test_registry_exposes_all_four_and_defaults_to_date(conn):
    assert set(ORGANIZERS) == {"date", "similarity", "camera", "place"}
    assert get_organizer(None).name == "date"
    assert get_organizer("place").name == "place"
    assert get_organizer("nonsense").name == "date"
