import pytest

from albums.by_date import ByDateOrganizer
from albums.memories import MemoriesOrganizer
from albums.memory_store import Memory, replace_memories
from organize.plan import PLAN_VERSION, build_plan
from tests.factories import add_photo

WHEN = "2026-08-19T09:12:44Z"


def plan_for(conn, organizer, grain=None):
    return build_plan(conn, 1, organizer, grain, generated_at=WHEN)


def targets(plan):
    return {f["hash"]: f["target"] for f in plan["files"]}


@pytest.fixture
def dated(conn):
    add_photo(conn, photo_id=1, content_hash="aaaa1111", thumb_key="1.jpg",
              shot_at="2024-06-14T18:30:12", bytes=100,
              sources=["iphone dump/IMG_4471.jpg"])
    add_photo(conn, photo_id=2, content_hash="bbbb2222", thumb_key="2.jpg",
              shot_at="2024-07-02T10:00:00", bytes=200,
              sources=["IMG_9000.jpg"])
    return conn


def test_a_date_month_plan_puts_each_photo_in_its_month_folder(dated):
    plan = plan_for(dated, ByDateOrganizer(), "month")
    assert plan["plan_version"] == PLAN_VERSION
    assert plan["organizer"] == "date"
    assert plan["grain"] == "month"
    assert plan["generated_at"] == WHEN
    assert targets(plan) == {
        "aaaa1111": "June 2024/IMG_4471.jpg",
        "bbbb2222": "July 2024/IMG_9000.jpg",
    }


def test_filenames_are_never_changed(dated):
    plan = plan_for(dated, ByDateOrganizer(), "month")
    assert all(f["target"].endswith(f["sources"][0].rsplit("/", 1)[-1])
               for f in plan["files"])


def test_every_album_carries_its_description_for_description_txt(dated):
    plan = plan_for(dated, ByDateOrganizer(), "month")
    june = next(a for a in plan["albums"] if a["folder"] == "June 2024")
    assert june["title"] == "June 2024"
    assert june["description"].startswith("1 photo")
    assert june["photo_count"] == 1
    assert june["kind"] == "album"


def test_sources_are_shallowest_then_alphabetical_and_the_first_one_is_kept(conn):
    add_photo(conn, photo_id=1, content_hash="aaaa1111", thumb_key="1.jpg",
              shot_at="2024-06-14T18:30:12", bytes=100,
              sources=["z/deep/IMG.jpg", "b.jpg", "a/IMG.jpg"])
    plan = plan_for(conn, ByDateOrganizer(), "month")
    entry = plan["files"][0]
    assert entry["sources"] == ["b.jpg", "a/IMG.jpg", "z/deep/IMG.jpg"]
    assert entry["target"] == "June 2024/b.jpg"


def test_two_photos_with_the_same_filename_get_a_hash_suffix(conn):
    add_photo(conn, photo_id=1, content_hash="aaaa1111", thumb_key="1.jpg",
              shot_at="2024-06-01T09:00:00", bytes=1, sources=["a/IMG.jpg"])
    add_photo(conn, photo_id=2, content_hash="bbbb2222", thumb_key="2.jpg",
              shot_at="2024-06-02T09:00:00", bytes=1, sources=["b/IMG.jpg"])
    assert set(targets(plan_for(conn, ByDateOrganizer(), "month")).values()) == {
        "June 2024/IMG.jpg", "June 2024/IMG_bbbb2222.jpg",
    }


def test_a_photo_no_album_claims_goes_to_unsorted(conn):
    add_photo(conn, photo_id=1, content_hash="aaaa1111", thumb_key="1.jpg",
              bytes=1, sources=["scan012.jpg"])          # no shot_at
    plan = plan_for(conn, ByDateOrganizer(), "month")
    assert targets(plan) == {"aaaa1111": "_unsorted/scan012.jpg"}
    assert [a["folder"] for a in plan["albums"]] == ["_unsorted"]


def test_a_photo_with_no_source_path_is_left_out_entirely(conn):
    add_photo(conn, photo_id=1, content_hash="aaaa1111", thumb_key="1.jpg",
              shot_at="2024-06-14T18:30:12", bytes=1)     # no sources
    assert plan_for(conn, ByDateOrganizer(), "month")["files"] == []


def test_the_event_owns_the_file_and_the_theme_only_references_it(conn):
    add_photo(conn, photo_id=1, content_hash="aaaa1111", thumb_key="1.jpg",
              bytes=1, sources=["IMG_4471.jpg"])
    replace_memories(conn, 1, [
        Memory("Waterfalls", "A thread across time.", [1], "sig", kind="theme"),
        Memory("A day at Borjomi", "One long day.", [1], "sig"),
    ])
    plan = plan_for(conn, MemoriesOrganizer())
    assert targets(plan) == {"aaaa1111": "A day at Borjomi/IMG_4471.jpg"}
    theme = next(a for a in plan["albums"] if a["folder"] == "Waterfalls")
    assert theme["kind"] == "theme"
    assert theme["references"] == ["A day at Borjomi/IMG_4471.jpg"]


def test_the_plan_is_incomplete_while_a_job_is_pending(conn):
    add_photo(conn, photo_id=1, content_hash="aaaa1111", thumb_key="1.jpg",
              shot_at="2024-06-14T18:30:12", bytes=1, sources=["IMG.jpg"])
    assert plan_for(conn, ByDateOrganizer(), "month")["complete"] is True
    conn.execute(
        "INSERT INTO jobs(photo_id, stage, status, updated_at)"
        " VALUES (1, 'caption', 'pending', '2026-01-01T00:00:00')"
    )
    assert plan_for(conn, ByDateOrganizer(), "month")["complete"] is False


def test_photo_count_matches_the_files_listed(dated):
    plan = plan_for(dated, ByDateOrganizer(), "month")
    assert plan["photo_count"] == len(plan["files"]) == 2
