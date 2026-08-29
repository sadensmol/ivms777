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


def test_memory_albums_carry_their_kind(conn):
    from albums.memories import MemoriesOrganizer
    from albums.memory_store import Memory, replace_memories
    _photo(conn, 1)
    replace_memories(conn, 1, [Memory("Waterfalls", "…", [1], "sig", kind="theme")])
    album = MemoriesOrganizer().organize(conn, 1)[0]
    assert album.meta["memory_kind"] == "theme"


def test_the_memory_agent_decides_the_order_photos_are_told_in():
    # Decided ONCE, when the memory is built: the album shows this order and every film
    # of the memory follows it, until the memory is rebuilt (§11).
    from albums.compose import _ordered

    assert _ordered([3, 1, 2], [1, 2, 3]) == [3, 1, 2]


def test_the_order_can_never_lose_or_invent_a_photograph():
    # Repaired rather than trusted: a memory that silently drops one of its own photos
    # is a bug the user can see and cannot explain.
    from albums.compose import _ordered

    assert _ordered([9, 3], [1, 2, 3]) == [3, 1, 2]      # 9 invented, 1 and 2 forgotten
    assert _ordered(None, [1, 2, 3]) == [1, 2, 3]
    assert _ordered([2, 2, 2], [1, 2, 3]) == [2, 1, 3]   # duplicates collapse
    assert sorted(_ordered([3], [1, 2, 3])) == [1, 2, 3]


def test_the_prompt_asks_for_the_order_and_says_who_owns_it():
    from albums.compose import _SYSTEM_PROMPT

    assert '"order"' in _SYSTEM_PROMPT
    assert "decided here and nowhere" in _SYSTEM_PROMPT
    assert "NEXT TO EACH OTHER" in _SYSTEM_PROMPT     # group related pictures
    assert "FOR that order" in _SYSTEM_PROMPT         # the description follows it


def test_rewriting_a_memory_keeps_its_identity_and_reorders_its_photos(conn):
    # In place, so `memory_notes` — which cascade on the group id — survive a retelling
    # (§11.3). `replace_memories` would take them with it.
    from albums.memory_store import (
        Memory,
        photo_ids_for_group,
        replace_memories,
        rewrite_memory,
    )
    from tests.factories import add_photo

    for pid in (1, 2, 3):
        add_photo(conn, photo_id=pid, content_hash=f"h{pid}", thumb_key=f"{pid}.jpg")
    replace_memories(conn, 1, [Memory("Old", "Old words.", [1, 2, 3], "sig")])
    gid = conn.execute("SELECT id FROM groups WHERE kind='memory'").fetchone()["id"]
    conn.execute("INSERT INTO memory_notes(group_id, at_seconds, text, created_at)"
                 " VALUES (?, 1.0, 'mine', 'now')", (gid,))

    rewrite_memory(conn, gid, Memory("New", "New words.", [3, 1, 2], "sig"))

    assert conn.execute("SELECT id FROM groups WHERE kind='memory'").fetchone()["id"] == gid
    row = conn.execute("SELECT name, description FROM groups WHERE id = ?", (gid,)).fetchone()
    assert row["name"] == "New" and row["description"] == "New words."
    assert photo_ids_for_group(conn, gid) == [3, 1, 2]
    assert conn.execute("SELECT COUNT(*) c FROM memory_notes").fetchone()["c"] == 1


def test_retelling_leaves_every_other_memory_alone(conn):
    from albums.memory_store import Memory, photo_ids_for_group, replace_memories, rewrite_memory
    from tests.factories import add_photo

    for pid in (1, 2, 3, 4):
        add_photo(conn, photo_id=pid, content_hash=f"h{pid}", thumb_key=f"{pid}.jpg")
    replace_memories(conn, 1, [Memory("A", "a", [1, 2], "sig"),
                               Memory("B", "b", [3, 4], "sig")])
    rows = conn.execute("SELECT id, name FROM groups WHERE kind='memory' ORDER BY id").fetchall()
    first, second = rows[0]["id"], rows[1]["id"]

    rewrite_memory(conn, first, Memory("A retold", "a2", [2, 1], "sig"))

    assert photo_ids_for_group(conn, first) == [2, 1]
    assert photo_ids_for_group(conn, second) == [3, 4]
    assert conn.execute("SELECT name FROM groups WHERE id = ?", (second,)).fetchone()["name"] == "B"
