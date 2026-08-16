from albums.memory_store import (
    Memory,
    album_key_for_group,
    current_signature,
    read_memories,
    replace_memories,
    stored_signature,
)
from tests.factories import add_photo


def _photo(conn, pid, **cols):
    return add_photo(conn, photo_id=pid, content_hash=f"h{pid}", thumb_key=f"{pid}.jpg", **cols)


def test_replace_then_read_round_trips_in_rank_order(conn):
    _photo(conn, 1); _photo(conn, 2); _photo(conn, 3)
    replace_memories(conn, owner_id=1, memories=[
        Memory(title="Family night in Ontario",
               description="A family having fun, 22 Nov 1999.",
               photo_ids=[2, 3, 1], signature="3:2026-01-01T00:00:00"),
    ])
    stored = read_memories(conn, owner_id=1)
    assert len(stored) == 1
    assert stored[0].title == "Family night in Ontario"
    assert stored[0].photo_ids == [2, 3, 1]  # cover (rank 0) first


def test_replace_clears_the_previous_memories(conn):
    _photo(conn, 1)
    sig = "1:2026-01-01T00:00:00"
    replace_memories(conn, 1, [Memory("Old", "old.", [1], sig)])
    replace_memories(conn, 1, [Memory("New", "new.", [1], sig)])
    titles = [m.title for m in read_memories(conn, 1)]
    assert titles == ["New"]


def _fts_hits(conn, term):
    return conn.execute(
        "SELECT count(*) FROM memory_fts WHERE memory_fts MATCH ?", (term,)
    ).fetchone()[0]


def test_replace_memories_indexes_and_reindexes_the_fts(conn):
    _photo(conn, 1); _photo(conn, 2)
    replace_memories(conn, 1, [Memory("Trip to Borjomi", "Hiking in Borjomi.", [1], "s")])
    assert _fts_hits(conn, "borjomi") == 1
    # A rebuild swaps the index in lockstep: the old term is gone, the new one indexed.
    replace_memories(conn, 1, [Memory("Ontario night", "Cozy at home.", [2], "s")])
    assert _fts_hits(conn, "borjomi") == 0
    assert _fts_hits(conn, "ontario") == 1


def test_backfill_indexes_memories_that_predate_the_fts(conn):
    from db.connection import _backfill_memory_fts
    _photo(conn, 1)
    conn.execute(
        "INSERT INTO groups(owner_id, kind, name, description, params, status, created_at)"
        " VALUES (1, 'memory', 'Trip to Borjomi', 'Hiking in Borjomi.', '{}', 'accepted', '2024-01-01T00:00:00')"
    )
    assert _fts_hits(conn, "borjomi") == 0  # built before memory_fts existed
    _backfill_memory_fts(conn)
    assert _fts_hits(conn, "borjomi") == 1
    _backfill_memory_fts(conn)  # idempotent — no duplicate rows
    assert _fts_hits(conn, "borjomi") == 1


def test_memories_are_owner_scoped(conn):
    add_photo(conn, photo_id=1, owner_id=1, content_hash="a", thumb_key="a.jpg")
    add_photo(conn, photo_id=2, owner_id=2, content_hash="b", thumb_key="b.jpg")
    replace_memories(conn, 1, [Memory("Mine", "d.", [1], "s")])
    replace_memories(conn, 2, [Memory("Theirs", "d.", [2], "s")])
    assert [m.title for m in read_memories(conn, 1)] == ["Mine"]


def test_signature_changes_when_a_photo_is_added(conn):
    _photo(conn, 1, caption="a", updated_at="2026-01-01T00:00:00")
    before = current_signature(conn, 1)
    _photo(conn, 2, caption="b", updated_at="2026-02-02T00:00:00")
    assert current_signature(conn, 1) != before


def test_album_key_for_group_matches_organize_position(conn):
    # The album key is the memory's position in read_memories order (size desc,
    # then id) — the same order MemoriesOrganizer numbers albums, so chat links
    # into the exact grid Organize shows.
    for pid in (1, 2, 3, 4, 5):
        _photo(conn, pid)
    replace_memories(conn, 1, [
        Memory("Small", "two.", [1, 2], "s"),          # inserted first → id smaller
        Memory("Big", "three.", [3, 4, 5], "s"),       # more photos → sorts first
    ])
    ids = {row["name"]: row["id"] for row in conn.execute(
        "SELECT id, name FROM groups WHERE owner_id = 1 AND kind = 'memory'")}
    assert album_key_for_group(conn, 1, ids["Big"]) == "memory-0"    # 3 photos, first
    assert album_key_for_group(conn, 1, ids["Small"]) == "memory-1"  # 2 photos, second
    assert album_key_for_group(conn, 1, 9999) is None                # unknown group


def test_album_key_for_group_is_owner_scoped(conn):
    add_photo(conn, photo_id=1, owner_id=1, content_hash="a", thumb_key="a.jpg")
    add_photo(conn, photo_id=2, owner_id=2, content_hash="b", thumb_key="b.jpg")
    replace_memories(conn, 2, [Memory("Theirs", "d.", [2], "s")])
    theirs = conn.execute(
        "SELECT id FROM groups WHERE owner_id = 2 AND kind = 'memory'").fetchone()["id"]
    assert album_key_for_group(conn, 1, theirs) is None  # not owner 1's memory


def test_stored_signature_reflects_the_built_set(conn):
    _photo(conn, 1, caption="a")
    assert stored_signature(conn, 1) is None
    replace_memories(conn, 1, [Memory("M", "d.", [1], "sig-42")])
    assert stored_signature(conn, 1) == "sig-42"
