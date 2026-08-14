from albums.memory_store import (
    Memory,
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


def test_stored_signature_reflects_the_built_set(conn):
    _photo(conn, 1, caption="a")
    assert stored_signature(conn, 1) is None
    replace_memories(conn, 1, [Memory("M", "d.", [1], "sig-42")])
    assert stored_signature(conn, 1) == "sig-42"
