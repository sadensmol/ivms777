import json

from albums.memories import MemoriesOrganizer
from albums.memories_build import build_memories
from albums.memory_store import read_memories
from inference.fakes import FakeInferenceClient
from tests.factories import add_photo


def _run(conn, pids, shot):
    for pid in pids:
        add_photo(conn, photo_id=pid, content_hash=f"h{pid}", thumb_key=f"{pid}.jpg",
                  caption="x", shot_at=shot.format(pid))


def _keep(title):
    return json.dumps({"action": "answer", "keep": True, "title": title,
                       "description": f"{title}.", "drop_photo_ids": []})


def test_build_writes_one_memory_per_kept_candidate(conn):
    _run(conn, (1, 2, 3), "2025-07-12T1{}:00:00")
    n = build_memories(conn, FakeInferenceClient([_keep("Beach day")]), "planner", owner_id=1)
    assert n == 1
    assert read_memories(conn, 1)[0].title == "Beach day"


def test_build_is_skipped_when_the_signature_matches(conn):
    _run(conn, (1, 2, 3), "2025-07-12T1{}:00:00")
    build_memories(conn, FakeInferenceClient([_keep("A")]), "planner", 1)
    # No queued responses: a second build must NOT call the model.
    empty = FakeInferenceClient([])
    n = build_memories(conn, empty, "planner", 1)
    assert n == 1
    assert empty.calls == []


def test_force_rebuilds_even_when_the_signature_matches(conn):
    _run(conn, (1, 2, 3), "2025-07-12T1{}:00:00")
    build_memories(conn, FakeInferenceClient([_keep("A")]), "planner", 1)
    build_memories(conn, FakeInferenceClient([_keep("B")]), "planner", 1, force=True)
    assert read_memories(conn, 1)[0].title == "B"


def test_build_reports_progress_per_candidate(conn):
    _run(conn, (1, 2, 3), "2025-07-12T1{}:00:00")   # one candidate
    calls: list[tuple[int, int]] = []
    build_memories(conn, FakeInferenceClient([_keep("A")]), "planner", 1,
                   progress=lambda done, total: calls.append((done, total)))
    assert calls[0] == (0, 1)     # starts at 0/total
    assert calls[-1] == (1, 1)    # ends at total/total


def test_stored_memory_shows_up_as_an_album(conn):
    _run(conn, (1, 2, 3), "2025-07-12T1{}:00:00")
    build_memories(conn, FakeInferenceClient([_keep("Beach day")]), "planner", 1)
    albums = MemoriesOrganizer().organize(conn, owner_id=1)
    assert albums[0].title == "Beach day"
    assert albums[0].cover_id in (1, 2, 3)
    assert albums[0].meta["kind"] == "memory"
