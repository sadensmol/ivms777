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


def test_one_failing_candidate_does_not_abort_the_whole_rebuild(conn):
    # The rebuild runs on a bare daemon thread (§11), so an exception escaping
    # build_memories killed it outright: "Exception in thread Thread-2". The
    # remaining candidates were never tried and replace_memories never ran, so a
    # single bad cluster left the library with ZERO memories.
    _run(conn, (1, 2, 3), "2025-07-12T1{}:00:00")
    _run(conn, (4, 5, 6), "2025-09-20T1{}:00:00")

    class FlakyClient:
        """Fails the first candidate, answers the second."""

        def __init__(self):
            self.calls = 0

        def complete(self, *a, **k):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("500 Internal Server Error")
            return _keep("Autumn trip")

    n = build_memories(conn, FlakyClient(), "planner", owner_id=1)
    assert n == 1                                            # the good one survived
    assert [m.title for m in read_memories(conn, 1)] == ["Autumn trip"]
