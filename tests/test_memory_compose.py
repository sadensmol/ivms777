import json

from albums.compose import compose_memory
from albums.memory_store import Memory
from albums.seeds import Candidate
from inference.fakes import FakeInferenceClient
from tests.factories import add_photo


def _photo(conn, pid, **cols):
    return add_photo(conn, photo_id=pid, content_hash=f"h{pid}", thumb_key=f"{pid}.jpg",
                     caption=cols.pop("caption", "a photo"), **cols)


def test_a_kept_candidate_becomes_a_named_described_memory(conn):
    for pid in (1, 2, 3):
        _photo(conn, pid, shot_at="1999-11-22T20:00:00")
    client = FakeInferenceClient([json.dumps({
        "action": "answer", "keep": True,
        "title": "Family night in Ontario",
        "description": "A family having fun, 22 Nov 1999.",
        "drop_photo_ids": [],
    })])
    memory = compose_memory(conn, client, "planner", owner_id=1,
                            candidate=Candidate([1, 2, 3]), signature="sig")
    assert isinstance(memory, Memory)
    assert memory.title == "Family night in Ontario"
    assert memory.photo_ids == [1, 2, 3]
    assert memory.signature == "sig"


def test_a_skipped_candidate_returns_none(conn):
    for pid in (1, 2, 3):
        _photo(conn, pid)
    client = FakeInferenceClient([json.dumps(
        {"action": "answer", "keep": False, "title": "", "description": "", "drop_photo_ids": []}
    )])
    assert compose_memory(conn, client, "planner", 1, Candidate([1, 2, 3]), signature="s") is None


def test_dropped_outliers_are_removed_from_the_memory(conn):
    for pid in (1, 2, 3, 4):
        _photo(conn, pid)
    client = FakeInferenceClient([json.dumps({
        "action": "answer", "keep": True, "title": "Beach day",
        "description": "An afternoon by the water.", "drop_photo_ids": [4],
    })])
    memory = compose_memory(conn, client, "planner", 1, Candidate([1, 2, 3, 4]), signature="s")
    assert 4 not in memory.photo_ids
    assert memory.photo_ids == [1, 2, 3]


def test_the_agent_can_request_context_then_answer(conn):
    for pid in (1, 2, 3):
        _photo(conn, pid)
    client = FakeInferenceClient([
        json.dumps({"action": "expand", "tool": "similar", "photo_id": 1}),
        json.dumps({"action": "answer", "keep": True, "title": "T",
                    "description": "d.", "drop_photo_ids": []}),
    ])
    memory = compose_memory(conn, client, "planner", 1, Candidate([1, 2, 3]),
                            signature="s", max_rounds=3)
    assert memory is not None
    assert len(client.calls) == 2  # one expand round, then the answer


def test_dropping_below_two_photos_skips_the_memory(conn):
    for pid in (1, 2):
        _photo(conn, pid)
    client = FakeInferenceClient([json.dumps({
        "action": "answer", "keep": True, "title": "T", "description": "d.",
        "drop_photo_ids": [2],
    })])
    assert compose_memory(conn, client, "planner", 1, Candidate([1, 2]), signature="s") is None


def test_unusable_model_output_is_skipped_not_raised(conn):
    for pid in (1, 2, 3):
        _photo(conn, pid)
    client = FakeInferenceClient(["not json at all"])
    assert compose_memory(conn, client, "planner", 1, Candidate([1, 2, 3]), signature="s") is None
