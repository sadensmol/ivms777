import json

from albums.compose import MAX_CAPTION_CHARS, MAX_SUMMARY_PHOTOS, compose_memory
from albums.memory_store import Memory
from albums.seeds import Candidate
from inference.fakes import FakeInferenceClient
from tests.factories import add_photo

_ANSWER = json.dumps({
    "action": "answer", "keep": True, "title": "A day",
    "description": "A day out.", "drop_photo_ids": [],
})


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


def test_a_huge_cluster_is_capped_so_the_prompt_fits_the_context(conn):
    # A seed candidate is every photo in a time/place cluster and has no upper
    # bound. Unbounded, ~40 tokens per line overran the jetson's 2048-token
    # context and llama-server rejected the whole request with a 500
    # ("request (3326 tokens) exceeds the available context size"), which used to
    # abort the entire rebuild. Regression for that.
    ids = list(range(1, 81))
    for pid in ids:
        _photo(conn, pid, shot_at="2025-07-12T10:00:00")
    client = FakeInferenceClient([_ANSWER])
    compose_memory(conn, client, "planner", 1, Candidate(ids), signature="s")

    prompt = client.calls[0][1][1]["content"]
    listed = [line for line in prompt.splitlines() if line.startswith("[")]
    assert len(listed) == MAX_SUMMARY_PHOTOS
    # The agent must know the cluster is bigger than what it can see, or it judges
    # coherence on a partial view believing it is the whole.
    assert "+56 more photos in this cluster" in prompt


def test_all_photos_stay_in_the_memory_even_when_the_prompt_is_capped(conn):
    # The cap is a PROMPT limit, not a memory limit — capping what the agent reads
    # must not silently shrink the memory it produces.
    ids = list(range(1, 81))
    for pid in ids:
        _photo(conn, pid, shot_at="2025-07-12T10:00:00")
    memory = compose_memory(conn, FakeInferenceClient([_ANSWER]), "planner", 1,
                            Candidate(ids), signature="s")
    assert memory is not None
    assert memory.photo_ids == ids


def test_an_overlong_caption_is_clipped(conn):
    _photo(conn, 1, caption="x" * 5000, shot_at="2025-07-12T10:00:00")
    for pid in (2, 3):
        _photo(conn, pid, shot_at="2025-07-12T10:00:00")
    client = FakeInferenceClient([_ANSWER])
    compose_memory(conn, client, "planner", 1, Candidate([1, 2, 3]), signature="s")
    assert "x" * (MAX_CAPTION_CHARS + 1) not in client.calls[0][1][1]["content"]


def _facet(conn, photo_id, key, value):
    conn.execute(
        "INSERT INTO photo_facets(photo_id, key, value_text) VALUES (?, ?, ?)",
        (photo_id, key, value),
    )


def test_the_summary_names_the_place_instead_of_coordinates(conn):
    # Fed "41.84,43.39" the model wrote "Activities in a location" and put raw
    # coordinates in the description. The facets stage already knows the name.
    for pid in (1, 2, 3):
        _photo(conn, pid, shot_at="2023-12-01T12:00:00", gps_lat=41.84, gps_lon=43.39)
        _facet(conn, pid, "place_city", "Borjomi")
        _facet(conn, pid, "place_country", "Georgia")
    client = FakeInferenceClient([_ANSWER])
    compose_memory(conn, client, "planner", 1, Candidate([1, 2, 3]), signature="s")

    prompt = client.calls[0][1][1]["content"]
    assert "Borjomi, Georgia" in prompt
    assert "41.84" not in prompt and "43.39" not in prompt


def test_a_photo_without_a_place_name_never_leaks_its_coordinates(conn):
    # GPS but no resolved facet (ingested before the place backfill): the agent gets
    # nothing rather than a number it would copy into the story.
    for pid in (1, 2):
        _photo(conn, pid, shot_at="2023-12-01T12:00:00", gps_lat=41.84, gps_lon=43.39)
    client = FakeInferenceClient([_ANSWER])
    compose_memory(conn, client, "planner", 1, Candidate([1, 2]), signature="s")

    prompt = client.calls[0][1][1]["content"]
    assert "place unknown" in prompt
    assert "41.84" not in prompt


def _tag(conn, photo_id, dimension, label, score=0.9):
    conn.execute("INSERT OR IGNORE INTO tags(dimension, label) VALUES (?, ?)", (dimension, label))
    tag_id = conn.execute(
        "SELECT id FROM tags WHERE dimension = ? AND label = ?", (dimension, label)
    ).fetchone()["id"]
    conn.execute(
        "INSERT INTO photo_tags(photo_id, tag_id, score, source) VALUES (?, ?, ?, 'model')",
        (photo_id, tag_id, score),
    )


def test_the_summary_carries_the_when_where_and_camera_of_the_cluster(conn):
    # The captions say what is in the frame; EXIF says what the day was.
    for pid in (1, 2):
        _photo(conn, pid, shot_at="2023-11-05T15:21:57")
        _facet(conn, pid, "place_city", "Tbilisi")
        _facet(conn, pid, "weekday", "Sunday")
        _facet(conn, pid, "time_of_day", "afternoon")
        _facet(conn, pid, "camera_model", "iPhone SE (2nd generation)")
    client = FakeInferenceClient([_ANSWER])
    compose_memory(conn, client, "planner", 1, Candidate([1, 2]), signature="s")

    prompt = client.calls[0][1][1]["content"]
    assert "When: 2023-11-05, Sunday, afternoon" in prompt
    assert "Where: Tbilisi" in prompt
    assert "Camera: iPhone SE (2nd generation)" in prompt


def test_technical_tags_never_reach_the_story(conn):
    # "sharp"/"pastel" describe the picture, not the day — the model wrote them into
    # the prose ("a moment filled with sharp, joyful holiday cheer").
    for pid in (1, 2):
        _photo(conn, pid, shot_at="2023-12-25T10:00:00")
        _tag(conn, pid, "quality", "sharp", 0.99)
        _tag(conn, pid, "palette", "pastel", 0.98)
        _tag(conn, pid, "occasion", "christmas", 0.5)
    client = FakeInferenceClient([_ANSWER])
    compose_memory(conn, client, "planner", 1, Candidate([1, 2]), signature="s")

    prompt = client.calls[0][1][1]["content"]
    assert "christmas" in prompt
    assert "sharp" not in prompt and "pastel" not in prompt


def test_a_failing_model_call_skips_the_candidate_instead_of_raising(conn):
    # The catch used to list only the parse errors, so an httpx.HTTPStatusError (a
    # 500 from /text/complete) escaped compose_memory, propagated through
    # build_memories, and KILLED the daemon rebuild thread — one bad cluster and
    # nothing at all was stored.
    for pid in (1, 2, 3):
        _photo(conn, pid)

    class Exploding:
        def complete(self, *a, **k):
            raise RuntimeError("500 Internal Server Error")

    assert compose_memory(conn, Exploding(), "planner", 1,
                          Candidate([1, 2, 3]), signature="s") is None
