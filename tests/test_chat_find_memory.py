from albums.memory_store import Memory, replace_memories
from chat.agent import agent_retrieve, find_memories, is_memory_show
from embedding.fakes import FakeEmbedder
from inference.fakes import FakeInferenceClient
from tests.factories import add_photo

DIMS = ["subject", "setting", "vibe"]


def _photo(conn, pid):
    add_photo(conn, photo_id=pid, content_hash=f"h{pid}", thumb_key=f"{pid}.jpg",
              shot_at=f"2024-01-0{pid}T00:00:00")


def _seed_memories(conn):
    for pid in range(1, 6):
        _photo(conn, pid)
    replace_memories(conn, 1, [
        Memory("Trip to Borjomi", "Hiking in Borjomi park, Georgia.", [1, 2], "sig"),
        Memory("Family night in Ontario", "A cozy evening at home.", [3, 4, 5], "sig"),
    ])


def _kw(**over):
    base = {"dimensions": DIMS, "caption_model": "fake", "tag_score_min": 0.2,
            "planner_model": "fake"}
    return {**base, **over}


def test_is_memory_show_distinguishes_show_from_count():
    assert is_memory_show("find memory in borjomi")
    assert is_memory_show("show me any of the memory")
    assert not is_memory_show("how many memories do I have?")  # a count, not a show
    assert not is_memory_show("a dog on a beach")


def test_find_memories_matches_by_place(conn):
    _seed_memories(conn)
    hits = find_memories(conn, 1, "borjomi")
    assert len(hits) == 1
    assert hits[0]["name"] == "Trip to Borjomi"
    assert hits[0]["photo_ids"] == [1, 2]  # its cover photos, in rank order


def test_find_memories_empty_query_returns_the_largest(conn):
    _seed_memories(conn)
    assert find_memories(conn, 1, "")[0]["name"] == "Family night in Ontario"  # 3 photos > 2


def test_find_memories_specific_no_match_is_empty_not_a_random_memory(conn):
    _seed_memories(conn)
    assert find_memories(conn, 1, "antarctica") == []


def test_find_memories_is_owner_scoped(conn):
    _photo(conn, 1)
    add_photo(conn, photo_id=2, owner_id=2, content_hash="h2", thumb_key="2.jpg")
    replace_memories(conn, 1, [Memory("Borjomi mine", "Borjomi.", [1], "s")])
    replace_memories(conn, 2, [Memory("Borjomi theirs", "Borjomi.", [2], "s")])
    hits = find_memories(conn, 1, "borjomi")
    assert [h["name"] for h in hits] == ["Borjomi mine"]


def test_agent_retrieve_finds_a_memory_by_place_without_a_model_call(conn):
    # Deterministic routing: the memory's place ("borjomi") is not in any photo
    # caption, so photo retrieval would find nothing — the memory index answers it,
    # returning the memory's own photos to show. The model is never consulted.
    _seed_memories(conn)
    ids, facts = agent_retrieve(conn, FakeEmbedder(), FakeInferenceClient(),
                                owner_id=1, question="find memory in borjomi", **_kw())
    assert ids == [1, 2]
    assert any("Borjomi" in f for f in facts)


def test_agent_retrieve_shows_any_memory(conn):
    _seed_memories(conn)
    ids, facts = agent_retrieve(conn, FakeEmbedder(), FakeInferenceClient(),
                                owner_id=1, question="show me any of the memory", **_kw())
    assert ids == [3, 4, 5]  # the largest memory
    assert any("Ontario" in f for f in facts)


def test_agent_retrieve_memory_miss_is_honest(conn):
    _seed_memories(conn)
    ids, facts = agent_retrieve(conn, FakeEmbedder(), FakeInferenceClient(),
                                owner_id=1, question="find memory in antarctica", **_kw())
    assert ids == []
    assert any("no memory matches" in f for f in facts)
