import json

from chat.agent import agent_retrieve, count_periods, count_photos, list_memories, retrieve
from embedding.fakes import FakeEmbedder
from embedding.store import write_caption_vector, write_vector
from embedding.vectors import l2_normalize
from inference.fakes import FakeInferenceClient
from tests.factories import add_photo

DIMS = ["subject", "setting", "vibe"]


def _photo(conn, pid, caption):
    add_photo(conn, photo_id=pid, content_hash=f"h{pid}", thumb_key=f"{pid}.jpg", caption=caption)
    # image vector (SigLIP space) so semantic fusion finds it...
    write_vector(conn, pid, FakeEmbedder().embed_texts([caption])[0])
    # ...and caption vector (inference-client space) so rerank can score it.
    write_caption_vector(conn, pid, l2_normalize(FakeInferenceClient().embed("fake", [caption])[0]))


def _photo_no_capvec(conn, pid, caption):
    # Captioned but its caption vector is not computed yet (mid-backfill library).
    add_photo(conn, photo_id=pid, content_hash=f"h{pid}", thumb_key=f"{pid}.jpg", caption=caption)
    write_vector(conn, pid, FakeEmbedder().embed_texts([caption])[0])


def _kw(**over):
    base = {"dimensions": DIMS, "caption_model": "fake", "tag_score_min": 0.2,
            "planner_model": "fake"}
    return {**base, **over}


def test_partial_captioning_still_returns_semantic_match(conn):
    # The live bug: a captioned photo whose caption VECTOR is not computed yet must
    # not be floored out of chat — its caption signal is unavailable, not weak.
    _photo_no_capvec(conn, 1, "a dog on a beach")
    client = FakeInferenceClient(responses=['{"semantic": "a dog on a beach"}'])
    ids = retrieve(conn, FakeEmbedder(), client, owner_id=1, question="a dog on a beach",
                   **_kw(k=8, floor=0.5))
    assert ids == [1]


def test_narrow_predicate_missing_in_taxonomy_keeps_fusion_match(conn):
    # Planner tags subject=dog, but the library never scored a subject tag: the soft
    # narrow must not wipe the real fusion match to empty.
    _photo(conn, 1, "a dog on a beach")
    client = FakeInferenceClient(responses=['{"semantic": "a dog", "tags": {"subject": ["dog"]}}'])
    ids = retrieve(conn, FakeEmbedder(), client, owner_id=1, question="a dog",
                   **_kw(k=8, floor=-1.0))
    assert 1 in ids


# --- Task 3: retriever v2 -----------------------------------------------------

def test_returns_only_photos_above_the_floor(conn):
    _photo(conn, 1, "a dog on a beach")
    _photo(conn, 2, "a plate of pasta")
    client = FakeInferenceClient(responses=['{"semantic": "a dog on a beach"}'])
    ids = retrieve(conn, FakeEmbedder(), client, owner_id=1, question="a dog on a beach",
                   **_kw(k=8, floor=0.5))
    assert ids == [1]


def test_empty_when_nothing_clears_the_floor(conn):
    _photo(conn, 1, "a plate of pasta")
    client = FakeInferenceClient(responses=['{"semantic": "a dog on a beach"}'])
    ids = retrieve(conn, FakeEmbedder(), client, owner_id=1, question="a dog on a beach",
                   **_kw(k=8, floor=0.9))
    assert ids == []


def test_planner_failure_falls_back_to_fusion(conn, monkeypatch):
    _photo(conn, 1, "a dog on a beach")
    client = FakeInferenceClient(responses=['{"semantic": "a dog on a beach"}'])
    monkeypatch.setattr("chat.agent.rerank",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError()))
    ids = retrieve(conn, FakeEmbedder(), client, owner_id=1, question="a dog on a beach",
                   **_kw(k=8, floor=0.5))
    assert 1 in ids  # degraded to chat.retrieve.retrieve, never crashes


# --- Task 4: bounded verify/refine loop ---------------------------------------

def test_agent_verifies_a_subset_of_the_seed(conn):
    _photo(conn, 1, "a dog on a beach")
    _photo(conn, 2, "a dog in a park")
    client = FakeInferenceClient(responses=[
        '{"semantic": "a dog"}',
        json.dumps({"action": "answer", "photo_ids": [1]}),
    ])
    ids, facts = agent_retrieve(conn, FakeEmbedder(), client, owner_id=1, question="a dog",
                                **_kw(k=8, floor=-1.0, max_rounds=3))
    assert ids == [1]
    assert facts == []


def test_agent_can_expand_then_answer(conn):
    _photo(conn, 1, "a dog on a beach")
    client = FakeInferenceClient(responses=[
        '{"semantic": "a dog"}',
        json.dumps({"action": "expand", "tool": "search", "query": "dog"}),
        json.dumps({"action": "answer", "photo_ids": [1]}),
    ])
    ids, facts = agent_retrieve(conn, FakeEmbedder(), client, owner_id=1, question="a dog",
                                **_kw(k=8, floor=-1.0, max_rounds=3))
    assert ids == [1]
    assert facts == []


def test_agent_answering_none_returns_empty(conn):
    _photo(conn, 1, "a plate of pasta")
    client = FakeInferenceClient(responses=[
        '{"semantic": "a dog"}',
        json.dumps({"action": "answer", "photo_ids": []}),
    ])
    ids, facts = agent_retrieve(conn, FakeEmbedder(), client, owner_id=1, question="a dog",
                                **_kw(k=8, floor=-1.0, max_rounds=3))
    assert ids == []
    assert facts == []


# --- Aggregate tools: count / memories / periods (§10, the "8 photos" bug fix) --

def test_agent_auto_counts_total_even_when_model_never_calls_the_tool(conn):
    # A weak model that just plans then answers (NO count tool-call) must still get
    # the real total via deterministic intent routing — the "8 photos" fix.
    for pid in range(1, 6):
        _photo(conn, pid, f"photo number {pid}")
    client = FakeInferenceClient(responses=[
        '{"semantic": "photos"}',
        json.dumps({"action": "answer", "photo_ids": [1]}),
    ])
    _ids, facts = agent_retrieve(conn, FakeEmbedder(), client, owner_id=1,
                                 question="how many photos do I have in total?",
                                 **_kw(k=8, floor=-1.0))
    assert any("5 photo" in f for f in facts)  # real total (5), not the retrieved count


def test_count_photos_total_when_query_is_empty_or_all(conn):
    for pid in range(1, 4):
        add_photo(conn, photo_id=pid, content_hash=f"h{pid}", thumb_key=f"{pid}.jpg")
    assert count_photos(conn, 1, "") == 3
    assert count_photos(conn, 1, "all") == 3


def test_count_photos_by_keyword_counts_real_matches_only(conn):
    add_photo(conn, photo_id=1, content_hash="h1", thumb_key="1.jpg")
    add_photo(conn, photo_id=2, content_hash="h2", thumb_key="2.jpg")
    conn.execute(
        "INSERT INTO photo_fts(rowid, caption, tags_text) VALUES (1, 'a dog on a beach', 'dog')"
    )
    conn.execute(
        "INSERT INTO photo_fts(rowid, caption, tags_text) VALUES (2, 'a plate of pasta', '')"
    )
    assert count_photos(conn, 1, "dog") == 1


def test_list_memories_returns_name_date_and_size(conn):
    add_photo(conn, photo_id=1, content_hash="h1", thumb_key="1.jpg", shot_at="2024-01-05T00:00:00")
    add_photo(conn, photo_id=2, content_hash="h2", thumb_key="2.jpg", shot_at="2024-01-09T00:00:00")
    gid = conn.execute(
        "INSERT INTO groups(owner_id, kind, name, description, params, status, created_at)"
        " VALUES (1, 'memory', 'Tbilisi trip', '', '{}', 'accepted', '2024-01-01T00:00:00')"
    ).lastrowid
    conn.executemany(
        "INSERT INTO group_photos(group_id, photo_id, rank) VALUES (?, ?, ?)",
        [(gid, 1, 0), (gid, 2, 1)],
    )
    assert list_memories(conn, 1) == [
        {"name": "Tbilisi trip", "date": "2024-01-05T00:00:00", "size": 2}
    ]


def test_count_periods_counts_distinct_months_and_years(conn):
    for pid, when in enumerate((
        "2024-01-01T00:00:00", "2024-01-15T00:00:00",
        "2024-02-01T00:00:00", "2025-03-01T00:00:00",
    ), start=1):
        add_photo(conn, photo_id=pid, content_hash=f"h{pid}", thumb_key=f"{pid}.jpg", shot_at=when)
    n_months, months = count_periods(conn, 1, "month")
    n_years, years = count_periods(conn, 1, "year")
    assert (n_months, months) == (3, ["2024-01", "2024-02", "2025-03"])
    assert (n_years, years) == (2, ["2024", "2025"])


def test_agent_calls_count_tool_and_threads_the_real_total_into_facts(conn):
    # The visible bug: "how many photos in total" must ground on the REAL total
    # via the count tool, never on the handful of seed candidates (here, 5).
    for pid in range(1, 6):
        _photo(conn, pid, f"photo number {pid}")
    client = FakeInferenceClient(responses=[
        '{"semantic": "photos"}',
        json.dumps({"action": "expand", "tool": "count", "query": ""}),
        json.dumps({"action": "answer", "photo_ids": [1]}),
    ])
    ids, facts = agent_retrieve(
        conn, FakeEmbedder(), client, owner_id=1,
        question="how many photos do I have in total?",
        **_kw(k=8, floor=-1.0, max_rounds=3),
    )
    assert ids == [1]
    assert any("5" in fact for fact in facts)
