import json

from chat.agent import (
    agent_retrieve,
    agentic_gather,
    count_periods,
    count_photos,
    direct_answer,
    list_memories,
    memories_for_show,
    retrieve,
)
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
    # ...and caption vector (dedicated text-embed space, the client's caption model —
    # "fake" here, nomic in prod, §4/§9) so the top-N caption ranking can score it.
    write_caption_vector(conn, pid, l2_normalize(FakeInferenceClient().embed("fake", [caption])[0]))


def _photo_no_capvec(conn, pid, caption):
    # Captioned but its caption vector is not computed yet (mid-backfill library).
    add_photo(conn, photo_id=pid, content_hash=f"h{pid}", thumb_key=f"{pid}.jpg", caption=caption)
    write_vector(conn, pid, FakeEmbedder().embed_texts([caption])[0])


def _kw(**over):
    base = {"dimensions": DIMS, "caption_embed_model": "fake",
            "tag_score_min": 0.2, "planner_model": "fake"}
    return {**base, **over}


def test_partial_captioning_still_returns_semantic_match(conn):
    # A captioned photo whose caption VECTOR is not computed yet must still be
    # returned — its caption signal is unavailable, so top-N keeps it (unscored) for
    # the agent to revalidate, never dropping it.
    _photo_no_capvec(conn, 1, "a dog on a beach")
    client = FakeInferenceClient(responses=['{"semantic": "a dog on a beach"}'])
    ids = retrieve(conn, FakeEmbedder(), client, owner_id=1, question="a dog on a beach",
                   **_kw(k=8))
    assert ids == [1]


def test_narrow_predicate_missing_in_taxonomy_keeps_fusion_match(conn):
    # Planner tags subject=dog, but the library never scored a subject tag: the soft
    # narrow must not wipe the real fusion match to empty.
    _photo(conn, 1, "a dog on a beach")
    client = FakeInferenceClient(responses=['{"semantic": "a dog", "tags": {"subject": ["dog"]}}'])
    ids = retrieve(conn, FakeEmbedder(), client, owner_id=1, question="a dog",
                   **_kw(k=8))
    assert 1 in ids


# --- Task 3: retriever v2 (top-N caption ranking, no floor — §10 / design §4) --

def test_ranks_the_matching_caption_first(conn):
    # retrieve() returns the top-N candidates ranked by caption-meaning cosine — no
    # floor gate (honest-empty is the agent's job). The exact-match caption ranks first.
    _photo(conn, 1, "a dog on a beach")
    _photo(conn, 2, "a plate of pasta")
    client = FakeInferenceClient(responses=['{"semantic": "a dog on a beach"}'])
    ids = retrieve(conn, FakeEmbedder(), client, owner_id=1, question="a dog on a beach",
                   **_kw(k=8))
    assert ids and ids[0] == 1


def test_planner_failure_falls_back_to_fusion(conn, monkeypatch):
    _photo(conn, 1, "a dog on a beach")
    client = FakeInferenceClient(responses=['{"semantic": "a dog on a beach"}'])
    monkeypatch.setattr("chat.agent.rerank",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError()))
    ids = retrieve(conn, FakeEmbedder(), client, owner_id=1, question="a dog on a beach",
                   **_kw(k=8))
    assert 1 in ids  # degraded to chat.retrieve.retrieve, never crashes


# --- Task 4: bounded verify/refine loop ---------------------------------------

def test_agent_verifies_a_subset_of_the_seed(conn):
    _photo(conn, 1, "a dog on a beach")
    _photo(conn, 2, "a dog in a park")
    client = FakeInferenceClient(responses=[
        '{"semantic": "a dog"}',
        json.dumps({"action": "answer", "photo_ids": [1]}),
    ])
    ids = agent_retrieve(conn, FakeEmbedder(), client, owner_id=1, question="a dog",
                         **_kw(k=8, max_rounds=3))
    assert ids == [1]


def test_agent_can_expand_then_answer(conn):
    _photo(conn, 1, "a dog on a beach")
    client = FakeInferenceClient(responses=[
        '{"semantic": "a dog"}',
        json.dumps({"action": "expand", "tool": "search", "query": "dog"}),
        json.dumps({"action": "answer", "photo_ids": [1]}),
    ])
    ids = agent_retrieve(conn, FakeEmbedder(), client, owner_id=1, question="a dog",
                         **_kw(k=8, max_rounds=3))
    assert ids == [1]


def test_agent_answering_none_returns_empty(conn):
    _photo(conn, 1, "a plate of pasta")
    client = FakeInferenceClient(responses=[
        '{"semantic": "a dog"}',
        json.dumps({"action": "answer", "photo_ids": []}),
    ])
    ids = agent_retrieve(conn, FakeEmbedder(), client, owner_id=1, question="a dog",
                         **_kw(k=8, max_rounds=3))
    assert ids == []


# --- Direct-DB tools: count / memories / periods (§10) ------------------------

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


def _seed_two_memories(conn):
    from albums.memory_store import Memory, replace_memories

    for pid in range(1, 6):
        add_photo(conn, photo_id=pid, content_hash=f"h{pid}", thumb_key=f"{pid}.jpg",
                  shot_at=f"2024-01-0{pid}T00:00:00")
    replace_memories(conn, 1, [
        Memory("A day at Borjomi", "Walk in Borjomi park.", [1, 2], "sig"),      # size 2
        Memory("Tbilisi evening", "Old town at night.", [3, 4, 5], "sig"),       # size 3
    ])


def test_memories_for_show_all_returns_every_memory(conn):
    # "show me all my memories" is a plural/all request — it must show EVERY
    # memory, not confabulate "no memory matches 'all'" (§10).
    _seed_two_memories(conn)
    names = {m["name"] for m in memories_for_show(conn, 1, "show me all my memories")}
    assert names == {"A day at Borjomi", "Tbilisi evening"}


def test_memories_for_show_specific_returns_one(conn):
    # A narrowing request still returns just the matched memory.
    _seed_two_memories(conn)
    mems = memories_for_show(conn, 1, "show me my borjomi memory")
    assert [m["name"] for m in mems] == ["A day at Borjomi"]


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


def test_direct_answer_counts_the_real_total_without_a_model(conn):
    # A total-count question is answered straight from the DB — no client passed,
    # so it cannot use the model (and, in chat_stream, takes no CHAT lease, §8.1).
    # The number is the REAL total, never the handful of candidates a loop would see.
    for pid in range(1, 6):
        _photo(conn, pid, f"photo {pid}")
    answer = direct_answer(conn, owner_id=1, question="how many images i have in my libray?")
    assert answer is not None and "5" in answer


def test_direct_answer_declines_a_relational_count(conn):
    # "how many similar to this dog" needs embedding similarity, not a keyword count.
    # direct_answer must DECLINE (return None) so the turn reaches the agent, never
    # confabulate the whole-library total (§10, the conservative-decline rule).
    for pid in range(1, 6):
        _photo(conn, pid, f"photo {pid}")
    q = "how many similar photos to this dog one you found we have?"
    assert direct_answer(conn, owner_id=1, question=q) is None


def test_agentic_gather_counts_via_the_count_tool(conn):
    # Direct-OFF fully-agentic path: the model calls count_photos, so the gathered
    # block carries the REAL total as a fact line (never inferred from candidates).
    for pid in range(1, 6):
        add_photo(conn, photo_id=pid, content_hash=f"h{pid}", thumb_key=f"{pid}.jpg")
    fake = FakeInferenceClient(responses=[
        json.dumps({"action": "count_photos", "query": "", "grain": None}),
        json.dumps({"action": "answer", "query": None, "grain": None}),
    ])
    block, grounded = agentic_gather(conn, FakeEmbedder(), fake, "fake", 1, "how many photos?")
    assert grounded is True
    assert "count: 5" in block


def test_agentic_gather_lists_memories_via_the_tool(conn):
    from albums.memory_store import Memory, replace_memories
    for pid in (1, 2):
        add_photo(conn, photo_id=pid, content_hash=f"h{pid}", thumb_key=f"{pid}.jpg",
                  shot_at=f"2023-12-0{pid}T10:00:00")
    replace_memories(conn, 1, [Memory("Borjomi", "A day out.", [1, 2], "sig")])
    fake = FakeInferenceClient(responses=[
        json.dumps({"action": "list_memories", "query": None, "grain": None}),
        json.dumps({"action": "answer", "query": None, "grain": None}),
    ])
    block, grounded = agentic_gather(conn, FakeEmbedder(), fake, "fake", 1, "what memories do I have?")
    assert grounded is True
    assert "Borjomi" in block


def test_agentic_system_prompt_reserves_search_for_library_questions():
    # `search` is the ONLY agentic tool that loads SigLIP, and on the Jetson SigLIP
    # evicts gemma (§8.1) — so the prompt must tell the model to answer world
    # questions with no tool, and to keep `search` for showing the user's own photos.
    from chat.agent import _AGENTIC_SYSTEM

    prompt = _AGENTIC_SYSTEM.lower()
    assert "own photos" in prompt
    assert "call no tool" in prompt
    assert "regions are in russia" in prompt  # the worked negative example
    assert "expensive" in prompt
    # The no-tool branch is stated BEFORE the tool menu, so the model reads it first.
    assert prompt.index("call no tool") < prompt.index('"action":"count_photos"')


def test_agentic_turns_are_deterministic_and_token_capped(conn):
    # A tool-selection turn emits ONE small JSON object. Uncapped and sampled, a
    # grammar-constrained decode can run to the whole KV context and come back EMPTY
    # (measured on the jetson: 1606 tokens, truncated, 56 s, no output).
    from chat.agent import _ROUTING_MAX_TOKENS

    fake = FakeInferenceClient(responses=[
        json.dumps({"action": "count_photos", "query": "", "grain": None}),
        json.dumps({"action": "answer", "query": None, "grain": None}),
    ])
    agentic_gather(conn, FakeEmbedder(), fake, "fake", 1, "how many photos?")

    assert fake.complete_kwargs, "no completion was issued"
    for kwargs in fake.complete_kwargs:
        assert kwargs["temperature"] == 0
        assert kwargs["max_tokens"] == _ROUTING_MAX_TOKENS


def test_route_is_deterministic_and_token_capped():
    from chat.agent import _ROUTING_MAX_TOKENS, route

    fake = FakeInferenceClient(responses=[json.dumps({"tool": "none", "query": None})])
    route(fake, "fake", "how many regions are in Russia?")

    assert fake.complete_kwargs[0]["temperature"] == 0
    assert fake.complete_kwargs[0]["max_tokens"] == _ROUTING_MAX_TOKENS


def test_agentic_gather_general_question_is_not_grounded(conn):
    # No tool call -> nothing gathered -> answer from general knowledge (empty block).
    fake = FakeInferenceClient(responses=[
        json.dumps({"action": "answer", "query": None, "grain": None}),
    ])
    block, grounded = agentic_gather(conn, FakeEmbedder(), fake, "fake", 1, "hi there")
    assert grounded is False
    assert block == ""


def test_is_app_topic_covers_counts_memories_albums_and_features():
    from chat.agent import is_app_topic
    for q in [
        "how many photos do I have?",
        "number of memories",
        "show me my memories",
        "what albums do I have?",
        "list my albums",
        "how many months of photos?",
        "how many pictures with dogs",
        "what's in my library",
        "how many uploads",
    ]:
        assert is_app_topic(q) is True, q


def test_is_app_topic_false_for_genuinely_off_topic():
    from chat.agent import is_app_topic
    for q in [
        "should I walk or drive to work?",
        "what is the capital of France?",
        "write me a python function",
        "how do I bake bread?",
    ]:
        assert is_app_topic(q) is False, q


def test_uppercase_queries_still_find_the_photo(conn):
    # SigLIP's text tower is trained on lower-case web captions and is case
    # SENSITIVE in a way that misleads: "DOG" reads as text PRINTED IN the image, so
    # it returned documents and ID cards while "dog" returned the dog at rank 1.
    # Measured on the board: "show me all photos with DOG on it!" found nothing.
    from embedding.fakes import FakeEmbedder
    from search.semantic import search_photos
    from tests.factories import add_photo

    class CaseSpy(FakeEmbedder):
        def __init__(self):
            super().__init__()
            self.seen = []

        def embed_texts(self, texts):
            self.seen.extend(texts)
            return super().embed_texts(texts)

    add_photo(conn, photo_id=1, content_hash="h1", thumb_key="1.jpg")
    spy = CaseSpy()
    search_photos(conn, spy, 1, "show me all photos with DOG on it!", 5)
    assert spy.seen == ["show me all photos with dog on it!"]


def test_priming_is_off_by_default_so_a_general_question_never_wakes_siglip(conn):
    # Default (guardrails off): gemma decides first. A question that never calls
    # `search` must not touch SigLIP at all — otherwise an already-resident gemma is
    # evicted and reloaded for nothing (§8.1).
    from chat.agent import agentic_gather
    from embedding.fakes import FakeEmbedder
    from inference.fakes import FakeInferenceClient
    from tests.factories import add_photo

    calls = []

    class CountingEmbedder(FakeEmbedder):
        def embed_texts(self, texts):
            calls.append(texts[0])
            return super().embed_texts(texts)

    add_photo(conn, photo_id=1, content_hash="h1", thumb_key="1.jpg", caption="a dog")
    client = FakeInferenceClient(
        responses=[json.dumps({"action": "answer", "query": None, "grain": None})]
    )
    agentic_gather(conn, CountingEmbedder(), client, "m", 1, "how many planets are there")
    assert calls == []


def test_priming_on_retrieves_before_the_model_and_only_once(conn):
    # Guardrails on: the question is about the library by policy, so retrieve up
    # front and let the first `search` serve from that pool — one SigLIP call.
    from chat.agent import agentic_gather
    from embedding.fakes import FakeEmbedder
    from inference.fakes import FakeInferenceClient
    from tests.factories import add_photo

    calls = []

    class CountingEmbedder(FakeEmbedder):
        def embed_texts(self, texts):
            calls.append(texts[0])
            return super().embed_texts(texts)

    for pid in (1, 2):
        add_photo(conn, photo_id=pid, content_hash=f"h{pid}", thumb_key=f"{pid}.jpg",
                  caption="a dog")
    client = FakeInferenceClient(responses=[
        json.dumps({"action": "search", "query": "dog", "grain": None}),
        json.dumps({"action": "answer", "query": None, "grain": None}),
    ])
    agentic_gather(conn, CountingEmbedder(), client, "m", 1, "show me photos of a dog",
                   prime=True)
    assert len(calls) == 1
    assert calls[0] == "show me photos of a dog"


def test_a_general_question_is_not_grounded_by_the_primed_pool(conn):
    # Priming must not make every message look photo-related: the pool is
    # candidates, not evidence, so `grounded` stays False when the model answers
    # without calling a tool.
    from chat.agent import agentic_gather
    from embedding.fakes import FakeEmbedder
    from inference.fakes import FakeInferenceClient
    from tests.factories import add_photo

    add_photo(conn, photo_id=1, content_hash="h1", thumb_key="1.jpg", caption="a dog")
    client = FakeInferenceClient(
        responses=[json.dumps({"action": "answer", "query": None, "grain": None})]
    )
    block, grounded = agentic_gather(
        conn, FakeEmbedder(), client, "m", 1, "what is the capital of France"
    )
    assert grounded is False and block == ""


def test_photo_show_intent_is_recognised_without_a_model():
    from chat.agent import is_photo_show

    for q in ("find me all photos with dog", "show me photos of a dog",
              "find all photos with man in black", "show my beach pictures",
              "list photos with cars"):
        assert is_photo_show(q), q
    # A count is not a show request, nor is a memory request, nor a general question.
    for q in ("how many photos do I have", "how many photos with dogs",
              "show me my Tbilisi memory", "what is the capital of France",
              "how do I bake bread"):
        assert not is_photo_show(q), q


def test_relative_strength_is_used_not_an_absolute_score(conn):
    # Measured on the real library: SigLIP's calibrated probability for a CORRECT
    # top hit is 0.76%-9.8%, and top-1 cosines for subjects that ARE present
    # (0.0889-0.1313) overlap those for subjects that are NOT (0.0137-0.0921) —
    # "a birthday cake" (present, 0.0889) scores below "sushi on a plate" (absent,
    # 0.0921). So no absolute threshold works, and printing ~1% would read as "no
    # match". Only the ordering within one query is meaningful.
    from embedding.fakes import FakeEmbedder
    from search.semantic import search_photos_scored
    from tests.factories import add_photo

    fe = FakeEmbedder()
    for pid, word in ((1, "beach"), (2, "keyboard"), (3, "mountain")):
        add_photo(conn, photo_id=pid, content_hash=f"h{pid}", thumb_key=f"{pid}.jpg")
        write_vector(conn, pid, fe.embed_texts([word])[0])

    scored = search_photos_scored(conn, fe, 1, "beach", 3)
    assert scored, "expected hits"
    assert scored[0][1] == 1.0                      # best hit is the reference point
    assert all(0.0 <= strength <= 1.0 for _, strength in scored)
    assert [s for _, s in scored] == sorted((s for _, s in scored), reverse=True)


def test_the_context_tells_the_model_what_the_image_model_saw(conn):
    # The caption cannot mention every attribute (clothing colour, small objects),
    # so the block must carry the visual rank — otherwise the model rejects rank 1
    # purely because the caption is worded differently, which is the "man in black"
    # bug: SigLIP ranked it #1 and gemma answered "no photos".
    from chat.context import build_context
    from tests.factories import add_photo

    add_photo(conn, photo_id=1, content_hash="h1", thumb_key="1.jpg",
              caption="A man with facial hair looks directly at the camera")
    block = build_context(conn, [1], strengths={1: 1.0})
    assert "visual match: rank 1" in block
    assert "100% as close as the best match" in block
    # Without strengths the block is unchanged — other callers must not grow a
    # meaningless rank line.
    assert "visual match" not in build_context(conn, [1])
