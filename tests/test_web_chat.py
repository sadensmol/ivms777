import json

import pytest
from fastapi.testclient import TestClient

from config import Settings
from embedding.fakes import FakeEmbedder
from embedding.store import write_caption_vector, write_vector
from embedding.vectors import l2_normalize
from inference.fakes import FakeInferenceClient
from tests.factories import add_photo
from web.app import create_app


@pytest.fixture
def chat_client(settings, monkeypatch):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    fake_inf = FakeInferenceClient(streams=[["A beach ", "[photo:1]", "."]])
    monkeypatch.setattr(
        Settings, "build_inference_client", lambda self: (fake_inf, "fake")
    )
    app = create_app(settings)
    conn = app.state.context.conn
    fe = FakeEmbedder()
    for pid, word in ((1, "beach"), (2, "keyboard")):
        add_photo(conn, photo_id=pid, content_hash=word, thumb_key=f"{word}.jpg")
        write_vector(conn, pid, fe.embed_texts([word])[0])
    with TestClient(app) as tc:
        yield tc


def test_chat_page_has_input_log_and_nav_order(chat_client):
    body = chat_client.get("/chat").text
    assert 'id="chat-form"' in body      # the message input
    assert 'id="chat-log"' in body       # the conversation-history container
    assert 'href="/chat"' in body        # nav link present
    # Chat now sits before Organize in the nav (review before reorganize).
    assert body.index('href="/chat"') < body.index('href="/organize"')


def test_stream_emits_tokens_then_done_no_candidate_strip(chat_client):
    body = chat_client.get("/chat/stream?q=beach").text
    assert "A beach " in body
    assert "[photo:1]" in body       # citation passes through untouched, rendered inline
    assert "event: done" in body
    # No raw-candidate strip: the loosely-related retrieved set is never pushed to
    # the client — only what the answer cites (design §6, §10).
    assert "event: sources" not in body


def test_chat_page_shows_the_model_in_use(chat_client):
    assert 'id="chat-model"' in chat_client.get("/chat").text


def test_stream_done_event_reports_model_and_decode_speed(chat_client):
    body = chat_client.get("/chat/stream?q=beach").text
    done = [ln for ln in body.splitlines() if ln.startswith("data:") and "model" in ln][-1]
    stats = json.loads(done[len("data:"):])
    assert stats["model"]  # the configured planner model, shown in the UI
    assert "tok_per_sec" in stats and "tokens" in stats  # decode-speed readout


def test_only_cited_photos_are_persisted_not_the_candidate_set(chat_client):
    # photo 2 (keyboard) is retrieved as a candidate but never cited; only the
    # cited photo 1 must be stored/shown, so the "30 thumbnails, 1 dog" mismatch
    # can't happen.
    chat_client.get("/chat/stream?q=beach")
    body = chat_client.get("/chat").text
    assert "/thumb/1" in body
    assert "/thumb/2" not in body


def test_stream_says_so_when_nothing_retrieved(chat_client, monkeypatch):
    # No query -> no retrieval; the model is handed the no-match sentinel.
    fake_inf = FakeInferenceClient(streams=[["I have no photos matching that."]])
    monkeypatch.setattr(Settings, "build_inference_client", lambda self: (fake_inf, "fake"))
    body = chat_client.get("/chat/stream?q=%20").text
    assert "no photos matching" in body.lower()


def test_history_persists_and_renders_on_load(chat_client):
    chat_client.get("/chat/stream?q=beach")          # streams + persists one turn
    body = chat_client.get("/chat").text             # re-rendered from the DB
    assert "beach" in body                            # the question survives reload
    assert "/thumb/1" in body                         # the persisted answer's citation


def test_new_session_clears_the_visible_history(chat_client):
    chat_client.get("/chat/stream?q=beach")
    chat_client.post("/chat/new")
    body = chat_client.get("/chat").text
    assert "beach" not in body                        # the fresh session is empty


def _searchable(conn, pid, caption, *, with_caption_vec=True):
    add_photo(conn, photo_id=pid, content_hash=f"h{pid}", thumb_key=f"{pid}.jpg", caption=caption)
    write_vector(conn, pid, FakeEmbedder().embed_texts([caption])[0])          # semantic fusion
    if with_caption_vec:
        # caption_vec in the dedicated text-embed space (the client's caption model —
        # "fake" in tests, nomic in prod, §4/§9), the same space the query uses.
        write_caption_vector(conn, pid, l2_normalize(FakeInferenceClient().embed("fake", [caption])[0]))


def test_chat_grounds_only_on_the_agent_verified_match(settings, monkeypatch):
    # The full agentic path runs: gate -> planner spec -> rerank -> verify loop.
    # Photo 2 (pasta) is a candidate but the agent verifies only photo 1, so only
    # photo 1 is ever grounded/cited — the "30 thumbnails, 1 dog" mismatch can't
    # recur.
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    fake = FakeInferenceClient(
        responses=["yes", '{"semantic": "a dog on a beach"}',
                   json.dumps({"action": "answer", "photo_ids": [1]})],
        streams=[["Here ", "[photo:1]", "."]],
    )
    monkeypatch.setattr(Settings, "build_inference_client", lambda self: (fake, "fake"))
    app = create_app(settings)
    conn = app.state.context.conn
    _searchable(conn, 1, "a dog on a beach")
    _searchable(conn, 2, "a plate of pasta")
    with TestClient(app) as tc:
        body = tc.get("/chat/stream", params={"q": "a dog on a beach"}).text
        assert "[photo:1]" in body
        page = tc.get("/chat").text
        assert "/thumb/1" in page
        assert "/thumb/2" not in page          # the pasta photo was never grounded


def test_chat_reports_nothing_found_when_floor_rejects_all(settings, monkeypatch):
    # A photo with no caption vector scores 0.0 < floor -> dropped, so retrieval is
    # empty and the model is handed the no-match sentinel with no sources.
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    fake = FakeInferenceClient(
        responses=["yes", '{"semantic": "a dog on a beach"}'],
        streams=[["I couldn't find any photos of that."]],
    )
    monkeypatch.setattr(Settings, "build_inference_client", lambda self: (fake, "fake"))
    app = create_app(settings)
    conn = app.state.context.conn
    _searchable(conn, 1, "a plate of pasta", with_caption_vec=False)
    with TestClient(app) as tc:
        body = tc.get("/chat/stream", params={"q": "a dog on a beach"}).text
        assert "couldn't find" in body.lower()
        assert "photo:" not in body            # nothing cited
        assert "/thumb/1" not in tc.get("/chat").text


def test_chat_total_count_question_answers_the_real_total_not_shown_photos(settings, monkeypatch):
    # The visible bug (§10): "how many photos in total" answered "8" (the
    # retrieved-candidate count) instead of the real total. Total-count questions
    # are now answered straight from the DB (§8.1), so the answer IS the real total
    # (12) by construction — no model, no retrieved-subset to miscount.
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    question = "how many photos do I have in total?"
    fake = FakeInferenceClient()  # must never be called for a count question
    monkeypatch.setattr(Settings, "build_inference_client", lambda self: (fake, "fake"))
    app = create_app(settings)
    conn = app.state.context.conn
    for pid in range(1, 13):
        add_photo(conn, photo_id=pid, content_hash=f"h{pid}", thumb_key=f"{pid}.jpg")
    with TestClient(app) as tc:
        body = tc.get("/chat/stream", params={"q": question}).text
    assert "12" in body           # the real total
    assert fake.calls == []        # answered deterministically, the model untouched


def test_normal_answer_streams_no_memory_card(chat_client):
    # A non-memory question never carries a memory card — the event is exclusive to
    # "show me a memory" turns.
    body = chat_client.get("/chat/stream?q=beach").text
    assert "event: memory" not in body


def _seed_one_memory(conn):
    from albums.memory_store import Memory, replace_memories
    for pid in (1, 2):
        add_photo(conn, photo_id=pid, content_hash=f"h{pid}", thumb_key=f"{pid}.jpg",
                  shot_at=f"2023-12-0{pid}T10:00:00")
    replace_memories(conn, 1, [Memory("Trip to Borjomi", "A day in Borjomi park.", [1, 2], "sig")])


def test_memory_show_streams_the_memory_card_with_chat_memory_links(settings, monkeypatch):
    # "show me my memory in borjomi" answers with the memory ITSELF: the stream
    # carries an `event: memory` with the Organize card, whose covers link into
    # ctx=chat-memory so drilling in pages within the memory (§10, §13.1).
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    fake = FakeInferenceClient(responses=["yes"], streams=[["Here is your Borjomi memory."]])
    monkeypatch.setattr(Settings, "build_inference_client", lambda self: (fake, "fake"))
    app = create_app(settings)
    _seed_one_memory(app.state.context.conn)
    with TestClient(app) as tc:
        body = tc.get("/chat/stream", params={"q": "show me my memory in borjomi"}).text
    assert "event: memory" in body
    assert "Trip to Borjomi" in body                     # the card's title
    assert "ctx=chat-memory:memory-0" in body            # covers link into the memory grid


def test_show_all_memories_streams_a_card_per_memory(settings, monkeypatch):
    # "show me all my memories" is a plural/all request: the stream carries a card
    # for EVERY memory, never a confabulated "no memory matches 'all'" (§10).
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    fake = FakeInferenceClient(responses=["yes"], streams=[["Here are your memories."]])
    monkeypatch.setattr(Settings, "build_inference_client", lambda self: (fake, "fake"))
    app = create_app(settings)
    from albums.memory_store import Memory, replace_memories
    conn = app.state.context.conn
    for pid in range(1, 5):
        add_photo(conn, photo_id=pid, content_hash=f"h{pid}", thumb_key=f"{pid}.jpg",
                  shot_at=f"2023-12-0{pid}T10:00:00")
    replace_memories(conn, 1, [
        Memory("Trip to Borjomi", "A day in Borjomi park.", [1, 2], "sig"),
        Memory("Tbilisi evening", "Old town at night.", [3, 4], "sig"),
    ])
    with TestClient(app) as tc:
        body = tc.get("/chat/stream", params={"q": "show me all my memories"}).text
    assert "event: memory" in body
    assert "Trip to Borjomi" in body and "Tbilisi evening" in body   # a card each
    assert 'no memory matches' not in body                           # no confabulation


def test_memory_card_survives_reload_in_history(settings, monkeypatch):
    # The card re-renders from the persisted turn (deterministic, no extra stored
    # state), so it survives navigation and restart like the answer text does.
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    fake = FakeInferenceClient(responses=["yes"], streams=[["Here is your Borjomi memory."]])
    monkeypatch.setattr(Settings, "build_inference_client", lambda self: (fake, "fake"))
    app = create_app(settings)
    _seed_one_memory(app.state.context.conn)
    with TestClient(app) as tc:
        tc.get("/chat/stream", params={"q": "show me my memory in borjomi"})
        page = tc.get("/chat").text
    assert "Trip to Borjomi" in page
    assert "ctx=chat-memory:memory-0" in page
    assert 'class="msg-memory"' in page


def test_count_question_streams_answer_not_no_answer(chat_client):
    body = chat_client.get("/chat/stream?q=how many images in my library?").text
    assert "data:" in body and "event: done" in body   # streamed → not "(no answer)"


def test_count_question_answers_without_taking_the_model_lease(settings, monkeypatch):
    # A count/aggregate question is answered straight from the DB (§8.1) — it must
    # NOT take the CHAT lease, so "how many photos?" answers instantly even while
    # ingest holds the caption model. The coordinator here refuses to hand over the
    # lease; the count must still come back (not "busy").
    from web.deps import AppContext

    settings.data_dir.mkdir(parents=True, exist_ok=True)

    class _NeverGivesLease:
        def require(self, workload):
            raise AssertionError("a count question must not require the model lease")

    monkeypatch.setattr(
        AppContext, "make_coordinator", lambda self, client, holder: _NeverGivesLease()
    )
    app = create_app(settings)
    conn = app.state.context.conn
    for pid in range(1, 4):
        add_photo(conn, photo_id=pid, content_hash=f"h{pid}", thumb_key=f"{pid}.jpg")
    with TestClient(app) as tc:
        body = tc.get("/chat/stream", params={"q": "how many photos do I have?"}).text
    assert "3" in body                       # the real total, from the DB
    assert "busy" not in body.lower()         # never fell into the lease-busy path
    assert "event: done" in body


def test_chat_stream_never_aborts_when_the_coordinator_lease_times_out(settings, monkeypatch):
    # §8.1 FIX I2: coordinator.require("CHAT") can raise TimeoutError (or
    # RefusedError / LeaseBusyError) BEFORE any yield — a background workload
    # that won't yield the lease within 30s, or an over-budget model-set. Without
    # a guard the SSE stream aborts with zero output — the "(no answer)" bug.
    # This must never recur: the stream always emits >=1 `data:` + `event: done`.
    from web.deps import AppContext

    settings.data_dir.mkdir(parents=True, exist_ok=True)

    class _TimeoutCoordinator:
        def require(self, workload):
            raise TimeoutError(f"could not acquire model lease for {workload} within 30.0s")

    monkeypatch.setattr(
        AppContext, "make_coordinator",
        lambda self, client, holder: _TimeoutCoordinator(),
    )
    app = create_app(settings)
    with TestClient(app) as tc:
        body = tc.get("/chat/stream?q=beach").text
    assert "data:" in body
    assert "busy" in body.lower()
    assert "event: done" in body


def test_off_topic_question_is_refused_without_retrieval(settings, monkeypatch):
    # The classifier says "no" -> canned refusal, no photos retrieved, still persisted.
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    fake = FakeInferenceClient(responses=["no"])
    monkeypatch.setattr(Settings, "build_inference_client", lambda self: (fake, "fake"))
    app = create_app(settings)
    with TestClient(app) as tc:
        body = tc.get("/chat/stream?q=should+I+walk+or+drive").text
        assert "only answer questions about your photos" in body
        assert "event: sources" not in body           # nothing retrieved, no strip
        history = tc.get("/chat").text
        assert "should I walk or drive" in history     # the refused turn is saved
