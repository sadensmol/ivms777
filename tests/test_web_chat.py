import json
import re

import pytest
from fastapi.testclient import TestClient

from chat.prefs import set_prefs
from config import Settings
from db.settings import set_setting
from embedding.fakes import FakeEmbedder
from embedding.store import write_caption_vector, write_vector
from embedding.vectors import l2_normalize
from inference.fakes import FakeInferenceClient
from tests.factories import add_photo
from web.app import create_app


def _input(body, name):
    return re.search(rf'<input[^>]*name="{name}"[^>]*>', body).group(0)


@pytest.fixture
def chat_client(settings, monkeypatch):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    # A routing response per turn, so the grounded path really runs and photo 1 is
    # in the context. Without it the router falls back to "none" (no photos), and a
    # [photo:1] citation is a fabrication the CitationFilter strips by design.
    fake_inf = FakeInferenceClient(
        responses=[json.dumps({"tool": "search_library", "query": "beach"})] * 4,
        streams=[["A beach ", "[photo:1]", "."]] * 4,
    )
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


def test_chat_names_the_selected_planner_slot_not_the_profile_default(chat_client):
    # The chat header must name the model the `planner` slot HOLDS (§4.1), not the
    # env/profile default: reading `settings.planner_model` said `gemma4-E2B` while
    # the resident model was the one picked in the settings popup.
    conn = chat_client.app.state.context.conn
    set_setting(conn, 1, "model_slot.planner", "qwen3-vl-8b")
    assert "qwen3-vl-8b" in chat_client.get("/chat").text

    done = [ln for ln in chat_client.get("/chat/stream?q=beach").text.splitlines()
            if ln.startswith("data:") and "model" in ln][-1]
    assert json.loads(done[len("data:"):])["model"] == "qwen3-vl-8b"


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
        responses=[json.dumps({"tool": "search_library", "query": "a dog on a beach"}),
                   "yes", '{"semantic": "a dog on a beach"}',
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


def test_stream_surfaces_error_instead_of_silent_no_answer(settings, monkeypatch):
    # When generation fails (the models service / llama-server is down — the jetson
    # OOM bug, §8.1), the UI must get a REAL error turn, never a silent "(no answer)".
    # Route resolves, then the stream raises (empty stream queue stands in for the
    # RemoteProtocolError the down llama-server produces).
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    fake = FakeInferenceClient(responses=['{"tool": "none", "query": null}'], streams=[])
    monkeypatch.setattr(Settings, "build_inference_client", lambda self: (fake, "fake"))
    app = create_app(settings)
    with TestClient(app) as tc:
        body = tc.get("/chat/stream?q=should+I+walk+or+drive").text
        assert "unavailable" in body.lower()   # a real error delta streamed to the UI
        assert "event: done" in body           # the stream still closes cleanly
        # the failed turn is persisted, so reload shows the error, not an empty bubble
        assert "unavailable" in tc.get("/chat").text.lower()


def test_count_question_streams_answer_not_no_answer(chat_client):
    body = chat_client.get("/chat/stream?q=how many images in my library?").text
    assert "data:" in body and "event: done" in body   # streamed → not "(no answer)"


def test_count_question_answers_from_the_db_without_a_model(settings):
    # A count/aggregate question is answered straight from the DB (§10) — no model,
    # no search. "how many photos?" comes back instantly with the real total.
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    app = create_app(settings)
    conn = app.state.context.conn
    for pid in range(1, 4):
        add_photo(conn, photo_id=pid, content_hash=f"h{pid}", thumb_key=f"{pid}.jpg")
    with TestClient(app) as tc:
        body = tc.get("/chat/stream", params={"q": "how many photos do I have?"}).text
    assert "3" in body                       # the real total, from the DB
    assert "event: done" in body


def test_off_topic_question_is_answered_generally(settings, monkeypatch):
    # Chat is NOT photo-limited (plan 17): a general question routes to `none` and is
    # answered directly by the model — no library search, no refusal — still persisted.
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    fake = FakeInferenceClient(
        responses=['{"tool": "none", "query": null}'],   # router: general, no tool
        streams=[["Walking", " is", " fine."]],           # the model's general answer
    )
    monkeypatch.setattr(Settings, "build_inference_client", lambda self: (fake, "fake"))
    app = create_app(settings)
    with TestClient(app) as tc:
        body = tc.get("/chat/stream?q=should+I+walk+or+drive").text
        assert "Walking" in body and "fine." in body      # answered from general knowledge
        assert "only answer questions about your photos" not in body  # not refused
        history = tc.get("/chat").text
        assert "should I walk or drive" in history         # the turn is saved
        assert "Walking is fine." in history               # persisted joined answer


# --- Global toggles: guardrails + direct-answer (§10) -------------------------

def test_chat_page_shows_toggles_with_defaults(chat_client):
    body = chat_client.get("/chat").text
    assert 'name="guardrails"' in body and 'name="direct_answers"' in body
    # direct answers ON by default (checked); guardrails OFF (unchecked)
    assert "checked" in _input(body, "direct_answers")
    assert "checked" not in _input(body, "guardrails")


def test_prefs_post_persists_and_reflects(chat_client):
    chat_client.post("/chat/prefs", data={"guardrails": "on"})  # direct omitted -> off
    body = chat_client.get("/chat").text
    assert "checked" in _input(body, "guardrails")
    assert "checked" not in _input(body, "direct_answers")


def test_guardrails_on_refuses_off_topic(settings, monkeypatch):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    fake = FakeInferenceClient(responses=['{"tool": "none", "query": null}'])  # route -> none
    monkeypatch.setattr(Settings, "build_inference_client", lambda self: (fake, "fake"))
    app = create_app(settings)
    set_prefs(app.state.context.conn, settings.owner_id, guardrails=True, direct_answers=True)
    with TestClient(app) as tc:
        body = tc.get("/chat/stream?q=should+I+walk+or+drive").text
    assert "only answer questions about your photos" in body.lower()
    assert "event: done" in body


def test_guardrails_off_answers_off_topic_generally(settings, monkeypatch):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    fake = FakeInferenceClient(
        responses=['{"tool": "none", "query": null}'], streams=[["Walking ", "is ", "fine."]]
    )
    monkeypatch.setattr(Settings, "build_inference_client", lambda self: (fake, "fake"))
    app = create_app(settings)
    with TestClient(app) as tc:
        body = tc.get("/chat/stream?q=should+I+walk+or+drive").text
    assert "Walking" in body
    assert "only answer questions about your photos" not in body.lower()


def test_direct_off_counts_go_through_the_agentic_count_tool(settings, monkeypatch):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    fake = FakeInferenceClient(
        responses=[
            '{"action": "count_photos", "query": "", "grain": null}',
            '{"action": "answer", "query": null, "grain": null}',
        ],
        streams=[["You have ", "7 ", "photos."]],
    )
    monkeypatch.setattr(Settings, "build_inference_client", lambda self: (fake, "fake"))
    app = create_app(settings)
    conn = app.state.context.conn
    for pid in range(1, 8):
        add_photo(conn, photo_id=pid, content_hash=f"h{pid}", thumb_key=f"{pid}.jpg")
    set_prefs(conn, settings.owner_id, guardrails=False, direct_answers=False)
    with TestClient(app) as tc:
        body = tc.get("/chat/stream?q=how+many+photos+do+I+have").text
    assert "event: done" in body
    # the REAL count reached the final grounded prompt as a fact line
    final_msgs = fake.calls[-1][1]
    assert "count: 7" in final_msgs[1]["content"]


def test_direct_off_memory_show_has_no_card(settings, monkeypatch):
    # whole direct-DB step skipped -> memory-show is prose, no rendered card event
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    fake = FakeInferenceClient(
        responses=['{"action": "answer", "query": null, "grain": null}'],
        streams=[["You have some memories."]],
    )
    monkeypatch.setattr(Settings, "build_inference_client", lambda self: (fake, "fake"))
    app = create_app(settings)
    set_prefs(app.state.context.conn, settings.owner_id, guardrails=False, direct_answers=False)
    with TestClient(app) as tc:
        body = tc.get("/chat/stream?q=show+me+my+memories").text
    assert "event: memory" not in body
    assert "event: done" in body


def test_guardrails_on_never_refuses_an_app_count_question(settings, monkeypatch):
    # App-specific questions (counts, memories, albums) are NEVER off-topic, even
    # when the weak router mislabels them `none` and Guardrails is on (§10). With
    # Direct answers off, the count is answered by the agentic count tool instead.
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    fake = FakeInferenceClient(
        responses=[
            '{"tool": "none", "query": null}',                       # router mislabels it
            '{"action": "count_photos", "query": "", "grain": null}',
            '{"action": "answer", "query": null, "grain": null}',
        ],
        streams=[["You have ", "4 ", "photos."]],
    )
    monkeypatch.setattr(Settings, "build_inference_client", lambda self: (fake, "fake"))
    app = create_app(settings)
    conn = app.state.context.conn
    for pid in range(1, 5):
        add_photo(conn, photo_id=pid, content_hash=f"h{pid}", thumb_key=f"{pid}.jpg")
    set_prefs(conn, settings.owner_id, guardrails=True, direct_answers=False)
    with TestClient(app) as tc:
        body = tc.get("/chat/stream?q=how+many+photos+do+I+have").text
    assert "only answer questions about your photos" not in body.lower()  # NOT refused
    assert "count: 4" in fake.calls[-1][1][1]["content"]                  # real count gathered


def _chat_app(settings, monkeypatch, fake_inf):
    """An app wired to `fake_inf`, with two photos to retrieve."""
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(Settings, "build_inference_client", lambda self: (fake_inf, "fake"))
    app = create_app(settings)
    conn = app.state.context.conn
    fe = FakeEmbedder()
    for pid, word in ((1, "beach"), (2, "keyboard")):
        add_photo(conn, photo_id=pid, content_hash=word, thumb_key=f"{word}.jpg",
                  caption=f"a {word}")
        write_vector(conn, pid, fe.embed_texts([word])[0])
    return app


def test_a_general_question_never_wakes_siglip(settings, monkeypatch):
    # THE rule (§10): never load SigLIP until something establishes the question is
    # about photos. gemma is usually already resident, so a speculative SigLIP load
    # evicts it and forces a ~9 s reload (§8.1) — "how many planets are in the solar
    # system" would pay two swaps for a search it never needs.
    fake = FakeInferenceClient(
        responses=[json.dumps({"tool": "none", "query": None})],
        streams=[["Eight."]],
    )
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(Settings, "build_inference_client", lambda self: (fake, "fake"))
    app = create_app(settings)
    conn = app.state.context.conn
    fe = FakeEmbedder()
    embedded = []

    class Spy(FakeEmbedder):
        def embed_texts(self, texts):
            embedded.extend(texts)
            return super().embed_texts(texts)

    monkeypatch.setattr(Settings, "build_embedder", lambda self: (Spy(), "fake"))
    add_photo(conn, photo_id=1, content_hash="h1", thumb_key="1.jpg", caption="a beach")
    write_vector(conn, 1, fe.embed_texts(["beach"])[0])
    set_prefs(conn, 1, guardrails=False, direct_answers=True)
    with TestClient(app) as tc:
        tc.get("/chat/stream?q=how%20many%20planets%20are%20in%20the%20solar%20system")

    assert embedded == []          # SigLIP never touched


def test_guardrails_on_retrieves_before_the_model(settings, monkeypatch):
    # Guardrails establishes on-topic BY POLICY, so retrieving up front always pays
    # off: the turn becomes siglip -> gemma, one swap instead of two.
    order = []

    class Spy(FakeEmbedder):
        def embed_texts(self, texts):
            order.append("siglip")
            return super().embed_texts(texts)

    class OrderedClient(FakeInferenceClient):
        def complete(self, *a, **k):
            order.append("gemma")
            return super().complete(*a, **k)

        def stream(self, *a, **k):
            order.append("gemma")
            return super().stream(*a, **k)

    fake = OrderedClient(
        responses=[json.dumps({"tool": "search_library", "query": "beach"})],
        streams=[["A beach."]],
    )
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(Settings, "build_inference_client", lambda self: (fake, "fake"))
    monkeypatch.setattr(Settings, "build_embedder", lambda self: (Spy(), "fake"))
    app = create_app(settings)
    conn = app.state.context.conn
    add_photo(conn, photo_id=1, content_hash="h1", thumb_key="1.jpg", caption="a beach")
    write_vector(conn, 1, FakeEmbedder().embed_texts(["beach"])[0])
    set_prefs(conn, 1, guardrails=True, direct_answers=True)
    with TestClient(app) as tc:
        tc.get("/chat/stream?q=show%20me%20the%20beach")

    assert order[0] == "siglip"                 # retrieval FIRST
    assert order.count("siglip") == 1           # and only once — no re-embed


def test_guardrails_on_still_routes_and_can_refuse(settings, monkeypatch):
    # Guardrails is an explicit opt-in to a stricter gate, and its refusal depends
    # on the router's verdict — so that path deliberately keeps the extra call.
    fake = FakeInferenceClient(responses=[json.dumps({"tool": "none", "query": None})])
    app = _chat_app(settings, monkeypatch, fake)
    set_prefs(app.state.context.conn, 1, guardrails=True, direct_answers=True)
    with TestClient(app) as tc:
        body = tc.get("/chat/stream?q=what%20is%20the%20capital%20of%20France").text
    assert len(fake.complete_kwargs) == 1    # the router DID run
    assert "only answer questions about your photos" in body


def test_the_photo_context_is_bounded_to_fit_the_smallest_context(conn):
    # Regression: the merged prompt is BIGGER than the old one (longer system text,
    # a memories line, and photo context on every turn), and with 12 photos it blew
    # jetson's 2048-token window — llama-server answered 400 and the SSE stream died
    # mid-answer with "the model service could not be reached".
    from chat.context import MAX_CONTEXT_CHARS, build_context

    ids = list(range(1, 31))
    for pid in ids:
        add_photo(conn, photo_id=pid, content_hash=f"h{pid}", thumb_key=f"{pid}.jpg",
                  caption="a beach with a dog and a long descriptive caption " * 6)
    block = build_context(conn, ids)
    assert len(block) <= MAX_CONTEXT_CHARS + 200      # bounded, not all 30 blocks
    assert "[photo:1]" in block                        # best matches kept, in order
    # The model must know it saw a truncated list, not the whole library.
    assert "lower-ranked photos not shown" in block


def test_the_memories_line_shares_the_photo_budget(conn):
    # Both go in the same prompt, so the photo block must shrink to make room —
    # otherwise adding memories silently reintroduces the overflow.
    from chat.context import MAX_CONTEXT_CHARS, build_context

    ids = list(range(1, 31))
    for pid in ids:
        add_photo(conn, photo_id=pid, content_hash=f"h{pid}", thumb_key=f"{pid}.jpg",
                  caption="a beach with a dog " * 10)
    mem_line = "memories: " + "; ".join(f"Trip {i} (5 photos)" for i in range(20))
    block = build_context(conn, ids, max_chars=MAX_CONTEXT_CHARS - len(mem_line))
    assert len(block) + len(mem_line) <= MAX_CONTEXT_CHARS + 200


def test_one_oversized_photo_still_yields_a_usable_prompt(conn):
    # The budget must never produce an EMPTY context: a single photo whose caption
    # alone exceeds the cap should still be shown, not dropped into nothing.
    from chat.context import build_context

    add_photo(conn, photo_id=1, content_hash="h1", thumb_key="1.jpg",
              caption="x" * 9000)
    block = build_context(conn, [1], max_chars=100)
    assert "[photo:1]" in block


def test_a_turn_records_and_renders_its_duration_and_time(settings, monkeypatch):
    # "thought for N s" has to survive a reload, so the wait is PERSISTED with the
    # turn, not just streamed — a turn that waited 40 s because it swapped models
    # should still say so tomorrow.
    from chat.history import add_message, session_messages

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        Settings, "build_inference_client",
        lambda self: (FakeInferenceClient(streams=[["hi"]]), "fake"),
    )
    app = create_app(settings)
    conn = app.state.context.conn
    with TestClient(app) as tc:
        body = tc.get("/chat/stream?q=hello").text
    # Announced BEFORE the answer, on its own event, so the UI can print it above
    # the bubble as the first token lands — not after the whole answer has streamed.
    assert body.index("event: thinking") < body.index("data:")
    assert '"elapsed_ms"' in body

    row = conn.execute("SELECT elapsed_ms FROM chat_messages ORDER BY id DESC").fetchone()
    assert row["elapsed_ms"] is not None and row["elapsed_ms"] >= 0

    session_id = conn.execute("SELECT id FROM chat_sessions ORDER BY id DESC").fetchone()["id"]
    shown = session_messages(conn, session_id)[-1]
    assert shown["took"]                                # rendered, e.g. "0.0 s"
    assert len(shown["at"]) == 5 and ":" in shown["at"]  # HH:MM

    # A turn written before this column existed must show no duration rather than a
    # misleading "0 s".
    add_message(conn, session_id, "old", "answer", [])
    assert session_messages(conn, session_id)[-1]["took"] == ""


def test_elapsed_is_formatted_for_humans():
    from chat.history import format_elapsed

    assert format_elapsed(None) == ""
    assert format_elapsed(800) == "0.8 s"
    assert format_elapsed(12400) == "12.4 s"
    assert format_elapsed(65000) == "1m 05s"


def test_a_count_line_can_never_become_a_photo_citation(settings, monkeypatch):
    # The real bug, end to end. "find me all photos with dog" made the agentic loop
    # gather ONLY `count: 1 photo(s) matching "dog"` — a QUANTITY, no photo blocks —
    # and gemma4-E2B emitted `[photo:1]`, reading the count as an id and inventing a
    # caption for it. Photo 1 was an unrelated portrait retrieval never returned.
    # The prompt forbids this in capitals and was ignored, so it is enforced in code.
    fake = FakeInferenceClient(
        responses=[json.dumps({"action": "count_photos", "query": "dog", "grain": None}),
                   json.dumps({"action": "answer", "query": None, "grain": None})],
        streams=[["You have 1 photo of a dog: ", "[photo:1]", "."]],
    )
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(Settings, "build_inference_client", lambda self: (fake, "fake"))
    app = create_app(settings)
    conn = app.state.context.conn
    add_photo(conn, photo_id=1, content_hash="h1", thumb_key="1.jpg",
              caption="A man with facial hair looks at the camera")
    set_prefs(conn, 1, guardrails=False, direct_answers=False)
    with TestClient(app) as tc:
        body = tc.get("/chat/stream?q=find%20me%20all%20photos%20with%20dog").text

    assert "[photo:1]" not in body            # the fabrication never reaches the user
    assert "You have 1 photo of a dog" in body  # the prose itself is untouched
    # ...and it is not persisted either, so a reload cannot resurrect it.
    row = conn.execute("SELECT answer, sources FROM chat_messages ORDER BY id DESC").fetchone()
    assert "[photo:1]" not in row["answer"]
    assert json.loads(row["sources"]) == []


def test_the_recorded_wait_excludes_streaming_time(settings, monkeypatch):
    # "thought for" is time-to-first-token, NOT the whole turn: once tokens are
    # arriving the user is reading, not waiting, so a long answer must not read as
    # a slow one. A stream that dawdles between chunks must not inflate it.
    import time as _time

    settings.data_dir.mkdir(parents=True, exist_ok=True)

    class SlowStream(FakeInferenceClient):
        def stream(self, *args, **kwargs):
            for chunk in ("hello ", "world"):
                _time.sleep(0.05)   # 100 ms of pure streaming, after the first token
                yield chunk

    monkeypatch.setattr(
        Settings, "build_inference_client", lambda self: (SlowStream(), "fake"),
    )
    app = create_app(settings)
    conn = app.state.context.conn
    with TestClient(app) as tc:
        tc.get("/chat/stream?q=hello")

    row = conn.execute("SELECT elapsed_ms FROM chat_messages ORDER BY id DESC").fetchone()
    # The second chunk's 50 ms sleep lands after the wait has already been stamped.
    assert row["elapsed_ms"] < 100
