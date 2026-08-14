import pytest
from fastapi.testclient import TestClient

from config import Settings
from embedding.fakes import FakeEmbedder
from embedding.store import write_vector
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
