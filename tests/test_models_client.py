"""`ModelsClient` round-tripped against the `models` service app (design §5.1)
via the FastAPI `TestClient`'s own sync-capable ASGI transport — no real
network, no real models."""

import httpx
import pytest
from fastapi.testclient import TestClient

from inference.models_client import ModelsClient
from modelsvc.app import create_models_app
from modelsvc.backends.fake import FakeBackend


def _client() -> ModelsClient:
    app = create_models_app(FakeBackend())
    # TestClient wraps a sync-capable ASGI transport (`httpx.ASGITransport` is
    # async-only); reuse it so `ModelsClient`'s plain `httpx.Client` can call the
    # app in-process, no real network.
    transport = TestClient(app)._transport
    return ModelsClient("http://modelsvc", transport=transport)


def test_embed_text_round_trips():
    vectors = _client().embed_text(["cat", "dog"])
    assert len(vectors) == 2
    assert all(isinstance(v, list) for v in vectors)


def test_embed_image_round_trips():
    vectors = _client().embed_image([b"one", b"two"])
    assert len(vectors) == 2
    assert vectors[0] != vectors[1]


def test_tag_round_trips():
    result = _client().tag(b"image-bytes", ["scene", "subject"])
    assert set(result.keys()) == {"scene", "subject"}


def test_caption_round_trips():
    result = _client().caption(b"image-bytes", ["scene"])
    assert set(result.keys()) == {"caption", "title", "description", "tags", "model"}


def test_caption_raises_for_error_statuses():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"detail": "boom"})

    client = ModelsClient("http://modelsvc", transport=httpx.MockTransport(handler))

    with pytest.raises(httpx.HTTPStatusError):
        client.caption(b"image-bytes", ["scene"])


def test_text_complete_round_trips():
    assert _client().text_complete("m", [{"role": "user", "content": "hi"}])


def test_text_embed_round_trips():
    vectors = _client().text_embed("m", ["a photo of a cat"])
    assert len(vectors) == 1


def test_text_warm_round_trips():
    assert _client().text_warm("m") is None


def test_text_evict_round_trips():
    assert _client().text_evict("m") is None


def test_resources_round_trips():
    data = _client().resources()
    assert data["ram_total_mb"] > 0
    assert isinstance(data["resident"], list)


def test_text_stream_yields_tokens_in_order():
    tokens = list(_client().text_stream("m", [{"role": "user", "content": "hi"}]))
    assert tokens == ["Hello", " from", " the fake backend."]
