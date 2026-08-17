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
    result = _client().caption(b"image-bytes")
    assert set(result.keys()) == {"caption", "title", "description", "model"}


def test_caption_raises_for_error_statuses():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"detail": "boom"})

    client = ModelsClient("http://modelsvc", transport=httpx.MockTransport(handler))

    with pytest.raises(httpx.HTTPStatusError):
        client.caption(b"image-bytes")


def test_text_complete_round_trips():
    assert _client().text_complete("m", [{"role": "user", "content": "hi"}])


def test_text_complete_forwards_temperature_and_max_tokens():
    # The decode controls must survive the app -> models-service hop. They used to be
    # dropped here, so a routing call sampled at the backend default with no cap and
    # could run to the whole KV context and return nothing.
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json

        seen.update(_json.loads(request.content))
        return httpx.Response(200, json={"text": "ok"})

    client = ModelsClient("http://modelsvc", transport=httpx.MockTransport(handler))
    client.text_complete("m", [{"role": "user", "content": "hi"}], temperature=0, max_tokens=128)

    assert seen["temperature"] == 0
    assert seen["max_tokens"] == 128


def test_text_complete_omits_decode_controls_when_unset():
    # Absent means "backend default" — never a hardcoded 0/None in the payload.
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json

        seen.update(_json.loads(request.content))
        return httpx.Response(200, json={"text": "ok"})

    client = ModelsClient("http://modelsvc", transport=httpx.MockTransport(handler))
    client.text_complete("m", [{"role": "user", "content": "hi"}])

    assert "temperature" not in seen
    assert "max_tokens" not in seen


def test_text_embed_round_trips():
    vectors = _client().text_embed("m", ["a photo of a cat"])
    assert len(vectors) == 1


def test_text_warm_round_trips():
    assert _client().text_warm("m") is None


def test_text_evict_round_trips():
    assert _client().text_evict("m") is None


def test_resources_round_trips_models_only_never_machine_metrics():
    # The service reports ONLY what it alone knows (§5.1): resident models and the
    # in-flight op. RAM/CPU/GPU/temperature are host reads `app` does itself, so the
    # bar survives this service being down — they must not reappear here.
    data = _client().resources()
    assert isinstance(data["resident"], list)
    assert "active" in data
    assert not {"ram_used_mb", "ram_total_mb", "cpu_pct", "gpu_pct", "cpu_c"} & data.keys()


def test_text_stream_yields_tokens_in_order():
    tokens = list(_client().text_stream("m", [{"role": "user", "content": "hi"}]))
    assert tokens == ["Hello", " from", " the fake backend."]
