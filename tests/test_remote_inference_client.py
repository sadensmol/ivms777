"""`RemoteInferenceClient` — the `inference.client.InferenceClient` protocol
implemented over `ModelsClient` (design §5.1, plan 15 task 4). Round-tripped
against `create_models_app` + `FakeBackend`, exactly like
`tests/test_remote_embedder.py` (task 2's mirror), so it never touches a real
Ollama.
"""

from fastapi.testclient import TestClient

from inference.models_client import ModelsClient
from inference.remote_inference_client import RemoteInferenceClient
from modelsvc.app import create_models_app
from modelsvc.backends.fake import FakeBackend


def _client() -> ModelsClient:
    app = create_models_app(FakeBackend())
    # TestClient wraps a sync-capable ASGI transport; reuse it so ModelsClient's
    # plain httpx.Client can call the app in-process, no real network.
    transport = TestClient(app)._transport
    return ModelsClient("http://modelsvc", transport=transport)


def test_complete_returns_the_fake_backend_string():
    ric = RemoteInferenceClient(_client())
    result = ric.complete("m", [{"role": "user", "content": "hi"}])
    assert isinstance(result, str) and result


def test_stream_yields_the_fake_backend_tokens_one_at_a_time():
    ric = RemoteInferenceClient(_client())
    tokens = list(ric.stream("m", [{"role": "user", "content": "hi"}]))
    assert tokens == ["Hello", " from", " the fake backend."]


def test_embed_returns_a_vector_per_text():
    ric = RemoteInferenceClient(_client())
    vectors = ric.embed("m", ["cat"])
    assert len(vectors) == 1
    assert len(vectors[0]) > 0


def test_warm_is_a_no_op():
    ric = RemoteInferenceClient(_client())
    assert ric.warm("m") is None  # no raise; residency is the service's concern


def test_evict_is_a_no_op():
    ric = RemoteInferenceClient(_client())
    assert ric.evict("m") is None
