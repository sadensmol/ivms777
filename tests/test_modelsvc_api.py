"""The `models` service HTTP surface (design §5.1) over `FakeBackend` — no
torch, no transformers, no Ollama. Task 1 of the models-service refactor."""

import base64
import json

from fastapi.testclient import TestClient

from modelsvc.app import create_models_app
from modelsvc.backends.fake import FakeBackend

_IMAGE_B64 = base64.b64encode(b"not-really-a-jpeg").decode()


def _client() -> TestClient:
    return TestClient(create_models_app(FakeBackend()))


def test_embed_image_returns_one_vector_per_image():
    with _client() as tc:
        resp = tc.post("/embed/image", json={"images": [_IMAGE_B64, _IMAGE_B64]})
    assert resp.status_code == 200
    vectors = resp.json()["vectors"]
    assert len(vectors) == 2
    assert vectors[0] == vectors[1]  # identical bytes -> identical vector
    assert all(isinstance(v, float) for v in vectors[0])


def test_embed_text_returns_a_vector():
    with _client() as tc:
        resp = tc.post("/embed/text", json={"texts": ["cat"]})
    assert resp.status_code == 200
    vectors = resp.json()["vectors"]
    assert len(vectors) == 1
    assert len(vectors[0]) > 0


def test_embed_text_is_deterministic():
    with _client() as tc:
        first = tc.post("/embed/text", json={"texts": ["cat"]}).json()["vectors"]
        second = tc.post("/embed/text", json={"texts": ["cat"]}).json()["vectors"]
    assert first == second


def test_tag_returns_labels_per_dimension():
    with _client() as tc:
        resp = tc.post(
            "/tag", json={"image": _IMAGE_B64, "dimensions": ["scene", "subject"]}
        )
    assert resp.status_code == 200
    data = resp.json()
    assert set(data.keys()) == {"scene", "subject"}
    assert all(isinstance(labels, list) and labels for labels in data.values())


def test_caption_returns_the_expected_shape():
    with _client() as tc:
        resp = tc.post("/caption", json={"image": _IMAGE_B64})
    assert resp.status_code == 200
    data = resp.json()
    # No tags — the caption model returns only the caption sentence (design §7).
    assert set(data.keys()) == {"caption", "title", "description", "model"}
    assert isinstance(data["caption"], str) and data["caption"]
    # The ACTUAL model the backend used (may differ from settings.caption_model).
    assert data["model"] == "fake"


def test_text_complete_returns_text():
    with _client() as tc:
        resp = tc.post(
            "/text/complete",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert resp.status_code == 200
    assert resp.json()["text"]


def test_text_embed_returns_a_vector():
    with _client() as tc:
        resp = tc.post("/text/embed", json={"model": "m", "texts": ["a photo of a cat"]})
    assert resp.status_code == 200
    assert len(resp.json()["vectors"]) == 1


def test_text_warm_returns_empty_body():
    with _client() as tc:
        resp = tc.post("/text/warm", json={"model": "m"})
    assert resp.status_code == 200
    assert resp.json() == {}


def test_text_evict_returns_empty_body():
    with _client() as tc:
        resp = tc.post("/text/evict", json={"model": "m"})
    assert resp.status_code == 200
    assert resp.json() == {}


def test_calibration_returns_fake_backend_values():
    with _client() as tc:
        resp = tc.get("/embed/calibration")
    assert resp.status_code == 200
    assert resp.json() == {"logit_scale": 10.0, "logit_bias": -5.0}


def test_resources_reports_resident_models_and_nothing_else():
    # §5.1: this endpoint carries no machine metrics. They are host-wide sysfs and
    # psutil reads needing no CUDA context, so `app` takes them directly and the
    # resource bar keeps showing them while this service is down or restarting.
    with _client() as tc:
        resp = tc.get("/resources")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data["resident"], list)
    assert set(data) == {"resident", "active"}


def test_text_stream_streams_tokens_as_sse():
    with _client() as tc, tc.stream(
        "POST",
        "/text/stream",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
    ) as resp:
        assert resp.status_code == 200
        lines = [line for line in resp.iter_lines() if line]

    tokens = []
    for line in lines:
        assert line.startswith("data:")
        data = line[len("data:") :].strip()
        if data == "[DONE]":
            continue
        tokens.append(json.loads(data)["token"])

    assert tokens == ["Hello", " from", " the fake backend."]
    assert lines[-1] == "data: [DONE]"
