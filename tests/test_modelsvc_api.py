"""The `models` service HTTP surface (design §5.1) over `FakeBackend` — no
torch, no transformers, no Ollama. Task 1 of the models-service refactor."""

import base64
import json

import pytest
from fastapi.testclient import TestClient

from modelsvc.app import create_models_app
from modelsvc.backends.fake import FAKE_CONTEXT, FakeBackend

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


def test_embed_spec_returns_fake_backend_calibration_and_preprocessing():
    with _client() as tc:
        resp = tc.get("/embed/spec")
    assert resp.status_code == 200
    assert resp.json() == {
        "logit_scale": 10.0,
        "logit_bias": -5.0,
        "preprocess": {"input_px": 384, "resample": "bilinear", "mode": "squash"},
        "generation": 0,
    }


def test_resources_reports_resident_models_and_nothing_else():
    # §5.1: this endpoint carries no machine metrics. They are host-wide sysfs and
    # psutil reads needing no CUDA context, so `app` takes them directly and the
    # resource bar keeps showing them while this service is down or restarting.
    with _client() as tc:
        resp = tc.get("/resources")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data["resident"], list)
    # No machine metrics here — they are host-wide reads `app` does itself (§5.1).
    assert not {"ram_used_mb", "ram_total_mb", "cpu_pct", "gpu_pct"} & data.keys()
    # It also reports the live slot selection, which only this service knows (§4.1).
    assert set(data) == {"resident", "active", "slots", "generation"}


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


def test_speak_returns_base64_wav_and_the_model_name():
    with _client() as tc:
        resp = tc.post("/speak", json={"text": "Beach day."})
    assert resp.status_code == 200
    body = resp.json()
    assert base64.b64decode(body["audio"])[:4] == b"RIFF"
    assert body["sample_rate"] == 24000
    assert body["model"] == "fake-voice"


def test_music_honours_the_requested_length():
    from storage.wav import duration_seconds

    with _client() as tc:
        resp = tc.post("/music", json={"prompt": "warm guitar", "seconds": 3})
    body = resp.json()
    assert duration_seconds(base64.b64decode(body["audio"])) == pytest.approx(3.0, abs=0.1)
    assert body["sample_rate"] == 44100


def test_music_is_clamped_to_the_models_ceiling():
    from storage.wav import duration_seconds

    with _client() as tc:
        resp = tc.post("/music", json={"prompt": "warm guitar", "seconds": 600})
    assert duration_seconds(base64.b64decode(resp.json()["audio"])) <= 47.0



def test_speak_defaults_to_english():
    with _client() as tc:
        assert tc.post("/speak", json={"text": "hello"}).status_code == 200


def test_the_language_picks_the_voice_without_the_user_choosing():
    # Kokoro cannot speak Russian, but asking for Russian must JUST WORK: the service
    # swaps the voice unit to one that can (§11.1). Making the user pick a model in
    # settings for a language is not a design, it is a bug.
    with _client() as tc:
        assert tc.post("/speak", json={"text": "привет", "language": "ru"}).status_code == 200
        assert tc.post("/speak", json={"text": "hello", "language": "en"}).status_code == 200




def test_music_always_has_a_model_to_use():
    # "No music" was an option; it is not any more (§11.1), so /music never 503s for
    # want of a model.
    with _client() as tc:
        assert tc.post("/music", json={"prompt": "x", "seconds": 3}).status_code == 200


def test_speak_always_has_a_voice_to_use():
    # "No narration" was an option; it is not any more (§11.1). Both languages just
    # work, because the language picks the model.
    with _client() as tc:
        assert tc.post("/speak", json={"text": "hi", "language": "en"}).status_code == 200
        assert tc.post("/speak", json={"text": "привет", "language": "ru"}).status_code == 200


def test_text_context_reports_what_the_server_accepts():
    # The story pass budgets its prompt against this (§11.1), so it is served.
    with _client() as tc:
        resp = tc.get("/text/context")
    assert resp.status_code == 200
    assert resp.json()["context"] == FAKE_CONTEXT


# --- the score (§11.1) --------------------------------------------------------------


def test_music_accepts_a_whole_score_and_says_where_each_section_landed():
    with _client() as tc:
        resp = tc.post("/music", json={"sections": [{"brief": "solo piano", "seconds": 4},
                                                    {"brief": "brass", "seconds": 6}]})
    assert resp.status_code == 200
    segments = resp.json()["segments"]
    assert len(segments) == 2
    assert segments[0]["end"] == segments[1]["start"]      # one piece, not two clips


def test_music_still_takes_a_single_prompt():
    with _client() as tc:
        resp = tc.post("/music", json={"prompt": "warm ambient", "seconds": 4})
    assert resp.status_code == 200
    assert len(resp.json()["segments"]) == 1


def test_music_with_neither_a_prompt_nor_sections_is_refused():
    with _client() as tc:
        assert tc.post("/music", json={"seconds": 4}).status_code == 422
