"""The `models` service HTTP surface (design §5.1) over `FakeBackend` — no
torch, no transformers, no Ollama. Task 1 of the models-service refactor."""

import base64
import hashlib
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from modelsvc.app import create_models_app
from modelsvc.backends.base import CloningUnavailable
from modelsvc.backends.fake import FAKE_BAKE_ADAPTER, FAKE_CONTEXT, FakeBackend
from storage.wav import encode_wav

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


_CLIP = b"RIFF-a-saved-example"


def _reference(audio: bytes = _CLIP, **overrides) -> dict:
    ref = {"audio": base64.b64encode(audio).decode(),
           "sha256": hashlib.sha256(audio).hexdigest(), "language": "en"}
    return ref | overrides


class _Recording(FakeBackend):
    """Records exactly how the route called `speak`."""

    def __init__(self):
        super().__init__()
        self.calls: list[tuple[tuple, dict]] = []

    def speak(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return super().speak(*args, **kwargs)


def _recording_client():
    backend = _Recording()
    return backend, TestClient(create_models_app(backend))


def test_speak_without_references_calls_the_backend_exactly_as_before():
    backend, tc = _recording_client()
    with tc:
        resp = tc.post("/speak", json={"text": "hi", "language": "en", "voice": "af_heart",
                                       "segments": ["hi"], "voices": [None]})
    assert resp.status_code == 200
    assert backend.calls == [(("hi", "en", "af_heart", ["hi"], [None]), {})]
    body = resp.json()
    assert body["revision"] is None and body["conditioning"] == {}


def test_speak_decodes_references_to_bytes_before_the_backend():
    backend, tc = _recording_client()
    stored = base64.b64encode(b"old-blob").decode()
    ref = _reference(transcript="the prompt",
                     conditioning=[{"model": "voxcpm2", "revision": "abc", "blob": stored}])
    with tc:
        resp = tc.post("/speak", json={"text": "hi", "voice": "saved:1",
                                       "references": {"saved:1": ref}})
    assert resp.status_code == 200
    (_, kwargs), = backend.calls
    assert kwargs["references"] == {"saved:1": {
        "audio": _CLIP, "sha256": hashlib.sha256(_CLIP).hexdigest(), "language": "en",
        "transcript": "the prompt",
        "conditioning": [{"model": "voxcpm2", "revision": "abc", "blob": b"old-blob"}],
    }}


def test_speak_reply_carries_the_revision_and_base64_blobs():
    class Cloning(FakeBackend):
        def speak(self, *args, references=None):
            return {"audio": b"RIFF", "sample_rate": 48000, "model": "voxcpm2",
                    "revision": "abc", "segments": [],
                    "conditioning": {
                        "saved:1": {"state": "built", "model": "voxcpm2", "revision": "abc",
                                    "reason": "no stored conditioning", "blob": b"\x00new"},
                        "saved:2": {"state": "restored", "model": "voxcpm2",
                                    "revision": "abc", "reason": None, "blob": None},
                    }}

    refs = {"saved:1": _reference(), "saved:2": _reference(b"other")}
    with TestClient(create_models_app(Cloning())) as tc:
        resp = tc.post("/speak", json={"text": "hi", "voice": "saved:1",
                                       "voices": ["saved:2"], "references": refs})
    assert resp.status_code == 200
    body = resp.json()
    assert body["revision"] == "abc"
    assert base64.b64decode(body["conditioning"]["saved:1"]["blob"]) == b"\x00new"
    assert body["conditioning"]["saved:1"]["state"] == "built"
    assert body["conditioning"]["saved:2"] == {"state": "restored", "model": "voxcpm2",
                                              "revision": "abc", "reason": None, "blob": None}


@pytest.mark.parametrize(("request_body", "reason"), [
    ({"voice": "saved:1", "references": {"saved:1": _reference(sha256="0" * 64)}},
     "does not match its sha256"),
    ({"voice": "saved:9", "references": {"saved:1": _reference()}},
     "'saved:9' must be one of the references"),
    ({"voice": "saved:1", "voices": ["af_heart"], "references": {"saved:1": _reference()}},
     "never mixes saved voices and preset speakers"),
    ({"references": {"saved:1": _reference()}}, "None must be one of the references"),
    ({"voice": "", "references": {"saved:1": _reference()}}, "'' must be one of the references"),
    ({"voice": "saved:1", "references": {f"saved:{i}": _reference() for i in range(1, 10)}},
     "at most 8 references"),
    ({"voice": "saved:1", "references": {"saved:1": _reference() | {"audio": "not base64!"}}},
     "not valid base64"),
    ({"voice": "saved:1", "references": {"saved:1": _reference(b"x" * (4 * 1024 * 1024 + 1))}},
     "the limit is 4194304"),
    ({"voice": "saved:1", "references": {"saved:1": _reference(
        conditioning=[{"model": "voxcpm2", "revision": "abc", "blob": "%%%"}])}},
     "conditioning blob of 'saved:1' is not valid base64"),
])
def test_a_malformed_saved_voice_request_is_a_422_with_the_reason(request_body, reason):
    backend, tc = _recording_client()
    with tc:
        resp = tc.post("/speak", json={"text": "hi"} | request_body)
    assert resp.status_code == 422
    assert reason in resp.json()["detail"]
    assert backend.calls == []


def test_eight_references_are_accepted():
    refs = {f"saved:{i}": _reference() for i in range(1, 9)}
    with _client() as tc:
        resp = tc.post("/speak", json={"text": "hi", "voice": "saved:1", "references": refs})
    assert resp.status_code == 200


def test_no_voice_copy_on_the_profile_is_a_503_with_the_reason():
    class Unavailable(FakeBackend):
        def speak(self, *args, references=None):
            raise CloningUnavailable("no voice-copy model speaks en on jetson")

    with TestClient(create_models_app(Unavailable())) as tc:
        resp = tc.post("/speak", json={"text": "hi", "voice": "saved:1",
                                       "references": {"saved:1": _reference()}})
    assert resp.status_code == 503
    assert resp.json()["detail"] == ("voice copy is not available: "
                                     "no voice-copy model speaks en on jetson")


def test_the_fake_refuses_voice_copy_where_the_profile_has_none():
    with TestClient(create_models_app(FakeBackend("jetson"))) as tc:
        resp = tc.post("/speak", json={"text": "hi", "voice": "saved:1",
                                       "references": {"saved:1": _reference()}})
    assert resp.status_code == 503
    assert resp.json()["detail"].startswith("voice copy is not available:")


def test_the_fake_speaks_a_saved_voice_and_builds_one_conditioning_per_token():
    refs = {"saved:1": _reference(), "saved:2": _reference(b"other")}
    with _client() as tc:
        resp = tc.post("/speak", json={"text": "hi", "voice": "saved:1",
                                       "voices": ["saved:2"], "references": refs})
    body = resp.json()
    assert base64.b64decode(body["audio"])[:4] == b"RIFF"
    assert body["model"] == "fake-voice" and body["revision"] is None
    assert set(body["conditioning"]) == {"saved:1", "saved:2"}
    assert all(c["state"] == "built" and c["blob"] for c in body["conditioning"].values())


_ADAPTER = b"safetensors-of-a-baked-voice"


def _adapter(blob: bytes = _ADAPTER, **overrides) -> dict:
    return {"blob": base64.b64encode(blob).decode(),
            "sha256": hashlib.sha256(blob).hexdigest()} | overrides


def test_speak_passes_the_decoded_adapter_and_use_clip_to_the_backend():
    backend, tc = _recording_client()
    with tc:
        resp = tc.post("/speak", json={"text": "hi", "voice": "saved:1", "references": {
            "saved:1": _reference(adapter=_adapter(), use_clip=False)}})
    assert resp.status_code == 200
    (_, kwargs), = backend.calls
    ref = kwargs["references"]["saved:1"]
    assert ref["adapter"] == {"blob": _ADAPTER, "sha256": hashlib.sha256(_ADAPTER).hexdigest()}
    assert ref["use_clip"] is False


def test_the_reply_carries_the_adapter_states():
    states = {"saved:1": {"state": "loaded", "sha256": "ab"}}

    class Baked(FakeBackend):
        def speak(self, *args, references=None):
            return {"audio": b"RIFF", "sample_rate": 48000, "model": "voxcpm2",
                    "revision": "abc", "segments": [], "conditioning": {}, "adapters": states}

    with TestClient(create_models_app(Baked())) as tc:
        resp = tc.post("/speak", json={"text": "hi", "voice": "saved:1", "references": {
            "saved:1": _reference(adapter=_adapter())}})
    assert resp.json()["adapters"] == states


def _model_only(**overrides) -> dict:
    return {"language": "ru", "adapter": _adapter(), "use_clip": False} | overrides


_CONTROLS = {"cfg_value": 1.5, "inference_timesteps": 20}


def test_a_plain_reference_reaches_the_backend_with_no_controls_key():
    backend, tc = _recording_client()
    with tc:
        resp = tc.post("/speak", json={"text": "hi", "voice": "saved:1", "references": {
            "saved:1": _reference(), "saved:2": _model_only()}})
    assert resp.status_code == 200
    (_, kwargs), = backend.calls
    assert all("controls" not in ref for ref in kwargs["references"].values())
    assert resp.json()["controls"] == {}


@pytest.mark.parametrize("reference", [
    pytest.param(_reference(controls=_CONTROLS), id="with a take"),
    pytest.param(_model_only(controls=_CONTROLS), id="the adapter alone"),
])
def test_controls_reach_the_backend_on_both_kinds_of_reference(reference):
    backend, tc = _recording_client()
    with tc:
        resp = tc.post("/speak", json={"text": "hi", "voice": "saved:1",
                                       "references": {"saved:1": reference}})
    assert resp.status_code == 200
    (_, kwargs), = backend.calls
    assert kwargs["references"]["saved:1"]["controls"] == _CONTROLS


@pytest.mark.parametrize("controls", [
    {"cfg_value": 1.5, "inference_timesteps": 10},
    {"cfg_value": 3.0, "inference_timesteps": 20},
])
def test_the_ends_of_the_control_ranges_are_accepted(controls):
    with _client() as tc:
        resp = tc.post("/speak", json={"text": "hi", "voice": "saved:1", "references": {
            "saved:1": _reference(controls=controls)}})
    assert resp.status_code == 200


@pytest.mark.parametrize("controls", [
    {"cfg_value": 1.49, "inference_timesteps": 10},
    {"cfg_value": 3.01, "inference_timesteps": 10},
    {"cfg_value": "2.0", "inference_timesteps": 10},
    {"cfg_value": True, "inference_timesteps": 10},
    {"cfg_value": 2.0, "inference_timesteps": 9},
    {"cfg_value": 2.0, "inference_timesteps": 21},
    {"cfg_value": 2.0, "inference_timesteps": 10.0},
    {"cfg_value": 2.0, "inference_timesteps": "10"},
    {"cfg_value": 2.0, "inference_timesteps": True},
    {"cfg_value": 2.0},
    {"inference_timesteps": 10},
    {"cfg_value": 2.0, "inference_timesteps": 10, "max_len": 64},
])
def test_controls_outside_their_contract_are_a_422_and_reach_no_backend(controls):
    backend, tc = _recording_client()
    with tc:
        resp = tc.post("/speak", json={"text": "hi", "voice": "saved:1", "references": {
            "saved:1": _reference(controls=controls)}})
    assert resp.status_code == 422
    assert backend.calls == []


def test_the_reply_carries_the_echo_of_the_controls():
    echo = {"saved:1": _CONTROLS}

    class Echoing(FakeBackend):
        def speak(self, *args, references=None):
            return {"audio": b"RIFF", "sample_rate": 48000, "model": "voxcpm2",
                    "revision": "abc", "segments": [], "controls": echo}

    with TestClient(create_models_app(Echoing())) as tc:
        resp = tc.post("/speak", json={"text": "hi", "voice": "saved:1", "references": {
            "saved:1": _reference(controls=_CONTROLS)}})
    assert resp.json()["controls"] == echo


@pytest.mark.parametrize(("references", "reason"), [
    ({"saved:1": _reference(adapter=_adapter() | {"blob": "not base64!"})},
     "the adapter of 'saved:1' is not valid base64"),
    ({"saved:1": _reference(adapter=_adapter(sha256="0" * 64))},
     "the adapter of 'saved:1' does not match its sha256"),
    ({f"saved:{i}": _reference(adapter=_adapter()) for i in range(1, 6)},
     "at most 4 references with an adapter, got 5"),
    ({"saved:1": _reference(adapter=_adapter(b"x" * (65 * 1024 * 1024)))},
     "the limit is 67108864"),
    ({"saved:1": _reference(use_clip=False)}, "use_clip=false needs an adapter"),
])
def test_a_malformed_adapter_is_a_422_with_the_reason(references, reason):
    backend, tc = _recording_client()
    with tc:
        resp = tc.post("/speak", json={"text": "hi", "voice": "saved:1",
                                       "references": references})
    assert resp.status_code == 422
    assert reason in resp.json()["detail"]
    assert backend.calls == []


def test_a_model_only_reference_with_no_audio_reaches_the_backend_without_audio():
    backend, tc = _recording_client()
    with tc:
        resp = tc.post("/speak", json={"text": "hi", "language": "ru", "voice": "saved:1",
                                       "references": {"saved:1": _model_only()}})
    assert resp.status_code == 200
    (_, kwargs), = backend.calls
    assert kwargs["references"] == {"saved:1": {
        "language": "ru", "use_clip": False,
        "adapter": {"blob": _ADAPTER, "sha256": hashlib.sha256(_ADAPTER).hexdigest()},
    }}
    body = resp.json()
    assert body["conditioning"] == {}
    assert body["adapters"] == {"saved:1": {"state": "loaded",
                                            "sha256": hashlib.sha256(_ADAPTER).hexdigest()}}


@pytest.mark.parametrize("reference", [
    {"language": "en"},
    {"language": "en", "use_clip": True},
    {"language": "en", "adapter": _adapter()},
    {"language": "en", "adapter": _adapter(), "use_clip": True},
    _reference() | {"sha256": None},
    {"language": "en", "sha256": hashlib.sha256(_CLIP).hexdigest()},
])
def test_a_reference_that_uses_its_clip_needs_its_audio(reference):
    backend, tc = _recording_client()
    with tc:
        resp = tc.post("/speak", json={"text": "hi", "voice": "saved:1",
                                       "references": {"saved:1": reference}})
    assert resp.status_code == 422
    assert resp.json()["detail"] == "the reference of 'saved:1' needs its audio"
    assert backend.calls == []


@pytest.mark.parametrize(("reference", "reason"), [
    (_reference(sha256="0" * 64, adapter=_adapter(), use_clip=False),
     "the audio of 'saved:1' does not match its sha256"),
    (_reference(sha256=None, adapter=_adapter(), use_clip=False),
     "the reference of 'saved:1' needs its audio"),
    (_model_only(adapter=None), "use_clip=false needs an adapter"),
    (_model_only(adapter=_adapter(sha256="0" * 64)),
     "the adapter of 'saved:1' does not match its sha256"),
])
def test_audio_sent_with_use_clip_false_is_still_checked(reference, reason):
    backend, tc = _recording_client()
    with tc:
        resp = tc.post("/speak", json={"text": "hi", "voice": "saved:1",
                                       "references": {"saved:1": reference}})
    assert resp.status_code == 422
    assert reason in resp.json()["detail"]
    assert backend.calls == []


class _Calls:
    """Records every scheduler and registry call."""

    def __init__(self):
        self.calls = []

    def run(self, needed, priority, fn):
        self.calls.append(("run", tuple(needed)))
        return fn()

    def ensure(self, name):
        self.calls.append(("ensure", name))

    def unload(self, name):
        self.calls.append(("unload", name))

    def resident(self):
        return []


def _bakes(**deps):
    from modelsvc.activity import Activity
    from modelsvc.voice_bakes import VoiceBakes

    return VoiceBakes(**{"slots": None, "registry": None, "governor": None, "scheduler": None,
                         "activity": Activity(), "bake_root": "unused", "profile": "mac"}
                      | deps)


def _composite(**parts):
    from modelsvc.backends.composite import CompositeBackend

    return CompositeBackend(embed=None, **parts)


def test_a_closed_gate_answers_409_to_a_manual_voice_ensure_and_loads_nothing():
    bakes, calls = _bakes(), _Calls()
    bakes.gate.close()
    for backend in (_composite(scheduler=calls, bakes=bakes),
                    _composite(registry=calls, bakes=bakes)):
        with TestClient(create_models_app(backend)) as tc:
            resp = tc.post("/models/voice/ensure")
        assert resp.status_code == 409
        assert resp.json()["detail"].startswith("a voice is being baked (0%)")
    assert calls.calls == []


def test_an_open_gate_lets_a_manual_voice_ensure_load():
    calls = _Calls()
    with TestClient(create_models_app(_composite(scheduler=calls, bakes=_bakes()))) as tc:
        assert tc.post("/models/voice/ensure").status_code == 200
    assert calls.calls == [("run", ("voice",))]


class _ActiveBakes:
    """A bake mid-run: `active()` is True and the gate is open or closed."""

    def __init__(self, *, gate_closed: bool):
        from modelsvc.voice_bakes import VoiceGate

        self.gate = VoiceGate()
        if gate_closed:
            self.gate.close()

    def active(self) -> bool:
        return True


@pytest.mark.parametrize("bakes", [
    pytest.param(lambda: None, id="no-bakes"),
    pytest.param(_bakes, id="inactive"),
    pytest.param(lambda: _ActiveBakes(gate_closed=False), id="active-gate-open"),
    pytest.param(lambda: _ActiveBakes(gate_closed=True), id="active-gate-closed"),
])
@pytest.mark.parametrize("dependency", ["scheduler", "registry"])
def test_voice_train_ensure_is_always_a_409(bakes, dependency):
    calls = _Calls()
    backend = _composite(bakes=bakes(), **{dependency: calls})
    with TestClient(create_models_app(backend)) as tc:
        resp = tc.post("/models/voice_train/ensure")
    assert resp.status_code == 409
    assert resp.json()["detail"] == "voice_train is loaded only by a bake"
    assert calls.calls == []


def test_speak_during_a_bake_is_a_409_with_the_percent():
    class _Voice:
        def speak(self, *args):
            raise AssertionError("a closed gate reaches no voice")

    bakes = _bakes()
    bakes.gate.close()
    with TestClient(create_models_app(_composite(voice=_Voice(), bakes=bakes))) as tc:
        resp = tc.post("/speak", json={"text": "hi"})
    assert resp.status_code == 409
    assert resp.json()["detail"] == ("a voice is being baked (0%); "
                                     "try again when it ends, or cancel it")


def test_unloading_the_leased_training_unit_is_a_409():
    from modelsvc.governor import MemoryGovernor
    from modelsvc.registry import ModelRegistry, ModelSpec

    freed = []
    registry = ModelRegistry()
    registry.register(ModelSpec("voice_train", lambda: None,
                                lambda: freed.append("voice_train"), 100))
    governor = MemoryGovernor(registry, measure_free_mb=lambda: 9999.0, budget_mb=9999)
    governor.acquire(["voice_train"])
    backend = _composite(registry=registry, governor=governor)
    with TestClient(create_models_app(backend)) as tc, governor.lease("voice_train"):
        resp = tc.post("/models/voice_train/unload")
    assert resp.status_code == 409
    assert freed == []


_BAKE = {"clips": [{"audio": base64.b64encode(b"RIFF").decode(), "sha256": "ab",
                    "text": "Hello.", "language": "en"}], "tag": {"check": "me"}}


def test_a_bake_walks_its_routes_from_start_to_delete():
    with _client() as tc:
        assert tc.get("/voice/bakes/current").status_code == 404
        started = tc.post("/voice/bakes", json=_BAKE)
        assert started.status_code == 202
        bake_id = started.json()["id"]
        assert started.json()["state"] == "running"

        assert tc.get("/voice/bakes/current").json()["id"] == bake_id
        assert tc.post("/voice/bakes", json=_BAKE).status_code == 409
        assert tc.get(f"/voice/bakes/{bake_id}/adapter").status_code == 409
        assert tc.delete(f"/voice/bakes/{bake_id}").status_code == 409

        states = [tc.get(f"/voice/bakes/{bake_id}").json()["state"] for _ in range(3)]
        assert states == ["running", "running", "done"]
        adapter = tc.get(f"/voice/bakes/{bake_id}/adapter")
        assert adapter.status_code == 200
        assert adapter.headers["content-type"] == "application/octet-stream"
        assert adapter.content == FAKE_BAKE_ADAPTER

        assert tc.delete(f"/voice/bakes/{bake_id}").status_code == 204
        assert tc.get(f"/voice/bakes/{bake_id}").status_code == 404
        assert tc.get("/voice/bakes/current").status_code == 404


def test_cancel_answers_the_status():
    with _client() as tc:
        bake_id = tc.post("/voice/bakes", json=_BAKE).json()["id"]
        resp = tc.post(f"/voice/bakes/{bake_id}/cancel")
    assert resp.status_code == 200
    assert resp.json()["id"] == bake_id


@pytest.mark.parametrize(("method", "path"), [
    ("get", "/voice/bakes/nope"), ("post", "/voice/bakes/nope/cancel"),
    ("get", "/voice/bakes/nope/adapter"), ("delete", "/voice/bakes/nope"),
])
def test_an_unknown_bake_is_a_404(method, path):
    with _client() as tc:
        assert getattr(tc, method)(path).status_code == 404


def test_a_refused_bake_is_a_422_with_the_reason():
    from modelsvc.voice_bakes import BakeRefused

    class Refusing(FakeBackend):
        def bake_start(self, request):
            raise BakeRefused("take 0: must be mono 16-bit PCM at 48000 Hz")

    with TestClient(create_models_app(Refusing())) as tc:
        resp = tc.post("/voice/bakes", json=_BAKE)
        missing = tc.post("/voice/bakes", json={"tag": {}})
    assert resp.status_code == 422
    assert resp.json()["detail"] == "take 0: must be mono 16-bit PCM at 48000 Hz"
    assert missing.status_code == 422


_TAKE = encode_wav(np.zeros(480, dtype=np.float32), 48000)


def _take(audio: bytes) -> dict:
    return {"audio": base64.b64encode(audio).decode(),
            "sha256": hashlib.sha256(audio).hexdigest(), "text": "Hello.", "language": "en"}


class _FailOnUse:
    """A bake dependency that records and refuses every use."""

    def __init__(self, name: str, used: list):
        self._name = name
        self._used = used

    def __getattr__(self, attr):
        self._used.append((self._name, attr))
        raise AssertionError(f"a refused bake reached {self._name}.{attr}")


class _TrainOffered(_FailOnUse):
    def has_unit(self, unit):
        return True


@pytest.mark.parametrize(("audio", "reason"), [
    pytest.param(_TAKE[:44], "take 1: is cut short: part of its audio is missing",
                 id="header-only"),
    pytest.param(encode_wav(np.zeros(0, dtype=np.float32), 48000), "take 1: holds no audio",
                 id="zero-frames"),
    pytest.param(_TAKE[:-1], "take 1: is cut short: part of its audio is missing",
                 id="last-byte-missing"),
])
def test_a_take_with_missing_audio_is_a_422_before_anything_runs(tmp_path, audio, reason):
    used = []
    root = tmp_path / "voice-bakes"
    bakes = _bakes(slots=_TrainOffered("slots", used), registry=_FailOnUse("registry", used),
                   governor=_FailOnUse("governor", used),
                   scheduler=_FailOnUse("scheduler", used), bake_root=root)
    with TestClient(create_models_app(_composite(bakes=bakes))) as tc:
        resp = tc.post("/voice/bakes", json={"clips": [_take(_TAKE), _take(audio)]})
        current = tc.get("/voice/bakes/current")
    assert resp.status_code == 422
    assert resp.json()["detail"] == reason
    assert current.status_code == 404
    assert used == []
    assert not root.exists()
    assert not bakes.active()
    assert not bakes.gate.closed()


def test_a_bake_where_none_is_offered_is_a_503_apart_from_voice_copy():
    with TestClient(create_models_app(FakeBackend("jetson"))) as tc:
        resp = tc.post("/voice/bakes", json=_BAKE)
    assert resp.status_code == 503
    assert resp.json()["detail"] == "baking a voice is not offered on this device yet"


def test_the_route_passes_the_request_to_the_backend():
    seen = []

    class Recording(FakeBackend):
        def bake_start(self, request):
            seen.append(request)
            return super().bake_start(request)

    with TestClient(create_models_app(Recording())) as tc:
        tc.post("/voice/bakes", json=_BAKE | {"model": "voxcpm2", "config": {"steps": 3}})
    assert seen == [_BAKE | {"model": "voxcpm2", "config": {"steps": 3}}]


def _wav_take(seconds: float = 6.0, *, channels: int = 1) -> bytes:
    import io
    import wave

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(48000)
        handle.writeframes(b"\x01\x00" * int(48000 * seconds) * channels)
    return buffer.getvalue()


def _transcribe_body(audio: bytes, language: str = "ru", **overrides) -> dict:
    return {"audio": base64.b64encode(audio).decode(),
            "sha256": hashlib.sha256(audio).hexdigest(), "language": language} | overrides


class _HearingFake(FakeBackend):
    """Records every take that reached the speech recognition."""

    def __init__(self, profile: str = "mac") -> None:
        super().__init__(profile)
        self.heard: list[tuple[bytes, str]] = []

    def transcribe(self, audio: bytes, language: str) -> dict:
        result = super().transcribe(audio, language)
        self.heard.append((audio, language))
        return result


def test_transcribe_answers_the_words_of_the_take():
    backend = _HearingFake()
    take = _wav_take()
    backend.transcripts[hashlib.sha256(take).hexdigest()] = "Мои слова."
    with TestClient(create_models_app(backend)) as tc:
        resp = tc.post("/voice/transcribe", json=_transcribe_body(take))
    assert resp.status_code == 200
    assert resp.json() == {"text": "Мои слова.", "model": "fake-asr", "revision": "fake",
                           "seconds": 6.0, "elapsed_s": 0.0}
    assert backend.heard == [(take, "ru")]


_BIG = b"\x00" * (5 * 1024 * 1024)


@pytest.mark.parametrize(("body", "reason"), [
    (_transcribe_body(_wav_take()) | {"audio": "not base64!"}, "not valid base64"),
    (_transcribe_body(_wav_take()) | {"sha256": "0" * 64}, "does not match its sha256"),
    (_transcribe_body(_BIG), "the limit is 4194304"),
    (_transcribe_body(_wav_take(channels=2)), "the take: must be mono 16-bit PCM at 48000 Hz"),
    (_transcribe_body(_wav_take(31.0)), "the take is 31.0 s long; the limit is 30 s"),
    (_transcribe_body(_wav_take(), language="de"), "'de'"),
])
def test_a_refused_take_is_a_422_with_its_reason_and_reaches_no_model(body, reason):
    backend = _HearingFake()
    with TestClient(create_models_app(backend)) as tc:
        resp = tc.post("/voice/transcribe", json=body)
    assert resp.status_code == 422
    assert reason in resp.json()["detail"]
    assert backend.heard == []


def test_transcribe_during_a_bake_is_a_409():
    from modelsvc.voice_bakes import BakeRunning

    class _Baking(FakeBackend):
        def transcribe(self, audio, language):
            raise BakeRunning("a voice is being baked (40%); try again when it ends, or cancel it")

    with TestClient(create_models_app(_Baking())) as tc:
        resp = tc.post("/voice/transcribe", json=_transcribe_body(_wav_take()))
    assert resp.status_code == 409
    assert resp.json()["detail"].startswith("a voice is being baked (40%)")


def test_transcribe_where_none_is_offered_is_a_503():
    with TestClient(create_models_app(FakeBackend("jetson"))) as tc:
        resp = tc.post("/voice/transcribe", json=_transcribe_body(_wav_take(), language="en"))
    assert resp.status_code == 503
    assert resp.json()["detail"] == "hearing the words of a take is not offered on this device yet"


def test_a_model_error_is_a_500_with_its_message():
    class _Missing(FakeBackend):
        def transcribe(self, audio, language):
            raise RuntimeError("Whisper weights for 41f01f3 are not on disk: run `make models`")

    with TestClient(create_models_app(_Missing())) as tc:
        resp = tc.post("/voice/transcribe", json=_transcribe_body(_wav_take()))
    assert resp.status_code == 500
    assert "not on disk: run `make models`" in resp.json()["detail"]


def _embed_body(audio: bytes, **overrides) -> dict:
    return {"audio": base64.b64encode(audio).decode(),
            "sha256": hashlib.sha256(audio).hexdigest()} | overrides


class _MeasuringFake(FakeBackend):
    """Records every clip that reached the speaker model."""

    def __init__(self, profile: str = "mac") -> None:
        super().__init__(profile)
        self.measured: list[bytes] = []

    def embed_voice(self, audio: bytes) -> dict:
        result = super().embed_voice(audio)
        self.measured.append(audio)
        return result


def test_voice_embed_answers_the_vector_of_a_clip_over_30_s():
    backend = _MeasuringFake()
    clip = _wav_take(40.0)
    with TestClient(create_models_app(backend)) as tc:
        resp = tc.post("/voice/embed", json=_embed_body(clip))
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["vector"]) == 192
    assert (body["model"], body["revision"], body["seconds"], body["pipeline"]) == (
        "fake-speaker", "fake", 40.0, 1)
    assert body["elapsed_s"] == 0.0
    assert backend.measured == [clip]


@pytest.mark.parametrize(("body", "reason"), [
    (_embed_body(_wav_take()) | {"audio": "not base64!"}, "not valid base64"),
    (_embed_body(_wav_take()) | {"sha256": "0" * 64}, "does not match its sha256"),
    (_embed_body(_BIG), "the limit is 4194304"),
    (_embed_body(_wav_take(channels=2)), "the clip: must be mono 16-bit PCM at 48000 Hz"),
    (_embed_body(b"not a wav"), "the clip: not a WAV file"),
])
def test_a_bad_clip_is_a_422_before_any_load(body, reason):
    backend = _MeasuringFake()
    with TestClient(create_models_app(backend)) as tc:
        resp = tc.post("/voice/embed", json=body)
    assert resp.status_code == 422
    assert reason in resp.json()["detail"]
    assert backend.measured == []


def test_a_clip_with_too_little_voiced_audio_is_a_422_with_the_reason():
    backend = FakeBackend()
    clip = _wav_take()
    backend.speaker_refusals[hashlib.sha256(clip).hexdigest()] = "the clip holds 0.20 s of voiced audio"
    with TestClient(create_models_app(backend)) as tc:
        resp = tc.post("/voice/embed", json=_embed_body(clip))
    assert resp.status_code == 422
    assert resp.json()["detail"] == "the clip holds 0.20 s of voiced audio"


def test_voice_embed_during_a_bake_is_a_409():
    from modelsvc.voice_bakes import BakeRunning

    class _Baking(FakeBackend):
        def embed_voice(self, audio):
            raise BakeRunning("a voice is being baked (40%); try again when it ends, or cancel it")

    with TestClient(create_models_app(_Baking())) as tc:
        resp = tc.post("/voice/embed", json=_embed_body(_wav_take()))
    assert resp.status_code == 409
    assert resp.json()["detail"].startswith("a voice is being baked (40%)")


def test_voice_embed_where_none_is_offered_is_a_503():
    with TestClient(create_models_app(FakeBackend("jetson"))) as tc:
        resp = tc.post("/voice/embed", json=_embed_body(_wav_take()))
    assert resp.status_code == 503
    assert resp.json()["detail"] == "measuring a voice is not offered on this device yet"


def test_a_speaker_model_error_is_a_500_with_its_message():
    class _Missing(FakeBackend):
        def embed_voice(self, audio):
            raise RuntimeError("ECAPA weights for 0f99f2d are not on disk: run `make models`")

    with TestClient(create_models_app(_Missing())) as tc:
        resp = tc.post("/voice/embed", json=_embed_body(_wav_take()))
    assert resp.status_code == 500
    assert "not on disk: run `make models`" in resp.json()["detail"]


_BUFFALO_REV = "d09715916a0778919a770c343533641e250b8699"
_FACE_IMAGE = b"\x89PNG-one-upright-photo"


def _unit_vector(dim: int = 512) -> list[float]:
    return [1.0] + [0.0] * (dim - 1)


_FACE = {"box": [0.375, 0.225, 0.25, 0.25], "score": 0.9, "vector": _unit_vector()}


class _FaceChild:
    """The `faces` child as the service sees it: `call("detect", image)` answers the faces
    of the image, or `{refused, revision}` for an image it cannot read."""

    def __init__(self, reply: dict, activity=None) -> None:
        self.reply = reply
        self.activity = activity
        self.calls: list[tuple] = []
        self.labels: list[str | None] = []

    def call(self, method, *args):
        self.calls.append((method, *args))
        if self.activity is not None:
            self.labels.append(self.activity.current())
        return self.reply


class _FaceSlots:
    def __init__(self, child, offered: bool = True) -> None:
        self.child, self.offered = child, offered

    def has_unit(self, unit):
        return self.offered and unit == "faces"

    def worker(self, unit):
        assert unit == "faces"
        return self.child


class _SeenScheduler:
    def __init__(self) -> None:
        self.seen: list[tuple] = []

    def run(self, needed, priority, fn):
        self.seen.append((tuple(needed), priority))
        return fn()

    def reap_idle(self, units):
        pass


def _face_app(child, *, offered=True, scheduler=None, activity=None, profile="mac"):
    from modelsvc.backends.composite import CompositeBackend

    backend = CompositeBackend(embed=None, slots=_FaceSlots(child, offered),
                               scheduler=scheduler, activity=activity, profile=profile)
    return TestClient(create_models_app(backend))


def _detect_body(image: bytes = _FACE_IMAGE) -> dict:
    return {"image": base64.b64encode(image).decode()}


@pytest.mark.parametrize("profile", ["mac", "jetson", "cloud"])
def test_faces_detect_answers_each_face_with_the_model_and_its_pinned_revision(profile):
    child = _FaceChild({"faces": [_FACE], "revision": _BUFFALO_REV, "elapsed_s": 0.4})
    with _face_app(child, profile=profile) as tc:
        resp = tc.post("/faces/detect", json=_detect_body())
    assert resp.status_code == 200
    body = resp.json()
    assert (body["model"], body["revision"]) == ("buffalo-l", _BUFFALO_REV)
    assert body["faces"] == [_FACE]
    assert child.calls == [("detect", _FACE_IMAGE)]


def test_faces_detect_runs_the_faces_unit_at_batch_priority_labelled_finding_faces():
    from modelsvc.activity import Activity
    from modelsvc.scheduler import Priority

    activity = Activity()
    scheduler = _SeenScheduler()
    child = _FaceChild({"faces": [], "revision": _BUFFALO_REV, "elapsed_s": 0.1}, activity)
    with _face_app(child, scheduler=scheduler, activity=activity) as tc:
        resp = tc.post("/faces/detect", json=_detect_body())
    assert resp.status_code == 200
    assert resp.json()["faces"] == []
    assert scheduler.seen == [(("faces",), Priority.BATCH)]
    assert child.labels == ["finding faces"]


def test_faces_detect_where_no_face_model_runs_is_a_503():
    child = _FaceChild({"faces": [_FACE], "revision": _BUFFALO_REV, "elapsed_s": 0.1})
    with _face_app(child, offered=False) as tc:
        resp = tc.post("/faces/detect", json=_detect_body())
    assert resp.status_code == 503
    assert resp.json()["detail"] == "finding faces is not offered on this device yet"
    assert child.calls == []


def test_an_image_the_face_child_cannot_read_is_a_422_with_its_reason():
    child = _FaceChild({"refused": "cannot identify image file", "revision": _BUFFALO_REV})
    with _face_app(child) as tc:
        resp = tc.post("/faces/detect", json=_detect_body(b"not an image"))
    assert resp.status_code == 422
    assert "cannot identify image file" in resp.json()["detail"]


def test_a_face_reply_of_another_revision_is_a_faces_identity_mismatch():
    child = _FaceChild({"faces": [_FACE], "revision": "another-commit", "elapsed_s": 0.1})
    with _face_app(child) as tc:
        resp = tc.post("/faces/detect", json=_detect_body())
    assert resp.status_code == 500
    assert "faces identity mismatch" in resp.json()["detail"]


def test_a_face_vector_of_another_size_is_an_error():
    short = {"box": [0.1, 0.1, 0.2, 0.2], "score": 0.9, "vector": _unit_vector(7)}
    child = _FaceChild({"faces": [_FACE, short], "revision": _BUFFALO_REV, "elapsed_s": 0.1})
    with _face_app(child) as tc:
        resp = tc.post("/faces/detect", json=_detect_body())
    assert resp.status_code == 500
    assert "7 values, not 512" in resp.json()["detail"]
