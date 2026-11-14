"""`ModelsClient` round-tripped against the `models` service app (design §5.1)
via the FastAPI `TestClient`'s own sync-capable ASGI transport — no real
network, no real models."""

import base64
import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from feed.saved_voices import model_reference
from inference.models_client import ModelsClient
from modelsvc.app import create_models_app
from modelsvc.backends.fake import FAKE_CONTEXT, FakeBackend


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


def test_speak_round_trips_raw_wav_bytes():
    result = _client().speak("Beach day.")
    assert isinstance(result["audio"], bytes)
    assert result["audio"][:4] == b"RIFF"
    assert result["sample_rate"] == 24000


def test_music_round_trips_raw_wav_bytes():
    result = _client().music("warm guitar", seconds=2.0)
    assert result["audio"][:4] == b"RIFF"
    assert result["sample_rate"] == 44100


def test_speak_503s_when_no_voice_covers_the_language():
    # The catalog has no `off` voice any more (CLAUDE.md § "No off / none options"), so
    # the only 503 left on this route is "nothing offered speaks this language" (§4.1).
    backend = FakeBackend()
    transport = TestClient(create_models_app(backend))._transport
    client = ModelsClient("http://modelsvc", transport=transport)
    with pytest.raises(httpx.HTTPStatusError) as excinfo:
        client.speak("hi", "ja")
    assert excinfo.value.response.status_code == 503


def test_speak_reports_the_models_own_message_on_a_failure():
    # Narration is mandatory (§11.1) and its failure is shown to the user, so the
    # reason must survive the hop. Losing it is what made a Silero refusal read as
    # "the models service refused to speak ru" with no way to tell why.
    class Refusing(FakeBackend):
        def speak(self, text, language="en", voice=None, segments=None, voices=None):
            raise RuntimeError("Model couldn't generate your text")

    transport = TestClient(create_models_app(Refusing()))._transport
    client = ModelsClient("http://modelsvc", transport=transport)
    with pytest.raises(httpx.HTTPStatusError) as excinfo:
        client.speak("hi")
    assert excinfo.value.response.status_code == 500
    assert excinfo.value.response.json()["detail"] == "Model couldn't generate your text"


def test_text_context_round_trips():
    assert _client().text_context() == FAKE_CONTEXT


def test_music_sends_the_whole_score():
    result = _client().music(sections=[{"brief": "piano", "seconds": 3},
                                       {"brief": "drums", "seconds": 3}])
    assert len(result["segments"]) == 2
    assert isinstance(result["audio"], bytes)


_REFS = {"saved:1": {"audio": b"RIFF-clip", "sha256": "ab", "language": "en",
                     "transcript": None,
                     "conditioning": [{"model": "voxcpm2", "revision": "abc",
                                       "blob": b"\x00old"}]}}
_REPLY = {"audio": base64.b64encode(b"RIFF").decode(), "sample_rate": 48000,
          "model": "voxcpm2", "revision": "abc", "segments": [],
          "conditioning": {
              "saved:1": {"state": "built", "model": "voxcpm2", "revision": "abc",
                          "reason": "clip changed",
                          "blob": base64.b64encode(b"\x00new").decode()},
              "saved:2": {"state": "restored", "model": "voxcpm2", "revision": "abc",
                          "reason": None, "blob": None},
          }}


def _capturing_client(seen: list) -> ModelsClient:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((json.loads(request.content), request.extensions["timeout"]))
        return httpx.Response(200, json=_REPLY)

    return ModelsClient("http://modelsvc", transport=httpx.MockTransport(handler))


def test_speak_encodes_the_references_as_base64():
    seen: list = []
    _capturing_client(seen).speak("hi", "en", "saved:1", references=_REFS)
    payload, _ = seen[0]
    assert payload["references"] == {"saved:1": {
        "audio": base64.b64encode(b"RIFF-clip").decode(), "sha256": "ab", "language": "en",
        "transcript": None,
        "conditioning": [{"model": "voxcpm2", "revision": "abc",
                          "blob": base64.b64encode(b"\x00old").decode()}],
    }}


def test_speak_without_references_sends_no_references_field():
    seen: list = []
    _capturing_client(seen).speak("hi")
    assert "references" not in seen[0][0]


def test_speak_waits_1800_s_with_references_and_300_s_without():
    seen: list = []
    client = _capturing_client(seen)
    client.speak("hi", references=_REFS, voice="saved:1")
    client.speak("hi")
    client.speak("hi", timeout=12.0)
    assert [timeout["read"] for _, timeout in seen] == [1800.0, 300.0, 12.0]


def test_speak_decodes_the_revision_and_the_conditioning_blobs():
    result = _capturing_client([]).speak("hi", "en", "saved:1", references=_REFS)
    assert result["revision"] == "abc"
    assert result["conditioning"]["saved:1"]["blob"] == b"\x00new"
    assert result["conditioning"]["saved:1"]["reason"] == "clip changed"
    assert result["conditioning"]["saved:2"]["blob"] is None


def test_speak_round_trips_a_saved_voice_through_the_service():
    import hashlib

    clip = b"RIFF-a-saved-example"
    refs = {"saved:1": {"audio": clip, "sha256": hashlib.sha256(clip).hexdigest(),
                        "language": "en", "transcript": None, "conditioning": []}}
    result = _client().speak("hi", "en", "saved:1", references=refs)
    assert result["audio"][:4] == b"RIFF"
    assert result["revision"] is None
    assert isinstance(result["conditioning"]["saved:1"]["blob"], bytes)


def test_speak_sends_the_adapter_as_base64_with_its_sha256_and_use_clip():
    seen: list = []
    refs = {"saved:1": _REFS["saved:1"] | {"adapter": {"blob": b"\x00lora", "sha256": "cd"},
                                           "use_clip": False}}
    _capturing_client(seen).speak("hi", "en", "saved:1", references=refs)
    sent = seen[0][0]["references"]["saved:1"]
    assert sent["adapter"] == {"blob": base64.b64encode(b"\x00lora").decode(), "sha256": "cd"}
    assert sent["use_clip"] is False


def test_speak_passes_the_adapter_states_through():
    states = {"saved:1": {"state": "kept", "sha256": "cd"}}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_REPLY | {"adapters": states})

    client = ModelsClient("http://modelsvc", transport=httpx.MockTransport(handler))
    assert client.speak("hi", "en", "saved:1", references=_REFS)["adapters"] == states


def test_speak_sends_no_audio_and_no_sha256_for_a_model_only_reference():
    seen: list = []
    refs = {"saved:1": {"language": "ru", "adapter": {"blob": b"\x00lora", "sha256": "cd"},
                        "use_clip": False}}
    _capturing_client(seen).speak("hi", "ru", "saved:1", references=refs)
    assert seen[0][0]["references"] == {"saved:1": {
        "language": "ru", "transcript": None, "conditioning": [],
        "adapter": {"blob": base64.b64encode(b"\x00lora").decode(), "sha256": "cd"},
        "use_clip": False,
    }}


def test_speak_round_trips_a_model_only_reference_through_the_service():
    import hashlib

    blob = b"safetensors-of-a-baked-voice"
    sha = hashlib.sha256(blob).hexdigest()
    refs = {"saved:1": {"language": "ru", "adapter": {"blob": blob, "sha256": sha},
                        "use_clip": False}}
    result = _client().speak("hi", "ru", "saved:1", references=refs)
    assert result["audio"][:4] == b"RIFF"
    assert result["conditioning"] == {}
    assert result["adapters"] == {"saved:1": {"state": "loaded", "sha256": sha}}


_CONTROLS = {"cfg_value": 3.0, "inference_timesteps": 20}


def test_speak_forwards_the_controls_of_a_reference():
    seen: list = []
    refs = {"saved:1": _REFS["saved:1"] | {"controls": _CONTROLS}}
    _capturing_client(seen).speak("hi", "en", "saved:1", references=refs)
    assert seen[0][0]["references"]["saved:1"]["controls"] == _CONTROLS


def test_speak_passes_the_echo_of_the_controls_through():
    echo = {"saved:1": _CONTROLS}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_REPLY | {"controls": echo})

    client = ModelsClient("http://modelsvc", transport=httpx.MockTransport(handler))
    assert client.speak("hi", "en", "saved:1", references=_REFS)["controls"] == echo


def test_a_reply_with_no_echo_has_empty_controls():
    reply = _capturing_client([]).speak("hi", "en", "saved:1", references=_REFS)
    assert reply["controls"] == {}


def test_the_reference_the_app_builds_sends_no_controls():
    seen: list = []
    model = SimpleNamespace(sha256="cd")
    refs = {"saved:1": model_reference(model, b"\x00lora", "ru")}
    _capturing_client(seen).speak("hi", "ru", "saved:1", references=refs)
    assert "controls" not in seen[0][0]["references"]["saved:1"]


def _bake_client(seen: list, status: int = 200, body=None, content: bytes = b"") -> ModelsClient:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append({"method": request.method, "path": request.url.path,
                     "timeout": request.extensions["timeout"]["read"],
                     "json": json.loads(request.content) if request.content else None})
        if body is not None:
            return httpx.Response(status, json=body)
        return httpx.Response(status, content=content)

    return ModelsClient("http://modelsvc", transport=httpx.MockTransport(handler))


_STATUS = {"id": "b1", "state": "running"}


@pytest.mark.parametrize(("call", "method", "path", "timeout"), [
    (lambda c: c.bake_status("b1"), "GET", "/voice/bakes/b1", 10.0),
    (lambda c: c.bake_current(), "GET", "/voice/bakes/current", 10.0),
    (lambda c: c.bake_cancel("b1"), "POST", "/voice/bakes/b1/cancel", 30.0),
])
def test_the_bake_status_calls_reach_their_paths_and_decode_the_status(
        call, method, path, timeout):
    seen: list = []
    assert call(_bake_client(seen, body=_STATUS)) == _STATUS
    assert (seen[0]["method"], seen[0]["path"], seen[0]["timeout"]) == (method, path, timeout)


def test_bake_start_encodes_the_takes_and_waits_600_s():
    seen: list = []
    clip = {"audio": b"RIFF-take", "sha256": "ab", "text": "Hello.", "language": "en"}
    result = _bake_client(seen, 202, body=_STATUS).bake_start(
        [clip], tag={"check": "me"}, config={"steps": 3})
    assert result == _STATUS
    assert (seen[0]["method"], seen[0]["path"], seen[0]["timeout"]) == (
        "POST", "/voice/bakes", 600.0)
    assert seen[0]["json"] == {
        "clips": [clip | {"audio": base64.b64encode(b"RIFF-take").decode()}],
        "config": {"steps": 3}, "tag": {"check": "me"}}


def test_bake_current_is_none_when_no_bake_exists():
    seen: list = []
    assert _bake_client(seen, 404, body={"detail": "no such bake"}).bake_current() is None


def test_bake_adapter_returns_the_raw_bytes():
    seen: list = []
    assert _bake_client(seen, content=b"\x00safetensors").bake_adapter("b1") == b"\x00safetensors"
    assert (seen[0]["method"], seen[0]["path"], seen[0]["timeout"]) == (
        "GET", "/voice/bakes/b1/adapter", 120.0)


def test_bake_delete_sends_a_delete_and_returns_nothing():
    seen: list = []
    assert _bake_client(seen, 204).bake_delete("b1") is None
    assert (seen[0]["method"], seen[0]["path"], seen[0]["timeout"]) == (
        "DELETE", "/voice/bakes/b1", 30.0)


def test_a_refused_bake_raises_for_its_status():
    with pytest.raises(httpx.HTTPStatusError):
        _bake_client([], 409, body={"detail": "a voice is already being baked"}).bake_status("b1")


def test_a_bake_round_trips_through_the_service():
    client = _client()
    started = client.bake_start([{"audio": b"RIFF", "sha256": "ab", "text": "Hi.",
                                  "language": "en"}], tag={"check": "me"})
    assert client.bake_current()["id"] == started["id"]
    assert [client.bake_status(started["id"])["state"] for _ in range(3)][-1] == "done"
    assert client.bake_adapter(started["id"]) == b"fake-adapter"
    client.bake_delete(started["id"])
    assert client.bake_current() is None


_HEARD = {"text": "Мои слова.", "model": "whisper-large-v3-turbo", "revision": "41f01f3",
          "seconds": 6.0, "elapsed_s": 1.5}


def test_transcribe_posts_the_take_its_sha256_and_language_and_waits_600_s():
    import hashlib

    seen: list = []
    result = _bake_client(seen, body=_HEARD).transcribe(b"RIFF-take", "ru")
    assert result == _HEARD
    assert (seen[0]["method"], seen[0]["path"], seen[0]["timeout"]) == (
        "POST", "/voice/transcribe", 600.0)
    assert seen[0]["json"] == {"audio": base64.b64encode(b"RIFF-take").decode(),
                               "sha256": hashlib.sha256(b"RIFF-take").hexdigest(),
                               "language": "ru"}


def test_a_refused_transcribe_raises_for_its_status():
    client = _bake_client([], 409, body={"detail": "a voice is being baked (40%)"})
    with pytest.raises(httpx.HTTPStatusError) as caught:
        client.transcribe(b"RIFF", "en")
    assert caught.value.response.status_code == 409


def test_a_transcribe_round_trips_through_the_service():
    import io
    import wave

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(48000)
        handle.writeframes(b"\x01\x00" * 48000 * 6)
    result = _client().transcribe(buffer.getvalue(), "en")
    assert result == {"text": "fake words", "model": "fake-asr", "revision": "fake",
                      "seconds": 6.0, "elapsed_s": 0.0}


_EMBEDDED = {"vector": [0.5] * 192, "model": "ecapa-voxceleb", "revision": "0f99f2d",
             "seconds": 6.0, "elapsed_s": 0.2, "pipeline": 1}


def test_embed_voice_posts_the_clip_and_its_sha256_and_waits_600_s():
    import hashlib

    seen: list = []
    result = _bake_client(seen, body=_EMBEDDED).embed_voice(b"RIFF-clip")
    assert result == _EMBEDDED
    assert (seen[0]["method"], seen[0]["path"], seen[0]["timeout"]) == (
        "POST", "/voice/embed", 600.0)
    assert seen[0]["json"] == {"audio": base64.b64encode(b"RIFF-clip").decode(),
                               "sha256": hashlib.sha256(b"RIFF-clip").hexdigest()}


@pytest.mark.parametrize("status", [409, 422, 503])
def test_a_refused_embed_raises_for_its_status(status):
    client = _bake_client([], status, body={"detail": "no"})
    with pytest.raises(httpx.HTTPStatusError) as caught:
        client.embed_voice(b"RIFF")
    assert caught.value.response.status_code == status


def test_an_embed_round_trips_through_the_service():
    import io
    import wave

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(48000)
        handle.writeframes(b"\x01\x00" * 48000 * 6)
    result = _client().embed_voice(buffer.getvalue())
    assert len(result["vector"]) == 192
    assert (result["model"], result["seconds"], result["pipeline"]) == ("fake-speaker", 6.0, 1)
