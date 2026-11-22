"""`ModelsClient` — the thin HTTP client `app`/`worker`/the CLI use to reach the
`models` service (design §5.1). They hold no models and no torch; every
embed/tag/caption/plan/chat call is an HTTP round trip through this client.
"""

import base64
import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass

import httpx

# Saved-voice narration (D28): VoxCPM2's speed on MPS float32 is unmeasured, and a 2-3
# minute narration plus a first load can outlast the preset voices' 300 s.
SPEAK_TIMEOUT = 300.0
SAVED_VOICE_SPEAK_TIMEOUT = 1800.0


@dataclass(frozen=True, slots=True, kw_only=True)
class DetectedFace:
    """One face `POST /faces/detect` found: the box in 0..1 of the upright image
    (x, y, w, h), the detector score, the L2-normalised face vector, and the model that
    made it as `<key>@<revision>`."""

    box: tuple[float, float, float, float]
    score: float
    vector: tuple[float, ...]
    model: str


def _detected_face(face: dict, model: str) -> DetectedFace:
    x, y, w, h = face["box"]
    return DetectedFace(box=(x, y, w, h), score=face["score"],
                        vector=tuple(face["vector"]), model=model)


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def _audio(body: dict) -> dict:
    """Decode a `/speak` or `/music` response: the wire carries base64, every caller
    wants bytes to write straight to storage."""
    conditioning = {
        token: entry | {"blob": None if entry.get("blob") is None  # pragma: allowlist secret
                        else base64.b64decode(entry["blob"])}
        for token, entry in (body.get("conditioning") or {}).items()
    }
    return {
        "audio": base64.b64decode(body["audio"]),
        "sample_rate": body["sample_rate"],
        "model": body["model"],
        "segments": body.get("segments", []),
        "revision": body.get("revision"),
        "conditioning": conditioning,
        "adapters": body.get("adapters") or {},
        "controls": body.get("controls") or {},
    }


def _reference_payload(ref: dict) -> dict:
    # A voice model alone has no take (D76): no `audio`, so no `sha256` either.
    payload = {} if ref.get("audio") is None else {
        "audio": _b64(ref["audio"]),
        "sha256": ref["sha256"],
    }
    payload |= {
        "language": ref["language"],
        "transcript": ref.get("transcript"),
        "conditioning": [
            {"model": c["model"], "revision": c["revision"], "blob": _b64(c["blob"])}
            for c in ref.get("conditioning") or []
        ],
    }
    if ref.get("adapter") is not None:
        payload["adapter"] = {"blob": _b64(ref["adapter"]["blob"]),
                              "sha256": ref["adapter"]["sha256"]}
    if "use_clip" in ref:
        payload["use_clip"] = ref["use_clip"]
    if "controls" in ref:
        payload["controls"] = ref["controls"]
    return payload


class ModelsClient:
    def __init__(self, base_url: str, transport: httpx.BaseTransport | None = None) -> None:
        self._client = httpx.Client(base_url=base_url.rstrip("/"), transport=transport)

    def embed_image(self, images: list[bytes], *, timeout: float = 60.0) -> list[list[float]]:
        payload = {"images": [base64.b64encode(image).decode() for image in images]}
        response = self._client.post("/embed/image", json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()["vectors"]

    def embed_text(self, texts: list[str], *, timeout: float = 60.0) -> list[list[float]]:
        response = self._client.post("/embed/text", json={"texts": texts}, timeout=timeout)
        response.raise_for_status()
        return response.json()["vectors"]

    def tag(
        self, image: bytes, dimensions: list[str], *, timeout: float = 60.0
    ) -> dict[str, list[str]]:
        payload = {"image": base64.b64encode(image).decode(), "dimensions": dimensions}
        response = self._client.post("/tag", json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()

    def caption(self, image: bytes, *, timeout: float = 120.0) -> dict:
        payload = {"image": base64.b64encode(image).decode()}
        response = self._client.post("/caption", json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()

    def speak(
        self, text: str, language: str = "en", voice: str | None = None,
        segments: list[str] | None = None,
        voices: list[str | None] | None = None, *,
        references: dict | None = None, timeout: float | None = None,
    ) -> dict:
        """One narration clip (design §11.1). `audio` comes back as raw WAV bytes.

        The timeout is generous on purpose: this is a BATCH call on a board that may
        first have to evict gemma and load the voice model (§8.1).

        `references` speaks SAVED voices: `{token: {audio: bytes, sha256, language,  # pragma: allowlist secret
        transcript, conditioning: [{model, revision, blob: bytes}]}}`, and optionally
        `adapter: {blob: bytes, sha256}` and `use_clip`; a reference with `use_clip`
        False may leave out `audio` and `sha256` (D76). A test may add `controls:
        {cfg_value, inference_timesteps}`. The reply's `conditioning` comes back with
        its blobs as bytes, and `adapters` and `controls` as the service sent them.
        `timeout=None` is 1800 s with references and 300 s without.
        """
        if timeout is None:
            timeout = SPEAK_TIMEOUT if references is None else SAVED_VOICE_SPEAK_TIMEOUT
        payload = {"text": text, "language": language, "voice": voice,
                   "segments": segments, "voices": voices}
        if references is not None:
            payload["references"] = {token: _reference_payload(ref)  # pragma: allowlist secret
                                     for token, ref in references.items()}
        response = self._client.post("/speak", json=payload, timeout=timeout)
        response.raise_for_status()
        return _audio(response.json())

    def music(
        self,
        prompt: str = "",
        seconds: float = 30.0,
        seed: int | None = None,
        *,
        sections: list[dict] | None = None,
        temperature: float | None = None,
        guidance: float | None = None,
        timeout: float = 1800.0,
    ) -> dict:
        """The film's score (design §11.1). `audio` comes back as raw WAV bytes.

        `sections` is the whole score — each one continues the previous, so the reply is
        a single piece whose `segments` say where each section landed.

        1800 s: MusicGen is autoregressive and measured at ~4.9x realtime on this mac,
        so a three-minute score is fifteen minutes of GPU time, and it may queue behind
        an in-flight caption.
        """
        payload = {"prompt": prompt, "seconds": seconds, "seed": seed,
                   "sections": sections, "temperature": temperature,
                   "guidance": guidance}
        response = self._client.post("/music", json=payload, timeout=timeout)
        response.raise_for_status()
        return _audio(response.json())

    def text_complete(
        self,
        model: str,
        messages: list[dict],
        *,
        json_schema: dict | None = None,
        timeout: float = 120.0,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        payload: dict = {"model": model, "messages": messages}
        if json_schema is not None:
            payload["json_schema"] = json_schema
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        response = self._client.post("/text/complete", json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()["text"]

    def text_stream(
        self, model: str, messages: list[dict], *, timeout: float = 120.0
    ) -> Iterator[str]:
        with self._client.stream(
            "POST", "/text/stream", json={"model": model, "messages": messages}, timeout=timeout
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break
                yield json.loads(data)["token"]

    def text_context(self, *, timeout: float = 5.0) -> int | None:
        """Tokens the text model's server accepts (§5.1). `None` when it cannot say.

        Read from the RUNNING server, so a caller assembling a large prompt (the story
        pass, §11.1) cuts it to what will be accepted instead of losing the whole call
        to a rejection.
        """
        response = self._client.get("/text/context", timeout=timeout)
        response.raise_for_status()
        value = response.json().get("context")
        return int(value) if value else None

    def text_embed(
        self, model: str, texts: list[str], *, timeout: float = 60.0
    ) -> list[list[float]]:
        response = self._client.post(
            "/text/embed", json={"model": model, "texts": texts}, timeout=timeout
        )
        response.raise_for_status()
        return response.json()["vectors"]

    def text_warm(self, model: str, *, timeout: float = 120.0) -> None:
        response = self._client.post("/text/warm", json={"model": model}, timeout=timeout)
        response.raise_for_status()

    def text_evict(self, model: str, *, timeout: float = 30.0) -> None:
        response = self._client.post("/text/evict", json={"model": model}, timeout=timeout)
        response.raise_for_status()

    def embed_spec(self, *, timeout: float = 10.0) -> dict:
        """Calibration + the selected image embedder's preprocessing contract +
        the slots' `generation` (design §4.1). One round trip: the caller needs
        all three to embed an image correctly."""
        response = self._client.get("/embed/spec", timeout=timeout)
        response.raise_for_status()
        return response.json()

    def catalog(self, *, timeout: float = 10.0) -> dict:
        response = self._client.get("/models/catalog", timeout=timeout)
        response.raise_for_status()
        return response.json()

    def set_slots(self, slots: dict, *, timeout: float = 30.0) -> dict:
        response = self._client.put("/models/slots", json={"slots": slots}, timeout=timeout)
        response.raise_for_status()
        return response.json()

    def download(self, slot: str, key: str, *, timeout: float = 10.0) -> dict:
        response = self._client.post(
            "/models/download", json={"slot": slot, "key": key}, timeout=timeout
        )
        response.raise_for_status()
        return response.json()

    def resources(self, *, timeout: float = 10.0) -> dict:
        response = self._client.get("/resources", timeout=timeout)
        response.raise_for_status()
        return response.json()

    def models_state(self, *, timeout: float = 10.0) -> dict:
        response = self._client.get("/models", timeout=timeout)
        response.raise_for_status()
        return response.json()

    def model_ensure(self, name: str, *, timeout: float = 120.0) -> None:
        response = self._client.post(f"/models/{name}/ensure", timeout=timeout)
        response.raise_for_status()

    def model_unload(self, name: str, *, timeout: float = 30.0) -> None:
        response = self._client.post(f"/models/{name}/unload", timeout=timeout)
        response.raise_for_status()

    def bake_start(self, clips: list[dict], *, tag: dict, config: dict | None = None,
                   timeout: float = 600.0) -> dict:
        """Start a bake; `clips` = `[{audio: bytes, sha256, text, language}]`. The
        service answers 202 with the status at once."""
        payload = {"clips": [clip | {"audio": _b64(clip["audio"])} for clip in clips],
                   "config": config, "tag": tag}
        response = self._client.post("/voice/bakes", json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()

    def bake_status(self, bake_id: str, *, timeout: float = 10.0) -> dict:
        response = self._client.get(f"/voice/bakes/{bake_id}", timeout=timeout)
        response.raise_for_status()
        return response.json()

    def bake_current(self, *, timeout: float = 10.0) -> dict | None:
        response = self._client.get("/voice/bakes/current", timeout=timeout)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()

    def bake_cancel(self, bake_id: str, *, timeout: float = 30.0) -> dict:
        response = self._client.post(f"/voice/bakes/{bake_id}/cancel", timeout=timeout)
        response.raise_for_status()
        return response.json()

    def bake_adapter(self, bake_id: str, *, timeout: float = 120.0) -> bytes:
        response = self._client.get(f"/voice/bakes/{bake_id}/adapter", timeout=timeout)
        response.raise_for_status()
        return response.content

    def bake_delete(self, bake_id: str, *, timeout: float = 30.0) -> None:
        response = self._client.delete(f"/voice/bakes/{bake_id}", timeout=timeout)
        response.raise_for_status()

    def transcribe(self, audio: bytes, language: str, *, timeout: float = 600.0) -> dict:
        """The words of one take, `{text, model, revision, seconds, elapsed_s}`. 600 s
        covers the first call's Whisper load and a wait for a scheduler slot. A non-2xx
        raises `httpx.HTTPStatusError`."""
        payload = {"audio": _b64(audio), "sha256": hashlib.sha256(audio).hexdigest(),
                   "language": language}
        response = self._client.post("/voice/transcribe", json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()

    def embed_voice(self, audio: bytes, *, timeout: float = 600.0) -> dict:
        """The speaker embedding of one clip, `{vector, model, revision, seconds, elapsed_s,
        pipeline}`. 600 s covers the first call's load and a wait for a scheduler slot. A
        non-2xx raises `httpx.HTTPStatusError`."""
        payload = {"audio": _b64(audio), "sha256": hashlib.sha256(audio).hexdigest()}
        response = self._client.post("/voice/embed", json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()

    def detect_faces(self, image: bytes, *, timeout: float = 600.0) -> list[DetectedFace]:
        """The faces in one upright image, as `POST /faces/detect` answers them. 600 s
        covers the first call's load and a wait for a scheduler slot. A non-2xx raises
        `httpx.HTTPStatusError`."""
        response = self._client.post(
            "/faces/detect", json={"image": _b64(image)}, timeout=timeout)
        response.raise_for_status()
        body = response.json()
        model = f"{body['model']}@{body['revision']}"
        return [_detected_face(face, model) for face in body["faces"]]
