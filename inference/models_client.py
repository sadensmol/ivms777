"""`ModelsClient` — the thin HTTP client `app`/`worker`/the CLI use to reach the
`models` service (design §5.1). They hold no models and no torch; every
embed/tag/caption/plan/chat call is an HTTP round trip through this client.
"""

import base64
import json
from collections.abc import Iterator

import httpx


def _audio(body: dict) -> dict:
    """Decode a `/speak` or `/music` response: the wire carries base64, every caller
    wants bytes to write straight to storage."""
    return {
        "audio": base64.b64decode(body["audio"]),
        "sample_rate": body["sample_rate"],
        "model": body["model"],
        "segments": body.get("segments", []),
    }


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
        voices: list[str | None] | None = None, *, timeout: float = 300.0,
    ) -> dict:
        """One narration clip (design §11.1). `audio` comes back as raw WAV bytes.

        The timeout is generous on purpose: this is a BATCH call on a board that may
        first have to evict gemma and load the voice model (§8.1).
        """
        payload = {"text": text, "language": language, "voice": voice,
                   "segments": segments, "voices": voices}
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
