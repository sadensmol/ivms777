"""`ModelsClient` — the thin HTTP client `app`/`worker`/the CLI use to reach the
`models` service (design §5.1). They hold no models and no torch; every
embed/tag/caption/plan/chat call is an HTTP round trip through this client.
"""

import base64
import json
from collections.abc import Iterator

import httpx


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
