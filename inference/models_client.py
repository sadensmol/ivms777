"""`ModelsClient` — the thin HTTP client `app`/`worker`/the CLI use to reach the
`models` service (design §5.1). They hold no models and no torch; every
embed/tag/caption/plan/chat call is an HTTP round trip through this client.
"""

import base64
import json
from collections.abc import Iterator

import httpx


class ModelsCaptionPreempted(RuntimeError):
    """The `models` service preempted an in-flight caption for a
    higher-priority interactive embed (§8.1, plan 15 task 5) — the
    client-side counterpart of `modelsvc.residency.CaptionPreempted`, raised
    on a 503 from `/caption`. `ingest.caption.caption_handler` catches this
    and maps it to `ingest.worker.Preempted` so the job stays pending
    instead of burning a retry."""


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

    def caption(
        self, image: bytes, dimensions: list[str], *, timeout: float = 120.0
    ) -> dict:
        payload = {"image": base64.b64encode(image).decode(), "dimensions": dimensions}
        response = self._client.post("/caption", json=payload, timeout=timeout)
        if response.status_code == 503:
            raise ModelsCaptionPreempted()
        response.raise_for_status()
        return response.json()

    def text_complete(
        self,
        model: str,
        messages: list[dict],
        *,
        json_schema: dict | None = None,
        timeout: float = 120.0,
    ) -> str:
        payload: dict = {"model": model, "messages": messages}
        if json_schema is not None:
            payload["json_schema"] = json_schema
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

    def calibration(self, *, timeout: float = 10.0) -> dict:
        response = self._client.get("/embed/calibration", timeout=timeout)
        response.raise_for_status()
        return response.json()

    def resources(self, *, timeout: float = 10.0) -> dict:
        response = self._client.get("/resources", timeout=timeout)
        response.raise_for_status()
        return response.json()
