import base64
import json
import threading
from collections.abc import Callable, Iterator
from typing import Protocol, TypedDict

import httpx

# How often the cancellable-completion watcher polls should_stop while the read
# blocks (§8.1). Small so a preempt aborts an in-flight caption within a tick.
_CANCEL_POLL_S = 0.25


class ChatMessage(TypedDict):
    role: str
    content: object


class InferenceCancelled(RuntimeError):
    """A streamed completion was stopped early because its `should_stop` callback
    fired — the caption force-preempt (§8.1). Closing the stream disconnects the
    backend, which stops generating; the caller turns this into a `Preempted`."""


def encode_image(data: bytes, mime: str = "image/jpeg") -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


class InferenceClient(Protocol):
    def complete(
        self,
        model: str,
        messages: list[ChatMessage],
        *,
        json_schema: dict | None = None,
        timeout: float = 120.0,
        should_stop: Callable[[], bool] | None = None,
    ) -> str: ...

    def stream(
        self,
        model: str,
        messages: list[ChatMessage],
        *,
        timeout: float = 120.0,
    ) -> Iterator[str]: ...

    def embed(
        self, model: str, texts: list[str], *, timeout: float = 60.0
    ) -> list[list[float]]: ...

    def warm(self, model: str, *, timeout: float = 120.0) -> None: ...

    def evict(self, model: str, *, timeout: float = 30.0) -> None: ...


class OpenAICompatClient:
    """Talks to anything speaking the OpenAI chat-completions API.

    That covers Ollama (mac, jetson) and vLLM (cloud), so swapping the inference
    backend is a base_url change, not a code change.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str = "unused",
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            transport=transport,
        )

    def complete(
        self,
        model: str,
        messages: list[ChatMessage],
        *,
        json_schema: dict | None = None,
        timeout: float = 120.0,
        should_stop: Callable[[], bool] | None = None,
    ) -> str:
        payload: dict = {"model": model, "messages": messages}
        if json_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": json_schema, "strict": True},
            }
        if should_stop is None:
            payload["stream"] = False
            response = self._client.post("/chat/completions", json=payload, timeout=timeout)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        # Cancellable path (§8.1): stream so the call can be abandoned. `iter_lines()`
        # BLOCKS while the backend prefills — a 7B VLM captioning a full image can
        # take tens of seconds before its first token — so polling `should_stop`
        # only between lines would miss a preempt for that whole wait. A watcher
        # thread polls `should_stop` and CLOSES the response the moment preempt
        # fires; closing unblocks the read (disconnecting the backend, which stops
        # generating), and we raise InferenceCancelled.
        payload["stream"] = True
        parts: list[str] = []
        cancelled = threading.Event()
        watcher_done = threading.Event()
        with self._client.stream(
            "POST", "/chat/completions", json=payload, timeout=timeout
        ) as response:
            response.raise_for_status()

            def _watch() -> None:
                while not watcher_done.wait(_CANCEL_POLL_S):
                    if should_stop():
                        cancelled.set()
                        response.close()  # unblock a read stuck in prefill
                        return

            watcher = threading.Thread(target=_watch, daemon=True)
            watcher.start()
            try:
                for line in response.iter_lines():
                    if should_stop():
                        cancelled.set()
                        break
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        break
                    delta = json.loads(data)["choices"][0].get("delta", {}).get("content")
                    if delta:
                        parts.append(delta)
            except (httpx.StreamClosed, httpx.ReadError, httpx.RemoteProtocolError):
                # The watcher closed the stream out from under the read — expected on
                # preempt; any other read error (real network fault) is re-raised.
                if not cancelled.is_set():
                    raise
            finally:
                watcher_done.set()
                watcher.join(timeout=1.0)
        if cancelled.is_set():
            raise InferenceCancelled(model)
        return "".join(parts)

    def stream(
        self,
        model: str,
        messages: list[ChatMessage],
        *,
        timeout: float = 120.0,
    ) -> Iterator[str]:
        payload = {"model": model, "messages": messages, "stream": True}
        with self._client.stream(
            "POST", "/chat/completions", json=payload, timeout=timeout
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                delta = json.loads(data)["choices"][0].get("delta", {}).get("content")
                if delta:
                    yield delta

    def embed(
        self, model: str, texts: list[str], *, timeout: float = 60.0
    ) -> list[list[float]]:
        """Text embeddings via the OpenAI-compatible `/embeddings` endpoint.

        Ollama and vLLM both expose it, so any loaded model can embed — we reuse
        the planner model that already stays resident, no extra model to pull.
        """
        response = self._client.post(
            "/embeddings", json={"model": model, "input": texts}, timeout=timeout
        )
        response.raise_for_status()
        return [item["embedding"] for item in response.json()["data"]]

    def _native_url(self, path: str) -> str:
        """Ollama's native endpoints (e.g. /api/generate) live one level above
        the OpenAI-compat /v1 base — strip it to get the host root.

        httpx normalizes a base_url that has a path to a trailing slash
        ("http://host:11434/v1/"), so strip it before checking the /v1 suffix.
        """
        base = str(self._client.base_url).rstrip("/")
        root = base[: -len("/v1")] if base.endswith("/v1") else base
        return root + path

    def warm(self, model: str, *, timeout: float = 120.0) -> None:
        # An empty prompt loads the model without generating tokens.
        self._client.post(
            self._native_url("/api/generate"),
            json={"model": model, "prompt": "", "stream": False},
            timeout=timeout,
        ).raise_for_status()

    def evict(self, model: str, *, timeout: float = 30.0) -> None:
        # keep_alive: 0 unloads the model immediately (Ollama-native semantics).
        self._client.post(
            self._native_url("/api/generate"),
            json={"model": model, "prompt": "", "keep_alive": 0, "stream": False},
            timeout=timeout,
        ).raise_for_status()
