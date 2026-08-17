"""`RemoteInferenceClient` — `inference.client.InferenceClient` implemented
over HTTP via `ModelsClient` (design §5.1, plan 15 task 4).

`config.Settings.build_inference_client()`'s non-fake path returns this. It is
a transparent shim: `search/planner.py`, `chat/agent.py`, `chat/retrieve.py`,
`web/app.py`, `ingest/caption.py` keep calling the `InferenceClient` protocol
(`complete`/`stream`/`embed`/`warm`/`evict`) exactly as before — they don't
know or care that Ollama now lives behind the `models` service, not in this
process. This module holds NO `OpenAICompatClient`/torch import; it only
calls `ModelsClient` (CLAUDE.md § "One model process").
"""

from collections.abc import Callable, Iterator

from inference.client import ChatMessage
from inference.models_client import ModelsClient


class RemoteInferenceClient:
    def __init__(self, client: ModelsClient) -> None:
        self._client = client

    def complete(
        self,
        model: str,
        messages: list[ChatMessage],
        *,
        json_schema: dict | None = None,
        timeout: float = 120.0,
        temperature: float | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> str:
        # `should_stop` is ignored: no app/worker text caller passes it (caption
        # preemption, §8.1, is internal to the service now) — see Ruling R1/R5.
        # `temperature` is accepted for interface parity; forwarding it through the
        # models-service text endpoint is pending plan-15 gateway work.
        return self._client.text_complete(
            model, messages, json_schema=json_schema, timeout=timeout
        )

    def stream(
        self,
        model: str,
        messages: list[ChatMessage],
        *,
        timeout: float = 120.0,
        temperature: float | None = None,
    ) -> Iterator[str]:
        yield from self._client.text_stream(model, messages, timeout=timeout)

    def embed(
        self, model: str, texts: list[str], *, timeout: float = 60.0
    ) -> list[list[float]]:
        return self._client.text_embed(model, texts, timeout=timeout)

    def warm(self, model: str, *, timeout: float = 120.0) -> None:
        # No-op: residency is the `models` service's own concern (Task 5); only
        # the soon-deleted coordinator (Task 7) ever called this.
        return None

    def evict(self, model: str, *, timeout: float = 30.0) -> None:
        return None
