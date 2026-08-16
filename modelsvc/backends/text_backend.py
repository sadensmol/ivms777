"""`TextBackend` — the `/text/*` sub-backend (design §5.1, plan 15 task 4).

Wraps ONE inference client (a real `OpenAICompatClient` talking to Ollama, or
a `FakeInferenceClient` in tests), injected by `build_backend` — the same
client instance `build_backend` also hands to the caption backend's
`OllamaCaptioner`, so the service holds exactly one Ollama gateway (CLAUDE.md
§ "One model process"; design §5.1). This module never constructs a client
itself and never imports `OpenAICompatClient` — that stays in
`modelsvc/backends/__init__.py::build_backend`, the one place that wires the
shared client into both sub-backends.
"""

from collections.abc import Iterator


class TextBackend:
    def __init__(self, client) -> None:
        self._client = client

    def text_complete(
        self, model: str, messages: list[dict], json_schema: dict | None = None
    ) -> str:
        return self._client.complete(model, messages, json_schema=json_schema)

    def text_stream(self, model: str, messages: list[dict]) -> Iterator[str]:
        yield from self._client.stream(model, messages)

    def text_embed(self, model: str, texts: list[str]) -> list[list[float]]:
        return self._client.embed(model, texts)

    def text_warm(self, model: str) -> None:
        self._client.warm(model)

    def text_evict(self, model: str) -> None:
        self._client.evict(model)

    def resident_models(self) -> list[str]:
        """Text models loaded in the backend (Ollama /api/ps) — so the resource
        bar shows the planner/chat model on mac/cloud, where text is not
        in-process and so is invisible to the residency manager (§13)."""
        fn = getattr(self._client, "loaded_models", None)
        return fn() if fn is not None else []
