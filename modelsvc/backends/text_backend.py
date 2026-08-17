"""`TextBackend` — the `/text/*` sub-backend (design §5.1, plan 15 task 4).

Wraps ONE inference client (a real `OpenAICompatClient` talking to llama-server
/ vLLM, or a `FakeInferenceClient` in tests) for text generation, injected by
`build_backend`. This module never constructs a client itself and never imports
`OpenAICompatClient` — that stays in `modelsvc/backends/__init__.py`.

Caption-meaning text embeddings (§9) are a SEPARATE model (`nomic-embed-text-v1.5`).
Since plan 16 dropped Ollama, on mac/jetson it has no server, so it runs
IN-PROCESS here (`embedding.text_embedder`); `text_embed_model` (+ `embed_device`)
select it. When `text_embed_model` is `None` (cloud, which still has an OpenAI
`/embeddings` backend) `text_embed` falls back to the injected client. The import
of `embedding.text_embedder` is lazy (inside `text_embed`), so this module stays
torch-free at import.
"""

from collections.abc import Iterator


class TextBackend:
    def __init__(
        self,
        client,
        *,
        text_embed_model: str | None = None,
        device: str | None = None,
        model_name: str | None = None,
    ) -> None:
        self._client = client
        self._text_embed_model = text_embed_model
        self._device = device or "cpu"
        # The text model llama-server/vLLM keeps resident — for the resource bar (§13).
        self._model_name = model_name

    def text_complete(
        self, model: str, messages: list[dict], json_schema: dict | None = None
    ) -> str:
        return self._client.complete(model, messages, json_schema=json_schema)

    def text_stream(self, model: str, messages: list[dict]) -> Iterator[str]:
        yield from self._client.stream(model, messages)

    def text_embed(self, model: str, texts: list[str]) -> list[list[float]]:
        if self._text_embed_model is None:
            return self._client.embed(model, texts)
        from embedding.text_embedder import get_text_embedder

        return get_text_embedder(self._text_embed_model, self._device).embed_texts(texts)

    def text_warm(self, model: str) -> None:
        self._client.warm(model)

    def text_evict(self, model: str) -> None:
        self._client.evict(model)

    def resident_models(self) -> list[str]:
        """Text model the backend holds resident, for the resource bar (§13).
        llama-server / vLLM keep their one model loaded for the process's whole life
        and expose no Ollama-style `/api/ps`, so report the configured `model_name`
        directly. Falls back to a client `loaded_models()` (if any), then `[]`."""
        if self._model_name:
            return [self._model_name]
        fn = getattr(self._client, "loaded_models", None)
        return fn() if fn is not None else []
