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

from collections.abc import Callable, Iterator


class TextBackend:
    def __init__(
        self,
        client,
        *,
        text_worker=None,
        model_name: Callable[[], str] | None = None,
    ) -> None:
        self._client = client
        # A PROVIDER of the text embedder's `TorchWorker` (plan 20), or None on cloud
        # (real /embeddings backend). A provider, not the worker itself: switching the
        # `text_embed` slot builds a new child (design §4.1).
        self._text_worker = text_worker
        # A PROVIDER of the text model llama-server/vLLM keeps resident — for the
        # resource bar (§13). A provider, not a name: switching the `planner` slot
        # restarts the child on a different GGUF (design §4.1), and a name captured at
        # build time would keep reporting the profile default forever.
        self._model_name = model_name

    def text_complete(
        self,
        model: str,
        messages: list[dict],
        json_schema: dict | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        return self._client.complete(
            model, messages, json_schema=json_schema,
            temperature=temperature, max_tokens=max_tokens,
        )

    def text_stream(self, model: str, messages: list[dict]) -> Iterator[str]:
        yield from self._client.stream(model, messages)

    def text_embed(self, model: str, texts: list[str]) -> list[list[float]]:
        # nomic runs in a killable child (plan 20, §8.1) so evicting it really
        # returns its RAM; cloud has a real /embeddings backend and no worker.
        if self._text_worker is None:
            return self._client.embed(model, texts)
        return self._text_worker().call("embed_texts", texts)

    def text_warm(self, model: str) -> None:
        self._client.warm(model)

    def text_evict(self, model: str) -> None:
        self._client.evict(model)

    def resident_models(self) -> list[str]:
        """Text model the backend holds resident, for the resource bar (§13).
        llama-server / vLLM keep their one model loaded for the process's whole life
        and expose no Ollama-style `/api/ps`, so report the slot's CURRENT
        `model_name`. Falls back to a client `loaded_models()` (if any), then `[]`."""
        if self._model_name is not None:
            name = self._model_name()
            if name:
                return [name]
        fn = getattr(self._client, "loaded_models", None)
        return fn() if fn is not None else []
