"""The `ModelBackend` protocol (design §5.1) — every model, LLM, embedder, and
heavy AI library the `models` service can be configured with implements this.

`modelsvc.app` never talks to torch/transformers/Ollama directly; it calls
whichever `ModelBackend` it was built with. Task 1 ships only `FakeBackend`
(no torch); the real SigLIP/VLM/Ollama-backed implementations arrive in later
tasks behind this same protocol, so the HTTP surface never changes.
"""

from collections.abc import Iterator
from typing import Protocol


class ModelBackend(Protocol):
    def embed_image(self, images: list[bytes]) -> list[list[float]]:
        """SigLIP image embeddings (ingest `embed`, "similar")."""
        ...

    def embed_text(self, texts: list[str]) -> list[list[float]]:
        """SigLIP text embeddings, same joint space as `embed_image` — search
        and chat query vectors."""
        ...

    def tag(self, image: bytes, dimensions: list[str]) -> dict[str, list[str]]:
        """Zero-shot label scoring against `vocab.yaml` dimensions (ingest
        `taxonomy`) — one label list per requested dimension."""
        ...

    def calibration(self) -> dict:
        """SigLIP's zero-shot calibration: `logit_scale`, `logit_bias`.

        `RemoteEmbedder` fetches this once and caches it, so its `Embedder`-
        protocol `logit_scale`/`logit_bias` attrs stay cheap after first use
        (taxonomy tagging, §9, computes probabilities client-side — ruling R2
        of plan 15 task 2).
        """
        ...

    def caption(self, image: bytes) -> dict:
        """A caption sentence for one image (no tags — tags are SigLIP-only, §7).

        Returns a dict with keys `caption`, `title`, `description`, `model`.
        """
        ...

    def text_complete(
        self,
        model: str,
        messages: list[dict],
        json_schema: dict | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Non-streaming text generation — query planner, agentic chat turns,
        `is_photo_question` (design §5.1, plan 15 task 4).

        `temperature`/`max_tokens` are the caller's decoding controls, passed straight
        to the backend; `None` leaves the backend default alone."""
        ...

    def text_stream(self, model: str, messages: list[dict]) -> Iterator[str]:
        """Chat answer generation, streamed one token at a time."""
        ...

    def text_embed(self, model: str, texts: list[str]) -> list[list[float]]:
        """Text embeddings over the text-generation model — caption-meaning
        vectors (§9) and any other caller of `InferenceClient.embed`."""
        ...

    def text_warm(self, model: str) -> None:
        """Load a text model without generating (residency, §8.1)."""
        ...

    def text_evict(self, model: str) -> None:
        """Unload a text model immediately (residency, §8.1)."""
        ...

    def resources(self) -> dict:
        """What is resident + memory, for the resource bar (§13).

        Returns a dict with keys `ram_used_mb`, `ram_total_mb`, `cpu_pct`,
        `gpu_pct` (GPU load 0-100, or `None` where no GPU is readable — the
        service is the only process with GPU access, §5.1), and `resident`
        (list of resident model names).
        """
        ...

    def models_state(self) -> dict:
        """Conveyor state for the control API (plan 18): keys `resident`,
        `budget_mb`, `free_mb`, `used_mb`, `active`."""
        ...

    def model_ensure(self, name: str) -> None:
        """Make a managed model resident now (evicting others to fit)."""
        ...

    def model_unload(self, name: str) -> None:
        """Unload a managed model now, freeing its memory."""
        ...
