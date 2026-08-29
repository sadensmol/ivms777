"""The `ModelBackend` protocol (design §5.1) — every model, LLM, embedder, and
heavy AI library the `models` service can be configured with implements this.

`modelsvc.app` never talks to torch/transformers/Ollama directly; it calls
whichever `ModelBackend` it was built with. Task 1 ships only `FakeBackend`
(no torch); the real SigLIP/VLM/Ollama-backed implementations arrive in later
tasks behind this same protocol, so the HTTP surface never changes.
"""

from collections.abc import Iterator
from typing import Protocol


class SlotOff(RuntimeError):
    """The requested slot holds the catalog's `off` entry (design §4.1).

    A distinct exception, not `NotImplementedError`: "the user turned this off" is a
    normal, expected answer that the HTTP layer turns into a 503 and the feed builder
    turns into a skipped artifact (§11.1). A wiring bug must not look the same.
    """


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

        Reached over HTTP as part of `embed_spec()`; taxonomy tagging (§9)
        computes probabilities client-side from it.
        """
        ...

    def embed_spec(self) -> dict:
        """`calibration()` plus the SELECTED image embedder's preprocessing
        contract and the slots' `generation` (design §4.1, §5.1).

        Keys: `logit_scale`, `logit_bias`, `preprocess`
        (`input_px`/`resample`/`mode`), `generation`. `RemoteEmbedder` caches it
        and resizes by it, so no caller holds a resolution constant.
        """
        ...

    def caption(self, image: bytes) -> dict:
        """A caption sentence for one image (no tags — tags are SigLIP-only, §7).

        Returns a dict with keys `caption`, `title`, `description`, `model`.
        """
        ...

    def speak(self, text: str, language: str = "en", voice: str | None = None,
              segments: list[str] | None = None,
              voices: list[str | None] | None = None) -> dict:
        """One narration clip for a memory film (design §11.1).

        Returns `{audio: bytes (WAV), sample_rate: int, model: str}`. Raises `SlotOff`
        when the `voice` slot holds `off`, or when the selected voice does not list
        `language` in its catalog `languages` (§4.1) — the film is then prepared
        without speech rather than narrated in the wrong language.
        """
        ...

    def music(
        self, prompt: str = "", seconds: float = 30.0, seed: int | None = None,
        sections: list[dict] | None = None, temperature: float | None = None,
        guidance: float | None = None,
    ) -> dict:
        """The score for a memory feed (design §11.1).

        `sections` is the whole score — `[{"brief", "seconds"}]`, each generated as a
        continuation of the one before it, so it is one piece that changes rather than
        one clip looped. A bare `prompt` is the one-section form. `segments` in the
        reply says where each section actually landed.

        `seconds` is CLAMPED by the backend to the model's ceiling (47 s for Stable
        Audio Open 1.0); the player loops the clip to cover a longer memory. Returns
        `{audio: bytes (WAV), sample_rate: int, model: str}`. Raises `SlotOff` when
        the `music` slot holds `off`.
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

    def text_context(self) -> int | None:
        """How many tokens the text server accepts (design §5.1, §11.1).

        Asked of the RUNNING server, so a caller that assembles a large prompt can cut
        it to fit instead of having the whole call rejected. `None` when no text
        backend can say."""
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

    def catalog(self) -> dict:
        """The repo catalog joined with live state (design §4.1): every entry this
        profile can offer, which one each slot holds, and each one's download
        state. Keys: `profile`, `slots`, `generation`, `entries`."""
        ...

    def set_slots(self, slots: dict) -> dict:
        """Point one or more slots at a different catalog entry. Evicts what it
        replaces; loads nothing eagerly. Raises `ValueError` for a key that is
        unknown or not offered on this profile. Returns `{slots, generation}`."""
        ...

    def download(self, slot: str, key: str) -> dict:
        """Start fetching a catalog entry's weights in the background and return
        its download status (`state`/`bytes`/`total`/`error`). Idempotent."""
        ...
