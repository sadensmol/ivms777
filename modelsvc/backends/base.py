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


class CloningUnavailable(SlotOff):
    """No voice-copy model speaks the language on this profile (§4.1, §11.1).

    A `SlotOff`, so every caller that treats "no voice" as a 503 keeps doing so; the
    HTTP layer gives it its own wording, because the film shows it to the user.
    """


class TranscribeNotOffered(RuntimeError):
    """No speech-recognition model runs on this profile: the service has no `asr`
    unit (`catalog.transcriber_for`, §4.1). The HTTP layer answers 503."""


class TranscribeRefused(ValueError):
    """A transcribe request the ASR entry cannot take (a language it does not list).
    The HTTP layer answers 422 with the reason."""


class SpeakerEmbedNotOffered(RuntimeError):
    """No speaker model runs on this profile: the service has no `speaker_embed` unit
    (`catalog.speaker_embedder_for`, §4.1). The HTTP layer answers 503."""


class SpeakerEmbedRefused(ValueError):
    """A clip the speaker model cannot score (too little voiced audio). The HTTP layer
    answers 422 with the reason."""


class FaceNotOffered(RuntimeError):
    """No face model runs on this profile: the service has no `faces` unit
    (`catalog.face_model_for`, §4.1). The HTTP layer answers 503."""


class FaceImageRefused(ValueError):
    """An image the face model cannot read. The HTTP layer answers 422 with the
    reason."""


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
              voices: list[str | None] | None = None,
              references: dict | None = None) -> dict:
        """One narration clip for a memory film (design §11.1).

        Returns `{audio: bytes (WAV), sample_rate: int, model: str}`. Raises `SlotOff`
        when the `voice` slot holds `off`, or when the selected voice does not list
        `language` in its catalog `languages` (§4.1) — the film is then prepared
        without speech rather than narrated in the wrong language.

        `references` speaks SAVED voices: `{token: {audio: bytes, sha256, language,  # pragma: allowlist secret
        transcript, conditioning: [{model, revision, blob: bytes}]}}`. It routes to the
        cloning entry for `language` and raises `CloningUnavailable` when there is none.
        The reply then also carries `revision` and `conditioning`
        (`{token: {state, model, revision, reason, blob: bytes | None}}`).  # pragma: allowlist secret
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
        """Make a managed model resident now (evicting others to fit).

        `voice`, `asr` and `speaker_embed` are counted in at the voice gate like a speak
        (`BakeRunning` while a bake runs); `voice_train` always raises `BakeOnly` (only a
        bake loads it)."""
        ...

    def model_unload(self, name: str) -> None:
        """Unload a managed model now, freeing its memory. A leased unit raises
        `UnitLeased`."""
        ...

    def bake_start(self, request: dict) -> dict:
        """Start baking a saved voice into a LoRA adapter (design §5.1 `/voice/bakes`)
        and return its status at once. Raises `BakeRunning`, `BakeRefused` (with the
        reason) or `BakeNotOffered`."""
        ...

    def bake_status(self, bake_id: str) -> dict | None:
        """The bake's status, or None for an unknown id."""
        ...

    def bake_current(self) -> dict | None:
        """The running bake, else the last ended one not yet deleted, else None."""
        ...

    def bake_cancel(self, bake_id: str) -> dict | None:
        """Ask the bake to stop; its status, or None for an unknown id."""
        ...

    def bake_adapter(self, bake_id: str) -> bytes | None:
        """The safetensors file of a `done` bake (`BakeRunning` before that), or None
        for an unknown id."""
        ...

    def bake_delete(self, bake_id: str) -> bool:
        """Remove an ended bake and its work folder; False for an unknown id,
        `BakeRunning` while it runs."""
        ...

    def transcribe(self, audio: bytes, language: str) -> dict:
        """The words of one take, as the bake's speech-recognition model decoded them
        (design §5.1 `/voice/transcribe`). `audio` is a checked mono PCM-16 WAV at
        48000 Hz of at most 30 s; `language` forces the decoding language.

        Returns `{text, model, revision, seconds, elapsed_s}`. Raises
        `TranscribeNotOffered` (no ASR on this profile), `TranscribeRefused` (a
        language the entry does not list) or `BakeRunning` (the voice gate is closed).
        """
        ...

    def embed_voice(self, audio: bytes) -> dict:
        """The speaker embedding of one clip (design §5.1 `/voice/embed`). `audio` is a
        checked mono PCM-16 WAV.

        Returns `{vector, model, revision, seconds, elapsed_s, pipeline}`. Raises
        `SpeakerEmbedNotOffered` (no speaker model on this profile), `SpeakerEmbedRefused`
        (too little voiced audio) or `BakeRunning` (the voice gate is closed).
        """
        ...

    def detect_faces(self, image: bytes) -> dict:
        """The faces of one upright image (design §5.1 `/faces/detect`).

        Returns `{faces: [{box: [x, y, w, h] in 0..1 of the image, score, vector}],
        model, revision, elapsed_s}`; every vector is L2-normalised and has the entry's
        `dim` values. Raises `FaceNotOffered` (no face model on this profile) or
        `FaceImageRefused` (an image it cannot read).
        """
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
