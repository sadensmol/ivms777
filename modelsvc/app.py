"""The `models` service — the one inference gateway (design §5.1).

`create_models_app(backend)` wires the HTTP surface over any `ModelBackend`.
Task 1 always passes `FakeBackend`; a real backend selected by profile
(SigLIP / caption VLM / Ollama) arrives in later tasks behind the same
protocol, so this module never imports torch/transformers/httpx-to-Ollama —
it only calls `backend`.
"""

import base64
import json
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from modelsvc.backends.base import ModelBackend, SlotOff


class ImageEmbedRequest(BaseModel):
    images: list[str]  # base64-encoded image bytes


class TextEmbedRequest(BaseModel):
    texts: list[str]


class EmbedResponse(BaseModel):
    vectors: list[list[float]]


class TagRequest(BaseModel):
    image: str  # base64
    dimensions: list[str]


class CaptionRequest(BaseModel):
    image: str  # base64


class CaptionResponse(BaseModel):
    caption: str
    title: str
    description: str
    # The ACTUAL model the backend used — the worker stores this, not the config
    # value. The caption model writes no tags; tags are SigLIP-only (design §7).
    model: str


class SpeakRequest(BaseModel):
    text: str
    language: str = "en"
    voice: str | None = None
    # The EXACT chunks to speak, in order. When given, the voice does not choose its own
    # sentence boundaries — it speaks these and reports where each one landed. That is
    # what lets a film start a shot on the sentence about THAT photograph (§11.1): a
    # boundary the caller did not choose is a boundary the caller cannot sync to.
    segments: list[str] | None = None
    # One speaker per segment, same length as `segments`. A viewer's `[voice]` bracket
    # applies to the PASSAGE it was written over and nowhere else (§11.3), so the film
    # is spoken in parts and merged: mostly its own narrator, with a different one for
    # the moments that asked for it. `None`, or a shorter list, falls back to `voice`.
    voices: list[str | None] | None = None


class MusicSection(BaseModel):
    """One stretch of a film's score (design §11.1) — what it sounds like, and for how
    long. Planned by `feed/score.py`."""

    brief: str
    seconds: float = 30.0


class MusicRequest(BaseModel):
    # A whole SCORE, in order. Each section continues the one before it, so the result
    # is one piece that changes rather than several clips. `prompt` is the one-section
    # form and stays for callers that want a single bed.
    sections: list[MusicSection] | None = None
    prompt: str = ""
    seconds: float = 30.0
    seed: int | None = None
    # Decoding controls. Left alone they are MusicGen's own defaults, which is what
    # every film sounded like: the same safe average whatever it was asked for.
    temperature: float | None = None
    guidance: float | None = None


class AudioResponse(BaseModel):
    """A generated clip for a memory feed (design §11.1). `audio` is a base64 WAV —
    base64 rather than a binary body because this is one artifact of a background
    job, not a stream, and the caller stores it whole."""

    audio: str
    sample_rate: int
    model: str
    # WHERE each spoken sentence lands inside the clip, in seconds — `[{text, start,
    # end}]`. The film cuts and titles ON the narration (§11.1), and only the voice
    # child knows how long each sentence took to say, so it has to travel with the
    # audio. Empty for music, and for a voice that cannot report it.
    segments: list[dict] = []


class TextCompleteRequest(BaseModel):
    model: str
    messages: list[dict[str, Any]]
    json_schema: dict[str, Any] | None = None
    # Decoding controls the CALLER owns. A routing/tool-selection call wants temp 0 and
    # a small token cap; a chat answer wants neither. Both are optional so an existing
    # caller that sends neither keeps the backend's defaults.
    temperature: float | None = None
    max_tokens: int | None = None


class TextCompleteResponse(BaseModel):
    text: str


class TextStreamRequest(BaseModel):
    model: str
    messages: list[dict[str, Any]]


class TextGenEmbedRequest(BaseModel):
    model: str
    texts: list[str]


class TextModelRequest(BaseModel):
    model: str


class Preprocess(BaseModel):
    """How the CALLER must resize an image before sending it (design §4.1)."""

    input_px: int
    resample: str
    mode: str  # "squash" | "native"


class EmbedSpecResponse(BaseModel):
    """Calibration AND the selected image embedder's preprocessing contract, in one
    round trip. `generation` moves when a slot is switched, which is the cache
    invalidation signal for `RemoteEmbedder`."""

    logit_scale: float
    logit_bias: float
    preprocess: Preprocess
    generation: int


class SetSlotsRequest(BaseModel):
    slots: dict[str, str]


class DownloadRequest(BaseModel):
    slot: str
    key: str


class ResourcesResponse(BaseModel):
    """Only what this service alone knows (design §5.1).

    Deliberately NO machine metrics: RAM/CPU/GPU-load/temperature are host-wide
    reads that need no CUDA context, so `app` takes them from psutil and sysfs
    directly and the bar survives this service being down.
    """

    resident: list[str]
    active: str | None = None  # current in-flight op for the bar (embedding/captioning/chat/…)
    # Which model each slot holds RIGHT NOW (design §4.1). `app` compares this with
    # its own stored selection on every poll and re-pushes when they diverge — that
    # is how a restarted service (which has no DB, so it boots on the profile
    # defaults) converges back without any shared store.
    slots: dict[str, str] = {}
    generation: int = 0


class ModelsStateResponse(BaseModel):
    """Conveyor state (plan 18) — what the governor holds resident vs its budget,
    plus the slot selection and its `generation` (plan 21): `app` compares that
    generation on its resources poll and re-pushes the stored slots when the
    service has restarted onto the profile defaults (design §4.1)."""

    resident: list[str]
    budget_mb: int
    free_mb: float
    used_mb: int
    active: str | None = None
    slots: dict[str, str] = {}
    generation: int = 0


def _audio_response(result: dict) -> "AudioResponse":
    return AudioResponse(
        audio=base64.b64encode(result["audio"]).decode(),
        sample_rate=result["sample_rate"],
        model=result["model"],
        segments=result.get("segments", []),
    )


def create_models_app(backend: ModelBackend) -> FastAPI:
    app = FastAPI(title="ivms777 models service")

    @app.post("/embed/image", response_model=EmbedResponse)
    def embed_image(req: ImageEmbedRequest) -> EmbedResponse:
        images = [base64.b64decode(image) for image in req.images]
        return EmbedResponse(vectors=backend.embed_image(images))

    @app.post("/embed/text", response_model=EmbedResponse)
    def embed_text(req: TextEmbedRequest) -> EmbedResponse:
        return EmbedResponse(vectors=backend.embed_text(req.texts))

    @app.post("/tag")
    def tag(req: TagRequest) -> dict[str, list[str]]:
        image = base64.b64decode(req.image)
        return backend.tag(image, req.dimensions)

    @app.get("/embed/spec", response_model=EmbedSpecResponse)
    def embed_spec() -> EmbedSpecResponse:
        return EmbedSpecResponse(**backend.embed_spec())

    @app.post("/caption", response_model=CaptionResponse)
    def caption(req: CaptionRequest) -> CaptionResponse:
        image = base64.b64decode(req.image)
        return CaptionResponse(**backend.caption(image))

    @app.post("/speak", response_model=AudioResponse)
    def speak(req: SpeakRequest) -> AudioResponse:
        # A slot the user turned off is a 503; no voice covering the language is the
        # same answer (§4.1). Anything else is a 500 that CARRIES THE MODEL'S OWN
        # MESSAGE: narration is mandatory (§11.1), so this text is shown to the user
        # on the failed film. Losing it is what made "no Russian voice" unreadable.
        try:
            result = backend.speak(req.text, req.language, req.voice, req.segments,
                                   req.voices)
        except SlotOff as exc:
            raise HTTPException(status_code=503, detail=f"voice slot is off: {exc}") from exc
        except Exception as exc:  # reported, never swallowed
            raise HTTPException(status_code=500, detail=str(exc) or type(exc).__name__) from exc
        return _audio_response(result)

    @app.post("/music", response_model=AudioResponse)
    def music(req: MusicRequest) -> AudioResponse:
        if not req.sections and not req.prompt:
            raise HTTPException(status_code=422, detail="music needs a prompt or sections")
        sections = ([s.model_dump() for s in req.sections] if req.sections else None)
        try:
            result = backend.music(req.prompt, req.seconds, req.seed, sections=sections,
                                   temperature=req.temperature, guidance=req.guidance)
        except SlotOff as exc:
            raise HTTPException(status_code=503, detail=f"music slot is off: {exc}") from exc
        return _audio_response(result)

    @app.post("/text/complete", response_model=TextCompleteResponse)
    def text_complete(req: TextCompleteRequest) -> TextCompleteResponse:
        return TextCompleteResponse(
            text=backend.text_complete(
                req.model, req.messages, json_schema=req.json_schema,
                temperature=req.temperature, max_tokens=req.max_tokens,
            )
        )

    @app.post("/text/stream")
    def text_stream(req: TextStreamRequest) -> StreamingResponse:
        def token_stream():
            for token in backend.text_stream(req.model, req.messages):
                yield f"data: {json.dumps({'token': token})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(token_stream(), media_type="text/event-stream")

    @app.get("/text/context")
    def text_context() -> dict:
        return {"context": backend.text_context()}

    @app.post("/text/embed", response_model=EmbedResponse)
    def text_embed(req: TextGenEmbedRequest) -> EmbedResponse:
        return EmbedResponse(vectors=backend.text_embed(req.model, req.texts))

    @app.post("/text/warm")
    def text_warm(req: TextModelRequest) -> dict:
        backend.text_warm(req.model)
        return {}

    @app.post("/text/evict")
    def text_evict(req: TextModelRequest) -> dict:
        backend.text_evict(req.model)
        return {}

    @app.get("/resources", response_model=ResourcesResponse)
    def resources() -> ResourcesResponse:
        return ResourcesResponse(**backend.resources())

    @app.get("/models", response_model=ModelsStateResponse)
    def models_state() -> ModelsStateResponse:
        return ModelsStateResponse(**backend.models_state())

    # --- slot control (design §4.1). Registered BEFORE `/models/{name}/…` so the
    # literal paths are not swallowed by the path parameter.
    @app.get("/models/catalog")
    def models_catalog() -> dict:
        return backend.catalog()

    @app.put("/models/slots")
    def set_slots(req: SetSlotsRequest) -> dict:
        try:
            return backend.set_slots(req.slots)
        except (ValueError, KeyError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/models/download")
    def download(req: DownloadRequest) -> dict:
        try:
            return backend.download(req.slot, req.key)
        except (ValueError, KeyError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/models/{name}/ensure")
    def model_ensure(name: str) -> dict:
        backend.model_ensure(name)
        return {}

    @app.post("/models/{name}/unload")
    def model_unload(name: str) -> dict:
        backend.model_unload(name)
        return {}

    return app
