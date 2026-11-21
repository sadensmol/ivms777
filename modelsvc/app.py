"""The `models` service — the one inference gateway (design §5.1).

`create_models_app(backend)` wires the HTTP surface over any `ModelBackend`.
Task 1 always passes `FakeBackend`; a real backend selected by profile
(SigLIP / caption VLM / Ollama) arrives in later tasks behind the same
protocol, so this module never imports torch/transformers/httpx-to-Ollama —
it only calls `backend`.
"""

import base64
import binascii
import hashlib
import json
from typing import Any

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt

from audio.whisper_asr import MAX_SECONDS as TRANSCRIBE_MAX_SECONDS
from embedding.face_model import WEIGHTS_MISSING as FACE_WEIGHTS_MISSING
from modelsvc.backends.base import (
    CloningUnavailable,
    FaceImageRefused,
    FaceNotOffered,
    ModelBackend,
    SlotOff,
    SpeakerEmbedNotOffered,
    SpeakerEmbedRefused,
    TranscribeNotOffered,
    TranscribeRefused,
)
from modelsvc.governor import UnitLeased
from modelsvc.voice_bakes import (
    MAX_CLIP_BYTES,
    BakeNotOffered,
    BakeOnly,
    BakeRefused,
    BakeRunning,
    _wav_seconds,
)


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


# The limits of one saved-voice request (§2.3). A film casts a narrator and a few
# bracket voices; an example is a 5-30 s mono PCM-16 clip at 48 kHz (< 3 MiB).
MAX_REFERENCES = 8
MAX_REFERENCE_BYTES = 4 * 1024 * 1024
# References with a baked voice model, per request; the estimate of one is ≈ 18 MB.
MAX_ADAPTERS = 4
MAX_ADAPTER_BYTES = 64 * 1024 * 1024


class VoiceConditioning(BaseModel):
    """What a cloning model built from one example, stored by `app` and sent back so the
    child can skip the encode. Valid only for the model + revision that made it."""

    model: str
    revision: str
    blob: str             # base64; opaque to everyone except the child that made it


class VoiceAdapter(BaseModel):
    """A saved voice's baked model (§5.1 `/voice/bakes`)."""

    blob: str             # base64 of the safetensors file a bake wrote
    sha256: str           # of the decoded bytes


class VoiceControls(BaseModel):
    """Generation settings of one reference, for `voice_bake_check.py --sweep` only (§5.1).
    Forbidding extras matters: an ignored field would speak at the fixed values."""

    model_config = ConfigDict(extra="forbid")

    cfg_value: StrictFloat = Field(ge=1.5, le=3.0)
    inference_timesteps: StrictInt = Field(ge=10, le=20)


class VoiceReference(BaseModel):
    """ONE saved example of a saved voice (§11.1), or its baked model alone (D76)."""

    # Both are required unless `use_clip` is false: the adapter alone needs no take.
    audio: str | None = None   # base64 PCM-16 WAV
    sha256: str | None = None  # hex sha256 of the decoded WAV bytes
    language: str
    # The heard words of the take. VoxCPM2 clones from the clip AND these words when
    # present, which keeps the person's pace; without them, timbre only.
    transcript: str | None = None
    conditioning: list[VoiceConditioning] = []
    adapter: VoiceAdapter | None = None
    # False speaks from the adapter alone, with no prompt take; only with an adapter.
    use_clip: bool = True
    controls: VoiceControls | None = None


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
    # SAVED voices (§11.1): voice token ("saved:<voice_id>") -> its example. When given,
    # `voice` and every named passage voice must be a token of this map — one film never
    # mixes saved voices and preset speakers (D17). `None` = preset speakers, as before.
    references: dict[str, VoiceReference] | None = None


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
    # The Hub commit the voice ran at. None for Kokoro, Silero and music.
    revision: str | None = None
    # Saved voices only: token -> {state: "built"|"restored", model, revision, reason,
    # blob: base64 | None}. A `built` blob is new; `restored` has none.
    conditioning: dict[str, dict] = {}
    # Saved voices only: token -> {state: "loaded"|"kept"|"none", sha256}, the state of
    # the token's last run.
    adapters: dict[str, dict] = {}
    # Only tokens that set `controls`: token -> {cfg_value, inference_timesteps}, the values
    # the child spoke with.
    controls: dict[str, dict] = {}


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


class BakeClip(BaseModel):
    audio: str            # base64 mono PCM-16 WAV at 48000 Hz
    sha256: str
    text: str
    language: str


class BakeRequest(BaseModel):
    # The entry that bakes on this profile when left out; any other key is a 422.
    model: str | None = None
    clips: list[BakeClip]
    config: dict[str, Any] | None = None
    tag: dict[str, Any] | None = None


class TranscribeRequest(BaseModel):
    audio: str            # base64 mono PCM-16 WAV at 48000 Hz, at most 30 s: one take
    sha256: str
    language: str         # the take's own label; it forces the decoding language


class TranscribeResponse(BaseModel):
    text: str             # as the model decoded it; `app` alone judges the words
    model: str
    revision: str
    seconds: float
    elapsed_s: float


class VoiceEmbedRequest(BaseModel):
    audio: str            # base64 mono PCM-16 WAV at 48000 Hz: one take or one test
    sha256: str


class VoiceEmbedResponse(BaseModel):
    vector: list[float]   # raw; `app` normalises and compares
    model: str
    revision: str
    seconds: float
    elapsed_s: float
    pipeline: int


class FaceDetectRequest(BaseModel):
    image: str            # base64 of one upright image, shrunk to the entry's input edge


class DetectedFace(BaseModel):
    box: list[float]      # [x, y, w, h], each in 0..1 of the image
    score: float
    vector: list[float]   # L2-normalised, the entry's `dim` values


class FaceDetectResponse(BaseModel):
    faces: list[DetectedFace]
    model: str
    revision: str
    elapsed_s: float


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
    conditioning = {
        token: entry | {"blob": None if entry.get("blob") is None  # pragma: allowlist secret
                        else base64.b64encode(entry["blob"]).decode()}
        for token, entry in (result.get("conditioning") or {}).items()
    }
    return AudioResponse(
        audio=base64.b64encode(result["audio"]).decode(),
        sample_rate=result["sample_rate"],
        model=result["model"],
        segments=result.get("segments", []),
        revision=result.get("revision"),
        conditioning=conditioning,
        adapters=result.get("adapters", {}),
        controls=result.get("controls", {}),
    )


def _b64(value: str, what: str) -> bytes:
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"{what} is not valid base64") from exc


def _decode_references(req: SpeakRequest) -> dict:
    """The request's saved-voice examples as the backend takes them (bytes, not
    base64), after every check of §2.3. Raises `ValueError` with the reason."""
    references = req.references or {}
    if len(references) > MAX_REFERENCES:
        raise ValueError(f"at most {MAX_REFERENCES} references, got {len(references)}")
    # The narrator itself must be a key; only the per-passage names may be empty.
    if not req.voice or req.voice not in references:
        raise ValueError(f"the voice {req.voice!r} must be one of the references")
    for name in req.voices or []:
        if name and name not in references:
            raise ValueError(f"voice {name!r} has no reference: a request never mixes "
                             "saved voices and preset speakers")
    adapters = sum(ref.adapter is not None for ref in references.values())
    if adapters > MAX_ADAPTERS:
        raise ValueError(f"at most {MAX_ADAPTERS} references with an adapter, got {adapters}")
    decoded = {}
    for token, ref in references.items():
        if not ref.use_clip and ref.adapter is None:
            raise ValueError(f"use_clip=false needs an adapter, and {token!r} has none")
        if ref.audio is None and not ref.use_clip:
            # The adapter alone (D76): no take, so no audio and no conditioning.
            decoded[token] = {"language": ref.language,
                              "adapter": _adapter(ref.adapter, token), "use_clip": False}
            if ref.controls is not None:
                decoded[token]["controls"] = ref.controls.model_dump()
            continue
        # Audio sent with use_clip=false is still checked, and the child does not use it.
        if ref.audio is None or ref.sha256 is None:
            raise ValueError(f"the reference of {token!r} needs its audio")
        audio = _b64(ref.audio, f"the audio of {token!r}")
        if len(audio) > MAX_REFERENCE_BYTES:
            raise ValueError(f"the audio of {token!r} is {len(audio)} bytes; "
                             f"the limit is {MAX_REFERENCE_BYTES}")
        if hashlib.sha256(audio).hexdigest() != ref.sha256.lower():
            raise ValueError(f"the audio of {token!r} does not match its sha256")
        decoded[token] = {
            "audio": audio,
            "sha256": ref.sha256,
            "language": ref.language,
            "transcript": ref.transcript,
            "conditioning": [
                {"model": c.model, "revision": c.revision,
                 "blob": _b64(c.blob, f"a conditioning blob of {token!r}")}
                for c in ref.conditioning
            ],
        }
        # Only a reference with an adapter carries the two keys, so a plain one reaches
        # the voice child exactly as before.
        if ref.adapter is not None:
            decoded[token] |= {"adapter": _adapter(ref.adapter, token), "use_clip": ref.use_clip}
        if ref.controls is not None:
            decoded[token]["controls"] = ref.controls.model_dump()
    return decoded


def _adapter(adapter: VoiceAdapter, token: str) -> dict:
    blob = _b64(adapter.blob, f"the adapter of {token!r}")
    if len(blob) > MAX_ADAPTER_BYTES:
        raise ValueError(f"the adapter of {token!r} is {len(blob)} bytes; "
                         f"the limit is {MAX_ADAPTER_BYTES}")
    if hashlib.sha256(blob).hexdigest() != adapter.sha256.lower():
        raise ValueError(f"the adapter of {token!r} does not match its sha256")
    return {"blob": blob, "sha256": adapter.sha256}


def _decode_take(req: TranscribeRequest) -> bytes:
    """The take of a transcribe request, after every check of §5.1, before any load.
    Raises `ValueError` with the reason."""
    audio = _b64(req.audio, "the take")
    if len(audio) > MAX_CLIP_BYTES:
        raise ValueError(f"the take is {len(audio)} bytes; the limit is {MAX_CLIP_BYTES}")
    if hashlib.sha256(audio).hexdigest() != req.sha256.lower():
        raise ValueError("the take does not match its sha256")
    try:
        seconds = _wav_seconds(audio, 0)
    except BakeRefused as exc:  # the `/voice/bakes` take checks, named for one take
        raise ValueError(str(exc).replace("take 0:", "the take:", 1)) from exc
    if seconds > TRANSCRIBE_MAX_SECONDS:
        raise ValueError(f"the take is {seconds:.1f} s long; "
                         f"the limit is {TRANSCRIBE_MAX_SECONDS:.0f} s")
    return audio


def _decode_clip(req: VoiceEmbedRequest) -> bytes:
    """The clip of an embed request, after every check of §5.1, before any load. It has
    no 30 s cap, unlike a take to hear. Raises `ValueError` with the reason."""
    audio = _b64(req.audio, "the clip")
    if len(audio) > MAX_CLIP_BYTES:
        raise ValueError(f"the clip is {len(audio)} bytes; the limit is {MAX_CLIP_BYTES}")
    if hashlib.sha256(audio).hexdigest() != req.sha256.lower():
        raise ValueError("the clip does not match its sha256")
    try:
        _wav_seconds(audio, 0)
    except BakeRefused as exc:  # the `/voice/bakes` take checks, named for one clip
        raise ValueError(str(exc).replace("take 0:", "the clip:", 1)) from exc
    return audio


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
        # A malformed saved-voice request is the caller's error: 422 with the reason.
        args = (req.text, req.language, req.voice, req.segments, req.voices)
        try:
            references = None if req.references is None else _decode_references(req)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        try:
            # `references=` only when the request has them, so a preset request reaches
            # the backend exactly as it always did.
            result = (backend.speak(*args) if references is None
                      else backend.speak(*args, references=references))
        except CloningUnavailable as exc:
            raise HTTPException(status_code=503,
                                detail=f"voice copy is not available: {exc}") from exc
        except SlotOff as exc:
            raise HTTPException(status_code=503, detail=f"voice slot is off: {exc}") from exc
        except BakeRunning as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
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
        try:
            backend.model_ensure(name)
        except (BakeRunning, BakeOnly) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {}

    @app.post("/models/{name}/unload")
    def model_unload(name: str) -> dict:
        try:
            backend.model_unload(name)
        except UnitLeased as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {}

    # --- voice bakes (§5.1). `current` is registered before `{bake_id}`.
    @app.post("/voice/bakes", status_code=202)
    def bake_start(req: BakeRequest) -> dict:
        try:
            return backend.bake_start(req.model_dump())
        except BakeRunning as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except BakeRefused as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except BakeNotOffered as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/voice/bakes/current")
    def bake_current() -> dict:
        return _found(backend.bake_current())

    @app.get("/voice/bakes/{bake_id}")
    def bake_status(bake_id: str) -> dict:
        return _found(backend.bake_status(bake_id))

    @app.post("/voice/bakes/{bake_id}/cancel")
    def bake_cancel(bake_id: str) -> dict:
        return _found(backend.bake_cancel(bake_id))

    @app.get("/voice/bakes/{bake_id}/adapter")
    def bake_adapter(bake_id: str) -> Response:
        try:
            blob = backend.bake_adapter(bake_id)
        except BakeRunning as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return Response(content=_found(blob), media_type="application/octet-stream")

    @app.delete("/voice/bakes/{bake_id}", status_code=204)
    def bake_delete(bake_id: str) -> Response:
        try:
            deleted = backend.bake_delete(bake_id)
        except BakeRunning as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if not deleted:
            raise HTTPException(status_code=404, detail="no such bake")
        return Response(status_code=204)

    @app.post("/voice/transcribe", response_model=TranscribeResponse)
    def transcribe(req: TranscribeRequest) -> TranscribeResponse:
        try:
            audio = _decode_take(req)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        try:
            result = backend.transcribe(audio, req.language)
        except TranscribeNotOffered as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except TranscribeRefused as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except BakeRunning as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:  # a load or model error, with its own message
            raise HTTPException(status_code=500, detail=str(exc) or type(exc).__name__) from exc
        return TranscribeResponse(**result)

    @app.post("/voice/embed", response_model=VoiceEmbedResponse)
    def voice_embed(req: VoiceEmbedRequest) -> VoiceEmbedResponse:
        try:
            audio = _decode_clip(req)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        try:
            result = backend.embed_voice(audio)
        except SpeakerEmbedNotOffered as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except SpeakerEmbedRefused as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except BakeRunning as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:  # a load or model error, with its own message
            raise HTTPException(status_code=500, detail=str(exc) or type(exc).__name__) from exc
        return VoiceEmbedResponse(**result)

    @app.post("/faces/detect", response_model=FaceDetectResponse)
    def faces_detect(req: FaceDetectRequest) -> FaceDetectResponse:
        try:
            image = _b64(req.image, "the image")
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        try:
            result = backend.detect_faces(image)
        except FaceNotOffered as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except FaceImageRefused as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:  # a load or model error, with its own message
            status = 503 if FACE_WEIGHTS_MISSING in str(exc) else 500
            raise HTTPException(status_code=status, detail=str(exc) or type(exc).__name__) from exc
        return FaceDetectResponse(**result)

    return app


def _found(value):
    if value is None:
        raise HTTPException(status_code=404, detail="no such bake")
    return value
