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

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from modelsvc.backends.base import ModelBackend


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


class CalibrationResponse(BaseModel):
    logit_scale: float
    logit_bias: float


class ResourcesResponse(BaseModel):
    """Only what this service alone knows (design §5.1).

    Deliberately NO machine metrics: RAM/CPU/GPU-load/temperature are host-wide
    reads that need no CUDA context, so `app` takes them from psutil and sysfs
    directly and the bar survives this service being down.
    """

    resident: list[str]
    active: str | None = None  # current in-flight op for the bar (embedding/captioning/chat/…)


class ModelsStateResponse(BaseModel):
    """Conveyor state (plan 18) — what the governor holds resident vs its budget."""

    resident: list[str]
    budget_mb: int
    free_mb: float
    used_mb: int
    active: str | None = None


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

    @app.get("/embed/calibration", response_model=CalibrationResponse)
    def calibration() -> CalibrationResponse:
        return CalibrationResponse(**backend.calibration())

    @app.post("/caption", response_model=CaptionResponse)
    def caption(req: CaptionRequest) -> CaptionResponse:
        image = base64.b64decode(req.image)
        return CaptionResponse(**backend.caption(image))

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

    @app.post("/models/{name}/ensure")
    def model_ensure(name: str) -> dict:
        backend.model_ensure(name)
        return {}

    @app.post("/models/{name}/unload")
    def model_unload(name: str) -> dict:
        backend.model_unload(name)
        return {}

    return app
