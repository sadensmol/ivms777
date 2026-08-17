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
    dimensions: list[str] = []


class CaptionResponse(BaseModel):
    caption: str
    title: str
    description: str
    tags: dict[str, list[str]]
    # The ACTUAL model the backend used (jetson's in-process VLM tag differs from
    # `settings.caption_model`) — the worker stores this, not the config value.
    model: str


class TextCompleteRequest(BaseModel):
    model: str
    messages: list[dict[str, Any]]
    json_schema: dict[str, Any] | None = None


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
    ram_used_mb: float
    ram_total_mb: float
    cpu_pct: float
    gpu_pct: float | None = None
    resident: list[str]
    active: str | None = None  # current in-flight op for the bar (embedding/captioning/chat/…)


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
        return CaptionResponse(**backend.caption(image, req.dimensions))

    @app.post("/text/complete", response_model=TextCompleteResponse)
    def text_complete(req: TextCompleteRequest) -> TextCompleteResponse:
        return TextCompleteResponse(
            text=backend.text_complete(req.model, req.messages, json_schema=req.json_schema)
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

    return app
