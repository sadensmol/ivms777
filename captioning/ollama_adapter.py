"""Captioning over the OpenAI-compatible inference backend (Ollama on mac/cloud).

Exactly today's caption path, wrapped as a `Captioner` adapter (design §4) —
mac/cloud behavior is unchanged.
"""

import json
from collections.abc import Callable

from captioning.base import CaptionResult
from inference.client import encode_image
from inference.prompts import CAPTION_SCHEMA, caption_messages

# Resident-MB estimates (design §8.1, relocated from the deleted
# `models/workloads.py`, plan 15 task 7 — this adapter is the sole remaining
# consumer). qwen2.5vl:3b ~3.3 GB; qwen2.5vl:7b ~6 GB.
_FOOTPRINT_MB: dict[str, int] = {"qwen2.5:3b": 2200, "qwen2.5vl:3b": 3300, "qwen2.5vl:7b": 6000}
_FALLBACK_LLM_MB = 3000  # unknown LLM tag → conservative estimate


class OllamaCaptioner:
    """Captioning over the OpenAI-compatible inference backend (Ollama on mac/cloud).
    Exactly today's caption path, wrapped as an adapter (design §4)."""

    def __init__(self, client, model: str, footprint_mb: int | None = None):
        self._client = client
        self.name = model
        self.caption_model = model
        self._footprint = footprint_mb

    def caption(
        self,
        image: bytes,
        dimensions: list[str],
        *,
        should_preempt: Callable[[], bool] = lambda: False,
    ) -> CaptionResult:
        msgs = caption_messages(self.caption_model, encode_image(image), dimensions)
        raw = self._client.complete(
            self.caption_model, msgs, json_schema=CAPTION_SCHEMA, should_stop=should_preempt
        )  # InferenceCancelled propagates; the caption stage maps it to Preempted
        obj = json.loads(raw)
        return CaptionResult(obj["caption"], obj["title"], obj["description"], obj.get("tags") or {})

    def load(self) -> None:
        self._client.warm(self.caption_model)

    def release(self) -> None:
        self._client.evict(self.caption_model)

    def footprint_mb(self) -> int:
        if self._footprint is not None:
            return self._footprint
        return _FOOTPRINT_MB.get(self.caption_model, _FALLBACK_LLM_MB)
