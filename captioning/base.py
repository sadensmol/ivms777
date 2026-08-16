"""Captioner protocol and result type (design §4).

`Captioner` is the seam between the caption pipeline stage and whatever
actually produces a caption — today's Ollama/OpenAI-compatible backend, or
(later) an in-process Jetson VLM. Both implementations return the exact same
`CaptionResult` shape.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class CaptionResult:
    caption: str
    title: str
    description: str
    tags: dict[str, list[str]]


class Captioner(Protocol):
    name: str
    caption_model: str

    def caption(self, image: bytes, dimensions: list[str], *,
                should_preempt: Callable[[], bool] = lambda: False) -> CaptionResult: ...

    def load(self) -> None: ...

    def release(self) -> None: ...

    def footprint_mb(self) -> int: ...
