"""Captioner protocol and result type (design §4).

`Captioner` is the seam between the caption pipeline stage and whatever
actually produces a caption. Since plan 16 there is one implementation —
`OpenAICaptioner` over an OpenAI-compatible backend (llama-server / vLLM) — but
the protocol stays so the `models` service's `CaptionBackend` is written against
it, not a concrete class.
"""

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

    def caption(self, image: bytes, dimensions: list[str]) -> CaptionResult: ...
