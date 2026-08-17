"""`CaptionBackend` — the `/caption` sub-backend (design §5.1).

Adapts a `captioning.Captioner` (`OpenAICaptioner`) to the `ModelBackend`
caption dict. Since plan 16, captioning is a plain OpenAI `/v1/chat/completions`
call to llama-server (mac/jetson) / vLLM (cloud) — the model is server-resident,
so there is nothing to load, evict, or preempt here. It is NOT a residency
concern: SigLIP is the only heavy in-process model left (design §8.1).
"""

from captioning.base import Captioner


class CaptionBackend:
    def __init__(self, captioner: Captioner) -> None:
        self._captioner = captioner

    def caption(self, image: bytes) -> dict:
        result = self._captioner.caption(image)
        return {
            "caption": result.caption,
            "title": result.title,
            "description": result.description,
            "model": self._captioner.caption_model,
        }
