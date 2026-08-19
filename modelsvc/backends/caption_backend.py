"""`CaptionBackend` — the `/caption` sub-backend (design §5.1).

Adapts a `captioning.Captioner` (`OpenAICaptioner`) to the `ModelBackend`
caption dict. Since plan 16, captioning is a plain OpenAI `/v1/chat/completions`
call to llama-server (mac/jetson) / vLLM (cloud) — the model is server-resident,
so there is nothing to load, evict, or preempt here. It is NOT a residency
concern: SigLIP is the only heavy in-process model left (design §8.1).
"""

from collections.abc import Callable

from captioning.base import Captioner


class CaptionBackend:
    def __init__(self, captioner: Callable[[], Captioner]) -> None:
        # A PROVIDER of the captioner, not a captioner: the `caption` slot is
        # switchable, and every caption must be produced — and STAMPED — by the model
        # the slot holds RIGHT NOW (design §4.1). A captioner captured at build time
        # is the profile default forever, so a switched-in model got the default's
        # prompt template and every photo was stamped with the default's name.
        self._captioner = captioner

    def caption(self, image: bytes) -> dict:
        captioner = self._captioner()
        result = captioner.caption(image)
        return {
            "caption": result.caption,
            "title": result.title,
            "description": result.description,
            "model": captioner.caption_model,
        }
