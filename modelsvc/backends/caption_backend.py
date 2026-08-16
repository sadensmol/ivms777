"""`CaptionBackend` — the `/caption` sub-backend (design §5.1, plan 15 task 3).

Wraps whichever `Captioner` (`captioning/`) the profile selects — `modelsvc`
is the ONE process allowed to actually load and run it (mac/cloud →
`OllamaCaptioner`, jetson → the in-process `VLMCaptioner`). This module (and
constructing a `VLMCaptioner`) is torch-free; only `.load()` — called lazily,
on first use — pulls torch/transformers.

Residency (plan 15 task 5, §8.1): when a `residency` is injected, `caption()`
registers the captioner's `load`/`release` with it and runs under
`residency.use("caption", priority=LOW)`, so an interactive embed can preempt
an in-flight caption — the captioner's `should_preempt` is wired to
`residency.should_preempt`. A preempted captioner raises
`inference.client.InferenceCancelled`, mapped here to `CaptionPreempted`
(caught by the `/caption` route -> HTTP 503, Deliverable 3). Without a
`residency` (e.g. `FakeBackend`'s path, or a bare unit test), behavior is
exactly the pre-Task-5 once-and-done load guard, no preemption possible.
"""

from captioning.base import Captioner
from inference.client import InferenceCancelled
from modelsvc.residency import LOW, CaptionPreempted, Residency


class CaptionBackend:
    def __init__(self, captioner: Captioner, residency: Residency | None = None) -> None:
        self._captioner = captioner
        self._loaded = False
        self._residency = residency
        if residency is not None:
            residency.register("caption", captioner.load, captioner.release)

    def caption(self, image: bytes, dimensions: list[str]) -> dict:
        if self._residency is None:
            if not self._loaded:
                self._captioner.load()
                self._loaded = True
            result = self._captioner.caption(image, dimensions)
        else:
            with self._residency.use("caption", priority=LOW):
                try:
                    result = self._captioner.caption(
                        image, dimensions, should_preempt=self._residency.should_preempt
                    )
                except InferenceCancelled as exc:
                    raise CaptionPreempted() from exc
        return {
            "caption": result.caption,
            "title": result.title,
            "description": result.description,
            "tags": result.tags,
            "model": self._captioner.caption_model,
        }
