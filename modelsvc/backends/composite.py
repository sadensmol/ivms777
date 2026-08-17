"""`CompositeBackend` — a `ModelBackend` assembled from independent
sub-backends, one per concern (design §5.1, plan 15 task 2).

Task 2 wires only `embed` (SigLIP). Task 3 slots in `caption`, Task 4 slots
in `text` — each without touching the others. A sub-backend not yet wired
raises `NotImplementedError` instead of silently returning nonsense, so a
premature call fails loudly at the call site.

Task 5 (§8.1) adds an optional `residency`: when given, `resources()` reports
`residency.resident()` — what is ACTUALLY loaded — instead of the Task-2
placeholder (`["siglip"]` whenever an embed sub-backend is wired).
"""

from collections.abc import Iterator

import psutil

from modelsvc.activity import Activity
from modelsvc.gpu import read_gpu_pct


class CompositeBackend:
    def __init__(self, *, embed, caption=None, text=None, residency=None) -> None:
        self._embed = embed
        self._caption = caption
        self._text = text
        self._residency = residency
        # The current in-flight op label, for the resource bar (§13). The service
        # is the one place that sees every call, so "what is the box doing now" is
        # tracked HERE — it replaces the removed cross-process lease workload.
        self._activity = Activity()

    def embed_image(self, images: list[bytes]) -> list[list[float]]:
        with self._activity.track("embedding"):
            return self._embed.embed_image(images)

    def embed_text(self, texts: list[str]) -> list[list[float]]:
        with self._activity.track("embedding"):
            return self._embed.embed_text(texts)

    def tag(self, image: bytes, dimensions: list[str]) -> dict[str, list[str]]:
        with self._activity.track("tagging"):
            return self._embed.tag(image, dimensions)

    def calibration(self) -> dict:
        return self._embed.calibration()

    def caption(self, image: bytes, dimensions: list[str]) -> dict:
        if self._caption is None:
            raise NotImplementedError("caption backend wired in Task 3")
        with self._activity.track("captioning"):
            return self._caption.caption(image, dimensions)

    def text_complete(
        self, model: str, messages: list[dict], json_schema: dict | None = None
    ) -> str:
        if self._text is None:
            raise NotImplementedError("text backend wired in Task 4")
        with self._activity.track("planning"):
            return self._text.text_complete(model, messages, json_schema=json_schema)

    def text_stream(self, model: str, messages: list[dict]) -> Iterator[str]:
        if self._text is None:
            raise NotImplementedError("text backend wired in Task 4")
        # A generator: keep the activity marked for the whole stream, not just
        # until the generator object is built — so the bar shows "chat" while the
        # answer streams, then clears when the last token is consumed.
        def stream() -> Iterator[str]:
            with self._activity.track("chat"):
                yield from self._text.text_stream(model, messages)

        return stream()

    def text_embed(self, model: str, texts: list[str]) -> list[list[float]]:
        if self._text is None:
            raise NotImplementedError("text backend wired in Task 4")
        with self._activity.track("embedding"):
            return self._text.text_embed(model, texts)

    def text_warm(self, model: str) -> None:
        if self._text is None:
            raise NotImplementedError("text backend wired in Task 4")
        self._text.text_warm(model)

    def text_evict(self, model: str) -> None:
        if self._text is None:
            raise NotImplementedError("text backend wired in Task 4")
        self._text.text_evict(model)

    def resources(self) -> dict:
        # Process-wide memory/CPU plus whichever sub-backends are actually
        # wired. `resident` prefers the residency manager's own view of what
        # is ACTUALLY loaded (§8.1); the Task-2 placeholder only applies when
        # no residency was injected (e.g. tests constructing this directly).
        vm = psutil.virtual_memory()
        active = self._activity.current()
        if self._residency is not None:
            resident = list(self._residency.resident())
        else:
            resident = ["siglip"] if self._embed is not None else []
        # The text model (llama-server / vLLM) is a SEPARATE always-on process that
        # keeps its one model loaded for its whole life and also serves captioning, so
        # it is resident in EVERY state — show it always (idle, embedding, captioning,
        # chat, planning), not only during a text op. (Under the old Ollama warm/unload
        # model this was gated to chat/planning; there is no unload now — §13.)
        if self._text is not None:
            text_fn = getattr(self._text, "resident_models", None)
            if text_fn is not None:
                for model in text_fn():
                    if model not in resident:
                        resident.append(model)
        return {
            "ram_used_mb": (vm.total - vm.available) / (1024 * 1024),
            "ram_total_mb": vm.total / (1024 * 1024),
            "cpu_pct": psutil.cpu_percent(),
            "gpu_pct": read_gpu_pct(),
            "resident": resident,
            "active": active,
        }
