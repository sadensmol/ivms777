"""`CompositeBackend` — a `ModelBackend` assembled from independent
sub-backends, one per concern (design §5.1), and the single funnel through
which every model op passes the conveyor (plan 18).

Each op acquires a scheduler slot (concurrency + priority) and asks the governor
to make the models it needs resident, then runs. Interactive ops (search/chat)
use `Priority.INTERACTIVE`; batch ingest captioning uses `Priority.BATCH`, so a
chat preempts a queued caption on the single-slot Jetson. When no `scheduler` is
injected (bare unit tests), ops run directly — same behaviour as before plan 18.

A sub-backend not yet wired raises `NotImplementedError` so a premature call
fails loudly.
"""

from collections.abc import Iterator

from models import catalog
from modelsvc.activity import Activity
from modelsvc.scheduler import Priority
from modelsvc.slots import LLM_UNITS, catalog_payload


class CompositeBackend:
    def __init__(
        self,
        *,
        embed,
        caption=None,
        text=None,
        registry=None,
        governor=None,
        scheduler=None,
        text_embed_needs: tuple[str, ...] = ("text_embed",),
        slots=None,
        downloads=None,
        profile: str = "mac",
    ) -> None:
        self._embed = embed
        self._caption = caption
        self._text = text
        self._registry = registry
        self._governor = governor
        self._scheduler = scheduler
        self._text_embed_needs = text_embed_needs
        # The `SlotManager` (design §4.1) — which model each slot holds, and the
        # generation `app` compares against. None in bare unit tests.
        self._slots = slots
        self._downloads = downloads
        self._profile = profile
        # The current in-flight op label, for the resource bar (§13).
        self._activity = Activity()

    # --- conveyor plumbing ---
    def _run(self, needed, priority, label, fn):
        def tracked():
            with self._activity.track(label):
                return fn()

        if self._scheduler is None:
            return tracked()
        result = self._scheduler.run(list(needed), priority, tracked)
        # An embedder op just ran; proactively unload the llm (either mode) if it has
        # gone idle past its TTL, freeing GB for the rest of a long batch (no-op on
        # mac/cloud).
        if not any(n.startswith("llm") for n in needed):
            self._scheduler.reap_idle(list(LLM_UNITS))
        return result

    # --- SigLIP ---
    def embed_image(self, images: list[bytes]) -> list[list[float]]:
        return self._run(["image_embed"], Priority.INTERACTIVE, "embedding",
                         lambda: self._embed.embed_image(images))

    def embed_text(self, texts: list[str]) -> list[list[float]]:
        return self._run(["image_embed"], Priority.INTERACTIVE, "embedding",
                         lambda: self._embed.embed_text(texts))

    def tag(self, image: bytes, dimensions: list[str]) -> dict[str, list[str]]:
        return self._run(["image_embed"], Priority.INTERACTIVE, "tagging",
                         lambda: self._embed.tag(image, dimensions))

    def calibration(self) -> dict:
        return self._run(["image_embed"], Priority.INTERACTIVE, "embedding",
                         lambda: self._embed.calibration())

    # --- captioning (llm_vision, batch) ---
    def caption(self, image: bytes) -> dict:
        # The ONLY op that sends an image to the llm, so the ONLY one that needs the
        # vision projector loaded (design §3.1). Chat/planner use text-only `llm`.
        if self._caption is None:
            raise NotImplementedError("caption backend not wired")
        return self._run(["llm_vision"], Priority.BATCH, "captioning",
                         lambda: self._caption.caption(image))

    # --- text generation (llm) ---
    def text_complete(
        self,
        model: str,
        messages: list[dict],
        json_schema: dict | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        if self._text is None:
            raise NotImplementedError("text backend not wired")
        return self._run(["llm"], Priority.INTERACTIVE, "planning",
                         lambda: self._text.text_complete(
                             model, messages, json_schema=json_schema,
                             temperature=temperature, max_tokens=max_tokens))

    def text_stream(self, model: str, messages: list[dict]) -> Iterator[str]:
        if self._text is None:
            raise NotImplementedError("text backend not wired")

        # A generator: hold the slot (and the llm) for the WHOLE stream, not just
        # until the generator object is built, so it is not evicted mid-answer.
        def stream() -> Iterator[str]:
            if self._scheduler is None:
                with self._activity.track("chat"):
                    yield from self._text.text_stream(model, messages)
                return
            with self._scheduler.reserve(["llm"], Priority.INTERACTIVE), self._activity.track("chat"):
                yield from self._text.text_stream(model, messages)

        return stream()

    def text_embed(self, model: str, texts: list[str]) -> list[list[float]]:
        if self._text is None:
            raise NotImplementedError("text backend not wired")
        return self._run(list(self._text_embed_needs), Priority.INTERACTIVE, "embedding",
                         lambda: self._text.text_embed(model, texts))

    def text_warm(self, model: str) -> None:
        if self._text is None:
            raise NotImplementedError("text backend not wired")
        self._text.text_warm(model)

    def text_evict(self, model: str) -> None:
        if self._text is None:
            raise NotImplementedError("text backend not wired")
        self._text.text_evict(model)

    # --- slot control (plan 21, design §4.1) ---
    def _slot_keys(self) -> dict:
        if self._slots is not None:
            return self._slots.state()["slots"]
        return dict(catalog.DEFAULTS[self._profile])

    def _generation(self) -> int:
        return self._slots.generation if self._slots is not None else 0

    def _download_status(self, entry) -> dict:
        if self._downloads is None:
            return {"state": "ready", "bytes": 0, "total": 0, "error": None}
        return self._downloads.status(entry)

    def catalog(self) -> dict:
        return catalog_payload(
            self._profile, self._slot_keys(), self._generation(), self._download_status
        )

    def set_slots(self, slots: dict) -> dict:
        if self._slots is None:
            raise ValueError("this backend has no switchable slots")
        self._slots.apply(slots)  # ValueError on an unknown / not-offered key
        return self._slots.state()

    def download(self, slot: str, key: str) -> dict:
        entry = catalog.get(slot, key)  # KeyError on an unknown model
        if self._downloads is None:
            raise ValueError("this backend downloads nothing")
        self._downloads.start(entry)
        return self._downloads.status(entry)

    def embed_spec(self) -> dict:
        """Calibration AND the image embedder's preprocessing contract, in one
        round trip (design §5.1). The caller resizes by this, never by a constant."""
        entry = (
            self._slots.entry("image_embed")
            if self._slots is not None
            else catalog.get("image_embed", catalog.default_key("image_embed", self._profile))
        )
        pre = entry.preprocess
        return self.calibration() | {
            "preprocess": {
                "input_px": pre.input_px,
                "resample": pre.resample,
                "mode": pre.mode,
            },
            "generation": self._generation(),
        }

    # --- conveyor control (plan 18, control API) ---
    def models_state(self) -> dict:
        slots = {"slots": self._slot_keys(), "generation": self._generation()}
        if self._governor is None:
            resident = list(self._registry.resident()) if self._registry is not None else []
            return {"resident": resident, "budget_mb": 0, "free_mb": 0.0,
                    "used_mb": 0, "active": self._activity.current()} | slots
        st = self._governor.state()
        return {"resident": st.resident, "budget_mb": st.budget_mb, "free_mb": st.free_mb,
                "used_mb": st.used_mb, "active": self._activity.current()} | slots

    def model_ensure(self, name: str) -> None:
        if self._scheduler is not None:
            self._scheduler.run([name], Priority.INTERACTIVE, lambda: None)
        elif self._registry is not None:
            self._registry.ensure(name)

    def model_unload(self, name: str) -> None:
        if self._registry is not None:
            self._registry.unload(name)

    def resources(self) -> dict:
        """What this service ALONE knows: which models are resident, and the op
        in flight. No machine metrics — RAM/CPU/GPU-load/temperature are host-wide
        sysfs/psutil reads that need no CUDA context, so `app` reads them itself
        and the bar keeps working when this service is down (design §5.1, §13).
        """
        active = self._activity.current()
        if self._registry is not None:
            resident = list(self._registry.resident())
        else:
            resident = ["image_embed"] if self._embed is not None else []
        # The text model is reported by the registry now (it is a managed
        # model). When no registry is wired (bare tests), fall back to the text
        # backend's own view so the resource bar still shows it.
        if self._registry is None and self._text is not None:
            text_fn = getattr(self._text, "resident_models", None)
            if text_fn is not None:
                for model in text_fn():
                    if model not in resident:
                        resident.append(model)
        return {
            "resident": resident,
            "active": active,
            "slots": self._slot_keys(),
            "generation": self._generation(),
        }
