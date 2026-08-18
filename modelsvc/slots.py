"""The four model slots, as the registry sees them (design §4.1, §8.1).

`SlotManager` is the only thing that turns a **catalog entry** into a registered
**residency unit**. The units are named by slot, never by model, so switching a
model renames nothing the governor, the scheduler or a "needs" list mentions:

| slot      | unit         | what it is                                  |
|-----------|--------------|---------------------------------------------|
| image_embed | `image_embed` | a `TorchWorker` child (SigLIP by default) |
| text_embed  | `text_embed`  | a `TorchWorker` child (nomic by default)  |
| planner     | `llm`         | `llama-server`, TEXT mode                 |
| caption     | `llm_vision`  | `llama-server`, VISION mode (+ projector) |

`llm` and `llm_vision` are one child on one port, so they are mutually exclusive:
loading either frees the other (§3.1). The two llm slots may name different GGUFs;
then the child restarts when the role changes, which is the same shape today's
text↔vision restart already has.

Everything is built LAZILY: `apply()` only registers `load`/`free` callables, so
constructing the service still spawns nothing and imports no torch (§5.1).
"""

from __future__ import annotations

from collections.abc import Callable

from models import catalog
from models.catalog import GgufSource, HfSource, ModelEntry
from modelsvc.registry import ModelRegistry, ModelSpec

# slot -> the residency unit it occupies.
UNIT_BY_SLOT: dict[str, str] = {
    "image_embed": "image_embed",
    "text_embed": "text_embed",
    "planner": "llm",
    "caption": "llm_vision",
}
LLM_UNITS = ("llm", "llm_vision")


class SlotManager:
    def __init__(
        self,
        settings,
        registry: ModelRegistry,
        *,
        llm_factory: Callable[[ModelEntry, ModelEntry], object],
        worker_factory: Callable[..., object] | None = None,
    ) -> None:
        self._settings = settings
        self._registry = registry
        self._llm_factory = llm_factory
        self._worker_factory = worker_factory or _default_worker_factory
        self._slots: dict[str, str] = {}
        self._workers: dict[str, object] = {}
        self._llm = None
        self.generation = 0

    # --- state ---------------------------------------------------------------
    def state(self) -> dict:
        return {"slots": dict(self._slots), "generation": self.generation}

    def entry(self, slot: str) -> ModelEntry:
        return catalog.get(slot, self._slots[slot])

    def worker(self, unit: str):
        """The LIVE child for a torch unit. The scheduler makes the unit resident
        before any op runs, so a missing worker here is a bug, not a race."""
        worker = self._workers.get(unit)
        if worker is None:
            raise RuntimeError(f"{unit} is not loaded")
        return worker

    def has_unit(self, unit: str) -> bool:
        return unit in self._registry._specs

    # --- switching -----------------------------------------------------------
    def apply(self, slots: dict[str, str]) -> None:
        """Set some or all slots. Unchanged slots are left completely alone —
        no eviction, no generation bump — so a re-push after a `models` restart
        costs nothing."""
        profile = self._settings.profile
        target = dict(self._slots)
        for slot, key in slots.items():
            if slot not in catalog.SLOTS:
                raise ValueError(f"unknown slot: {slot}")
            if not catalog.has(slot, key):
                raise ValueError(f"unknown model for {slot}: {key}")
            if profile not in catalog.get(slot, key).profiles:
                raise ValueError(f"{key} is not offered on {profile}")
            target[slot] = key
        for slot in catalog.SLOTS:
            target.setdefault(slot, catalog.default_key(slot, profile))

        changed = {slot for slot in catalog.SLOTS if target[slot] != self._slots.get(slot)}
        if not changed:
            return

        self._slots = target
        if "image_embed" in changed:
            self._register_image_embed()
        if "text_embed" in changed:
            self._register_text_embed()
        if changed & {"caption", "planner"}:
            self._register_llm()
        self.generation += 1

    # --- unit registration ---------------------------------------------------
    def _cost(self, unit: str, entry: ModelEntry) -> int:
        # `IVMS777_MODEL_COST_MB` overrides the catalog per UNIT — the board-side
        # escape hatch for a cost that measures differently than declared (§8.1).
        return self._settings.model_cost_mb.get(unit, entry.cost_mb)

    def _register_torch_unit(self, unit: str, entry: ModelEntry, target: str, warm=None) -> None:
        # Drop whatever occupies the unit first: `unload` kills the child, which is
        # the only thing that returns its memory (§8.1).
        self._registry.unload(unit)
        self._workers.pop(unit, None)
        assert isinstance(entry.source, HfSource)
        args = (entry.source.repo, self._settings.embed_device)

        def load() -> None:
            worker = self._workers.get(unit)
            if worker is None:
                worker = self._worker_factory(target, args, warm)
                self._workers[unit] = worker
            worker.start()

        def free() -> None:
            worker = self._workers.pop(unit, None)
            if worker is not None:
                worker.stop()

        def alive() -> bool:
            worker = self._workers.get(unit)
            return worker is not None and worker.is_alive()

        self._registry.register(ModelSpec(unit, load, free, self._cost(unit, entry), alive=alive))

    def _register_image_embed(self) -> None:
        self._register_torch_unit(
            "image_embed", self.entry("image_embed"), "embedding.siglip:SiglipEmbedder"
        )

    def _register_text_embed(self) -> None:
        # cloud keeps the OpenAI `/embeddings` path, so it hosts no text embedder.
        if self._settings.profile == "cloud":
            return
        self._register_torch_unit(
            "text_embed",
            self.entry("text_embed"),
            "embedding.text_embedder:TextEmbedder",
            warm="warm",  # this encoder loads lazily; resident must mean resident
        )

    def _register_llm(self) -> None:
        for unit in LLM_UNITS:
            self._registry.unload(unit)  # kills the running llama-server, if any
        self._llm = None
        planner, caption = self.entry("planner"), self.entry("caption")

        def llm():
            if self._llm is None:
                self._llm = self._llm_factory(planner, caption)
            return self._llm

        def load(vision: bool) -> None:
            # One child, one port: making either mode resident frees the other, so
            # the registry's resident set stays honest (§3.1).
            self._registry.unload("llm" if vision else "llm_vision")
            llm().load(vision=vision)

        def free() -> None:
            if self._llm is not None:
                self._llm.free()

        def alive() -> bool:
            return self._llm is not None and self._llm.is_loaded()

        self._registry.register(
            ModelSpec("llm", lambda: load(False), free, self._cost("llm", planner), alive=alive)
        )
        self._registry.register(
            ModelSpec(
                "llm_vision",
                lambda: load(True),
                free,
                self._cost("llm_vision", caption),
                alive=alive,
            )
        )


def entry_payload(entry: ModelEntry, *, current: bool, switchable: bool, download: dict) -> dict:
    """One catalog entry as the settings popup needs it (design §13)."""
    preprocess = entry.preprocess
    return {
        "slot": entry.slot,
        "key": entry.key,
        "display": entry.display,
        "size_mb": entry.size_mb,
        "cost_mb": entry.cost_mb,
        "cost_measured": entry.cost_measured,
        "dim": entry.dim,
        "preprocess": (
            None
            if preprocess is None
            else {
                "input_px": preprocess.input_px,
                "resample": preprocess.resample,
                "mode": preprocess.mode,
            }
        ),
        "note": entry.note,
        "current": current,
        "switchable": switchable,
        "download": download,
    }


def catalog_payload(profile: str, slots: dict[str, str], generation: int, download_status) -> dict:
    """`GET /models/catalog` — the repo catalog joined with live state (§5.1)."""
    entries = []
    for slot in catalog.SLOTS:
        for entry in catalog.entries_for(slot, profile):
            entries.append(
                entry_payload(
                    entry,
                    current=slots.get(slot) == entry.key,
                    switchable=catalog.is_switchable(slot, profile),
                    download=download_status(entry),
                )
            )
    return {
        "profile": profile,
        "slots": dict(slots),
        "generation": generation,
        "entries": entries,
    }


def _default_worker_factory(target: str, args: tuple, warm=None):
    from modelsvc.torch_process import TorchWorker

    return TorchWorker(target, args, warm=warm)


def gguf_paths(entry: ModelEntry, model_dir) -> tuple[str, str | None]:
    """`(weights, projector)` on disk for a GGUF entry — the paths `llama-server`
    is pointed at and the downloader writes to, so they are derived in ONE place."""
    assert isinstance(entry.source, GgufSource)
    weights = str(model_dir / entry.source.file)
    mmproj = str(model_dir / entry.source.mmproj_file) if entry.source.mmproj_file else None
    return weights, mmproj
