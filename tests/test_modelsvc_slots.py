"""`SlotManager` — the registry's four units, built FROM the catalog (design §4.1).

No torch, no llama-server: the worker and llm factories are injected, so this is
about registration, eviction on switch, and the generation counter.
"""

import pytest

from config import Settings
from models import catalog
from modelsvc.registry import ModelRegistry
from modelsvc.slots import SlotManager


class _FakeWorker:
    def __init__(self, target, args, warm=None):
        self.target, self.args, self.warm = target, args, warm
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def is_alive(self):
        return self.started and not self.stopped

    def call(self, method, *args):
        return ("called", method, args)


class _FakeLlm:
    def __init__(self, planner, caption):
        self.planner, self.caption = planner, caption
        self.loaded = None
        self.freed = 0

    def load(self, *, vision=False):
        self.loaded = "vision" if vision else "text"

    def free(self):
        self.loaded = None
        self.freed += 1

    def is_loaded(self):
        return self.loaded is not None


def _manager(tmp_path, profile="jetson", **kw):
    settings = Settings(data_dir=tmp_path, profile=profile, **kw)
    registry = ModelRegistry()
    built = {"workers": [], "llms": []}

    def worker_factory(target, args, warm=None):
        w = _FakeWorker(target, args, warm)
        built["workers"].append(w)
        return w

    def llm_factory(planner, caption):
        llm = _FakeLlm(planner, caption)
        built["llms"].append(llm)
        return llm

    manager = SlotManager(
        settings, registry, llm_factory=llm_factory, worker_factory=worker_factory
    )
    return manager, registry, built, settings


def test_apply_registers_the_four_units(tmp_path):
    manager, registry, _, _ = _manager(tmp_path)
    manager.apply(catalog.DEFAULTS["jetson"])
    assert set(registry._specs) == {"image_embed", "text_embed", "llm", "llm_vision"}


def test_costs_come_from_the_catalog_entry(tmp_path):
    manager, registry, _, _ = _manager(tmp_path)
    manager.apply(catalog.DEFAULTS["jetson"])
    assert registry.cost_mb("image_embed") == 3400
    assert registry.cost_mb("text_embed") == 2200
    assert registry.cost_mb("llm") == 3800
    assert registry.cost_mb("llm_vision") == 4300


def test_config_can_override_a_measured_cost_per_unit(tmp_path):
    # `IVMS777_MODEL_COST_MB` stays the board-side escape hatch (§8.1): correct a
    # cost without editing the catalog.
    manager, registry, _, _ = _manager(tmp_path, model_cost_mb={"image_embed": 1234})
    manager.apply(catalog.DEFAULTS["jetson"])
    assert registry.cost_mb("image_embed") == 1234


def test_switching_a_slot_evicts_the_old_unit_and_rebuilds_it(tmp_path):
    manager, registry, built, _ = _manager(tmp_path)
    manager.apply(catalog.DEFAULTS["jetson"])
    registry.ensure("image_embed")
    old = built["workers"][0]
    assert old.started

    manager.apply({"image_embed": "siglip2-so400m-512"})

    assert old.stopped, "the outgoing worker must be killed, not left resident"
    assert not registry.is_resident("image_embed")
    assert registry.cost_mb("image_embed") == catalog.get(
        "image_embed", "siglip2-so400m-512"
    ).cost_mb
    # the new worker is built lazily from the NEW entry
    registry.ensure("image_embed")
    assert built["workers"][-1].args[0] == "google/siglip2-so400m-patch16-512"


def test_switching_bumps_the_generation(tmp_path):
    manager, _, _, _ = _manager(tmp_path)
    manager.apply(catalog.DEFAULTS["jetson"])
    first = manager.generation
    manager.apply({"caption": "qwen3-vl-4b"})
    assert manager.generation == first + 1


def test_applying_the_same_slots_changes_nothing(tmp_path):
    manager, registry, built, _ = _manager(tmp_path)
    manager.apply(catalog.DEFAULTS["jetson"])
    registry.ensure("image_embed")
    generation = manager.generation
    manager.apply(catalog.DEFAULTS["jetson"])
    assert manager.generation == generation
    assert registry.is_resident("image_embed")  # no pointless eviction
    assert built["workers"][0].stopped is False


def test_an_unknown_key_is_refused_and_changes_nothing(tmp_path):
    manager, _registry, _, _ = _manager(tmp_path)
    manager.apply(catalog.DEFAULTS["jetson"])
    generation = manager.generation
    with pytest.raises(ValueError):
        manager.apply({"caption": "no-such-model"})
    assert manager.generation == generation
    assert manager.state()["slots"]["caption"] == "gemma4-E2B"


def test_a_key_not_offered_on_this_profile_is_refused(tmp_path):
    # qwen3-vl-8b is mac-only; on jetson the governor could never load it (§8.1).
    manager, _, _, _ = _manager(tmp_path, profile="jetson")
    manager.apply(catalog.DEFAULTS["jetson"])
    with pytest.raises(ValueError):
        manager.apply({"caption": "qwen3-vl-8b"})


def test_switching_an_llm_slot_rebuilds_the_child_for_both_modes(tmp_path):
    manager, registry, built, _ = _manager(tmp_path)
    manager.apply(catalog.DEFAULTS["jetson"])
    registry.ensure("llm")
    first_llm = built["llms"][-1]

    manager.apply({"caption": "qwen3-vl-4b"})

    assert first_llm.freed >= 1, "the running llama-server must be killed on a switch"
    assert not registry.is_resident("llm")
    registry.ensure("llm_vision")
    new_llm = built["llms"][-1]
    assert new_llm is not first_llm
    assert new_llm.caption.key == "qwen3-vl-4b"
    assert new_llm.planner.key == "gemma4-E2B"


def test_llm_modes_stay_mutually_exclusive_after_a_switch(tmp_path):
    manager, registry, _, _ = _manager(tmp_path)
    manager.apply(catalog.DEFAULTS["jetson"])
    manager.apply({"planner": "qwen3-4b-2507"})
    registry.ensure("llm")
    registry.ensure("llm_vision")
    assert registry.resident() == ["llm_vision"]  # loading one frees the other


def test_cloud_registers_no_text_embed_unit(tmp_path):
    # cloud keeps the OpenAI /embeddings path, so there is no in-process embedder.
    manager, registry, _, _ = _manager(tmp_path, profile="cloud")
    manager.apply(catalog.DEFAULTS["cloud"])
    assert "text_embed" not in registry._specs


def test_state_reports_slots_and_generation(tmp_path):
    manager, _, _, _ = _manager(tmp_path)
    manager.apply(catalog.DEFAULTS["mac"])
    state = manager.state()
    assert state["slots"] == catalog.DEFAULTS["mac"]
    assert state["generation"] == manager.generation


def test_entry_exposes_the_live_selection(tmp_path):
    manager, _, _, _ = _manager(tmp_path)
    manager.apply(catalog.DEFAULTS["jetson"])
    manager.apply({"image_embed": "siglip2-so400m-512"})
    assert manager.entry("image_embed").preprocess.input_px == 512


def test_alive_probe_follows_the_live_worker(tmp_path):
    # A model that can die behind our back needs a probe bound to the CURRENT child,
    # not the one that existed when the spec was first registered (§8.1).
    manager, registry, built, _ = _manager(tmp_path)
    manager.apply(catalog.DEFAULTS["jetson"])
    registry.ensure("image_embed")
    manager.apply({"image_embed": "siglip2-so400m-512"})
    registry.ensure("image_embed")
    assert registry._specs["image_embed"].alive() is True
    built["workers"][-1].stop()
    assert registry._specs["image_embed"].alive() is False
