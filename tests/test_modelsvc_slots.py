"""`SlotManager` — the registry's four units, built FROM the catalog (design §4.1).

No torch, no llama-server: the worker and llm factories are injected, so this is
about registration, eviction on switch, and the generation counter.
"""

import pytest

from config import Settings
from models import catalog
from modelsvc.governor import MemoryGovernor
from modelsvc.registry import ModelRegistry
from modelsvc.slots import SlotManager

VOXCPM2_LORA_JSON = (
    '{"alpha": 16, "dropout": 0.0, "enable_dit": true, "enable_lm": true, "enable_proj": false, '
    '"r": 8, "target_modules_dit": ["q_proj", "v_proj", "k_proj", "o_proj"], '
    '"target_modules_lm": ["q_proj", "v_proj", "k_proj", "o_proj"]}'
)
TRAINER_TARGET = "audio.voxcpm_trainer:VoxCPMTrainer"


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


def test_apply_registers_a_unit_per_slot(tmp_path):
    manager, registry, _, _ = _manager(tmp_path)
    manager.apply(catalog.DEFAULTS["jetson"])
    assert set(registry._specs) == {
        "image_embed", "text_embed", "llm", "llm_vision", "voice", "music",
    }


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
    # On the mac PROFILE: its defaults name mac-only models (musicgen-medium, §4.1),
    # and `apply` rightly refuses a model the running profile does not offer.
    manager, _, _, _ = _manager(tmp_path, profile="mac")
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


def test_voice_and_music_register_torch_units(tmp_path):
    manager, registry, _, _ = _manager(tmp_path)
    manager.apply(catalog.DEFAULTS["jetson"])
    assert manager.has_unit("voice") and manager.has_unit("music")
    assert registry.cost_mb("voice") == catalog.get("voice", "kokoro-82m").cost_mb
    assert registry.cost_mb("music") == catalog.get("music", "musicgen-small").cost_mb


def test_the_audio_units_are_built_from_the_audio_package(tmp_path):
    # The children must be resolved BY NAME through TorchWorker so the models parent
    # process never imports torch (CLAUDE.md, §5.1).
    manager, registry, built, _ = _manager(tmp_path)
    manager.apply(catalog.DEFAULTS["jetson"])
    registry.ensure("voice")
    registry.ensure("music")
    targets = {w.target for w in built["workers"]}
    assert "audio.kokoro_voice:KokoroVoice" in targets
    assert "audio.musicgen:MusicGen" in targets






def test_the_language_picks_the_voice_model(tmp_path):
    # The user never chooses a voice: asking for Russian swaps the unit to a model
    # that speaks Russian (§11.1).
    manager, _, _, _ = _manager(tmp_path)
    manager.apply(catalog.DEFAULTS["jetson"])
    assert manager.entry("voice").key == "kokoro-82m"

    assert manager.ensure_voice_for("ru") is True
    assert manager.entry("voice").key == "silero-v4-ru"

    assert manager.ensure_voice_for("en") is True
    assert manager.entry("voice").key == "kokoro-82m"


def test_a_language_nothing_speaks_is_refused(tmp_path):
    manager, _, _, _ = _manager(tmp_path)
    manager.apply(catalog.DEFAULTS["jetson"])
    assert manager.ensure_voice_for("klingon") is False



def test_a_saved_voice_routes_to_the_cloning_child_with_its_revision(tmp_path):
    manager, registry, built, settings = _manager(tmp_path, profile="mac")
    manager.apply(catalog.DEFAULTS["mac"])
    generation = manager.generation

    assert manager.ensure_voice_for("en", cloning=True) is True
    assert manager.entry("voice").key == "voxcpm2"
    assert manager.generation == generation + 1
    registry.ensure("voice")
    worker = built["workers"][-1]
    source = catalog.get("voice", "voxcpm2").source
    assert worker.target == "audio.voxcpm_voice:VoxCPMVoice"
    assert worker.args == (source.repo, settings.embed_device, source.revision, VOXCPM2_LORA_JSON)
    assert registry.cost_mb("voice") == catalog.get("voice", "voxcpm2").cost_mb


def test_preset_routing_never_picks_the_cloning_entry(tmp_path):
    manager, _, _, _ = _manager(tmp_path, profile="mac")
    manager.apply(catalog.DEFAULTS["mac"])
    for language in ("en", "ru"):
        assert manager.voice_entry_for(language).clones is False

    # Also when the cloning child is the one resident: the selection no longer wins.
    assert manager.ensure_voice_for("ru", cloning=True) is True
    assert manager.voice_entry_for("en").key == "kokoro-82m"
    assert manager.voice_entry_for("ru").key == "silero-v4-ru"
    assert manager.ensure_voice_for("ru") is True
    assert manager.entry("voice").key == "silero-v4-ru"


def test_a_saved_voice_never_reaches_a_preset_child(tmp_path):
    manager, _, _, _ = _manager(tmp_path, profile="mac")
    manager.apply(catalog.DEFAULTS["mac"])
    assert manager.entry("voice").key == "kokoro-82m"   # selected, speaks en
    assert manager.voice_entry_for("en", cloning=True).key == "voxcpm2"


def test_jetson_has_no_voice_copy_and_keeps_its_voice(tmp_path):
    manager, _, _, _ = _manager(tmp_path, profile="jetson")
    manager.apply(catalog.DEFAULTS["jetson"])
    generation = manager.generation
    assert manager.ensure_voice_for("en", cloning=True) is False
    assert manager.entry("voice").key == "kokoro-82m"
    assert manager.generation == generation


def test_an_unpinned_voice_keeps_two_child_args(tmp_path):
    manager, registry, built, settings = _manager(tmp_path)
    manager.apply(catalog.DEFAULTS["jetson"])
    registry.ensure("voice")
    voice = next(w for w in built["workers"] if w.target == "audio.kokoro_voice:KokoroVoice")
    assert voice.args == ("hexgrad/Kokoro-82M", settings.embed_device)


def test_every_catalog_voice_has_a_child_class():
    from modelsvc.slots import VOICE_TARGETS

    for entry in catalog.CATALOG:
        if entry.slot == "voice":
            assert entry.key in VOICE_TARGETS, entry.key


def test_a_voice_with_no_child_class_raises_instead_of_falling_back(tmp_path, monkeypatch):
    # No silent Kokoro: a stored identity must never be spoken by another model.
    from modelsvc import slots

    manager, _, _, _ = _manager(tmp_path)
    manager.apply(catalog.DEFAULTS["jetson"])
    monkeypatch.delitem(slots.VOICE_TARGETS, "silero-v4-ru")
    with pytest.raises(KeyError):
        manager.ensure_voice_for("ru")


def test_a_silero_voice_keeps_two_child_args(tmp_path):
    manager, registry, built, settings = _manager(tmp_path)
    manager.apply(catalog.DEFAULTS["jetson"])
    manager.ensure_voice_for("ru")
    registry.ensure("voice")
    voice = built["workers"][-1]
    assert voice.target == "audio.silero_voice:SileroVoice"
    assert voice.args == (
        str(tmp_path / "models" / catalog.get("voice", "silero-v4-ru").source.file),
        settings.embed_device,
    )



def test_mac_registers_the_training_unit_from_the_adapter_entry(tmp_path):
    manager, registry, built, settings = _manager(tmp_path, profile="mac")
    manager.apply(catalog.DEFAULTS["mac"])
    registry.ensure("voice_train")
    worker = built["workers"][-1]
    source = catalog.get("voice", "voxcpm2").source
    assert worker.target == TRAINER_TARGET
    assert worker.args == (
        source.repo,
        settings.embed_device,
        source.revision,
        VOXCPM2_LORA_JSON,
        str(tmp_path / "voice-bakes"),
    )
    assert worker.warm is None
    assert registry.cost_mb("voice_train") == 14000


def test_config_overrides_the_training_cost(tmp_path):
    manager, registry, _, _ = _manager(tmp_path, profile="mac", model_cost_mb={"voice_train": 9000})
    manager.apply(catalog.DEFAULTS["mac"])
    assert registry.cost_mb("voice_train") == 9000


def test_jetson_registers_no_training_unit(tmp_path):
    manager, registry, _, _ = _manager(tmp_path, profile="jetson")
    manager.apply(catalog.DEFAULTS["jetson"])
    assert "voice_train" not in registry._specs


def test_a_slot_switch_never_touches_a_leased_training_unit(tmp_path):
    manager, registry, built, _ = _manager(tmp_path, profile="mac")
    manager.apply(catalog.DEFAULTS["mac"])
    governor = MemoryGovernor(registry, measure_free_mb=lambda: 100000.0, budget_mb=100000)
    governor.acquire(["voice_train"])
    trainer = built["workers"][-1]

    with governor.lease("voice_train"):
        manager.apply({"music": "musicgen-small"})

        assert "voice_train" in registry._specs
        assert registry.is_resident("voice_train")
        assert trainer.started and not trainer.stopped


ASR_TARGET = "audio.whisper_asr:WhisperASR"


@pytest.mark.parametrize(("profile", "device"), [("mac", "mps"), ("cloud", "cuda")])
def test_the_asr_unit_is_registered_from_the_transcriber_entry(tmp_path, profile, device):
    manager, registry, built, _ = _manager(tmp_path, profile=profile)
    manager.apply(catalog.DEFAULTS[profile])
    registry.ensure("asr")
    worker = built["workers"][-1]
    source = catalog.transcriber_for(profile).source
    assert worker.target == ASR_TARGET
    assert worker.args == (source.repo, device, source.revision)
    assert worker.warm is None
    assert registry.cost_mb("asr") == 4000


def test_config_overrides_the_asr_cost(tmp_path):
    manager, registry, _, _ = _manager(tmp_path, profile="mac", model_cost_mb={"asr": 3500})
    manager.apply(catalog.DEFAULTS["mac"])
    assert registry.cost_mb("asr") == 3500


def test_a_slot_switch_leaves_a_running_asr_child_alone(tmp_path):
    manager, registry, built, _ = _manager(tmp_path, profile="mac")
    manager.apply(catalog.DEFAULTS["mac"])
    registry.ensure("asr")
    whisper = built["workers"][-1]

    manager.apply({"music": "musicgen-small"})

    assert "asr" in registry._specs
    assert registry.is_resident("asr")
    assert whisper.started and not whisper.stopped
    assert manager.worker("asr") is whisper


def test_no_asr_unit_on_jetson(tmp_path):
    manager, registry, _, _ = _manager(tmp_path, profile="jetson")
    manager.apply(catalog.DEFAULTS["jetson"])
    assert "asr" not in registry._specs
    assert not manager.has_unit("asr")


SPEAKER_TARGET = "audio.speaker_embed:SpeakerEmbedder"


def test_mac_registers_the_speaker_unit_from_its_catalog_entry(tmp_path):
    manager, registry, built, settings = _manager(tmp_path, profile="mac")
    manager.apply(catalog.DEFAULTS["mac"])
    registry.ensure("speaker_embed")
    worker = built["workers"][-1]
    source = catalog.speaker_embedder_for("mac").source
    assert worker.target == SPEAKER_TARGET
    assert worker.args == (source.repo, settings.embed_device, source.revision)
    assert worker.warm is None
    assert registry.cost_mb("speaker_embed") == 1000


def test_config_overrides_the_speaker_cost(tmp_path):
    manager, registry, _, _ = _manager(tmp_path, profile="mac",
                                       model_cost_mb={"speaker_embed": 700})
    manager.apply(catalog.DEFAULTS["mac"])
    assert registry.cost_mb("speaker_embed") == 700


def test_a_slot_switch_leaves_a_running_speaker_child_alone(tmp_path):
    manager, registry, built, _ = _manager(tmp_path, profile="mac")
    manager.apply(catalog.DEFAULTS["mac"])
    registry.ensure("speaker_embed")
    child = built["workers"][-1]

    manager.apply({"music": "musicgen-small"})

    assert registry.is_resident("speaker_embed")
    assert manager.worker("speaker_embed") is child
    assert child.started and not child.stopped


@pytest.mark.parametrize("profile", ["jetson", "cloud"])
def test_no_speaker_unit_where_no_embedder_is_offered(tmp_path, profile):
    manager, registry, _, _ = _manager(tmp_path, profile=profile)
    manager.apply(catalog.DEFAULTS[profile])
    assert "speaker_embed" not in registry._specs
    assert not manager.has_unit("speaker_embed")
