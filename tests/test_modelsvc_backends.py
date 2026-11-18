"""`build_backend` (config -> `ModelBackend`) and `CompositeBackend` routing
through the conveyor (design §5.1, plan 18). The real embedder/llm loads are lazy,
so building the backend stays torch-free and never spawns llama-server. The units
are named by SLOT since plan 21 (design §4.1).
"""

import pytest

from config import Settings
from modelsvc.backends import build_backend
from modelsvc.backends.composite import CompositeBackend
from modelsvc.backends.fake import FakeBackend
from modelsvc.backends.text_backend import TextBackend
from modelsvc.governor import MemoryGovernor
from modelsvc.registry import ModelRegistry, ModelSpec
from modelsvc.scheduler import Scheduler


def test_build_backend_returns_fake_backend_when_use_fake_embedder(tmp_path):
    settings = Settings(data_dir=tmp_path, use_fake_embedder=True)
    assert isinstance(build_backend(settings), FakeBackend)


def test_build_backend_real_path_wires_text_and_shares_one_inference_client(tmp_path):
    # Construct-only, no network / no torch: everything is lazy. Proves the real
    # path wires text=TextBackend and that caption + text share ONE inference
    # client (the single llama-server/vLLM gateway, design §5.1).
    settings = Settings(data_dir=tmp_path, profile="mac")
    backend = build_backend(settings)
    assert isinstance(backend, CompositeBackend)
    assert backend._text is not None
    assert backend._caption is not None
    assert backend._caption._captioner()._client is backend._text._client


def test_caption_backend_follows_a_slot_switch(tmp_path):
    # The captioner must be read from the LIVE `caption` slot, never snapshotted at
    # build time: the service boots on the profile defaults and `app` pushes the
    # user's stored selection afterwards (§4.1). A snapshot stamps every caption
    # with the default model's name and sends the default model's prompt template.
    backend = build_backend(Settings(data_dir=tmp_path, profile="mac"))
    before = backend._caption._captioner()
    assert before.caption_model == "gemma4-E2B"

    backend.set_slots({"caption": "qwen3-vl-4b"})

    after = backend._caption._captioner()
    assert after.caption_model == "qwen3-vl-4b"
    assert after.prompt_key == "qwen3-vl"
    assert after._client is before._client  # still ONE gateway client (§5.1)


def test_text_backend_reports_the_planner_slot_after_a_switch(tmp_path):
    # Same for the resource bar's text model (§13): it names what llama-server holds
    # right now, not what the profile defaulted to at boot.
    backend = build_backend(Settings(data_dir=tmp_path, profile="mac"))
    assert backend._text.resident_models() == ["gemma4-E2B"]

    backend.set_slots({"planner": "qwen3-4b-2507"})

    assert backend._text.resident_models() == ["qwen3-4b-2507"]


def test_build_backend_wires_conveyor_on_jetson(tmp_path):
    backend = build_backend(Settings(data_dir=tmp_path, profile="jetson"))
    assert isinstance(backend._scheduler, Scheduler)
    assert backend._scheduler.slots == 1  # jetson serializes the GPU
    # One managed unit per slot (design §4.1), plus the face unit, which sits outside
    # the slots and is registered on every profile (§8.1). There is no "off" model any
    # more, so every slot has one.
    assert set(backend._registry._specs) == {
        "image_embed", "text_embed", "llm", "llm_vision", "voice", "music", "faces",
    }
    assert backend._text_embed_needs == ("text_embed",)


def test_llm_specs_probe_the_supervised_child_for_liveness(tmp_path):
    # The llm is a separate process that can abort on its own (CUDA OOM at image
    # decode). Both llm specs must carry a liveness probe bound to that child, so
    # a dead one is respawned instead of 500-ing forever (§8.1).
    backend = build_backend(Settings(data_dir=tmp_path, profile="jetson"))
    specs = backend._registry._specs
    for name in ("llm", "llm_vision"):
        assert specs[name].alive is not None
        assert specs[name].alive() is False  # nothing spawned yet


def test_torch_models_are_registered_as_killable_workers(tmp_path):
    # Evicting SigLIP/nomic in-process does NOT return their memory (§8.1), so both
    # live in a child the registry kills — and both carry its liveness probe.
    backend = build_backend(Settings(data_dir=tmp_path, profile="jetson"))
    for name in ("image_embed", "text_embed"):
        spec = backend._registry._specs[name]
        assert spec.alive is not None
        assert spec.alive() is False  # nothing spawned at build time


def test_building_the_backend_never_imports_torch_into_the_parent(tmp_path):
    # The parent must stay torch-free forever: a CUDA context here is exactly the
    # ~2.7 GB gemma cannot get back (§8.1).
    import sys

    build_backend(Settings(data_dir=tmp_path, profile="jetson"))
    assert "torch" not in sys.modules


def test_build_backend_mac_runs_parallel(tmp_path):
    backend = build_backend(Settings(data_dir=tmp_path, profile="mac"))
    assert backend._scheduler.slots == 3


def test_build_backend_cloud_has_no_text_embed_unit_and_remote_llm(tmp_path):
    backend = build_backend(Settings(data_dir=tmp_path, profile="cloud"))
    assert "text_embed" not in backend._registry._specs  # cloud uses remote /embeddings
    assert {"image_embed", "llm"} <= set(backend._registry._specs)
    assert backend._text_embed_needs == ()


class _RecordingWorker:
    """Stands in for a `TorchWorker`; the real child process is covered by
    tests/test_torch_process.py."""

    def __init__(self):
        self.calls = []

    def call(self, method, *args):
        self.calls.append((method, args))
        if method.startswith("embed"):
            return [[0.5]]
        return {"logit_scale": 1.0, "logit_bias": 0.0}


def test_siglip_backend_routes_every_op_through_the_worker():
    from modelsvc.backends.siglip_backend import SiglipBackend

    worker = _RecordingWorker()
    # A PROVIDER, not a worker: a slot switch builds a new child (design §4.1).
    backend = SiglipBackend(lambda: worker)
    assert backend.embed_image([b"jpegbytes"]) == [[0.5]]
    assert backend.embed_text(["hi"]) == [[0.5]]
    assert backend.calibration() == {"logit_scale": 1.0, "logit_bias": 0.0}
    assert [c[0] for c in worker.calls] == ["embed_image_bytes", "embed_texts", "calibration"]
    # Raw bytes cross the pipe: decoding is the child's job, not the parent's.
    assert worker.calls[0][1] == ([b"jpegbytes"],)


class _StubClient:
    """Minimal inference client: only `embed` is exercised here."""

    def __init__(self):
        self.embedded = None

    def embed(self, model, texts):
        self.embedded = (model, texts)
        return [[0.1] for _ in texts]


def test_text_embed_uses_the_worker_when_wired():
    from modelsvc.backends.text_backend import TextBackend

    worker = _RecordingWorker()
    backend = TextBackend(
        _StubClient(), text_worker=lambda: worker, model_name=lambda: "gemma4-E2B"
    )
    assert backend.text_embed("nomic-1.5", ["a caption"]) == [[0.5]]
    assert worker.calls == [("embed_texts", (["a caption"],))]


def test_text_embed_falls_back_to_the_client_without_a_worker():
    from modelsvc.backends.text_backend import TextBackend

    client = _StubClient()
    backend = TextBackend(client, model_name=lambda: "qwen")
    backend.text_embed("text-embed-3", ["a caption"])
    assert client.embedded == ("text-embed-3", ["a caption"])


class _StubEmbed:
    def embed_image(self, images):
        return [[0.1] for _ in images]

    def embed_text(self, texts):
        return [[0.2] for _ in texts]

    def tag(self, image, dimensions):
        return {d: [] for d in dimensions}

    def calibration(self):
        return {"logit_scale": 1.0, "logit_bias": 2.0}


def _conveyor(costs, free=9999.0, concurrency=1):
    log = []
    reg = ModelRegistry()
    for name, cost in costs.items():
        reg.register(
            ModelSpec(name, (lambda n=name: log.append(("load", n))),
                      (lambda n=name: log.append(("free", n))), cost)
        )
    gov = MemoryGovernor(reg, measure_free_mb=lambda: free, budget_mb=6000)
    return reg, gov, Scheduler(gov, concurrency=concurrency), log


def test_embed_ops_route_through_scheduler_needing_the_image_embedder():
    reg, gov, sched, log = _conveyor({"image_embed": 1600})
    seen = []
    real = sched.run
    sched.run = lambda needed, prio, fn, **kw: (seen.append((list(needed), prio)), real(needed, prio, fn, **kw))[1]
    backend = CompositeBackend(embed=_StubEmbed(), registry=reg, governor=gov, scheduler=sched)
    assert backend.embed_image([b"x"]) == [[0.1]]
    assert seen[0][0] == ["image_embed"]
    assert ("load", "image_embed") in log  # governor loaded it via the spec


def test_composite_without_scheduler_runs_directly():
    backend = CompositeBackend(embed=_StubEmbed())
    assert backend.embed_image([b"x"]) == [[0.1]]
    assert backend.embed_text(["x"]) == [[0.2]]
    assert backend.tag(b"x", ["scene"]) == {"scene": []}
    assert backend.calibration() == {"logit_scale": 1.0, "logit_bias": 2.0}


def test_composite_backend_raises_when_caption_sub_backend_missing():
    backend = CompositeBackend(embed=_StubEmbed())
    with pytest.raises(NotImplementedError):
        backend.caption(b"x")


def test_composite_backend_raises_when_text_sub_backend_missing():
    backend = CompositeBackend(embed=_StubEmbed())
    with pytest.raises(NotImplementedError):
        backend.text_complete("m", [{"role": "user", "content": "hi"}])
    with pytest.raises(NotImplementedError):
        list(backend.text_stream("m", [{"role": "user", "content": "hi"}]))
    with pytest.raises(NotImplementedError):
        backend.text_embed("m", ["hello"])
    with pytest.raises(NotImplementedError):
        backend.text_warm("m")
    with pytest.raises(NotImplementedError):
        backend.text_evict("m")


def test_resources_reports_registry_resident_when_wired():
    reg, gov, sched, _ = _conveyor({"image_embed": 1600, "llm": 2200})
    gov.acquire(["image_embed"])
    backend = CompositeBackend(embed=_StubEmbed(), registry=reg, governor=gov, scheduler=sched)
    res = backend.resources()
    assert res["resident"] == ["image_embed"]  # from the registry, not a placeholder
    # Models and slots only — the machine metrics moved to `app` (§5.1).
    assert set(res) == {"resident", "active", "slots", "generation"}


def test_resources_falls_back_to_placeholder_without_registry():
    backend = CompositeBackend(embed=_StubEmbed())
    assert backend.resources()["resident"] == ["image_embed"]


class _RecordingScheduler:
    """Records what each op asked the conveyor for (needed units, priority)."""

    def __init__(self) -> None:
        self.seen: list[tuple[tuple[str, ...], object]] = []

    def run(self, needed, priority, fn):
        self.seen.append((tuple(needed), priority))
        return fn()

    def reap_idle(self, units):
        pass


class _Voice:
    def speak(self, text, voice, segments=None, voices=None):
        return {"audio": b"RIFF", "sample_rate": 24000, "model": "v"}


class _Music:
    def __init__(self):
        self.calls = []

    def music(self, prompt, seconds, seed, sections=None, temperature=None, guidance=None):
        self.calls.append({"prompt": prompt, "seconds": seconds, "sections": sections,
                           "temperature": temperature, "guidance": guidance})
        return {"audio": b"RIFF", "sample_rate": 44100, "model": "m", "segments": []}


def test_composite_runs_speak_on_the_voice_unit_at_batch_priority():
    from modelsvc.scheduler import Priority

    scheduler = _RecordingScheduler()
    backend = CompositeBackend(embed=None, voice=_Voice(), scheduler=scheduler)
    assert backend.speak("hi")["sample_rate"] == 24000
    assert scheduler.seen == [(("voice",), Priority.BATCH)]


def test_composite_runs_music_on_the_music_unit_at_batch_priority():
    from modelsvc.scheduler import Priority

    scheduler = _RecordingScheduler()
    backend = CompositeBackend(embed=None, music=_Music(), scheduler=scheduler)
    assert backend.music("warm guitar", 10.0)["sample_rate"] == 44100
    assert scheduler.seen == [(("music",), Priority.BATCH)]


def test_composite_raises_slot_off_when_no_voice_is_wired():
    from modelsvc.backends.base import SlotOff

    with pytest.raises(SlotOff):
        CompositeBackend(embed=None).speak("hi")


def test_composite_raises_slot_off_when_the_unit_is_unregistered():
    # A slot switched to `off` unregisters its residency unit (§4.1); asking the
    # scheduler for it would KeyError deep in the governor, so the composite must
    # check first and answer with the honest reason.
    class _Slots:
        def has_unit(self, unit):
            return False

    from modelsvc.backends.base import SlotOff

    backend = CompositeBackend(embed=None, music=_Music(), slots=_Slots())
    with pytest.raises(SlotOff):
        backend.music("x", 5.0)


class _Slots:
    """Just the two members CompositeBackend asks a SlotManager for."""

    def __init__(self, languages):
        self._languages = languages

    def has_unit(self, unit):
        return True

    def entry(self, slot):
        from models import catalog

        return catalog.get("voice", "kokoro-82m") if self._languages == ("en",) \
            else catalog.get("voice", "silero-v4-ru")


class _RoutingSlots:
    """A SlotManager stub that records which language it was asked to prepare for."""

    def __init__(self, can: bool = True):
        self.can = can
        self.asked: list[str] = []

    def has_unit(self, unit):
        return True

    def ensure_voice_for(self, language):
        self.asked.append(language)
        return self.can


def test_composite_asks_the_manager_to_route_the_voice_by_language():
    slots = _RoutingSlots()
    backend = CompositeBackend(embed=None, voice=_Voice(), slots=slots,
                               scheduler=_RecordingScheduler())
    assert backend.speak("привет", "ru")["sample_rate"] == 24000
    assert slots.asked == ["ru"]


def test_composite_refuses_only_when_no_voice_covers_the_language():
    from modelsvc.backends.base import SlotOff

    backend = CompositeBackend(embed=None, voice=_Voice(), slots=_RoutingSlots(can=False),
                               scheduler=_RecordingScheduler())
    with pytest.raises(SlotOff):
        backend.speak("hello", "en")


# --- the served context (§5.1, §11.1) ----------------------------------------------


class _ContextClient:
    def __init__(self, served):
        self._served = served

    def context_window(self):
        return self._served


def test_text_context_prefers_the_running_server():
    # The setting only takes effect on a restart, so the server is the authority.
    backend = TextBackend(_ContextClient(32768), context_default=4096)
    assert backend.text_context() == 32768


def test_text_context_falls_back_to_the_configured_value():
    backend = TextBackend(_ContextClient(None), context_default=4096)
    assert backend.text_context() == 4096


def test_text_context_is_none_when_nobody_can_say():
    assert TextBackend(object()).text_context() is None


def test_composite_passes_the_whole_score_to_the_music_backend():
    music = _Music()
    backend = CompositeBackend(embed=None, music=music)
    backend.music(sections=[{"brief": "piano", "seconds": 4}], temperature=1.2, guidance=4.0)
    assert music.calls[0]["sections"] == [{"brief": "piano", "seconds": 4}]
    assert music.calls[0]["temperature"] == 1.2 and music.calls[0]["guidance"] == 4.0


_REFS = {"saved:1": {"audio": b"RIFF", "sha256": "ab", "language": "en",
                     "transcript": None, "conditioning": []}}
_VOXCPM_REVISION = "32279effe8c19989596f05d353d1447f51d9e915"


class _CloningVoice:
    """A voice sub-backend that answers as the given model + revision."""

    def __init__(self, model="voxcpm2", revision=_VOXCPM_REVISION):
        self.reply = {"audio": b"RIFF", "sample_rate": 48000, "model": model,
                      "revision": revision, "conditioning": {}}
        self.calls = []

    def speak(self, text, voice, segments=None, voices=None, references=None):
        self.calls.append((text, voice, segments, voices, references))
        return self.reply


class _CloningSlots:
    """A SlotManager stub that routes a saved voice to `voxcpm2`, or to nothing."""

    def __init__(self, can: bool = True):
        self.can = can
        self.asked: list[tuple[str, bool]] = []

    def has_unit(self, unit):
        return True

    def ensure_voice_for(self, language, *, cloning=False):
        self.asked.append((language, cloning))
        return self.can

    def entry(self, slot):
        from models import catalog

        return catalog.get("voice", "voxcpm2")


def test_a_saved_voice_routes_to_the_cloning_entry_on_the_voice_unit_at_batch():
    from modelsvc.scheduler import Priority

    voice, slots, scheduler = _CloningVoice(), _CloningSlots(), _RecordingScheduler()
    backend = CompositeBackend(embed=None, voice=voice, slots=slots, scheduler=scheduler)
    result = backend.speak("hi", "ru", "saved:1", ["hi"], ["saved:1"], references=_REFS)
    assert result["model"] == "voxcpm2"
    assert slots.asked == [("ru", True)]
    assert scheduler.seen == [(("voice",), Priority.BATCH)]
    assert voice.calls == [("hi", "saved:1", ["hi"], ["saved:1"], _REFS)]


def test_a_preset_request_still_routes_without_the_cloning_flag():
    slots = _CloningSlots()
    backend = CompositeBackend(embed=None, voice=_Voice(), slots=slots,
                               scheduler=_RecordingScheduler())
    backend.speak("hi", "en")
    assert slots.asked == [("en", False)]


def test_no_voice_copy_for_the_language_is_cloning_unavailable():
    from modelsvc.backends.base import CloningUnavailable

    voice = _CloningVoice()
    backend = CompositeBackend(embed=None, voice=voice, slots=_CloningSlots(can=False),
                               scheduler=_RecordingScheduler(), profile="jetson")
    with pytest.raises(CloningUnavailable, match="no voice-copy model speaks en on jetson"):
        backend.speak("hi", "en", "saved:1", references=_REFS)
    assert voice.calls == []


@pytest.mark.parametrize(("model", "revision"), [
    ("kokoro-82m", None),                 # a preset child answered
    ("voxcpm2", "0" * 40),                # another Hub commit answered
    ("voxcpm2", None),
])
def test_a_reply_from_another_model_or_revision_is_refused(model, revision):
    backend = CompositeBackend(embed=None, voice=_CloningVoice(model, revision),
                               slots=_CloningSlots(), scheduler=_RecordingScheduler())
    with pytest.raises(RuntimeError, match="voice identity mismatch"):
        backend.speak("hi", "en", "saved:1", references=_REFS)


def test_without_a_slot_manager_references_pass_straight_to_the_voice():
    voice = _CloningVoice(model="anything", revision=None)
    backend = CompositeBackend(embed=None, voice=voice)
    assert backend.speak("hi", "en", "saved:1", references=_REFS)["model"] == "anything"
    assert voice.calls[0][4] is _REFS


def test_voice_backend_sends_references_only_when_given():
    from modelsvc.backends.voice_backend import VoiceBackend

    worker = _RecordingWorker()
    backend = VoiceBackend(lambda: worker)
    backend.speak("hi", "af_heart", ["hi"], [None])
    backend.speak("hi", "saved:1", ["hi"], ["saved:1"], _REFS)
    assert worker.calls == [
        ("speak", ("hi", "af_heart", ["hi"], [None])),
        ("speak", ("hi", "saved:1", ["hi"], ["saved:1"], _REFS)),
    ]


@pytest.mark.parametrize(("language", "preset"), [("en", "kokoro-82m"), ("ru", "silero-v4-ru")])
def test_fake_backend_never_speaks_a_preset_with_a_cloning_entry(language, preset):
    fake = FakeBackend("mac")
    fake.set_slots({"voice": "voxcpm2"})
    fake.speak("hi", language)
    assert fake.resources()["slots"]["voice"] == preset


def test_fake_backend_speaks_saved_voices_without_touching_the_slot():
    from storage.wav import duration_seconds

    fake = FakeBackend("mac")
    result = fake.speak("hi", "ru", "saved:1", references=_REFS)
    assert duration_seconds(result["audio"]) > 0
    assert result["model"] == "fake-voice" and result["revision"] is None
    assert result["conditioning"]["saved:1"]["state"] == "built"
    assert isinstance(result["conditioning"]["saved:1"]["blob"], bytes)
    assert fake.resources()["slots"]["voice"] == "kokoro-82m"


def _bakes():
    from modelsvc.activity import Activity
    from modelsvc.voice_bakes import VoiceBakes

    return VoiceBakes(slots=None, registry=None, governor=None, scheduler=None,
                      activity=Activity(), bake_root="unused", profile="mac")


def _drains(gate) -> bool:
    import threading

    gate.close()
    draining = threading.Thread(target=gate.drain, daemon=True)
    draining.start()
    draining.join(1.0)
    return not draining.is_alive()


def test_speak_raises_bake_running_while_the_gate_is_closed():
    from modelsvc.voice_bakes import BakeRunning

    bakes = _bakes()
    voice = _CloningVoice()
    backend = CompositeBackend(embed=None, voice=voice, bakes=bakes)
    bakes.gate.close()
    with pytest.raises(BakeRunning, match=r"a voice is being baked \(0%\)"):
        backend.speak("hi")
    with pytest.raises(BakeRunning):
        backend.speak("hi", "en", "saved:1", references=_REFS)
    assert voice.calls == []


def test_a_speak_that_raised_leaves_the_gate():
    class _Failing:
        def speak(self, *args):
            raise RuntimeError("the voice child died")

    bakes = _bakes()
    backend = CompositeBackend(embed=None, voice=_Failing(), bakes=bakes)
    with pytest.raises(RuntimeError, match="died"):
        backend.speak("hi")
    assert _drains(bakes.gate)


def test_model_unload_goes_through_the_governor_which_refuses_a_leased_unit():
    from modelsvc.governor import UnitLeased

    reg, gov, sched, log = _conveyor({"voice_train": 1000, "music": 100})
    gov.acquire(["voice_train", "music"])
    backend = CompositeBackend(embed=None, registry=reg, governor=gov, scheduler=sched)
    with gov.lease("voice_train"), pytest.raises(UnitLeased):
        backend.model_unload("voice_train")
    backend.model_unload("music")
    assert ("free", "voice_train") not in log
    assert ("free", "music") in log


class _StrictRecorder:
    """Records every method called on it, whatever its name."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self.calls.append(name)

        return record


class _StageBakes:
    """The two things `CompositeBackend` reads from the bakes, frozen at one stage."""

    def __init__(self, *, active: bool, gate_closed: bool):
        from modelsvc.voice_bakes import VoiceGate

        self._active = active
        self.gate = VoiceGate()
        if gate_closed:
            self.gate.close()

    def active(self) -> bool:
        return self._active


@pytest.mark.parametrize("bakes", [
    pytest.param(lambda: None, id="no-bakes"),
    pytest.param(lambda: _StageBakes(active=False, gate_closed=False), id="inactive"),
    pytest.param(lambda: _StageBakes(active=True, gate_closed=False), id="prepare"),
    pytest.param(lambda: _StageBakes(active=True, gate_closed=True), id="drain"),
    pytest.param(lambda: _StageBakes(active=True, gate_closed=True), id="teardown"),
])
@pytest.mark.parametrize("dependency", ["scheduler", "registry"])
def test_voice_train_ensure_always_raises_bake_only_and_touches_nothing(bakes, dependency):
    from modelsvc.voice_bakes import BakeOnly

    recorder = _StrictRecorder()
    backend = CompositeBackend(embed=None, bakes=bakes(), **{dependency: recorder})
    with pytest.raises(BakeOnly, match="voice_train is loaded only by a bake"):
        backend.model_ensure("voice_train")
    assert recorder.calls == []


def test_without_bakes_a_bake_is_not_offered():
    from modelsvc.voice_bakes import BakeNotOffered

    backend = CompositeBackend(embed=None)
    with pytest.raises(BakeNotOffered):
        backend.bake_start({"clips": []})
    assert backend.bake_current() is None
    assert backend.bake_status("x") is None
    assert backend.bake_delete("x") is False


def test_the_fake_scripted_bake_walks_from_running_to_done_and_returns_its_bytes():
    from modelsvc.backends.fake import FAKE_BAKE_ADAPTER
    from modelsvc.voice_bakes import BakeRunning

    fake = FakeBackend("mac")
    bake_id = fake.bake_start({"clips": [{}], "tag": {"check": "me"}})["id"]
    with pytest.raises(BakeRunning):
        fake.bake_adapter(bake_id)
    states = [fake.bake_status(bake_id)["state"] for _ in range(3)]
    assert states == ["running", "running", "done"]
    assert fake.bake_adapter(bake_id) == FAKE_BAKE_ADAPTER
    assert fake.bake_current()["tag"] == {"check": "me"}
    assert fake.bake_delete(bake_id) is True
    assert fake.bake_current() is None


def test_the_fake_bake_can_be_cancelled_and_is_not_offered_on_jetson():
    from modelsvc.voice_bakes import BakeNotOffered

    fake = FakeBackend("mac")
    bake_id = fake.bake_start({"clips": [{}]})["id"]
    assert fake.bake_cancel(bake_id)["state"] == "cancelled"
    assert fake.bake_status(bake_id)["state"] == "cancelled"
    with pytest.raises(BakeNotOffered):
        FakeBackend("jetson").bake_start({"clips": [{}]})


def test_build_backend_wires_the_bakes_and_sweeps_their_old_folders(tmp_path):
    from modelsvc.voice_bakes import VoiceBakes

    leftover = tmp_path / "voice-bakes" / "0123abcd" / "clips"
    leftover.mkdir(parents=True)
    backend = build_backend(Settings(data_dir=tmp_path, profile="mac"))
    assert isinstance(backend._bakes, VoiceBakes)
    assert backend._gate is backend._bakes.gate
    assert list((tmp_path / "voice-bakes").iterdir()) == []


def test_the_fake_speaks_a_model_only_reference_with_no_conditioning():
    fake = FakeBackend("mac")
    refs = {"saved:1": {"language": "ru", "adapter": {"blob": b"lora", "sha256": "cd"},
                        "use_clip": False},
            "saved:2": _REFS["saved:1"]}
    result = fake.speak("hi", "ru", "saved:1", voices=["saved:2"], references=refs)
    assert set(result["conditioning"]) == {"saved:2"}
    assert result["conditioning"]["saved:2"]["state"] == "built"
    assert result["adapters"] == {"saved:1": {"state": "loaded", "sha256": "cd"}}


_WHISPER_REV = "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"


class _AsrChild:
    def __init__(self, revision=_WHISPER_REV):
        self.revision = revision
        self.calls = []

    def call(self, method, *args):
        self.calls.append((method, *args))
        return {"text": " heard words ", "revision": self.revision, "device": "mps",
                "seconds": 6.0, "elapsed_s": 0.5}


class _AsrSlots:
    def __init__(self, child, offered=True):
        self.child, self.offered = child, offered

    def has_unit(self, unit):
        return self.offered and unit == "asr"

    def worker(self, unit):
        assert unit == "asr"
        return self.child


class _AsrRegistry:
    def __init__(self, resident=()):
        self.resident = set(resident)

    def is_resident(self, name):
        return name in self.resident


class _AsrGovernor:
    def __init__(self, states):
        self.states = list(states)

    def state(self):
        import types

        resident, free_mb = self.states.pop(0) if len(self.states) > 1 else self.states[0]
        return types.SimpleNamespace(resident=list(resident), free_mb=free_mb)


class _LoadingScheduler(_RecordingScheduler):
    """Makes the needed units resident before the op, as the conveyor does."""

    def __init__(self, registry):
        super().__init__()
        self.registry = registry

    def run(self, needed, priority, fn):
        self.registry.resident.update(needed)
        return super().run(needed, priority, fn)


def _asr_backend(*, child=None, offered=True, resident=(), states=(((), 20000.0),),
                 bakes=None, profile="mac"):
    registry = _AsrRegistry(resident)
    scheduler = _LoadingScheduler(registry)
    backend = CompositeBackend(embed=None, slots=_AsrSlots(child or _AsrChild(), offered),
                               registry=registry, governor=_AsrGovernor(states),
                               scheduler=scheduler, bakes=bakes, profile=profile)
    return backend, scheduler, registry


def test_transcribe_runs_the_asr_unit_at_batch_and_names_the_entry():
    from modelsvc.scheduler import Priority

    child = _AsrChild()
    backend, scheduler, _ = _asr_backend(child=child)

    result = backend.transcribe(b"take", "ru")

    assert scheduler.seen == [(("asr",), Priority.BATCH)]
    assert child.calls == [("transcribe", b"take", "ru")]
    assert result == {"text": " heard words ", "model": "whisper-large-v3-turbo",
                      "revision": _WHISPER_REV, "seconds": 6.0, "elapsed_s": 0.5}


def test_transcribe_with_the_gate_closed_is_bake_running_and_never_scheduled():
    from modelsvc.voice_bakes import BakeRunning

    bakes = _bakes()
    backend, scheduler, _ = _asr_backend(bakes=bakes)
    bakes.gate.close()
    with pytest.raises(BakeRunning, match="a voice is being baked"):
        backend.transcribe(b"take", "en")
    assert scheduler.seen == []


def test_a_transcribe_that_raised_leaves_the_gate():
    class _Dying(_AsrChild):
        def call(self, method, *args):
            raise RuntimeError("Whisper weights for x are not on disk: run `make models`")

    bakes = _bakes()
    backend, _, _ = _asr_backend(child=_Dying(), bakes=bakes)
    with pytest.raises(RuntimeError, match="not on disk"):
        backend.transcribe(b"take", "en")
    assert _drains(bakes.gate)


def test_a_reply_of_another_revision_is_an_identity_mismatch():
    backend, _, _ = _asr_backend(child=_AsrChild(revision="other"))
    with pytest.raises(RuntimeError, match="asr identity mismatch"):
        backend.transcribe(b"take", "en")


@pytest.mark.parametrize(("offered", "profile"), [(False, "mac"), (True, "jetson")])
def test_no_asr_unit_is_transcribe_not_offered(offered, profile):
    from modelsvc.backends.base import TranscribeNotOffered

    backend, scheduler, _ = _asr_backend(offered=offered, profile=profile)
    with pytest.raises(TranscribeNotOffered, match="not offered on this device yet"):
        backend.transcribe(b"take", "en")
    assert scheduler.seen == []


def test_no_slot_manager_is_transcribe_not_offered():
    from modelsvc.backends.base import TranscribeNotOffered

    with pytest.raises(TranscribeNotOffered):
        CompositeBackend(embed=None).transcribe(b"take", "en")


def test_a_language_the_entry_does_not_list_is_refused_before_any_load():
    from modelsvc.backends.base import TranscribeRefused

    backend, scheduler, _ = _asr_backend()
    with pytest.raises(TranscribeRefused, match="'de'"):
        backend.transcribe(b"take", "de")
    assert scheduler.seen == []


_FORBIDDEN = ("cost", "drop", "measured")


def test_a_call_that_loads_the_child_logs_its_net_free_memory_change(caplog):
    backend, _, _ = _asr_backend(states=((["voice"], 20000.0), (["voice", "asr"], 16500.0)))
    with caplog.at_level("INFO", logger="modelsvc.backends.composite"):
        backend.transcribe(b"take", "en")
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("asr ")]
    assert lines == [("asr whisper-large-v3-turbo@41f01f3 loaded by this request: "
                      "free_mb 20000 → 16500, net change -3500 MB; evicted: none")]
    assert not any(word in lines[0] for word in _FORBIDDEN)


def test_the_load_line_names_the_units_the_governor_evicted(caplog):
    backend, _, _ = _asr_backend(
        states=((["voice", "music"], 9000.0), (["voice", "asr"], 7000.0)))
    with caplog.at_level("INFO", logger="modelsvc.backends.composite"):
        backend.transcribe(b"take", "en")
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("asr ")]
    assert len(lines) == 1
    assert "loaded by this request" in lines[0]
    assert "net change -2000 MB" in lines[0]
    assert lines[0].endswith("evicted: music")
    assert not any(word in lines[0] for word in _FORBIDDEN)


def test_a_call_to_a_resident_child_logs_nothing(caplog):
    backend, _, _ = _asr_backend(resident={"asr"})
    with caplog.at_level("INFO", logger="modelsvc.backends.composite"):
        backend.transcribe(b"take", "en")
    assert not [r for r in caplog.records if r.getMessage().startswith("asr ")]


def test_a_manual_asr_ensure_enters_the_gate():
    from modelsvc.voice_bakes import BakeRunning

    bakes = _bakes()
    scheduler = _RecordingScheduler()
    backend = CompositeBackend(embed=None, scheduler=scheduler, bakes=bakes)

    entered = []
    real_enter = bakes.gate.enter
    bakes.gate.enter = lambda: (entered.append(True), real_enter())[1]
    backend.model_ensure("asr")
    assert entered == [True] and len(scheduler.seen) == 1
    assert _drains(bakes.gate)

    with pytest.raises(BakeRunning):
        backend.model_ensure("asr")
    assert len(scheduler.seen) == 1


def test_the_fake_transcribes_a_scripted_text_per_take():
    import hashlib
    import io
    import wave

    from modelsvc.backends.base import TranscribeNotOffered, TranscribeRefused

    def take(sample: bytes) -> bytes:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(48000)
            handle.writeframes(sample * 48000 * 6)
        return buffer.getvalue()

    fake = FakeBackend("mac")
    scripted, other = take(b"\x01\x00"), take(b"\x02\x00")
    fake.transcripts[hashlib.sha256(scripted).hexdigest()] = "♪"
    assert fake.transcribe(scripted, "ru") == {
        "text": "♪", "model": "fake-asr", "revision": "fake", "seconds": 6.0, "elapsed_s": 0.0}
    assert fake.transcribe(other, "en")["text"] == "fake words"
    with pytest.raises(TranscribeRefused):
        fake.transcribe(other, "de")
    with pytest.raises(TranscribeNotOffered):
        FakeBackend("jetson").transcribe(other, "en")


_ECAPA_REV = "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"


class _SpeakerChild:
    def __init__(self, reply=None):
        self.reply = reply or {"vector": [0.5] * 192, "revision": _ECAPA_REV, "seconds": 40.0,
                               "elapsed_s": 0.2, "pipeline": 1}
        self.calls = []

    def call(self, method, *args):
        self.calls.append((method, *args))
        return self.reply


class _SpeakerSlots:
    def __init__(self, child, offered=True):
        self.child, self.offered = child, offered

    def has_unit(self, unit):
        return self.offered and unit == "speaker_embed"

    def worker(self, unit):
        assert unit == "speaker_embed"
        return self.child


def _speaker_backend(*, child=None, offered=True, bakes=None, profile="mac"):
    scheduler = _RecordingScheduler()
    backend = CompositeBackend(embed=None, slots=_SpeakerSlots(child or _SpeakerChild(), offered),
                               scheduler=scheduler, bakes=bakes, profile=profile)
    return backend, scheduler


def test_embed_voice_runs_the_speaker_unit_at_batch_and_names_the_entry():
    from modelsvc.scheduler import Priority

    child = _SpeakerChild()
    backend, scheduler = _speaker_backend(child=child)

    result = backend.embed_voice(b"clip")

    assert scheduler.seen == [(("speaker_embed",), Priority.BATCH)]
    assert child.calls == [("embed", b"clip")]
    assert result == {"vector": [0.5] * 192, "model": "ecapa-voxceleb", "revision": _ECAPA_REV,
                      "seconds": 40.0, "elapsed_s": 0.2, "pipeline": 1}


def test_embed_voice_with_the_gate_closed_is_bake_running_and_never_scheduled():
    from modelsvc.voice_bakes import BakeRunning

    bakes = _bakes()
    assert _drains(bakes.gate)
    backend, scheduler = _speaker_backend(bakes=bakes)

    with pytest.raises(BakeRunning):
        backend.embed_voice(b"clip")
    assert scheduler.seen == []


def test_embed_voice_is_counted_in_at_the_gate_and_leaves_it_when_it_raises():
    class _Dying(_SpeakerChild):
        def call(self, method, *args):
            raise RuntimeError("the child died")

    bakes = _bakes()
    entered = []
    real_enter = bakes.gate.enter
    bakes.gate.enter = lambda: (entered.append(True), real_enter())[1]
    backend, _ = _speaker_backend(child=_Dying(), bakes=bakes)

    with pytest.raises(RuntimeError, match="the child died"):
        backend.embed_voice(b"clip")

    assert entered == [True]
    assert _drains(bakes.gate)


def test_a_refusal_of_the_child_is_speaker_embed_refused():
    from modelsvc.backends.base import SpeakerEmbedRefused

    child = _SpeakerChild({"refused": "the clip holds 0.20 s of voiced audio",
                           "revision": _ECAPA_REV})
    backend, _ = _speaker_backend(child=child)
    with pytest.raises(SpeakerEmbedRefused, match="0.20 s of voiced audio"):
        backend.embed_voice(b"clip")


def test_another_revision_is_a_speaker_embed_identity_mismatch():
    child = _SpeakerChild({"vector": [0.5] * 192, "revision": "other", "seconds": 6.0,
                           "elapsed_s": 0.2, "pipeline": 1})
    backend, _ = _speaker_backend(child=child)
    with pytest.raises(RuntimeError, match="speaker_embed identity mismatch"):
        backend.embed_voice(b"clip")


def test_a_vector_of_another_size_is_an_error():
    child = _SpeakerChild({"vector": [0.5] * 7, "revision": _ECAPA_REV, "seconds": 6.0,
                           "elapsed_s": 0.2, "pipeline": 1})
    backend, _ = _speaker_backend(child=child)
    with pytest.raises(RuntimeError, match="7 values, not 192"):
        backend.embed_voice(b"clip")


@pytest.mark.parametrize(("offered", "profile"), [(False, "mac"), (True, "jetson")])
def test_no_speaker_unit_is_speaker_embed_not_offered(offered, profile):
    from modelsvc.backends.base import SpeakerEmbedNotOffered

    backend, scheduler = _speaker_backend(offered=offered, profile=profile)
    with pytest.raises(SpeakerEmbedNotOffered, match="not offered on this device yet"):
        backend.embed_voice(b"clip")
    assert scheduler.seen == []


def test_a_manual_speaker_ensure_enters_the_gate():
    from modelsvc.voice_bakes import BakeRunning

    bakes = _bakes()
    scheduler = _RecordingScheduler()
    backend = CompositeBackend(embed=None, scheduler=scheduler, bakes=bakes)

    entered = []
    real_enter = bakes.gate.enter
    bakes.gate.enter = lambda: (entered.append(True), real_enter())[1]
    backend.model_ensure("speaker_embed")
    assert entered == [True] and len(scheduler.seen) == 1
    assert _drains(bakes.gate)

    with pytest.raises(BakeRunning):
        backend.model_ensure("speaker_embed")
    assert len(scheduler.seen) == 1


def _clip(sample: bytes, seconds: int = 40) -> bytes:
    import io
    import wave

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(48000)
        handle.writeframes(sample * 48000 * seconds)
    return buffer.getvalue()


def test_the_fake_embeds_each_clip_stably_with_no_30_s_cap():
    fake = FakeBackend("mac")
    first, other = _clip(b"\x01\x00"), _clip(b"\x02\x00")

    result = fake.embed_voice(first)

    assert len(result["vector"]) == 192
    assert result["seconds"] == 40.0
    assert (result["model"], result["revision"], result["pipeline"]) == ("fake-speaker", "fake", 1)
    assert fake.embed_voice(first)["vector"] == result["vector"]
    assert fake.embed_voice(other)["vector"] != result["vector"]


def test_the_fake_answers_a_scripted_vector_or_refusal_per_clip():
    import hashlib

    from modelsvc.backends.base import SpeakerEmbedNotOffered, SpeakerEmbedRefused

    fake = FakeBackend("mac")
    scripted, refused = _clip(b"\x01\x00", 6), _clip(b"\x02\x00", 6)
    fake.speaker_vectors[hashlib.sha256(scripted).hexdigest()] = [1.0] * 192
    fake.speaker_refusals[hashlib.sha256(refused).hexdigest()] = "too little voiced audio"

    assert fake.embed_voice(scripted)["vector"] == [1.0] * 192
    with pytest.raises(SpeakerEmbedRefused, match="too little voiced audio"):
        fake.embed_voice(refused)
    with pytest.raises(SpeakerEmbedNotOffered):
        FakeBackend("jetson").embed_voice(scripted)
