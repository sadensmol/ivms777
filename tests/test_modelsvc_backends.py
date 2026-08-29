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
    # One managed unit per slot (design §4.1). There is no "off" model any more, so
    # every slot has one.
    assert set(backend._registry._specs) == {
        "image_embed", "text_embed", "llm", "llm_vision", "voice", "music",
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
