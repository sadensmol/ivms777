"""`build_backend` (config -> `ModelBackend`) and `CompositeBackend` routing
through the conveyor (design §5.1, plan 18). The real SigLIP/nomic/gemma loads
are lazy, so building the backend stays torch-free and never spawns llama-server.
"""

import pytest

from config import Settings
from modelsvc.backends import build_backend
from modelsvc.backends.composite import CompositeBackend
from modelsvc.backends.fake import FakeBackend
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
    assert backend._caption._captioner._client is backend._text._client


def test_build_backend_wires_conveyor_on_jetson(tmp_path):
    backend = build_backend(Settings(data_dir=tmp_path, profile="jetson"))
    assert isinstance(backend._scheduler, Scheduler)
    assert backend._scheduler.slots == 1  # jetson serializes the GPU
    # SigLIP + nomic + gemma all registered as managed models.
    assert {"siglip", "nomic", "gemma"} <= set(backend._registry._specs)
    assert backend._text_embed_needs == ("nomic",)


def test_gemma_specs_probe_the_supervised_child_for_liveness(tmp_path):
    # gemma is a separate process that can abort on its own (CUDA OOM at image
    # decode). Both gemma specs must carry a liveness probe bound to that child, so
    # a dead one is respawned instead of 500-ing forever (§8.1).
    backend = build_backend(Settings(data_dir=tmp_path, profile="jetson"))
    specs = backend._registry._specs
    for name in ("gemma", "gemma-vision"):
        assert specs[name].alive is not None
        assert specs[name].alive() is False  # nothing spawned yet


def test_torch_models_are_registered_as_killable_workers(tmp_path):
    # Evicting SigLIP/nomic in-process does NOT return their memory (§8.1), so both
    # live in a child the registry kills — and both carry its liveness probe.
    backend = build_backend(Settings(data_dir=tmp_path, profile="jetson"))
    for name in ("siglip", "nomic"):
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


def test_build_backend_cloud_has_no_nomic_and_remote_llm(tmp_path):
    backend = build_backend(Settings(data_dir=tmp_path, profile="cloud"))
    assert "nomic" not in backend._registry._specs  # cloud uses remote /embeddings
    assert {"siglip", "gemma"} <= set(backend._registry._specs)
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
    backend = SiglipBackend(worker)
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
    backend = TextBackend(_StubClient(), text_worker=worker, model_name="gemma4-E2B")
    assert backend.text_embed("nomic", ["a caption"]) == [[0.5]]
    assert worker.calls == [("embed_texts", (["a caption"],))]


def test_text_embed_falls_back_to_the_client_without_a_worker():
    from modelsvc.backends.text_backend import TextBackend

    client = _StubClient()
    backend = TextBackend(client, model_name="qwen")
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


def test_embed_ops_route_through_scheduler_needing_siglip():
    reg, gov, sched, log = _conveyor({"siglip": 1600})
    seen = []
    real = sched.run
    sched.run = lambda needed, prio, fn, **kw: (seen.append((list(needed), prio)), real(needed, prio, fn, **kw))[1]
    backend = CompositeBackend(embed=_StubEmbed(), registry=reg, governor=gov, scheduler=sched)
    assert backend.embed_image([b"x"]) == [[0.1]]
    assert seen[0][0] == ["siglip"]
    assert ("load", "siglip") in log  # governor loaded it via the spec


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
    reg, gov, sched, _ = _conveyor({"siglip": 1600, "gemma": 2200})
    gov.acquire(["siglip"])
    backend = CompositeBackend(embed=_StubEmbed(), registry=reg, governor=gov, scheduler=sched)
    res = backend.resources()
    assert res["resident"] == ["siglip"]  # from the registry, not a placeholder
    # Models only — the machine metrics moved to `app` (§5.1).
    assert set(res) == {"resident", "active"}


def test_resources_falls_back_to_placeholder_without_registry():
    backend = CompositeBackend(embed=_StubEmbed())
    assert backend.resources()["resident"] == ["siglip"]
