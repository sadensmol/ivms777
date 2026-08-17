"""`build_backend` (config -> `ModelBackend`) and `CompositeBackend` delegation
(design §5.1, plan 15 task 2). The real SigLIP path is board-only and never
instantiated here — `use_fake_embedder=True` is the only path exercised.
"""

from contextlib import contextmanager

import pytest

from config import Settings
from modelsvc.backends import build_backend
from modelsvc.backends.composite import CompositeBackend
from modelsvc.backends.fake import FakeBackend
from modelsvc.backends.siglip_backend import SiglipBackend
from modelsvc.residency import HIGH, Residency


class _RecordingResidency:
    """Duck-typed `Residency` double: records `register`/`use` calls without
    touching a real model — proves `SiglipBackend` wires itself into
    residency correctly (concurrency semantics live in tests/test_residency.py)."""

    def __init__(self):
        self.registered: dict[str, tuple] = {}
        self.use_calls: list[tuple[str, int]] = []

    def register(self, name, load, release):
        self.registered[name] = (load, release)

    @contextmanager
    def use(self, name, *, priority):
        self.use_calls.append((name, priority))
        yield

    def should_preempt(self):
        return False

    def resident(self):
        return list(self.registered)


def test_build_backend_returns_fake_backend_when_use_fake_embedder(tmp_path):
    settings = Settings(data_dir=tmp_path, use_fake_embedder=True)
    assert isinstance(build_backend(settings), FakeBackend)


def test_build_backend_real_path_wires_text_and_shares_one_inference_client(tmp_path):
    # Construct-only, no network: SiglipBackend/OpenAICaptioner/TextBackend are
    # all lazy about actually calling out. Proves the real path wires
    # text=TextBackend and that caption + text share ONE inference client
    # (the single llama-server/vLLM gateway, design §5.1).
    settings = Settings(data_dir=tmp_path, profile="mac")
    backend = build_backend(settings)
    assert isinstance(backend, CompositeBackend)
    assert backend._text is not None
    assert backend._caption is not None
    assert backend._caption._captioner._client is backend._text._client


class _StubEmbed:
    def embed_image(self, images):
        return [[0.1] for _ in images]

    def embed_text(self, texts):
        return [[0.2] for _ in texts]

    def tag(self, image, dimensions):
        return {d: [] for d in dimensions}

    def calibration(self):
        return {"logit_scale": 1.0, "logit_bias": 2.0}


def test_composite_backend_delegates_embed_calls_to_the_embed_sub_backend():
    backend = CompositeBackend(embed=_StubEmbed())
    assert backend.embed_image([b"x"]) == [[0.1]]
    assert backend.embed_text(["x"]) == [[0.2]]
    assert backend.tag(b"x", ["scene"]) == {"scene": []}
    assert backend.calibration() == {"logit_scale": 1.0, "logit_bias": 2.0}


def test_composite_backend_raises_when_caption_sub_backend_missing():
    backend = CompositeBackend(embed=_StubEmbed())
    with pytest.raises(NotImplementedError):
        backend.caption(b"x", ["scene"])


def test_composite_backend_raises_when_text_sub_backend_missing():
    backend = CompositeBackend(embed=_StubEmbed())
    with pytest.raises(NotImplementedError):
        backend.text_complete("m", [{"role": "user", "content": "hi"}])
    with pytest.raises(NotImplementedError):
        backend.text_stream("m", [{"role": "user", "content": "hi"}])
    with pytest.raises(NotImplementedError):
        backend.text_embed("m", ["hello"])
    with pytest.raises(NotImplementedError):
        backend.text_warm("m")
    with pytest.raises(NotImplementedError):
        backend.text_evict("m")


def test_composite_backend_resources_reports_siglip_resident():
    backend = CompositeBackend(embed=_StubEmbed())
    resources = backend.resources()
    assert resources["resident"] == ["siglip"]
    assert resources["ram_total_mb"] > 0
    assert isinstance(resources["cpu_pct"], float)
    # GPU load is read from the box's own tool (tegrastats/nvidia-smi); absent on
    # a machine with neither, so the reader returns None (§13).
    assert resources["gpu_pct"] is None or isinstance(resources["gpu_pct"], float)


def test_composite_backend_resources_reports_the_residency_resident_list_when_wired():
    residency = Residency()
    residency.register("siglip", lambda: None, lambda: None)
    with residency.use("siglip", priority=1):
        pass

    backend = CompositeBackend(embed=_StubEmbed(), residency=residency)

    assert backend.resources()["resident"] == ["siglip"]  # from the residency, not the placeholder


def test_composite_backend_resources_falls_back_to_the_placeholder_without_residency():
    # No residency injected -> unchanged Task-2 behavior.
    backend = CompositeBackend(embed=_StubEmbed())
    assert backend.resources()["resident"] == ["siglip"]


def test_siglip_backend_registers_itself_with_residency_on_construction():
    residency = _RecordingResidency()

    backend = SiglipBackend("siglip2-so400m-patch14-384", "cpu", residency=residency)

    assert "siglip" in residency.registered
    load, release = residency.registered["siglip"]
    assert callable(load) and callable(release)
    assert backend is not None  # constructed without touching torch


def test_siglip_backend_wraps_embed_text_in_residency_use_at_high_priority():
    residency = _RecordingResidency()
    backend = SiglipBackend("siglip2-so400m-patch14-384", "cpu", residency=residency)

    class _FakeEmbedder:
        def embed_texts(self, texts):
            return [[0.1] for _ in texts]

    backend._embedder = lambda: _FakeEmbedder()  # bypass the real (torch) embedder

    vectors = backend.embed_text(["cat"])

    assert vectors == [[0.1]]
    assert residency.use_calls == [("siglip", HIGH)]


def test_siglip_backend_wraps_embed_image_in_residency_use():
    from tests.fixtures import jpeg_bytes

    residency = _RecordingResidency()
    backend = SiglipBackend("siglip2-so400m-patch14-384", "cpu", residency=residency)

    class _FakeEmbedder:
        def embed_images(self, images):
            return [[0.2] for _ in images]

    backend._embedder = lambda: _FakeEmbedder()

    vectors = backend.embed_image([jpeg_bytes()])

    assert vectors == [[0.2]]
    assert residency.use_calls == [("siglip", HIGH)]


def test_siglip_backend_wraps_calibration_in_residency_use():
    residency = _RecordingResidency()
    backend = SiglipBackend("siglip2-so400m-patch14-384", "cpu", residency=residency)

    class _FakeEmbedder:
        logit_scale = 10.0
        logit_bias = -5.0

    backend._embedder = lambda: _FakeEmbedder()

    assert backend.calibration() == {"logit_scale": 10.0, "logit_bias": -5.0}
    assert residency.use_calls == [("siglip", HIGH)]


def test_siglip_backend_without_residency_behaves_as_before():
    backend = SiglipBackend("siglip2-so400m-patch14-384", "cpu")

    class _FakeEmbedder:
        def embed_texts(self, texts):
            return [[0.3] for _ in texts]

    backend._embedder = lambda: _FakeEmbedder()

    assert backend.embed_text(["dog"]) == [[0.3]]


def test_build_backend_wires_a_non_exclusive_residency_on_jetson(tmp_path):
    # Since plan 16 there is no in-process caption VLM, so residency is always
    # non-exclusive and SigLIP is the only model wired into it (the captioner is a
    # remote OpenAI call).
    settings = Settings(data_dir=tmp_path, profile="jetson", use_fake_inference=True)
    backend = build_backend(settings)
    assert isinstance(backend, CompositeBackend)
    assert backend._residency is not None
    assert backend._residency._exclusive is False
    assert backend._embed._residency is backend._residency


def test_build_backend_wires_a_non_exclusive_residency_on_mac(tmp_path):
    settings = Settings(data_dir=tmp_path, profile="mac")
    backend = build_backend(settings)
    assert backend._residency._exclusive is False
    assert backend._embed._residency is backend._residency
