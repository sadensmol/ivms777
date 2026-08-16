"""`CaptionBackend` + `build_caption_backend` (design §5.1, plan 15 task 3) — the
`models` service's caption sub-backend. `modelsvc/` is the ONE process allowed
to load a captioner; `CaptionBackend` wraps whichever `Captioner` the profile
selects (`captioning/`) and ensures it is loaded before use. Construction-only
here — `VLMCaptioner` is never `.load()`-ed, so this stays torch-free (same
guard as tests/test_captioner_vlm.py / tests/test_remote_embedder.py)."""

import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

from captioning.base import CaptionResult
from captioning.ollama_adapter import OllamaCaptioner
from captioning.vlm_adapter import VLMCaptioner
from config import Settings
from inference.client import InferenceCancelled
from modelsvc.backends import build_caption_backend
from modelsvc.backends.caption_backend import CaptionBackend
from modelsvc.residency import LOW, CaptionPreempted

_REPO_ROOT = Path(__file__).resolve().parent.parent


class FakeCaptioner:
    caption_model = "fake-vlm"

    def __init__(self):
        self.load_calls = 0

    def load(self) -> None:
        self.load_calls += 1

    def release(self) -> None:
        pass

    def footprint_mb(self) -> int:
        return 0

    def caption(self, image, dimensions, *, should_preempt=lambda: False):
        self.last_should_preempt = should_preempt
        return CaptionResult(
            caption="a cat", title="Cat", description="a cat photo", tags={"subject": ["cat"]}
        )


class _CancellingCaptioner(FakeCaptioner):
    """Simulates a captioner whose generation was force-preempted (§8.1) —
    `VLMCaptioner`'s StoppingCriteria / `OllamaCaptioner`'s should_stop watcher
    both raise `InferenceCancelled` the same way."""

    def caption(self, image, dimensions, *, should_preempt=lambda: False):
        raise InferenceCancelled("fake-vlm")


class _RecordingResidency:
    """A minimal duck-typed `Residency` double: records `register`/`use` calls
    without touching any real model — used to prove `CaptionBackend` wires
    itself into residency correctly, independent of `Residency`'s own
    concurrency semantics (covered by tests/test_residency.py)."""

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


def test_caption_backend_returns_the_dict_with_the_model_key():
    captioner = FakeCaptioner()
    backend = CaptionBackend(captioner)

    result = backend.caption(b"image-bytes", ["subject"])

    assert result == {
        "caption": "a cat",
        "title": "Cat",
        "description": "a cat photo",
        "tags": {"subject": ["cat"]},
        "model": "fake-vlm",
    }


def test_caption_backend_loads_the_captioner_before_first_use():
    captioner = FakeCaptioner()
    backend = CaptionBackend(captioner)

    backend.caption(b"image-bytes", [])

    assert captioner.load_calls == 1


def test_caption_backend_loads_only_once_across_calls():
    captioner = FakeCaptioner()
    backend = CaptionBackend(captioner)

    backend.caption(b"one", [])
    backend.caption(b"two", [])

    assert captioner.load_calls == 1


def test_caption_backend_registers_the_captioner_with_residency_on_construction():
    captioner = FakeCaptioner()
    residency = _RecordingResidency()

    CaptionBackend(captioner, residency=residency)

    assert residency.registered["caption"] == (captioner.load, captioner.release)


def test_caption_backend_wraps_caption_in_residency_use_at_low_priority():
    captioner = FakeCaptioner()
    residency = _RecordingResidency()
    backend = CaptionBackend(captioner, residency=residency)

    backend.caption(b"image-bytes", [])

    assert residency.use_calls == [("caption", LOW)]


def test_caption_backend_passes_residency_should_preempt_to_the_captioner():
    captioner = FakeCaptioner()
    residency = _RecordingResidency()
    backend = CaptionBackend(captioner, residency=residency)

    backend.caption(b"image-bytes", [])

    # Bound methods aren't `is`-identical across accesses; `==` is the right
    # check (same underlying function, same bound instance).
    assert captioner.last_should_preempt == residency.should_preempt


def test_caption_backend_maps_inference_cancelled_to_caption_preempted():
    captioner = _CancellingCaptioner()
    residency = _RecordingResidency()
    backend = CaptionBackend(captioner, residency=residency)

    with pytest.raises(CaptionPreempted):
        backend.caption(b"image-bytes", [])


def test_caption_backend_without_residency_behaves_as_before():
    # No residency injected (e.g. FakeBackend path) -> no wrapping, no
    # should_preempt threaded through; exactly the pre-Task-5 behavior.
    captioner = FakeCaptioner()
    backend = CaptionBackend(captioner)

    result = backend.caption(b"image-bytes", [])

    assert result["caption"] == "a cat"
    assert captioner.load_calls == 1


def test_build_caption_backend_selects_vlm_captioner_for_inprocess(tmp_path):
    s = Settings(profile="jetson", data_dir=tmp_path, use_fake_embedder=True)
    backend = build_caption_backend(s)
    assert isinstance(backend, CaptionBackend)
    assert isinstance(backend._captioner, VLMCaptioner)
    assert backend._captioner._model is None  # constructed only — never loaded


def test_build_caption_backend_selects_ollama_captioner_for_mac(tmp_path):
    s = Settings(profile="mac", data_dir=tmp_path, use_fake_embedder=True)
    backend = build_caption_backend(s)
    assert isinstance(backend._captioner, OllamaCaptioner)
    assert backend._captioner.caption_model == s.caption_model


def test_build_caption_backend_fake_path_forces_ollama_over_fake_client(tmp_path):
    s = Settings(profile="jetson", data_dir=tmp_path, use_fake_embedder=True, use_fake_inference=True)
    backend = build_caption_backend(s)
    assert isinstance(backend._captioner, OllamaCaptioner)  # fake always wins over inprocess
    assert backend._captioner.caption_model == "fake"


def test_importing_caption_backend_and_constructing_a_vlm_captioner_is_torch_free():
    code = (
        "import modelsvc.backends.caption_backend; "
        "from captioning.vlm_adapter import VLMCaptioner; "
        "VLMCaptioner('Qwen/Qwen2.5-VL-3B-Instruct'); "
        "import sys; assert 'torch' not in sys.modules, 'torch imported'"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
