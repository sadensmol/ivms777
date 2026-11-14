"""The bake's Whisper child (`audio/whisper_asr.py`, design §5.1 `/voice/transcribe`).

Fakes only: no torch, no transformers, no scipy, no model load.
"""

import io
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

from audio.whisper_asr import RATE_ASR, RATE_IN, WhisperASR, take_samples


def _wav(seconds: float = 6.0, *, rate: int = RATE_IN, channels: int = 1, width: int = 2,
         cut: int = 0) -> bytes:
    frames = int(seconds * rate)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(width)
        handle.setframerate(rate)
        handle.writeframes(b"\x10\x00" * (frames * channels * width // 2))
    data = buffer.getvalue()
    return data[:len(data) - cut] if cut else data


def test_importing_the_module_imports_no_model_library():
    code = (
        "import sys\n"
        "heavy = ('torch', 'transformers', 'huggingface_hub', 'scipy')\n"
        "before = {m for m in heavy if m in sys.modules}\n"
        "import audio.whisper_asr\n"
        "after = {m for m in heavy if m in sys.modules}\n"
        "print(sorted(after - before))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True, cwd=Path(__file__).resolve().parents[1])
    assert out.stdout.strip() == "[]"


@pytest.mark.parametrize(("data", "reason"), [
    (_wav(channels=2), "must be mono 16-bit PCM at 48000 Hz"),
    (_wav(rate=44100), "must be mono 16-bit PCM at 48000 Hz"),
    (_wav(width=1), "must be mono 16-bit PCM at 48000 Hz"),
    (_wav(0.0), "holds no audio"),
    (_wav(cut=10), "is cut short"),
    (_wav(30.5), "the limit is 30 s"),
    (b"not a wav", "not a WAV file"),
])
def test_take_samples_refuses_a_take_with_its_reason(data, reason):
    with pytest.raises(ValueError, match=reason):
        take_samples(data)


def test_take_samples_returns_float32_samples_of_the_take():
    samples = take_samples(_wav(6.0))
    assert samples.dtype == np.float32
    assert len(samples) == 6 * RATE_IN
    assert np.allclose(samples, 16 / 32768.0)


def test_a_cpu_device_is_refused_before_any_import():
    with pytest.raises(RuntimeError, match="mps or cuda only"):
        WhisperASR("repo", "cpu", "rev")


def test_the_weights_load_on_one_thread(monkeypatch):
    # transformers' 4 loader threads each cast float16 to float32 on MPS at once, and
    # torch's MPS kernel cache crashed the child (SIGSEGV) or spun it forever at 0/587.
    import os
    import types

    seen = {}

    class _Loaded:
        def eval(self):
            return self

    class _Whisper:
        @staticmethod
        def from_pretrained(path, **kw):
            seen["async_off"] = os.environ.get("HF_DEACTIVATE_ASYNC_LOAD")
            return _Loaded()

    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = lambda repo, revision, local_files_only: f"/hub/{revision}"
    errors = types.ModuleType("huggingface_hub.errors")
    errors.LocalEntryNotFoundError = type("LocalEntryNotFoundError", (Exception,), {})
    torch = types.ModuleType("torch")
    torch.float32, torch.inference_mode = "float32", None
    transformers = types.ModuleType("transformers")
    transformers.WhisperForConditionalGeneration = _Whisper
    transformers.WhisperProcessor = type(
        "P", (), {"from_pretrained": staticmethod(lambda path: object())})
    for name, module in (("huggingface_hub", hub), ("huggingface_hub.errors", errors),
                         ("torch", torch), ("transformers", transformers)):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delenv("HF_DEACTIVATE_ASYNC_LOAD", raising=False)

    WhisperASR("repo", "mps", "rev")

    assert seen["async_off"] == "1"


class _Features:
    def __init__(self):
        self.moved = None

    def to(self, device, dtype):
        self.moved = (device, dtype)
        return self


class _Processor:
    def __init__(self):
        self.calls = []
        self.features = _Features()
        self.decoded = None

    def __call__(self, x, *, sampling_rate, return_tensors):
        self.calls.append((x, sampling_rate, return_tensors))
        return type("Out", (), {"input_features": self.features})()

    def batch_decode(self, ids, *, skip_special_tokens):
        self.decoded = (ids, skip_special_tokens)
        return ["  Мои слова, как есть. "]


class _Model:
    def __init__(self):
        self.calls = []

    def generate(self, features, **kw):
        self.calls.append((features, kw))
        return ["ids"]


def _asr():
    model, processor, resampled = _Model(), _Processor(), []

    def resample(samples):
        resampled.append(samples)
        return samples[::3]

    asr = WhisperASR.loaded(model, processor, "rev123", resample=resample, device="mps")
    return asr, model, processor, resampled


def test_transcribe_runs_the_take_through_whisper_and_returns_the_text_as_decoded():
    asr, model, processor, resampled = _asr()

    result = asr.transcribe(_wav(6.0), "ru")

    assert len(resampled) == 1 and len(resampled[0]) == 6 * RATE_IN
    x, rate, tensors = processor.calls[0]
    assert len(x) == 6 * RATE_ASR
    assert rate == RATE_ASR == 16000
    assert tensors == "pt"
    assert processor.features.moved[0] == "mps"
    features, kw = model.calls[0]
    assert features is processor.features
    assert kw == {"language": "ru", "task": "transcribe"}
    assert processor.decoded == (["ids"], True)
    assert result["text"] == "  Мои слова, как есть. "
    assert result["revision"] == "rev123"
    assert result["device"] == "mps"
    assert result["seconds"] == 6.0
    assert result["elapsed_s"] >= 0


def test_a_language_whisper_is_not_asked_for_is_refused_before_generate():
    asr, model, _, _ = _asr()
    with pytest.raises(ValueError, match="'de'"):
        asr.transcribe(_wav(6.0), "de")
    assert model.calls == []
