"""The speaker-similarity child (`audio/speaker_embed.py`, design §5.1 `/voice/embed`).

Fakes only: no torch, no speechbrain, no scipy, no model load.
"""

import io
import math
import subprocess
import sys
import types
import wave
from pathlib import Path

import numpy as np
import pytest

from audio.speaker_embed import (
    MIN_VOICED_SECONDS,
    PIPELINE,
    RATE_MODEL,
    SpeakerEmbedder,
    _resample,
    clip_samples,
    voiced_seconds,
)

LOUD = 3277  # 0.1 of full scale
QUIET = 33  # about 0.001, under VOICED_RMS
REVISION = "rev123"


def _wav(seconds: float = 6.0, *, rate: int = 48000, channels: int = 1, width: int = 2,
         level: int = LOUD, cut: int = 0) -> bytes:
    frames = int(seconds * rate)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(width)
        handle.setframerate(rate)
        handle.writeframes(level.to_bytes(2, "little") * (frames * channels * width // 2))
    data = buffer.getvalue()
    return data[:len(data) - cut] if cut else data


class _Embedding:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32).reshape(1, 1, -1)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class _Classifier:
    def __init__(self, values=None):
        self.values = np.arange(1, 193) if values is None else values
        self.batches = []

    def encode_batch(self, batch):
        self.batches.append(batch)
        return _Embedding(self.values)


def _embedder(classifier=None):
    classifier = classifier or _Classifier()
    resampled = []

    def resample(samples, rate):
        resampled.append(rate)
        return samples[::rate // RATE_MODEL]

    embedder = SpeakerEmbedder.loaded(classifier, REVISION, resample=resample,
                                      tensor=lambda x: ("tensor", x))
    return embedder, classifier, resampled


def test_importing_the_module_imports_no_model_library():
    code = (
        "import sys\n"
        "heavy = ('torch', 'speechbrain', 'huggingface_hub', 'scipy')\n"
        "before = {m for m in heavy if m in sys.modules}\n"
        "import audio.speaker_embed\n"
        "after = {m for m in heavy if m in sys.modules}\n"
        "print(sorted(after - before))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True, cwd=Path(__file__).resolve().parents[1])
    assert out.stdout.strip() == "[]"


@pytest.mark.parametrize(("data", "reason"), [
    (_wav(channels=2), "must be mono 16-bit PCM"),
    (_wav(width=1), "must be mono 16-bit PCM"),
    (_wav(rate=8000), "the minimum is 16000 Hz"),
    (_wav(0.0), "holds no audio"),
    (_wav(cut=10), "is cut short"),
    (b"not a wav", "not a WAV file"),
])
def test_clip_samples_refuses_a_clip_with_its_reason(data, reason):
    with pytest.raises(ValueError, match=reason):
        clip_samples(data)


@pytest.mark.parametrize("rate", [16000, 24000, 44100, 48000])
def test_clip_samples_takes_any_rate_from_the_models_and_has_no_30_s_cap(rate):
    samples, found = clip_samples(_wav(43.0, rate=rate))
    assert found == rate
    assert samples.dtype == np.float32
    assert len(samples) == 43 * rate
    assert np.allclose(samples, LOUD / 32768.0)


def test_voiced_seconds_counts_only_the_frames_at_or_over_the_rms():
    silence, speech = np.zeros(RATE_MODEL, np.float32), np.full(RATE_MODEL // 2, 0.1, np.float32)
    assert voiced_seconds(np.concatenate([silence, speech, silence])) == pytest.approx(0.5, abs=0.03)


def test_voiced_seconds_of_a_clip_under_one_frame_is_zero():
    assert voiced_seconds(np.full(100, 0.5, np.float32)) == 0


@pytest.mark.parametrize(("rate", "up", "down"), [(48000, 1, 3), (44100, 160, 441), (24000, 2, 3)])
def test_resampling_reaches_16_khz_from_any_rate(monkeypatch, rate, up, down):
    seen = []
    signal = types.ModuleType("scipy.signal")
    signal.resample_poly = lambda samples, a, b: (seen.append((a, b)), samples)[1]
    monkeypatch.setitem(sys.modules, "scipy", types.ModuleType("scipy"))
    monkeypatch.setitem(sys.modules, "scipy.signal", signal)

    _resample(np.zeros(10, np.float32), rate)

    assert seen == [(up, down)]


def test_embed_returns_the_vector_with_its_revision_seconds_and_pipeline():
    embedder, classifier, resampled = _embedder()

    result = embedder.embed(_wav(6.0))

    assert resampled == [48000]
    kind, x = classifier.batches[0]
    assert kind == "tensor" and len(x) == 6 * RATE_MODEL
    assert result["vector"] == [float(n) for n in range(1, 193)]
    assert result["revision"] == REVISION
    assert result["seconds"] == 6.0
    assert result["pipeline"] == PIPELINE
    assert result["elapsed_s"] >= 0


def test_a_clip_at_the_models_rate_is_not_resampled():
    embedder, classifier, resampled = _embedder()
    embedder.embed(_wav(6.0, rate=RATE_MODEL))
    assert resampled == []
    assert len(classifier.batches[0][1]) == 6 * RATE_MODEL


def test_a_long_clip_is_embedded_whole():
    embedder, _, _ = _embedder()
    assert embedder.embed(_wav(43.0))["seconds"] == 43.0


@pytest.mark.parametrize("data", [
    _wav(6.0, level=QUIET),
    _wav(MIN_VOICED_SECONDS - 0.2),
])
def test_too_little_voiced_audio_is_refused_with_its_reason_before_the_model(data):
    embedder, classifier, _ = _embedder()

    result = embedder.embed(data)

    assert "voiced audio" in result["refused"]
    assert "the minimum is 1.0 s" in result["refused"]
    assert result["revision"] == REVISION
    assert "vector" not in result
    assert classifier.batches == []


@pytest.mark.parametrize("values", [[math.nan] * 192, [math.inf] * 192, [0.0] * 192])
def test_a_vector_that_is_not_finite_or_is_zero_is_an_error(values):
    embedder, _, _ = _embedder(_Classifier(values))
    with pytest.raises(RuntimeError, match="not finite or has a zero norm"):
        embedder.embed(_wav(6.0))


def test_a_cpu_device_is_refused_before_any_import():
    with pytest.raises(RuntimeError, match="mps or cuda only"):
        SpeakerEmbedder("repo", "cpu", "rev")


def _fake_load(monkeypatch, snapshot_download):
    """The libraries of `__init__`, faked. Returns what `from_hparams` was given."""
    seen = {}

    class _EncoderClassifier:
        @staticmethod
        def from_hparams(**kw):
            seen.update(kw)
            return _Classifier()

    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = snapshot_download
    errors = types.ModuleType("huggingface_hub.errors")
    errors.LocalEntryNotFoundError = _Missing
    torch = types.ModuleType("torch")
    torch.from_numpy, torch.inference_mode = None, None
    classifiers = types.ModuleType("speechbrain.inference.classifiers")
    classifiers.EncoderClassifier = _EncoderClassifier
    fetching = types.ModuleType("speechbrain.utils.fetching")
    fetching.LocalStrategy = types.SimpleNamespace(NO_LINK="no-link")
    for name, module in (("huggingface_hub", hub), ("huggingface_hub.errors", errors),
                         ("torch", torch), ("speechbrain", types.ModuleType("speechbrain")),
                         ("speechbrain.inference", types.ModuleType("speechbrain.inference")),
                         ("speechbrain.inference.classifiers", classifiers),
                         ("speechbrain.utils", types.ModuleType("speechbrain.utils")),
                         ("speechbrain.utils.fetching", fetching)):
        monkeypatch.setitem(sys.modules, name, module)
    return seen


class _Missing(Exception):
    pass


def test_the_model_loads_from_the_pinned_snapshot_on_disk_only(monkeypatch):
    asked = []

    def download(repo, revision, local_files_only):
        asked.append((repo, revision, local_files_only))
        return f"/hub/snapshots/{revision}"

    seen = _fake_load(monkeypatch, download)

    SpeakerEmbedder("repo", "mps", "rev")

    assert asked == [("repo", "rev", True)]
    assert seen == {"source": "/hub/snapshots/rev",
                    "overrides": {"pretrained_path": "/hub/snapshots/rev"},
                    "local_strategy": "no-link", "run_opts": {"device": "mps"}}


def test_weights_that_are_not_on_disk_name_make_models(monkeypatch):
    def download(repo, revision, local_files_only):
        raise _Missing

    seen = _fake_load(monkeypatch, download)
    with pytest.raises(RuntimeError, match=r"ECAPA weights for rev are not on disk: run `make models`"):
        SpeakerEmbedder("repo", "mps", "rev")
    assert seen == {}


def test_a_snapshot_of_another_revision_is_refused(monkeypatch):
    seen = _fake_load(monkeypatch, lambda repo, revision, local_files_only: "/hub/snapshots/other")
    with pytest.raises(RuntimeError, match="is not the pinned revision"):
        SpeakerEmbedder("repo", "mps", "rev")
    assert seen == {}
