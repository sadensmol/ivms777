"""The speaker-similarity meter's model in a `TorchWorker` child (design §4.1, §5.1
`/voice/embed`, §8.1 `speaker_embed`, §11.1).

Models-service-only, like every other module in `audio/`. It turns ONE clip, a take or
a generated test, into the raw speaker embedding of ECAPA-TDNN. It only measures:
nothing here touches synthesis, and `app` does the arithmetic.

Module scope imports no torch, no speechbrain, no huggingface_hub and no scipy, exactly
like `audio/whisper_asr.py`: `clip_samples` runs in the fake backend, and importing it
must not build a torch context in the importing process (CLAUDE.md § "One model
process").

Licence: Apache-2.0 for the weights (`speechbrain/spkrec-ecapa-voxceleb`) and the code.
"""

import io
import time
import wave
from contextlib import nullcontext
from math import gcd
from pathlib import Path

import numpy as np

RATE_MODEL = 16000  # the model card's input rate
# The preprocessing version. Goes up when a change here moves the vector of the same
# clip, so a stored take vector is never compared with one made another way (§6).
PIPELINE = 1
# A stated choice, not calibrated (§5.1). An energy check, not voice detection.
MIN_VOICED_SECONDS = 1.0
VOICED_RMS = 0.01
FRAME_SAMPLES = 480  # 30 ms at RATE_MODEL


def clip_samples(data: bytes) -> tuple[np.ndarray, int]:
    """The clip as float32 samples in [-1, 1], and its rate. Only a mono PCM-16 WAV of at
    least 16000 Hz with every declared frame is accepted; else `ValueError` with the
    reason. No length cap: the HTTP layer bounds the bytes."""
    try:
        with wave.open(io.BytesIO(data), "rb") as handle:
            shape = (handle.getcomptype(), handle.getsampwidth(), handle.getnchannels())
            rate = handle.getframerate()
            frames = handle.getnframes()
            payload = handle.readframes(frames)
    except (wave.Error, EOFError) as exc:
        raise ValueError("the clip is not a WAV file") from exc
    if shape != ("NONE", 2, 1):
        raise ValueError("the clip must be mono 16-bit PCM")
    if rate < RATE_MODEL:
        raise ValueError(f"the clip is {rate} Hz; the minimum is {RATE_MODEL} Hz")
    if frames <= 0:
        raise ValueError("the clip holds no audio")
    if len(payload) != frames * 2:
        raise ValueError("the clip is cut short: part of its audio is missing")
    return np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0, rate


def voiced_seconds(samples: np.ndarray) -> float:
    """The seconds of 30 ms frames at `RATE_MODEL` with an RMS of at least `VOICED_RMS`."""
    count = len(samples) // FRAME_SAMPLES
    frames = samples[:count * FRAME_SAMPLES].reshape(count, FRAME_SAMPLES).astype(np.float64)
    voiced = np.sqrt((frames ** 2).mean(axis=1)) >= VOICED_RMS
    return int(voiced.sum()) * FRAME_SAMPLES / RATE_MODEL


def _resample(samples: np.ndarray, rate: int) -> np.ndarray:
    from scipy.signal import resample_poly

    common = gcd(rate, RATE_MODEL)
    return resample_poly(samples, RATE_MODEL // common, rate // common).astype(np.float32)


class SpeakerEmbedder:
    def __init__(self, repo: str, device: str, revision: str) -> None:
        # GPU only (`docs/design.md` §5): there is no CPU fallback.
        if device != "mps" and not str(device).startswith("cuda"):
            raise RuntimeError(f"ECAPA runs on mps or cuda only, not {device!r}")

        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError

        # The PINNED snapshot on disk only: a child never downloads inside a test.
        try:
            path = snapshot_download(repo, revision=revision, local_files_only=True)
        except LocalEntryNotFoundError as exc:
            raise RuntimeError(
                f"ECAPA weights for {revision} are not on disk: run `make models`"
            ) from exc
        # The snapshot folder is the commit; anything else is another revision.
        if Path(path).name != revision:
            raise RuntimeError(
                f"ECAPA snapshot {Path(path).name!r} is not the pinned revision {revision!r}"
            )

        import torch
        from speechbrain.inference.classifiers import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy

        # The pinned `hyperparams.yaml` names the Hub repo as `pretrained_path`, so the
        # override is what keeps every checkpoint on the snapshot. No `savedir`: it would
        # leave a folder of links in the working directory.
        classifier = EncoderClassifier.from_hparams(
            source=path, overrides={"pretrained_path": path},
            local_strategy=LocalStrategy.NO_LINK, run_opts={"device": device})
        self._setup(classifier, revision, _resample, torch.from_numpy, torch.inference_mode)

    @classmethod
    def loaded(cls, classifier, revision: str, *, resample, tensor) -> "SpeakerEmbedder":
        """A child around an already built classifier — the test seam.

        `resample` replaces `scipy.signal.resample_poly` and `tensor` replaces
        `torch.from_numpy`, and no inference mode is used, so a test drives `embed` with
        fakes and never imports torch.
        """
        embedder = cls.__new__(cls)
        embedder._setup(classifier, revision, resample, tensor, nullcontext)
        return embedder

    def _setup(self, classifier, revision: str, resample, tensor, inference) -> None:
        self._classifier = classifier
        self.revision = revision
        self._resample = resample
        self._tensor = tensor
        self._inference = inference

    def embed(self, audio: bytes) -> dict:
        """The speaker embedding of one clip: `{vector, revision, seconds, elapsed_s,
        pipeline}`, or `{refused, revision}` with the reason when the clip holds too
        little voiced audio. A refusal is an answer, not an error: a child's exception
        reaches the parent only as text."""
        started = time.monotonic()
        samples, rate = clip_samples(audio)
        x = samples if rate == RATE_MODEL else self._resample(samples, rate)
        voiced = voiced_seconds(x)
        if voiced < MIN_VOICED_SECONDS:
            return {"refused": f"the clip holds {voiced:.2f} s of voiced audio; "
                               f"the minimum is {MIN_VOICED_SECONDS:.1f} s",
                    "revision": self.revision}
        with self._inference():
            embedding = self._classifier.encode_batch(self._tensor(x))
        vector = embedding.detach().cpu().numpy().reshape(-1)
        if not np.isfinite(vector).all() or not vector.any():
            raise RuntimeError("the speaker embedding is not finite or has a zero norm")
        return {"vector": vector.tolist(), "revision": self.revision,
                "seconds": len(samples) / rate, "pipeline": PIPELINE,
                "elapsed_s": round(time.monotonic() - started, 3)}
