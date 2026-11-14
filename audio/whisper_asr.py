"""The bake's speech recognition in a `TorchWorker` child (design §4.1, §5.1
`/voice/transcribe`, §8.1 `asr`).

Models-service-only, like every other module in `audio/`. It hears the words of ONE
saved-voice take before a bake: a mono PCM-16 WAV at 48 kHz of 5-30 s, so it fits
Whisper's one 30 s window in one pass. The take is resampled to Whisper's 16 kHz
exactly (48000 / 16000 = 3) and decoded with the take's language forced; nothing
detects a language. The text comes back as decoded: `app` alone judges whether it
holds usable words.

Module scope imports no torch, no transformers, no huggingface_hub and no scipy,
exactly like `audio/voxcpm_voice.py`: `take_samples` is unit-tested, and importing it
must not build a torch context in the importing process (CLAUDE.md § "One model
process").

Licence: MIT for the weights (`openai/whisper-large-v3-turbo`).
"""

import io
import os
import time
import wave
from contextlib import nullcontext
from pathlib import Path

import numpy as np

RATE_IN = 48000  # the take format (§5.1)
RATE_ASR = 16000  # Whisper's `preprocessor_config.json` sampling_rate
MAX_SECONDS = 30.0  # Whisper's window; a take is 5-30 s
LANGUAGES = ("en", "ru")


def take_samples(data: bytes) -> np.ndarray:
    """The take as float32 samples in [-1, 1]. Only a mono PCM-16 WAV at 48000 Hz with
    every declared frame and 0 < seconds <= 30 is accepted; else `ValueError` with the
    reason."""
    try:
        with wave.open(io.BytesIO(data), "rb") as handle:
            shape = (handle.getcomptype(), handle.getsampwidth(), handle.getnchannels(),
                     handle.getframerate())
            frames = handle.getnframes()
            payload = handle.readframes(frames)
    except (wave.Error, EOFError) as exc:
        raise ValueError("the take is not a WAV file") from exc
    if shape != ("NONE", 2, 1, RATE_IN):
        raise ValueError(f"the take must be mono 16-bit PCM at {RATE_IN} Hz")
    if frames <= 0:
        raise ValueError("the take holds no audio")
    if len(payload) != frames * 2:
        raise ValueError("the take is cut short: part of its audio is missing")
    seconds = frames / RATE_IN
    if seconds > MAX_SECONDS:
        raise ValueError(f"the take is {seconds:.1f} s long; the limit is {MAX_SECONDS:.0f} s")
    return np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0


def _resample(samples: np.ndarray) -> np.ndarray:
    from scipy.signal import resample_poly

    return resample_poly(samples, 1, RATE_IN // RATE_ASR).astype(np.float32)


class WhisperASR:
    def __init__(self, repo: str, device: str, revision: str) -> None:
        # GPU only (`docs/design.md` §5): "cpu" would silently run an 809M model on the CPU.
        if device != "mps" and not str(device).startswith("cuda"):
            raise RuntimeError(f"Whisper runs on mps or cuda only, not {device!r}")

        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError

        # The PINNED snapshot on disk only: a child never downloads inside a bake.
        try:
            path = snapshot_download(repo, revision=revision, local_files_only=True)
        except LocalEntryNotFoundError as exc:
            raise RuntimeError(
                f"Whisper weights for {revision} are not on disk: run `make models`"
            ) from exc
        # The snapshot folder is the commit; anything else is another revision.
        if Path(path).name != revision:
            raise RuntimeError(
                f"Whisper snapshot {Path(path).name!r} is not the pinned revision {revision!r}"
            )

        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        processor = WhisperProcessor.from_pretrained(path)
        # ONE loader thread. transformers' 4 threads each cast float16 to float32 on MPS
        # at once, and torch's MPS kernel cache crashed this child (SIGSEGV) or spun it
        # forever at 0 of 587 weights (§5.1 `asr`). The child holds Whisper alone.
        os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"
        # float32: no float16 run on MPS was measured here (§4.1).
        model = WhisperForConditionalGeneration.from_pretrained(
            path, dtype=torch.float32, device_map=device).eval()
        self.repo = repo
        self._setup(model, processor, revision, _resample, device, torch.float32,
                    torch.inference_mode)

    @classmethod
    def loaded(cls, model, processor, revision: str, *, resample,
               device: str = "mps") -> "WhisperASR":
        """A child around an already built model and processor — the test seam.

        `resample` replaces `scipy.signal.resample_poly`, and no torch dtype or
        inference mode is used, so a test drives `transcribe` with fakes and never
        imports torch.
        """
        asr = cls.__new__(cls)
        asr.repo = None
        asr._setup(model, processor, revision, resample, device, None, nullcontext)
        return asr

    def _setup(self, model, processor, revision: str, resample, device: str, dtype,
               inference) -> None:
        self._model = model
        self._processor = processor
        self.revision = revision
        self._resample = resample
        self.device = device
        self._dtype = dtype
        self._inference = inference

    def transcribe(self, audio: bytes, language: str) -> dict:
        """The words of one take: `{text, revision, device, seconds, elapsed_s}`."""
        if language not in LANGUAGES:
            raise ValueError(f"Whisper hears {', '.join(LANGUAGES)} here, not {language!r}")
        started = time.monotonic()
        samples = take_samples(audio)
        x = self._resample(samples)
        features = self._processor(x, sampling_rate=RATE_ASR, return_tensors="pt").input_features
        features = features.to(self.device, self._dtype)
        with self._inference():
            ids = self._model.generate(features, language=language, task="transcribe")
        text = self._processor.batch_decode(ids, skip_special_tokens=True)[0]
        return {"text": text, "revision": self.revision, "device": self.device,
                "seconds": len(samples) / RATE_IN,
                "elapsed_s": round(time.monotonic() - started, 3)}
