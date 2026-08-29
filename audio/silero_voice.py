"""Silero v4 Russian TTS in a `TorchWorker` child (design §5.1, §8.1, §11.1).

Models-service-only, like every other module in `audio/`.

Chosen for Russian because Kokoro does not speak it and Silero needs **no system
dependency at all** — no `espeak-ng` for phonemes, no `uroman` for Cyrillic. It is a
single `torch.package` archive driven by plain torch, which is already the one heavy
library this process is allowed to load.

It is **not distributed on Hugging Face**, so its catalog entry is a `UrlSource` and
the downloader streams the archive into `data_dir/models` the same way it does a GGUF.
This class is handed that local path, never a repo id.

Licence: CC-BY-NC (Silero's MIT applies only to their base cis-tts models).
"""

import re

import numpy as np

from audio.segments import speak_passages
from storage.wav import encode_wav

SAMPLE_RATE = 48000
DEFAULT_VOICE = "xenia"

# Silero v4 warns past 1 000 symbols and then RAISES — "Model couldn't generate your
# text, probably it's too long". The real ceiling is the generated audio, not the
# character count, so the same 931-symbol script succeeds as `xenia` and fails as
# `aidar`. A film's script is 200-400 words, i.e. always over it, which is why every
# Russian film came out mute. 500 leaves room for the slowest speaker.
MAX_SYMBOLS = 500
# 0.25 s between chunks: a sentence boundary read with no gap at all sounds rushed.
_GAP_SECONDS = 0.25
_SENTENCE = re.compile(r"[^.!?…]*[.!?…]+\s*|[^.!?…]+$")


def split_text(text: str, limit: int = MAX_SYMBOLS) -> list[str]:
    """Cut `text` into pieces of at most `limit` symbols, at SENTENCE boundaries.

    Pure, so the cutting is unit-tested without loading a model. A single sentence
    longer than the limit is split on whitespace rather than dropped — mid-word is
    audible, but a missing paragraph is worse.
    """
    pieces: list[str] = []
    buffer = ""
    for match in _SENTENCE.finditer(text.strip()):
        sentence = match.group(0).strip()
        if not sentence:
            continue
        while len(sentence) > limit:
            cut = sentence.rfind(" ", 0, limit)
            cut = cut if cut > 0 else limit
            if buffer:
                pieces.append(buffer)
                buffer = ""
            pieces.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if buffer and len(buffer) + 1 + len(sentence) <= limit:
            buffer = f"{buffer} {sentence}"
        else:
            if buffer:
                pieces.append(buffer)
            buffer = sentence
    if buffer:
        pieces.append(buffer)
    return pieces


class SileroVoice:
    def __init__(self, path: str, device: str) -> None:
        # Imported HERE, not at module scope, exactly as `KokoroVoice` imports kokoro:
        # `split_text` is pure and is unit-tested, and importing this module for it must
        # not build a torch (and CUDA) context in whatever process did the importing
        # (CLAUDE.md § "One model process").
        import torch

        # `path` is the archive `Downloads` placed in `data_dir/models` — the models
        # service resolves it from the slot's `UrlSource.file` (§4.1).
        self.path = path
        self.device = device
        self._model = torch.package.PackageImporter(path).load_pickle("tts_models", "model")
        self._model.to(torch.device(device))

    def warm(self) -> None:
        """Resident must MEAN resident (§8.1) — the archive builds its graph lazily, so
        speak once at load time to make the governor's accounting honest."""
        self.speak("готово")

    def speak(self, text: str, voice: str | None = None,
              segments: list[str] | None = None,
              voices: list[str | None] | None = None) -> dict:
        """One WAV for the whole script, however long it is.

        The passages are grouped into RUNS that share a voice and each run is spoken in
        ONE call (`audio/segments.py`). Synthesising every passage separately restarts
        the prosody each time, so an eight-passage film came out sounding like eight
        different readers even where the speaker never changed.
        """
        default = voice or DEFAULT_VOICE
        chunks = ([c for c in (segments or []) if c and c.strip()]
                  or split_text(text) or [text])
        # Silero refuses anything much past a thousand symbols, so a run is split again
        # here if joining its passages pushed it over the limit.
        def speak_one(passage: str, speaker: str):
            parts = split_text(passage) or [passage]
            spoken = [self._model.apply_tts(text=part, speaker=speaker,
                                            sample_rate=SAMPLE_RATE).cpu().numpy()
                      for part in parts]
            return np.concatenate(spoken) if spoken else np.zeros(1, dtype=np.float32)

        samples, timings = speak_passages(
            chunks, list(voices or []), default, speak_one, SAMPLE_RATE, _GAP_SECONDS
        )
        return {
            "audio": encode_wav(samples, SAMPLE_RATE),
            "sample_rate": SAMPLE_RATE,
            "model": "silero-v4-ru",
            "segments": timings,
        }
