"""Kokoro-82M in a `TorchWorker` child (design §5.1, §8.1, §11.1).

Models-service-only: this module imports torch transitively through `kokoro`, so it
lives in `audio/` and is never imported by `app`, `worker`, or `feed`
(CLAUDE.md § "One model process"). The parent resolves it BY NAME through
`TorchWorker`, so nothing torch-shaped ever crosses the boundary — only WAV bytes do.

Needs the `espeak-ng` system package: kokoro's grapheme-to-phoneme stage (`misaki`)
shells out to it. The Dockerfiles install it; on mac it is `brew install espeak-ng`.
"""

import numpy as np

from audio.segments import speak_passages
from storage.wav import encode_wav

SAMPLE_RATE = 24000
DEFAULT_VOICE = "af_heart"


class KokoroVoice:
    def __init__(self, repo: str, device: str) -> None:
        # `repo` is the HF id the `voice` SLOT holds (§4.1) — never hardcoded, so
        # switching the slot rebuilds this child against the new checkpoint.
        import espeakng_loader
        from kokoro import KPipeline
        from phonemizer.backend.espeak.wrapper import EspeakWrapper

        from audio.espeak_path import short_data_path

        # Importing kokoro made misaki point espeak at the bundled data. From a long
        # venv path espeak-ng drops it and exit(1)s the child (audio/espeak_path.py).
        EspeakWrapper.set_data_path(short_data_path(espeakng_loader.get_data_path()))

        self.repo = repo
        self.device = device
        # `lang_code='a'` is kokoro's American-English pack, which the default `af_*`
        # voices belong to. A different language is a different catalog entry, not a
        # runtime flag — the voice list and the G2P pack have to agree.
        self._pipeline = KPipeline(lang_code="a", repo_id=repo, device=device)

    def warm(self) -> None:
        """Resident must MEAN resident (§8.1). The pipeline builds its models lazily
        on the first call, so a one-word synthesis at load time is what makes the
        governor's accounting honest instead of optimistic."""
        self.speak("ready")

    def speak(self, text: str, voice: str | None = None,
              segments: list[str] | None = None,
              voices: list[str | None] | None = None) -> dict:
        """One WAV for the whole text.

        Passages that share a voice are spoken in ONE call (`audio/segments.py`):
        synthesising each separately restarts the prosody every time, which reads as a
        different narrator per sentence. kokoro yields per-sentence pieces within a
        call; they are concatenated HERE, in the child that owns torch, so exactly one
        clip crosses the pipe and the parent never sees a tensor.
        """
        default = voice or DEFAULT_VOICE
        chunks = [c for c in (segments or []) if c and c.strip()] or [text]

        def speak_one(passage: str, speaker: str):
            spoken = [np.asarray(audio, dtype=np.float32)
                      for _, _, audio in self._pipeline(passage, voice=speaker)]
            return np.concatenate(spoken) if spoken else np.zeros(1, dtype=np.float32)

        samples, timings = speak_passages(
            chunks, list(voices or []), default, speak_one, SAMPLE_RATE
        )
        return {
            "audio": encode_wav(samples, SAMPLE_RATE),
            "sample_rate": SAMPLE_RATE,
            "model": self.repo,
            "segments": timings,
        }
