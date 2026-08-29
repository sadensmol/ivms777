"""MusicGen in a `TorchWorker` child (design §5.1, §8.1, §11.1).

Models-service-only, like every other module in `audio/`.

**Why not Stable Audio Open.** It was the first choice — better model, 44.1 kHz
stereo, permissive licence — but its sampler is `CosineDPMSolverMultistepScheduler`,
which is SDE-only and drives `torchsde`. On this stack (torch 2.13 / Python 3.13 /
torchsde 0.2.6) `BrownianInterval._split` recurses without terminating near the
smallest sigma: raising the recursion limit to 20 000 did not help, CPU failed the
same way at step 7 of 8, and the scheduler exposes no ODE alternative. MusicGen has no
scheduler at all — it is an autoregressive transformer under plain `transformers`, so
there is nothing to go wrong in a sampler.

Licence: the weights are **CC-BY-NC 4.0** — non-commercial. Fine for a personal
library, disqualifying for a commercial build. The code is MIT.
"""

import numpy as np

from storage.wav import encode_wav

# Generation is autoregressive at the audio encoder's frame rate, so length is a token
# count, not a fixed ceiling. These caps are about COST and quality, not a model limit:
# MusicGen wanders past roughly half a minute, and every second costs GPU time in a job
# the user is waiting on — MEASURED at ~4.9x realtime for `musicgen-small` on an M-series
# mac (15 s of audio in 73 s).
MAX_SECONDS = 40.0
# The whole score. A film longer than this ends its music early rather than doubling a
# wait nobody asked for (`feed/score.py` caps the plan to the same figure).
SCORE_MAX_SECONDS = 180.0

# How much of the previous section is fed back in as the audio prompt for the next one.
# This is what makes a score one piece instead of several clips: MusicGen continues what
# it hears, so section two arrives in the same key, tempo and instrumentation as section
# one even when its brief asks for a different feel.
CONTINUATION_SECONDS = 2.0
# The continuation RETURNS the prompt audio followed by the new material, so the prompt
# region is dropped — all but this much of it, which is crossfaded against the audio we
# already have. The overlap is musically aligned (it is the model's own reconstruction
# of what it was given), so the join is inaudible.
CROSSFADE_SECONDS = 0.35
# The film's own top and tail. A score that starts at full level on the first frame
# sounds like a clip being pasted in, and one that stops dead sounds broken.
FADE_IN_SECONDS = 0.4
FADE_OUT_SECONDS = 1.2

# MusicGen's own defaults, named here because they are the difference between a score
# and the same safe average every time. Guidance is how hard the brief is enforced; a
# livelier memory is worth pushing harder on both.
DEFAULT_GUIDANCE = 3.0


class MusicGen:
    def __init__(self, repo: str, device: str) -> None:
        import torch
        from transformers import AutoProcessor, MusicgenForConditionalGeneration

        self.repo = repo
        self.device = device
        self._processor = AutoProcessor.from_pretrained(repo)
        # fp32 everywhere: musicgen-small is ~2 GB, so half precision buys little, and
        # MPS is fussy about half in the audio decoder.
        self._model = MusicgenForConditionalGeneration.from_pretrained(repo).to(device).eval()
        self._sample_rate = int(self._model.config.audio_encoder.sampling_rate)
        self._frame_rate = int(self._model.config.audio_encoder.frame_rate)
        self._torch = torch

    def warm(self) -> None:
        """Nothing lazy is left: `from_pretrained` performs the whole load. Present so
        the unit registers with `warm="warm"` uniformly with the other torch units
        (`modelsvc/slots.py`)."""

    def music(
        self, prompt: str = "", seconds: float = 30.0, seed: int | None = None,
        sections: list[dict] | None = None, temperature: float | None = None,
        guidance: float | None = None,
    ) -> dict:
        """One score, as one continuous piece (design §11.1).

        `sections` is `[{"brief": str, "seconds": float}]` — each generated as a
        CONTINUATION of the one before it, so the music changes where the story changes
        instead of being one clip looped under the whole film. A bare `prompt` is the
        same thing with one section, which is what every caller did before.

        Returns the track plus `segments`: where each section actually landed, since a
        token count never divides exactly into the seconds that were asked for.
        """
        torch = self._torch
        plan = sections or [{"brief": prompt, "seconds": seconds}]
        if seed is not None:
            # ONE seed for the whole score, so the same memory scores the same way
            # twice — and a different memory never gets this memory's music.
            torch.manual_seed(seed)
        pieces: list[np.ndarray] = []
        budget = SCORE_MAX_SECONDS
        for index, section in enumerate(plan):
            length = max(1.0, min(float(section.get("seconds", seconds)), MAX_SECONDS,
                                  budget))
            if length < 1.0:
                break
            budget -= length
            brief = str(section.get("brief") or prompt)
            tail = self._tail(pieces[-1]) if pieces else None
            audio = self._generate(brief, length, tail, temperature, guidance)
            if tail is not None:
                # Drop the prompt the model was given back, keeping one crossfade's
                # worth to overlap against what we already have.
                keep = max(0, tail.shape[-1] - self._samples(CROSSFADE_SECONDS))
                audio = audio[:, keep:]
            pieces.append(audio)
            if index and audio.shape[-1] <= self._samples(CROSSFADE_SECONDS):
                break
        track, segments = self._splice(pieces)
        return {
            "audio": encode_wav(track, self._sample_rate),
            "sample_rate": self._sample_rate,
            "model": self.repo,
            "segments": segments,
        }

    def _generate(
        self, brief: str, length: float, tail: np.ndarray | None,
        temperature: float | None, guidance: float | None,
    ) -> np.ndarray:
        """One section. With `tail`, MusicGen continues that audio rather than starting
        from nothing — the returned audio therefore BEGINS with the prompt."""
        torch = self._torch
        kwargs: dict = {"text": [brief], "padding": True, "return_tensors": "pt"}
        if tail is not None:
            kwargs["audio"] = tail
            kwargs["sampling_rate"] = self._sample_rate
        inputs = self._processor(**kwargs).to(self.device)
        decode: dict = {}
        if temperature is not None:
            decode["temperature"] = float(temperature)
        with torch.no_grad():
            values = self._model.generate(
                **inputs, do_sample=True,
                guidance_scale=float(DEFAULT_GUIDANCE if guidance is None else guidance),
                max_new_tokens=int(length * self._frame_rate), **decode,
            )
        # (batch, channels, samples); batch is 1 and musicgen-small is mono, so this is
        # the (channels, samples) layout `encode_wav` expects.
        return values[0].to(torch.float32).cpu().numpy()

    def _samples(self, seconds: float) -> int:
        return int(seconds * self._sample_rate)

    def _tail(self, audio: np.ndarray) -> np.ndarray:
        """The end of a section, as the audio prompt for the next one. Mono: the
        processor takes one waveform, not a channel pair."""
        return audio[0][-min(audio.shape[-1], self._samples(CONTINUATION_SECONDS)):]

    def _splice(self, pieces: list[np.ndarray]) -> tuple[np.ndarray, list[dict]]:
        """Join the sections, crossfading each into the last, and report where each one
        landed. Peak-normalised ONCE over the whole track: the sampler's absolute level
        varies run to run, and normalising per section would make the joins audible as
        steps in volume."""
        fade = self._samples(CROSSFADE_SECONDS)
        track = pieces[0]
        segments = [{"start": 0.0, "end": track.shape[-1] / self._sample_rate}]
        for piece in pieces[1:]:
            overlap = min(fade, track.shape[-1], piece.shape[-1])
            start = (track.shape[-1] - overlap) / self._sample_rate
            if overlap:
                ramp = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
                track[:, -overlap:] = track[:, -overlap:] * (1.0 - ramp) + piece[:, :overlap] * ramp
                piece = piece[:, overlap:]
            track = np.concatenate([track, piece], axis=-1)
            segments.append({"start": round(start, 3),
                             "end": round(track.shape[-1] / self._sample_rate, 3)})
        track = self._shape(track)
        peak = float(np.max(np.abs(track))) or 1.0
        return track / peak, segments

    def _shape(self, track: np.ndarray) -> np.ndarray:
        """Top and tail the score so it arrives and leaves instead of being cut in."""
        length = track.shape[-1]
        head = min(self._samples(FADE_IN_SECONDS), length // 2)
        tail = min(self._samples(FADE_OUT_SECONDS), length // 2)
        if head:
            track[:, :head] *= np.linspace(0.0, 1.0, head, dtype=np.float32)
        if tail:
            track[:, -tail:] *= np.linspace(1.0, 0.0, tail, dtype=np.float32)
        return track
