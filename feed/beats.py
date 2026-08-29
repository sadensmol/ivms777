"""The music's pulse (design §11.1) — a beat grid for one generated soundtrack.

PURE and offline: WAV bytes in, a list of beat times out. numpy and nothing else, so
this is importable from `feed/` on the torch-free side of the one-model-process
boundary (CLAUDE.md). No model is involved — beat tracking is signal processing, and a
30-second clip is milliseconds of arithmetic.

Why it exists: a film whose cuts land wherever a fixed 4.5 s timer expires is three
things playing at once — picture, score, voice. Cutting ON the beat is the single
change that fuses them into one thing, and it is the reason the timeline carries
absolute times rather than durations (§11.1).

The method is the standard one: spectral flux -> onset envelope -> tempo by
autocorrelation -> a phase-aligned pulse train. It is deliberately NOT a full
dynamic-programming beat tracker: the music here is our own generated ambient bed,
slow and steady by construction, so a constant tempo with a fitted phase is both
sufficient and far easier to test.
"""

import numpy as np

from storage.wav import decode_wav

_HOP = 512
_WINDOW = 2048
# Ambient scores sit far below dance tempo; anything outside this is a mis-fit, not a
# groove. 40-140 BPM covers "slow piano bed" through "walking pace".
_MIN_BPM = 40.0
_MAX_BPM = 140.0
# A film needs a beat to cut on every few seconds, not every half second. Beats are
# grouped into BARS of this many, and the director cuts on bars.
_BEATS_PER_BAR = 4
_DEFAULT_BPM = 72.0


def onset_envelope(samples: np.ndarray, rate: int) -> tuple[np.ndarray, float]:
    """Positive spectral flux per hop, and the seconds one hop covers.

    Rectified difference of consecutive magnitude spectra: energy APPEARING is an
    onset, energy decaying is not, which is why the negative half is discarded.
    """
    if samples.size < _WINDOW:
        return np.zeros(0, dtype=np.float32), _HOP / rate
    window = np.hanning(_WINDOW).astype(np.float32)
    count = 1 + (samples.size - _WINDOW) // _HOP
    frames = np.lib.stride_tricks.as_strided(
        samples,
        shape=(count, _WINDOW),
        strides=(samples.strides[0] * _HOP, samples.strides[0]),
    )
    spectra = np.abs(np.fft.rfft(frames * window, axis=1))
    flux = np.diff(spectra, axis=0)
    envelope = np.maximum(flux, 0.0).sum(axis=1)
    # Subtracting a local mean removes the slow swell of a pad, leaving the transients.
    if envelope.size > 16:
        kernel = np.ones(16, dtype=np.float32) / 16.0
        envelope = envelope - np.convolve(envelope, kernel, mode="same")
    envelope = np.maximum(envelope, 0.0)
    peak = envelope.max()
    return (envelope / peak if peak else envelope), _HOP / rate


def estimate_bpm(envelope: np.ndarray, hop_seconds: float) -> float:
    """Tempo by autocorrelating the onset envelope, restricted to a musical range."""
    if envelope.size < 8 or hop_seconds <= 0:
        return _DEFAULT_BPM
    centred = envelope - envelope.mean()
    correlation = np.correlate(centred, centred, mode="full")[centred.size - 1:]
    low = max(1, round(60.0 / _MAX_BPM / hop_seconds))
    high = min(correlation.size - 1, round(60.0 / _MIN_BPM / hop_seconds))
    if high <= low:
        return _DEFAULT_BPM
    lag = low + int(np.argmax(correlation[low : high + 1]))
    return 60.0 / (lag * hop_seconds)


def beat_times(data: bytes) -> list[float]:
    """Every beat in the clip, in seconds from its start.

    An empty list when there is no music: the caller then paces the film on its own
    clock rather than pretending to a groove that is not there.
    """
    samples, rate = decode_wav(data)
    if samples.size == 0:
        return []
    envelope, hop = onset_envelope(samples, rate)
    if envelope.size == 0:
        return []
    bpm = estimate_bpm(envelope, hop)
    period = 60.0 / bpm
    duration = samples.size / rate
    # Fit the PHASE by scoring every candidate offset against the onset envelope: the
    # grid that lines up with the actual transients wins. Without this the beats are at
    # the right rate but in the wrong place, which is worse than no grid at all.
    step = period / 16.0
    best_offset, best_score = 0.0, -1.0
    offset = 0.0
    while offset < period:
        times = np.arange(offset, duration, period)
        indices = np.clip((times / hop).astype(int), 0, envelope.size - 1)
        score = float(envelope[indices].sum())
        if score > best_score:
            best_offset, best_score = offset, score
        offset += step
    return [round(t, 4) for t in np.arange(best_offset, duration, period)]


def bar_times(data: bytes, beats_per_bar: int = _BEATS_PER_BAR) -> list[float]:
    """Every downbeat — the moments a film may cut on without feeling early or late."""
    beats = beat_times(data)
    return beats[::beats_per_bar]


def snap(time: float, grid: list[float], *, window: float) -> float:
    """`time` moved to the nearest grid point within `window` seconds, else unchanged.

    The window matters: a cut dragged two seconds to reach a beat is no longer the cut
    the director wrote. Sync serves the edit, it does not overrule it.
    """
    if not grid:
        return time
    nearest = min(grid, key=lambda g: abs(g - time))
    return nearest if abs(nearest - time) <= window else time
