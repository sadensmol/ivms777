"""WAV bytes in and out — numpy plus the stdlib `wave` module, nothing else.

Deliberately dependency-free. `audio/` (which owns torch) encodes with it and `feed/`
(which must never see torch) measures with it, so it has to be importable from BOTH
sides of the one-model-process boundary (CLAUDE.md, design §5.1). That also rules out
`soundfile` and `torchaudio`, either of which would drag a native audio stack into
`app` for the sake of reading a header.
"""

import io
import wave

import numpy as np


def encode_wav(samples: np.ndarray, sample_rate: int) -> bytes:
    """PCM-16 WAV bytes for `samples`.

    Accepts `(n,)` mono or `(channels, n)` — the two shapes the model families
    produce. Float input is taken to be in [-1, 1] and is CLIPPED before scaling: a
    diffusion sampler occasionally overshoots, and a wrapped int16 is an audible click.
    """
    array = np.asarray(samples)
    if array.ndim == 1:
        channels, frames = 1, array.reshape(1, -1)
    else:
        channels, frames = array.shape[0], array
    if frames.dtype != np.int16:
        frames = (np.clip(frames.astype(np.float32), -1.0, 1.0) * 32767.0).astype(np.int16)
    # (channels, n) -> n*channels, L R L R …: a channel-major write plays as a stutter.
    interleaved = frames.T.reshape(-1)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(interleaved.tobytes())
    return buffer.getvalue()


def decode_wav(data: bytes) -> tuple[np.ndarray, int]:
    """`(mono float32 in [-1, 1], sample_rate)` for a PCM-16 WAV.

    The counterpart of `encode_wav`, and the only decoder in the app. `feed/beats.py`
    needs samples to find the music's pulse, and it must reach them WITHOUT torch,
    torchaudio or soundfile (CLAUDE.md § "One model process") — so this stays on the
    stdlib `wave` module plus numpy, exactly like the encoder.
    """
    with wave.open(io.BytesIO(data), "rb") as handle:
        rate = handle.getframerate()
        channels = handle.getnchannels()
        frames = np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16)
    if frames.size == 0:
        return np.zeros(0, dtype=np.float32), rate
    if channels > 1:
        frames = frames.reshape(-1, channels).mean(axis=1)
    return (frames.astype(np.float32) / 32767.0), rate


def duration_seconds(data: bytes) -> float:
    """Seconds of audio in a WAV payload, read from its header.

    0.0 for anything unreadable: the caller uses this to FIT a slideshow to a
    narration, so a broken clip must degrade to "no narration timing" rather than
    raise in the middle of a background build.
    """
    try:
        with wave.open(io.BytesIO(data), "rb") as handle:
            rate = handle.getframerate()
            return handle.getnframes() / rate if rate else 0.0
    except Exception:  # noqa: BLE001 - unreadable audio is a fact here, not an error
        return 0.0
