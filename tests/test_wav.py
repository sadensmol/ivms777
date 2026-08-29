"""WAV encode/measure (design §11.1). numpy + stdlib only — no torch, no soundfile:
`feed/` (a thin client) and `audio/` (models-service-only) both import this."""

import numpy as np
import pytest

from storage.wav import duration_seconds, encode_wav


def test_mono_float_round_trips_to_a_measurable_duration():
    data = encode_wav(np.zeros(24000, dtype=np.float32), 24000)
    assert data[:4] == b"RIFF"
    assert duration_seconds(data) == pytest.approx(1.0)


def test_stereo_is_interleaved_and_keeps_its_duration():
    data = encode_wav(np.zeros((2, 44100), dtype=np.float32), 44100)
    assert duration_seconds(data) == pytest.approx(1.0)


def test_stereo_channels_stay_separate():
    left = np.full(3, 0.5, dtype=np.float32)
    right = np.full(3, -0.5, dtype=np.float32)
    decoded = np.frombuffer(encode_wav(np.stack([left, right]), 8000)[44:], dtype="<i2")
    # L R L R L R, not LLL RRR — a channel-major write plays as a stutter, not stereo.
    assert decoded[0] > 0 and decoded[1] < 0 and decoded[2] > 0


def test_float_samples_are_scaled_and_clipped_to_int16():
    samples = np.array([0.0, 1.0, -1.0, 2.0, -2.0], dtype=np.float32)
    decoded = np.frombuffer(encode_wav(samples, 8000)[44:], dtype="<i2")
    assert decoded.tolist() == [0, 32767, -32767, 32767, -32767]


def test_int16_samples_are_written_unchanged():
    samples = np.array([0, 1000, -1000], dtype=np.int16)
    decoded = np.frombuffer(encode_wav(samples, 8000)[44:], dtype="<i2")
    assert decoded.tolist() == [0, 1000, -1000]


def test_duration_of_a_non_wav_payload_is_zero():
    # A broken clip must degrade to "no narration timing", never raise inside a
    # background build (§11.1).
    assert duration_seconds(b"not audio at all") == 0.0
