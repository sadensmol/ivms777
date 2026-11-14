import io
import json
import random
import struct
import wave

import numpy as np
import pytest

from models import catalog
from scripts import voice_check
from scripts.voice_check import (
    ClipError,
    check_clip,
    is_void,
    plan_calls,
    run_check,
    segments_monotonic,
    split_sentences,
    wav_metrics,
)
from storage.wav import encode_wav

REVISION = catalog.get("voice", "voxcpm2").source.revision


def tone(seconds: float, rate: int = 48000, amplitude: float = 0.3) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    return (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def clip_bytes(seconds: float = 12.0, rate: int = 48000, channels: int = 1,
               width: int = 2) -> bytes:
    frames = int(seconds * rate)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(width)
        handle.setframerate(rate)
        if width == 2:
            samples = (tone(seconds, rate) * 32767).astype(np.int16)
            handle.writeframes(np.repeat(samples, channels).tobytes())
        else:
            handle.writeframes(b"\x10" * (frames * channels * width))
    return buffer.getvalue()


class FakeClient:
    """A models service stand-in: builds a blob on a call with no stored conditioning
    and restores when handed it back."""

    def __init__(self, *, model: str = "voxcpm2", revision: str = REVISION,
                 free_mb=(20000.0, 9500.0)) -> None:
        self.model, self.revision = model, revision
        self.free = list(free_mb)
        self.events: list[tuple] = []

    def model_unload(self, name: str) -> None:
        self.events.append(("unload", name))

    def models_state(self) -> dict:
        self.events.append(("state",))
        return {"free_mb": self.free.pop(0) if self.free else 0.0}

    def speak(self, text, language, voice, segments, voices, *, references=None,
              timeout=None):
        self.events.append(("speak", text, language, voice, segments, voices, references))
        ref = references[voice]
        stored = ref["conditioning"]
        if stored and stored[0]["blob"] == b"blob-1":
            state = {"state": "restored", "model": "voxcpm2", "revision": self.revision,
                     "reason": None, "blob": None}
        else:
            state = {"state": "built", "model": "voxcpm2", "revision": self.revision,
                     "reason": "no stored conditioning", "blob": b"blob-1"}
        audio = encode_wav(tone(2.0), 48000)
        parts = segments or [text]
        timings = [{"text": part, "start": i * 0.4, "end": (i + 1) * 0.4}
                   for i, part in enumerate(parts)]
        return {"audio": audio, "sample_rate": 48000, "model": self.model,
                "revision": self.revision, "segments": timings,
                "conditioning": {voice: state}}

    def speaks(self) -> list[tuple]:
        return [event for event in self.events if event[0] == "speak"]


def run(tmp_path, client=None, **kwargs):
    client = client or FakeClient()
    clip = check_clip(clip_bytes())
    report = run_check(client, clip, "en", tmp_path, rng=random.Random(0), **kwargs)
    return client, report


def test_a_valid_clip_is_accepted():
    data = clip_bytes(12.0)
    clip = check_clip(data)
    assert clip.seconds == pytest.approx(12.0)
    assert clip.rms > voice_check.SILENCE_RMS
    assert len(clip.sha256) == 64


@pytest.mark.parametrize("data, text", [
    (clip_bytes(rate=44100), "this file is 16-bit mono at 44100 Hz"),
    (clip_bytes(channels=2), "this file is 16-bit stereo at 48000 Hz"),
    (clip_bytes(width=3), "this file is 24-bit mono at 48000 Hz"),
    (clip_bytes()[:-1000], "truncated"),
    (clip_bytes(4.9), "4.90 s"),
    (clip_bytes(30.5), "30.50 s"),
    (b"not a wav at all", "not a WAV file"),
])
def test_the_clip_contract_refuses_a_breach(data, text):
    with pytest.raises(ClipError, match=text):
        check_clip(data)


def test_a_silent_clip_is_refused():
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(48000)
        handle.writeframes(b"\x00\x00" * 48000 * 6)
    with pytest.raises(ClipError, match="silent"):
        check_clip(buffer.getvalue())


def test_a_clip_over_4_mib_is_refused():
    with pytest.raises(ClipError, match="limit"):
        check_clip(b"RIFF" + b"\x00" * (4 * 1024 * 1024))


def test_a_compressed_header_is_refused():
    # A WAVE_FORMAT_ADPCM (2) header: stdlib `wave` raises on it.
    fmt = struct.pack("<HHIIHH", 2, 1, 48000, 48000, 2, 4)
    data = (b"RIFF" + struct.pack("<I", 36) + b"WAVE" + b"fmt " + struct.pack("<I", 16)
            + fmt + b"data" + struct.pack("<I", 0))
    with pytest.raises(ClipError):
        check_clip(data)


@pytest.mark.parametrize("data", [
    clip_bytes(rate=44100), clip_bytes(channels=2), clip_bytes(width=3), clip_bytes()[:-1000],
])
def test_main_refuses_a_bad_clip_before_any_client_call(tmp_path, data, capsys):
    path = tmp_path / "clip.wav"
    path.write_bytes(data)
    client = FakeClient()
    code = voice_check.main(["--wav", str(path), "--language", "en", "--out", str(tmp_path)],
                            client=client)
    assert code == 1
    assert client.events == []
    assert "clip" in capsys.readouterr().err.lower()


def test_the_long_sentences_split_into_five_passages():
    for language in ("en", "ru"):
        long_text = voice_check.SENTENCES[language][-1][1]
        parts = split_sentences(long_text)
        assert len(parts) == 5
        assert " ".join(parts) == long_text


def test_plan_calls_without_and_with_cross():
    calls = plan_calls("en", cross=False)
    assert [c.id for c in calls] == ["EN-1", "EN-2", "EN-3", "EN-4", "EN-5", "EN-5-segments"]
    assert calls[-1].segments and len(calls[-1].segments) == 5
    crossed = plan_calls("ru", cross=True)
    assert [c.id for c in crossed][-5:] == ["EN-1", "EN-2", "EN-3", "EN-4", "EN-5"]
    assert all(c.cross and c.language == "en" for c in crossed[-5:])


def test_wav_metrics():
    metrics = wav_metrics(encode_wav(tone(2.0, amplitude=0.5), 48000))
    assert metrics["duration_s"] == pytest.approx(2.0)
    assert metrics["wav_sample_rate"] == 48000
    assert metrics["bytes"] == 44 + 2 * 96000
    assert metrics["rms"] == pytest.approx(0.5 / np.sqrt(2), rel=1e-3)
    assert metrics["nan"] is False
    assert metrics["silent"] is False


def test_wav_metrics_of_silence():
    metrics = wav_metrics(encode_wav(np.zeros(4800, dtype=np.float32), 48000))
    assert metrics["silent"] is True
    assert metrics["rms_dbfs"] is None


def test_segments_monotonic():
    good = [{"start": 0.0, "end": 1.0}, {"start": 1.0, "end": 2.5}]
    assert segments_monotonic(good, 2)
    assert not segments_monotonic(good, 5)
    assert not segments_monotonic([{"start": 0.0, "end": 1.0}, {"start": 0.5, "end": 2.0}], 2)
    assert not segments_monotonic([{"start": 1.0, "end": 0.5}], None)
    assert segments_monotonic([], None)


def test_is_void():
    assert not is_void({"model": "voxcpm2", "revision": REVISION}, REVISION)
    assert is_void({"model": "hexgrad/Kokoro-82M", "revision": None}, REVISION)
    assert is_void({"model": "voxcpm2", "revision": "0" * 40}, REVISION)
    assert is_void({"model": "voxcpm2"}, REVISION)


def test_a_reply_from_another_model_is_marked_void(tmp_path):
    _client, report = run(tmp_path, FakeClient(model="fake-voice"))
    assert all(row["void"] for row in report["calls"])
    assert report["checks"]["no VOID reply"] is False


def test_a_reply_at_another_revision_is_marked_void(tmp_path):
    _client, report = run(tmp_path, FakeClient(revision="f" * 40))
    assert all(row["void"] for row in report["calls"])


def test_unload_runs_before_each_pass(tmp_path):
    client, _report = run(tmp_path)
    kinds = [event[0] for event in client.events]
    assert kinds[0] == "unload"
    first_unload, second_unload = [i for i, k in enumerate(kinds) if k == "unload"]
    calls = len(plan_calls("en", False))
    assert kinds[first_unload + 1:second_unload].count("speak") == calls
    assert kinds[second_unload + 1:].count("speak") == calls
    assert all(event[1] == "voice" for event in client.events if event[0] == "unload")


def test_pass_2_sends_the_pass_1_blob_and_reads_restored(tmp_path):
    client, report = run(tmp_path)
    speaks = client.speaks()
    half = len(speaks) // 2
    for event in speaks[:half]:
        assert event[6]["check"]["conditioning"] == []
    for event in speaks[half:]:
        assert event[6]["check"]["conditioning"] == [
            {"model": "voxcpm2", "revision": REVISION, "blob": b"blob-1"}]
    states = {(row["pass"], row["conditioning_state"]) for row in report["calls"]}
    assert states == {("fresh", "built"), ("restored", "restored")}
    assert report["checks"]["pass 2 restored"] is True


def test_the_reference_carries_the_clip(tmp_path):
    client, _report = run(tmp_path)
    _kind, text, _language, voice, segments, voices, refs = client.speaks()[0]
    clip = clip_bytes()
    assert voice == "check"
    assert refs["check"]["audio"] == clip
    assert refs["check"]["language"] == "en"
    assert len(refs["check"]["sha256"]) == 64
    assert text == voice_check.SENTENCES["en"][0][1]
    assert segments is None and voices is None


def test_the_segments_call_sends_five_passages(tmp_path):
    client, report = run(tmp_path)
    segment_call = next(e for e in client.speaks() if e[4])
    assert len(segment_call[4]) == 5
    row = next(r for r in report["calls"] if r["id"] == "EN-5-segments")
    assert row["segments_monotonic"] is True


def test_resident_cost_is_the_free_mb_drop_across_the_first_call(tmp_path):
    _client, report = run(tmp_path, FakeClient(free_mb=(20000.0, 9500.0)))
    assert report["resident_cost_mb"] == 10500.0


def test_ab_order_is_recorded_in_key_json(tmp_path):
    _client, report = run(tmp_path)
    key = json.loads((tmp_path / "key.json").read_text())
    assert set(key) == {c.id for c in plan_calls("en", False)}
    for call_id, order in key.items():
        assert sorted(order.values()) == ["fresh", "restored"]
        assert (tmp_path / f"{call_id}-A.wav").exists()
        assert (tmp_path / f"{call_id}-B.wav").exists()
    for row in report["calls"]:
        label = "A" if key[row["id"]]["A"] == row["pass"] else "B"
        assert row["file"] == f"{row['id']}-{label}.wav"
    # The order is random, not always fresh first.
    assert {order["A"] for order in key.values()} == {"fresh", "restored"}


def test_report_json_has_every_per_call_field(tmp_path):
    run(tmp_path, cross=True)
    report = json.loads((tmp_path / "report.json").read_text())
    assert len(report["calls"]) == 2 * len(plan_calls("en", True))
    row = report["calls"][0]
    for field in ("duration_s", "sample_rate", "bytes", "rms", "nan", "model", "revision",
                  "conditioning_state", "wall_s", "rtf", "chars_per_s", "segments_monotonic",
                  "void"):
        assert field in row
    assert report["expected"] == {"model": "voxcpm2", "revision": REVISION,
                                  "sample_rate": 48000}
    assert all(report["checks"].values())


def test_rtf_and_chars_per_second_come_from_the_clock(tmp_path):
    ticks = iter(float(i) for i in range(1000))
    client = FakeClient()
    clip = check_clip(clip_bytes())
    report = run_check(client, clip, "en", tmp_path, rng=random.Random(0),
                       clock=lambda: next(ticks))
    row = report["calls"][0]
    assert row["wall_s"] == 1.0
    assert row["rtf"] == pytest.approx(0.5)
    assert row["chars_per_s"] == pytest.approx(len(voice_check.SENTENCES["en"][0][1]) / 2.0,
                                               rel=1e-2)


def test_a_failed_call_is_reported_and_the_rest_run(tmp_path):
    class Flaky(FakeClient):
        def speak(self, text, *args, **kwargs):
            if text.startswith("Julia"):
                raise RuntimeError("503 voice copy is not available")
            return super().speak(text, *args, **kwargs)

    client, report = run(tmp_path, Flaky())
    errors = [row for row in report["calls"] if "error" in row]
    assert [row["id"] for row in errors] == ["EN-2", "EN-2"]
    assert errors[0]["file"] is None
    assert report["checks"]["every call answered"] is False
    # The failed call raised before the fake recorded it; every other call still ran.
    assert len(report["calls"]) == 2 * len(plan_calls("en", False))
    assert len(client.speaks()) == 2 * len(plan_calls("en", False)) - 2


def test_main_prints_the_summary_and_writes_the_report(tmp_path, capsys):
    path = tmp_path / "clip.wav"
    path.write_bytes(clip_bytes())
    out = tmp_path / "out"
    code = voice_check.main(["--wav", str(path), "--language", "en", "--out", str(out)],
                            client=FakeClient())
    printed = capsys.readouterr().out
    assert code == 0
    assert "resident cost" in printed and "10500 MB" in printed
    assert "RTF <= 1.0" in printed
    assert (out / "report.json").exists()
