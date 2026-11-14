import hashlib
import io
import json
import math
import random
import re
import threading
import wave
from collections import Counter
from pathlib import Path

import httpx
import numpy as np
import pytest

from audio.silero_voice import split_text
from audio.voxcpm_voice import CONTROLS, _normalise
from config import Settings
from db.connection import connect, migrate
from feed.saved_voices import (
    NO_WORDS,
    RECORDING_KIT,
    RECORDING_PROMPTS,
    add_example,
    create_voice,
    request_example_deletion,
    set_heard_words,
)
from feed.voice_preview import PREVIEW_SENTENCES
from models import catalog
from modelsvc.backends.base import SpeakerEmbedRefused
from modelsvc.backends.fake import FakeBackend
from scripts import voice_bake_check
from scripts.voice_check import SENTENCES
from storage.local import LocalStorage
from storage.wav import encode_wav
from tests.test_feed_build import ADAPTER as STORED_ADAPTER
from tests.test_feed_build import bake

OWNER = 1
REVISION = catalog.get("voice", "voxcpm2").source.revision
ADAPTER = b"adapter-safetensors-bytes"
ADAPTER_SHA = hashlib.sha256(ADAPTER).hexdigest()
EN_WORDS = next(iter(RECORDING_PROMPTS["en"].values()))
RU_WORDS = next(iter(RECORDING_PROMPTS["ru"].values()))
GIVEN_WORDS = "the words I said in the uploaded take"
ASR = "whisper-large-v3-turbo@41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
B_LABEL = "today's films on the LoRA-built child"


def tone(seconds: float, freq: float) -> bytes:
    t = np.arange(int(seconds * 48000)) / 48000
    return encode_wav((0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32), 48000)


def status(state="running", *, step=0, stage="training", device="mps", error=None,
           min_free_mb=None) -> dict:
    """A `/voice/bakes` status in the shape `modelsvc/voice_bakes.py` snapshots."""
    return {
        "id": "bake-1", "state": state, "stage": stage, "fraction": 0.5, "step": step,
        "steps": 4, "loss_first": 2.5, "loss_last": 1.25, "seconds_per_step": 3.0,
        "device": device, "stage_seconds": {"training": 12.0}, "elapsed_s": 20.0,
        "free_mb_before": 9000.0, "min_free_mb": min_free_mb, "peak_device_bytes": 123456,
        "error": error,
        "adapter": ({"sha256": ADAPTER_SHA, "bytes": len(ADAPTER), "keys": 96}
                    if state == "done" else None),
        "model": "voxcpm2", "revision": REVISION, "config": {"steps": 4},
    }


class FakeClient:
    """A models service stand-in: the bake follows `statuses`, one per `bake_status`. An
    exception in `statuses` is raised by its `bake_status`; the first `cancel_failures`
    cancels raise a timeout."""

    def __init__(self, statuses=None, *, free_mb=(9000.0, 8000.0, 7000.0, 8500.0),
                 cancel_failures: int = 0) -> None:
        self.statuses = list(statuses if statuses is not None
                             else [status(step=2, min_free_mb=8000.0),
                                   status("done", step=4, stage="done", min_free_mb=7000.0)])
        self.free = list(free_mb)
        self.events: list[tuple] = []
        self.status_timeouts: list[float | None] = []
        self.cancel_failures = cancel_failures
        self.current = status() | {"elapsed_s": 0.0}

    def models_state(self) -> dict:
        self.events.append(("state",))
        return {"free_mb": self.free.pop(0) if len(self.free) > 1 else self.free[0]}

    def bake_start(self, clips, *, tag, config=None):
        self.events.append(("start", clips, tag, config))
        return self.current

    def bake_status(self, bake_id, *, timeout=None):
        self.events.append(("status", bake_id))
        self.status_timeouts.append(timeout)
        answer = self.statuses.pop(0)
        if isinstance(answer, Exception):
            raise answer
        self.current = answer
        return self.current

    def bake_cancel(self, bake_id):
        self.events.append(("cancel", bake_id))
        if self.cancel_failures:
            self.cancel_failures -= 1
            raise httpx.ReadTimeout("timed out")
        return self.current

    def bake_adapter(self, bake_id):
        self.events.append(("adapter", bake_id))
        return ADAPTER

    def bake_delete(self, bake_id):
        self.events.append(("delete", bake_id))

    def speak(self, text, language, voice, segments, voices, *, references=None,
              timeout=None):
        self.events.append(("speak", text, language, voice, references))
        ref = references[voice]
        reply = {"audio": tone(2.0, 330.0), "sample_rate": 48000, "model": "voxcpm2",
                 "revision": REVISION, "segments": [], "conditioning": {}, "adapters": {}}
        if ref.get("use_clip", True):
            reply["conditioning"] = {voice: {"state": "built", "model": "voxcpm2",
                                             "revision": REVISION, "reason": None,
                                             "blob": b"blob"}}
        if ref.get("adapter") is not None:
            reply["adapters"] = {voice: {"state": "loaded", "sha256": ref["adapter"]["sha256"]}}
        return reply

    def kinds(self) -> list[str]:
        return [event[0] for event in self.events]

    def speaks(self) -> list[tuple]:
        return [event for event in self.events if event[0] == "speak"]


@pytest.fixture
def world(tmp_path, monkeypatch):
    """The user's data dir: a migrated DB with the voice «me» (a recorded en take whose
    words the app's bake heard, an uploaded en take with no words, a recorded ru take
    stored before ASR with its shown text, and one en take being deleted)."""
    monkeypatch.setenv("IVMS777_DATA_DIR", str(tmp_path))
    settings = Settings(data_dir=tmp_path, models_base_url="http://models.invalid:9000")
    conn = connect(settings.db_path)
    migrate(conn)
    store = LocalStorage(settings.voices_dir)
    voice = create_voice(conn, OWNER, "me")

    def add(language, source, freq):
        return add_example(conn, store, OWNER, voice.id, language=language, source=source,
                           data=tone(6.0, freq))

    takes = {"recorded": add("en", "recorded", 220.0), "uploaded": add("en", "uploaded", 240.0),
             "ru": add("ru", "recorded", 260.0), "deleting": add("en", "recorded", 280.0)}
    set_heard_words(conn, OWNER, voice.id, takes["recorded"].id, text=EN_WORDS, model=ASR)
    conn.execute("UPDATE voice_examples SET transcript = ? WHERE id = ?", (RU_WORDS, takes["ru"].id))
    request_example_deletion(conn, OWNER, voice.id, takes["deleting"].id)
    conn.close()
    return settings, takes


def add_unusable_take(settings) -> int:
    """One more en take whose heard words held none (D94)."""
    conn = connect(settings.db_path)
    store = LocalStorage(settings.voices_dir)
    (voice,) = conn.execute("SELECT id FROM saved_voices").fetchall()
    example = add_example(conn, store, OWNER, voice["id"], language="en", source="uploaded",
                          data=tone(6.0, 300.0))
    set_heard_words(conn, OWNER, voice["id"], example.id, text="♪ ♪", model=ASR)
    conn.close()
    return example.id


def run_main(settings, args, client, *, sleep=lambda _seconds: None, out=None):
    argv = ["--voice", "me", *args]
    if out is not None:
        argv += ["--out", str(out)]
    return voice_bake_check.main(argv, client=client, settings=settings, sleep=sleep,
                                 rng=random.Random(0))


def confirm(takes) -> list[str]:
    return ["--confirm-texts", "--text", f"{takes['uploaded'].id}={GIVEN_WORDS}"]


def ctrl_c_once():
    calls = []

    def sleep(_seconds):
        calls.append(1)
        if len(calls) == 1:
            raise KeyboardInterrupt

    return sleep


def snapshot(root: Path) -> dict[str, bytes]:
    return {path.relative_to(root).as_posix(): path.read_bytes()
            for path in sorted(root.rglob("*")) if path.is_file()}


def test_list_prints_every_take_with_its_words_and_starts_no_bake(world, capsys):
    settings, takes = world
    client = FakeClient()
    code = run_main(settings, ["--list"], client)
    printed = capsys.readouterr().out
    assert code == 0
    assert client.events == []
    assert f"take {takes['recorded'].id} · en" in printed
    assert f"take {takes['ru'].id} · ru" in printed
    assert f"take {takes['deleting'].id} " not in printed
    lines = printed.splitlines()
    assert f"take {takes['recorded'].id} · en · 6.0 s · recorded · heard by whisper-large-v3-turbo@41f01f3" in lines
    assert f"take {takes['ru'].id} · ru · 6.0 s · recorded · shown text, not heard" in lines
    assert f"take {takes['uploaded'].id} · en · 6.0 s · uploaded · NO WORDS" in lines
    assert "reading text" not in printed and "not checked" not in printed
    assert EN_WORDS in printed and RU_WORDS in printed
    assert "5-10 minutes" in printed


def test_list_names_a_take_with_no_usable_words_and_its_reason(world, capsys):
    settings, _takes = world
    unusable = add_unusable_take(settings)
    assert run_main(settings, ["--list"], FakeClient()) == 0
    printed = capsys.readouterr().out
    assert f"take {unusable} · en · 6.0 s · uploaded · no usable words: {NO_WORDS}" in printed
    assert printed.count("NO WORDS") == 1


def test_a_take_with_no_words_stops_the_run_and_is_named(world, tmp_path, capsys):
    settings, takes = world
    client = FakeClient()
    code = run_main(settings, ["--confirm-texts"], client, out=tmp_path / "out")
    assert code == 2
    assert f"take {takes['uploaded'].id}" in capsys.readouterr().err
    assert "start" not in client.kinds()


def test_a_take_with_no_usable_words_still_needs_its_text(world, tmp_path, capsys):
    settings, takes = world
    unusable = add_unusable_take(settings)
    client = FakeClient()
    code = run_main(settings, confirm(takes), client, out=tmp_path / "out")
    assert code == 2
    assert f"take {unusable}" in capsys.readouterr().err
    assert "start" not in client.kinds()
    given = [*confirm(takes), "--text", f"{unusable}={GIVEN_WORDS}"]
    assert run_main(settings, given, FakeClient(), out=tmp_path / "given") == 0


def test_a_run_without_confirm_texts_is_refused(world, tmp_path):
    settings, takes = world
    client = FakeClient()
    with pytest.raises(SystemExit) as stopped:
        run_main(settings, ["--text", f"{takes['uploaded'].id}={GIVEN_WORDS}"], client)
    assert stopped.value.code == 2
    assert client.events == []


def test_text_for_an_unknown_or_deleting_take_is_refused(world, tmp_path, capsys):
    settings, takes = world
    client = FakeClient()
    code = run_main(settings, [*confirm(takes), "--text", f"{takes['deleting'].id}=words"],
                    client, out=tmp_path / "out")
    assert code == 2
    assert str(takes["deleting"].id) in capsys.readouterr().err
    assert client.events == []


def test_bake_start_gets_every_take_with_its_words(world, tmp_path):
    settings, takes = world
    client = FakeClient()
    run_main(settings, [*confirm(takes), "--steps", "4", "--learning-rate", "0.0002"],
             client, out=tmp_path / "out")
    (_kind, clips, tag, config), = [e for e in client.events if e[0] == "start"]
    store = LocalStorage(settings.voices_dir)
    expected = [
        {"audio": store.read(takes[name].storage_key), "sha256": takes[name].content_hash,
         "text": words, "language": takes[name].language}
        for name, words in (("recorded", EN_WORDS), ("uploaded", GIVEN_WORDS),
                            ("ru", RU_WORDS))
    ]
    assert clips == expected
    assert tag == {"check": "me"}
    assert config == {"steps": 4, "learning_rate": 0.0002}


def test_no_loop_flag_sends_no_config(world, tmp_path):
    settings, takes = world
    client = FakeClient()
    run_main(settings, confirm(takes), client, out=tmp_path / "out")
    (_kind, _clips, _tag, config), = [e for e in client.events if e[0] == "start"]
    assert config is None


def test_ctrl_c_cancels_waits_for_cancelled_and_exits_130(world, tmp_path, capsys):
    settings, takes = world
    out = tmp_path / "out"
    client = FakeClient([status(step=2), status("cancelled", step=2)])
    code = run_main(settings, confirm(takes), client, sleep=ctrl_c_once(), out=out)
    assert code == 130
    kinds = client.kinds()
    assert kinds.index("cancel") < kinds.index("delete")
    assert "adapter" not in kinds and "speak" not in kinds
    assert kinds[-1] == "delete"
    report = json.loads((out / "report.json").read_text())
    assert report["cancelled_by_user"] is True
    assert report["bake"]["state"] == "cancelled"
    assert report["adapter"] is None
    assert "cancelled at step 2" in capsys.readouterr().out


@pytest.mark.parametrize("end", ["done", "failed"])
def test_an_end_other_than_cancelled_after_ctrl_c_still_ends_the_wait(world, tmp_path, end):
    settings, takes = world
    out = tmp_path / "out"
    client = FakeClient([status(step=3), status(end, step=4, error="boom" if end == "failed"
                                                else None)])
    code = run_main(settings, confirm(takes), client, sleep=ctrl_c_once(), out=out)
    assert code == 130
    assert "cancel" in client.kinds()
    assert "adapter" not in client.kinds()
    assert "delete" in client.kinds()
    assert not (out / "adapter.safetensors").exists()
    assert json.loads((out / "report.json").read_text())["adapter"] is None


def test_a_failed_bake_reports_the_error_and_exits_1(world, tmp_path, capsys):
    settings, takes = world
    out = tmp_path / "out"
    client = FakeClient([status("failed", step=1, error="InsufficientMemory: no room")])
    code = run_main(settings, confirm(takes), client, out=out)
    assert code == 1
    assert "delete" in client.kinds() and "adapter" not in client.kinds()
    report = json.loads((out / "report.json").read_text())
    assert report["bake"]["error"] == "InsufficientMemory: no room"
    assert report["checks"]["bake done on mps"] is False
    assert "InsufficientMemory" in capsys.readouterr().out


@pytest.fixture
def done_run(world, tmp_path):
    settings, takes = world
    out = tmp_path / "out"
    client = FakeClient()
    code = run_main(settings, confirm(takes), client, out=out)
    return code, client, out, takes


def test_done_writes_the_adapter_and_deletes_the_bake(done_run):
    code, client, out, _takes = done_run
    assert code == 0
    assert (out / "adapter.safetensors").read_bytes() == ADAPTER
    kinds = client.kinds()
    assert kinds.count("delete") == 1
    assert kinds.index("adapter") < kinds.index("delete")


def test_three_speak_calls_per_sentence(done_run):
    _code, client, _out, takes = done_run
    speaks = client.speaks()
    sentences = [(text, language) for language in ("en", "ru")
                 for _sid, text in SENTENCES[language]]
    assert len(speaks) == 3 * len(sentences)
    for index, (text, language) in enumerate(sentences):
        group = speaks[3 * index:3 * index + 3]
        assert {(e[1], e[2], e[3]) for e in group} == {(text, language, "saved:check")}
        refs = [e[4]["saved:check"] for e in group]
        prompt = takes["uploaded"] if language == "en" else takes["ru"]
        assert all(ref["sha256"] == prompt.content_hash for ref in refs)
        shapes = sorted((ref.get("adapter") is not None, ref.get("use_clip")) for ref in refs)
        assert shapes == [(False, None), (True, False), (True, True)]
        for ref in refs:
            if ref.get("adapter") is not None:
                assert ref["adapter"] == {"blob": ADAPTER, "sha256": ADAPTER_SHA}


def test_the_prompt_take_is_the_newest_take_of_each_language(done_run):
    _code, client, out, takes = done_run
    by_language = {e[2]: e[4]["saved:check"]["sha256"] for e in client.speaks()}
    # The newest take that is not being deleted.
    assert by_language == {"en": takes["uploaded"].content_hash,
                           "ru": takes["ru"].content_hash}
    report = json.loads((out / "report.json").read_text())
    assert report["prompt_takes"] == {"en": {"example_id": takes["uploaded"].id, "rule": "newest"},
                                      "ru": {"example_id": takes["ru"].id, "rule": "newest"}}


def test_an_old_film_take_pin_is_ignored(world, tmp_path):
    settings, takes = world
    conn = connect(settings.db_path)
    conn.execute(
        "INSERT INTO voice_film_examples(voice_id, language, example_id, chosen_at)"
        " VALUES (?, 'en', ?, '2026-01-01')",
        (takes["recorded"].voice_id, takes["recorded"].id),
    )
    conn.close()
    client = FakeClient()
    assert run_main(settings, confirm(takes), client, out=tmp_path / "out") == 0
    english = {e[4]["saved:check"]["sha256"] for e in client.speaks() if e[2] == "en"}
    assert english == {takes["uploaded"].content_hash}


def test_files_get_random_xyz_names_and_key_json_maps_them(done_run):
    _code, _client, out, _takes = done_run
    key = json.loads((out / "key.json").read_text())
    ids = [sid for language in ("en", "ru") for sid, _text in SENTENCES[language]]
    assert sorted(key) == sorted(ids)
    for sentence_id, order in key.items():
        assert sorted(order) == ["X", "Y", "Z"]
        assert sorted(order.values()) == ["A", "B", "C"]
        for label in order:
            assert (out / f"{sentence_id}-{label}.wav").exists()
    assert len({order["X"] for order in key.values()}) > 1
    report = json.loads((out / "report.json").read_text())
    for row in report["calls"]:
        label = next(lab for lab, variant in key[row["id"]].items() if variant == row["variant"])
        assert row["file"] == f"{row['id']}-{label}.wav"


def test_report_json_holds_the_fields_of_step_7(done_run):
    _code, _client, out, _takes = done_run
    report = json.loads((out / "report.json").read_text())
    bake = report["bake"]
    for name in ("stage_seconds", "steps", "seconds_per_step", "loss_first", "loss_last",
                 "peak_device_bytes", "device"):
        assert name in bake
    assert report["free_mb_before"] == 9000.0
    assert report["min_free_mb"] == 7000.0
    assert report["adapter"] == {"file": "adapter.safetensors", "sha256": ADAPTER_SHA,
                                 "bytes": len(ADAPTER), "keys": 96}
    assert report["takes"]["count"] == 3
    assert report["takes"]["seconds_by_language"] == {"en": 12.0, "ru": 6.0}
    assert report["takes"]["total_seconds"] == 18.0
    row = report["calls"][0]
    for name in ("model", "revision", "adapter_state", "conditioning_state", "wall_s", "rtf",
                 "duration_s", "rms", "nan"):
        assert name in row
    assert set(report["mean_rtf"]) == {"A", "B", "C"}
    assert all(mean["rtf"] is not None for mean in report["mean_rtf"].values())
    assert all(report["checks"].values())


def test_the_report_labels_b_as_todays_films_on_the_lora_built_child(capsys, done_run):
    _code, _client, out, _takes = done_run
    report = json.loads((out / "report.json").read_text())
    assert report["mean_rtf"]["B"]["label"] == B_LABEL
    assert B_LABEL in capsys.readouterr().out


def file_rows(printed: str) -> list[list[str]]:
    rows = [line.split() for line in printed.splitlines()]
    return [row for row in rows if len(row) > 1 and row[1].endswith(".wav")]


def test_the_console_names_every_file_without_its_variant(world, tmp_path, capsys):
    settings, takes = world
    out = tmp_path / "out"
    run_main(settings, confirm(takes), FakeClient(), out=out)
    printed = capsys.readouterr().out
    report = json.loads((out / "report.json").read_text())
    header = next(line for line in printed.splitlines() if line.startswith("sentence"))
    assert header.split() == ["sentence", "file", "dur", "s", "dBFS", "identity"]
    expected = [[row["id"], row["file"], f"{row['duration_s']:.2f}", f"{row['rms_dbfs']:.1f}",
                 "ok"] for row in report["calls"]]
    assert file_rows(printed) == expected
    key = json.loads((out / "key.json").read_text())
    assert {row[1] for row in file_rows(printed)} == {
        f"{sentence_id}-{label}.wav" for sentence_id, order in key.items() for label in order}
    for variant, label in voice_bake_check.VARIANTS.items():
        assert f"mean RTF {variant} ({label}): " in printed


def test_an_error_row_prints_its_file_but_not_its_reason(world, tmp_path, capsys):
    reason = "the service fell over"

    class FirstCallFails(FakeClient):
        def speak(self, *args, **kwargs):
            if not self.speaks():
                self.events.append(("speak", *args))
                raise RuntimeError(reason)
            return super().speak(*args, **kwargs)

    settings, takes = world
    out = tmp_path / "out"
    run_main(settings, confirm(takes), FirstCallFails(), out=out)
    printed = capsys.readouterr().out
    first = json.loads((out / "report.json").read_text())["calls"][0]
    assert first["error"] == reason
    assert first["file"] == f"{first['id']}-X.wav"
    assert [first["id"], first["file"], "ERROR"] in file_rows(printed)
    assert reason not in printed


def test_min_free_mb_takes_the_lower_service_minimum(world, tmp_path):
    settings, takes = world
    out = tmp_path / "out"
    client = FakeClient([status("done", step=4, stage="done", min_free_mb=500.0)])
    run_main(settings, confirm(takes), client, out=out)
    report = json.loads((out / "report.json").read_text())
    assert report["min_free_mb"] == 500.0
    assert report["measured_thresholds"]["min_free_mb >= 1024"] is False


def test_a_bake_on_another_device_fails_the_automatic_check(world, tmp_path):
    settings, takes = world
    out = tmp_path / "out"
    client = FakeClient([status("done", step=4, stage="done", device="cpu")])
    code = run_main(settings, confirm(takes), client, out=out)
    assert code == 1
    assert json.loads((out / "report.json").read_text())["checks"]["bake done on mps"] is False


def test_an_adapter_reported_none_fails_the_automatic_check(world, tmp_path):
    class NoAdapter(FakeClient):
        def speak(self, *args, **kwargs):
            reply = super().speak(*args, **kwargs)
            reply["adapters"] = {token: {"state": "none", "sha256": None}
                                 for token in reply["adapters"]}
            return reply

    settings, takes = world
    out = tmp_path / "out"
    code = run_main(settings, confirm(takes), NoAdapter(), out=out)
    report = json.loads((out / "report.json").read_text())
    assert code == 1
    assert report["checks"]["adapter loaded or kept in every A and C reply"] is False


def test_the_database_and_the_voices_store_are_unchanged(world, tmp_path):
    settings, takes = world
    db_before = settings.db_path.read_bytes()
    voices_before = snapshot(settings.voices_dir)
    run_main(settings, ["--list"], FakeClient())
    run_main(settings, confirm(takes), FakeClient(), out=tmp_path / "out")
    assert settings.db_path.read_bytes() == db_before
    assert snapshot(settings.voices_dir) == voices_before


DIM = 192
ARM_COSINE = {"A": 0.9, "B": 0.8, "C": 0.6}
REPEAT_STEP = 0.01
WINDOW_COSINE = 0.5
LONG_SECONDS = 25.0
WINDOW_BYTES = 10 * 48000 * 2


def at_cosine(value: float) -> list[float]:
    """A unit vector at cosine `value` with the takes' centroid, the first axis."""
    return [value, math.sqrt(1 - value ** 2)] + [0.0] * (DIM - 2)


def arm_of(reference: dict) -> str:
    if reference.get("adapter") is None:
        return "B"
    return "A" if reference.get("use_clip") else "C"


class MeterClient(FakeClient):
    """Every speak call makes a new clip whose scripted speaker vector follows its arm and
    its repeat: every take sits on the first axis, so the expected cosines are literals.
    `embed_voice` answers as the service: a refusal is a 422. `refuse_windows` names, per
    arm, the window positions of a long clip that the meter refuses."""

    def __init__(self, *, down: bool = False, refuse_arm: str | None = None,
                 refuse_windows: dict[str, list[int]] | None = None) -> None:
        super().__init__()
        self.backend = FakeBackend()
        self.down, self.refuse_arm = down, refuse_arm
        self.refuse_windows = refuse_windows or {}
        self.embedded: list[bytes] = []
        self.per_text: dict[tuple[str, str], int] = {}

    def embed_voice(self, audio):
        self.embedded.append(audio)
        request = httpx.Request("POST", "http://models/voice/embed")
        if self.down:
            raise httpx.HTTPStatusError("503", request=request, response=httpx.Response(
                503, request=request, json={"detail": "not offered"}))
        sha256 = hashlib.sha256(audio).hexdigest()
        if sha256 not in self.backend.speaker_vectors and sha256 not in self.backend.speaker_refusals:
            first = 1.0 if len(audio) < WINDOW_BYTES else WINDOW_COSINE
            self.backend.speaker_vectors[sha256] = at_cosine(first)
        try:
            return self.backend.embed_voice(audio)
        except SpeakerEmbedRefused as exc:
            raise httpx.HTTPStatusError("422", request=request, response=httpx.Response(
                422, request=request, json={"detail": str(exc)})) from exc

    def speak(self, text, language, voice, segments, voices, *, references=None, timeout=None):
        reply = super().speak(text, language, voice, segments, voices, references=references,
                              timeout=timeout)
        arm = arm_of(references[voice])
        repeat = self.per_text[text, arm] = self.per_text.get((text, arm), 0) + 1
        seconds = LONG_SECONDS if len(text) > 500 else 2.0
        # A whole number of hertz repeats every second: all windows would be the same bytes.
        audio = tone(seconds, 300.37 + len(self.events))
        reply["audio"] = audio
        sha256 = hashlib.sha256(audio).hexdigest()
        for position in self.refuse_windows.get(arm, []) if seconds == LONG_SECONDS else []:
            clip = voice_bake_check.fixed_windows(audio)[position][2]
            self.backend.speaker_refusals[hashlib.sha256(clip).hexdigest()] = "too little voiced audio"
        if arm == self.refuse_arm:
            self.backend.speaker_refusals[sha256] = "too little voiced audio"
        elif seconds < 10.0:
            self.backend.speaker_vectors[sha256] = at_cosine(ARM_COSINE[arm] - REPEAT_STEP * repeat)
        return reply


@pytest.fixture
def baked_world(world):
    settings, takes = world
    conn = connect(settings.db_path)
    bake(conn, LocalStorage(settings.voices_dir), takes["recorded"].voice_id)
    conn.close()
    return settings, takes


def measure(settings, client, out, *args):
    return run_main(settings, ["--measure", *args], client, out=out)


def report_of(out: Path) -> dict:
    return json.loads((out / "report.json").read_text())


def speaks_of(client, arm: str) -> list[tuple]:
    return [event for event in client.speaks() if arm_of(event[4][event[3]]) == arm]


def test_the_long_passage_crosses_a_piece_boundary_and_is_unseen():
    (text_id, text), = voice_bake_check.LONG_PASSAGES["ru"]
    assert len(text) > 500
    assert len(split_text(text)) > 1
    seen = [kit.text for kit in RECORDING_KIT["ru"]] + list(PREVIEW_SENTENCES["ru"].values())
    assert all(sentence.strip() not in other for sentence in text.split(".") if sentence.strip()
               for other in seen)
    assert text_id in voice_bake_check.WINDOWED


def test_measure_starts_no_bake_and_speaks_every_text_per_arm_and_repeat(baked_world, tmp_path):
    settings, _ = baked_world
    client = MeterClient()
    assert measure(settings, client, tmp_path / "out", "--repeats", "2") == 0
    assert "start" not in client.kinds()
    texts = len(SENTENCES["en"]) + len(SENTENCES["ru"]) + 1
    assert len(client.speaks()) == texts * 3 * 2
    long_text = voice_bake_check.LONG_PASSAGES["ru"][0][1]
    assert sum(event[1] == long_text for event in client.speaks()) == 3 * 2


def test_repeats_default_to_five(baked_world, tmp_path):
    settings, _ = baked_world
    client = MeterClient()
    measure(settings, client, tmp_path / "out")
    assert report_of(tmp_path / "out")["repeats"] == 5
    assert len([e for e in speaks_of(client, "C") if e[1] == SENTENCES["en"][0][1]]) == 5


def test_measure_speaks_c_exactly_as_a_film_sends_the_stored_model(baked_world, tmp_path):
    settings, _ = baked_world
    client = MeterClient()
    measure(settings, client, tmp_path / "out", "--repeats", "1")
    sha256 = hashlib.sha256(STORED_ADAPTER).hexdigest()
    for event in speaks_of(client, "C"):
        assert event[4][event[3]] == {"language": event[2], "use_clip": False,
                                      "adapter": {"blob": STORED_ADAPTER, "sha256": sha256}}
    for event in speaks_of(client, "A"):
        assert event[4][event[3]]["adapter"]["sha256"] == sha256
        assert "audio" in event[4][event[3]]
    assert all("adapter" not in event[4][event[3]] for event in speaks_of(client, "B"))


def test_a_voice_with_no_stored_model_stops_with_no_call(world, tmp_path, capsys):
    settings, _ = world
    client = MeterClient()
    assert measure(settings, client, tmp_path / "out") == voice_bake_check.EXIT_USAGE
    assert "no stored voice model" in capsys.readouterr().err
    assert client.embedded == [] and client.speaks() == []


def test_a_meter_that_cannot_run_stops_before_any_speech(baked_world, tmp_path):
    settings, _ = baked_world
    client = MeterClient(down=True)
    assert measure(settings, client, tmp_path / "out") == 1
    assert client.speaks() == []


def test_every_file_is_scored_against_the_takes_of_its_language(baked_world, tmp_path):
    settings, takes = baked_world
    measure(settings, MeterClient(), tmp_path / "out", "--repeats", "2")
    report = report_of(tmp_path / "out")
    ru = report["frozen"]["reference"]["ru"]
    en = report["frozen"]["reference"]["en"]
    assert [row["example_id"] for row in ru["takes"]] == [takes["ru"].id]
    assert [row["example_id"] for row in en["takes"]] == [takes["recorded"].id, takes["uploaded"].id]
    rows = [row for row in report["calls"] if row["id"] == "EN-1"]
    assert sorted((row["variant"], row["repeat"], row["cosine"]) for row in rows) == [
        ("A", 1, pytest.approx(0.89)), ("A", 2, pytest.approx(0.88)),
        ("B", 1, pytest.approx(0.79)), ("B", 2, pytest.approx(0.78)),
        ("C", 1, pytest.approx(0.59)), ("C", 2, pytest.approx(0.58))]
    assert all((tmp_path / "out" / row["file"]).exists() for row in report["calls"])


def test_the_own_range_is_the_only_scale_in_the_report(baked_world, tmp_path):
    settings, _ = baked_world
    measure(settings, MeterClient(), tmp_path / "out", "--repeats", "1")
    report = report_of(tmp_path / "out")
    en, ru = report["frozen"]["reference"]["en"], report["frozen"]["reference"]["ru"]
    assert en["own"] == {"min": 1.0, "median": 1.0, "max": 1.0}
    assert ru["own"] is None
    assert "thresholds" not in report
    assert "other_voice" not in (tmp_path / "out" / "report.json").read_text()


def test_the_report_gives_mean_worst_and_spread_per_arm_and_per_text(baked_world, tmp_path):
    settings, _ = baked_world
    measure(settings, MeterClient(), tmp_path / "out", "--repeats", "2")
    summary = report_of(tmp_path / "out")["likeness"]
    assert summary["by_text"]["EN-1"]["C"] == {
        "runs": 2, "failures": 0, "mean": pytest.approx(0.585), "worst": pytest.approx(0.58),
        "spread": pytest.approx(0.01)}
    assert summary["by_arm"]["C"]["runs"] == 2 * (len(SENTENCES["en"]) + len(SENTENCES["ru"]) + 1)
    assert summary["by_arm"]["A"]["mean"] > summary["by_arm"]["C"]["mean"]


def test_the_report_freezes_its_inputs(baked_world, tmp_path):
    settings, _ = baked_world
    measure(settings, MeterClient(), tmp_path / "out", "--repeats", "1")
    frozen = report_of(tmp_path / "out")["frozen"]
    assert frozen["adapter_sha256"] == hashlib.sha256(STORED_ADAPTER).hexdigest()
    assert frozen["base_revision"] == REVISION
    assert frozen["controls"] == CONTROLS
    assert frozen["texts"]["RU-5"] == dict(SENTENCES["ru"])["RU-5"]
    assert frozen["texts"]["RU-LONG"] == voice_bake_check.LONG_PASSAGES["ru"][0][1]
    assert frozen["mode"]["ru"]["C"] == "adapter alone"
    assert frozen["mode"]["ru"]["B"] == "ref_continuation"
    reference = frozen["reference"]["ru"]
    assert (reference["evaluator"], reference["pipeline"]) == ("fake-speaker@fake", 1)
    assert len(reference["reference_hash"]) == 64


def test_every_result_row_holds_nothing_of_the_model(baked_world, tmp_path):
    settings, _ = baked_world
    measure(settings, MeterClient(), tmp_path / "out", "--repeats", "1", "--label", "lora-r8")
    report = report_of(tmp_path / "out")
    assert report["label"] == "lora-r8"
    row = next(entry for entry in report["results"]
               if (entry["text_id"], entry["arm"]) == ("EN-1", "C"))
    assert row == {"label": "lora-r8", "arm": "C", "model": "voxcpm2", "revision": REVISION,
                   "text_id": "EN-1", "repeat": 1, "cosine": pytest.approx(0.59), "seconds": 2.0,
                   "windows": [], "error": None}


def test_a_run_with_no_label_is_named_after_its_voice_model(baked_world, tmp_path):
    settings, _ = baked_world
    measure(settings, MeterClient(), tmp_path / "out", "--repeats", "1")
    assert report_of(tmp_path / "out")["label"] == "voxcpm2 v1"


def test_the_long_passage_is_scored_in_windows_of_one_length(baked_world, tmp_path):
    settings, _ = baked_world
    client = MeterClient()
    measure(settings, client, tmp_path / "out", "--repeats", "1")
    report = report_of(tmp_path / "out")
    row = next(entry for entry in report["calls"] if entry["id"] == "RU-LONG")
    assert [(w["start_s"], w["end_s"]) for w in row["windows"]] == [(0.0, 10.0), (10.0, 20.0), (15.0, 25.0)]
    assert row["cosine"] == pytest.approx(WINDOW_COSINE)
    positions = report["likeness"]["windows"]["RU-LONG"]["C"]
    assert [position["runs"] for position in positions] == [1, 1, 1]


def test_a_window_is_the_exact_audio_of_its_span():
    audio = tone(LONG_SECONDS, 440.0)
    with wave.open(io.BytesIO(audio), "rb") as handle:
        pcm = handle.readframes(handle.getnframes())
    start, end, clip = voice_bake_check.fixed_windows(audio)[-1]
    with wave.open(io.BytesIO(clip), "rb") as handle:
        assert (handle.getframerate(), handle.getnchannels()) == (48000, 1)
        assert handle.readframes(handle.getnframes()) == pcm[int(start * 96000):int(end * 96000)]


def test_a_clip_under_one_window_is_one_window():
    audio = tone(4.0, 440.0)
    assert voice_bake_check.fixed_windows(audio) == [(0.0, 4.0, audio)]


def test_a_refused_file_keeps_its_reason_and_counts_as_a_failure(baked_world, tmp_path):
    settings, _ = baked_world
    assert measure(settings, MeterClient(refuse_arm="B"), tmp_path / "out", "--repeats", "1") == 1
    report = report_of(tmp_path / "out")
    row = next(entry for entry in report["calls"]
               if (entry["id"], entry["variant"]) == ("EN-1", "B"))
    assert "too little voiced audio" in row["score_error"]
    assert "cosine" not in row
    assert report["likeness"]["by_text"]["EN-1"]["B"]["failures"] == 1
    assert report["likeness"]["by_text"]["EN-1"]["B"]["mean"] is None
    assert not report["checks"]["every file scored"]


def long_row(report: dict, arm: str) -> dict:
    return next(entry for entry in report["calls"]
                if (entry["id"], entry["variant"]) == ("RU-LONG", arm))


def test_one_refused_window_leaves_the_long_file_with_no_score(baked_world, tmp_path):
    settings, _ = baked_world
    client = MeterClient(refuse_windows={"C": [1]})
    assert measure(settings, client, tmp_path / "out", "--repeats", "1") == 1
    report = report_of(tmp_path / "out")
    row = long_row(report, "C")
    assert "cosine" not in row
    assert row["score_error"] == "1 of 3 windows could not be scored"
    assert [window.get("cosine") for window in row["windows"]] == [
        pytest.approx(WINDOW_COSINE), None, pytest.approx(WINDOW_COSINE)]
    assert "too little voiced audio" in row["windows"][1]["score_error"]
    result = next(entry for entry in report["results"]
                  if (entry["text_id"], entry["arm"]) == ("RU-LONG", "C"))
    assert result["cosine"] is None
    assert result["error"] == "1 of 3 windows could not be scored"
    assert report["likeness"]["by_text"]["RU-LONG"]["C"]["failures"] == 1
    assert report["likeness"]["by_text"]["RU-LONG"]["C"]["mean"] is None
    assert report["likeness"]["by_arm"]["C"]["failures"] == 1
    assert report["likeness"]["by_arm"]["A"]["failures"] == 0
    assert not report["checks"]["every file scored"]


def test_a_file_with_every_window_refused_counts_them_all(baked_world, tmp_path):
    settings, _ = baked_world
    client = MeterClient(refuse_windows={"C": [0, 1, 2]})
    assert measure(settings, client, tmp_path / "out", "--repeats", "1") == 1
    report = report_of(tmp_path / "out")
    row = long_row(report, "C")
    assert "cosine" not in row
    assert row["score_error"] == "3 of 3 windows could not be scored"
    assert all("too little voiced audio" in window["score_error"] for window in row["windows"])
    assert report["likeness"]["by_arm"]["C"]["failures"] == 1
    assert not report["checks"]["every file scored"]


def test_the_measure_console_prints_no_score_and_no_arm(baked_world, tmp_path, capsys):
    settings, _ = baked_world
    measure(settings, MeterClient(), tmp_path / "out", "--repeats", "1")
    printed = capsys.readouterr().out
    assert "cosine" not in printed.lower() and "similarity" not in printed.lower()
    assert "0.59" not in printed and "0.89" not in printed
    assert all(row[1].rsplit("-", 1)[1] in ("X.wav", "Y.wav", "Z.wav") for row in file_rows(printed))


def test_key_json_maps_every_measure_file_to_its_arm(baked_world, tmp_path):
    settings, _ = baked_world
    measure(settings, MeterClient(), tmp_path / "out", "--repeats", "2")
    key = json.loads((tmp_path / "out" / "key.json").read_text())
    report = report_of(tmp_path / "out")
    for row in report["calls"]:
        label = row["file"].removesuffix(".wav").rsplit("-", 1)[1]
        assert key[f"{row['id']}-{row['repeat']}"][label] == row["variant"]


def test_measure_leaves_the_database_and_the_store_unchanged(baked_world, tmp_path):
    settings, _ = baked_world
    db_before = settings.db_path.read_bytes()
    voices_before = snapshot(settings.voices_dir)
    measure(settings, MeterClient(), tmp_path / "out", "--repeats", "1")
    assert settings.db_path.read_bytes() == db_before
    assert snapshot(settings.voices_dir) == voices_before
    conn = connect(settings.db_path)
    assert conn.execute("SELECT COUNT(*) FROM voice_take_embeddings").fetchone()[0] == 0
    conn.close()


def test_a_ctrl_c_keeps_the_files_made_so_far_and_scores_them(baked_world, tmp_path):
    settings, _ = baked_world

    class Stopped(MeterClient):
        def speak(self, *args, **kwargs):
            if len(self.speaks()) == 4:
                raise KeyboardInterrupt
            return super().speak(*args, **kwargs)

    code = measure(settings, Stopped(), tmp_path / "out", "--repeats", "1")
    assert code == voice_bake_check.EXIT_INTERRUPTED
    report = report_of(tmp_path / "out")
    assert report["cancelled_by_user"] is True
    assert len(report["calls"]) == 4
    assert all("cosine" in row for row in report["calls"])


def test_repeats_under_one_is_refused(baked_world, tmp_path):
    settings, _ = baked_world
    with pytest.raises(SystemExit) as stopped:
        measure(settings, MeterClient(), tmp_path / "out", "--repeats", "0")
    assert stopped.value.code == voice_bake_check.EXIT_USAGE


def test_the_bake_path_scores_every_file(world, tmp_path, capsys):
    settings, takes = world
    code = run_main(settings, confirm(takes), MeterClient(), out=tmp_path / "out")
    assert code == 0
    report = report_of(tmp_path / "out")
    assert all("cosine" in row for row in report["calls"])
    assert report["likeness"]["by_arm"]["C"]["runs"] == len(SENTENCES["en"]) + len(SENTENCES["ru"])
    assert report["frozen"]["adapter_sha256"] == ADAPTER_SHA
    assert report["results"][0]["label"] == "voxcpm2 bake"
    printed = capsys.readouterr().out
    assert "cosine" not in printed.lower()


def test_a_meter_that_fails_after_a_bake_keeps_the_bake(world, tmp_path):
    settings, takes = world
    assert run_main(settings, confirm(takes), MeterClient(down=True), out=tmp_path / "out") == 0
    report = report_of(tmp_path / "out")
    assert report["likeness"] == {"error": "503 not offered"}
    assert (tmp_path / "out" / voice_bake_check.ADAPTER_FILE).read_bytes() == ADAPTER


SWEEP_CALLS = 24
BASELINE = {"cfg_value": 2.0, "inference_timesteps": 10}
GRID = [{"cfg_value": cfg_value, "inference_timesteps": steps}
        for cfg_value in (1.5, 2.0, 3.0) for steps in (10, 20)]
PILOT_SENTENCES = (
    "Мы нашли старую коробку с письмами на чердаке и читали их до самого вечера, пока не стемнело.",
    ("Когда самолёт наконец приземлился, все захлопали, а маленький мальчик у окна радостно "
     "засмеялся."),
)


class SweepClient(MeterClient):
    """Answers as the voice child does: a reference's `controls` come back in the reply."""

    def speak(self, text, language, voice, segments, voices, *, references=None, timeout=None):
        reply = super().speak(text, language, voice, segments, voices, references=references,
                              timeout=timeout)
        sent = references[voice].get("controls")
        reply["controls"] = {} if sent is None else {voice: sent}
        return reply

    def controls_sent(self) -> list[dict]:
        return [event[4][event[3]]["controls"] for event in self.speaks()]


@pytest.fixture
def v2_world(baked_world):
    """The voice as the user has it: the stored voice model is version 2."""
    settings, takes = baked_world
    conn = connect(settings.db_path)
    version = bake(conn, LocalStorage(settings.voices_dir), takes["recorded"].voice_id).version
    conn.close()
    assert version == voice_bake_check.SWEEP_MODEL_VERSION
    return settings, takes


def sweep(settings, client, out, *args):
    return run_main(settings, ["--sweep", *args], client, out=out)


def key_of(out: Path) -> dict:
    return json.loads((out / "key.json").read_text())


def test_the_sweep_sentence_is_new_and_one_synthesis_piece():
    text = voice_bake_check.SWEEP_TEXT
    seen = [kit.text for kit in RECORDING_KIT["ru"]] + list(PREVIEW_SENTENCES["ru"].values())
    seen += [known for _id, known in SENTENCES["ru"] + voice_bake_check.LONG_PASSAGES["ru"]]
    seen += PILOT_SENTENCES
    assert all(text not in other for other in seen)
    assert 90 <= len(text) <= 100
    assert split_text(text) == [text]


def test_the_sweep_speaks_the_whole_grid_in_arm_c_of_the_stored_model(v2_world, tmp_path):
    settings, _ = v2_world
    client = SweepClient()
    assert sweep(settings, client, tmp_path / "out") == 0
    assert "start" not in client.kinds()
    speaks = client.speaks()
    assert len(speaks) == SWEEP_CALLS
    assert {(event[1], event[2]) for event in speaks} == {(voice_bake_check.SWEEP_TEXT, "ru")}
    sha256 = hashlib.sha256(STORED_ADAPTER).hexdigest()
    for event in speaks:
        ref = event[4][event[3]]
        assert ref == {"language": "ru", "use_clip": False, "controls": ref["controls"],
                       "adapter": {"blob": STORED_ADAPTER, "sha256": sha256}}
    assert all(client.controls_sent().count(cell) == 4 for cell in GRID)


def test_each_repeat_is_one_shuffled_block_of_all_six_cells(v2_world, tmp_path):
    settings, _ = v2_world
    client = SweepClient()
    sweep(settings, client, tmp_path / "out")
    sent = client.controls_sent()
    blocks = [sent[start:start + len(GRID)] for start in range(0, SWEEP_CALLS, len(GRID))]
    assert all(sorted(block, key=str) == sorted(GRID, key=str) for block in blocks)
    assert len({str(block) for block in blocks}) > 1
    rows = report_of(tmp_path / "out")["calls"]
    assert [row["repeat"] for row in rows] == [1] * 6 + [2] * 6 + [3] * 6 + [4] * 6


def test_file_names_hold_neither_the_cell_nor_the_call_order(v2_world, tmp_path):
    settings, _ = v2_world
    out = tmp_path / "out"
    client = SweepClient()
    sweep(settings, client, out)
    rows = report_of(out)["calls"]
    names = [row["file"] for row in rows]
    assert sorted(names) == [f"S-{number:02d}.wav" for number in range(1, SWEEP_CALLS + 1)]
    assert names != sorted(names)
    assert all((out / name).exists() for name in names)
    sent = client.controls_sent()
    key = key_of(out)
    assert sorted(key) == sorted(names)
    assert [{name: key[row["file"]][name] for name in BASELINE} for row in rows] == sent


def test_the_sweep_report_freezes_its_own_inputs(v2_world, tmp_path):
    settings, _ = v2_world
    sweep(settings, SweepClient(), tmp_path / "out")
    report = report_of(tmp_path / "out")
    frozen = report["frozen"]
    sha256 = hashlib.sha256(STORED_ADAPTER).hexdigest()
    assert frozen["controls"] == [CONTROLS | cell for cell in GRID]
    assert frozen["texts"] == {voice_bake_check.SWEEP_TEXT_ID: voice_bake_check.SWEEP_TEXT}
    assert frozen["voice_model"] == {"model_key": "voxcpm2", "version": 2, "sha256": sha256}
    assert frozen["adapter_sha256"] == sha256
    assert frozen["base_revision"] == REVISION
    assert frozen["planned_calls"] == SWEEP_CALLS
    assert frozen["pass_rule"] == voice_bake_check.SWEEP_PASS_RULE
    assert frozen["mode"] == {"ru": {"C": "adapter alone"}}
    assert report["complete"] is True
    assert report["cells"] == [cell | {"valid_clips": 4} for cell in GRID]
    assert all(row["controls_echo"] == {name: row[name] for name in BASELINE}
               for row in report["calls"])


def test_the_sweep_console_prints_no_cell_and_no_score(v2_world, tmp_path, capsys):
    settings, _ = v2_world
    sweep(settings, SweepClient(), tmp_path / "out")
    printed = capsys.readouterr().out
    assert "cfg_value" not in printed and "inference_timesteps" not in printed
    assert "cosine" not in printed.lower() and "similarity" not in printed.lower()
    assert len(file_rows(printed)) == SWEEP_CALLS


def test_a_score_the_meter_refuses_leaves_the_sweep_complete(v2_world, tmp_path):
    settings, _ = v2_world
    client = SweepClient(refuse_arm="C")
    assert sweep(settings, client, tmp_path / "out") == 0
    report = report_of(tmp_path / "out")
    assert report["complete"] is True
    assert all("score_error" in row for row in report["calls"])


class WrongEcho(SweepClient):
    def __init__(self, at_call: int, echo: dict) -> None:
        super().__init__()
        self.at_call, self.echo = at_call, echo

    def speak(self, *args, **kwargs):
        reply = super().speak(*args, **kwargs)
        return reply | {"controls": self.echo} if len(self.speaks()) == self.at_call else reply


@pytest.mark.parametrize("echo", [
    pytest.param({}, id="missing"),
    pytest.param({voice_bake_check.TOKEN: {"cfg_value": 9.0, "inference_timesteps": 99}},
                 id="wrong"),
])
def test_a_missing_or_wrong_echo_stops_the_run_keeps_the_rows_and_exits_1(
        v2_world, tmp_path, capsys, echo):
    settings, _ = v2_world
    out = tmp_path / "out"
    client = WrongEcho(3, echo)
    assert sweep(settings, client, out) == 1
    assert len(client.speaks()) == 3
    report = report_of(out)
    rows = report["calls"]
    assert len(rows) == 3
    assert report["stopped_by_echo"] == rows[-1]["file"]
    assert "echoed" in rows[-1]["error"]
    assert (out / rows[-1]["file"]).exists()
    assert rows[-1]["wall_s"] is not None
    assert all("error" not in row for row in rows[:-1])
    assert sorted(key_of(out)) == sorted(row["file"] for row in rows)
    assert report["complete"] is False
    printed = capsys.readouterr().out
    assert f"stopped at {rows[-1]['file']}" in printed
    assert "9.0" not in printed


class FailsOnce(SweepClient):
    """The first call of `cell` fails; a reroll would show as a 25th call."""

    def __init__(self, cell: dict) -> None:
        super().__init__()
        self.cell, self.failed = cell, False

    def speak(self, text, language, voice, segments, voices, *, references=None, timeout=None):
        if references[voice]["controls"] == self.cell and not self.failed:
            self.failed = True
            self.events.append(("speak", text, language, voice, references))
            raise RuntimeError("the service fell over")
        return super().speak(text, language, voice, segments, voices, references=references,
                             timeout=timeout)


def test_an_incomplete_baseline_keeps_its_row_with_no_reroll_and_exits_1(v2_world, tmp_path):
    settings, _ = v2_world
    out = tmp_path / "out"
    client = FailsOnce(BASELINE)
    assert sweep(settings, client, out) == 1
    assert len(client.speaks()) == SWEEP_CALLS
    report = report_of(out)
    (failed,) = [row for row in report["calls"] if "error" in row]
    assert failed["error"] == "the service fell over"
    assert {name: failed[name] for name in BASELINE} == BASELINE
    assert "wall_s" in failed
    assert report["complete"] is False
    assert report["stopped_by_echo"] is None
    assert report["checks"][f"all {SWEEP_CALLS} clips valid"] is False
    assert [cell["valid_clips"] for cell in report["cells"]] == [
        3 if {name: cell[name] for name in BASELINE} == BASELINE else 4 for cell in report["cells"]]


def test_a_ctrl_c_keeps_what_the_sweep_made(v2_world, tmp_path):
    settings, _ = v2_world
    out = tmp_path / "out"

    class Stopped(SweepClient):
        def speak(self, *args, **kwargs):
            if len(self.speaks()) == 7:
                raise KeyboardInterrupt
            return super().speak(*args, **kwargs)

    assert sweep(settings, Stopped(), out) == voice_bake_check.EXIT_INTERRUPTED
    report = report_of(out)
    assert report["cancelled_by_user"] is True
    assert report["complete"] is False
    assert len(report["calls"]) == 7
    assert sorted(key_of(out)) == sorted(row["file"] for row in report["calls"])
    assert all((out / row["file"]).exists() for row in report["calls"])


def test_another_voice_model_version_is_refused_before_any_call(baked_world, tmp_path, capsys):
    settings, _ = baked_world
    client = SweepClient()
    assert sweep(settings, client, tmp_path / "out") == voice_bake_check.EXIT_USAGE
    assert "v2" in capsys.readouterr().err
    assert client.events == [] and client.embedded == []
    assert not (tmp_path / "out").exists()


def test_a_voice_model_that_cannot_speak_russian_is_refused_before_any_call(v2_world, tmp_path, capsys):
    settings, takes = v2_world
    conn = connect(settings.db_path)
    conn.execute("UPDATE voice_examples SET language = 'en' WHERE id = ?", (takes["ru"].id,))
    conn.close()
    client = SweepClient()
    assert sweep(settings, client, tmp_path / "out") == voice_bake_check.EXIT_USAGE
    assert "cannot speak ru" in capsys.readouterr().err
    assert client.events == [] and client.embedded == []


def test_a_take_that_says_the_sweep_sentence_is_refused_before_any_call(v2_world, tmp_path, capsys):
    settings, takes = v2_world
    client = SweepClient()
    said = voice_bake_check.SWEEP_TEXT.upper().replace(",", "")
    code = sweep(settings, client, tmp_path / "out", "--text", f"{takes['ru'].id}={said}")
    assert code == voice_bake_check.EXIT_USAGE
    assert f"take {takes['ru'].id}" in capsys.readouterr().err
    assert client.events == [] and client.embedded == []


def test_the_sweep_is_a_mode_of_its_own(v2_world, tmp_path):
    settings, _ = v2_world
    client = SweepClient()
    with pytest.raises(SystemExit) as stopped:
        measure(settings, client, tmp_path / "out", "--sweep")
    assert stopped.value.code == voice_bake_check.EXIT_USAGE
    assert client.events == []


def test_a_meter_that_cannot_run_stops_the_sweep_before_any_speech(v2_world, tmp_path):
    settings, _ = v2_world
    client = SweepClient(down=True)
    assert sweep(settings, client, tmp_path / "out") == 1
    assert client.speaks() == []


def test_the_sweep_leaves_the_database_and_the_store_unchanged(v2_world, tmp_path):
    settings, _ = v2_world
    db_before = settings.db_path.read_bytes()
    voices_before = snapshot(settings.voices_dir)
    sweep(settings, SweepClient(), tmp_path / "out")
    assert settings.db_path.read_bytes() == db_before
    assert snapshot(settings.voices_dir) == voices_before


HELDOUT_CALLS = 28
CANDIDATE = {"cfg_value": 3.0, "inference_timesteps": 20}
CELLS = [BASELINE, CANDIDATE]
HELD_LONG = "RU-HELD-LONG"
SHORT_IDS = ["RU-HELD-1", "RU-HELD-2", "RU-HELD-3", "RU-HELD-4"]
HELD_SHA = hashlib.sha256(STORED_ADAPTER).hexdigest()


@pytest.fixture
def heldout_world(v2_world, monkeypatch):
    """The stored voice model is the one the held-out check names."""
    monkeypatch.setattr(voice_bake_check, "HELDOUT_MODEL_SHA256", HELD_SHA)
    return v2_world


@pytest.fixture
def clocked_run(heldout_world, tmp_path):
    """Builds a held-out run on a clock the test moves: `main` takes no clock."""
    settings, _ = heldout_world
    conn = connect(settings.db_path)
    store = voice_bake_check.VoiceStore(conn, LocalStorage(settings.voices_dir), OWNER)
    voice = voice_bake_check.find_voice(conn, OWNER, "me")
    takes = voice_bake_check.takes_of(voice, {})

    def build(clock):
        return voice_bake_check.CheckRun(store, voice, takes, tmp_path / "out", random.Random(0),
                                         clock)

    yield build
    conn.close()


def heldout(settings, client, out, *args):
    return run_main(settings, ["--heldout", *args], client, out=out)


def pairs_sent(client) -> list[tuple[str, tuple]]:
    ids = {text: text_id for text_id, text in voice_bake_check.HELDOUT_TEXTS.items()}
    return [(ids[event[1]], tuple(event[4][event[3]]["controls"].values()))
            for event in client.speaks()]


def words_of(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold())


def ngrams(text: str, size: int) -> set[tuple[str, ...]]:
    words = words_of(text)
    return {tuple(words[start:start + size]) for start in range(len(words) - size + 1)}


def test_the_held_out_texts_are_new():
    kit = [kit.text for texts in RECORDING_KIT.values() for kit in texts]
    kit += [text for prompts in RECORDING_PROMPTS.values() for text in prompts.values()]
    seen = kit + [text for texts in PREVIEW_SENTENCES.values() for text in texts.values()]
    seen += [text for texts in SENTENCES.values() for _id, text in texts]
    seen += [text for texts in voice_bake_check.LONG_PASSAGES.values() for _id, text in texts]
    seen += [voice_bake_check.SWEEP_TEXT, *PILOT_SENTENCES]
    known = [" ".join(words_of(text)) for text in seen]
    for text in voice_bake_check.HELDOUT_TEXTS.values():
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            assert all(" ".join(words_of(sentence)) not in other for other in known)
        assert not any(ngrams(text, 4) & ngrams(other, 4) for other in kit)
        assert not any(ngrams(text, 3) & ngrams(other, 3) for other in PILOT_SENTENCES)


def test_the_long_text_crosses_a_piece_boundary_and_each_short_text_does_not():
    texts = voice_bake_check.HELDOUT_TEXTS
    assert list(texts) == [*SHORT_IDS, HELD_LONG]
    assert len(split_text(_normalise(texts[HELD_LONG]))) >= 2
    assert all(len(split_text(_normalise(texts[text_id]))) == 1 for text_id in SHORT_IDS)


def test_the_long_text_is_windowed_and_stays_out_of_measure():
    assert HELD_LONG in voice_bake_check.WINDOWED
    assert HELD_LONG not in [text_id for passages in voice_bake_check.LONG_PASSAGES.values()
                             for text_id, _ in passages]


def test_the_held_out_run_speaks_every_text_on_both_cells_in_arm_c(heldout_world, tmp_path):
    settings, _ = heldout_world
    client = SweepClient()
    assert heldout(settings, client, tmp_path / "out") == 0
    assert "start" not in client.kinds()
    speaks = client.speaks()
    assert len(speaks) == HELDOUT_CALLS
    assert {event[2] for event in speaks} == {"ru"}
    for event in speaks:
        ref = event[4][event[3]]
        assert ref == {"language": "ru", "use_clip": False, "controls": ref["controls"],
                       "adapter": {"blob": STORED_ADAPTER, "sha256": HELD_SHA}}
    counts = Counter(pairs_sent(client))
    assert counts == {(text_id, tuple(cell.values())): 2 if text_id == HELD_LONG else 3
                      for text_id in [*SHORT_IDS, HELD_LONG] for cell in CELLS}
    sent = client.controls_sent()
    assert sent.count(BASELINE) == sent.count(CANDIDATE) == HELDOUT_CALLS / 2


def test_each_repeat_is_one_shuffled_block_and_the_long_repeats_come_first(heldout_world, tmp_path):
    settings, _ = heldout_world
    client = SweepClient()
    heldout(settings, client, tmp_path / "out")
    rows = report_of(tmp_path / "out")["calls"]
    assert [row["repeat"] for row in rows] == [1] * 10 + [2] * 10 + [3] * 8
    pairs = pairs_sent(client)
    blocks = [pairs[:10], pairs[10:20], pairs[20:]]
    assert len(set(blocks[0])) == len(set(blocks[1])) == 10 and len(set(blocks[2])) == 8
    assert blocks[0] != blocks[1]
    assert [index for index, row in enumerate(rows) if row["id"] == HELD_LONG][-1] < 20
    assert all(text_id != HELD_LONG for text_id, _ in blocks[2])


def test_held_out_names_hold_neither_the_cell_the_text_nor_the_call_order(heldout_world, tmp_path):
    settings, _ = heldout_world
    out = tmp_path / "out"
    client = SweepClient()
    heldout(settings, client, out)
    rows = report_of(out)["calls"]
    names = [row["file"] for row in rows]
    assert sorted(names) == [f"H-{number:02d}.wav" for number in range(1, HELDOUT_CALLS + 1)]
    assert names != sorted(names)
    assert all((out / name).exists() for name in names)
    key = key_of(out)
    assert sorted(key) == sorted(names)
    assert [{name: key[row["file"]][name] for name in BASELINE} for row in rows] == (
        client.controls_sent())
    assert [key[row["file"]]["id"] for row in rows] == [text_id for text_id, _ in pairs_sent(client)]


class WatchesDisk(SweepClient):
    """Notes what the output folder holds when the first call arrives."""

    def __init__(self, out: Path) -> None:
        super().__init__()
        self.out = out
        self.first_call: dict | None = None

    def speak(self, *args, **kwargs):
        if self.first_call is None:
            self.first_call = {"files": sorted(path.name for path in self.out.iterdir()),
                               "key": key_of(self.out), "embedded": len(self.embedded),
                               "frozen": json.loads((self.out / "frozen.json").read_text())}
        return super().speak(*args, **kwargs)


def test_key_and_frozen_are_on_disk_before_the_first_speak(heldout_world, tmp_path):
    settings, _ = heldout_world
    out = tmp_path / "out"
    client = WatchesDisk(out)
    assert heldout(settings, client, out) == 0
    seen = client.first_call
    assert seen["files"] == ["frozen.json", "key.json"]
    assert len(seen["key"]) == HELDOUT_CALLS
    assert seen["embedded"] > 0
    assert seen["frozen"] == report_of(out)["frozen"]


def test_the_held_out_report_freezes_its_inputs_and_has_the_sweep_keys(heldout_world, tmp_path):
    settings, _ = heldout_world
    sweep(settings, SweepClient(), tmp_path / "sweep")
    heldout(settings, SweepClient(), tmp_path / "out")
    report = report_of(tmp_path / "out")
    assert set(report) == set(report_of(tmp_path / "sweep")) | {
        "started", "ended", "elapsed_s", "stopped_by_deadline", "timed_out"}
    frozen = report["frozen"]
    assert frozen == json.loads((tmp_path / "out" / "frozen.json").read_text())
    assert frozen["texts"] == voice_bake_check.HELDOUT_TEXTS
    assert frozen["controls"] == [CONTROLS | cell for cell in CELLS]
    assert frozen["voice_model"] == {"model_key": "voxcpm2", "version": 2, "sha256": HELD_SHA}
    assert frozen["adapter_sha256"] == HELD_SHA
    assert frozen["repeats"] == {"short": 3, "long": 2}
    assert frozen["planned_calls"] == HELDOUT_CALLS
    assert [row["file"] for row in frozen["planned_rows"]] == [
        row["file"] for row in report["calls"]]
    assert frozen["pass_rule"] == voice_bake_check.HELDOUT_PASS_RULE
    assert frozen["wall_budget"] == voice_bake_check.HELDOUT_WALL_BUDGET
    assert frozen["mode"] == {"ru": {"C": "adapter alone"}}
    assert report["complete"] is True
    assert report["stopped_by_deadline"] is False and report["timed_out"] is None
    assert report["started"] <= report["ended"] and report["elapsed_s"] >= 0
    assert report["cells"] == [BASELINE | {"valid_clips": 14}, CANDIDATE | {"valid_clips": 14}]
    assert all(row["windows"] for row in report["calls"] if row["id"] == HELD_LONG)


def test_the_held_out_console_prints_no_cell_and_no_score(heldout_world, tmp_path, capsys):
    settings, _ = heldout_world
    heldout(settings, SweepClient(), tmp_path / "out")
    printed = capsys.readouterr().out
    assert "cfg_value" not in printed and "inference_timesteps" not in printed
    assert "2.0/10" not in printed and "3.0/20" not in printed
    assert "cosine" not in printed.lower() and "similarity" not in printed.lower()
    assert len(file_rows(printed)) == HELDOUT_CALLS
    assert voice_bake_check.HELDOUT_LISTEN_FIRST in printed


def test_a_score_the_meter_refuses_leaves_the_held_out_run_complete(heldout_world, tmp_path):
    settings, _ = heldout_world
    assert heldout(settings, SweepClient(refuse_arm="C"), tmp_path / "out") == 0
    report = report_of(tmp_path / "out")
    assert report["complete"] is True
    assert all("score_error" in row for row in report["calls"] if row["id"] != HELD_LONG)


def assert_refused_before_any_call(client, out: Path) -> None:
    assert client.events == [] and client.embedded == []
    assert not (out / "key.json").exists() and not (out / "frozen.json").exists()


def test_another_voice_model_version_is_refused_by_the_held_out_run(baked_world, tmp_path, capsys):
    settings, _ = baked_world
    client = SweepClient()
    assert heldout(settings, client, tmp_path / "out") == voice_bake_check.EXIT_USAGE
    assert "v2" in capsys.readouterr().err
    assert_refused_before_any_call(client, tmp_path / "out")


def test_another_voice_model_hash_is_refused_by_the_held_out_run(v2_world, tmp_path, capsys):
    settings, _ = v2_world
    client = SweepClient()
    assert heldout(settings, client, tmp_path / "out") == voice_bake_check.EXIT_USAGE
    assert HELD_SHA[:12] in capsys.readouterr().err
    assert_refused_before_any_call(client, tmp_path / "out")


def test_a_voice_with_no_russian_model_is_refused_by_the_held_out_run(heldout_world, tmp_path, capsys):
    settings, takes = heldout_world
    conn = connect(settings.db_path)
    conn.execute("UPDATE voice_examples SET language = 'en' WHERE id = ?", (takes["ru"].id,))
    conn.close()
    client = SweepClient()
    assert heldout(settings, client, tmp_path / "out") == voice_bake_check.EXIT_USAGE
    assert "cannot speak ru" in capsys.readouterr().err
    assert_refused_before_any_call(client, tmp_path / "out")


@pytest.mark.parametrize("text_id", [*SHORT_IDS, HELD_LONG])
def test_a_take_holding_a_held_out_text_is_refused(heldout_world, tmp_path, capsys, text_id):
    settings, takes = heldout_world
    client = SweepClient()
    text = voice_bake_check.HELDOUT_TEXTS[text_id]
    said = f"сначала {text.upper().replace(',', '')} и потом ещё слова"
    code = heldout(settings, client, tmp_path / "out", "--text", f"{takes['ru'].id}={said}")
    assert code == voice_bake_check.EXIT_USAGE
    assert f"take {takes['ru'].id}" in capsys.readouterr().err
    assert_refused_before_any_call(client, tmp_path / "out")


def test_a_non_empty_output_folder_is_refused(heldout_world, tmp_path, capsys):
    settings, _ = heldout_world
    out = tmp_path / "out"
    out.mkdir()
    (out / "H-01.wav").write_bytes(b"an earlier clip")
    client = SweepClient()
    assert heldout(settings, client, out) == voice_bake_check.EXIT_USAGE
    assert "not empty" in capsys.readouterr().err
    assert_refused_before_any_call(client, out)
    assert (out / "H-01.wav").read_bytes() == b"an earlier clip"


def test_a_meter_that_cannot_run_stops_the_held_out_run_before_any_speech(heldout_world, tmp_path):
    settings, _ = heldout_world
    client = SweepClient(down=True)
    assert heldout(settings, client, tmp_path / "out") == 1
    assert client.speaks() == []
    assert not (tmp_path / "out" / "key.json").exists()


def test_a_wrong_echo_stops_the_held_out_run(heldout_world, tmp_path, capsys):
    settings, _ = heldout_world
    out = tmp_path / "out"
    client = WrongEcho(3, {})
    assert heldout(settings, client, out) == 1
    assert len(client.speaks()) == 3
    report = report_of(out)
    rows = report["calls"]
    assert len(rows) == 3
    assert report["stopped_by_echo"] == rows[-1]["file"]
    assert "echoed" in rows[-1]["error"]
    assert (out / rows[-1]["file"]).exists()
    assert report["stopped_by_deadline"] is False and report["complete"] is False
    assert len(key_of(out)) == HELDOUT_CALLS
    assert f"stopped at {rows[-1]['file']}" in capsys.readouterr().out


def test_a_failed_call_with_time_left_is_a_row_and_the_run_goes_on(heldout_world, tmp_path):
    settings, _ = heldout_world
    out = tmp_path / "out"
    client = FailsOnce(BASELINE)
    assert heldout(settings, client, out) == 1
    assert len(client.speaks()) == HELDOUT_CALLS
    report = report_of(out)
    (failed,) = [row for row in report["calls"] if "error" in row]
    assert "timed_out" not in failed
    assert report["timed_out"] is None and report["stopped_by_deadline"] is False
    assert report["complete"] is False


class SlowClient(SweepClient):
    """Every call takes `seconds` on a fake clock; call `fails_at` then raises."""

    def __init__(self, seconds: float, *, fails_at: int | None = None) -> None:
        super().__init__()
        self.seconds, self.fails_at = seconds, fails_at
        self.now = 0.0
        self.timeouts: list[float] = []

    def clock(self) -> float:
        return self.now

    def speak(self, *args, timeout=None, **kwargs):
        self.timeouts.append(timeout)
        self.now += self.seconds
        if len(self.timeouts) == self.fails_at:
            raise RuntimeError("the service fell over")
        return super().speak(*args, timeout=timeout, **kwargs)


def test_no_call_starts_after_the_budget_and_each_gets_the_time_left(clocked_run, tmp_path):
    client = SlowClient(600.0)
    report, code = voice_bake_check.run_heldout(client, clocked_run(client.clock))
    assert code == 1
    assert client.timeouts == [1500.0, 900.0, 300.0]
    assert report["stopped_by_deadline"] is True
    assert report["timed_out"] is None
    assert report["elapsed_s"] == 1800.0
    assert report["complete"] is False
    assert len(report["calls"]) == 3
    assert len(key_of(tmp_path / "out")) == HELDOUT_CALLS
    assert all((tmp_path / "out" / row["file"]).exists() for row in report["calls"])
    printed = voice_bake_check.format_heldout_summary(report, tmp_path / "out")
    assert "wall budget" in printed


def test_a_failure_past_the_budget_gets_no_replacement_call(clocked_run):
    client = SlowClient(600.0, fails_at=3)
    report, code = voice_bake_check.run_heldout(client, clocked_run(client.clock))
    assert code == 1
    assert len(client.timeouts) == 3
    assert report["stopped_by_deadline"] is True
    assert report["calls"][-1]["error"] == "the service fell over"
    assert "timed_out" not in report["calls"][-1]


def test_no_budget_left_at_the_start_makes_no_call(heldout_world, tmp_path, monkeypatch):
    settings, _ = heldout_world
    monkeypatch.setattr(voice_bake_check, "HELDOUT_WALL_SECONDS", 0)
    client = SweepClient()
    assert heldout(settings, client, tmp_path / "out") == 1
    assert client.speaks() == []
    report = report_of(tmp_path / "out")
    assert report["calls"] == [] and report["stopped_by_deadline"] is True


class HoldsCall(SweepClient):
    """Holds one call until the test lets go, as a service still computing that clip."""

    def __init__(self, at_call: int) -> None:
        super().__init__()
        self.at_call = at_call
        self.go_on = threading.Event()
        self.timeouts: list[float] = []
        self.worker: threading.Thread | None = None

    def speak(self, *args, timeout=None, **kwargs):
        self.timeouts.append(timeout)
        if len(self.timeouts) == self.at_call:
            self.worker = threading.current_thread()
            self.go_on.wait(5.0)
        return super().speak(*args, timeout=timeout, **kwargs)


def test_a_call_held_past_the_deadline_ends_the_run_and_its_reply_is_ignored(
        heldout_world, tmp_path, monkeypatch, capsys):
    settings, _ = heldout_world
    out = tmp_path / "out"
    budget = 0.25
    monkeypatch.setattr(voice_bake_check, "HELDOUT_WALL_SECONDS", budget)
    client = HoldsCall(2)
    assert heldout(settings, client, out) == 1
    assert client.worker.is_alive()
    report_bytes = (out / "report.json").read_bytes()
    report = report_of(out)
    first, held = report["calls"]
    assert report["stopped_by_deadline"] is True
    assert report["timed_out"] == held["file"] and held["timed_out"] is True
    assert budget <= report["elapsed_s"] < 5.0
    assert client.timeouts[0] <= budget and client.timeouts[1] < client.timeouts[0]
    assert (out / first["file"]).exists() and "error" not in first
    assert len(key_of(out)) == HELDOUT_CALLS
    printed = capsys.readouterr().out
    assert f"stopped at {held['file']}" in printed and "wall budget" in printed

    client.go_on.set()
    client.worker.join(5.0)
    assert not client.worker.is_alive()
    assert len(client.timeouts) == 2
    assert not (out / held["file"]).exists()
    assert (out / "report.json").read_bytes() == report_bytes


class TimesOut(SweepClient):
    def __init__(self, at_call: int, timeout: Exception) -> None:
        super().__init__()
        self.at_call, self.timeout = at_call, timeout

    def speak(self, text, language, voice, segments, voices, *, references=None, timeout=None):
        if len(self.speaks()) + 1 == self.at_call:
            self.events.append(("speak", text, language, voice, references))
            raise self.timeout
        return super().speak(text, language, voice, segments, voices, references=references,
                             timeout=timeout)


@pytest.mark.parametrize("timeout", [
    pytest.param(httpx.ReadTimeout("timed out"), id="httpx"),
    pytest.param(TimeoutError(), id="builtin"),
])
def test_any_timeout_ends_the_run_with_time_left(heldout_world, tmp_path, timeout):
    settings, _ = heldout_world
    out = tmp_path / "out"
    client = TimesOut(3, timeout)
    assert heldout(settings, client, out) == 1
    assert len(client.speaks()) == 3
    report = report_of(out)
    rows = report["calls"]
    assert len(rows) == 3
    assert rows[-1]["timed_out"] is True
    assert report["timed_out"] == rows[-1]["file"]
    assert report["stopped_by_deadline"] is False
    assert not (out / rows[-1]["file"]).exists()
    assert all((out / row["file"]).exists() for row in rows[:-1])


def test_a_ctrl_c_keeps_what_the_held_out_run_made(heldout_world, tmp_path):
    settings, _ = heldout_world
    out = tmp_path / "out"

    class Stopped(SweepClient):
        def speak(self, *args, **kwargs):
            if len(self.speaks()) == 7:
                raise KeyboardInterrupt
            return super().speak(*args, **kwargs)

    assert heldout(settings, Stopped(), out) == voice_bake_check.EXIT_INTERRUPTED
    report = report_of(out)
    assert report["cancelled_by_user"] is True
    assert report["complete"] is False
    assert len(report["calls"]) == 7
    assert all((out / row["file"]).exists() for row in report["calls"])
    assert len(key_of(out)) == HELDOUT_CALLS
    assert (out / "frozen.json").exists()


def test_the_held_out_run_is_a_mode_of_its_own(heldout_world, tmp_path):
    settings, _ = heldout_world
    client = SweepClient()
    with pytest.raises(SystemExit) as stopped:
        sweep(settings, client, tmp_path / "out", "--heldout")
    assert stopped.value.code == voice_bake_check.EXIT_USAGE
    assert client.events == []


def test_the_held_out_run_leaves_the_database_and_the_store_unchanged(heldout_world, tmp_path):
    settings, _ = heldout_world
    db_before = settings.db_path.read_bytes()
    voices_before = snapshot(settings.voices_dir)
    heldout(settings, SweepClient(), tmp_path / "out")
    assert settings.db_path.read_bytes() == db_before
    assert snapshot(settings.voices_dir) == voices_before


def test_the_sweep_sends_no_budget_and_speaks_on_the_main_thread(v2_world, tmp_path):
    settings, _ = v2_world
    seen = []

    class Watched(SweepClient):
        def speak(self, *args, timeout=None, **kwargs):
            seen.append((timeout, threading.current_thread()))
            return super().speak(*args, timeout=timeout, **kwargs)

    assert sweep(settings, Watched(), tmp_path / "out") == 0
    assert set(seen) == {(None, threading.main_thread())}


def test_a_timeout_in_a_sweep_stays_a_failed_row_and_the_sweep_goes_on(v2_world, tmp_path):
    settings, _ = v2_world
    client = TimesOut(3, httpx.ReadTimeout("timed out"))
    assert sweep(settings, client, tmp_path / "out") == 1
    assert len(client.speaks()) == SWEEP_CALLS
    rows = report_of(tmp_path / "out")["calls"]
    assert all("timed_out" not in row for row in rows)
    assert rows[2]["error"] == "timed out"


def test_measure_sends_no_controls(baked_world, tmp_path):
    settings, _ = baked_world
    client = MeterClient()
    measure(settings, client, tmp_path / "out", "--repeats", "1")
    assert all("controls" not in event[4][event[3]] for event in client.speaks())
    assert all("controls_echo" not in row for row in report_of(tmp_path / "out")["calls"])


LONGER_CALLS = 28
LONGER_STEPS = 1000
ROLE_OF = {ADAPTER_SHA: "candidate", HELD_SHA: "baseline"}


def longer_status(state="running", *, step, dataset="", seconds_per_step=37.7,
                  min_free_mb=1700.0) -> dict:
    return status(state, step=step, stage="done" if state == "done" else "training",
                  min_free_mb=min_free_mb) | {
        "steps": LONGER_STEPS, "seconds_per_step": seconds_per_step,
        "config": {"steps": LONGER_STEPS}, "dataset_sha256": dataset}


class LongerClient(SweepClient):
    """A 1000-step bake that ends `done`, then the speaking of `SweepClient`."""

    def __init__(self, dataset: str, statuses: list[dict] | None = None) -> None:
        super().__init__()
        self.statuses = statuses or [longer_status(step=500, dataset=dataset),
                                     longer_status("done", step=LONGER_STEPS, dataset=dataset)]
        self.cancel_requests = 0

    def bake_cancel(self, bake_id):
        self.cancel_requests += 1
        return super().bake_cancel(bake_id)


@pytest.fixture
def longer_world(heldout_world):
    """The held-out world and the dataset hash of its stored v2."""
    settings, takes = heldout_world
    conn = connect(settings.db_path)
    (dataset,) = conn.execute("SELECT dataset_hash FROM voice_models").fetchone()
    conn.close()
    return settings, takes, dataset


def longer(settings, client, out, *args, sleep=lambda _seconds: None):
    return run_main(settings, ["--longer-bake", *args], client, out=out, sleep=sleep)


def roles_sent(client) -> list[tuple[str, str]]:
    ids = {text: text_id for text_id, text in voice_bake_check.HELDOUT_TEXTS.items()}
    return [(ids[event[1]], ROLE_OF[event[4][event[3]]["adapter"]["sha256"]])
            for event in client.speaks()]


def ticking(step_seconds: float):
    """`(clock, sleep)`: each poll's sleep moves the clock on by `step_seconds`."""
    now = [0.0]

    def sleep(_seconds):
        now[0] += step_seconds

    return (lambda: now[0]), sleep


@pytest.mark.parametrize(("statuses", "step_seconds", "trigger", "elapsed"), [
    pytest.param([status(step=2), status(step=3), status("cancelled", step=3)], 21600.0,
                 "12 h passed", 43200.0, id="12h"),
    pytest.param([status(step=2, min_free_mb=1023.0), status("cancelled", step=2)], 2.0,
                 "free memory under 1024 MiB", 2.0, id="low-memory"),
    pytest.param([status(step=51) | {"seconds_per_step": 45.5}, status("cancelled", step=51)],
                 2.0, "over 45 s per step", 2.0, id="slow-step"),
    pytest.param([status("failed", step=7, error="boom")], 2.0, "failed", 2.0, id="failure"),
])
def test_each_watchdog_trigger_requests_a_cancel_and_is_recorded(statuses, step_seconds, trigger,
                                                                 elapsed):
    client = FakeClient(statuses)
    clock, sleep = ticking(step_seconds)
    result = voice_bake_check.run_bake(client, [], {}, None, sleep=sleep, watchdog=True,
                                       clock=clock)
    assert client.kinds().count("cancel") == 1
    assert result.stop == {"trigger": trigger, "elapsed_s": elapsed}
    assert result.adapter is None and "adapter" not in client.kinds()
    assert client.kinds()[-1] == "delete"


@pytest.mark.parametrize("step", [10, 50])
def test_a_slow_step_up_to_step_50_requests_no_cancel(step):
    client = FakeClient([status(step=step) | {"seconds_per_step": 300.0},
                         status("done", step=4, stage="done")])
    clock, sleep = ticking(2.0)
    result = voice_bake_check.run_bake(client, [], {}, None, sleep=sleep, watchdog=True,
                                       clock=clock)
    assert "cancel" not in client.kinds()
    assert result.stop is None and result.adapter == ADAPTER


def test_a_done_bake_after_a_watchdog_cancel_keeps_no_adapter():
    client = FakeClient([status(step=2, min_free_mb=10.0), status("done", step=4, stage="done")])
    result = voice_bake_check.run_bake(client, [], {}, None, sleep=lambda _s: None,
                                       watchdog=True)
    assert result.stop["trigger"] == "free memory under 1024 MiB"
    assert result.adapter is None


def test_without_the_watchdog_no_trigger_cancels():
    client = FakeClient([status(step=60, min_free_mb=10.0) | {"seconds_per_step": 300.0},
                         status("done", step=4, stage="done")])
    clock, sleep = ticking(LONGER_STEPS * 3600.0)
    result = voice_bake_check.run_bake(client, [], {}, None, sleep=sleep, clock=clock)
    assert "cancel" not in client.kinds()
    assert result.stop is None and result.adapter == ADAPTER


def test_the_bake_check_runs_with_no_watchdog(world, tmp_path):
    settings, takes = world
    client = FakeClient([status(step=60, min_free_mb=10.0) | {"seconds_per_step": 300.0},
                         status("done", step=4, stage="done")])
    run_main(settings, confirm(takes), client, out=tmp_path / "out")
    assert "cancel" not in client.kinds()
    assert (tmp_path / "out" / voice_bake_check.ADAPTER_FILE).read_bytes() == ADAPTER


def test_a_ctrl_c_still_cancels_under_the_watchdog_and_is_recorded():
    client = FakeClient([status("cancelled", step=1)])
    clock, _ = ticking(0.0)
    result = voice_bake_check.run_bake(client, [], {}, None, sleep=ctrl_c_once(), watchdog=True,
                                       clock=clock)
    assert result.interrupted is True
    assert result.stop == {"trigger": "Ctrl-C", "elapsed_s": 0.0}
    assert client.kinds().count("cancel") == 1


def test_the_longer_bake_sends_v2s_takes_and_only_the_steps(longer_world, tmp_path):
    settings, takes, dataset = longer_world
    client = LongerClient(dataset)
    assert longer(settings, client, tmp_path / "out") == 0
    (start,) = [event for event in client.events if event[0] == "start"]
    _, clips, _, config = start
    assert config == {"steps": LONGER_STEPS}
    learned = sorted(take.id for name, take in takes.items() if name != "deleting")
    assert [take["example_id"] for take in report_of(tmp_path / "out")["frozen"]["takes"]] == learned
    assert len(clips) == len(learned)


class WatchesBakeStart(LongerClient):
    def __init__(self, dataset: str, out: Path) -> None:
        super().__init__(dataset)
        self.out = out
        self.at_start: dict | None = None

    def bake_start(self, clips, *, tag, config=None):
        self.at_start = {"files": sorted(path.name for path in self.out.iterdir()),
                         "frozen": json.loads((self.out / "frozen.json").read_text())}
        return super().bake_start(clips, tag=tag, config=config)


def test_frozen_json_is_on_disk_before_the_bake_starts(longer_world, tmp_path):
    settings, _, dataset = longer_world
    out = tmp_path / "out"
    client = WatchesBakeStart(dataset, out)
    assert longer(settings, client, out) == 0
    seen = client.at_start
    assert seen["files"] == ["frozen.json"]
    frozen = seen["frozen"]
    assert frozen["steps"] == LONGER_STEPS
    assert frozen["config_sent"] == {"steps": LONGER_STEPS}
    assert frozen["recipe"] == {"steps": LONGER_STEPS}
    assert frozen["dataset_sha256"] == dataset
    assert frozen["base_revision"] == REVISION
    assert frozen["voice_model"]["sha256"] == HELD_SHA and frozen["voice_model"]["version"] == 2
    assert all(len(take["fingerprint"]) == 64 for take in frozen["takes"])
    assert frozen["takes_left_out"] == []
    assert frozen["watchdog"] == {"max_seconds": 43200, "min_free_mb": 1024,
                                  "max_seconds_per_step": 45.0, "slow_after_step": 50}
    assert "planned_rows" not in frozen


class WatchesLongerDisk(LongerClient):
    def __init__(self, dataset: str, out: Path) -> None:
        super().__init__(dataset)
        self.out = out
        self.first_call: dict | None = None

    def speak(self, *args, **kwargs):
        if self.first_call is None:
            self.first_call = {"files": sorted(path.name for path in self.out.iterdir()),
                               "key": key_of(self.out),
                               "frozen": json.loads((self.out / "frozen.json").read_text())}
        return super().speak(*args, **kwargs)


def test_key_and_the_full_frozen_json_are_on_disk_before_the_first_speak(longer_world, tmp_path):
    settings, _, dataset = longer_world
    out = tmp_path / "out"
    client = WatchesLongerDisk(dataset, out)
    assert longer(settings, client, out) == 0
    seen = client.first_call
    assert seen["files"] == ["adapter.safetensors", "bake.json", "frozen.json", "key.json"]
    assert len(seen["key"]) == LONGER_CALLS
    frozen = seen["frozen"]
    assert frozen == report_of(out)["frozen"]
    assert frozen["adapter_sha256"] == {"candidate": ADAPTER_SHA, "baseline": HELD_SHA}
    assert frozen["planned_calls"] == LONGER_CALLS and len(frozen["planned_rows"]) == LONGER_CALLS
    assert frozen["pass_rule"] == voice_bake_check.LONGER_BAKE_PASS_RULE
    assert frozen["wall_budget"] == voice_bake_check.HELDOUT_WALL_BUDGET
    assert frozen["texts"] == voice_bake_check.HELDOUT_TEXTS
    assert frozen["controls"] == CONTROLS


def test_the_longer_bake_speaks_28_calls_in_arm_c_with_no_controls(longer_world, tmp_path):
    settings, _, dataset = longer_world
    out = tmp_path / "out"
    client = LongerClient(dataset)
    assert longer(settings, client, out) == 0
    speaks = client.speaks()
    assert len(speaks) == LONGER_CALLS
    blobs = {ADAPTER_SHA: ADAPTER, HELD_SHA: STORED_ADAPTER}
    for event in speaks:
        ref = event[4][event[3]]
        sha256 = ref["adapter"]["sha256"]
        assert ref == {"language": "ru", "use_clip": False,
                       "adapter": {"blob": blobs[sha256], "sha256": sha256}}
    assert Counter(roles_sent(client)) == {
        (text_id, role): 2 if text_id == HELD_LONG else 3
        for text_id in [*SHORT_IDS, HELD_LONG] for role in ("candidate", "baseline")}
    rows = report_of(out)["calls"]
    assert [row["repeat"] for row in rows] == [1] * 10 + [2] * 10 + [3] * 8
    assert [index for index, row in enumerate(rows) if row["id"] == HELD_LONG][-1] < 20
    names = [row["file"] for row in rows]
    assert sorted(names) == [f"L-{number:02d}.wav" for number in range(1, LONGER_CALLS + 1)]
    assert names != sorted(names)
    key = key_of(out)
    assert [(key[name]["id"], key[name]["role"]) for name in names] == roles_sent(client)


def test_the_longer_report_holds_the_bake_and_the_speaking(longer_world, tmp_path):
    settings, _, dataset = longer_world
    out = tmp_path / "out"
    assert longer(settings, LongerClient(dataset), out) == 0
    report = report_of(out)
    assert report["bake"]["state"] == "done"
    assert report["steps_done"] == LONGER_STEPS
    assert report["seconds_per_step"] == 37.7
    assert report["loss_first"] == 2.5 and report["loss_last"] == 1.25
    assert report["peak_device_bytes"] == 123456
    assert report["min_free_mb"] == 1700.0 and report["free_mb_before"] == 9000.0
    assert report["stop"] is None
    assert report["attached"] is False and report["bake_id"] == "bake-1"
    assert report["poll_failures"] == 0
    assert report["measured_thresholds"] == {"seconds_per_step <= 6.0": False}
    assert report["complete"] is True and all(report["checks"].values())
    assert report["roles"] == [{"role": "candidate", "valid_clips": 14},
                               {"role": "baseline", "valid_clips": 14}]
    assert report["stopped_by_deadline"] is False and report["timed_out"] is None
    assert report["adapter"]["file"] == voice_bake_check.ADAPTER_FILE
    assert (out / voice_bake_check.ADAPTER_FILE).read_bytes() == ADAPTER


def test_a_bake_of_other_takes_or_recipe_fails_its_checks(longer_world, tmp_path):
    settings, _, dataset = longer_world
    statuses = [longer_status("done", step=LONGER_STEPS, dataset="other")
                | {"config": {"steps": LONGER_STEPS, "learning_rate": 0.5}}]
    assert longer(settings, LongerClient(dataset, statuses), tmp_path / "out") == 1
    checks = report_of(tmp_path / "out")["checks"]
    assert checks["the bake trained the takes of v2"] is False
    assert checks["the recipe is v2's with 1000 steps"] is False


@pytest.mark.parametrize("statuses", [
    pytest.param([longer_status(step=60, seconds_per_step=50.0),
                  longer_status("cancelled", step=61)], id="watchdog"),
    pytest.param([longer_status("done", step=999)], id="fewer-steps"),
    pytest.param([longer_status("failed", step=3)], id="failed"),
])
def test_a_bake_that_did_not_end_done_with_1000_steps_speaks_nothing(longer_world, tmp_path,
                                                                      statuses, capsys):
    settings, _, dataset = longer_world
    out = tmp_path / "out"
    client = LongerClient(dataset, statuses)
    assert longer(settings, client, out) == 1
    assert client.speaks() == [] and client.embedded == []
    assert not (out / "key.json").exists()
    report = report_of(out)
    assert report["calls"] == [] and report["complete"] is False
    assert report["checks"] == {"bake done with all 1000 steps": False}
    assert "nothing was spoken" in capsys.readouterr().out


def test_a_watchdog_stop_is_in_the_report_and_the_console(longer_world, tmp_path, capsys):
    settings, _, dataset = longer_world
    out = tmp_path / "out"
    client = LongerClient(dataset, [longer_status(step=60, seconds_per_step=50.0),
                                    longer_status("cancelled", step=61)])
    longer(settings, client, out)
    report = report_of(out)
    assert client.cancel_requests == 1
    assert report["stop"]["trigger"] == "over 45 s per step"
    assert report["stop"]["elapsed_s"] >= 0
    assert report["steps_done"] == 61
    assert not (out / voice_bake_check.ADAPTER_FILE).exists()
    assert "over 45 s per step" in capsys.readouterr().out


def test_a_ctrl_c_during_the_longer_bake_speaks_nothing_and_exits_130(longer_world, tmp_path):
    settings, _, dataset = longer_world
    out = tmp_path / "out"
    client = LongerClient(dataset, [longer_status("cancelled", step=1)])
    assert longer(settings, client, out, sleep=ctrl_c_once()) == voice_bake_check.EXIT_INTERRUPTED
    assert client.speaks() == []
    report = report_of(out)
    assert report["stop"]["trigger"] == "Ctrl-C" and report["cancelled_by_user"] is True


def assert_no_bake_call(client, out: Path) -> None:
    assert client.events == [] and client.embedded == []
    assert not (out / "frozen.json").exists()


def test_another_voice_model_version_is_refused_by_the_longer_bake(baked_world, tmp_path, capsys):
    settings, _ = baked_world
    client = LongerClient("")
    assert longer(settings, client, tmp_path / "out") == voice_bake_check.EXIT_USAGE
    assert "v2" in capsys.readouterr().err
    assert_no_bake_call(client, tmp_path / "out")


def test_another_voice_model_hash_is_refused_by_the_longer_bake(v2_world, tmp_path, capsys):
    settings, _ = v2_world
    client = LongerClient("")
    assert longer(settings, client, tmp_path / "out") == voice_bake_check.EXIT_USAGE
    assert HELD_SHA[:12] in capsys.readouterr().err
    assert_no_bake_call(client, tmp_path / "out")


def test_a_voice_with_no_russian_model_is_refused_by_the_longer_bake(longer_world, tmp_path, capsys):
    settings, takes, dataset = longer_world
    conn = connect(settings.db_path)
    conn.execute("UPDATE voice_model_takes SET language = 'en' WHERE example_id = ?",
                 (takes["ru"].id,))
    conn.close()
    client = LongerClient(dataset)
    assert longer(settings, client, tmp_path / "out") == voice_bake_check.EXIT_USAGE
    assert "cannot speak ru" in capsys.readouterr().err
    assert_no_bake_call(client, tmp_path / "out")


def test_a_take_of_v2_being_deleted_is_refused(longer_world, tmp_path, capsys):
    settings, takes, dataset = longer_world
    conn = connect(settings.db_path)
    request_example_deletion(conn, OWNER, takes["uploaded"].voice_id, takes["uploaded"].id)
    conn.close()
    client = LongerClient(dataset)
    assert longer(settings, client, tmp_path / "out") == voice_bake_check.EXIT_USAGE
    assert f"take {takes['uploaded'].id}" in capsys.readouterr().err
    assert_no_bake_call(client, tmp_path / "out")


def test_a_take_with_other_words_than_v2_learned_is_refused(longer_world, tmp_path, capsys):
    settings, takes, dataset = longer_world
    client = LongerClient(dataset)
    code = longer(settings, client, tmp_path / "out", "--text", f"{takes['ru'].id}=другие слова")
    assert code == voice_bake_check.EXIT_USAGE
    assert f"take {takes['ru'].id} differs" in capsys.readouterr().err
    assert_no_bake_call(client, tmp_path / "out")


def test_a_take_v2_did_not_learn_is_left_out_and_named(longer_world, tmp_path):
    settings, takes, dataset = longer_world
    conn = connect(settings.db_path)
    new = add_example(conn, LocalStorage(settings.voices_dir), OWNER, takes["ru"].voice_id,
                      language="en", source="recorded", data=tone(6.0, 310.0))
    conn.close()
    client = LongerClient(dataset)
    assert longer(settings, client, tmp_path / "out") == 0
    report = report_of(tmp_path / "out")
    assert report["frozen"]["takes_left_out"] == [new.id]
    assert new.id not in report["takes"]["ids"]
    (start,) = [event for event in client.events if event[0] == "start"]
    assert len(start[1]) == len(report["takes"]["ids"])


def test_a_non_empty_output_folder_is_refused_by_the_longer_bake(longer_world, tmp_path, capsys):
    settings, _, dataset = longer_world
    out = tmp_path / "out"
    out.mkdir()
    (out / "L-01.wav").write_bytes(b"an earlier clip")
    client = LongerClient(dataset)
    assert longer(settings, client, out) == voice_bake_check.EXIT_USAGE
    assert "not empty" in capsys.readouterr().err
    assert_no_bake_call(client, out)


TIMEOUT = httpx.ReadTimeout("timed out")
DROPPED = httpx.ConnectError("connection refused")
TWELVE_HOURS = 43200.0


def test_a_failed_poll_is_retried_on_the_same_bake_and_the_bake_completes(capsys):
    client = FakeClient([TIMEOUT, DROPPED, status(step=3), status("done", step=4, stage="done")])
    clock, sleep = ticking(2.0)
    result = voice_bake_check.run_bake(client, [], {}, None, sleep=sleep, watchdog=True,
                                       clock=clock)
    assert result.adapter == ADAPTER and result.stop is None
    assert result.poll_failures == 2
    assert {event[1] for event in client.events if event[0] == "status"} == {"bake-1"}
    assert client.kinds().count("start") == 1 and "cancel" not in client.kinds()
    assert client.kinds()[-1] == "delete"
    assert "timed out" in capsys.readouterr().out


def test_no_poll_asks_for_the_models_state():
    client = FakeClient([status(step=2, min_free_mb=8000.0), status(step=3, min_free_mb=6000.0),
                         status("done", step=4, stage="done", min_free_mb=6500.0)])
    result = voice_bake_check.run_bake(client, [], {}, None, sleep=lambda _s: None,
                                       watchdog=True)
    assert client.kinds()[:2] == ["state", "start"]
    assert client.kinds().count("state") == 1
    assert result.free_mb_before == 9000.0 and result.min_free_mb == 6500.0


def test_the_deadline_fires_across_failed_polls_with_no_status_request():
    client = FakeClient([TIMEOUT, status("cancelled", step=3)])
    clock, sleep = ticking(TWELVE_HOURS / 2)
    result = voice_bake_check.run_bake(client, [], {}, None, sleep=sleep, watchdog=True,
                                       clock=clock)
    assert client.kinds() == ["state", "start", "status", "cancel", "status", "delete"]
    assert result.stop == {"trigger": "12 h passed", "elapsed_s": TWELVE_HOURS}
    assert result.poll_failures == 1


def test_the_poll_timeout_shrinks_to_the_time_left_before_12_hours():
    client = FakeClient([status(step=3), status("cancelled", step=3)])
    clock, sleep = ticking(TWELVE_HOURS - 25.0)
    voice_bake_check.run_bake(client, [], {}, None, sleep=sleep, watchdog=True, clock=clock)
    assert client.status_timeouts == [25.0, 60.0]


def test_a_poll_waits_60_seconds_far_from_the_deadline():
    client = FakeClient()
    clock, sleep = ticking(2.0)
    voice_bake_check.run_bake(client, [], {}, None, sleep=sleep, watchdog=True, clock=clock)
    assert client.status_timeouts == [60.0, 60.0]


def test_the_attach_clock_counts_the_time_before_the_attach():
    client = FakeClient([status("cancelled", step=3)])
    clock, sleep = ticking(10.0)
    attached = voice_bake_check.Attach(status(step=900) | {"elapsed_s": TWELVE_HOURS - 10.0}, 0.0)
    result = voice_bake_check.run_bake(client, [], {}, None, sleep=sleep, watchdog=True,
                                       clock=clock, attach=attached)
    assert client.kinds() == ["cancel", "status", "delete"]
    assert result.stop == {"trigger": "12 h passed", "elapsed_s": TWELVE_HOURS}
    assert result.attached is True


def test_an_attach_past_12_hours_cancels_before_any_wait():
    client = FakeClient([status("cancelled", step=3)])
    slept = []
    attached = voice_bake_check.Attach(status(step=900) | {"elapsed_s": TWELVE_HOURS}, 0.0)
    voice_bake_check.run_bake(client, [], {}, None, sleep=slept.append, watchdog=True,
                              clock=lambda: 0.0, attach=attached)
    assert client.kinds()[0] == "cancel" and slept == [voice_bake_check.POLL_SECONDS]


def test_a_failed_cancel_is_retried_until_the_bake_ends():
    client = FakeClient([status(step=2, min_free_mb=10.0), status(step=2),
                         status("cancelled", step=2)], cancel_failures=2)
    result = voice_bake_check.run_bake(client, [], {}, None, sleep=lambda _s: None,
                                       watchdog=True)
    assert client.kinds() == ["state", "start", "status", "cancel", "cancel", "status", "cancel",
                              "status", "delete"]
    assert result.status["state"] == "cancelled"
    assert result.stop["trigger"] == "free memory under 1024 MiB"


def test_a_second_ctrl_c_leaves_and_deletes_no_running_bake(capsys):
    client = FakeClient([status(step=2)])

    def sleep(_seconds):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        voice_bake_check.run_bake(client, [], {}, None, sleep=sleep, watchdog=True)
    assert client.kinds() == ["state", "start", "cancel"]
    assert "bake-1 still runs" in capsys.readouterr().err


def test_a_run_that_fails_while_the_bake_runs_deletes_nothing():
    client = FakeClient([httpx.HTTPStatusError(
        "boom", request=httpx.Request("GET", "http://x"), response=httpx.Response(500))])
    with pytest.raises(httpx.HTTPStatusError):
        voice_bake_check.run_bake(client, [], {}, None, sleep=lambda _s: None, watchdog=True)
    assert "delete" not in client.kinds()


class WatchesBakeId(LongerClient):
    def __init__(self, dataset: str, out: Path) -> None:
        super().__init__(dataset)
        self.out = out
        self.at_first_poll: dict | None = None

    def bake_status(self, bake_id, *, timeout=None):
        if self.at_first_poll is None:
            self.at_first_poll = json.loads((self.out / "bake.json").read_text())
        return super().bake_status(bake_id, timeout=timeout)


def test_a_new_longer_bake_writes_its_bake_id_right_after_the_start(longer_world, tmp_path):
    settings, _, dataset = longer_world
    out = tmp_path / "out"
    client = WatchesBakeId(dataset, out)
    assert longer(settings, client, out) == 0
    assert client.at_first_poll == {"id": "bake-1"}


LORA = json.dumps(catalog.get("voice", "voxcpm2").adapter.lora, sort_keys=True)


class StartFails(LongerClient):
    def bake_start(self, clips, *, tag, config=None):
        raise httpx.ReadTimeout("timed out")


@pytest.fixture
def lost_bake(longer_world, tmp_path):
    """The folder of a longer bake whose client failed at the start: only `frozen.json`."""
    settings, _, dataset = longer_world
    out = tmp_path / "out"
    assert longer(settings, StartFails(dataset), out) == 1
    assert sorted(path.name for path in out.iterdir()) == ["frozen.json"]
    takes = len(json.loads((out / "frozen.json").read_text())["takes"])
    running = longer_status(step=1, dataset=dataset) | {
        "tag": {"check": "me"}, "lora": LORA, "clips": takes, "elapsed_s": 100.0}
    return settings, dataset, out, running


def attach(settings, client, out, bake_id="bake-1"):
    return longer(settings, client, out, "--attach", bake_id)


def test_attach_follows_the_bake_starts_none_and_speaks(lost_bake):
    settings, dataset, out, running = lost_bake
    client = LongerClient(dataset, [running, longer_status(step=500, dataset=dataset),
                                    longer_status("done", step=LONGER_STEPS, dataset=dataset)])
    assert attach(settings, client, out) == 0
    assert "start" not in client.kinds() and "state" not in client.kinds()
    assert len(client.speaks()) == LONGER_CALLS
    report = report_of(out)
    assert report["attached"] is True and report["bake_id"] == "bake-1"
    assert report["poll_failures"] == 0
    assert report["free_mb_before"] == 9000.0
    assert json.loads((out / "bake.json").read_text()) == {"id": "bake-1"}


def test_attach_accepts_a_bake_json_of_the_same_id(lost_bake):
    settings, dataset, out, running = lost_bake
    (out / "bake.json").write_text(json.dumps({"id": "bake-1"}))
    client = LongerClient(dataset, [running,
                                    longer_status("done", step=LONGER_STEPS, dataset=dataset)])
    assert attach(settings, client, out) == 0
    assert "start" not in client.kinds()


@pytest.mark.parametrize(("field", "value"), [
    ("id", "bake-2"),
    ("tag", {"check": "you"}),
    ("model", "voxcpm1"),
    ("revision", "0" * 40),
    ("lora", json.dumps({"r": 32})),
    ("config", {"steps": 520}),
    ("clips", 99),
    ("dataset_sha256", "f" * 64),
])
def test_attach_refuses_a_bake_that_is_not_this_runs(lost_bake, capsys, field, value):
    settings, dataset, out, running = lost_bake
    before = snapshot(out)
    client = LongerClient(dataset, [running | {field: value}])
    assert attach(settings, client, out) == voice_bake_check.EXIT_USAGE
    assert client.kinds() == ["status"]
    assert field in capsys.readouterr().err
    assert snapshot(out) == before


def edit_frozen(out: Path) -> None:
    frozen = json.loads((out / "frozen.json").read_text())
    frozen["steps"] = 520
    (out / "frozen.json").write_text(json.dumps(frozen))


@pytest.mark.parametrize("dirty", [
    pytest.param(lambda out: (out / "L-01.wav").write_bytes(b"clip"), id="another-file"),
    pytest.param(lambda out: (out / "bake.json").write_text(json.dumps({"id": "bake-2"})),
                 id="another-bake-id"),
    pytest.param(lambda out: (out / "frozen.json").unlink(), id="no-frozen"),
    pytest.param(edit_frozen, id="another-frozen"),
])
def test_attach_refuses_a_folder_that_is_not_this_runs(lost_bake, dirty):
    settings, dataset, out, running = lost_bake
    dirty(out)
    before = snapshot(out)
    client = LongerClient(dataset, [running])
    assert attach(settings, client, out) == voice_bake_check.EXIT_USAGE
    assert client.events == []
    assert snapshot(out) == before


def test_attach_needs_the_longer_bake(world, tmp_path):
    settings, takes = world
    client = FakeClient()
    with pytest.raises(SystemExit) as stopped:
        run_main(settings, [*confirm(takes), "--attach", "bake-1"], client, out=tmp_path / "out")
    assert stopped.value.code == voice_bake_check.EXIT_USAGE
    assert client.events == []


def test_attach_survives_a_lost_poll_and_records_it(lost_bake):
    settings, dataset, out, running = lost_bake
    client = LongerClient(dataset, [running, TIMEOUT,
                                    longer_status("done", step=LONGER_STEPS, dataset=dataset)])
    assert attach(settings, client, out) == 0
    assert report_of(out)["poll_failures"] == 1


@pytest.mark.parametrize("args", [["--steps", "999"], ["--learning-rate", "0.0002"]])
def test_other_steps_or_a_learning_rate_are_refused(longer_world, tmp_path, args):
    settings, _, dataset = longer_world
    client = LongerClient(dataset)
    with pytest.raises(SystemExit) as stopped:
        longer(settings, client, tmp_path / "out", *args)
    assert stopped.value.code == voice_bake_check.EXIT_USAGE
    assert_no_bake_call(client, tmp_path / "out")


def test_steps_of_1000_are_accepted(longer_world, tmp_path):
    settings, _, dataset = longer_world
    assert longer(settings, LongerClient(dataset), tmp_path / "out", "--steps", "1000") == 0


def test_the_longer_bake_leaves_the_database_and_the_store_unchanged(longer_world, tmp_path):
    settings, _, dataset = longer_world
    db_before = settings.db_path.read_bytes()
    voices_before = snapshot(settings.voices_dir)
    assert longer(settings, LongerClient(dataset), tmp_path / "out") == 0
    assert settings.db_path.read_bytes() == db_before
    assert snapshot(settings.voices_dir) == voices_before


def test_the_longer_console_prints_no_score_and_no_model_per_file(longer_world, tmp_path, capsys):
    settings, _, dataset = longer_world
    longer(settings, LongerClient(dataset), tmp_path / "out")
    printed = capsys.readouterr().out
    assert "candidate" not in printed and "baseline" not in printed
    assert ADAPTER_SHA[:12] not in printed and HELD_SHA[:12] not in printed
    assert "cosine" not in printed.lower() and "similarity" not in printed.lower()
    assert len(file_rows(printed)) == LONGER_CALLS
    assert voice_bake_check.LONGER_LISTEN_FIRST in printed


def test_a_meter_that_cannot_run_after_the_bake_still_speaks(longer_world, tmp_path):
    settings, _, dataset = longer_world
    client = LongerClient(dataset)
    client.down = True
    assert longer(settings, client, tmp_path / "out") == 0
    report = report_of(tmp_path / "out")
    assert len(client.speaks()) == LONGER_CALLS
    assert "error" in report["likeness"]
    assert all("cosine" not in row for row in report["calls"])


def test_the_longer_bake_is_a_mode_of_its_own(longer_world, tmp_path):
    settings, _, dataset = longer_world
    client = LongerClient(dataset)
    with pytest.raises(SystemExit) as stopped:
        longer(settings, client, tmp_path / "out", "--heldout")
    assert stopped.value.code == voice_bake_check.EXIT_USAGE
    assert client.events == []


def neutral_report(label: str, cosine: float, *, model="other-tts", texts=None, reference="hash-1",
                   pipeline=1, failed=0) -> dict:
    """A report of another architecture: only the fields `--compare` may need."""
    results = [{"label": label, "arm": "C", "model": model, "revision": "abcdef1234",
                "text_id": "T-1", "repeat": repeat, "cosine": cosine + 0.02 * repeat,
                "seconds": 3.0, "windows": [], "error": None} for repeat in (0, 1)]
    results += [{"label": label, "arm": "C", "model": None, "revision": None, "text_id": "T-1",
                 "repeat": 2 + index, "cosine": None, "seconds": None, "windows": [],
                 "error": "failed"} for index in range(failed)]
    return {"label": label, "results": results,
            "frozen": {"texts": texts or {"T-1": "words"},
                       "reference": {"ru": {"reference_hash": reference, "pipeline": pipeline}}}}


def compare(tmp_path, old: dict, new: dict, capsys) -> str:
    paths = []
    for name, report in (("old.json", old), ("new.json", new)):
        (tmp_path / name).write_text(json.dumps(report))
        paths.append(str(tmp_path / name))
    assert voice_bake_check.main(["--compare", *paths]) == 0
    return capsys.readouterr().out


def test_compare_prints_mean_worst_spread_and_deltas_per_arm_and_text(tmp_path, capsys):
    printed = compare(tmp_path, neutral_report("before", 0.5), neutral_report("after", 0.6), capsys)
    assert "old: before (other-tts @ abcdef1)" in printed
    assert "new: after (other-tts @ abcdef1)" in printed
    rows = [line.split() for line in printed.splitlines()]
    expected = ["C", "0.5100", "0.5000", "0.0200", "0", "0.6100", "0.6000", "0.0200", "0",
                "+0.1000", "+0.1000", "+0.0000"]
    assert ["all", *expected] in rows
    assert ["T-1", *expected] in rows
    assert "frozen inputs that differ: none" in printed


def test_compare_prints_no_delta_when_the_new_report_has_a_failed_file(tmp_path, capsys):
    printed = compare(tmp_path, neutral_report("before", 0.5), neutral_report("after", 0.6, failed=1), capsys)
    rows = [line.split() for line in printed.splitlines()]
    expected = ["C", "0.5100", "0.5000", "0.0200", "0", "0.6100", "0.6000", "0.0200", "1",
                "-", "-", "-"]
    assert ["all", *expected] in rows
    assert ["T-1", *expected] in rows


def test_compare_prints_no_delta_when_the_old_report_has_a_failed_file(tmp_path, capsys):
    printed = compare(tmp_path, neutral_report("before", 0.5, failed=1), neutral_report("after", 0.6), capsys)
    rows = [line.split() for line in printed.splitlines()]
    assert ["all", "C", "0.5100", "0.5000", "0.0200", "1", "0.6100", "0.6000", "0.0200", "0",
            "-", "-", "-"] in rows


def test_a_failed_hard_file_gives_no_delta_and_a_clean_text_keeps_its_own(tmp_path, capsys):
    old = neutral_report("before", 0.5)
    new = neutral_report("after", 0.6, failed=1)
    for report, cosine in ((old, 0.7), (new, 0.8)):
        report["results"].append(report["results"][0] | {"text_id": "T-2", "cosine": cosine})
    rows = [line.split() for line in compare(tmp_path, old, new, capsys).splitlines()]
    assert ["all", "C", "0.5733", "0.5000", "0.2000", "0", "0.6733", "0.6000", "0.2000", "1",
            "-", "-", "-"] in rows
    assert ["T-1", "C", "0.5100", "0.5000", "0.0200", "0", "0.6100", "0.6000", "0.0200", "1",
            "-", "-", "-"] in rows
    assert ["T-2", "C", "0.7000", "0.7000", "0.0000", "0", "0.8000", "0.8000", "0.0000", "0",
            "+0.1000", "+0.1000", "+0.0000"] in rows


def test_compare_names_every_frozen_input_that_differs(tmp_path, capsys):
    new = neutral_report("after", 0.6, texts={"T-1": "other words"}, reference="hash-2", pipeline=2)
    printed = compare(tmp_path, neutral_report("before", 0.5), new, capsys)
    assert ("frozen inputs that differ: reference.ru.pipeline, reference.ru.reference_hash, "
            "texts.T-1") in printed


def test_compare_shows_an_arm_or_a_text_of_only_one_report(tmp_path, capsys):
    old = neutral_report("before", 0.5)
    new = neutral_report("after", 0.6)
    for result in new["results"]:
        result |= {"arm": "D", "text_id": "T-2"}
    rows = [line.split() for line in compare(tmp_path, old, new, capsys).splitlines()]
    assert ["all", "C", "0.5100", "0.5000", "0.0200", "0", "-", "-", "-", "0", "-", "-", "-"] in rows
    assert ["T-2", "D", "-", "-", "-", "0", "0.6100", "0.6000", "0.0200", "0", "-", "-", "-"] in rows


def test_compare_needs_no_voice_and_a_measure_report_compares_with_another_model(
        baked_world, tmp_path, capsys):
    settings, _ = baked_world
    measure(settings, MeterClient(), tmp_path / "out", "--repeats", "1", "--label", "voxcpm")
    capsys.readouterr()
    printed = compare(tmp_path, report_of(tmp_path / "out"), neutral_report("other", 0.5), capsys)
    assert "old: voxcpm (voxcpm2 @" in printed and "new: other (other-tts @" in printed


def test_compare_of_a_missing_report_stops_with_a_usage_code(tmp_path, capsys):
    code = voice_bake_check.main(["--compare", str(tmp_path / "a"), str(tmp_path / "b")])
    assert code == voice_bake_check.EXIT_USAGE


def test_no_mode_but_compare_runs_without_a_voice():
    with pytest.raises(SystemExit) as stopped:
        voice_bake_check.main(["--measure"])
    assert stopped.value.code == voice_bake_check.EXIT_USAGE


def test_a_silent_clip_prints_a_dash_for_its_level():
    lines = voice_bake_check._file_lines([{"id": "EN-1", "file": "X.wav", "duration_s": 1.0,
                                           "rms_dbfs": None, "void": False}])
    assert lines[1].split()[3] == "-"
