"""The bake job and the voice gate (design §5.1 `/voice/bakes`, §8.1 `voice_train`).

Fakes only: a scripted training child, a recording governor and scheduler, a fake clock.
No torch, no voxcpm, no datasets, no model load.
"""

import base64
import hashlib
import json
import threading
import time
import types
from contextlib import contextmanager

import numpy as np
import pytest

from modelsvc import voice_bakes
from modelsvc.activity import Activity
from modelsvc.backends.composite import CompositeBackend
from modelsvc.governor import MemoryGovernor
from modelsvc.registry import ModelRegistry, ModelSpec
from modelsvc.scheduler import Priority, Scheduler
from modelsvc.voice_bakes import (
    BakeNotOffered,
    BakeOnly,
    BakeRefused,
    BakeRunning,
    VoiceBakes,
    VoiceGate,
)
from storage.wav import encode_wav

ADAPTER_BYTES = b"safetensors-of-a-baked-voice"
DIED = "audio.voxcpm_trainer:VoxCPMTrainer worker died during status"
WAIT_S = 5.0


def wait_for(condition, timeout: float = WAIT_S) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("the condition never held")
        time.sleep(0.005)


def wav(seconds: float = 0.05, rate: int = 48000, channels: int = 1) -> bytes:
    frames = int(rate * seconds)
    shape = frames if channels == 1 else (channels, frames)
    return encode_wav(np.zeros(shape, dtype=np.float32), rate)


def a_clip(audio: bytes | None = None, text: str = "Hello there.", language: str = "en",
           **overrides) -> dict:
    audio = wav() if audio is None else audio
    return {"audio": base64.b64encode(audio).decode(),
            "sha256": hashlib.sha256(audio).hexdigest(),
            "text": text, "language": language} | overrides


def a_request(*clips, **overrides) -> dict:
    return {"model": "voxcpm2", "clips": list(clips) or [a_clip()],
            "config": {"steps": 2}, "tag": {"check": "me"}} | overrides


def child(state: str = "running", stage: str | None = None, step: int = 0, steps: int = 2,
          **extra) -> dict:
    return {"state": state, "stage": stage, "step": step, "steps": steps,
            "loss_first": None, "loss_last": None, "seconds_per_step": None,
            "peak_device_bytes": None, "device": "mps", "error": None,
            "adapter": None} | extra


DONE_SCRIPT = (child(stage="reading the takes"), child(stage="training", step=1),
               child("done", stage="saving the voice model", step=2))


class ScriptedChild:
    """The `voice_train` child: answers `start`, `status` and `cancel` like the trainer."""

    def __init__(self, harness, script=DONE_SCRIPT, *, fail_start=False, respond=None):
        self.h = harness
        self.script = list(script)
        self.fail_start = fail_start
        self.respond = respond
        self.job: dict | None = None
        self.polls = 0
        self.cancels = 0

    def call(self, method, *args):
        self.h.log.append(("call", method))
        if method == "start":
            self.job = args[0]
            (self.h.root / "hf-cache").mkdir(parents=True, exist_ok=True)
            if self.fail_start:
                raise RuntimeError("the training child refused the job")
            return None
        if method == "cancel":
            self.cancels += 1
            return None
        self.polls += 1
        row = (self.respond(self) if self.respond is not None
               else self.script[min(self.polls, len(self.script)) - 1])
        if row["state"] == "done" and row["adapter"] is None:
            path = f"{self.job['work_dir']}/adapter.safetensors"
            with open(path, "wb") as handle:
                handle.write(ADAPTER_BYTES)
            row = row | {"adapter": {"path": path, "bytes": len(ADAPTER_BYTES), "keys": 3,
                                     "sha256": hashlib.sha256(ADAPTER_BYTES).hexdigest()}}
        return row


class FakeSlots:
    def __init__(self, worker, offered=True):
        self._worker = worker
        self._offered = offered

    def has_unit(self, unit):
        return self._offered

    def worker(self, unit):
        return self._worker


class FakeRegistry:
    def __init__(self):
        self.resident: set[str] = set()

    def is_resident(self, name):
        return name in self.resident


class FakeGovernor:
    def __init__(self, harness, free_mb=(5000.0,)):
        self.h = harness
        self.free_mb = list(free_mb)
        self.seen: dict[str, tuple] = {}
        self.on_unload = None

    def unload(self, name):
        self.h.log.append(("unload", name))
        self.seen[name] = (self.h.bakes.gate.closed(), self.h.bakes.active(),
                           (self.h.root / "hf-cache").exists(), self.h.bakes.current())
        self.h.registry.resident.discard(name)
        if self.on_unload is not None:
            self.on_unload(name)

    @contextmanager
    def lease(self, name):
        self.h.log.append(("lease", name))
        try:
            yield
        finally:
            self.h.log.append(("release", name))

    def state(self):
        value = self.free_mb.pop(0) if len(self.free_mb) > 1 else self.free_mb[0]
        return types.SimpleNamespace(free_mb=value)


class FakeScheduler:
    def __init__(self, harness):
        self.h = harness
        self.on_reserve = None

    @contextmanager
    def reserve(self, needed, priority, *, pinned=frozenset()):
        self.h.log.append(("reserve", tuple(needed), priority))
        if self.on_reserve is not None:
            self.on_reserve()
        self.h.registry.resident.update(needed)
        try:
            yield
        finally:
            self.h.log.append(("leave reserve",))


class FakeClock:
    def __init__(self):
        self.t = 1000.0
        self.on_sleep = None

    def now(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds
        if self.on_sleep is not None:
            self.on_sleep()


class Harness:
    def __init__(self, tmp_path, *, profile="mac", offered=True, free_mb=(5000.0,), **child):
        self.log: list = []
        self.root = tmp_path / "voice-bakes"
        self.clock = FakeClock()
        self.registry = FakeRegistry()
        self.governor = FakeGovernor(self, free_mb)
        self.scheduler = FakeScheduler(self)
        self.child = ScriptedChild(self, **child)
        self.bakes = VoiceBakes(
            slots=FakeSlots(self.child, offered), registry=self.registry,
            governor=self.governor, scheduler=self.scheduler, activity=Activity(),
            bake_root=self.root, profile=profile, clock=self.clock.now,
            sleep=self.clock.sleep)

    def ended(self, bake_id: str) -> dict:
        wait_for(lambda: self.bakes.status(bake_id)["state"] != "running")
        return self.bakes.status(bake_id)

    def steps(self) -> list:
        return [event for event in self.log if event != ("call", "status")]


TEARDOWN = [("release", "voice_train"), ("unload", "voice_train"), ("leave reserve",)]
SETUP = [("unload", "voice"), ("unload", "asr"), ("unload", "speaker_embed"),
         ("lease", "voice_train"), ("reserve", ("voice_train",), Priority.BATCH),
         ("call", "start")]


def _cancel_on_first_poll(h):
    def cancel():
        h.bakes.cancel(h.bakes.current()["id"])
    return cancel


@pytest.mark.parametrize(("setup", "state"), [
    ({}, "done"),
    ({"script": (child("failed", error="RuntimeError: the loss is not a number"),)}, "failed"),
    ({"respond": lambda c: child("cancelled") if c.cancels else child(stage="training")},
     "cancelled"),
    ({"fail_start": True}, "failed"),
])
def test_a_bake_runs_in_order_and_always_tears_down_before_the_gate_opens(
        tmp_path, setup, state):
    h = Harness(tmp_path, **setup)
    if state == "cancelled":
        h.clock.on_sleep = _cancel_on_first_poll(h)

    bake_id = h.bakes.start(a_request())["id"]
    status = h.ended(bake_id)

    assert status["state"] == state
    steps = h.steps()
    assert steps[:len(SETUP)] == SETUP
    assert steps[-3:] == TEARDOWN
    gate_closed, active, _, _ = h.governor.seen["voice"]
    assert gate_closed and active
    gate_closed, active, _, _ = h.governor.seen["voice_train"]
    assert gate_closed and active
    assert not (h.root / "hf-cache").exists()
    assert not h.bakes.active()
    assert not h.bakes.gate.closed()


def test_an_exception_reports_the_reason_and_tears_down(tmp_path):
    h = Harness(tmp_path, fail_start=True)
    status = h.ended(h.bakes.start(a_request())["id"])
    assert status["error"] == "the training child refused the job"
    assert h.steps()[-3:] == TEARDOWN


def test_the_end_state_shows_only_after_the_teardown(tmp_path):
    h = Harness(tmp_path)
    bake_id = h.bakes.start(a_request())["id"]
    h.ended(bake_id)
    *_, seen_at_unload = h.governor.seen["voice_train"]
    assert seen_at_unload["state"] == "running"


def test_a_profile_with_no_baking_entry_is_not_offered(tmp_path):
    for h in (Harness(tmp_path, profile="jetson"), Harness(tmp_path, offered=False)):
        with pytest.raises(BakeNotOffered, match="not offered on this device yet"):
            h.bakes.start(a_request())
        assert h.log == []


class _BlockingVoice:
    def __init__(self, log):
        self.log = log
        self.entered = threading.Event()
        self.release = threading.Event()

    def speak(self, text, voice, segments=None, voices=None, references=None):
        self.entered.set()
        self.release.wait(WAIT_S)
        self.log.append(("speak returned",))
        return {"audio": b"RIFF", "sample_rate": 24000, "model": "v"}


class _HeldScheduler:
    """A scheduler whose slot is busy: `run` waits, as in `_admit`, until released."""

    def __init__(self, log):
        self.log = log
        self.waiting = threading.Event()
        self.release = threading.Event()

    def run(self, needed, priority, fn):
        self.waiting.set()
        self.release.wait(WAIT_S)
        self.log.append(("speak admitted",))
        return fn()

    def reap_idle(self, units):
        pass


class _RecordingSlots:
    def __init__(self):
        self.asked = []

    def has_unit(self, unit):
        return True

    def ensure_voice_for(self, language, *, cloning=False):
        self.asked.append(language)
        return True


def _in_thread(fn) -> threading.Thread:
    thread = threading.Thread(target=fn, daemon=True)
    thread.start()
    return thread


def test_a_bake_waits_for_a_speak_held_inside_the_backend(tmp_path):
    h = Harness(tmp_path)
    voice = _BlockingVoice(h.log)
    backend = CompositeBackend(embed=None, voice=voice, bakes=h.bakes)
    speaking = _in_thread(lambda: backend.speak("hi"))
    assert voice.entered.wait(WAIT_S)

    bake_id = h.bakes.start(a_request())["id"]
    wait_for(lambda: h.bakes.status(bake_id)["stage"] == "waiting for the voice calls to end")
    time.sleep(0.05)  # proves the unload did NOT happen while the speak runs
    assert ("unload", "voice") not in h.log

    voice.release.set()
    speaking.join(WAIT_S)
    assert h.ended(bake_id)["state"] == "done"
    assert h.log.index(("speak returned",)) < h.log.index(("unload", "voice"))


def test_a_speak_that_waits_for_a_slot_is_counted_the_same(tmp_path):
    h = Harness(tmp_path)
    scheduler = _HeldScheduler(h.log)
    backend = CompositeBackend(embed=None, voice=_BlockingVoice(h.log), scheduler=scheduler,
                               bakes=h.bakes)
    backend._voice.release.set()
    speaking = _in_thread(lambda: backend.speak("hi"))
    assert scheduler.waiting.wait(WAIT_S)

    bake_id = h.bakes.start(a_request())["id"]
    wait_for(lambda: h.bakes.status(bake_id)["stage"] == "waiting for the voice calls to end")
    time.sleep(0.05)  # proves the unload did NOT happen while the speak waits
    assert ("unload", "voice") not in h.log

    scheduler.release.set()
    speaking.join(WAIT_S)
    assert h.ended(bake_id)["state"] == "done"
    assert h.log.index(("speak returned",)) < h.log.index(("unload", "voice"))


def test_a_speak_after_the_close_is_refused_before_routing_and_runs_after_the_bake(tmp_path):
    h = Harness(tmp_path)
    held = threading.Event()
    h.scheduler.on_reserve = lambda: held.wait(WAIT_S)
    slots = _RecordingSlots()
    voice = _BlockingVoice(h.log)
    voice.release.set()
    backend = CompositeBackend(embed=None, voice=voice, slots=slots, bakes=h.bakes)

    bake_id = h.bakes.start(a_request())["id"]
    wait_for(lambda: ("reserve", ("voice_train",), Priority.BATCH) in h.log)
    with pytest.raises(BakeRunning, match=r"a voice is being baked \(\d+%\); try again"):
        backend.speak("hi", "en")
    assert slots.asked == []

    held.set()
    h.ended(bake_id)
    assert backend.speak("hi", "en")["model"] == "v"
    assert slots.asked == ["en"]


class _Conveyor:
    """A real registry, governor and concurrency=1 scheduler with logging units."""

    def __init__(self, h, *, hold_voice=False, fail_voice=False):
        self.h = h
        self.voice_loading = threading.Event()
        self.voice_release = threading.Event()
        if not hold_voice:
            self.voice_release.set()
        self.registry = ModelRegistry()
        for name, cost in (("music", 100), ("voice", 1000), ("voice_train", 1400)):
            self.registry.register(ModelSpec(
                name, (lambda n=name: self._load(n, fail_voice)),
                (lambda n=name: h.log.append(("free", n))), cost))
        self.governor = MemoryGovernor(self.registry, measure_free_mb=lambda: 99999.0,
                                       budget_mb=99999)
        self.scheduler = Scheduler(self.governor, concurrency=1)
        h.bakes = VoiceBakes(
            slots=FakeSlots(h.child), registry=self.registry, governor=self.governor,
            scheduler=self.scheduler, activity=Activity(), bake_root=h.root, profile="mac",
            clock=h.clock.now, sleep=h.clock.sleep)
        self.backend = CompositeBackend(embed=None, registry=self.registry,
                                        governor=self.governor, scheduler=self.scheduler,
                                        bakes=h.bakes)

    def _load(self, name, fail_voice):
        if name == "voice":
            self.voice_loading.set()
            self.voice_release.wait(WAIT_S)
            if fail_voice:
                raise RuntimeError("the voice child did not start")
        if name == "voice_train":
            self.h.log.append(("voice resident at the trainer load",
                               self.registry.is_resident("voice")))
        self.h.log.append(("load", name))


def test_an_ensure_queued_for_the_slot_loads_voice_before_the_bake_unloads_it(tmp_path):
    h = Harness(tmp_path)
    conveyor = _Conveyor(h)
    slot_free = threading.Event()
    holding = _in_thread(lambda: conveyor.scheduler.run(
        ["music"], Priority.BATCH, lambda: slot_free.wait(WAIT_S)))
    wait_for(lambda: ("load", "music") in h.log)
    ensuring = _in_thread(lambda: conveyor.backend.model_ensure("voice"))
    wait_for(lambda: conveyor.scheduler._waiters)

    bake_id = h.bakes.start(a_request())["id"]
    wait_for(lambda: h.bakes.status(bake_id)["stage"] == "waiting for the voice calls to end")
    slot_free.set()
    holding.join(WAIT_S)
    ensuring.join(WAIT_S)

    assert h.ended(bake_id)["state"] == "done"
    loads = [e for e in h.log if e[0] in ("load", "free")]
    assert loads == [("load", "music"), ("load", "voice"), ("free", "voice"),
                     ("load", "voice_train"), ("free", "voice_train")]
    assert ("voice resident at the trainer load", False) in h.log


def test_a_bake_waits_for_a_held_manual_voice_load(tmp_path):
    h = Harness(tmp_path)
    conveyor = _Conveyor(h, hold_voice=True)
    ensuring = _in_thread(lambda: conveyor.backend.model_ensure("voice"))
    assert conveyor.voice_loading.wait(WAIT_S)

    bake_id = h.bakes.start(a_request())["id"]
    wait_for(lambda: h.bakes.status(bake_id)["stage"] == "waiting for the voice calls to end")
    time.sleep(0.05)  # proves the unload did NOT happen while the load runs
    assert ("free", "voice") not in h.log

    conveyor.voice_release.set()
    ensuring.join(WAIT_S)
    assert h.ended(bake_id)["state"] == "done"
    assert h.log.index(("load", "voice")) < h.log.index(("free", "voice")) \
        < h.log.index(("load", "voice_train"))


def _drains(gate: VoiceGate) -> bool:
    gate.close()
    draining = _in_thread(gate.drain)
    draining.join(1.0)
    return not draining.is_alive()


def test_a_failed_voice_load_through_the_scheduler_releases_its_count(tmp_path):
    h = Harness(tmp_path)
    conveyor = _Conveyor(h, fail_voice=True)
    with pytest.raises(RuntimeError, match="did not start"):
        conveyor.backend.model_ensure("voice")
    assert _drains(h.bakes.gate)


def test_a_failed_voice_load_through_the_registry_alone_releases_its_count(tmp_path):
    h = Harness(tmp_path)
    conveyor = _Conveyor(h, fail_voice=True)
    backend = CompositeBackend(embed=None, registry=conveyor.registry, bakes=h.bakes)
    with pytest.raises(RuntimeError, match="did not start"):
        backend.model_ensure("voice")
    assert _drains(h.bakes.gate)


def test_a_closed_gate_refuses_a_manual_voice_ensure_with_no_load(tmp_path):
    h = Harness(tmp_path)
    conveyor = _Conveyor(h)
    h.bakes.gate.close()
    for backend in (conveyor.backend,
                    CompositeBackend(embed=None, registry=conveyor.registry, bakes=h.bakes)):
        with pytest.raises(BakeRunning):
            backend.model_ensure("voice")
    assert h.log == []


def test_voice_train_is_loaded_only_by_a_bake(tmp_path):
    h = Harness(tmp_path)
    conveyor = _Conveyor(h)
    with pytest.raises(BakeOnly, match="voice_train is loaded only by a bake"):
        conveyor.backend.model_ensure("voice_train")
    assert h.log == []


def test_the_work_folder_holds_every_take_and_a_manifest(tmp_path):
    h = Harness(tmp_path)
    takes = [wav(0.05), wav(0.1)]
    bake_id = h.bakes.start(a_request(a_clip(takes[0], text="  One.  "),
                                      a_clip(takes[1], text="Two.", language="ru")))["id"]
    status = h.ended(bake_id)

    work_dir = h.root / bake_id
    assert [(work_dir / "clips" / f"{n}.wav").read_bytes() for n in range(2)] == takes
    lines = [json.loads(line) for line in (work_dir / "train.jsonl").read_text().splitlines()]
    assert lines == [{"audio": str((work_dir / "clips" / "0.wav").resolve()), "text": "One."},
                     {"audio": str((work_dir / "clips" / "1.wav").resolve()), "text": "Two."}]
    assert h.child.job == {"manifest": str(work_dir / "train.jsonl"), "work_dir": str(work_dir),
                           "clips": 2, "config": status["config"],
                           "dataset_sha256": status["dataset_sha256"]}
    assert status["clips"] == 2 and status["seconds"] == pytest.approx(0.15)
    assert status["tag"] == {"check": "me"}
    assert status["model"] == "voxcpm2"


def test_the_dataset_hash_does_not_depend_on_the_order_of_the_takes(tmp_path):
    one, two = a_clip(wav(0.05), text="One."), a_clip(wav(0.1), text="Two.", language="ru")
    expected = hashlib.sha256("\n".join(sorted([
        f"{one['sha256']}\ten\tOne.", f"{two['sha256']}\tru\tTwo."])).encode()).hexdigest()
    h = Harness(tmp_path)
    first = h.ended(h.bakes.start(a_request(one, two))["id"])["dataset_sha256"]
    second = h.ended(h.bakes.start(a_request(two, one))["id"])["dataset_sha256"]
    assert first == second == expected


def test_the_load_stage_moves_on_the_clock_and_stops_below_ten_percent(tmp_path):
    h = Harness(tmp_path)
    seen = []

    def loading():
        h.clock.t += 90.0
        seen.append(h.bakes.current())
        h.clock.t += 1000.0
        seen.append(h.bakes.current())

    h.scheduler.on_reserve = loading
    h.ended(h.bakes.start(a_request())["id"])
    assert [s["stage"] for s in seen] == ["loading the voice model for training"] * 2
    assert seen[0]["fraction"] == pytest.approx(0.02 + 0.08 * 90.0 / 180.0)
    assert seen[1]["fraction"] == pytest.approx(0.099)
    at_unload = h.governor.seen["voice"][3]
    assert (at_unload["stage"], at_unload["fraction"]) == ("waiting for the GPU", 0.02)


def test_the_status_maps_the_child_stages_and_keeps_the_extremes(tmp_path):
    script = (child(stage="reading the takes", peak_device_bytes=10),
              child(stage="training", step=50, steps=200, peak_device_bytes=30,
                    loss_first=2.0, loss_last=1.5),
              child(stage="saving the voice model", step=200, steps=200, peak_device_bytes=20,
                    loss_first=2.0, loss_last=1.0),
              child("done", stage="saving the voice model", step=200, steps=200,
                    loss_first=2.0, loss_last=1.0))
    h = Harness(tmp_path, script=script, free_mb=(5000.0, 4000.0, 3000.0, 3500.0, 3200.0))
    seen = []
    h.clock.on_sleep = lambda: seen.append(h.bakes.current())

    status = h.ended(h.bakes.start(a_request())["id"])

    assert [(s["stage"], round(s["fraction"], 4)) for s in seen] == [
        ("reading the takes", 0.10), ("reading the takes", 0.10),
        ("training", round(0.12 + 0.83 * 50 / 200, 4)), ("saving the voice model", 0.95)]
    assert (status["state"], status["stage"], status["fraction"]) == ("done", "done", 1.0)
    assert status["free_mb_before"] == 5000.0
    assert status["min_free_mb"] == 3000.0
    assert status["peak_device_bytes"] == 30
    assert (status["loss_first"], status["step"], status["steps"]) == (2.0, 200, 200)
    assert status["adapter"] == {"sha256": hashlib.sha256(ADAPTER_BYTES).hexdigest(),
                                 "bytes": len(ADAPTER_BYTES), "keys": 3}
    assert {"waiting for the GPU", "training", "done"} <= set(status["stage_seconds"])
    assert status["elapsed_s"] == pytest.approx(4.0)


def test_a_cancel_before_the_reserve_ends_the_bake_with_no_training_call(tmp_path):
    h = Harness(tmp_path)
    voice = _BlockingVoice(h.log)
    backend = CompositeBackend(embed=None, voice=voice, bakes=h.bakes)
    speaking = _in_thread(lambda: backend.speak("hi"))
    assert voice.entered.wait(WAIT_S)
    bake_id = h.bakes.start(a_request())["id"]
    wait_for(lambda: h.bakes.status(bake_id)["stage"] == "waiting for the voice calls to end")

    assert h.bakes.cancel(bake_id)["state"] == "running"
    voice.release.set()
    speaking.join(WAIT_S)

    assert h.ended(bake_id)["state"] == "cancelled"
    assert not any(e[0] in ("reserve", "call") for e in h.log)
    assert not h.bakes.gate.closed()


def test_a_cancel_during_training_is_sent_once_and_shows_when_the_child_says_so(tmp_path):
    def respond(c):
        return child("cancelled") if c.cancels and c.polls > 4 else child(stage="training",
                                                                         step=1)

    h = Harness(tmp_path, respond=respond)
    seen = []

    def on_sleep():
        current = h.bakes.current()
        if h.child.polls == 1:
            h.bakes.cancel(current["id"])
        seen.append((h.child.cancels, current["state"]))

    h.clock.on_sleep = on_sleep
    status = h.ended(h.bakes.start(a_request())["id"])

    assert status["state"] == "cancelled"
    assert h.child.cancels == 1
    assert [state for cancels, state in seen if cancels] == ["running"] * 3
    with pytest.raises(BakeRunning):
        h.bakes.adapter(status["id"])


def test_one_bake_at_a_time_and_its_adapter_and_folder_follow_its_state(tmp_path):
    h = Harness(tmp_path)
    held = threading.Event()
    h.scheduler.on_reserve = lambda: held.wait(WAIT_S)
    bake_id = h.bakes.start(a_request())["id"]
    wait_for(lambda: ("reserve", ("voice_train",), Priority.BATCH) in h.log)

    with pytest.raises(BakeRunning):
        h.bakes.start(a_request())
    with pytest.raises(BakeRunning):
        h.bakes.adapter(bake_id)
    with pytest.raises(BakeRunning):
        h.bakes.delete(bake_id)

    held.set()
    assert h.ended(bake_id)["state"] == "done"
    assert h.bakes.adapter(bake_id) == ADAPTER_BYTES
    assert h.bakes.delete(bake_id) is True
    assert not (h.root / bake_id).exists()
    assert h.bakes.status(bake_id) is None and h.bakes.current() is None
    assert h.bakes.delete(bake_id) is False


@pytest.mark.parametrize(("setup", "state"), [
    ({}, "done"),
    ({"script": (child("failed", error="RuntimeError: the loss is not a number"),)}, "failed"),
    ({"respond": lambda c: child("cancelled") if c.cancels else child(stage="training")},
     "cancelled"),
])
def test_a_start_during_the_hand_off_waits_for_the_end_and_the_new_bake_owns_the_gate(
        tmp_path, monkeypatch, setup, state):
    h = Harness(tmp_path, **setup)
    if state == "cancelled":
        h.clock.on_sleep = _cancel_on_first_poll(h)
    gate = h.bakes.gate
    first_go, second_go, resume = threading.Event(), threading.Event(), threading.Event()
    first_reserved, second_reserved, pausing = (threading.Event(), threading.Event(),
                                                threading.Event())
    gate_events: list = []
    removed_at: list = []
    second: dict = {}
    real_open, real_close, real_rmtree = gate.open, gate.close, voice_bakes.shutil.rmtree

    def on_reserve():
        if not first_reserved.is_set():
            first_reserved.set()
            first_go.wait(WAIT_S)
        else:
            second_reserved.set()
            second_go.wait(WAIT_S)

    def close():
        gate_events.append(("close", threading.current_thread()))
        real_close()

    def open_paused_once():
        gate_events.append(("open", threading.current_thread()))
        if not pausing.is_set():
            pausing.set()
            resume.wait(WAIT_S)
        real_open()

    def rmtree(path, *args, **kwargs):
        if path == first.work_dir:
            removed_at.append(first.state)
        real_rmtree(path, *args, **kwargs)

    def start_second():
        second["status"] = h.bakes.start(a_request())

    h.scheduler.on_reserve = on_reserve
    monkeypatch.setattr(gate, "close", close)
    monkeypatch.setattr(gate, "open", open_paused_once)
    monkeypatch.setattr(voice_bakes.shutil, "rmtree", rmtree)
    starting = None
    try:
        first_id = h.bakes.start(a_request())["id"]
        first = h.bakes._bake
        first_go.set()
        assert pausing.wait(WAIT_S)

        starting = _in_thread(start_second)
        starting.join(0.05)  # proves the second start did NOT return during the hand-off
        assert starting.is_alive()
        assert (h.root / first_id).exists()
        assert gate.closed()

        resume.set()
        starting.join(WAIT_S)
        assert not starting.is_alive()
        assert removed_at == [state]
        assert second_reserved.wait(WAIT_S)
        with pytest.raises(BakeRunning):
            gate.enter()
        first_thread = gate_events[0][1]
        first_thread.join(WAIT_S)
        assert not first_thread.is_alive()
        with pytest.raises(BakeRunning):
            gate.enter()

        second_go.set()
        h.ended(second["status"]["id"])
        second_close = [n for n, (event, _) in enumerate(gate_events) if event == "close"][1]
        second_thread = gate_events[second_close][1]
        assert second_thread is not first_thread
        opens_after = [thread for event, thread in gate_events[second_close:] if event == "open"]
        assert opens_after and all(thread is second_thread for thread in opens_after)
    finally:
        for event in (first_go, resume, second_go):
            event.set()
        if starting is not None:
            starting.join(WAIT_S)
        for _, thread in gate_events:
            thread.join(WAIT_S)


def test_a_new_bake_removes_the_folder_of_the_last_ended_one(tmp_path):
    h = Harness(tmp_path)
    first = h.ended(h.bakes.start(a_request())["id"])["id"]
    second = h.bakes.start(a_request())["id"]
    assert not (h.root / first).exists()
    assert h.bakes.status(first) is None
    h.ended(second)


def test_sweep_removes_every_leftover_folder(tmp_path):
    h = Harness(tmp_path)
    (h.root / "0123abcd" / "clips").mkdir(parents=True)
    (h.root / "hf-cache").mkdir()
    h.bakes.sweep()
    assert list(h.root.iterdir()) == []


def test_unknown_ids_are_none(tmp_path):
    h = Harness(tmp_path)
    assert h.bakes.status("nope") is None
    assert h.bakes.cancel("nope") is None
    assert h.bakes.adapter("nope") is None
    assert h.bakes.delete("nope") is False
    assert h.bakes.current() is None


@pytest.mark.parametrize(("request_body", "reason"), [
    (a_request() | {"clips": []}, "1-300 takes, not 0"),
    (a_request(*[a_clip()] * 301), "1-300 takes, not 301"),
    (a_request(a_clip(b"x" * (5 * 1024 * 1024))), "the limit is 4194304"),
    (a_request(a_clip(sha256="0" * 64)), "does not match its sha256"),
    (a_request(a_clip(wav(channels=2))), "must be mono 16-bit PCM at 48000 Hz"),
    (a_request(a_clip(wav(rate=44100))), "must be mono 16-bit PCM at 48000 Hz"),
    (a_request(a_clip(wav(0.01)[:44])), "take 0: is cut short: part of its audio is missing"),
    (a_request(a_clip(wav(0))), "take 0: holds no audio"),
    (a_request(a_clip(wav(0.01)[:-1])), "take 0: is cut short: part of its audio is missing"),
    (a_request(a_clip(text="   ")), "1-1000 characters, not 0"),
    (a_request(a_clip(text="a" * 1001)), "1-1000 characters, not 1001"),
    (a_request(a_clip(language="de")), "does not speak 'de'"),
    (a_request(config={"steps": 0}), "config: steps must be 1-5000"),
    (a_request(tag={"note": "x" * 70 * 1024}), "the limit is 65536"),
    (a_request(model="kokoro-82m"), "model must be 'voxcpm2'"),
    (a_request(a_clip() | {"audio": "%%%"}), "not valid base64"),
])
def test_a_malformed_request_is_refused_and_writes_nothing(tmp_path, request_body, reason):
    h = Harness(tmp_path)
    with pytest.raises(BakeRefused, match=reason):
        h.bakes.start(request_body)
    assert not h.root.exists()
    assert not h.bakes.active()
    assert h.log == []


def test_a_child_that_dies_fails_the_bake_with_its_text_and_tears_down(tmp_path):
    def respond(c):
        raise RuntimeError(DIED)

    h = Harness(tmp_path, respond=respond)
    status = h.ended(h.bakes.start(a_request())["id"])
    assert (status["state"], status["error"]) == ("failed", DIED)
    assert h.steps()[-3:] == TEARDOWN
    assert not h.bakes.gate.closed()


def test_a_bake_unloads_asr_right_after_voice_and_before_its_lease(tmp_path):
    h = Harness(tmp_path)
    h.registry.resident.update({"voice", "asr"})

    h.ended(h.bakes.start(a_request())["id"])

    steps = h.steps()
    assert steps.index(("unload", "voice")) + 1 == steps.index(("unload", "asr"))
    assert steps.index(("unload", "asr")) < steps.index(("lease", "voice_train"))
    gate_closed, active, _, _ = h.governor.seen["asr"]
    assert gate_closed and active
    assert "asr" not in h.registry.resident


def test_a_bake_unloads_the_speaker_model_right_after_asr_and_before_its_lease(tmp_path):
    h = Harness(tmp_path)
    h.registry.resident.update({"speaker_embed"})

    h.ended(h.bakes.start(a_request())["id"])

    steps = h.steps()
    assert steps.index(("unload", "asr")) + 1 == steps.index(("unload", "speaker_embed"))
    assert steps.index(("unload", "speaker_embed")) < steps.index(("lease", "voice_train"))
    gate_closed, active, _, _ = h.governor.seen["speaker_embed"]
    assert gate_closed and active
    assert "speaker_embed" not in h.registry.resident


def test_with_no_asr_registered_the_bake_runs_as_before(tmp_path):
    # The real registry returns at once for a unit that is not loaded or not registered.
    from modelsvc.registry import ModelRegistry

    h = Harness(tmp_path)
    registry = ModelRegistry()
    unload = h.governor.unload

    def through_registry(name):
        registry.unload(name)
        unload(name)

    h.governor.unload = through_registry

    status = h.ended(h.bakes.start(a_request())["id"])

    assert status["state"] == "done"
    assert h.steps()[:len(SETUP)] == SETUP
