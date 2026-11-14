"""Baking a saved voice into a LoRA adapter — the parent side of a bake (design §5.1
`/voice/bakes`, §8.1 `voice_train`).

The parent imports no torch: the `voice_train` child trains, and this module writes the
takes, runs the child under a governor lease and one scheduler slot, polls it and tears
it down. `VoiceGate` is the one admission barrier of the voice path: every speak and
every manual `voice` ensure is counted in, and a bake closes the gate and waits for the
count to drop to zero before it unloads `voice`, so no accepted voice call can load a
second VoxCPM2 beside the training child.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import json
import logging
import secrets
import shutil
import threading
import time
import wave
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

from audio.voxcpm_trainer import train_config
from models import catalog
from modelsvc.scheduler import Priority
from modelsvc.slots import ASR_UNIT, SPEAKER_EMBED_UNIT, TRAIN_UNIT

log = logging.getLogger(__name__)

MAX_BAKE_CLIPS = 300
MAX_CLIP_BYTES = 4 * 1024 * 1024
MAX_BAKE_AUDIO_BYTES = 192 * 1024 * 1024
TEXT_MAX = 1000
TAG_MAX_BYTES = 64 * 1024
POLL_SECONDS = 1.0
# Display only: moves the bar while the training child loads, which nothing reports.
LOAD_ESTIMATE_S = 180.0
SAMPLE_RATE = 48000

PREPARING = "preparing the takes"
DRAINING = "waiting for the voice calls to end"
WAITING = "waiting for the GPU"
LOADING = "loading the voice model for training"
READING = "reading the takes"
TRAINING = "training"
SAVING = "saving the voice model"
DONE = "done"

_CHILD_STAGES = {READING: 0.10, TRAINING: 0.12, SAVING: 0.95}
_ENDED = ("done", "failed", "cancelled")
_MEASURED = ("step", "steps", "loss_first", "loss_last", "seconds_per_step", "device")


class BakeRunning(RuntimeError):
    """A second bake, a voice request during a bake, or a bake that is not done yet."""


class BakeRefused(ValueError):
    """A malformed bake request; the message is the reason."""


class BakeNotOffered(RuntimeError):
    """No catalog entry bakes on this profile (`catalog.adapter_entry_for`)."""


class BakeOnly(RuntimeError):
    """`voice_train` is asked for outside a bake."""


class VoiceGate:
    """Counts the voice calls in; a closed gate refuses new ones.

    The closed check and the count-in happen under one condition, so a call either
    counts before `close()` (and `drain()` waits for it) or is refused after it. No
    caller holds the condition across a scheduler wait or a load.
    """

    def __init__(self, percent: Callable[[], int] = lambda: 0) -> None:
        self._percent = percent
        self._cond = threading.Condition()
        self._closed = False
        self._inside = 0

    def enter(self) -> None:
        with self._cond:
            if not self._closed:
                self._inside += 1
                return
        raise BakeRunning(f"a voice is being baked ({self._percent()}%); "
                          "try again when it ends, or cancel it")

    def leave(self) -> None:
        with self._cond:
            self._inside -= 1
            self._cond.notify_all()

    @contextmanager
    def admitted(self) -> Iterator[None]:
        self.enter()
        try:
            yield
        finally:
            self.leave()

    def close(self) -> None:
        with self._cond:
            self._closed = True

    def drain(self) -> None:
        with self._cond:
            self._cond.wait_for(lambda: self._inside == 0)

    def open(self) -> None:
        with self._cond:
            self._closed = False

    def closed(self) -> bool:
        with self._cond:
            return self._closed


class _Bake:
    """One bake's progress. Every read and write happens under `VoiceBakes._lock`."""

    def __init__(self, bake_id: str, work_dir: Path, now: float, facts: dict) -> None:
        self.id = bake_id
        self.work_dir = work_dir
        self.facts = facts
        self.state = "running"
        self.stage = PREPARING
        self.fraction = 0.02
        self.stage_started = now
        self.started = now
        self.ended: float | None = None
        self.stage_seconds: dict[str, float] = {}
        self.measured: dict = {"step": 0, "steps": facts["config"]["steps"],
                               "loss_first": None, "loss_last": None,
                               "seconds_per_step": None, "device": None}
        self.free_mb_before: float | None = None
        self.min_free_mb: float | None = None
        self.peak_device_bytes: int | None = None
        self.error: str | None = None
        self.adapter: dict | None = None
        self.cancel_requested = False
        self.outcome: dict | None = None

    def enter_stage(self, stage: str, fraction: float, now: float) -> None:
        if stage == self.stage:
            self.fraction = fraction
            return
        self.stage_seconds[self.stage] = (self.stage_seconds.get(self.stage, 0.0)
                                          + now - self.stage_started)
        self.stage, self.fraction, self.stage_started = stage, fraction, now

    def shown_fraction(self, now: float) -> float:
        if self.state == "running" and self.stage == LOADING:
            return min(0.099, 0.02 + 0.08 * (now - self.stage_started) / LOAD_ESTIMATE_S)
        return self.fraction

    def snapshot(self, now: float) -> dict:
        until = self.ended if self.ended is not None else now
        stage_seconds = dict(self.stage_seconds)
        stage_seconds[self.stage] = stage_seconds.get(self.stage, 0.0) + until - self.stage_started
        adapter = None if self.adapter is None else {
            key: self.adapter[key] for key in ("sha256", "bytes", "keys")}
        return {
            "id": self.id, "state": self.state, "stage": self.stage,
            "fraction": self.shown_fraction(now), **self.measured,
            "stage_seconds": stage_seconds, "elapsed_s": until - self.started,
            "free_mb_before": self.free_mb_before, "min_free_mb": self.min_free_mb,
            "peak_device_bytes": self.peak_device_bytes, "error": self.error,
            "adapter": adapter, **self.facts,
        }


class VoiceBakes:
    """One bake at a time per service (D63); see the module docstring."""

    def __init__(self, *, slots, registry, governor, scheduler, activity, bake_root: Path,
                 profile: str, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self._slots = slots
        self._registry = registry
        self._governor = governor
        self._scheduler = scheduler
        self._activity = activity
        self._root = Path(bake_root)
        self._profile = profile
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._active = False
        self._bake: _Bake | None = None
        self.gate = VoiceGate(self._percent)

    def _percent(self) -> int:
        with self._lock:
            bake = self._bake
            return 0 if bake is None else int(bake.shown_fraction(self._clock()) * 100)

    def active(self) -> bool:
        with self._lock:
            return self._active

    def start(self, request: dict) -> dict:
        entry = catalog.adapter_entry_for(self._profile)
        if entry is None or not self._slots.has_unit(TRAIN_UNIT):
            raise BakeNotOffered("baking a voice is not offered on this device yet")
        with self._lock:
            if self._active:
                raise BakeRunning("a voice is already being baked")
            self._active = True
        try:
            clips, config, tag = _checked(request, entry)
            bake = self._prepare(entry, clips, config, tag)
        except BaseException:
            with self._lock:
                self._active = False
            raise
        log.info("voice bake %s started: %d takes, %d steps", bake.id, len(clips),
                 config["steps"])
        threading.Thread(target=self._run, args=(bake,), daemon=True,
                         name=f"voice-bake-{bake.id}").start()
        with self._lock:
            return bake.snapshot(self._clock())

    def _prepare(self, entry, clips: list[dict], config: dict, tag: dict) -> _Bake:
        with self._lock:
            previous, self._bake = self._bake, None
        if previous is not None:
            shutil.rmtree(previous.work_dir, ignore_errors=True)
        bake_id = secrets.token_hex(8)
        work_dir = self._root / bake_id
        try:
            (work_dir / "clips").mkdir(parents=True)
            lines = []
            for n, clip in enumerate(clips):
                path = work_dir / "clips" / f"{n}.wav"
                path.write_bytes(clip["audio"])
                lines.append(json.dumps({"audio": str(path.resolve()), "text": clip["text"]}))
            (work_dir / "train.jsonl").write_text("\n".join(lines) + "\n")
        except BaseException:
            shutil.rmtree(work_dir, ignore_errors=True)
            raise
        dataset = sorted(f"{c['sha256']}\t{c['language']}\t{c['text']}" for c in clips)
        facts = {
            "model": entry.key, "revision": entry.source.revision,
            "lora": json.dumps(entry.adapter.lora, sort_keys=True), "config": config,
            "dataset_sha256": hashlib.sha256("\n".join(dataset).encode()).hexdigest(),
            "clips": len(clips), "seconds": sum(c["seconds"] for c in clips), "tag": tag,
        }
        bake = _Bake(bake_id, work_dir, self._clock(), facts)
        with self._lock:
            self._bake = bake
        return bake

    def status(self, bake_id: str) -> dict | None:
        with self._lock:
            bake = self._bake
            if bake is None or bake.id != bake_id:
                return None
            return bake.snapshot(self._clock())

    def current(self) -> dict | None:
        with self._lock:
            return None if self._bake is None else self._bake.snapshot(self._clock())

    def cancel(self, bake_id: str) -> dict | None:
        with self._lock:
            bake = self._bake
            if bake is None or bake.id != bake_id:
                return None
            if bake.state == "running":
                bake.cancel_requested = True
            return bake.snapshot(self._clock())

    def adapter(self, bake_id: str) -> bytes | None:
        with self._lock:
            bake = self._bake
            if bake is None or bake.id != bake_id:
                return None
            if bake.state != "done":
                raise BakeRunning("the bake is not done")
            path = Path(bake.adapter["path"])
        return path.read_bytes()

    def delete(self, bake_id: str) -> bool:
        with self._lock:
            bake = self._bake
            if bake is None or bake.id != bake_id:
                return False
            if bake.state == "running":
                raise BakeRunning("the bake is still running; cancel it first")
            self._bake = None
        shutil.rmtree(bake.work_dir, ignore_errors=True)
        return True

    def sweep(self) -> None:
        if not self._root.exists():
            return
        for path in self._root.iterdir():
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)

    def _stage(self, bake: _Bake, stage: str, fraction: float) -> None:
        with self._lock:
            bake.enter_stage(stage, fraction, self._clock())

    def _cancelled(self, bake: _Bake) -> bool:
        with self._lock:
            if bake.cancel_requested:
                bake.outcome = {"state": "cancelled"}
            return bake.cancel_requested

    def _run(self, bake: _Bake) -> None:
        with self._activity.track("baking a voice"):
            self.gate.close()
            try:
                self._stage(bake, DRAINING, 0.02)
                self.gate.drain()
                if not self._cancelled(bake):
                    self._stage(bake, WAITING, 0.02)
                    self._governor.unload("voice")
                    # Whisper and the speaker model never share memory with the training
                    # child; a no-op when not loaded or not registered (§8.1).
                    self._governor.unload(ASR_UNIT)
                    self._governor.unload(SPEAKER_EMBED_UNIT)
                    free_mb = self._governor.state().free_mb
                    with self._lock:
                        bake.free_mb_before = free_mb
                    if not self._cancelled(bake):
                        self._hold_trainer(bake)
            except Exception as exc:  # noqa: BLE001 - the status carries it to the caller
                with self._lock:
                    bake.outcome = {"state": "failed", "error": str(exc) or type(exc).__name__}
            finally:
                shutil.rmtree(self._root / "hf-cache", ignore_errors=True)
                self._finish(bake)

    def _hold_trainer(self, bake: _Bake) -> None:
        """Lease, then reserve (the governor loads the child inside it), then train.
        The teardown releases the lease, unloads the child and only then leaves the
        reserve, so no op can take the slot while the child still holds its memory."""
        lease = self._governor.lease(TRAIN_UNIT)
        reserve = self._scheduler.reserve([TRAIN_UNIT], Priority.BATCH)
        reserved = False
        lease.__enter__()
        try:
            self._stage(bake, LOADING, 0.02)
            reserve.__enter__()
            reserved = True
            self._train(bake)
        finally:
            if reserved:
                self._last_status(bake)
            lease.__exit__(None, None, None)
            try:
                self._governor.unload(TRAIN_UNIT)
            finally:
                if reserved:
                    reserve.__exit__(None, None, None)

    def _train(self, bake: _Bake) -> None:
        if self._cancelled(bake):
            return
        self._stage(bake, READING, 0.10)
        worker = self._slots.worker(TRAIN_UNIT)
        with self._lock:
            job = {"manifest": str(bake.work_dir / "train.jsonl"), "work_dir": str(bake.work_dir),
                   "clips": bake.facts["clips"], "config": bake.facts["config"],
                   "dataset_sha256": bake.facts["dataset_sha256"]}
        worker.call("start", job)
        cancel_sent = False
        while True:
            with self._lock:
                cancel = bake.cancel_requested and not cancel_sent
            if cancel:
                worker.call("cancel")
                cancel_sent = True
            self._sleep(POLL_SECONDS)
            child = worker.call("status")
            free_mb = self._governor.state().free_mb
            with self._lock:
                self._observe(bake, child, free_mb)
            if child["state"] in _ENDED:
                return

    def _observe(self, bake: _Bake, child: dict, free_mb: float | None) -> None:
        bake.measured.update({key: child[key] for key in _MEASURED if key in child})
        peak = child.get("peak_device_bytes")
        if peak is not None:
            bake.peak_device_bytes = max(peak, bake.peak_device_bytes or 0)
        if free_mb is not None:
            bake.min_free_mb = free_mb if bake.min_free_mb is None else min(bake.min_free_mb, free_mb)
        stage = child.get("stage")
        if child["state"] == "running" and stage in _CHILD_STAGES:
            fraction = _CHILD_STAGES[stage]
            if stage == TRAINING and bake.measured["steps"]:
                fraction += 0.83 * bake.measured["step"] / bake.measured["steps"]
            bake.enter_stage(stage, fraction, self._clock())
        elif child["state"] == "done":
            bake.outcome = {"state": "done", "adapter": child["adapter"]}
        elif child["state"] == "failed":
            bake.outcome = {"state": "failed", "error": child.get("error")}
        elif child["state"] == "cancelled":
            bake.outcome = {"state": "cancelled"}

    def _last_status(self, bake: _Bake) -> None:
        if not self._registry.is_resident(TRAIN_UNIT):
            return
        try:
            child = self._slots.worker(TRAIN_UNIT).call("status")
        except Exception:  # best effort: the bake already has its outcome
            log.warning("voice bake %s: no last status from the training child", bake.id,
                        exc_info=True)
            return
        with self._lock:
            bake.measured.update({key: child[key] for key in _MEASURED if key in child})
            peak = child.get("peak_device_bytes")
            if peak is not None:
                bake.peak_device_bytes = max(peak, bake.peak_device_bytes or 0)

    def _finish(self, bake: _Bake) -> None:
        """Publish the end state only after the teardown, so a caller that sees it can
        speak at once: the training child is gone. The end state, the open gate and the
        free bake slot change under one lock, so a new `start` sees all three or none:
        it cannot remove this bake's folder before the end, and this bake cannot open
        the gate after the next bake has closed it."""
        with self._lock:
            outcome = bake.outcome or {"state": "failed",
                                       "error": "the training child ended with no result"}
            now = self._clock()
            if outcome["state"] == "done":
                bake.enter_stage(DONE, 1.0, now)
                bake.adapter = outcome["adapter"]
            bake.state = outcome["state"]
            bake.error = outcome.get("error")
            bake.ended = now
            self.gate.open()
            self._active = False
        log.info("voice bake %s ended: %s", bake.id, outcome["state"])


def _checked(request: dict, entry) -> tuple[list[dict], dict, dict]:
    """The request's clips (decoded), its loop values and its tag; `BakeRefused` names
    the first breach. Nothing is written before every check has passed."""
    model = request.get("model")
    if model is not None and model != entry.key:
        raise BakeRefused(f"model must be {entry.key!r} on this device, not {model!r}")
    raw = request.get("clips") or []
    if not 1 <= len(raw) <= MAX_BAKE_CLIPS:
        raise BakeRefused(f"a bake takes 1-{MAX_BAKE_CLIPS} takes, not {len(raw)}")
    clips, total = [], 0
    for n, clip in enumerate(raw):
        try:
            audio = base64.b64decode(clip["audio"], validate=True)
        except (binascii.Error, ValueError) as exc:
            raise BakeRefused(f"take {n}: the audio is not valid base64") from exc
        if len(audio) > MAX_CLIP_BYTES:
            raise BakeRefused(f"take {n}: {len(audio)} bytes; the limit is {MAX_CLIP_BYTES}")
        sha256 = clip["sha256"].lower()
        if hashlib.sha256(audio).hexdigest() != sha256:
            raise BakeRefused(f"take {n}: the audio does not match its sha256")
        seconds = _wav_seconds(audio, n)
        total += len(audio)
        if total > MAX_BAKE_AUDIO_BYTES:
            raise BakeRefused(f"the takes hold more than {MAX_BAKE_AUDIO_BYTES} bytes")
        text = clip["text"].strip()
        if not 1 <= len(text) <= TEXT_MAX:
            raise BakeRefused(f"take {n}: the words must be 1-{TEXT_MAX} characters, "
                              f"not {len(text)}")
        if clip["language"] not in entry.languages:
            raise BakeRefused(f"take {n}: {entry.key} does not speak {clip['language']!r}")
        clips.append({"audio": audio, "sha256": sha256, "text": text,
                      "language": clip["language"], "seconds": seconds})
    try:
        config = asdict(train_config(request.get("config") or {}, len(clips)))
    except (TypeError, ValueError) as exc:
        raise BakeRefused(f"config: {exc}") from exc
    tag = request.get("tag") or {}
    size = len(json.dumps(tag).encode())
    if size > TAG_MAX_BYTES:
        raise BakeRefused(f"the tag is {size} bytes as JSON; the limit is {TAG_MAX_BYTES}")
    return clips, config, tag


def _wav_seconds(audio: bytes, n: int) -> float:
    """The seconds of a mono PCM-16 WAV at 48000 Hz (the `/speak` reference format)."""
    try:
        with wave.open(io.BytesIO(audio), "rb") as handle:
            shape = (handle.getcomptype(), handle.getsampwidth(), handle.getnchannels(),
                     handle.getframerate())
            frames = handle.getnframes()
            payload = handle.readframes(frames)
    except (wave.Error, EOFError) as exc:
        raise BakeRefused(f"take {n}: not a WAV file") from exc
    if shape != ("NONE", 2, 1, SAMPLE_RATE):
        raise BakeRefused(f"take {n}: must be mono 16-bit PCM at {SAMPLE_RATE} Hz")
    if frames <= 0:
        raise BakeRefused(f"take {n}: holds no audio")
    if len(payload) != frames * 2:
        raise BakeRefused(f"take {n}: is cut short: part of its audio is missing")
    return frames / SAMPLE_RATE
