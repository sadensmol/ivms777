"""The bake of one saved voice in `app` (design §11.1 "Baking a saved voice"): hear the
words of every take that waits for them, write the immutable snapshot, train a voice
model from it in the models service and store the result.

A new start hears its takes first: one `POST /voice/transcribe` per take that waits,
then, under the job lock, the snapshot read that finds no waiting take writes the run
row (`start_bake_run`). A take that arrives during the stage sends the job back to the
stage; a take that arrives after that read is not in this bake. An attach (after an app
restart) reads only the stored snapshot of its run and hears nothing.

The whole job holds the voice lock, so films and tests wait for it. The job lock is the
publication barrier: the hand-off to the run row and the store run under it, and no
HTTP call does. Cancel is logical between calls and one `bake_cancel` during training;
the job then waits for ANY end of the service bake and stores nothing.

No models: this module is a thin client (CLAUDE.md § "One model process").
"""

import hashlib
import logging
import sqlite3
import threading
import time
from contextlib import nullcontext

import httpx

from feed.build import Cancelled, _reason, voice_turn
from feed.saved_voices import (
    BakeCancelled,
    SnapshotTake,
    VoiceError,
    VoiceExample,
    VoiceModel,
    bake_snapshot,
    end_bake_run,
    not_used_text,
    read_bake_run,
    read_example_audio,
    save_voice_model,
    set_heard_words,
    start_bake_run,
    take_to_hear,
    takes_to_hear,
)
from storage.base import Storage

logger = logging.getLogger(__name__)

POLL_SECONDS = 1.0
NO_TAKE = "No take yet. Record takes on Recordings."
RESTARTED = "The models service restarted, so the bake stopped. Bake again."
VOICE_GONE = "The voice was deleted during the bake, so the voice model was dropped."
NOT_OFFERED = "Hearing the words of a take is not offered on this device yet."
TRANSCRIBING = "transcribing the takes"
WAITING_FOR_OTHER = "waiting for the other bake to stop"
SENDING = "sending the takes"
STORING = "storing the voice model"


class BakeBlocked(VoiceError):
    """No take a bake can learn from; the text says why."""


class BakeFailed(VoiceError):
    """The bake ended without a voice model, for the reason the page shows."""


class BakeBusy(VoiceError):
    """The models service runs a bake this job did not start; `status` is its status
    (None when it ended meanwhile), from which the page asks its question (D83)."""

    def __init__(self, status: dict | None) -> None:
        super().__init__("The models service is baking another voice.")
        self.status = status


def run_bake(
    conn: sqlite3.Connection,
    client,
    voices: Storage,
    owner_id: int,
    voice_id: int,
    *,
    voice_lock: "threading.Lock | None" = None,
    job_lock: "threading.Lock | None" = None,
    progress=None,
    cancelled=None,
    attach: dict | None = None,
    wait_for_other: bool = False,
    on_started=None,
    on_saved=None,
) -> VoiceModel:
    """Bake the voice model of `voice_id` from all its takes and store it (D74, D92).

    `attach` is the service status of a running bake of this app (`reattach`): its run is
    read back by the tag's `bake_key`; `voice_id` is the voice of that run. `on_started(
    bake_key)` and `on_saved(model, done_text)` run while `job_lock` is held, so the job
    can flag that run on Cancel (D88) and mark itself done before a cancel is answered.

    Raises `BakeBlocked` before the voice lock when no take can be learned from,
    `BakeBusy` when another bake holds the service, `Cancelled`, and `BakeFailed`.
    """
    highest = [0.0]

    def report(stage: str, fraction: float) -> None:
        highest[0] = max(highest[0], fraction)
        if progress is not None:
            progress(stage, highest[0])

    def stop_if_cancelled(stage: str) -> None:
        if cancelled is not None and cancelled():
            raise Cancelled(stage)

    publication = job_lock if job_lock is not None else nullcontext()

    if attach is not None:
        bake_key = attach["tag"]["bake_key"]
        run = read_bake_run(conn, owner_id, bake_key)
        dropped = run is None or run.cancel_requested_at is not None
        takes, dataset_hash = (run.takes, run.dataset_hash) if run is not None else ((), "")

        def stop() -> bool:
            return dropped or (cancelled is not None and cancelled())

        with voice_turn(voice_lock, progress, cancelled, 0.0):
            with publication:
                if on_started is not None:
                    on_started(bake_key)
            return _follow_and_store(
                conn, client, voices, owner_id, voice_id, attach, bake_key, takes, dataset_hash,
                (), report, stop, publication, on_saved)

    snapshot = bake_snapshot(conn, owner_id, voice_id)
    if not (snapshot.takes or snapshot.waiting or snapshot.unusable):
        raise BakeBlocked(NO_TAKE)
    if not (snapshot.takes or snapshot.waiting):
        raise BakeBlocked(f"No take can be learned from: {not_used_text(snapshot.unusable)}. "
                          "Record or upload a take where you speak.")

    with voice_turn(voice_lock, progress, cancelled, 0.0):
        current = client.bake_current()
        if current is not None and current["state"] == "running":
            if not wait_for_other:
                raise BakeBusy(current)
            _wait_for_other(client, report, stop_if_cancelled)

        heard = 0
        while True:
            heard = _hear(conn, client, voices, owner_id, voice_id,
                          takes_to_hear(conn, owner_id, voice_id), heard, report, stop_if_cancelled)
            with publication:
                stop_if_cancelled(TRANSCRIBING)
                snapshot = bake_snapshot(conn, owner_id, voice_id)
                if snapshot.waiting:
                    continue
                if not snapshot.takes:
                    raise BakeBlocked(f"No take can be learned from: {not_used_text(snapshot.unusable)}. "
                                      "Record or upload a take where you speak.")
                bake_key = start_bake_run(conn, owner_id, voice_id, snapshot)
                if on_started is not None:
                    on_started(bake_key)
                break

        status: dict | None = None
        try:
            report(SENDING, 0.15)
            clips = []
            for take in snapshot.takes:
                try:
                    audio = read_example_audio(voices, take)
                except Exception as exc:
                    raise BakeFailed(
                        f"The audio of take #{take.id} could not be read ({_reason(exc)}).") from exc
                clips.append({"audio": audio, "sha256": take.content_hash, "text": take.transcript,
                              "language": take.language})
            tag = {"source": "app", "owner_id": owner_id, "voice_id": voice_id, "bake_key": bake_key}
            while True:
                stop_if_cancelled(SENDING)
                try:
                    status = client.bake_start(clips, tag=tag)
                    break
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code != 409:
                        raise
                    current = client.bake_current()
                    if not wait_for_other:
                        raise BakeBusy(current) from exc
                    _wait_for_other(client, report, stop_if_cancelled)
            logger.info("voice bake %s of voice %s started: %d takes, run %s", status["id"],
                        voice_id, len(clips), bake_key)
        except BaseException:
            end_bake_run(conn, bake_key)
            if status is not None:
                _delete(client, status["id"])
            raise
        run = read_bake_run(conn, owner_id, bake_key)
        return _follow_and_store(
            conn, client, voices, owner_id, voice_id, status, bake_key,
            run.takes if run is not None else (), snapshot.dataset_hash, snapshot.unusable,
            report, lambda: cancelled is not None and cancelled(), publication, on_saved)


def _wait_for_other(client, report, stop_if_cancelled) -> None:
    """Until the bake this job did not start has ended; never cancels or deletes it."""
    report(WAITING_FOR_OTHER, 0.01)
    while True:
        stop_if_cancelled(WAITING_FOR_OTHER)
        time.sleep(POLL_SECONDS)
        current = client.bake_current()
        if current is None or current["state"] != "running":
            return


def _hear(conn, client, voices: Storage, owner_id: int, voice_id: int,
          queue: list[VoiceExample], heard: int, report, stop_if_cancelled) -> int:
    """One round of the stage: every take of `queue` that still waits is heard and its
    words stored. Returns the takes of the stage so far; `heard` = those of the rounds
    before. Cancel is checked before each call; the call in flight is never interrupted."""
    total = heard + len(queue)
    for take in queue:
        heard += 1
        stage = f"{TRANSCRIBING} · take {heard} of {total}"
        report(stage, 0.01 + 0.14 * heard / total)
        stop_if_cancelled(TRANSCRIBING)
        if take_to_hear(conn, owner_id, voice_id, take.id) is None:
            continue
        try:
            audio = read_example_audio(voices, take)
        except Exception as exc:
            if take_to_hear(conn, owner_id, voice_id, take.id) is None:
                continue
            raise BakeFailed(
                f"The audio of take #{take.id} could not be read ({_reason(exc)}).") from exc
        try:
            reply = client.transcribe(audio, take.language)
        except Exception as exc:
            response = getattr(exc, "response", None)
            code = response.status_code if response is not None else None
            if code == 409:
                raise BakeBusy(client.bake_current()) from exc
            if code == 503:
                raise BakeFailed(NOT_OFFERED) from exc
            raise BakeFailed(f"Could not hear the words of take #{take.id} ({_reason(exc)}).") from exc
        set_heard_words(conn, owner_id, voice_id, take.id, text=reply["text"],
                        model=f"{reply['model']}@{reply['revision']}")
    return heard


def _delete(client, bake_id: str) -> None:
    try:
        client.bake_delete(bake_id)
    except Exception:
        logger.exception("deleting voice bake %s failed", bake_id)


def _stage_of(status: dict) -> str:
    if status["stage"] == "training" and status.get("steps"):
        return f"training · step {status['step']} of {status['steps']}"
    return status["stage"]


def _follow_and_store(
    conn, client, voices: Storage, owner_id: int, voice_id: int, status: dict, bake_key: str,
    takes: tuple[SnapshotTake, ...], dataset_hash: str, unusable: tuple[VoiceExample, ...],
    report, stop, publication, on_saved,
) -> VoiceModel:
    """Follow the service bake to ANY end, store a `done` one that nobody cancelled, and
    delete the bake and end the run on every way out."""
    bake_id = status["id"]
    try:
        cancel_sent = False
        try:
            while status["state"] == "running":
                if not cancel_sent and stop():
                    status = client.bake_cancel(bake_id)
                    cancel_sent = True
                    continue
                time.sleep(POLL_SECONDS)
                status = client.bake_status(bake_id)
                report(_stage_of(status), 0.16 + 0.81 * status["fraction"])
            if cancel_sent or stop() or status["state"] == "cancelled":
                raise Cancelled(status["stage"])
            if status["state"] == "failed":
                raise BakeFailed(f"The bake failed ({status.get('error')}).")
            report(STORING, 0.98)
            data = client.bake_adapter(bake_id)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise BakeFailed(RESTARTED) from exc
            raise
        if hashlib.sha256(data).hexdigest() != (status.get("adapter") or {}).get("sha256"):
            raise BakeFailed("The bake failed (the voice model file does not match its sha256).")
        with publication:
            if stop():
                raise Cancelled(STORING)
            try:
                model = save_voice_model(conn, voices, owner_id, voice_id, bake_key=bake_key,
                                         data=data, status=status, takes=takes,
                                         dataset_hash=dataset_hash)
            except BakeCancelled as exc:
                raise Cancelled(STORING) from exc
            if model is None:
                raise BakeFailed(VOICE_GONE)
            done = (f"Voice model v{model.version} is stored. Not used: {not_used_text(unusable)}."
                    if unusable else f"Voice model v{model.version} is stored.")
            logger.info("voice model v%s of voice %s stored from bake %s (%d takes)",
                        model.version, voice_id, bake_id, len(takes))
            if on_saved is not None:
                on_saved(model, done)
        return model
    finally:
        _delete(client, bake_id)
        end_bake_run(conn, bake_key)
