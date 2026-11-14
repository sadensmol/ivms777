"""The Voices pages (design §6, §11.1, §13, §13.1): the grid `/voices` and its one leaf
`/voices/{id}`. The old address `/organize/voices` answers 308.

Every action is a `fetch` that answers a fragment — the cards (`_voice_cards.html`) on the
grid, the leaf body (`_voice_body.html`) on a voice — or a JSON `{"detail": …}` error; none
pushes history. A delete only records the intent
(`feed/saved_voices.py`, D33) and replies with the row "deleting…". The worker drains it
in `drain_pass`; without a worker (`inline_drain`), `VoiceDeletionDrain` drains it here,
on one background thread, so the request never waits for the cascade.

Whether voice copy is offered follows from the catalog and this app's own profile
(`cloning_entry_for`, `cloning_languages`) — no call to the models service (§11.1).

A voice test (`VoicePreviews`) runs in `app` like a film build: one job per process, a
polled fragment (`_voice_preview.html`) with a stage, a percentage and Cancel. It speaks
from the take films of its language speak from (`film_take`, §11.1).

The hearing job (`VoiceHearings`) runs the same way: one per process, a polled fragment
(`_voice_hearing.html`). A saved take starts it, so films and tests can speak from the
take once its words are heard.

`app` starts no bake (§11.1). `VoiceBakeJobs` stays only to finish or clean up a bake this
app had started before a restart (`reattach`, D74, D83); no page shows it.
"""

import logging
import secrets
import threading
import time
from collections.abc import Callable

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates

from feed import voice_bake
from feed.build import Cancelled, _reason
from feed.language import LANGUAGES
from feed.saved_voices import (
    KIT_STYLES,
    MAX_BYTES,
    RECORDING_KIT,
    SavedVoice,
    VoiceDeleting,
    VoiceError,
    VoiceNameTaken,
    add_example,
    create_voice,
    film_take,
    get_voice,
    list_voices,
    pending_voice_deletions,
    process_voice_deletions,
    read_bake_run,
    read_example_audio,
    read_test_audio,
    rename_voice,
    request_bake_cancel,
    request_example_deletion,
    request_voice_deletion,
    voice_model_for,
    voice_test,
)
from feed.voice_bake import BakeBusy, run_bake
from feed.voice_hearing import NOT_OFFERED as HEAR_NOT_OFFERED
from feed.voice_hearing import HearingFailed, hear_takes
from feed.voice_likeness import measured_older_takes
from feed.voice_preview import (
    CUSTOM_TEXT_MAX,
    PreviewFailed,
    PreviewJob,
    _cannot_test,
    preview_text,
    run_preview,
)
from models.catalog import adapter_entry_for, cloning_entry_for, cloning_languages, transcriber_for
from web.deps import AppContext
from web.upload_api import FREE_SPACE_FLOOR

logger = logging.getLogger("ivms777.voices")

# The worker's `POLL_SECONDS` (ingest/cli.py): a drain pass that left rows behind (a
# failed unlink, an in-flight write) runs again this much later.
RETRY_SECONDS = 10

# English names for the page and the narrator labels ("no Russian example"). The feed's
# own language buttons keep `feed.language.DISPLAY` (each language in itself).
LANGUAGE_NAMES: dict[str, str] = {"en": "English", "ru": "Russian"}

TAKE_LABEL_CHARS = 80
NOT_AVAILABLE = "Voice copy is not available on this device."
BAKE_CANCELLED = "The bake was cancelled, so no voice model was stored."

READY_PROBLEMS: dict[str, str] = {
    "language": "no take yet",
    "waiting": "the words of its takes are not heard yet",
    "unusable": "no take with usable words",
    "deleting": "the voice is being deleted",
}

# The leaf's tabs (§13.1): state of the leaf, carried in `?tab`, never history.
TABS: tuple[tuple[str, str], ...] = (("recordings", "Recordings"), ("test", "Test"))


def language_name(code: str) -> str:
    return LANGUAGE_NAMES.get(code, code)


def clock(seconds: float) -> str:
    """`m:ss` of a take or of the takes of one language."""
    whole = round(seconds)
    return f"{whole // 60}:{whole % 60:02d}"


def percent(fraction: float | None) -> int:
    return min(100, max(0, int(100 * (fraction or 0.0))))


def foreign_bake_question(status: dict, name: str) -> str:
    """The question before a bake stops one that `app` did not start (D83)."""
    return (f"The models service is baking a voice for the bake check ({percent(status.get('fraction'))}%). "
            f"Stop it and bake «{name}» instead?")


def _is_app_bake(status: dict | None, owner_id: int) -> bool:
    tag = (status or {}).get("tag") or {}
    return tag.get("source") == "app" and tag.get("owner_id") == owner_id


def _recordings(voice: SavedVoice, languages: tuple[str, ...]) -> list[dict]:
    """The Recordings tab (K1, D101): per language voice copy speaks here, also with no
    takes, the recording kit by style with the takes on each kit text; then "Other takes"
    (uploads and recordings with no kit text). A language voice copy does not speak here
    keeps its takes, with no Record controls. Takes are listed newest first."""
    sections = []
    for code in LANGUAGES:
        takes = [take for take in reversed(voice.examples) if take.language == code]
        recordable = code in languages
        if not recordable and not takes:
            continue
        kit = RECORDING_KIT[code]
        on_text = {text.id: [take for take in takes if take.kit_id == text.id] for text in kit}
        groups = []
        for style, (label, instruction) in KIT_STYLES.items():
            rows = [(text, on_text[text.id]) for text in kit
                    if text.style == style and (recordable or on_text[text.id])]
            if rows:
                groups.append({"label": label, "instruction": instruction, "rows": rows})
        kept = [take for take in takes if not take.deleting]
        sections.append({
            "code": code,
            "name": language_name(code),
            "recordable": recordable,
            "kit_done": sum(1 for found in on_text.values() if any(not t.deleting for t in found)),
            "kit_total": len(kit),
            "seconds": sum(take.seconds for take in kept),
            "groups": groups,
            "other": [take for take in takes if take.kit_id not in on_text],
        })
    return sections


def _test_texts(voice: SavedVoice, languages: list[str]) -> list[tuple[str, str, str, str]]:
    """The Test tab's recordings, `(value, language, label, words)`: per language every
    heard take, newest first (D101). The label names the words, so takes can be told
    apart; the option carries them whole."""
    texts = []
    for code in languages:
        for take in reversed(voice.examples):
            if take.language == code and not take.deleting and take.words_state == "heard":
                words = take.transcript
                if len(words) > TAKE_LABEL_CHARS:
                    words = f"{words[:TAKE_LABEL_CHARS].rstrip()}…"
                texts.append((f"take:{take.id}", code, f"Take #{take.id} — {words}",
                              take.transcript))
    return texts


class VoiceDeletionDrain:
    """Runs `process_voice_deletions` on ONE background thread when no worker does (D33).

    `running` and `rerun` live under `lock`, never `Thread.is_alive()`: a `schedule()`
    that lands while the thread is finishing sets `rerun` under the same lock the thread
    takes to decide to stop, so it always gets a run.
    """

    def __init__(self, context: Callable[[], AppContext]) -> None:
        self._context = context
        self.lock = threading.Lock()
        self.running = False
        self.rerun = False
        self.thread: threading.Thread | None = None

    def schedule(self) -> None:
        """Make sure a drain runs soon. Returns at once; with a worker it does nothing."""
        if not self._context().settings.inline_drain:
            return
        with self.lock:
            if self.running:
                self.rerun = True
                return
            self.running = True
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def wake(self) -> None:
        """`schedule()` only when the outbox holds a row: an intent, a replaced blob, or
        the write-ahead row of a write that did not finish."""
        if pending_voice_deletions(self._context().conn) > 0:
            self.schedule()

    def join(self) -> None:
        """Test seam: wait until no drain thread runs."""
        while True:
            thread = self.thread
            if thread is None:
                return
            thread.join()
            if self.thread is thread:
                return

    def _run(self) -> None:
        # `ctx.conn` is thread-local (web/deps.py), so this thread uses its own.
        ctx = self._context()
        while True:
            try:
                left = process_voice_deletions(ctx.conn, ctx.voices, ctx.settings.owner_id)
            except Exception:  # never for one row (§2.7) — but the thread must not die
                logger.exception("voice deletion drain failed; retried in %s s", RETRY_SECONDS)
                left = -1
            if left != 0:
                time.sleep(RETRY_SECONDS)
                continue
            with self.lock:
                if self.rerun:
                    self.rerun = False
                    continue
                self.running = False
                return


class VoicePreviews:
    """The one voice test of this process (§2.16 E): the film's job pattern, no queue.

    `run` names the current test; `start` and `cancel` bump it, so an older thread is
    stale and its result is dropped. `lock` is also `run_preview`'s publication barrier:
    a test is stored and marked done under it, so it is never stored after the `cancel()`
    or `start()` that superseded it has returned. `epoch` tells this process from the one
    before a reload: an identity, never an order. `completed` = the run of the last test
    this instance stored (0 = none).
    """

    def __init__(
        self,
        context: Callable[[], AppContext],
        voice_lock: threading.Lock,
        drain: VoiceDeletionDrain,
    ) -> None:
        self._context = context
        self._voice_lock = voice_lock
        self._drain = drain
        self.lock = threading.Lock()
        self.run = 0
        self.epoch = secrets.token_hex(16)
        self.completed = 0
        self.state: dict = {"state": "idle", "run": 0}
        self.thread: threading.Thread | None = None
        self._threads: list[threading.Thread] = []

    def start(self, job: PreviewJob) -> None:
        """Start a test of `job`; a running one is replaced. Returns at once."""
        with self.lock:
            self.run += 1
            run = self.run
            self.state = {"state": "running", "run": run, "job": job, "stage": "starting",
                          "fraction": 0.0}
            self.thread = threading.Thread(target=self._run, args=(run, job), daemon=True)
            # A stale thread may still wait for its model call; `join` waits for it too.
            self._threads = [held for held in self._threads if held.is_alive()] + [self.thread]
            self.thread.start()

    def cancel(self) -> None:
        """End a running test for the page, or clear a finished one. Never waits for a
        model call; it may wait for a publication in progress."""
        with self.lock:
            self.run += 1
            self.state = {"state": "idle", "run": self.run}

    def join(self) -> None:
        """Test seam: wait until no test thread runs, stale ones included."""
        while True:
            with self.lock:
                alive = [held for held in self._threads if held.is_alive()]
            if not alive:
                return
            for held in alive:
                held.join()

    def snapshot(self) -> dict:
        with self.lock:
            state = dict(self.state)
            completed = self.completed
        job: PreviewJob | None = state.get("job")
        text = job.text.text if job is not None else ""
        return {
            "state": state["state"],
            "run": state["run"],
            "epoch": self.epoch,
            "completed": completed,
            "voice_id": job.voice_id if job is not None else None,
            "language": job.language if job is not None else None,
            "text": text if len(text) <= 80 else f"{text[:80]}…",
            "stage": state.get("stage", ""),
            "percent": percent(state.get("fraction")),
            "error": state.get("error"),
        }

    def _run(self, run: int, job: PreviewJob) -> None:
        # `ctx.conn` is thread-local (web/deps.py), so this thread uses its own.
        ctx = self._context()

        def progress(stage: str, fraction: float) -> None:
            with self.lock:
                if self.run == run:
                    self.state = self.state | {"stage": stage, "fraction": fraction}

        def on_saved(_preview) -> None:
            # `run_preview` holds `self.lock` here, and it is not re-entrant.
            self.completed = run
            self.state = {"state": "done", "run": run, "job": job}

        try:
            run_preview(
                ctx.conn, ctx.settings.build_models_client(), ctx.voices,
                ctx.settings.owner_id, job, voice_lock=self._voice_lock, job_lock=self.lock,
                on_saved=on_saved, progress=progress, cancelled=lambda: self.run != run,
            )
        except Cancelled:
            logger.info("voice test of voice %s cancelled", job.voice_id)
        except PreviewFailed as error:
            self._fail(run, job, str(error))
        except Exception as error:  # a failed test must never kill the app
            logger.exception("voice test of voice %s failed", job.voice_id)
            self._fail(run, job, f"The test failed ({_reason(error)}).")
        finally:
            # A replaced test WAV, a replaced blob or a write-ahead row waits in the outbox.
            try:
                self._drain.wake()
            except Exception:  # a cleanup must never break the job state
                logger.exception("waking the voice drain after a voice test failed")

    def _fail(self, run: int, job: PreviewJob, error: str) -> None:
        with self.lock:
            if self.run == run:
                self.state = {"state": "failed", "run": run, "job": job, "error": error}


class VoiceHearings:
    """The one hearing job of this process (§11.1): the words of every take that waits for
    them, one voice at a time, in the order their takes were saved.

    `queue` holds the voices still to hear. A voice is taken off it BEFORE it is heard, so
    a take saved while its voice is heard queues the voice again and is never left
    waiting. `current` is the voice heard now, with its stage; `passes` names each pass, and
    a Cancel of the voice heard now bumps it, so that pass stops before its next take and
    the job goes on with the next voice. `errors` keeps why the pass of a voice failed,
    until that voice is heard again or the failure is closed; a failure never drops the
    work of another voice. `working` is True exactly while the thread lives. Everything is
    guarded by `lock`.
    """

    def __init__(self, context: Callable[[], AppContext], voice_lock: threading.Lock) -> None:
        self._context = context
        self._voice_lock = voice_lock
        self.lock = threading.Lock()
        self.passes = 0
        self.queue: list[int] = []
        self.current: dict | None = None
        self.errors: dict[int, str] = {}
        self.working = False
        self._threads: list[threading.Thread] = []

    def start(self, voice_id: int) -> None:
        """Hear the waiting takes of `voice_id`, now or after the voices before it."""
        with self.lock:
            self.errors.pop(voice_id, None)
            if voice_id not in self.queue:
                self.queue.append(voice_id)
            if self.working:
                return
            self.working = True
            thread = threading.Thread(target=self._run, daemon=True)
            self._threads = [held for held in self._threads if held.is_alive()] + [thread]
            thread.start()

    def cancel(self, voice_id: int) -> None:
        """Stop the work of `voice_id` (after the take in hand, when it is heard now), or
        close its failure. The other voices are heard as before."""
        with self.lock:
            self.errors.pop(voice_id, None)
            if voice_id in self.queue:
                self.queue.remove(voice_id)
            if self.current is not None and self.current["voice_id"] == voice_id:
                self.passes += 1
                self.current = None

    def hears(self, voice_id: int) -> bool:
        """The job hears `voice_id` now or will hear it."""
        with self.lock:
            return self._hears_locked(voice_id)

    def _hears_locked(self, voice_id: int) -> bool:
        heard_now = self.current is not None and self.current["voice_id"] == voice_id
        return heard_now or voice_id in self.queue

    def view(self, voice_id: int) -> dict:
        """The job as the page of `voice_id` shows it: "running" (its takes are heard now),
        "queued" (after another voice), "failed" (with `error`) or "idle"."""
        with self.lock:
            current = self.current
            if current is not None and current["voice_id"] == voice_id:
                return {"state": "running", "run": current["pass"], "voice_id": voice_id,
                        "stage": current["stage"], "percent": percent(current["fraction"]),
                        "error": None}
            if voice_id in self.queue:
                state, error = "queued", None
            elif voice_id in self.errors:
                state, error = "failed", self.errors[voice_id]
            else:
                state, error = "idle", None
            return {"state": state, "run": self.passes, "voice_id": voice_id, "stage": "",
                    "percent": 0, "error": error}

    def join(self) -> None:
        """Test seam: wait until no hearing thread runs."""
        while True:
            with self.lock:
                alive = [held for held in self._threads if held.is_alive()]
            if not alive:
                return
            for held in alive:
                held.join()

    def _run(self) -> None:
        # `ctx.conn` is thread-local (web/deps.py), so this thread uses its own.
        ctx = self._context()
        while True:
            with self.lock:
                if not self.queue:
                    self.current = None
                    self.working = False
                    return
                voice_id = self.queue.pop(0)
                self.passes += 1
                this_pass = self.passes
                self.current = {"voice_id": voice_id, "pass": this_pass, "stage": "starting",
                                "fraction": 0.0}

            def progress(stage: str, fraction: float, this_pass: int = this_pass) -> None:
                with self.lock:
                    if self.passes == this_pass and self.current is not None:
                        self.current = self.current | {"stage": stage, "fraction": fraction}

            error = None
            try:
                hear_takes(ctx.conn, ctx.settings.build_models_client(), ctx.voices,
                           ctx.settings.owner_id, voice_id, voice_lock=self._voice_lock,
                           progress=progress,
                           cancelled=lambda this_pass=this_pass: self.passes != this_pass)
            except Cancelled:
                logger.info("hearing the words of voice %s cancelled", voice_id)
            except HearingFailed as failed:
                error = str(failed)
            except Exception as failed:  # a failed pass must never kill the app or the job
                logger.exception("hearing the words of voice %s failed", voice_id)
                error = f"Hearing the words failed ({_reason(failed)})."
            with self.lock:
                if self.passes == this_pass:
                    self.current = None
                    # A take saved during the failed pass queued its voice again: that new
                    # pass is the retry, so no failure is shown for it.
                    if error is not None and voice_id not in self.queue:
                        self.errors[voice_id] = error


def next_version(conn, owner_id: int, voice_id: int) -> int:
    """The version a bake of this voice stores: "Baking v<N+1>"."""
    found = voice_model_for(conn, owner_id, voice_id).model
    return (found.version if found is not None else 0) + 1


class VoiceBakeJobs:
    """The one bake of this process (§2.17 K4, D74): the `VoicePreviews` pattern.

    `run` names the current bake; `start`, `cancel` and a voice delete bump it, so an older
    thread is stale: `run_bake` reads it as cancelled and stores nothing. `lock` is
    `run_bake`'s job lock, the publication barrier: the hand-off to the run row and the
    store run under it, and Cancel, a replace and a voice delete take it to write the
    cancel request on the run row before they answer (D88). `on_started` and `on_saved`
    run while `run_bake` holds it, so they only set state (t48).

    Every new thread first joins the one before it, so a replacement starts only after the
    old run has deleted its bake. `reattach` finds this app's own bake again after an app
    restart; a bake that `app` did not start is never adopted, cancelled or deleted here
    (D83).
    """

    def __init__(
        self,
        context: Callable[[], AppContext],
        voice_lock: threading.Lock,
        drain: VoiceDeletionDrain,
    ) -> None:
        self._context = context
        self._voice_lock = voice_lock
        self._drain = drain
        self.lock = threading.Lock()
        self.run = 0
        self.state: dict = {"state": "idle", "run": 0}
        self.thread: threading.Thread | None = None
        self._threads: list[threading.Thread] = []
        self._reattach_lock = threading.Lock()
        self.reattached = False

    def snapshot(self) -> dict:
        with self.lock:
            return self._view_locked()

    def _view_locked(self) -> dict:
        state = self.state
        return {
            "state": state["state"],
            "run": state["run"],
            "voice_id": state.get("voice_id"),
            "version": state.get("version"),
            "stage": state.get("stage", ""),
            "percent": percent(state.get("fraction")),
            "error": state.get("error"),
            "text": state.get("text"),
            "question": state.get("question"),
            "name": state.get("name"),
        }

    def start(self, voice_id: int, version: int, *, attach: dict | None = None,
              wait_for_other: bool = False, only_if_idle: bool = False,
              replace: bool = False) -> dict | None:
        """Start a bake of `voice_id`, or adopt the running bake `attach`. Returns at once:
        None = started, else the running job's snapshot and nothing changed.

        The check and the start are one step under `lock`, so two starts never both pass
        it. A running bake is replaced only with `replace` (the user's yes, D74), and never
        by a bake of its own voice or with `only_if_idle`; its cancel request is written
        first (D88)."""
        with self.lock:
            if self.state["state"] == "running":
                if only_if_idle or not replace or self.state.get("voice_id") == voice_id:
                    return self._view_locked()
                self._request_cancel()
            self.run += 1
            run = self.run
            self.state = {"state": "running", "run": run, "voice_id": voice_id,
                          "version": version, "stage": "starting", "fraction": 0.0}
            self.thread = self._spawn_locked(self._run, run, voice_id, attach, wait_for_other,
                                             self.thread)
        return None

    def cancel(self, *, close: bool = False) -> None:
        """Cancel the running bake, or clear an ended one (`close`). Never waits for the
        models service; it may wait for a publication in progress, and when that stored the
        model first, the job says so: the store wins (D88)."""
        with self.lock:
            state = self.state
            if state["state"] == "running":
                self._request_cancel()
                self.run += 1
                self.state = {"state": "idle", "run": self.run}
            elif state["state"] == "done" and not close:
                self.state = state | {
                    "text": f"The bake already finished. Voice model v{state['version']} is stored."}
            else:
                self.state = {"state": "idle", "run": self.run}

    def cancel_voice(self, voice_id: int) -> None:
        """A voice delete: its running bake stores nothing (K4)."""
        with self.lock:
            if self.state["state"] == "running" and self.state.get("voice_id") == voice_id:
                self._request_cancel()
                self.run += 1
                self.state = {"state": "idle", "run": self.run}

    def join(self) -> None:
        """Test seam: wait until no bake, clean-up or reattach thread runs."""
        while True:
            with self.lock:
                alive = [held for held in self._threads if held.is_alive()]
            if not alive:
                return
            for held in alive:
                held.join()

    def reattach(self) -> None:
        """After an app restart, find this app's own bake in the models service again (D74,
        D83): only a bake tagged `source: "app"` of this owner. With its run row it is
        adopted, stored once or shown once; a row with a cancel request is cleaned up and
        stores nothing (D88); with no row it is cancelled if it runs and deleted. Done once
        per process, when the models service answers."""
        with self._reattach_lock:
            if self.reattached:
                return
            ctx = self._context()
            owner_id = ctx.settings.owner_id
            try:
                client = ctx.settings.build_models_client()
                current = client.bake_current()
            except Exception as error:  # noqa: BLE001 — the service is down: the next idle read tries again
                logger.warning("voice bake reattach: no answer from the models service (%s)",
                               _reason(error))
                return
            self.reattached = True
            if current is None or not _is_app_bake(current, owner_id):
                return
            bake_key = current["tag"].get("bake_key")
            run = read_bake_run(ctx.conn, owner_id, bake_key) if bake_key else None
            if run is None:
                # The run ended — stored, dropped or never written: nothing is stored.
                self._spawn(self._clean_up, client, current)
            elif run.cancel_requested_at is not None:
                self._spawn(self._drop, client, run.voice_id, current)
            elif self.start(run.voice_id, next_version(ctx.conn, owner_id, run.voice_id),
                            attach=current, only_if_idle=True) is None:
                logger.info("voice bake %s of voice %s adopted after a restart", current["id"],
                            run.voice_id)

    def reattach_later(self) -> None:
        self._spawn(self.reattach)

    def _spawn_locked(self, target, *args) -> threading.Thread:
        thread = threading.Thread(target=target, args=args, daemon=True)
        self._threads = [held for held in self._threads if held.is_alive()] + [thread]
        thread.start()
        return thread

    def _spawn(self, target, *args) -> threading.Thread:
        with self.lock:
            return self._spawn_locked(target, *args)

    def _request_cancel(self) -> None:
        # Under `lock`, before the reply: the request outlives an app restart (D88). With
        # no run row yet (the transcription stage) the logical cancel is all there is.
        bake_key = self.state.get("bake_key")
        if bake_key is not None and not request_bake_cancel(self._context().conn, bake_key):
            logger.info("voice bake run %s had already ended", bake_key)

    def _end(self, run: int, voice_id: int, fields: dict) -> None:
        with self.lock:
            if self.run == run:
                self.state = {"run": run, "voice_id": voice_id} | fields

    def _run(self, run: int, voice_id: int, attach: dict | None, wait_for_other: bool,
             previous: threading.Thread | None) -> None:
        if previous is not None:
            previous.join()   # the old run has deleted its bake (K4)
        # `ctx.conn` is thread-local (web/deps.py), so this thread uses its own.
        ctx = self._context()

        def progress(stage: str, fraction: float) -> None:
            with self.lock:
                if self.run == run:
                    self.state = self.state | {"stage": stage, "fraction": fraction}

        def on_started(bake_key: str) -> None:
            # `run_bake` holds `self.lock` here, and it is not re-entrant (t48).
            if self.run == run:
                self.state = self.state | {"bake_key": bake_key}

        def on_saved(model, text: str) -> None:
            if self.run == run:
                self.state = {"state": "done", "run": run, "voice_id": voice_id,
                              "version": model.version, "text": text}

        try:
            run_bake(
                ctx.conn, ctx.settings.build_models_client(), ctx.voices, ctx.settings.owner_id,
                voice_id, voice_lock=self._voice_lock, job_lock=self.lock, progress=progress,
                cancelled=lambda: self.run != run, attach=attach, wait_for_other=wait_for_other,
                on_started=on_started, on_saved=on_saved,
            )
        except Cancelled:
            # A cancel of this process bumped the run, so this shows only for a bake that
            # ended cancelled on its own (an adopted one).
            logger.info("voice bake of voice %s cancelled", voice_id)
            self._end(run, voice_id, {"state": "failed", "error": BAKE_CANCELLED})
        except BakeBusy as busy:
            # The bake check started a bake meanwhile: ask, never a dead end (D83).
            status = busy.status
            if status is not None and status.get("state") == "running" \
                    and not _is_app_bake(status, ctx.settings.owner_id):
                voice = get_voice(ctx.conn, ctx.settings.owner_id, voice_id)
                name = voice.name if voice is not None else ""
                self._end(run, voice_id, {"state": "question", "name": name,
                                          "question": foreign_bake_question(status, name)})
            else:
                self._end(run, voice_id, {"state": "failed", "error": str(busy)})
        except VoiceError as error:   # BakeBlocked, BakeFailed, VoiceDeleting: the page shows why
            self._end(run, voice_id, {"state": "failed", "error": str(error)})
        except Exception as error:  # a failed bake must never kill the app
            logger.exception("voice bake of voice %s failed", voice_id)
            self._end(run, voice_id, {"state": "failed",
                                      "error": f"The bake failed ({_reason(error)})."})
        finally:
            # A replaced voice model file waits in the outbox.
            try:
                self._drain.wake()
            except Exception:  # a cleanup must never break the job state
                logger.exception("waking the voice drain after a voice bake failed")

    def _clean_up(self, client, status: dict) -> None:
        """A bake of this app with no run row: cancel it if it runs, wait for ANY end,
        delete it. Nothing is stored."""
        bake_id = status["id"]
        try:
            if status["state"] == "running":
                status = client.bake_cancel(bake_id)
            while status["state"] == "running":
                time.sleep(voice_bake.POLL_SECONDS)
                status = client.bake_status(bake_id)
            client.bake_delete(bake_id)
        except Exception:  # best effort: the service sweeps its folder at its next start
            logger.exception("cleaning up voice bake %s failed", bake_id)

    def _drop(self, client, voice_id: int, status: dict) -> None:
        """A run with a cancel request (D88): `run_bake` cancels the bake if it runs,
        waits for any end, deletes it, ends the run row and stores nothing."""
        ctx = self._context()
        try:
            run_bake(ctx.conn, client, ctx.voices, ctx.settings.owner_id, voice_id,
                     voice_lock=self._voice_lock, job_lock=self.lock, attach=status)
        except Cancelled:
            logger.info("voice bake %s was cancelled before the restart; nothing stored",
                        status["id"])
        except Exception:  # the run is ended in run_bake's own finally
            logger.exception("dropping cancelled voice bake %s failed", status["id"])


def register(
    app: FastAPI,
    context: Callable[[], AppContext],
    templates: Jinja2Templates,
) -> None:
    drain = VoiceDeletionDrain(context)
    app.state.voice_drain = drain
    app.state.await_voice_drain = drain.join
    tests = VoicePreviews(context, app.state.voice_calls, drain)
    app.state.voice_previews = tests
    app.state.await_voice_preview = tests.join
    bakes = VoiceBakeJobs(context, app.state.voice_calls, drain)
    app.state.voice_bakes = bakes
    app.state.await_voice_bake = bakes.join
    hearings = VoiceHearings(context, app.state.voice_calls)
    app.state.voice_hearings = hearings
    app.state.await_voice_hearing = hearings.join
    templates.env.filters["clock"] = clock

    def _cards_context(ctx: AppContext, created: int | None = None) -> dict:
        voices = list_voices(ctx.conn, ctx.settings.owner_id)
        return {
            "voices": voices,
            "ready": {voice.id: [language_name(code) for code in LANGUAGES
                                 if film_take(ctx.conn, ctx.settings.owner_id, voice.id, code).take]
                      for voice in voices if not voice.deleting},
            "all_languages": [(code, language_name(code)) for code in LANGUAGES],
            "created": created,
        }

    def _cards(ctx: AppContext, created: int | None = None) -> str:
        return templates.get_template("_voice_cards.html").render(_cards_context(ctx, created))

    def _preview_context(ctx: AppContext, preview: dict) -> dict:
        voice = (get_voice(ctx.conn, ctx.settings.owner_id, preview["voice_id"])
                 if preview["voice_id"] is not None else None)
        return {"preview": preview, "language_names": LANGUAGE_NAMES,
                "preview_voice": voice.name if voice is not None else ""}

    def _voice_context(ctx: AppContext, voice: SavedVoice, languages: tuple[str, ...]) -> dict:
        """The header and the Test tab of one voice (§11.1): per language the take films
        and tests speak from, or why there is none."""
        owner_id = ctx.settings.owner_id
        takes = [take for take in voice.examples if not take.deleting]
        found = {code: film_take(ctx.conn, owner_id, voice.id, code) for code in LANGUAGES}
        ready_lines = []
        for code in LANGUAGES:
            of_language = [take for take in reversed(takes) if take.language == code]
            if not of_language and code not in languages:
                continue
            name, take = language_name(code), found[code].take
            if take is None:
                ready_lines.append((f"{name} · {READY_PROBLEMS[found[code].problem]}", False))
                continue
            line = f"{name} · films and tests speak from take #{take.id}"
            newest = of_language[0]
            if newest.id != take.id and newest.words_state == "waiting":
                line = f"{line} · take #{newest.id} waits for its words"
            elif newest.id != take.id:
                line = f"{line} · take #{newest.id} has no usable words"
            ready_lines.append((line, True))
        tested = [code for code in languages if found[code].take is not None]
        test = voice_test(ctx.conn, owner_id, voice.id)
        original = None
        if test is not None and test.example_id is not None:
            original = next((take for take in takes if take.id == test.example_id), None)
        now = found[test.language].take if test is not None and test.language in found else None
        return {
            "ready_lines": ready_lines,
            "waiting_takes": any(take.words_state == "waiting" for take in takes),
            "hearing_here": hearings.hears(voice.id),
            "hear_offered": transcriber_for(ctx.settings.profile) is not None,
            "hear_not_offered": HEAR_NOT_OFFERED,
            "test_ready": bool(tested),
            "test_languages": [(code, language_name(code), found[code].take.id) for code in tested],
            "test_texts": _test_texts(voice, tested),
            "test": test,
            "test_take_now": now.id if now is not None else None,
            "test_older_takes": (test is not None and test.likeness is not None
                                 and measured_older_takes(test.likeness, voice, test.language)),
            "original": original,
        }

    def _leaf_context(ctx: AppContext, voice_id: int, tab: str = "recordings") -> dict:
        # Read before the voice: a test stored meanwhile makes the body newer than its
        # certificate, and the page then fetches it again — never the other way round.
        preview = tests.snapshot()
        hearing = hearings.view(voice_id)
        voice = get_voice(ctx.conn, ctx.settings.owner_id, voice_id)
        if voice is None:
            raise HTTPException(status_code=404, detail="Unknown saved voice.")
        languages = cloning_languages(ctx.settings.profile)
        return _preview_context(ctx, preview) | {
            "voice": voice,
            "hearing": hearing,
            "tab": tab,
            "tabs": TABS,
            "available": bool(languages),
            "not_available": NOT_AVAILABLE,
            "languages": [(code, language_name(code)) for code in languages],
            "recordings": _recordings(voice, languages),
        } | ({} if voice.deleting else _voice_context(ctx, voice, languages))

    def _body(ctx: AppContext, voice_id: int) -> str:
        return templates.get_template("_voice_body.html").render(_leaf_context(ctx, voice_id))

    def _preview_fragment() -> str:
        return templates.get_template("_voice_preview.html").render(
            _preview_context(context(), tests.snapshot())
        )

    def _hearing_fragment(voice_id: int) -> str:
        return templates.get_template("_voice_hearing.html").render({"hearing": hearings.view(voice_id)})

    def _voice_or_404(ctx: AppContext, voice_id: int):
        voice = get_voice(ctx.conn, ctx.settings.owner_id, voice_id)
        if voice is None:
            raise HTTPException(status_code=404, detail="Unknown saved voice.")
        return voice

    def _refuse(error: VoiceError) -> HTTPException:
        status = 409 if isinstance(error, VoiceNameTaken | VoiceDeleting) else 400
        return HTTPException(status_code=status, detail=str(error))

    @app.get("/organize/voices")
    def voices_moved() -> RedirectResponse:
        # Old bookmarks of the page that lived under Organize (D20 reversed, D67).
        return RedirectResponse("/voices", status_code=308)

    # The literal paths come before /voices/{voice_id}: `voice_id` is an int, so a later
    # literal path would answer 422 (§2.17 V1).
    @app.get("/voices", response_class=HTMLResponse)
    def voices_page(request: Request) -> HTMLResponse:
        ctx = context()
        return templates.TemplateResponse(
            request,
            "voices.html",
            _cards_context(ctx) | {
                "available": bool(cloning_languages(ctx.settings.profile)),
                "not_available": NOT_AVAILABLE,
            },
        )

    @app.get("/voices/list", response_class=HTMLResponse)
    def voices_list() -> HTMLResponse:
        return HTMLResponse(_cards(context()))

    @app.post("/voices", response_class=HTMLResponse)
    def voices_create(name: str = Form(default="")) -> HTMLResponse:
        ctx = context()
        if not cloning_languages(ctx.settings.profile):
            raise HTTPException(status_code=409, detail=NOT_AVAILABLE)
        try:
            voice = create_voice(ctx.conn, ctx.settings.owner_id, name)
        except VoiceError as error:
            raise _refuse(error) from error
        return HTMLResponse(_cards(ctx, created=voice.id))

    @app.get("/voices/preview", response_class=HTMLResponse)
    def voices_preview_state() -> HTMLResponse:
        # Polled every 1 s while a test runs, and the page's reconciliation read: a
        # cached copy would show a test of another run or another process.
        return HTMLResponse(_preview_fragment(), headers={"Cache-Control": "no-store"})

    @app.post("/voices/preview/cancel", response_class=HTMLResponse)
    def voices_preview_cancel() -> HTMLResponse:
        tests.cancel()
        return HTMLResponse(_preview_fragment())

    @app.get("/voices/hearing", response_class=HTMLResponse)
    def voices_hearing_state(voice: int = 0) -> HTMLResponse:
        # Polled every 1 s while the takes of the open voice `voice` are heard or queued.
        return HTMLResponse(_hearing_fragment(voice), headers={"Cache-Control": "no-store"})

    @app.post("/voices/hearing/cancel", response_class=HTMLResponse)
    def voices_hearing_cancel(voice: int = 0) -> HTMLResponse:
        hearings.cancel(voice)
        return HTMLResponse(_hearing_fragment(voice))

    @app.get("/voices/{voice_id}", response_class=HTMLResponse)
    def voice_page(request: Request, voice_id: int, tab: str = "recordings") -> HTMLResponse:
        known = {key for key, _ in TABS}
        return templates.TemplateResponse(
            request,
            "voice.html",
            _leaf_context(context(), voice_id, tab if tab in known else "recordings") | {
                "kit": RECORDING_KIT,
                "styles": KIT_STYLES,
                "custom_max": CUSTOM_TEXT_MAX,
            },
        )

    @app.get("/voices/{voice_id}/list", response_class=HTMLResponse)
    def voice_body(voice_id: int) -> HTMLResponse:
        return HTMLResponse(_body(context(), voice_id))

    @app.post("/voices/{voice_id}/preview", response_class=HTMLResponse)
    def voices_preview_start(
        voice_id: int,
        language: str = Form(default=""),
        text_kind: str = Form(default=""),
        text_id: str = Form(default=""),
        text: str = Form(default=""),
    ) -> HTMLResponse:
        # Every refusal comes at once, so Generate never starts a test that must fail (K5).
        ctx = context()
        voice = _voice_or_404(ctx, voice_id)
        if voice.deleting:
            raise HTTPException(status_code=409, detail="This voice is being deleted.")
        entry = cloning_entry_for(ctx.settings.profile, language)
        if entry is None:
            raise HTTPException(status_code=409, detail=NOT_AVAILABLE)
        problem = film_take(ctx.conn, ctx.settings.owner_id, voice.id, language).problem
        if problem is not None:
            raise HTTPException(status_code=409, detail=_cannot_test(problem, language))
        free = ctx.voices.free_bytes()
        if free < FREE_SPACE_FLOOR:
            raise HTTPException(
                status_code=507, detail=f"only {free // (1024 * 1024)} MB free on the server"
            )
        try:
            words = preview_text(voice, language, text_kind, text_id, text)
        except VoiceError as error:
            raise _refuse(error) from error
        tests.start(PreviewJob(voice.id, language, words, entry.source.revision))
        return HTMLResponse(_preview_fragment())

    @app.get("/voices/{voice_id}/preview/audio")
    def voices_preview_audio(voice_id: int) -> Response:
        ctx = context()
        _voice_or_404(ctx, voice_id)
        test = voice_test(ctx.conn, ctx.settings.owner_id, voice_id)
        if test is None:
            raise HTTPException(status_code=404, detail="This voice has no test yet.")
        try:
            data = read_test_audio(ctx.voices, test)
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail="This voice has no test yet.") from error
        return Response(data, media_type="audio/wav")

    @app.post("/voices/{voice_id}/hear", response_class=HTMLResponse)
    def voices_hear(voice_id: int) -> HTMLResponse:
        ctx = context()
        voice = _voice_or_404(ctx, voice_id)
        if voice.deleting:
            raise HTTPException(status_code=409, detail="This voice is being deleted.")
        if transcriber_for(ctx.settings.profile) is None:
            raise HTTPException(status_code=409, detail=HEAR_NOT_OFFERED)
        hearings.start(voice_id)
        return HTMLResponse(_hearing_fragment(voice_id))

    @app.post("/voices/{voice_id}/rename", response_class=HTMLResponse)
    def voices_rename(voice_id: int, name: str = Form(default="")) -> HTMLResponse:
        # Allowed on every profile: the name is the user's, not the model's.
        ctx = context()
        _voice_or_404(ctx, voice_id)
        try:
            rename_voice(ctx.conn, ctx.settings.owner_id, voice_id, name)
        except VoiceError as error:
            raise _refuse(error) from error
        return HTMLResponse(_body(ctx, voice_id))

    @app.post("/voices/{voice_id}/delete", response_class=HTMLResponse)
    def voices_delete(voice_id: int) -> HTMLResponse:
        # Enqueue, render, THEN schedule: the reply always shows "deleting…", however
        # fast the drain is. No drain runs inside the request (D33).
        ctx = context()
        if not request_voice_deletion(ctx.conn, ctx.settings.owner_id, voice_id):
            raise HTTPException(status_code=404, detail="Unknown saved voice.")
        # Its running bake stores nothing: the cancel request is written before the reply.
        bakes.cancel_voice(voice_id)
        body = _body(ctx, voice_id)
        drain.schedule()
        return HTMLResponse(body)

    @app.post("/voices/{voice_id}/examples", response_class=HTMLResponse)
    def voices_add_example(
        voice_id: int,
        language: str = Form(default=""),
        source: str = Form(default=""),
        prompt_id: str = Form(default=""),
        file: UploadFile = File(...),  # noqa: B008 - FastAPI's parameter-in-default idiom
    ) -> HTMLResponse:
        ctx = context()
        try:
            _voice_or_404(ctx, voice_id)
            languages = cloning_languages(ctx.settings.profile)
            if not languages or (language in LANGUAGES and language not in languages):
                raise HTTPException(status_code=409, detail=NOT_AVAILABLE)
            free = ctx.voices.free_bytes()
            if free < FREE_SPACE_FLOOR:
                raise HTTPException(
                    status_code=507, detail=f"only {free // (1024 * 1024)} MB free on the server"
                )
            data = file.file.read(MAX_BYTES + 1)
            if len(data) > MAX_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"The example is larger than {MAX_BYTES // (1024 * 1024)} MiB.",
                )
            try:
                add_example(
                    ctx.conn, ctx.voices, ctx.settings.owner_id, voice_id,
                    language=language, source=source, data=data,
                    prompt_id=prompt_id or None,
                )
            except VoiceError as error:
                raise _refuse(error) from error
            if transcriber_for(ctx.settings.profile) is not None:
                hearings.start(voice_id)
            return HTMLResponse(_body(ctx, voice_id))
        finally:
            # A refused or failed write can leave a write-ahead row (§2.7); it must not
            # wait for the next delete or restart.
            try:
                drain.wake()
            except Exception:
                logger.exception("waking the voice drain after adding an example failed")

    @app.post(
        "/voices/{voice_id}/examples/{example_id}/delete", response_class=HTMLResponse
    )
    def voices_delete_example(voice_id: int, example_id: int) -> HTMLResponse:
        ctx = context()
        if not request_example_deletion(ctx.conn, ctx.settings.owner_id, voice_id, example_id):
            raise HTTPException(status_code=404, detail="Unknown example.")
        body = _body(ctx, voice_id)
        drain.schedule()
        return HTMLResponse(body)

    @app.get("/voices/{voice_id}/examples/{example_id}/audio")
    def voices_example_audio(voice_id: int, example_id: int) -> Response:
        ctx = context()
        voice = _voice_or_404(ctx, voice_id)
        example = next((e for e in voice.examples if e.id == example_id), None)
        if example is None or example.deleting:
            raise HTTPException(status_code=404, detail="Unknown example.")
        try:
            data = read_example_audio(ctx.voices, example)
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail="Unknown example.") from error
        return Response(data, media_type="audio/wav")

    # The start-up run: intents left by a stopped process drain without a request.
    drain.wake()
    # A bake of this app may still run in the models service after a reload (D74). Off the
    # start-up path: the call waits for the service.
    if adapter_entry_for(context().settings.profile) is not None:
        bakes.reattach_later()
