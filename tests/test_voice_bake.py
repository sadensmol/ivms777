"""The bake job of one saved voice in `app` (design §11.1 "Baking a saved voice", PLAN
§2.17 K4) — a fake models service, a temp DB made by `migrate`, a temp voices store."""

import hashlib
import json
import threading
import time
from contextlib import nullcontext

import httpx
import pytest

from db.connection import connect
from feed import voice_bake
from feed.build import Cancelled
from feed.saved_voices import (
    NO_WORDS,
    TOO_LONG,
    add_example,
    create_voice,
    get_voice,
    process_voice_deletions,
    request_bake_cancel,
    request_example_deletion,
    request_voice_deletion,
    set_heard_words,
    voice_model_for,
)
from feed.voice_bake import (
    NO_TAKE,
    NOT_OFFERED,
    RESTARTED,
    VOICE_GONE,
    BakeBlocked,
    BakeBusy,
    BakeFailed,
    run_bake,
)
from models import catalog
from tests.test_feed_build import ADAPTER, clip
from tests.test_voice_preview import HookedStorage

OWNER = 1
BAKE_ID = "bake-1"
ASR_MODEL = "whisper-large-v3-turbo"
ASR_REVISION = "41f01f3"
HEARD_BY = f"{ASR_MODEL}@{ASR_REVISION}"
SERVICE_STAGES = (("loading the voice model for training", 0.05), ("training", 0.50),
                  ("saving the voice model", 0.95))
FOREIGN = {"id": "check-1", "state": "running", "stage": "training", "fraction": 0.4,
           "tag": {"check": "me"}}
JOIN_TIMEOUT = 10.0
BLOCKED_WAIT = 0.3


def http_error(code: int, detail: str) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://models/voice")
    return httpx.HTTPStatusError(str(code), request=request,
                                 response=httpx.Response(code, request=request, json={"detail": detail}))


def service_dataset(clips: list[dict]) -> str:
    """`dataset_sha256` computed the way the models service does (modelsvc/voice_bakes.py)."""
    lines = sorted(f"{c['sha256']}\t{c['language']}\t{c['text'].strip()}" for c in clips)
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


class FakeService:
    """The models service as the bake job sees it. `words` = heard text per sha256;
    `currents` = the answers of `bake_current`, one per call, then None; `on_*` hooks run
    inside a call; each call notes whether `job_lock` or `voice_lock` was held."""

    def __init__(self, *, words: dict | None = None, currents: list | None = None,
                 end: str = "done", end_after_cancel: str = "cancelled",
                 start_refusals: int = 0, job_lock=None, voice_lock=None) -> None:
        self.words = words or {}
        self.currents = list(currents or [])
        self.end, self.end_after_cancel = end, end_after_cancel
        self.start_refusals = start_refusals
        self.job_lock, self.voice_lock = job_lock, voice_lock
        self.transcribed: list[tuple[str, str]] = []
        self.starts: list[tuple[list[dict], dict]] = []
        self.cancels: list[str] = []
        self.deletes: list[str] = []
        self.current_calls = 0
        self.polls_after_cancel = 0
        self.lock_held_in_call = False
        self.voice_lock_free_in_call = False
        self.transcribe_error = None
        self.gone = False
        self.adapter = ADAPTER
        self.on_transcribe = None
        self.on_status = None
        self.dataset = ""
        self.tag: dict = {}
        self.step = 0

    def _note_locks(self) -> None:
        if self.job_lock is not None and self.job_lock.locked():
            self.lock_held_in_call = True
        if self.voice_lock is not None and not self.voice_lock.locked():
            self.voice_lock_free_in_call = True

    def transcribe(self, audio: bytes, language: str) -> dict:
        self._note_locks()
        sha = hashlib.sha256(audio).hexdigest()
        self.transcribed.append((sha, language))
        if self.on_transcribe is not None:
            self.on_transcribe(len(self.transcribed))
        if self.transcribe_error is not None and len(self.transcribed) == self.transcribe_error[0]:
            raise self.transcribe_error[1]
        return {"text": self.words.get(sha, f"Words of {sha[:6]}."), "model": ASR_MODEL,
                "revision": ASR_REVISION, "seconds": 6.0, "elapsed_s": 0.1}

    def bake_current(self):
        self.current_calls += 1
        return self.currents.pop(0) if self.currents else None

    def _status(self, state: str, stage: str, fraction: float, error: str | None = None) -> dict:
        entry = catalog.get("voice", "voxcpm2")
        return {"id": BAKE_ID, "state": state, "stage": stage, "fraction": fraction,
                "step": 3, "steps": 6, "error": error,
                "adapter": {"sha256": hashlib.sha256(self.adapter).hexdigest()} if state == "done" else None,
                "model": "voxcpm2", "revision": entry.source.revision,
                "lora": json.dumps(entry.adapter.lora, sort_keys=True), "config": {"steps": 6},
                "dataset_sha256": self.dataset, "tag": self.tag}

    def bake_start(self, clips: list[dict], *, tag: dict, config=None) -> dict:
        self._note_locks()
        if self.start_refusals:
            self.start_refusals -= 1
            raise http_error(409, "a voice is already being baked")
        for c in clips:
            assert hashlib.sha256(c["audio"]).hexdigest() == c["sha256"]
        self.starts.append((clips, tag))
        self.dataset, self.tag = service_dataset(clips), tag
        return self._status("running", "preparing the takes", 0.02)

    def bake_status(self, bake_id: str) -> dict:
        self._note_locks()
        if self.gone:
            raise http_error(404, "no such bake")
        if self.on_status is not None:
            self.on_status(self.step)
        if self.cancels:
            self.polls_after_cancel += 1
            state = self.end_after_cancel
            return self._status(state, "training", 0.5, "boom" if state == "failed" else None)
        if self.step < len(SERVICE_STAGES):
            stage, fraction = SERVICE_STAGES[self.step]
            self.step += 1
            return self._status("running", stage, fraction)
        return self._status(self.end, "done", 1.0, "boom" if self.end == "failed" else None)

    def bake_cancel(self, bake_id: str) -> dict:
        self.cancels.append(bake_id)
        return self._status("running", "training", 0.5)

    def bake_adapter(self, bake_id: str) -> bytes:
        return self.adapter

    def bake_delete(self, bake_id: str) -> None:
        self.deletes.append(bake_id)
        if self.gone:
            raise http_error(404, "no such bake")


@pytest.fixture(autouse=True)
def no_poll_wait(monkeypatch):
    monkeypatch.setattr(voice_bake, "POLL_SECONDS", 0.0)


@pytest.fixture
def voices(tmp_path):
    return HookedStorage(tmp_path / "voices")


def a_voice(conn, voices, takes: int = 2, *, language: str = "en"):
    voice = create_voice(conn, OWNER, "Mum")
    for n in range(takes):
        add_example(conn, voices, OWNER, voice.id, language=language, source="uploaded",
                    data=clip(200.0 + 10 * n))
    return get_voice(conn, OWNER, voice.id)


def add_take(conn, voices, voice_id: int, freq: float, language: str = "en"):
    return add_example(conn, voices, OWNER, voice_id, language=language, source="uploaded",
                       data=clip(freq))


def sha_of(take) -> str:
    return take.content_hash


def runs(conn) -> list[dict]:
    return [dict(r) for r in conn.execute("SELECT * FROM voice_bake_runs")]


def models(conn) -> list[dict]:
    return [dict(r) for r in conn.execute("SELECT * FROM voice_models")]


def model_takes(conn) -> list[int]:
    return [r["example_id"] for r in conn.execute("SELECT example_id FROM voice_model_takes ORDER BY example_id")]


def take_row(conn, example_id: int) -> dict:
    return dict(conn.execute("SELECT * FROM voice_examples WHERE id = ?", (example_id,)).fetchone())


def in_thread(settings, client, voices, voice_id, **kwargs) -> tuple[threading.Thread, dict]:
    outcome: dict = {}

    def run() -> None:
        own = connect(settings.db_path)
        try:
            outcome["result"] = run_bake(own, client, voices, OWNER, voice_id, **kwargs)
        except Exception as exc:  # noqa: BLE001 - the test asserts on it
            outcome["error"] = exc
        finally:
            own.close()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, outcome


def joined(thread: threading.Thread) -> None:
    thread.join(timeout=JOIN_TIMEOUT)
    assert not thread.is_alive()


def test_every_waiting_take_is_heard_once_by_id_and_its_words_are_stored(conn, voices):
    voice = a_voice(conn, voices, 3)
    first, second, third = voice.examples
    set_heard_words(conn, OWNER, voice.id, second.id, text="Already heard.", model="old@1")
    client = FakeService()

    run_bake(conn, client, voices, OWNER, voice.id)

    assert client.transcribed == [(sha_of(first), "en"), (sha_of(third), "en")]
    assert take_row(conn, first.id)["transcript"] == f"Words of {sha_of(first)[:6]}."
    assert take_row(conn, first.id)["transcript_model"] == HEARD_BY
    assert take_row(conn, second.id)["transcript"] == "Already heard."


def test_an_unusable_take_is_never_sent_again(conn, voices):
    voice = a_voice(conn, voices, 2)
    set_heard_words(conn, OWNER, voice.id, voice.examples[0].id, text="♪", model="old@1")
    client = FakeService()

    run_bake(conn, client, voices, OWNER, voice.id)

    assert client.transcribed == [(sha_of(voice.examples[1]), "en")]


def test_a_take_heard_as_no_words_is_left_out_and_named_in_the_done_text(conn, voices):
    voice = a_voice(conn, voices, 2)
    music, speech = voice.examples
    client = FakeService(words={sha_of(music): "♪ ♪"})
    done: list[str] = []

    model = run_bake(conn, client, voices, OWNER, voice.id, on_saved=lambda m, text: done.append(text))

    assert take_row(conn, music.id)["transcript_error"] == NO_WORDS
    assert [c["sha256"] for c in client.starts[0][0]] == [sha_of(speech)]
    assert model_takes(conn) == [speech.id]
    assert done == [f"Voice model v1 is stored. Not used: take #{music.id} — {NO_WORDS}."]
    assert model.version == 1


def test_a_bake_with_every_take_usable_says_only_that_the_model_is_stored(conn, voices):
    voice = a_voice(conn, voices, 1)
    done: list[str] = []

    run_bake(conn, FakeService(), voices, OWNER, voice.id, on_saved=lambda m, text: done.append(text))

    assert done == ["Voice model v1 is stored."]


def test_every_take_heard_as_unusable_blocks_with_each_reason(conn, voices):
    voice = a_voice(conn, voices, 2)
    short, long = voice.examples
    client = FakeService(words={sha_of(short): "...", sha_of(long): "a" * 1001})

    with pytest.raises(BakeBlocked) as raised:
        run_bake(conn, client, voices, OWNER, voice.id)

    assert str(raised.value) == (f"No take can be learned from: take #{short.id} — {NO_WORDS}; "
                                 f"take #{long.id} — {TOO_LONG}. Record or upload a take where you speak.")
    assert runs(conn) == [] and client.starts == []
    assert take_row(conn, short.id)["transcript_error"] == NO_WORDS


def test_takes_already_unusable_block_before_the_voice_lock(conn, voices):
    voice = a_voice(conn, voices, 1)
    set_heard_words(conn, OWNER, voice.id, voice.examples[0].id, text="♪", model="old@1")
    lock = threading.Lock()
    lock.acquire()
    client = FakeService()

    with pytest.raises(BakeBlocked, match="No take can be learned from: take #"):
        run_bake(conn, client, voices, OWNER, voice.id, voice_lock=lock)

    assert client.current_calls == 0 and client.transcribed == []


def test_no_take_at_all_is_blocked_before_the_voice_lock(conn, voices):
    voice = a_voice(conn, voices, 0)
    lock = threading.Lock()
    lock.acquire()
    client = FakeService()

    with pytest.raises(BakeBlocked) as raised:
        run_bake(conn, client, voices, OWNER, voice.id, voice_lock=lock)

    assert str(raised.value) == NO_TAKE
    assert client.current_calls == 0


def test_cancel_before_the_second_call_keeps_the_first_words_and_starts_nothing(conn, voices):
    voice = a_voice(conn, voices, 2)
    client = FakeService()
    stop = {"now": False}
    client.on_transcribe = lambda n: stop.update(now=True)

    with pytest.raises(Cancelled):
        run_bake(conn, client, voices, OWNER, voice.id, cancelled=lambda: stop["now"])

    assert len(client.transcribed) == 1
    assert take_row(conn, voice.examples[0].id)["transcript_model"] == HEARD_BY
    assert take_row(conn, voice.examples[1].id)["transcript_model"] is None
    assert runs(conn) == [] and client.starts == []


def test_a_409_from_transcribe_is_busy_with_the_current_bake(conn, voices):
    voice = a_voice(conn, voices, 1)
    client = FakeService(currents=[None, FOREIGN])
    client.transcribe_error = (1, http_error(409, "a voice is being baked"))

    with pytest.raises(BakeBusy) as raised:
        run_bake(conn, client, voices, OWNER, voice.id)

    assert raised.value.status == FOREIGN
    assert take_row(conn, voice.examples[0].id)["transcript_model"] is None


def test_a_foreign_bake_before_the_stage_is_busy_and_hears_nothing(conn, voices):
    voice = a_voice(conn, voices, 1)
    client = FakeService(currents=[FOREIGN])

    with pytest.raises(BakeBusy) as raised:
        run_bake(conn, client, voices, OWNER, voice.id)

    assert raised.value.status == FOREIGN
    assert client.transcribed == [] and client.cancels == [] and client.deletes == []


def test_wait_for_other_waits_for_the_foreign_bake_then_hears_and_bakes(conn, voices):
    voice = a_voice(conn, voices, 1)
    ended = FOREIGN | {"state": "cancelled"}
    client = FakeService(currents=[FOREIGN, FOREIGN, ended])
    stages: list[str] = []

    run_bake(conn, client, voices, OWNER, voice.id, wait_for_other=True,
             progress=lambda stage, fraction: stages.append(stage))

    assert stages[0] == "waiting for the other bake to stop"
    assert stages.index("transcribing the takes · take 1 of 1") > 0
    assert client.current_calls == 3 and len(client.starts) == 1
    assert "check-1" not in client.deletes and client.cancels == []


def test_a_503_from_transcribe_says_hearing_is_not_offered(conn, voices):
    voice = a_voice(conn, voices, 1)
    client = FakeService()
    client.transcribe_error = (1, http_error(503, "not offered"))

    with pytest.raises(BakeFailed) as raised:
        run_bake(conn, client, voices, OWNER, voice.id)

    assert str(raised.value) == NOT_OFFERED


def test_another_transcribe_error_names_the_take_and_keeps_earlier_words(conn, voices):
    voice = a_voice(conn, voices, 2)
    first, second = voice.examples
    client = FakeService()
    client.transcribe_error = (2, http_error(500, "Whisper weights are not on disk"))

    with pytest.raises(BakeFailed) as raised:
        run_bake(conn, client, voices, OWNER, voice.id)

    assert str(raised.value) == f"Could not hear the words of take #{second.id} (Whisper weights are not on disk)."
    assert take_row(conn, first.id)["transcript_model"] == HEARD_BY
    assert take_row(conn, second.id)["transcript_model"] is None
    assert runs(conn) == []


def test_the_clips_carry_the_heard_words_and_the_service_hash_matches(conn, voices):
    voice = a_voice(conn, voices, 2, language="ru")
    words = {sha_of(voice.examples[0]): "  Мои   слова. ", sha_of(voice.examples[1]): "1 2 3 4 5"}
    client = FakeService(words=words)

    model = run_bake(conn, client, voices, OWNER, voice.id)

    clips, _ = client.starts[0]
    assert [(c["text"], c["language"]) for c in clips] == [("Мои слова.", "ru"), ("1 2 3 4 5", "ru")]
    assert model.dataset_hash == client.dataset
    assert model_takes(conn) == [t.id for t in voice.examples]
    assert models(conn)[0]["storage_key"] == model.storage_key


def test_the_tag_names_the_app_the_owner_the_voice_and_the_run(conn, voices):
    voice = a_voice(conn, voices, 1)
    client = FakeService()
    keys: list[str] = []

    run_bake(conn, client, voices, OWNER, voice.id, on_started=keys.append)

    _, tag = client.starts[0]
    assert tag == {"source": "app", "owner_id": OWNER, "voice_id": voice.id, "bake_key": keys[0]}
    assert models(conn)[0]["bake_key"] == keys[0]


def test_done_stores_the_model_deletes_the_bake_and_ends_the_run(conn, voices):
    voice = a_voice(conn, voices, 1)
    client = FakeService()

    model = run_bake(conn, client, voices, OWNER, voice.id)

    assert voice_model_for(conn, OWNER, voice.id).model == model
    assert voices.read(model.storage_key) == ADAPTER
    assert client.deletes == [BAKE_ID]
    assert runs(conn) == []


def test_job_lock_is_never_held_during_a_call_and_the_voice_lock_always_is(conn, voices):
    voice = a_voice(conn, voices, 2)
    job_lock, voice_lock = threading.Lock(), threading.Lock()
    client = FakeService(job_lock=job_lock, voice_lock=voice_lock)

    run_bake(conn, client, voices, OWNER, voice.id, job_lock=job_lock, voice_lock=voice_lock)

    assert client.transcribed and client.starts
    assert not client.lock_held_in_call
    assert not client.voice_lock_free_in_call
    assert not voice_lock.locked() and not job_lock.locked()


@pytest.mark.parametrize("end", ["done", "failed", "cancelled"])
def test_the_voice_lock_is_released_after_every_end(conn, voices, end):
    voice = a_voice(conn, voices, 1)
    voice_lock = threading.Lock()
    client = FakeService(end=end, voice_lock=voice_lock)

    with pytest.raises((BakeFailed, Cancelled)) if end != "done" else nullcontext():
        run_bake(conn, client, voices, OWNER, voice.id, voice_lock=voice_lock)

    assert not client.voice_lock_free_in_call
    assert not voice_lock.locked()


def test_a_held_voice_lock_shows_its_wait_and_cancel_works(conn, voices):
    voice = a_voice(conn, voices, 1)
    voice_lock = threading.Lock()
    voice_lock.acquire()
    client = FakeService()
    stages: list[str] = []
    stop = {"now": False}

    def progress(stage, fraction):
        stages.append(stage)
        stop["now"] = True

    with pytest.raises(Cancelled):
        run_bake(conn, client, voices, OWNER, voice.id, voice_lock=voice_lock, progress=progress,
                 cancelled=lambda: stop["now"])

    assert stages == ["waiting for the voice model"]
    assert client.transcribed == [] and client.current_calls == 0


def test_progress_runs_through_the_stage_then_mirrors_the_service(conn, voices):
    voice = a_voice(conn, voices, 2)
    seen: list[tuple[str, float]] = []

    run_bake(conn, FakeService(), voices, OWNER, voice.id,
             progress=lambda stage, fraction: seen.append((stage, round(fraction, 4))))

    assert seen == [
        ("transcribing the takes · take 1 of 2", 0.08),
        ("transcribing the takes · take 2 of 2", 0.15),
        ("sending the takes", 0.15),
        ("loading the voice model for training", round(0.16 + 0.81 * 0.05, 4)),
        ("training · step 3 of 6", round(0.16 + 0.81 * 0.50, 4)),
        ("saving the voice model", round(0.16 + 0.81 * 0.95, 4)),
        ("done", 0.97),
        ("storing the voice model", 0.98),
    ]


def test_a_cancel_during_the_hand_off_waits_and_never_leaves_an_unflagged_run(conn, settings, voices,
                                                                               monkeypatch):
    voice = a_voice(conn, voices, 1)
    job_lock = threading.Lock()
    entered, release = threading.Event(), threading.Event()
    real_start = voice_bake.start_bake_run

    def held_start(*args):
        entered.set()
        release.wait(JOIN_TIMEOUT)
        return real_start(*args)

    monkeypatch.setattr(voice_bake, "start_bake_run", held_start)
    cancel_done = threading.Event()
    voices.on_read = lambda key: cancel_done.wait(JOIN_TIMEOUT) if entered.is_set() else None
    job = {"cancelled": False, "key": None, "flagged": None}
    client = FakeService()
    thread, outcome = in_thread(settings, client, voices, voice.id, job_lock=job_lock,
                                cancelled=lambda: job["cancelled"],
                                on_started=lambda key: job.update(key=key))
    assert entered.wait(JOIN_TIMEOUT)

    def cancel() -> None:
        own = connect(settings.db_path)
        with job_lock:
            job["cancelled"] = True
            if job["key"] is not None:
                job["flagged"] = request_bake_cancel(own, job["key"])
                job["row"] = dict(own.execute("SELECT cancel_requested_at FROM voice_bake_runs"
                                              " WHERE bake_key = ?", (job["key"],)).fetchone())
        own.close()
        cancel_done.set()

    canceller = threading.Thread(target=cancel, daemon=True)
    canceller.start()
    time.sleep(BLOCKED_WAIT)
    assert canceller.is_alive()
    release.set()
    joined(canceller)
    joined(thread)

    assert job["flagged"] is True and job["row"]["cancel_requested_at"] is not None
    assert isinstance(outcome["error"], Cancelled)
    assert client.starts == [] and runs(conn) == [] and models(conn) == []


def test_a_take_added_during_the_last_call_is_heard_and_baked(conn, voices):
    voice = a_voice(conn, voices, 1)
    client = FakeService()
    late: list = []
    client.on_transcribe = lambda n: late.append(add_take(conn, voices, voice.id, 400.0)) if n == 1 else None

    run_bake(conn, client, voices, OWNER, voice.id)

    assert [sha for sha, _ in client.transcribed] == [sha_of(voice.examples[0]), sha_of(late[0])]
    assert [c["sha256"] for c in client.starts[0][0]] == [sha_of(voice.examples[0]), sha_of(late[0])]
    assert model_takes(conn) == [voice.examples[0].id, late[0].id]


def test_a_take_added_after_the_final_snapshot_read_waits_for_the_next_bake(conn, voices, monkeypatch):
    voice = a_voice(conn, voices, 1)
    real_start = voice_bake.start_bake_run
    late: list = []

    def start_then_add(*args):
        key = real_start(*args)
        late.append(add_take(conn, voices, voice.id, 400.0))
        return key

    monkeypatch.setattr(voice_bake, "start_bake_run", start_then_add)
    client = FakeService()

    run_bake(conn, client, voices, OWNER, voice.id)

    assert [sha for sha, _ in client.transcribed] == [sha_of(voice.examples[0])]
    assert model_takes(conn) == [voice.examples[0].id]
    assert voice_model_for(conn, OWNER, voice.id).changes.added == 1
    assert take_row(conn, late[0].id)["transcript_model"] is None


def test_a_queued_take_deleted_and_drained_is_skipped(conn, voices):
    voice = a_voice(conn, voices, 3)
    first, gone, last = voice.examples
    client = FakeService()

    def delete_second(n):
        if n == 1:
            request_example_deletion(conn, OWNER, voice.id, gone.id)
            process_voice_deletions(conn, voices, OWNER)

    client.on_transcribe = delete_second
    reads: list[str] = []
    voices.on_read = reads.append

    run_bake(conn, client, voices, OWNER, voice.id)

    assert [sha for sha, _ in client.transcribed] == [sha_of(first), sha_of(last)]
    assert gone.storage_key not in reads
    assert model_takes(conn) == [first.id, last.id]


def test_a_take_deleted_during_its_call_is_not_stored_and_the_bake_goes_on(conn, voices):
    voice = a_voice(conn, voices, 2)
    kept, gone = voice.examples
    client = FakeService()
    client.on_transcribe = (lambda n: request_example_deletion(conn, OWNER, voice.id, gone.id)
                            if n == 2 else None)

    run_bake(conn, client, voices, OWNER, voice.id)

    assert len(client.transcribed) == 2
    assert take_row(conn, gone.id)["transcript_model"] is None
    assert model_takes(conn) == [kept.id]


def test_an_audio_read_that_fails_for_a_take_still_there_fails_the_bake(conn, voices):
    voice = a_voice(conn, voices, 1)
    take = voice.examples[0]

    def broken(key):
        raise OSError("disk error")

    voices.on_read = broken

    with pytest.raises(BakeFailed) as raised:
        run_bake(conn, FakeService(), voices, OWNER, voice.id)

    assert str(raised.value) == f"The audio of take #{take.id} could not be read (disk error)."
    assert runs(conn) == []


def test_an_audio_read_that_fails_for_a_take_gone_meanwhile_skips_it(conn, voices):
    voice = a_voice(conn, voices, 2)
    gone, kept = voice.examples

    def vanish(key):
        if key == gone.storage_key:
            request_example_deletion(conn, OWNER, voice.id, gone.id)
            process_voice_deletions(conn, voices, OWNER)
            raise FileNotFoundError(key)

    voices.on_read = vanish
    client = FakeService()

    run_bake(conn, client, voices, OWNER, voice.id)

    assert [sha for sha, _ in client.transcribed] == [sha_of(kept)]
    assert model_takes(conn) == [kept.id]


@pytest.mark.parametrize("end", ["cancelled", "done", "failed"])
def test_cancel_during_training_sends_one_cancel_waits_for_any_end_and_stores_nothing(conn, voices, end):
    voice = a_voice(conn, voices, 1)
    client = FakeService(end_after_cancel=end)
    stop = {"now": False}
    client.on_status = lambda step: stop.update(now=True) if step == 1 else None

    with pytest.raises(Cancelled):
        run_bake(conn, client, voices, OWNER, voice.id, cancelled=lambda: stop["now"])

    assert client.cancels == [BAKE_ID]
    assert client.polls_after_cancel >= 1
    assert models(conn) == [] and runs(conn) == []
    assert client.deletes == [BAKE_ID]


def test_a_failed_bake_says_why_and_is_deleted(conn, voices):
    voice = a_voice(conn, voices, 1)
    client = FakeService(end="failed")

    with pytest.raises(BakeFailed) as raised:
        run_bake(conn, client, voices, OWNER, voice.id)

    assert str(raised.value) == "The bake failed (boom)."
    assert client.deletes == [BAKE_ID] and models(conn) == [] and runs(conn) == []


def test_a_service_cancel_nobody_asked_for_stores_nothing(conn, voices):
    voice = a_voice(conn, voices, 1)
    client = FakeService(end="cancelled")

    with pytest.raises(Cancelled):
        run_bake(conn, client, voices, OWNER, voice.id)

    assert client.deletes == [BAKE_ID] and models(conn) == []


def test_a_404_says_the_service_restarted_and_still_deletes(conn, voices):
    voice = a_voice(conn, voices, 1)
    client = FakeService()
    client.on_status = lambda step: setattr(client, "gone", True)

    with pytest.raises(BakeFailed) as raised:
        run_bake(conn, client, voices, OWNER, voice.id)

    assert str(raised.value) == RESTARTED
    assert client.deletes == [BAKE_ID] and runs(conn) == []


def test_an_adapter_with_another_sha256_is_not_stored(conn, voices):
    voice = a_voice(conn, voices, 1)
    client = FakeService()

    def swap(step):
        if step == len(SERVICE_STAGES):
            client.adapter = b"other"

    client.on_status = swap
    client.bake_adapter = lambda bake_id: ADAPTER

    with pytest.raises(BakeFailed, match="does not match its sha256"):
        run_bake(conn, client, voices, OWNER, voice.id)

    assert models(conn) == []


def test_a_409_from_bake_start_is_busy_and_ends_the_run(conn, voices):
    voice = a_voice(conn, voices, 1)
    client = FakeService(currents=[None, FOREIGN], start_refusals=1)

    with pytest.raises(BakeBusy) as raised:
        run_bake(conn, client, voices, OWNER, voice.id)

    assert raised.value.status == FOREIGN
    assert client.starts == [] and runs(conn) == [] and client.deletes == []


def test_a_409_from_bake_start_with_wait_for_other_waits_then_starts_once(conn, voices):
    voice = a_voice(conn, voices, 1)
    ended = FOREIGN | {"state": "done"}
    client = FakeService(currents=[None, FOREIGN, FOREIGN, ended], start_refusals=1)
    stages: list[str] = []

    run_bake(conn, client, voices, OWNER, voice.id, wait_for_other=True,
             progress=lambda stage, fraction: stages.append(stage))

    assert "waiting for the other bake to stop" in stages
    assert client.current_calls == 4 and len(client.starts) == 1
    assert client.deletes == [BAKE_ID] and client.cancels == []


def test_cancel_while_waiting_for_the_other_bake_starts_nothing(conn, voices):
    voice = a_voice(conn, voices, 1)
    client = FakeService(currents=[FOREIGN, FOREIGN, FOREIGN])
    stop = {"now": False}

    def progress(stage, fraction):
        if stage == "waiting for the other bake to stop":
            stop["now"] = True

    with pytest.raises(Cancelled):
        run_bake(conn, client, voices, OWNER, voice.id, wait_for_other=True, progress=progress,
                 cancelled=lambda: stop["now"])

    assert client.starts == [] and client.transcribed == []
    assert client.cancels == [] and client.deletes == []


def test_a_take_deleted_during_the_training_does_not_drop_the_model(conn, voices):
    voice = a_voice(conn, voices, 2)
    client = FakeService()

    def delete_one(step):
        if step == 1:
            request_example_deletion(conn, OWNER, voice.id, voice.examples[0].id)
            process_voice_deletions(conn, voices, OWNER)

    client.on_status = delete_one

    model = run_bake(conn, client, voices, OWNER, voice.id)

    assert model.takes == 2
    assert model_takes(conn) == [t.id for t in voice.examples]


def test_the_voice_deleted_during_the_training_drops_the_model(conn, voices):
    voice = a_voice(conn, voices, 1)
    client = FakeService()
    client.on_status = lambda step: request_voice_deletion(conn, OWNER, voice.id) if step == 1 else None

    with pytest.raises(BakeFailed) as raised:
        run_bake(conn, client, voices, OWNER, voice.id)

    assert str(raised.value) == VOICE_GONE
    assert models(conn) == [] and client.deletes == [BAKE_ID]


def test_a_run_flagged_for_cancel_is_not_stored_when_the_service_reports_done(conn, voices):
    voice = a_voice(conn, voices, 1)
    client = FakeService()
    keys: list[str] = []
    client.on_status = lambda step: request_bake_cancel(conn, keys[0]) if step == 1 else None

    with pytest.raises(Cancelled):
        run_bake(conn, client, voices, OWNER, voice.id, on_started=keys.append)

    assert models(conn) == [] and runs(conn) == []
    assert client.deletes == [BAKE_ID]


def test_a_cancel_during_the_publication_waits_and_the_store_wins(conn, settings, voices):
    voice = a_voice(conn, voices, 1)
    job_lock = threading.Lock()
    entered, release = threading.Event(), threading.Event()
    job = {"cancelled": False, "key": None}

    def held_publication(model, text):
        entered.set()
        release.wait(JOIN_TIMEOUT)

    thread, outcome = in_thread(settings, FakeService(), voices, voice.id, job_lock=job_lock,
                                cancelled=lambda: job["cancelled"],
                                on_started=lambda key: job.update(key=key), on_saved=held_publication)
    assert entered.wait(JOIN_TIMEOUT)

    def cancel() -> None:
        own = connect(settings.db_path)
        with job_lock:
            job["cancelled"] = True
            job["open"] = request_bake_cancel(own, job["key"])
        own.close()

    canceller = threading.Thread(target=cancel, daemon=True)
    canceller.start()
    time.sleep(BLOCKED_WAIT)
    assert canceller.is_alive()
    release.set()
    joined(thread)
    joined(canceller)

    assert outcome["result"].version == 1
    assert job["open"] is False
    assert len(models(conn)) == 1


def test_a_cancel_before_the_publication_stores_nothing(conn, voices):
    voice = a_voice(conn, voices, 1)
    client = FakeService()
    stop = {"now": False}
    real_adapter = client.bake_adapter

    def adapter_then_cancel(bake_id):
        stop["now"] = True
        return real_adapter(bake_id)

    client.bake_adapter = adapter_then_cancel

    with pytest.raises(Cancelled):
        run_bake(conn, client, voices, OWNER, voice.id, cancelled=lambda: stop["now"])

    assert models(conn) == [] and runs(conn) == [] and client.deletes == [BAKE_ID]


def started_bake(conn, voices, voice, monkeypatch) -> tuple[FakeService, dict, str]:
    """A bake this app started and left running (an app reload): its run row and the
    service status `reattach` passes as `attach`."""
    client = FakeService()
    keys: list[str] = []

    def leave(step):
        raise SystemExit  # the reload kills the job thread mid-bake

    client.on_status = leave
    with monkeypatch.context() as patched:
        patched.setattr(voice_bake, "end_bake_run", lambda c, k: None)
        with pytest.raises(SystemExit):
            run_bake(conn, client, voices, OWNER, voice.id, on_started=keys.append)
    client.on_status = None
    client.deletes.clear()
    client.starts_before = list(client.starts)
    client.transcribed_before = list(client.transcribed)
    return client, client._status("running", "training", 0.5), keys[0]


def test_attach_stores_the_stored_snapshot_after_the_takes_changed(conn, voices, monkeypatch):
    voice = a_voice(conn, voices, 2)
    client, status, key = started_bake(conn, voices, voice, monkeypatch)
    added = add_take(conn, voices, voice.id, 400.0)
    request_example_deletion(conn, OWNER, voice.id, voice.examples[0].id)
    process_voice_deletions(conn, voices, OWNER)
    reads: list[str] = []
    voices.on_read = reads.append

    run_bake(conn, client, voices, OWNER, voice.id, attach=status)

    assert client.starts == client.starts_before and client.transcribed == client.transcribed_before
    assert reads == []
    assert model_takes(conn) == [t.id for t in voice.examples]
    assert take_row(conn, added.id)["transcript_model"] is None
    assert models(conn)[0]["bake_key"] == key
    assert client.deletes == [BAKE_ID] and runs(conn) == []


def test_attach_after_every_take_was_deleted_and_drained_still_stores(conn, voices, monkeypatch):
    voice = a_voice(conn, voices, 2)
    client, status, _ = started_bake(conn, voices, voice, monkeypatch)
    for take in voice.examples:
        request_example_deletion(conn, OWNER, voice.id, take.id)
    process_voice_deletions(conn, voices, OWNER)

    model = run_bake(conn, client, voices, OWNER, voice.id, attach=status)

    assert model.takes == 2
    assert model_takes(conn) == [t.id for t in voice.examples]


def test_attach_of_a_flagged_run_cancels_and_stores_nothing(conn, voices, monkeypatch):
    voice = a_voice(conn, voices, 1)
    client, status, key = started_bake(conn, voices, voice, monkeypatch)
    request_bake_cancel(conn, key)

    with pytest.raises(Cancelled):
        run_bake(conn, client, voices, OWNER, voice.id, attach=status)

    assert client.cancels == [BAKE_ID]
    assert models(conn) == [] and runs(conn) == [] and client.deletes == [BAKE_ID]
