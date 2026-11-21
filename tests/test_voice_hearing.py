"""The hearing job of one saved voice in `app` (design §11.1 "The words of a take are heard
right after it is saved") — a fake models service, a temp DB made by `migrate`, a temp
voices store."""

import hashlib
import threading

import pytest

from feed.build import Cancelled
from feed.saved_voices import (
    NO_WORDS,
    add_example,
    create_voice,
    film_take,
    get_voice,
    process_voice_deletions,
    request_example_deletion,
)
from feed.voice_hearing import BAKE_CHECK, NOT_OFFERED, HearingFailed, hear_takes
from tests.test_feed_build import clip
from tests.test_voice_bake import ASR_MODEL, ASR_REVISION, HEARD_BY, http_error
from tests.test_voice_preview import HookedStorage

OWNER = 1


class FakeASR:
    """`POST /voice/transcribe` as the job sees it: `words` = heard text per sha256;
    `on_call(n)` runs inside the n-th call; `error` = (n, exception) raised by the n-th
    call; `voice_lock_held` notes whether each call ran under the voice lock."""

    def __init__(self, words: dict | None = None, voice_lock: "threading.Lock | None" = None) -> None:
        self.words = words or {}
        self.voice_lock = voice_lock
        self.heard: list[tuple[str, str]] = []
        self.voice_lock_held: list[bool] = []
        self.on_call = None
        self.error = None

    def transcribe(self, audio: bytes, language: str) -> dict:
        sha = hashlib.sha256(audio).hexdigest()
        self.heard.append((sha, language))
        if self.voice_lock is not None:
            self.voice_lock_held.append(self.voice_lock.locked())
        if self.on_call is not None:
            self.on_call(len(self.heard))
        if self.error is not None and len(self.heard) == self.error[0]:
            raise self.error[1]
        return {"text": self.words.get(sha, f"Words of {sha[:6]}."), "model": ASR_MODEL,
                "revision": ASR_REVISION, "seconds": 6.0, "elapsed_s": 0.1}


@pytest.fixture
def voices(tmp_path):
    return HookedStorage(tmp_path / "voices")


def a_take(conn, voices, voice_id, freq, *, language="en"):
    return add_example(conn, voices, OWNER, voice_id, language=language, source="uploaded",
                       data=clip(freq))


def test_every_waiting_take_is_heard_and_the_newest_becomes_the_film_take(conn, voices):
    voice = create_voice(conn, OWNER, "Mum")
    first = a_take(conn, voices, voice.id, 220.0)
    second = a_take(conn, voices, voice.id, 330.0)
    client = FakeASR({second.content_hash: "  Мои   слова. "})
    assert film_take(conn, OWNER, voice.id, "en").problem == "waiting"
    assert hear_takes(conn, client, voices, OWNER, voice.id) == 2
    assert client.heard == [(first.content_hash, "en"), (second.content_hash, "en")]
    takes = {take.id: take for take in get_voice(conn, OWNER, voice.id).examples}
    assert (takes[second.id].transcript, takes[second.id].transcript_model) == ("Мои слова.", HEARD_BY)
    assert takes[first.id].words_state == "heard"
    assert film_take(conn, OWNER, voice.id, "en").take == takes[second.id]


def test_a_take_is_heard_once(conn, voices):
    voice = create_voice(conn, OWNER, "Mum")
    a_take(conn, voices, voice.id, 220.0)
    client = FakeASR()
    hear_takes(conn, client, voices, OWNER, voice.id)
    assert hear_takes(conn, client, voices, OWNER, voice.id) == 0
    assert len(client.heard) == 1


def test_a_take_saved_while_the_job_runs_is_heard_in_the_same_job(conn, voices):
    voice = create_voice(conn, OWNER, "Mum")
    a_take(conn, voices, voice.id, 220.0)
    client = FakeASR()
    late = []
    client.on_call = lambda n: late.append(a_take(conn, voices, voice.id, 440.0)) if n == 1 else None
    progress: list[tuple[str, float]] = []
    assert hear_takes(conn, client, voices, OWNER, voice.id,
                      progress=lambda stage, fraction: progress.append((stage, fraction))) == 2
    assert client.heard[1] == (late[0].content_hash, "en")
    assert [stage for stage, _ in progress] == ["hearing the words · take 1 of 1",
                                                "hearing the words · take 2 of 2"]


def test_a_take_with_no_words_is_heard_and_never_spoken_from(conn, voices):
    voice = create_voice(conn, OWNER, "Mum")
    silent = a_take(conn, voices, voice.id, 220.0, language="ru")
    hear_takes(conn, FakeASR({silent.content_hash: "♪"}), voices, OWNER, voice.id)
    heard = get_voice(conn, OWNER, voice.id).examples[0]
    assert (heard.words_state, heard.transcript_error) == ("unusable", NO_WORDS)
    assert film_take(conn, OWNER, voice.id, "ru").problem == "unusable"


def test_a_deleted_take_is_skipped(conn, voices):
    voice = create_voice(conn, OWNER, "Mum")
    gone = a_take(conn, voices, voice.id, 220.0)
    kept = a_take(conn, voices, voice.id, 330.0)
    client = FakeASR()
    request_example_deletion(conn, OWNER, voice.id, gone.id)
    assert hear_takes(conn, client, voices, OWNER, voice.id) == 1
    assert client.heard == [(kept.content_hash, "en")]


def test_a_take_deleted_during_its_audio_read_is_skipped_and_the_next_take_is_heard(conn, voices):
    voice = create_voice(conn, OWNER, "Mum")
    gone = a_take(conn, voices, voice.id, 220.0)
    kept = a_take(conn, voices, voice.id, 330.0)

    def vanish(key):
        if key == gone.storage_key:
            request_example_deletion(conn, OWNER, voice.id, gone.id)
            process_voice_deletions(conn, voices, OWNER)
            raise FileNotFoundError(key)

    voices.on_read = vanish
    client = FakeASR()
    assert hear_takes(conn, client, voices, OWNER, voice.id) == 1
    assert client.heard == [(kept.content_hash, "en")]
    assert [take.id for take in get_voice(conn, OWNER, voice.id).examples] == [kept.id]
    assert film_take(conn, OWNER, voice.id, "en").take.id == kept.id


def test_a_take_deleted_during_its_call_gets_no_words_and_the_next_take_is_heard(conn, voices):
    voice = create_voice(conn, OWNER, "Mum")
    gone = a_take(conn, voices, voice.id, 220.0)
    kept = a_take(conn, voices, voice.id, 330.0)
    client = FakeASR()

    def delete_first(n):
        if n == 1:
            request_example_deletion(conn, OWNER, voice.id, gone.id)
            process_voice_deletions(conn, voices, OWNER)

    client.on_call = delete_first
    assert hear_takes(conn, client, voices, OWNER, voice.id) == 1
    assert client.heard == [(gone.content_hash, "en"), (kept.content_hash, "en")]
    assert conn.execute("SELECT COUNT(*) FROM voice_examples WHERE id = ?", (gone.id,)).fetchone()[0] == 0
    assert film_take(conn, OWNER, voice.id, "en").take.id == kept.id


def test_cancel_stops_before_the_next_take_and_keeps_the_words_heard(conn, voices):
    voice = create_voice(conn, OWNER, "Mum")
    first = a_take(conn, voices, voice.id, 220.0)
    a_take(conn, voices, voice.id, 330.0)
    client = FakeASR()
    stop = threading.Event()
    client.on_call = lambda n: stop.set()
    with pytest.raises(Cancelled):
        hear_takes(conn, client, voices, OWNER, voice.id, cancelled=stop.is_set)
    assert client.heard == [(first.content_hash, "en")]
    states = [take.words_state for take in get_voice(conn, OWNER, voice.id).examples]
    assert states == ["heard", "waiting"]


def test_each_call_runs_under_the_voice_lock_and_releases_it(conn, voices):
    voice = create_voice(conn, OWNER, "Mum")
    a_take(conn, voices, voice.id, 220.0)
    a_take(conn, voices, voice.id, 330.0)
    lock = threading.Lock()
    client = FakeASR(voice_lock=lock)
    hear_takes(conn, client, voices, OWNER, voice.id, voice_lock=lock)
    assert client.voice_lock_held == [True, True] and not lock.locked()


@pytest.mark.parametrize("error, reason", [
    (http_error(409, "a voice is being baked"), BAKE_CHECK),
    (http_error(503, "not offered"), NOT_OFFERED),
    (RuntimeError("boom"), "Could not hear the words of take #{id} (boom)."),
])
def test_a_failed_call_ends_the_job_with_its_reason_and_stores_nothing_for_that_take(
        conn, voices, error, reason):
    voice = create_voice(conn, OWNER, "Mum")
    first = a_take(conn, voices, voice.id, 220.0)
    second = a_take(conn, voices, voice.id, 330.0)
    client = FakeASR()
    client.error = (2, error)
    with pytest.raises(HearingFailed) as failed:
        hear_takes(conn, client, voices, OWNER, voice.id)
    assert str(failed.value) == reason.format(id=second.id)
    states = {take.id: take.words_state for take in get_voice(conn, OWNER, voice.id).examples}
    assert states == {first.id: "heard", second.id: "waiting"}


def test_an_unreadable_take_fails_with_its_reason(conn, voices):
    voice = create_voice(conn, OWNER, "Mum")
    lost = a_take(conn, voices, voice.id, 220.0)

    def gone(_key):
        raise FileNotFoundError("no such file")

    voices.on_read = gone
    with pytest.raises(HearingFailed) as failed:
        hear_takes(conn, FakeASR(), voices, OWNER, voice.id)
    assert str(failed.value) == f"The audio of take #{lost.id} could not be read (no such file)."
