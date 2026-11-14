"""The test of one saved voice from the take films speak from (design §11.1) — fully faked."""

import json
import threading
import time

import httpx
import numpy as np
import pytest

import scripts.voice_check
from albums.memory_store import Memory, replace_memories
from db.connection import connect
from feed import saved_voices
from feed.build import Cancelled
from feed.saved_voices import (
    VoiceError,
    add_example,
    create_voice,
    get_voice,
    process_voice_deletions,
    reference_for,
    request_example_deletion,
    request_voice_deletion,
    set_heard_words,
    set_narrator,
    token,
    voice_test,
)
from feed.voice_preview import (
    CUSTOM_TEXT_MAX,
    MEASURE_FROM,
    MEASURE_TO,
    PREVIEW_SENTENCES,
    PreviewFailed,
    PreviewJob,
    PreviewText,
    preview_text,
    run_preview,
)
from modelsvc.backends.fake import FakeBackend
from storage.local import LocalStorage
from storage.wav import encode_wav
from tests.factories import add_photo
from tests.test_feed_build import FAKE_ASR, WORDS, clip

OWNER = 1
REVISION = "rev"
MODEL = "voxcpm2"
WAITING = "waiting for the voice model"
TEST_SECONDS = 2.0
SPEAK_SECONDS = 0.05
JOIN_TIMEOUT = 10.0
SENTENCE = PreviewText("sentence", "EN-2", None, PREVIEW_SENTENCES["en"]["EN-2"])
SHOWN_TEXT = "The text this take was shown before speech recognition."
CHANGED = ("The take of the test was deleted while the test ran, or the voice is being deleted. "
           "Generate it again.")
MODEL_CHANGED = "The voice-copy model changed while the test ran. Generate it again."


class FakeClient:
    """`POST /speak` of a saved voice: records each call and answers with a fixed reply
    (`model`, `revision`, and per reference the conditioning built from its take, as §5.1
    and D86 say). `state` replaces that entry's state; `drop_conditioning` leaves it out."""

    def __init__(self, *, revision: str = REVISION, during_speak=None,
                 refuse: str | None = None, state: str = "built",
                 drop_conditioning: bool = False) -> None:
        self.revision = revision
        self.during_speak, self.refuse = during_speak, refuse
        self.state, self.drop_conditioning = state, drop_conditioning
        self.calls: list[tuple] = []
        self.references: list[dict] = []

    def speak(self, text, language="en", voice=None, segments=None, voices=None, *,
              references=None):
        self.calls.append((text, language, voice, segments, voices))
        self.references.append(references)
        if self.refuse is not None:
            request = httpx.Request("POST", "http://models/speak")
            raise httpx.HTTPStatusError("503", request=request, response=httpx.Response(
                503, request=request, json={"detail": self.refuse}))
        time.sleep(SPEAK_SECONDS)
        if self.during_speak is not None:
            self.during_speak()
        conditioning = {} if self.drop_conditioning else {
            name: {"state": self.state, "model": MODEL, "revision": self.revision, "reason": None,
                   "blob": None}
            for name in (references or {})}
        audio = encode_wav(np.zeros(int(24000 * TEST_SECONDS), dtype=np.float32), 24000)
        return {"audio": audio, "sample_rate": 24000, "model": MODEL, "revision": self.revision,
                "segments": [], "conditioning": conditioning,
                "adapters": {name: {"state": "none", "sha256": None} for name in (references or {})}}


class MeasuringClient(FakeClient):
    """`FakeClient` with `POST /voice/embed`: the scripted speaker model of `FakeBackend`,
    or the service's `embed_status` answer. `during_embed` runs inside each call."""

    def __init__(self, *, embed_status: tuple[int, str] | None = None, during_embed=None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.backend = FakeBackend()
        self.embed_status, self.during_embed = embed_status, during_embed
        self.embedded: list[bytes] = []

    def embed_voice(self, audio):
        self.embedded.append(audio)
        if self.during_embed is not None:
            self.during_embed()
        if self.embed_status is not None:
            code, detail = self.embed_status
            request = httpx.Request("POST", "http://models/voice/embed")
            raise httpx.HTTPStatusError(str(code), request=request, response=httpx.Response(
                code, request=request, json={"detail": detail}))
        return self.backend.embed_voice(audio)


class HookedStorage(LocalStorage):
    """The saved-voice store, recording each written key; `on_read` / `on_write` run before
    a read / a write once set."""

    def __init__(self, root):
        super().__init__(root)
        self.written: list[str] = []
        self.on_read = None
        self.on_write = None

    def read(self, key):
        if self.on_read is not None:
            self.on_read(key)
        return super().read(key)

    def write(self, key, data):
        if self.on_write is not None:
            self.on_write(key)
        self.written.append(key)
        super().write(key, data)


@pytest.fixture
def voices(tmp_path):
    return HookedStorage(tmp_path / "voices")


@pytest.fixture
def memory_group(conn):
    for pid in (1, 2, 3):
        add_photo(conn, photo_id=pid, content_hash=f"h{pid}", thumb_key=f"{pid}.jpg",
                  width=4000, height=3000, ai_title=f"caption {pid}",
                  shot_at=f"2025-07-12T1{pid}:00:00")
    replace_memories(conn, OWNER, [Memory("Beach day", "Warm afternoon.", [1, 2, 3], "sig")])
    return conn.execute("SELECT id FROM groups WHERE kind = 'memory'").fetchone()["id"]


def a_voice(conn, voices, *, language="en", name="Mum", heard=True):
    """A saved voice with one uploaded take; `heard` stores the words ASR heard in it."""
    voice = create_voice(conn, OWNER, name)
    added = add_example(conn, voices, OWNER, voice.id, language=language, source="uploaded",
                        data=clip(220.0))
    if heard:
        set_heard_words(conn, OWNER, voice.id, added.id, text=WORDS, model=FAKE_ASR)
    voices.written.clear()
    return get_voice(conn, OWNER, voice.id)


def a_heard_take(conn, voices, voice, freq):
    added = add_example(conn, voices, OWNER, voice.id, language="en", source="uploaded",
                        data=clip(freq))
    return set_heard_words(conn, OWNER, voice.id, added.id, text=WORDS, model=FAKE_ASR)


def a_job(voice, text=SENTENCE, *, language="en", revision=REVISION) -> PreviewJob:
    return PreviewJob(voice.id, language, text, revision)


def stored_tests(conn) -> list:
    return [dict(r) for r in conn.execute("SELECT * FROM voice_tests")]


def in_thread(settings, voices, client, job, **kwargs) -> tuple[threading.Thread, dict]:
    """`run_preview` on its own connection in a daemon thread; `outcome` gets its
    result or its exception."""
    outcome: dict = {}

    def run() -> None:
        own = connect(settings.db_path)
        try:
            outcome["result"] = run_preview(own, client, voices, OWNER, job, **kwargs)
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


def test_the_preview_sentences_are_the_voice_check_sentences():
    assert PREVIEW_SENTENCES == {lang: dict(pairs)
                                 for lang, pairs in scripts.voice_check.SENTENCES.items()}


def test_no_preview_sentence_is_a_reading_text():
    readings = [*scripts.voice_check.RECORDING_PROMPTS.values(),
                *(text for texts in saved_voices.RECORDING_PROMPTS.values()
                  for text in texts.values())]
    for sentences in PREVIEW_SENTENCES.values():
        for sentence in sentences.values():
            assert not any(sentence in reading for reading in readings), sentence


def test_a_sentence_of_the_language_is_its_text(conn, voices):
    assert preview_text(a_voice(conn, voices), "en", "sentence", "EN-2", "") == SENTENCE


def test_a_sentence_of_the_chosen_language_is_used_whatever_the_takes_language(conn, voices):
    voice = a_voice(conn, voices, language="en")
    assert preview_text(voice, "ru", "sentence", "RU-1", "") == PreviewText(
        "sentence", "RU-1", None, PREVIEW_SENTENCES["ru"]["RU-1"])


@pytest.mark.parametrize("text_id", ["RU-2", "EN-9"])
def test_an_unknown_or_other_language_sentence_is_refused(conn, voices, text_id):
    with pytest.raises(VoiceError, match=f"Unknown test sentence for English: '{text_id}'"):
        preview_text(a_voice(conn, voices), "en", "sentence", text_id, "")


def test_a_blank_own_text_is_refused(conn, voices):
    with pytest.raises(VoiceError, match="Type a text to test"):
        preview_text(a_voice(conn, voices), "en", "custom", "", "  ")


def test_an_own_text_over_the_limit_is_refused(conn, voices):
    with pytest.raises(VoiceError, match="at most 300 characters"):
        preview_text(a_voice(conn, voices), "en", "custom", "", "a" * (CUSTOM_TEXT_MAX + 1))


def test_an_own_text_at_the_limit_is_stripped_and_accepted(conn, voices):
    text = "a" * CUSTOM_TEXT_MAX
    assert preview_text(a_voice(conn, voices), "en", "custom", "", f"  {text}\n") == PreviewText(
        "custom", None, None, text)


def test_a_heard_take_is_its_heard_words(conn, voices):
    voice = a_voice(conn, voices)
    [take] = voice.examples
    assert preview_text(voice, "en", "take", str(take.id), "") == PreviewText(
        "take", None, take.id, WORDS)


def test_a_waiting_take_is_refused_until_its_words_are_heard(conn, voices):
    voice = a_voice(conn, voices, heard=False)
    [take] = voice.examples
    with pytest.raises(VoiceError) as refused:
        preview_text(voice, "en", "take", str(take.id), "")
    assert str(refused.value) == f"The words of take #{take.id} are not heard yet."


def test_a_take_stored_with_its_shown_text_is_waiting_never_its_shown_text(conn, voices):
    voice = a_voice(conn, voices, heard=False)
    [take] = voice.examples
    conn.execute("UPDATE voice_examples SET transcript = ? WHERE id = ?", (SHOWN_TEXT, take.id))
    conn.commit()
    voice = get_voice(conn, OWNER, voice.id)
    with pytest.raises(VoiceError) as refused:
        preview_text(voice, "en", "take", str(take.id), "")
    assert str(refused.value) == f"The words of take #{take.id} are not heard yet."


@pytest.mark.parametrize(("heard", "reason"), [
    ("♪ ♪", saved_voices.NO_WORDS),
    ("a" * (saved_voices.TRANSCRIPT_MAX + 1), saved_voices.TOO_LONG),
])
def test_an_unusable_take_is_refused_with_its_own_reason(conn, voices, heard, reason):
    voice = a_voice(conn, voices, heard=False)
    [take] = voice.examples
    assert set_heard_words(conn, OWNER, voice.id, take.id, text=heard, model=FAKE_ASR)
    voice = get_voice(conn, OWNER, voice.id)
    with pytest.raises(VoiceError) as refused:
        preview_text(voice, "en", "take", str(take.id), "")
    assert str(refused.value) == f"Take #{take.id} has no words to speak: {reason}."


def test_a_take_of_another_language_a_pending_or_an_unknown_take_is_refused(conn, voices):
    voice = a_voice(conn, voices)
    [take] = voice.examples
    for language, text_id in (("ru", str(take.id)), ("en", str(take.id + 1)), ("en", "x")):
        with pytest.raises(VoiceError, match="^Unknown take.$"):
            preview_text(voice, language, "take", text_id, "")
    assert request_example_deletion(conn, OWNER, voice.id, take.id)
    with pytest.raises(VoiceError, match="^Unknown take.$"):
        preview_text(get_voice(conn, OWNER, voice.id), "en", "take", str(take.id), "")


@pytest.mark.parametrize("kind", ["other", "reading"])
def test_an_unknown_or_the_old_reading_kind_of_text_is_refused(conn, voices, kind):
    with pytest.raises(VoiceError, match="Unknown kind of test text"):
        preview_text(a_voice(conn, voices), "en", kind, "", "")


def test_a_test_speaks_from_the_newest_heard_take_and_is_stored(conn, voices):
    voice = a_voice(conn, voices)
    newest = a_heard_take(conn, voices, voice, 330.0)
    voices.written.clear()
    client = FakeClient()
    test = run_preview(conn, client, voices, OWNER, a_job(voice))
    speaker = token(voice.id)
    assert client.calls == [(SENTENCE.text, "en", speaker, None, None)]
    assert client.references == [{speaker: reference_for(conn, voices, newest)}]
    assert client.references[0][speaker]["transcript"] == WORDS
    assert "adapter" not in client.references[0][speaker]
    assert (test.voice_id, test.language, test.text_kind, test.text_id, test.example_id,
            test.text) == (voice.id, "en", "sentence", "EN-2", None, SENTENCE.text)
    assert (test.prompt_example_id, test.model_version, test.model_sha256) == (newest.id, 0, "")
    assert (test.model_key, test.revision) == (MODEL, REVISION)
    assert test.elapsed_seconds >= SPEAK_SECONDS and test.seconds == TEST_SECONDS
    assert voice_test(conn, OWNER, voice.id) == test
    assert voices.exists(test.storage_key)
    assert voices.written == [test.storage_key]                    # no conditioning blob
    assert conn.execute("SELECT COUNT(*) FROM voice_conditionings").fetchone()[0] == 0


def test_a_new_test_replaces_the_last_one(conn, voices):
    voice = a_voice(conn, voices)
    run_preview(conn, FakeClient(), voices, OWNER, a_job(voice))
    custom = PreviewText("custom", None, None, "Hello there.")
    run_preview(conn, FakeClient(), voices, OWNER, a_job(voice, custom))
    [row] = stored_tests(conn)
    assert (row["text_kind"], row["text"]) == ("custom", "Hello there.")


def test_a_voice_whose_takes_wait_for_their_words_is_not_tested(conn, voices):
    voice = a_voice(conn, voices, heard=False)
    client = FakeClient()
    with pytest.raises(PreviewFailed, match=r"^The words of the English takes are not heard yet\.$"):
        run_preview(conn, client, voices, OWNER, a_job(voice))
    assert client.calls == [] and stored_tests(conn) == []


def test_a_voice_whose_takes_have_no_usable_words_is_not_tested(conn, voices):
    voice = a_voice(conn, voices, heard=False)
    [take] = voice.examples
    set_heard_words(conn, OWNER, voice.id, take.id, text="♪", model=FAKE_ASR)
    client = FakeClient()
    with pytest.raises(PreviewFailed, match=r"^No English take has usable words\.$"):
        run_preview(conn, client, voices, OWNER, a_job(voice))
    assert client.calls == []


def test_a_language_with_no_take_is_not_tested(conn, voices):
    voice = a_voice(conn, voices, language="en")
    client = FakeClient()
    ru = PreviewText("sentence", "RU-1", None, PREVIEW_SENTENCES["ru"]["RU-1"])
    with pytest.raises(PreviewFailed, match=r"^Record a Russian take first\.$"):
        run_preview(conn, client, voices, OWNER, a_job(voice, ru, language="ru"))
    assert client.calls == []


def test_a_voice_being_deleted_is_not_tested(conn, voices):
    voice = a_voice(conn, voices)
    assert request_voice_deletion(conn, OWNER, voice.id)
    client = FakeClient()
    with pytest.raises(PreviewFailed, match=r"^This voice is being deleted\.$"):
        run_preview(conn, client, voices, OWNER, a_job(voice))
    assert client.calls == []


def test_a_missing_take_file_fails_the_test(conn, voices):
    voice = a_voice(conn, voices)
    [take] = voice.examples
    voices.delete(take.storage_key)
    client = FakeClient()
    with pytest.raises(PreviewFailed, match=f"Take #{take.id} of «Mum» could not be read"):
        run_preview(conn, client, voices, OWNER, a_job(voice))
    assert client.calls == []


@pytest.mark.parametrize("client", [FakeClient(drop_conditioning=True), FakeClient(state="failed")],
                         ids=["no entry", "no conditioning"])
def test_a_reply_that_did_not_speak_from_the_take_is_not_stored(conn, voices, client):
    voice = a_voice(conn, voices)
    with pytest.raises(PreviewFailed, match=r"^The models service did not speak «Mum» from "
                                            r"its take\.$"):
        run_preview(conn, client, voices, OWNER, a_job(voice))
    assert stored_tests(conn) == []


def test_a_restored_conditioning_is_a_reply_from_the_take(conn, voices):
    voice = a_voice(conn, voices)
    [take] = voice.examples
    test = run_preview(conn, FakeClient(state="restored"), voices, OWNER, a_job(voice))
    assert test.prompt_example_id == take.id


def test_a_reply_of_another_revision_is_not_stored(conn, voices):
    voice = a_voice(conn, voices)
    with pytest.raises(PreviewFailed) as failed:
        run_preview(conn, FakeClient(revision="other"), voices, OWNER, a_job(voice))
    assert str(failed.value) == MODEL_CHANGED
    assert stored_tests(conn) == []


def test_a_voice_deleted_during_the_test_drops_it(conn, voices):
    voice = a_voice(conn, voices)

    def delete_now() -> None:
        assert request_voice_deletion(conn, OWNER, voice.id)

    with pytest.raises(PreviewFailed) as failed:
        run_preview(conn, FakeClient(during_speak=delete_now), voices, OWNER, a_job(voice))
    assert str(failed.value) == CHANGED
    assert stored_tests(conn) == []


def test_a_take_deleted_during_the_test_is_stored_without_it(conn, voices):
    voice = a_voice(conn, voices)
    [take] = voice.examples
    prompt = a_heard_take(conn, voices, voice, 330.0)

    def delete_now() -> None:
        assert request_example_deletion(conn, OWNER, voice.id, take.id)
        assert process_voice_deletions(conn, voices, OWNER) == 0

    words = PreviewText("take", None, take.id, WORDS)
    test = run_preview(conn, FakeClient(during_speak=delete_now), voices, OWNER,
                       a_job(voice, words))
    assert (test.text_kind, test.example_id, test.prompt_example_id) == ("take", None, prompt.id)
    assert stored_tests(conn)[0]["example_id"] is None


def test_a_test_whose_take_was_deleted_while_it_spoke_is_dropped(conn, voices):
    voice = a_voice(conn, voices)
    [take] = voice.examples

    def delete_now() -> None:
        assert request_example_deletion(conn, OWNER, voice.id, take.id)

    with pytest.raises(PreviewFailed) as failed:
        run_preview(conn, FakeClient(during_speak=delete_now), voices, OWNER, a_job(voice))
    assert str(failed.value) == CHANGED
    assert stored_tests(conn) == []


def test_a_take_heard_while_the_test_waited_is_the_one_it_speaks_from(conn, settings, voices):
    voice = a_voice(conn, voices)
    lock = threading.Lock()
    lock.acquire()
    waited = threading.Event()
    client = FakeClient()
    thread, outcome = in_thread(settings, voices, client, a_job(voice), voice_lock=lock,
                                progress=lambda stage, f: stage == WAITING and waited.set())
    assert waited.wait(timeout=JOIN_TIMEOUT)
    newest = a_heard_take(conn, voices, voice, 330.0)     # heard while the job held the lock
    lock.release()
    joined(thread)
    assert client.references[0][token(voice.id)]["sha256"] == newest.content_hash
    assert outcome["result"].prompt_example_id == newest.id


def test_a_refusal_fails_the_test_with_the_services_words(conn, voices):
    voice = a_voice(conn, voices)
    client = FakeClient(refuse="voice copy is not available: no cloning model")
    with pytest.raises(PreviewFailed) as failed:
        run_preview(conn, client, voices, OWNER, a_job(voice))
    assert "voice copy is not available: no cloning model" in str(failed.value)
    assert len(client.calls) == 1
    assert stored_tests(conn) == []


def test_a_cancel_during_the_call_drops_the_test(conn, voices):
    voice = a_voice(conn, voices)
    flag = threading.Event()
    with pytest.raises(Cancelled):
        run_preview(conn, FakeClient(during_speak=flag.set), voices, OWNER, a_job(voice),
                    cancelled=flag.is_set)
    assert stored_tests(conn) == []
    assert voices.written == []


def test_a_cancel_while_waiting_for_the_voice_model_makes_no_call(conn, settings, voices):
    voice = a_voice(conn, voices)
    lock = threading.Lock()
    lock.acquire()
    stages: list[str] = []
    waited = threading.Event()

    def progress(stage, fraction):
        stages.append(stage)
        if stage == WAITING:
            waited.set()

    client = FakeClient()
    thread, outcome = in_thread(settings, voices, client, a_job(voice), voice_lock=lock,
                                progress=progress, cancelled=waited.is_set)
    joined(thread)
    lock.release()
    assert isinstance(outcome.get("error"), Cancelled)
    assert client.calls == []
    assert WAITING in stages


def test_a_cancel_just_before_the_call_makes_no_call(conn, voices):
    voice = a_voice(conn, voices)
    flag = threading.Event()
    voices.on_read = lambda key: flag.set()
    lock = threading.Lock()
    client = FakeClient()
    with pytest.raises(Cancelled):
        run_preview(conn, client, voices, OWNER, a_job(voice), voice_lock=lock,
                    cancelled=flag.is_set)
    assert client.calls == []
    assert lock.acquire(blocking=False)


def test_a_cancel_reported_with_the_generating_stage_makes_no_call(conn, voices):
    voice = a_voice(conn, voices)
    voice_lock, job_lock = threading.Lock(), threading.Lock()
    flag = threading.Event()

    def progress(stage, fraction):
        # The job's own callback takes the job lock, so it must never be held here.
        assert job_lock.acquire(blocking=False)
        try:
            if stage == "generating the voice":
                flag.set()
        finally:
            job_lock.release()

    # A refusing client: a call made before the cancel check would fail, not cancel.
    client = FakeClient(refuse="test refusal")
    with pytest.raises(Cancelled):
        run_preview(conn, client, voices, OWNER, a_job(voice), voice_lock=voice_lock,
                    job_lock=job_lock, progress=progress, cancelled=flag.is_set)
    assert client.calls == []
    assert stored_tests(conn) == []
    assert voices.written == []
    assert voice_lock.acquire(blocking=False)


def test_a_test_never_changes_the_cast_or_the_takes(conn, voices, memory_group):
    voice = a_voice(conn, voices)
    set_narrator(conn, memory_group, "en", voice.id)

    def cast() -> tuple[list, list]:
        return ([dict(r) for r in conn.execute("SELECT * FROM memory_narrators")],
                [dict(r) for r in conn.execute("SELECT * FROM voice_examples")])

    before = cast()
    run_preview(conn, FakeClient(), voices, OWNER, a_job(voice))
    assert cast() == before


def test_the_stages_come_in_order_and_never_go_back(conn, voices):
    voice = a_voice(conn, voices)
    lock = threading.Lock()
    lock.acquire()
    reported: list[tuple[str, float]] = []

    def progress(stage, fraction):
        reported.append((stage.split(" — ")[0], fraction))
        if stage == WAITING and lock.locked():
            lock.release()

    run_preview(conn, FakeClient(), voices, OWNER, a_job(voice), voice_lock=lock,
                progress=progress)
    stages = [stage for i, (stage, _) in enumerate(reported)
              if i == 0 or reported[i - 1][0] != stage]
    assert stages == [WAITING, "reading the take", "generating the voice",
                      "measuring the voice", "saving the test"]
    fractions = [fraction for _, fraction in reported]
    assert fractions == sorted(fractions)


def test_a_test_stores_its_measure_in_its_one_row(conn, voices):
    voice = a_voice(conn, voices)
    test = run_preview(conn, MeasuringClient(), voices, OWNER, a_job(voice))
    (row,) = stored_tests(conn)
    assert row["likeness_error"] is None
    assert json.loads(row["likeness_json"]) == test.likeness
    assert list(json.loads(row["likeness_json"])) == sorted(test.likeness)
    assert test.likeness["takes_count"] == 1
    assert voice_test(conn, OWNER, voice.id) == test


def test_the_measure_scores_the_stored_wav(conn, voices):
    voice = a_voice(conn, voices)
    client = MeasuringClient()
    test = run_preview(conn, client, voices, OWNER, a_job(voice))
    assert client.embedded[0] == voices.read(test.storage_key)


def test_the_measure_runs_outside_the_voice_lock_and_the_job_lock(conn, voices):
    voice = a_voice(conn, voices)
    voice_lock, job_lock = threading.Lock(), threading.Lock()
    held: list[bool] = []
    client = MeasuringClient(
        during_embed=lambda: held.append(voice_lock.locked() or job_lock.locked()))
    run_preview(conn, client, voices, OWNER, a_job(voice), voice_lock=voice_lock, job_lock=job_lock)
    assert held == [False, False]


def test_the_measure_reports_each_take_between_the_speaking_and_the_saving(conn, voices):
    voice = a_voice(conn, voices)
    add_example(conn, voices, OWNER, voice.id, language="en", source="uploaded", data=clip(240.0))
    reported: list[tuple[str, float]] = []
    run_preview(conn, MeasuringClient(), voices, OWNER, a_job(voice),
                progress=lambda stage, fraction: reported.append((stage, fraction)))
    measuring = [entry for entry in reported if entry[0].startswith("measuring the voice")]
    assert [stage for stage, _ in measuring] == [
        "measuring the voice", "measuring the voice · take 1 of 2", "measuring the voice · take 2 of 2"]
    fractions = [fraction for _, fraction in measuring]
    assert fractions == sorted(fractions)
    assert fractions[0] == pytest.approx(MEASURE_FROM)
    assert fractions[-1] < MEASURE_TO
    assert reported[-1] == ("saving the test", MEASURE_TO)


def test_a_cancel_between_two_embeddings_stops_the_measure_and_stores_nothing(conn, voices):
    voice = a_voice(conn, voices)
    add_example(conn, voices, OWNER, voice.id, language="en", source="uploaded", data=clip(240.0))
    flag = threading.Event()
    client = MeasuringClient(during_embed=lambda: len(client.embedded) == 2 and flag.set())
    with pytest.raises(Cancelled):
        run_preview(conn, client, voices, OWNER, a_job(voice), cancelled=flag.is_set)
    assert len(client.embedded) == 2
    assert stored_tests(conn) == []


def test_a_cancel_before_the_first_embedding_makes_no_call(conn, voices):
    voice = a_voice(conn, voices)
    flag = threading.Event()
    client = MeasuringClient(during_speak=flag.set)
    with pytest.raises(Cancelled):
        run_preview(conn, client, voices, OWNER, a_job(voice), cancelled=flag.is_set)
    assert client.embedded == []


@pytest.mark.parametrize("status", [(409, "a bake is running"), (503, "not offered on this device"),
                                    (422, "the clip holds 0.20 s of voiced audio"),
                                    (500, "ECAPA weights are not on disk")])
def test_a_failed_measure_never_fails_the_test(conn, voices, status):
    voice = a_voice(conn, voices)
    test = run_preview(conn, MeasuringClient(embed_status=status), voices, OWNER, a_job(voice))
    assert test.likeness is None
    assert test.likeness_error == status[1]
    (row,) = stored_tests(conn)
    assert row["likeness_json"] is None
    assert row["likeness_error"] == status[1]


def test_a_client_error_in_the_measure_is_stored_too(conn, voices):
    voice = a_voice(conn, voices)
    test = run_preview(conn, FakeClient(), voices, OWNER, a_job(voice))
    assert test.likeness is None
    assert "embed_voice" in test.likeness_error


def test_a_new_test_replaces_the_old_measure_with_its_own_failure(conn, voices):
    voice = a_voice(conn, voices)
    run_preview(conn, MeasuringClient(), voices, OWNER, a_job(voice))
    run_preview(conn, MeasuringClient(embed_status=(503, "no")), voices, OWNER, a_job(voice))
    (row,) = stored_tests(conn)
    assert (row["likeness_json"], row["likeness_error"]) == (None, "no")


def test_a_new_measure_clears_the_old_failure(conn, voices):
    voice = a_voice(conn, voices)
    run_preview(conn, MeasuringClient(embed_status=(503, "no")), voices, OWNER, a_job(voice))
    run_preview(conn, MeasuringClient(), voices, OWNER, a_job(voice))
    (row,) = stored_tests(conn)
    assert row["likeness_error"] is None and row["likeness_json"] is not None


def test_no_measure_runs_when_the_reply_did_not_speak_from_the_take(conn, voices):
    voice = a_voice(conn, voices)
    client = MeasuringClient(drop_conditioning=True)
    with pytest.raises(PreviewFailed):
        run_preview(conn, client, voices, OWNER, a_job(voice))
    assert client.embedded == []


def test_a_measure_never_changes_the_takes(conn, voices):
    voice = a_voice(conn, voices)
    run_preview(conn, MeasuringClient(), voices, OWNER, a_job(voice))
    assert get_voice(conn, OWNER, voice.id) == voice


def test_a_cancel_while_the_job_lock_is_held_stores_nothing(conn, settings, voices):
    voice = a_voice(conn, voices)
    job_lock = threading.Lock()
    job_lock.acquire()
    saving, flag = threading.Event(), threading.Event()
    saved: list = []
    thread, outcome = in_thread(
        settings, voices, FakeClient(), a_job(voice), job_lock=job_lock,
        on_saved=saved.append, cancelled=flag.is_set,
        progress=lambda stage, f: stage == "saving the test" and saving.set())
    assert saving.wait(timeout=JOIN_TIMEOUT)
    flag.set()
    job_lock.release()
    joined(thread)
    assert isinstance(outcome.get("error"), Cancelled)
    assert stored_tests(conn) == []
    assert saved == []


def test_the_job_lock_is_held_through_the_save(conn, settings, voices):
    voice = a_voice(conn, voices)
    job_lock = threading.Lock()
    writing, go_on = threading.Event(), threading.Event()

    def wait_on_the_test_wav(key: str) -> None:
        if key.endswith(".wav"):
            writing.set()
            go_on.wait(timeout=JOIN_TIMEOUT)

    voices.on_write = wait_on_the_test_wav
    saved: list = []
    thread, outcome = in_thread(settings, voices, FakeClient(), a_job(voice),
                                job_lock=job_lock, on_saved=saved.append)
    assert writing.wait(timeout=JOIN_TIMEOUT)
    assert not job_lock.acquire(blocking=False)
    go_on.set()
    joined(thread)
    assert "error" not in outcome
    assert len(stored_tests(conn)) == 1
    assert saved == [outcome["result"]]
