"""The Voices pages and their deletion drain (design §6, §11.1, §13, §13.1; PLAN §2.9, §2.17 V,
§2.17 K1, K4-K7, D20-D22, D33, D67, D68, D74, D76-D80, D83, D87, D88, D94, D95, D101).

`web/voices_api.py`: the routes, the status codes, the deleting state, the
`VoiceDeletionDrain` that drains the outbox when no worker runs (`inline_drain`), the
hearing job `VoiceHearings`, and the reattach of an earlier bake by `VoiceBakeJobs`, against
a fake models service.
`web/static/voices.js`: its pure parts and its lifecycle, under node.
"""

import hashlib
import html
import io
import json
import pathlib
import re
import shutil
import subprocess
import threading
import time
import wave
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from config import Settings
from feed import saved_voices as sv
from feed import voice_bake
from feed.saved_voices import (
    RECORDING_PROMPTS,
    add_example,
    create_voice,
    request_voice_deletion,
)
from feed.voice_preview import PREVIEW_SENTENCES
from models.catalog import cloning_entry_for
from tests.test_feed_build import WORDS
from tests.test_saved_voices import hand_header, make_wav
from tests.test_voice_bake import BAKE_ID, FOREIGN, FakeService, http_error
from web import voices_api
from web.app import create_app
from web.voices_api import NOT_AVAILABLE, VoiceDeletionDrain

OWNER = 1


def clip(seconds: float = 6.0, *, freq: float = 220.0) -> bytes:
    """A valid example: mono PCM-16 at 48 kHz, a tone loud enough not to be silent."""
    t = np.arange(round(seconds * 48000)) / 48000
    samples = (0.2 * 32767 * np.sin(2 * np.pi * freq * t)).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(48000)
        handle.writeframes(samples.tobytes())
    return buffer.getvalue()


class HeardModels:
    """The models service's `/voice/transcribe` alone: every take holds `WORDS`. It keeps
    the hearing job a saved take starts off the network."""

    def __init__(self) -> None:
        self.heard: list[str] = []

    def transcribe(self, audio: bytes, language: str) -> dict:
        self.heard.append(hashlib.sha256(audio).hexdigest())
        return {"text": WORDS, "model": "fake-asr", "revision": "fake", "seconds": 6.0,
                "elapsed_s": 0.0}


@pytest.fixture(autouse=True)
def heard_models(monkeypatch):
    fake = HeardModels()
    monkeypatch.setattr(Settings, "build_models_client", lambda self: fake)
    return fake


def _client(settings):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return TestClient(create_app(settings))


@pytest.fixture
def client(settings):
    with _client(settings) as test_client:
        yield test_client


@pytest.fixture
def queued(settings):
    """A run with a worker: the request side only enqueues; nothing drains here."""
    settings.inline_drain = False
    with _client(settings) as test_client:
        yield test_client


@pytest.fixture
def jetson(settings):
    settings.profile = "jetson"
    with _client(settings) as test_client:
        yield test_client


def ctx(client):
    return client.app.state.context


def rows(client, table: str) -> int:
    return ctx(client).conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def post_example(client, voice_id, data=None, *, language="en", source="uploaded",
                 **extra):
    return client.post(
        f"/voices/{voice_id}/examples",
        data={"language": language, "source": source} | extra,
        files={"file": ("example.wav", clip() if data is None else data, "audio/wav")},
    )


def leaf(client, voice_id, **params) -> str:
    """The page of one voice, `/voices/{id}`."""
    return client.get(f"/voices/{voice_id}", params=params).text


def body_of(client, voice_id) -> str:
    """The leaf body fragment, `/voices/{id}/list`."""
    return client.get(f"/voices/{voice_id}/list").text


def voice_with_example(client, name="Grandma", language="en", data=None):
    c = ctx(client)
    voice = create_voice(c.conn, OWNER, name)
    example = add_example(c.conn, c.voices, OWNER, voice.id, language=language,
                          source="uploaded", data=data or clip())
    return voice, example


def test_the_leaf_renders_the_voice_and_prompts_and_asks_no_consent(client):
    voice, _ = voice_with_example(client, "Grandma")
    body = leaf(client, voice.id)
    assert "Grandma" in body
    for texts in RECORDING_PROMPTS.values():
        for text in texts.values():
            assert text in body
    assert "voice-consent" not in body
    assert "Add a take" in body and "Record" in body and "Upload a file" in body
    assert "Organize files" not in body
    assert "voices.js" in body


def test_the_grid_has_new_voice_and_the_script(client):
    body = client.get("/voices").text
    assert "<h1>Voices</h1>" in body
    assert ("A voice is a person's own voice. It can narrate a film in each language it has a "
            "take of.") in body
    assert "New voice" in body and 'action="/voices"' in body
    assert "voices.js" in body


def test_the_add_example_select_lists_only_cloning_languages(client):
    voice, _ = voice_with_example(client)
    body = leaf(client, voice.id)
    select = re.search(r'<select class="voice-language".*?</select>', body, re.DOTALL).group(0)
    assert 'value="en"' in select and 'value="ru"' in select


def test_create_returns_the_cards_with_the_new_one_marked(client):
    voice, _ = voice_with_example(client, "Babushka")
    response = client.post("/voices", data={"name": "  Grandma "})
    assert response.status_code == 200
    assert response.text.lstrip().startswith('<div id="voice-cards"')
    assert "<html" not in response.text
    created = sv.list_voices(ctx(client).conn, OWNER)
    grandma = next(v for v in created if v.name == "Grandma")
    cards = re.findall(r'<a class="voice-card"[^>]*>', response.text)
    assert [card for card in cards if "data-created" in card] == [
        f'<a class="voice-card" href="/voices/{grandma.id}" data-voice-id="{grandma.id}" data-created>']
    assert f'href="/voices/{voice.id}" data-voice-id="{voice.id}">' in response.text


def test_rename_returns_the_fragment(client):
    voice, _ = voice_with_example(client)
    response = client.post(f"/voices/{voice.id}/rename", data={"name": "Nana"})
    assert response.status_code == 200
    assert 'value="Nana"' in response.text
    assert sv.get_voice(ctx(client).conn, OWNER, voice.id).name == "Nana"


def test_the_list_endpoints_answer_their_fragments(client):
    voice, _ = voice_with_example(client, "Grandma")
    cards = client.get("/voices/list")
    assert cards.status_code == 200
    assert cards.text.lstrip().startswith('<div id="voice-cards"') and "Grandma" in cards.text
    assert "data-created" not in cards.text
    body = client.get(f"/voices/{voice.id}/list")
    assert body.status_code == 200
    assert body.text.lstrip().startswith('<div id="voices"') and "Grandma" in body.text


def test_an_uploaded_example_is_stored(client):
    voice = create_voice(ctx(client).conn, OWNER, "Grandma")
    data = clip(7.0)
    response = post_example(client, voice.id, data, source="recorded")
    assert response.status_code == 200
    stored = sv.get_voice(ctx(client).conn, OWNER, voice.id).examples
    assert len(stored) == 1 and stored[0].source == "recorded"
    assert ctx(client).voices.read(stored[0].storage_key) == data
    assert "0:07" in response.text and "recorded" in response.text


def test_the_audio_of_an_example_is_served(client):
    voice, example = voice_with_example(client)
    response = client.get(f"/voices/{voice.id}/examples/{example.id}/audio")
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.content == ctx(client).voices.read(example.storage_key)


@pytest.mark.parametrize("name", ["", "   ", "x" * (sv.NAME_MAX + 1)])
def test_a_bad_name_is_400(client, name):
    response = client.post("/voices", data={"name": name})
    assert response.status_code == 400
    assert response.json()["detail"]


def test_a_duplicate_name_is_409(client):
    client.post("/voices", data={"name": "Grandma"})
    response = client.post("/voices", data={"name": "GRANDMA"})
    assert response.status_code == 409
    assert "already exists" in response.json()["detail"]


def test_an_unknown_voice_is_404(client):
    assert client.get("/voices/99").status_code == 404
    assert client.get("/voices/99/list").status_code == 404
    assert client.post("/voices/99/film-example",
                       data={"language": "en", "example_id": "1"}).status_code == 404
    assert client.post("/voices/99/examples/1/preview").status_code == 404
    assert client.get("/voices/99/examples/1/preview/audio").status_code == 404
    assert client.post("/voices/99/rename", data={"name": "x"}).status_code == 404
    assert client.post("/voices/99/delete").status_code == 404
    assert post_example(client, 99).status_code == 404
    assert client.post("/voices/99/examples/1/delete").status_code == 404
    assert client.get("/voices/99/examples/1/audio").status_code == 404


def test_an_unknown_example_is_404(client):
    voice, _ = voice_with_example(client)
    assert client.post(f"/voices/{voice.id}/examples/99/delete").status_code == 404
    assert client.get(f"/voices/{voice.id}/examples/99/audio").status_code == 404


def test_a_foreign_voice_is_404(client):
    c = ctx(client)
    other = create_voice(c.conn, 2, "Theirs")
    assert client.post(f"/voices/{other.id}/rename", data={"name": "x"}).status_code == 404
    assert client.post(f"/voices/{other.id}/delete").status_code == 404
    assert "Theirs" not in client.get("/voices").text
    for path in (f"/voices/{other.id}", f"/voices/{other.id}/list"):
        assert client.get(path).status_code == 404


@pytest.mark.parametrize(
    "language, source, text", [("de", "uploaded", "language"), ("en", "cloned", "source")]
)
def test_an_unknown_language_or_source_is_400(client, language, source, text):
    voice = create_voice(ctx(client).conn, OWNER, "Grandma")
    response = post_example(client, voice.id, language=language, source=source)
    assert response.status_code == 400
    assert text in response.json()["detail"]


@pytest.mark.parametrize(
    "seconds, text", [(4.99, "at least"), (30.01, "at most")]
)
def test_an_example_out_of_bounds_is_400(client, seconds, text):
    voice = create_voice(ctx(client).conn, OWNER, "Grandma")
    response = post_example(client, voice.id, clip(seconds))
    assert response.status_code == 400
    assert text in response.json()["detail"]


def test_a_silent_example_is_400(client):
    voice = create_voice(ctx(client).conn, OWNER, "Grandma")
    response = post_example(client, voice.id, make_wav(6.0, amp=0.0))
    assert response.status_code == 400
    assert "silent" in response.json()["detail"]


@pytest.mark.parametrize(
    "data, text",
    [
        (make_wav(6.0, width=1), "8-bit mono at 48000 Hz"),
        (make_wav(6.0, width=3), "24-bit mono at 48000 Hz"),
        (make_wav(6.0, width=4), "32-bit mono at 48000 Hz"),
        (make_wav(6.0, channels=2), "16-bit stereo at 48000 Hz"),
        (make_wav(6.0, channels=6), "16-bit 6-channel at 48000 Hz"),
        (make_wav(6.0, rate=44100), "16-bit mono at 44100 Hz"),
        (make_wav(6.0, rate=16000), "16-bit mono at 16000 Hz"),
    ],
)
def test_a_wrong_format_is_400_naming_the_format_found(client, data, text):
    voice = create_voice(ctx(client).conn, OWNER, "Grandma")
    response = post_example(client, voice.id, data)
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "must be mono 16-bit PCM at 48000 Hz" in detail and text in detail


@pytest.mark.parametrize(
    "data, text",
    [
        (hand_header(3), None),               # IEEE float
        (hand_header(2, width=2), None),      # ADPCM (compressed)
        (b"this is not audio at all, just some garbage bytes", "not a WAV"),
        (b"", "not a WAV"),
        (make_wav(0.0), "no audio"),
        (clip(6.0)[:-1000], "cut short"),
    ],
    ids=["float", "compressed", "garbage", "empty", "zero-frames", "truncated"],
)
def test_a_file_that_is_not_a_pcm_wav_is_400(client, data, text):
    voice = create_voice(ctx(client).conn, OWNER, "Grandma")
    response = post_example(client, voice.id, data)
    assert response.status_code == 400
    if text is not None:
        assert text in response.json()["detail"]
    assert rows(client, "voice_examples") == 0


def test_an_example_over_4_mib_is_413(client):
    voice = create_voice(ctx(client).conn, OWNER, "Grandma")
    response = post_example(client, voice.id, b"\0" * (sv.MAX_BYTES + 1))
    assert response.status_code == 413
    assert rows(client, "voice_examples") == 0


def test_low_disk_is_507(client, monkeypatch):
    voice = create_voice(ctx(client).conn, OWNER, "Grandma")
    monkeypatch.setattr(ctx(client).voices, "free_bytes", lambda: 0)
    response = post_example(client, voice.id)
    assert response.status_code == 507
    assert rows(client, "voice_examples") == 0


def test_a_duplicate_example_is_400(client):
    voice, _ = voice_with_example(client)
    response = post_example(client, voice.id)
    assert response.status_code == 400
    assert "already has" in response.json()["detail"]


def test_jetson_shows_the_notice_and_refuses_with_409(jetson):
    voice = create_voice(ctx(jetson).conn, OWNER, "Grandma")
    body = jetson.get("/voices").text
    assert "Voice copy is not available on this device." in body
    assert "voices-new" not in body and "New voice" not in body
    assert f'<a class="voice-card" href="/voices/{voice.id}"' in body   # still opens
    page = leaf(jetson, voice.id)
    assert "Voice copy is not available on this device." in page
    assert 'class="voice-record"' not in page and "voice-upload" not in page
    assert "voice-rename" in page and "voice-delete" in page
    assert jetson.post("/voices", data={"name": "Nana"}).status_code == 409
    response = post_example(jetson, voice.id)
    assert response.status_code == 409
    assert response.json()["detail"] == "Voice copy is not available on this device."
    renamed = jetson.post(f"/voices/{voice.id}/rename", data={"name": "Nana"})
    assert renamed.status_code == 200
    assert jetson.post(f"/voices/{voice.id}/delete").status_code == 200
    jetson.app.state.await_voice_drain()
    assert rows(jetson, "saved_voices") == 0


def test_a_voice_delete_only_enqueues(queued):
    voice, example = voice_with_example(queued)
    response = queued.post(f"/voices/{voice.id}/delete")
    assert response.status_code == 200
    assert "deleting…" in response.text and "data-deleting" in response.text
    assert rows(queued, "saved_voices") == 1 and rows(queued, "voice_examples") == 1
    assert ctx(queued).voices.exists(example.storage_key)
    assert rows(queued, "voice_deletions") == 1
    assert queued.app.state.voice_drain.thread is None


def test_an_example_delete_only_enqueues(queued):
    voice, example = voice_with_example(queued)
    response = queued.post(f"/voices/{voice.id}/examples/{example.id}/delete")
    assert response.status_code == 200
    assert "deleting…" in response.text
    assert rows(queued, "voice_examples") == 1
    assert ctx(queued).voices.exists(example.storage_key)
    assert queued.app.state.voice_drain.thread is None


def test_a_second_delete_returns_the_same_fragment_and_one_intent(queued):
    voice, _ = voice_with_example(queued)
    first = queued.post(f"/voices/{voice.id}/delete")
    second = queued.post(f"/voices/{voice.id}/delete")
    assert second.status_code == 200 and second.text == first.text
    assert rows(queued, "voice_deletions") == 1


def test_a_pending_voice_takes_no_edit(queued):
    voice, example = voice_with_example(queued)
    queued.post(f"/voices/{voice.id}/delete")
    assert queued.post(f"/voices/{voice.id}/rename", data={"name": "x"}).status_code == 409
    assert post_example(queued, voice.id, clip(8.0)).status_code == 409
    assert queued.get(
        f"/voices/{voice.id}/examples/{example.id}/audio"
    ).status_code == 404


def test_the_audio_of_a_pending_example_is_404(queued):
    voice, example = voice_with_example(queued)
    queued.post(f"/voices/{voice.id}/examples/{example.id}/delete")
    assert queued.get(
        f"/voices/{voice.id}/examples/{example.id}/audio"
    ).status_code == 404


def test_a_pending_voice_row_has_no_controls(queued):
    voice, _ = voice_with_example(queued)
    body = queued.post(f"/voices/{voice.id}/delete").text
    assert "Grandma" in body
    assert "voice-rename" not in body and "voice-delete" not in body
    assert "voice-add" not in body and "<audio" not in body


def test_after_the_drain_the_voice_is_gone_and_a_delete_is_404(queued):
    voice, example = voice_with_example(queued)
    queued.post(f"/voices/{voice.id}/delete")
    c = ctx(queued)
    assert sv.process_voice_deletions(c.conn, c.voices, OWNER) == 0
    body = queued.get("/voices/list").text
    assert "Grandma" not in body and "data-deleting" not in body
    assert queued.get(f"/voices/{voice.id}/list").status_code == 404
    assert not c.voices.exists(example.storage_key)
    assert queued.post(f"/voices/{voice.id}/delete").status_code == 404


def test_the_delete_reply_does_not_wait_for_the_drain(client, monkeypatch):
    voice, example = voice_with_example(client)
    release = threading.Event()
    real = sv.process_voice_deletions

    def slow(conn, storage, owner_id):
        release.wait(5)
        return real(conn, storage, owner_id)

    monkeypatch.setattr(voices_api, "process_voice_deletions", slow)
    response = client.post(f"/voices/{voice.id}/delete")
    assert response.status_code == 200 and "deleting…" in response.text
    assert not release.is_set()
    release.set()
    client.app.state.await_voice_drain()
    assert rows(client, "saved_voices") == 0 and rows(client, "voice_deletions") == 0
    assert not ctx(client).voices.exists(example.storage_key)


def test_two_deletes_during_one_drain_use_one_thread(client, monkeypatch):
    first, _ = voice_with_example(client, "Grandma")
    second, _ = voice_with_example(client, "Grandpa", data=clip(freq=330.0))
    release = threading.Event()
    threads: set[int] = set()
    real = sv.process_voice_deletions

    def slow(conn, storage, owner_id):
        threads.add(threading.get_ident())
        release.wait(5)
        return real(conn, storage, owner_id)

    monkeypatch.setattr(voices_api, "process_voice_deletions", slow)
    client.post(f"/voices/{first.id}/delete")
    client.post(f"/voices/{second.id}/delete")
    release.set()
    client.app.state.await_voice_drain()
    assert len(threads) == 1
    assert rows(client, "saved_voices") == 0 and rows(client, "voice_deletions") == 0
    assert not client.app.state.voice_drain.running


def _pending_library(settings):
    """A library with a voice whose deletion was requested before the app stopped."""
    settings.inline_drain = False
    with _client(settings) as first:
        voice, example = voice_with_example(first)
        request_voice_deletion(ctx(first).conn, OWNER, voice.id)
        return example.storage_key


def test_start_up_drains_a_pending_intent(settings):
    key = _pending_library(settings)
    settings.inline_drain = True
    with _client(settings) as client:
        client.app.state.await_voice_drain()
        assert rows(client, "saved_voices") == 0 and rows(client, "voice_deletions") == 0
        assert not ctx(client).voices.exists(key)


def test_start_up_with_a_worker_leaves_the_intent_for_it(settings):
    key = _pending_library(settings)
    with _client(settings) as client:
        assert client.app.state.voice_drain.thread is None
        assert rows(client, "voice_deletions") == 1
        assert ctx(client).voices.exists(key)


class FlakyStorage:
    """The voices store, whose first `delete` fails."""

    def __init__(self, real):
        self.real = real
        self.failures = 1

    def delete(self, key):
        if self.failures:
            self.failures -= 1
            raise OSError("disk busy")
        self.real.delete(key)

    def __getattr__(self, name):
        return getattr(self.real, name)


def test_a_failed_unlink_is_retried(client, monkeypatch):
    voice, example = voice_with_example(client)
    flaky = FlakyStorage(ctx(client).voices)
    monkeypatch.setattr(ctx(client), "voices", flaky)
    monkeypatch.setattr(voices_api, "RETRY_SECONDS", 0)
    client.post(f"/voices/{voice.id}/delete")
    client.app.state.await_voice_drain()
    assert flaky.failures == 0
    assert not flaky.real.exists(example.storage_key)
    assert rows(client, "voice_deletions") == 0 and rows(client, "voice_file_deletions") == 0


def test_a_schedule_in_the_exit_window_gets_one_more_pass(monkeypatch):
    go = threading.Event()
    passes: list[int] = []

    def stub(conn, storage, owner_id):
        go.wait(5)
        passes.append(1)
        return 0

    monkeypatch.setattr(voices_api, "process_voice_deletions", stub)
    context = SimpleNamespace(
        settings=SimpleNamespace(inline_drain=True, owner_id=OWNER), conn=None, voices=None
    )
    drain = VoiceDeletionDrain(lambda: context)
    drain.schedule()
    with drain.lock:
        go.set()                        # the pass returns 0; the thread waits for the lock
        time.sleep(0.1)
        late = threading.Thread(target=drain.schedule)
        late.start()                    # ...and so does this schedule()
        time.sleep(0.1)
    late.join()
    drain.join()
    assert len(passes) == 2
    assert not drain.running and not drain.rerun


def test_an_add_example_wakes_the_drain_for_a_pending_key(client):
    c = ctx(client)
    c.voices.write("orphan.cond", b"left by a dead write")
    c.conn.execute(
        "INSERT INTO voice_file_deletions(storage_key, deletion_id, requested_at)"
        " VALUES ('orphan.cond', NULL, 'now')"
    )
    voice = create_voice(c.conn, OWNER, "Grandma")
    assert post_example(client, voice.id).status_code == 200
    client.app.state.await_voice_drain()
    assert not c.voices.exists("orphan.cond")
    assert rows(client, "voice_file_deletions") == 0


def test_an_add_example_with_nothing_pending_starts_no_thread(client):
    voice = create_voice(ctx(client).conn, OWNER, "Grandma")
    assert post_example(client, voice.id).status_code == 200
    assert client.app.state.voice_drain.thread is None


TIP = ("Record clean, normal speech in a quiet room. Films and tests of «Grandma» speak from "
       "its newest take of the language, once its words are heard.")


def _voice_adds(body: str) -> list[str]:
    return re.findall(r'<div class="voice-add".*?</div>', body, re.DOTALL)


@pytest.mark.parametrize("path", ["/voices/{id}", "/voices/{id}/list"])
def test_add_a_take_shows_the_recording_tip_before_record(client, path):
    voice, _ = voice_with_example(client, "Grandma")
    adds = _voice_adds(client.get(path.format(id=voice.id)).text)
    assert len(adds) == 1
    for add in adds:
        assert '<span class="voice-add-title">Add a take</span>' in add
        assert f'<p class="voices-tip muted">{TIP}</p>' in add
        assert add.index("voices-tip") < add.index('class="voice-record"')
        assert add.index("voices-tip") < add.index("Upload a file")


def test_every_reading_text_is_in_the_recorder_with_its_language_and_id(client):
    voice = create_voice(ctx(client).conn, OWNER, "Grandma")
    body = leaf(client, voice.id)
    recorder = body[body.index('id="voice-recorder"'):]
    assert body.index('id="voices"') < body.index('id="voice-recorder"')
    assert "recorder-prompt" not in body[body.index('id="voices"'):body.index('id="voice-recorder"')]
    for language, texts in RECORDING_PROMPTS.items():
        for prompt_id, text in texts.items():
            assert (f'<p class="recorder-prompt" data-lang="{language}" '
                    f'data-prompt-id="{prompt_id}" hidden>{text}</p>') in recorder


def test_the_preparation_buttons_sit_in_the_recorder_outside_the_list(client):
    voice, _ = voice_with_example(client)
    body = leaf(client, voice.id)
    listing = body[body.index('id="voices"'):body.index('id="voice-recorder"')]
    recorder = body[body.index('id="voice-recorder"'):]
    assert '<div class="recorder-prepare" hidden>' in recorder
    assert '<button type="button" class="recorder-next">Different text</button>' in recorder
    assert '<button type="button" class="recorder-start">Start recording</button>' in recorder
    assert "recorder-next" not in listing and "recorder-start" not in listing


def _take_row_of(client, column: str):
    return ctx(client).conn.execute(
        f"SELECT {column} FROM voice_examples ORDER BY id DESC LIMIT 1").fetchone()


@pytest.fixture
def held_hearings(client, monkeypatch):
    """The hearing job a saved take starts writes the heard words into the row on its own
    thread; held here, the row is read as the add route wrote it."""
    started: list[int] = []
    monkeypatch.setattr(client.app.state.voice_hearings, "start", started.append)
    return started


def test_a_recording_stores_its_kit_text_and_no_words(client, held_hearings):
    voice = create_voice(ctx(client).conn, OWNER, "Babushka")
    response = post_example(client, voice.id, language="ru", source="recorded", prompt_id="ru-rain")
    assert response.status_code == 200
    assert held_hearings == [voice.id]
    stored = _take_row_of(client, "kit_id, transcript, transcript_model, transcript_error")
    assert tuple(stored) == ("ru-rain", None, None, None)


def test_a_recording_without_a_prompt_id_stores_no_kit_text(client, held_hearings):
    voice = create_voice(ctx(client).conn, OWNER, "Babushka")
    assert post_example(client, voice.id, language="ru", source="recorded").status_code == 200
    assert held_hearings == [voice.id]
    assert tuple(_take_row_of(client, "kit_id, transcript")) == (None, None)


def test_the_add_route_stores_no_words_even_when_the_form_carries_them(client, held_hearings):
    voice = create_voice(ctx(client).conn, OWNER, "Babushka")
    response = post_example(client, voice.id, source="recorded", prompt_id="en-rain",
                            transcript="The words I typed.")
    assert response.status_code == 200
    assert held_hearings == [voice.id]
    assert tuple(_take_row_of(client, "kit_id, transcript, transcript_model")) == (
        "en-rain", None, None)


def test_a_prompt_id_of_another_language_is_400(client):
    voice = create_voice(ctx(client).conn, OWNER, "Babushka")
    response = post_example(client, voice.id, language="ru", source="recorded", prompt_id="en-rain")
    assert response.status_code == 400
    assert response.json()["detail"] == "Unknown reading text for Russian: 'en-rain'."
    assert rows(client, "voice_examples") == 0


def test_the_script_sends_the_prompt_id_and_unwinds_esc_in_order():
    import pathlib

    source = pathlib.Path("web/static/voices.js").read_text()
    save = source[source.index("async function saveTake"):]
    assert "body.append('prompt_id', take.promptId)" in save[:save.index("\n  }\n")]
    escape = source[source.index("if (event.key !== 'Escape'"):]
    escape = escape[:escape.index("\n  });\n")]
    order = [escape.index(step) for step in (
        "if (live) { stopRecording(true); return; }", "if (take) { discardTake(); return; }",
        "if (prepared) { closePanel(); return; }",
        "active.matches('input[type=\"text\"], textarea')) { active.blur(); return; }",
        "close();")]
    assert order == sorted(order)
    # Tabs and the test are no Esc level any more (§2.17 V4).
    assert "closePreview" not in source and "previewTarget" not in escape


def _js_function(source: str, head: str) -> str:
    body = source[source.index(head):]
    return body[:body.index("\n  }\n")]


def test_the_microphone_wait_is_live_and_a_cancelled_wait_ends_quietly():
    import pathlib

    source = pathlib.Path("web/static/voices.js").read_text()
    start = _js_function(source, "async function startRecording")
    wait = start.index("await navigator.mediaDevices.getUserMedia")
    # The session is LIVE before the await, so every `live || take` guard blocks the wait.
    assert start.index("live = session;") < wait
    assert start.index("openPanel(target, true);") < wait
    after = start[wait:]
    # A late deny of a cancelled wait shows nothing; a late allow releases the microphone.
    denied = after[after.index("catch (_)"):]
    assert denied.index("if (live !== session) { return; }") < denied.index("showError(")
    allowed = after[after.index("if (live !== session) {", after.index("catch (_)") + 1):]
    assert allowed.index("track.stop()") < allowed.index("new MediaRecorder")
    # onstop touches the page only for its own session.
    onstop = after[after.index("recorder.onstop"):]
    assert onstop.index("if (live !== session) { return; }") < onstop.index("live = null;")


def test_stop_or_esc_during_the_wait_closes_the_panel_and_upload_waits():
    import pathlib

    source = pathlib.Path("web/static/voices.js").read_text()
    stop = _js_function(source, "function stopRecording")
    assert "if (!live.recorder) { live = null; closePanel(); return; }" in stop
    change = source[source.index("document.addEventListener('change'"):]
    change = change[:change.index("\n  });\n")]
    assert change.index("if (live || take) { return; }") < change.index("makeTake(")


def test_the_decode_wait_is_the_unsaved_take_and_a_discarded_one_ends_quietly():
    import pathlib

    source = pathlib.Path("web/static/voices.js").read_text()
    make = _js_function(source, "async function makeTake")
    held = make.index("take = pending;")
    assert held < make.index("await blob.arrayBuffer()") < make.index("await context.decodeAudioData")
    assert make.index("try {") < make.index("new AudioContext")
    stale = "if (take !== pending) { return; }"
    assert make.index(stale) < make.index("await context.decodeAudioData")
    after_decode = make[make.index("} finally {"):]
    assert "if (context) { context.close(); }" in after_decode[:after_decode.index("\n    }\n")]
    assert after_decode.index(stale) < after_decode.index("MIN_SECONDS") < after_decode.index("createObjectURL")
    failed = make[make.index("catch (_)"):make.index("} finally {")]
    assert failed.index(stale) < failed.index("take = null;") < failed.index("showError(")
    too_long = after_decode[after_decode.index("if (seconds < MIN_SECONDS"):]
    assert too_long.index("take = null;") < too_long.index("showError(")
    assert "if (!take || !take.wav) { return; }" in _js_function(source, "async function saveTake")
    discard = _js_function(source, "function discardTake")
    assert "if (take.url) { URL.revokeObjectURL(take.url); }" in discard
    assert "if (live || take) { return; }" in _js_function(source, "function prepare")
    assert "if (!prepared || live || take) { return; }" in _js_function(source, "async function startRecording")
    assert "upload && (live || take ||" in source


REVISION = cloning_entry_for("mac", "en").source.revision
ASR = "fake-asr@fake"


def _example_row(body: str, example_id: int) -> str:
    rows_ = [row for row in re.findall(r'<li class="voice-example.*?</li>', body, re.DOTALL)
             if f">#{example_id}<" in row]
    assert len(rows_) == 1
    return rows_[0]


def _kit_row(body: str, prompt_id: str) -> str:
    start = body.index(f'<li class="kit-row" data-lang="{prompt_id[:2]}" data-prompt-id="{prompt_id}">')
    end = body.index('<li class="kit-row"', start + 1) if body.find('<li class="kit-row"', start + 1) > 0 \
        else body.index("</ul>", start)
    return body[start:end]


def _part(body: str, name: str) -> str:
    """The server-rendered block of one tab inside #voices."""
    start = body.index(f'data-tab-part="{name}"')
    return body[start:body.index("</section>", start)]


def _text(fragment: str) -> str:
    """The visible text of a fragment, one space between words."""
    return " ".join(re.sub(r"<[^>]+>", " ", fragment).split())


def _options(select: str) -> list[tuple[str, str, bool, bool]]:
    """(value, label, selected, disabled) of each option."""
    return [(value, label, " selected" in attrs, " disabled" in attrs)
            for value, attrs, label in re.findall(
                r'<option value="([^"]*)"([^>]*)>(.*?)</option>', select, re.DOTALL)]


def _add(client, voice_id, *, language="en", source="uploaded", freq=330.0, prompt_id=None,
         seconds=6.0):
    c = ctx(client)
    return add_example(c.conn, c.voices, OWNER, voice_id, language=language, source=source,
                       data=clip(seconds, freq=freq), prompt_id=prompt_id)


def _hear(client, voice_id, example, text):
    """What the hearing job stores for one take (D91)."""
    assert sv.set_heard_words(ctx(client).conn, OWNER, voice_id, example.id, text=text,
                              model=ASR) is not None


def _hear_all(client, voice_id):
    """What the hearing job stores: `WORDS` on every take of the voice that waits for them."""
    c = ctx(client)
    for take in sv.takes_to_hear(c.conn, OWNER, voice_id):
        _hear(client, voice_id, take, WORDS)


def _drain_delete(client, voice_id, example_id):
    assert client.post(f"/voices/{voice_id}/examples/{example_id}/delete").status_code == 200
    client.app.state.await_voice_drain()


def _store_test(client, voice_id, *, text_kind="sentence", example_id=None, text="Hello there.",
                audio=None, language="en"):
    c = ctx(client)
    return sv.save_test(
        c.conn, c.voices, OWNER, voice_id, language=language, text_kind=text_kind,
        text_id="EN-1" if text_kind == "sentence" else None, example_id=example_id, text=text,
        prompt=sv.film_take(c.conn, OWNER, voice_id, language).take, model="voxcpm2",
        revision=REVISION, elapsed_seconds=12.34, audio=audio or clip(2.0, freq=550.0))


def _likeness(example, **fields) -> dict:
    """A measure of one take, as `feed/voice_likeness.py` stores it."""
    return {"cosine": 0.3125, "own": {"min": 0.7212, "median": 0.8, "max": 0.8469}, "skipped": [],
            "reference_hash": hashlib.sha256(example.content_hash.encode()).hexdigest()} | fields


def _measure_line(client, voice_id) -> str:
    compare = _compare(body_of(client, voice_id))
    return html.unescape(re.search(r'test-likeness">(.*?)</span>', compare).group(1))


def _store_measured(client, voice_id, **kwargs):
    c = ctx(client)
    return sv.save_test(
        c.conn, c.voices, OWNER, voice_id, language="en", text_kind="sentence", text_id="EN-1",
        example_id=None, text="Hello there.", prompt=sv.film_take(c.conn, OWNER, voice_id, "en").take,
        model="voxcpm2", revision=REVISION, elapsed_seconds=1.0, audio=clip(2.0, freq=550.0), **kwargs)


def test_the_measure_reads_as_a_cosine_beside_the_own_take_range(client):
    voice, example = voice_with_example(client)
    _hear_all(client, voice.id)
    _store_measured(client, voice.id, likeness=_likeness(example))
    assert _measure_line(client, voice.id) == "Speaker similarity 0.31 (cosine) · your own takes 0.72-0.85"


def test_under_two_takes_the_line_has_no_range_and_keeps_the_score(client):
    voice, example = voice_with_example(client)
    _hear_all(client, voice.id)
    _store_measured(client, voice.id, likeness=_likeness(example, own=None))
    assert _measure_line(client, voice.id) == "Speaker similarity 0.31 (cosine)"


def test_a_measure_never_reads_as_a_percent(client):
    voice, example = voice_with_example(client)
    _hear_all(client, voice.id)
    _store_measured(client, voice.id, likeness=_likeness(example))
    assert "%" not in _measure_line(client, voice.id)


def test_a_new_take_marks_the_measure_as_older(client):
    voice, example = voice_with_example(client)
    _hear_all(client, voice.id)
    _store_measured(client, voice.id, likeness=_likeness(example))
    post_example(client, voice.id, clip(6.0, freq=300.0))
    client.app.state.await_voice_hearing()
    assert _measure_line(client, voice.id).endswith("· measured against older takes")


def test_a_failed_measure_names_its_reason_and_no_number(client):
    voice, _ = voice_with_example(client)
    _hear_all(client, voice.id)
    _store_measured(client, voice.id, likeness_error="a bake <is> running")
    assert _measure_line(client, voice.id) == "Speaker similarity not measured: a bake <is> running"
    assert "a bake <is>" not in _compare(body_of(client, voice.id))


def test_a_test_from_before_the_meter_reads_not_measured(client):
    voice, _ = voice_with_example(client)
    _hear_all(client, voice.id)
    _store_test(client, voice.id)
    assert _measure_line(client, voice.id) == "Speaker similarity not measured"


def test_no_part_of_the_leaf_or_the_grid_names_a_take_for_films_or_asks_for_words(client):
    voice, heard = voice_with_example(client)
    _hear(client, voice.id, heard, "Hello there, this is me.")
    _hear_all(client, voice.id)
    silent = _add(client, voice.id, freq=440.0)
    _hear(client, voice.id, silent, "♪")
    _add(client, voice.id, language="ru", source="recorded", prompt_id="ru-rain")
    _store_test(client, voice.id)
    pages = {"leaf": leaf(client, voice.id), "body": body_of(client, voice.id),
             "grid": client.get("/voices").text}
    for name, page in pages.items():
        for gone in ('class="example-preview"', "film-take-select", 'class="take-compare"',
                     "films use take", "Films: take", "one example per language",
                     "chosen for films", "What you said", "words checked", "check the words",
                     "needs its words", "need their words", "Save words", "(checked)",
                     "not checked", "Your checked words", "no speech", "edited"):
            assert gone not in page, (name, gone)


def test_the_old_per_take_routes_and_a_words_route_are_gone(client):
    voice, example = voice_with_example(client)
    e = example.id
    assert client.post(f"/voices/{voice.id}/film-example",
                       data={"language": "en", "example_id": str(e)}).status_code in (404, 405)
    assert client.post(f"/voices/{voice.id}/examples/{e}/preview",
                       data={"text_kind": "sentence", "text_id": "EN-1"}).status_code in (404, 405)
    assert client.get(f"/voices/{voice.id}/examples/{e}/preview/audio").status_code in (404, 405)
    assert client.post(f"/voices/{voice.id}/examples/{e}/transcript",
                       data={"text": "My words."}).status_code in (404, 405)
    assert ctx(client).conn.execute("SELECT transcript FROM voice_examples").fetchone()[0] is None


def test_the_recordings_tab_shows_the_kit_of_every_language_also_with_no_takes(client):
    voice = create_voice(ctx(client).conn, OWNER, "Grandma")
    recordings = _part(leaf(client, voice.id), "recordings")
    for language, name in (("en", "English"), ("ru", "Russian")):
        kit = sv.RECORDING_KIT[language]
        assert len(kit) == 26
        assert f'<h2 class="voice-lang">{name} · 0 of 26 kit texts · 0:00 recorded</h2>' in recordings
        for text in kit:
            row = _kit_row(recordings, text.id)
            assert f'<span class="kit-id">{text.id}</span>' in row
            assert '<button type="button" class="kit-record">Record</button>' in row
    for label, instruction in sv.KIT_STYLES.values():
        assert recordings.count(f'<h3 class="kit-style">{label} — {instruction}</h3>') == 2
    assert "Other takes" not in recordings


def test_a_take_sits_on_its_kit_row_and_the_other_takes_follow(client):
    c = ctx(client)
    voice = create_voice(c.conn, OWNER, "Grandma")
    on_kit = _add(client, voice.id, source="recorded", prompt_id="en-rain", freq=220.0)
    no_kit = _add(client, voice.id, source="recorded", freq=330.0)
    upload = _add(client, voice.id, freq=440.0, seconds=7.0)
    recordings = _part(leaf(client, voice.id), "recordings")
    assert '<h2 class="voice-lang">English · 1 of 26 kit texts · 0:19 recorded</h2>' in recordings
    row = _kit_row(recordings, "en-rain")
    assert f">#{on_kit.id}<" in row
    assert "kit-record" not in row[:row.index('<ul class="voice-examples">')]
    take = _text(_example_row(recordings, on_kit.id))
    assert take == (f"#{on_kit.id} 0:06 recorded {on_kit.created_at[:10]} words not heard yet "
                    "Record again Delete")
    assert f'src="/voices/{voice.id}/examples/{on_kit.id}/audio"' in _example_row(recordings, on_kit.id)
    assert "data-confirm" not in _example_row(recordings, on_kit.id)   # a delete asks nothing (D87)
    other = recordings[recordings.index('<h3 class="kit-style">Other takes</h3>'):]
    assert other.index(f">#{upload.id}<") < other.index(f">#{no_kit.id}<")   # newest first
    assert f">#{on_kit.id}<" not in other
    for loose in (upload, no_kit):
        assert "kit-record" not in _example_row(recordings, loose.id)
    assert _text(_example_row(other, upload.id)) == (
        f"#{upload.id} 0:07 uploaded {upload.created_at[:10]} words not heard yet Delete")


def test_a_take_with_no_usable_words_says_why_on_its_row(client):
    voice, heard = voice_with_example(client)
    silent = _add(client, voice.id, freq=330.0)
    long = _add(client, voice.id, freq=440.0)
    waiting = _add(client, voice.id, freq=550.0)
    _hear(client, voice.id, heard, "Hello there.")
    _hear(client, voice.id, silent, " ♪ … ")
    _hear(client, voice.id, long, "word " * 250)
    body = leaf(client, voice.id)
    assert ("not used: the speech recognition found no words in it"
            in _text(_example_row(body, silent.id)))
    assert ("not used: it holds more than 1000 characters of words"
            in _text(_example_row(body, long.id)))
    for fine in (heard, waiting):
        assert "not used" not in _example_row(body, fine.id)
    assert "words not heard yet" in _text(_example_row(body, waiting.id))
    assert "words not heard yet" not in _example_row(body, heard.id)
    assert "Hello there." not in _part(body, "recordings")   # no row shows the heard words


def test_takes_of_a_language_voice_copy_does_not_speak_stay_with_no_record(jetson):
    c = ctx(jetson)
    voice = create_voice(c.conn, OWNER, "Grandma")
    take = add_example(c.conn, c.voices, OWNER, voice.id, language="en", source="recorded",
                       data=clip(), prompt_id="en-rain")
    recordings = _part(leaf(jetson, voice.id), "recordings")
    assert '<h2 class="voice-lang">English · 1 of 26 kit texts · 0:06 recorded</h2>' in recordings
    assert f">#{take.id}<" in _kit_row(recordings, "en-rain")
    assert recordings.count('<li class="kit-row"') == 1
    assert "kit-record" not in recordings and "Russian" not in recordings


def test_the_recorder_shows_the_style_the_text_and_that_any_speech_is_heard(client):
    voice = create_voice(ctx(client).conn, OWNER, "Grandma")
    body = leaf(client, voice.id)
    recorder = body[body.index('id="voice-recorder"'):body.index('<div id="voice-preview"')]
    for language, kit in sv.RECORDING_KIT.items():
        for text in kit:
            label, instruction = sv.KIT_STYLES[text.style]
            assert (f'<p class="recorder-instruction" data-lang="{language}" '
                    f'data-prompt-id="{text.id}" hidden>{label} — {instruction}</p>') in recorder
    assert ('<p class="recorder-hint muted" hidden>Read it, or say anything else: the app hears '
            "what you say.</p>") in recorder
    take = recorder[recorder.index('<div class="recorder-take"'):]
    assert "<textarea" not in take and "<input" not in take
    assert take.index('class="recorder-save">Save</button>') < take.index(
        'class="recorder-again">Record again</button>') < take.index('class="recorder-discard">Discard')


def _ready_lines(body: str) -> list[str]:
    return [_text(line) for line in re.findall(r'<p class="voice-ready[^"]*">.*?</p>', body, re.DOTALL)]


def test_the_header_names_the_take_films_speak_from_per_language(client):
    voice, older = voice_with_example(client)
    newer = _add(client, voice.id, freq=330.0)
    _hear(client, voice.id, older, "First words.")
    _hear(client, voice.id, newer, "Second words.")
    assert _ready_lines(body_of(client, voice.id)) == [
        f"English · films and tests speak from take #{newer.id}",
        "Russian · no take yet",
    ]


def test_a_newer_take_with_no_usable_words_yet_is_named_beside_the_take_films_use(client):
    voice, heard = voice_with_example(client)
    _hear(client, voice.id, heard, "Words.")
    silent = _add(client, voice.id, freq=330.0)
    _hear(client, voice.id, silent, "♪")
    assert _ready_lines(body_of(client, voice.id))[0] == (
        f"English · films and tests speak from take #{heard.id} · take #{silent.id} has no usable words")
    waiting = _add(client, voice.id, freq=440.0)
    assert _ready_lines(body_of(client, voice.id))[0] == (
        f"English · films and tests speak from take #{heard.id} · take #{waiting.id} waits for its words")


def test_the_header_says_why_a_language_is_not_ready(client):
    voice, waiting = voice_with_example(client)
    silent = _add(client, voice.id, language="ru", freq=330.0)
    _hear(client, voice.id, silent, "♪")
    body = body_of(client, voice.id)
    assert _ready_lines(body)[:2] == ["English · the words of its takes are not heard yet",
                                      "Russian · no take with usable words"]
    assert 'class="voice-ready voice-unready"' in body
    assert f'<button type="button" class="voice-hear" data-voice-id="{voice.id}">Hear the words</button>' in body
    del waiting


def test_hear_the_words_shows_only_while_a_take_waits_and_no_job_hears_it(client):
    voice, example = voice_with_example(client)
    assert 'class="voice-hear"' in body_of(client, voice.id)
    _hear(client, voice.id, example, "Words.")
    assert 'class="voice-hear"' not in body_of(client, voice.id)


def test_a_device_with_voice_copy_and_no_speech_recognition_says_so(client, monkeypatch):
    monkeypatch.setattr(voices_api, "transcriber_for", lambda profile: None)
    voice, _ = voice_with_example(client)
    body = body_of(client, voice.id)
    assert "Hearing the words of a take is not offered on this device yet." in body
    assert 'class="voice-hear"' not in body


def _card(body: str, voice_id: int) -> list[str]:
    card = re.search(rf'<a class="voice-card" href="/voices/{voice_id}"[^>]*>.*?</a>', body,
                     re.DOTALL).group(0)
    return re.findall(r'<span class="voice-card-(?:line|ready)[^"]*">(.*?)</span>', card)


def test_each_card_names_its_takes_and_the_languages_it_is_ready_for(client):
    c = ctx(client)
    waiting, _ = voice_with_example(client, "Anna")
    ready, _ = voice_with_example(client, "Boris", data=clip(freq=330.0))
    _hear_all(client, ready.id)
    both, _ = voice_with_example(client, "Clara", data=clip(freq=440.0))
    _add(client, both.id, language="ru", freq=550.0)
    _hear_all(client, both.id)
    empty = create_voice(c.conn, OWNER, "Dora")
    body = client.get("/voices").text
    assert _card(body, waiting.id) == ["English · 0:06 · 1 take", "Not ready for films yet"]
    assert _card(body, ready.id) == ["English · 0:06 · 1 take", "For films: English"]
    assert _card(body, both.id) == ["English · 0:06 · 1 take", "Russian · 0:06 · 1 take",
                                    "For films: English, Russian"]
    assert _card(body, empty.id) == ["No takes yet", "Not ready for films yet"]
    assert "data-baking" not in body


def test_no_page_offers_a_bake_or_names_a_voice_model(client):
    voice, example = voice_with_example(client)
    _hear(client, voice.id, example, "Words.")
    pages = {"leaf": leaf(client, voice.id), "body": body_of(client, voice.id),
             "grid": client.get("/voices").text, "model tab": leaf(client, voice.id, tab="model")}
    for name, page in pages.items():
        for gone in ("voice-bake", "Bake", "Rebake", "voice model", "Voice model", "data-tab-part=\"model\"",
                     "voice-model-tab", "Baking"):
            assert gone not in page, (name, gone)


def test_the_test_tab_offers_the_ready_languages_and_their_texts(client):
    voice, older = voice_with_example(client)
    newer = _add(client, voice.id, freq=330.0)
    silent = _add(client, voice.id, freq=550.0)
    russian = _add(client, voice.id, language="ru", freq=660.0)
    _hear(client, voice.id, older, "First words.")
    _hear(client, voice.id, newer, "Second words.")
    _hear(client, voice.id, silent, "♪")
    _hear(client, voice.id, russian, "Русские слова.")
    waiting = _add(client, voice.id, freq=440.0)   # not heard yet
    body = body_of(client, voice.id)
    assert re.search(r'<div id="voices"[^>]*data-test-ready="1">', body)
    part = _part(body, "test")
    language = re.search(r'<select class="test-language".*?</select>', part, re.DOTALL).group(0)
    assert [(value, label) for value, label, _, _ in _options(language)] == [
        ("en", "English"), ("ru", "Russian")]
    assert f'<option value="en" data-take="{newer.id}">' in language
    assert f'<option value="ru" data-take="{russian.id}">' in language
    assert "the newest take of the language with heard words speaks" in part
    texts = re.search(r'<select class="test-text".*?</select>', part, re.DOTALL).group(0)
    found = re.findall(r'<option value="([^"]*)" data-lang="(\w+)" data-words="([^"]*)">(.*?)'
                       r"</option>", texts)
    expected = []
    for code, takes in (("en", [(newer, "Second words."), (older, "First words.")]),
                        ("ru", [(russian, "Русские слова.")])):
        expected += [(f"take:{take.id}", code, words, f"Take #{take.id} — {words}")
                     for take, words in takes]
    assert found == expected
    # Only the takes open in the list: the "nothing chosen" option is hidden from it.
    assert texts.count("<option") == len(expected) + 1
    assert ('<option value="custom" hidden disabled selected>'
            "Choose a recording to compare with</option>") in texts
    assert f"Take #{waiting.id}" not in texts and f"Take #{silent.id}" not in texts


def test_a_long_take_is_cut_in_its_label_and_whole_in_its_words(client):
    voice, example = voice_with_example(client)
    words = "word " * 40 + "end"
    _hear(client, voice.id, example, words)
    _hear_all(client, voice.id)
    texts = re.search(r'<select class="test-text".*?</select>', _part(body_of(client, voice.id), "test"),
                      re.DOTALL).group(0)
    assert f'data-words="{words}"' in texts
    assert f">Take #{example.id} — {words[:80].rstrip()}…</option>" in texts


def test_the_test_tab_offers_only_the_ready_languages(client):
    voice, _ = voice_with_example(client, language="ru")
    _hear_all(client, voice.id)
    _add(client, voice.id, language="en", freq=330.0)   # not heard yet: English is not ready
    part = _part(body_of(client, voice.id), "test")
    language = re.search(r'<select class="test-language".*?</select>', part, re.DOTALL).group(0)
    assert [value for value, _, _, _ in _options(language)] == ["ru"]


def test_with_no_ready_language_the_test_tab_asks_for_a_take(client):
    voice, _ = voice_with_example(client)
    body = body_of(client, voice.id)
    part = _part(body, "test")
    assert ("Record a take first. Tests speak from the newest take of the language, once its "
            "words are heard.") in part
    assert "test-language" not in part and "test-text" not in part
    assert "data-test-ready" not in body


def _compare(body: str) -> str:
    return re.search(r'<template class="test-compare">.*?</template>', body, re.DOTALL).group(0)


def _test_line(client, voice_id) -> str:
    return re.search(r'<span class="example-test muted">(.*?)</span>',
                     _compare(body_of(client, voice_id))).group(1)


def test_the_comparison_shows_the_stored_test_with_its_take(client):
    voice, first = voice_with_example(client)
    _hear_all(client, voice.id)
    test = _store_test(client, voice.id)
    compare = _compare(body_of(client, voice.id))
    assert _test_line(client, voice.id) == (
        f"English · “Hello there.” · from take #{first.id} · voxcpm2 @ {REVISION[:7]} · made "
        f"in 12.3 s · {test.created_at[:10]}")
    assert f'src="/voices/{voice.id}/preview/audio?v={test.storage_key[:12]}"' in compare
    assert "Original" not in compare
    newer = _add(client, voice.id, freq=330.0)
    _hear_all(client, voice.id)
    assert _test_line(client, voice.id).endswith(
        f"· films of «Grandma» now speak from take #{newer.id} — generate it again")
    _drain_delete(client, voice.id, first.id)
    assert "· from a deleted take · voxcpm2" in _test_line(client, voice.id)


def test_a_test_made_through_a_voice_model_keeps_its_version(client):
    voice, _ = voice_with_example(client)
    _hear_all(client, voice.id)
    test = _store_test(client, voice.id)
    ctx(client).conn.execute("UPDATE voice_tests SET model_version = 2, prompt_example_id = NULL")
    assert _test_line(client, voice.id).startswith("English · “Hello there.” · voice model v2 · ")
    del test


def test_a_test_of_a_takes_words_plays_that_take_as_the_original(client):
    voice, example = voice_with_example(client)
    _hear(client, voice.id, example, "My own words.")
    _hear_all(client, voice.id)
    _store_test(client, voice.id, text_kind="take", example_id=example.id, text="My own words.")
    compare = _compare(body_of(client, voice.id))
    original = compare[compare.index('<span class="example-label">Original</span>'):]
    assert f'<span class="compare-what">take #{example.id}: “My own words.”</span>' in original
    assert f'src="/voices/{voice.id}/examples/{example.id}/audio"' in original
    _drain_delete(client, voice.id, example.id)
    compare = _compare(body_of(client, voice.id))
    assert '<span class="compare-what">the take was deleted</span>' in compare
    assert f"/examples/{example.id}/audio" not in compare


def test_the_test_audio_is_served(client):
    voice, _ = voice_with_example(client)
    _hear_all(client, voice.id)
    audio = clip(2.0, freq=550.0)
    _store_test(client, voice.id, audio=audio)
    response = client.get(f"/voices/{voice.id}/preview/audio")
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.content == audio


def test_the_test_audio_is_404_without_a_test_or_its_file(client):
    voice, _ = voice_with_example(client)
    url = f"/voices/{voice.id}/preview/audio"
    response = client.get(url)
    assert response.status_code == 404
    assert response.json()["detail"] == "This voice has no test yet."
    assert client.get("/voices/99/preview/audio").status_code == 404
    _hear_all(client, voice.id)
    test = _store_test(client, voice.id)
    ctx(client).voices.delete(test.storage_key)
    assert client.get(url).status_code == 404


def test_start_recording_pauses_every_audio_before_the_microphone():
    import pathlib

    source = pathlib.Path("web/static/voices.js").read_text()
    start = _js_function(source, "async function startRecording")
    pause = start.index("document.querySelectorAll('audio').forEach((audio) => audio.pause());")
    assert pause < start.index("await navigator.mediaDevices.getUserMedia")


VOXCPM = cloning_entry_for("mac", "en")
EN_2 = PREVIEW_SENTENCES["en"]["EN-2"]
EN_3 = PREVIEW_SENTENCES["en"]["EN-3"]


def _audio_for(text: str) -> bytes:
    return clip(2.0, freq=200.0 + len(text))


class VoiceCopyModels(HeardModels):
    """The models service's `/speak` with a take reference (§5.1): it answers as the
    voice-copy model and reports, per reference, the conditioning it built from the take
    (D86). A text in `gates` waits for its Event."""

    def __init__(self):
        super().__init__()
        self.calls: list[str] = []
        self.references: list[dict] = []
        self.gates: dict[str, threading.Event] = {}
        self.blocked = threading.Event()

    def hold(self, text: str) -> threading.Event:
        self.gates[text] = threading.Event()
        return self.gates[text]

    def speak(self, text, language="en", voice=None, segments=None, voices=None, *,
              references=None, **kw):
        self.calls.append(text)
        self.references.append(references)
        gate = self.gates.get(text)
        if gate is not None:
            self.blocked.set()
            assert gate.wait(10)
        return {
            "audio": _audio_for(text), "sample_rate": 48000, "model": VOXCPM.key,
            "revision": VOXCPM.source.revision,
            "conditioning": {token: {"state": "built", "model": VOXCPM.key,
                                     "revision": VOXCPM.source.revision, "reason": None, "blob": None}
                             for token in references},
            "adapters": {token: {"state": "none", "sha256": None} for token in references},
        }


@pytest.fixture
def models(monkeypatch):
    fake = VoiceCopyModels()
    monkeypatch.setattr(Settings, "build_models_client", lambda self: fake)
    yield fake
    for gate in fake.gates.values():   # never leave a test thread waiting
        gate.set()


def ready_voice(client, name="Grandma", language="en"):
    """A saved voice with one take whose words were heard: ready for its language."""
    voice, example = voice_with_example(client, name, language)
    _hear_all(client, voice.id)
    return voice, example


def _test(client, voice_id, kind="sentence", text_id="EN-2", text="", language="en"):
    return client.post(f"/voices/{voice_id}/preview",
                       data={"language": language, "text_kind": kind, "text_id": text_id,
                             "text": text})


def _state(client) -> str:
    return client.get("/voices/preview").text


def _wait_for(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.05)


def _stored_tests(client) -> list[tuple[int, str]]:
    return [tuple(row) for row in ctx(client).conn.execute(
        "SELECT voice_id, text FROM voice_tests ORDER BY voice_id")]


def test_the_test_panel_sits_outside_the_list_and_the_texts_inside_it(client):
    voice, _ = ready_voice(client)
    body = leaf(client, voice.id)
    end = _voices_end(body)
    assert "voice-preview" not in body[body.index('<div id="voices"'):end]
    assert body.index('id="voice-recorder"') < body.index('<div id="voice-preview"')
    panel = body[body.index('<div id="voice-preview"'):]
    assert '<p class="preview-count muted" hidden></p>' in panel
    assert '<textarea class="preview-custom" maxlength="300"' in panel
    assert f'placeholder="Type what «{voice.name}» should say"' in panel
    assert '<button type="button" class="preview-generate">Generate</button>' in panel
    assert '<span class="preview-through muted"></span>' in panel
    assert '<div id="voice-preview-job" data-state="idle"' in panel
    assert "test-text" not in panel and "test-language" not in panel
    listing = body[body.index('<div id="voices"'):end]
    for text_id, sentence in PREVIEW_SENTENCES["en"].items():
        assert sentence not in listing and f"sentence:{text_id}" not in listing


def test_a_test_is_stored_and_shown_as_generated(client, models):
    voice, take = ready_voice(client)
    response = _test(client, voice.id)
    assert response.status_code == 200
    assert response.text.lstrip().startswith('<div id="voice-preview-job"')
    client.app.state.await_voice_preview()
    state = _state(client)
    assert 'data-state="done"' in state
    assert "Done: the new test is under ▶ Generated." not in state
    line = re.search(r'<span class="example-test muted">(.*?)</span>',
                     _compare(body_of(client, voice.id))).group(1)
    assert line.startswith(f"English · “{EN_2}” · from take #{take.id} · {VOXCPM.key} @ "
                           f"{VOXCPM.source.revision[:7]} · made in ")
    assert "generate it again" not in line
    audio = client.get(f"/voices/{voice.id}/preview/audio")
    assert audio.status_code == 200 and audio.headers["content-type"] == "audio/wav"
    assert audio.content == _audio_for(EN_2)
    reference = models.references[0][sv.token(voice.id)]
    assert reference["sha256"] == take.content_hash and "adapter" not in reference
    assert reference["transcript"] == WORDS
    _test(client, voice.id, text_id="EN-3")
    client.app.state.await_voice_preview()
    assert _stored_tests(client) == [(voice.id, EN_3)]
    assert models.calls == [EN_2, EN_3]


def test_a_test_of_own_text_speaks_it(client, models):
    voice, _ = ready_voice(client)
    assert _test(client, voice.id, "custom", "", "  My own words.  ").status_code == 200
    client.app.state.await_voice_preview()
    assert _stored_tests(client) == [(voice.id, "My own words.")]


def test_a_test_carries_the_running_job_while_it_runs(client, models):
    voice, _ = ready_voice(client)
    models.hold(EN_2)
    body = _test(client, voice.id).text
    assert f"Testing «Grandma» (English): “{EN_2}”" in body
    assert '<p class="preview-other" hidden>Testing «Grandma» · ' in body
    assert 'class="preview-cancel">Cancel</button>' in body
    assert models.blocked.wait(10)
    page = leaf(client, voice.id)
    job = re.search(r'<div id="voice-preview-job"[^>]*>', page).group(0)
    assert 'data-state="running"' in job and 'data-language="en"' in job
    assert "data-example-id" not in job


@pytest.mark.parametrize(("kind", "text_id", "text", "detail"), [
    ("bogus", "", "", "Unknown kind of test text."),
    ("reading", "", "", "Unknown kind of test text."),
    ("sentence", "RU-2", "", "Unknown test sentence for English: 'RU-2'."),
    ("custom", "", "   ", "Type a text to test."),
    ("custom", "", "x" * 301, "A test text can have at most 300 characters."),
])
def test_a_refused_test_text_is_400(client, models, kind, text_id, text, detail):
    voice, _ = ready_voice(client)
    response = _test(client, voice.id, kind, text_id, text)
    assert response.status_code == 400
    assert response.json()["detail"] == detail
    assert client.app.state.voice_previews.thread is None


def test_a_test_of_the_heard_words_of_a_take_plays_it_as_the_original(client, models):
    voice, example = voice_with_example(client)
    _hear(client, voice.id, example, "My own words.")
    _hear_all(client, voice.id)
    response = _test(client, voice.id, "take", str(example.id))
    assert response.status_code == 200, response.text
    client.app.state.await_voice_preview()
    assert models.calls == ["My own words."]
    compare = _compare(body_of(client, voice.id))
    assert f'<span class="compare-what">take #{example.id}: “My own words.”</span>' in compare


def test_a_take_that_has_no_words_to_test_is_400(client, models):
    voice, _ = voice_with_example(client)
    silent = _add(client, voice.id)
    _hear(client, voice.id, silent, "♪")
    _hear_all(client, voice.id)
    waiting = _add(client, voice.id, freq=440.0)   # not heard yet
    cases = {
        str(waiting.id): f"The words of take #{waiting.id} are not heard yet.",
        str(silent.id): (f"Take #{silent.id} has no words to speak: the speech recognition "
                         "found no words in it."),
        "999": "Unknown take.",
    }
    for text_id, detail in cases.items():
        response = _test(client, voice.id, "take", text_id)
        assert (response.status_code, response.json()["detail"]) == (400, detail)
    assert models.calls == []


def test_a_test_of_an_unknown_or_foreign_voice_is_404(client, models):
    ready_voice(client)
    foreign = create_voice(ctx(client).conn, 2, "Theirs")
    for voice_id in (99, foreign.id):
        response = _test(client, voice_id)
        assert (response.status_code, response.json()["detail"]) == (404, "Unknown saved voice.")
    assert models.calls == []


def test_a_test_of_a_pending_voice_is_409(queued, models):
    voice, _ = ready_voice(queued)
    queued.post(f"/voices/{voice.id}/delete")
    response = _test(queued, voice.id)
    assert (response.status_code, response.json()["detail"]) == (409, "This voice is being deleted.")


@pytest.mark.parametrize("language", ["de", ""])
def test_a_test_in_a_language_voice_copy_does_not_speak_is_409(client, models, language):
    voice, _ = ready_voice(client)
    response = _test(client, voice.id, language=language)
    assert (response.status_code, response.json()["detail"]) == (409, NOT_AVAILABLE)


def test_a_test_on_jetson_is_409(jetson, models):
    voice, _ = voice_with_example(jetson)
    response = _test(jetson, voice.id)
    assert (response.status_code, response.json()["detail"]) == (409, NOT_AVAILABLE)


def test_a_voice_that_cannot_be_tested_is_refused_at_once(client, models):
    waiting, _ = voice_with_example(client, "Anna")
    silent, take = voice_with_example(client, "Boris", data=clip(freq=330.0))
    _hear(client, silent.id, take, "♪")
    english, _ = ready_voice(client, "Clara")
    cases = [
        (waiting.id, "en", "The words of the English takes are not heard yet."),
        (silent.id, "en", "No English take has usable words."),
        (english.id, "ru", "Record a Russian take first."),
    ]
    for voice_id, language, detail in cases:
        response = _test(client, voice_id, text_id="RU-1" if language == "ru" else "EN-2",
                         language=language)
        assert (response.status_code, response.json()["detail"]) == (409, detail)
    assert models.calls == [] and client.app.state.voice_previews.thread is None


def test_a_test_on_low_disk_is_507(client, models, monkeypatch):
    voice, _ = ready_voice(client)
    monkeypatch.setattr(ctx(client).voices, "free_bytes", lambda: 0)
    assert _test(client, voice.id).status_code == 507
    assert models.calls == []


def test_cancel_does_not_wait_for_the_model_and_stores_nothing(client, models):
    voice, _ = ready_voice(client)
    gate = models.hold(EN_2)
    _test(client, voice.id)
    assert models.blocked.wait(10)
    response = client.post("/voices/preview/cancel")
    assert response.status_code == 200
    assert 'data-state="idle"' in response.text and not gate.is_set()
    assert client.app.state.voice_calls.locked()
    gate.set()
    client.app.state.await_voice_preview()
    assert rows(client, "voice_tests") == 0
    assert rows(client, "voice_conditionings") == 0
    assert not client.app.state.voice_calls.locked()


def test_a_test_waits_for_a_cancelled_call_and_then_runs(client, models):
    voice, _ = ready_voice(client)
    gate = models.hold(EN_2)
    _test(client, voice.id)
    assert models.blocked.wait(10)
    client.post("/voices/preview/cancel")
    _test(client, voice.id, text_id="EN-3")
    _wait_for(lambda: "waiting for the voice model" in _state(client))
    assert models.calls == [EN_2]
    gate.set()
    client.app.state.await_voice_preview()
    assert models.calls == [EN_2, EN_3]
    assert _stored_tests(client) == [(voice.id, EN_3)]


def test_a_newer_test_replaces_a_running_one(client, models):
    first, _ = ready_voice(client, "Anna")
    second, _ = ready_voice(client, "Boris", language="ru")
    gate = models.hold(EN_2)
    _test(client, first.id)
    assert models.blocked.wait(10)
    _test(client, second.id, text_id="RU-1", language="ru")
    gate.set()
    client.app.state.await_voice_preview()
    assert _stored_tests(client) == [(second.id, PREVIEW_SENTENCES["ru"]["RU-1"])]


def test_a_replaced_test_of_the_same_voice_that_returns_last_is_dropped(client, models,
                                                                        monkeypatch):
    # Without the voice lock, so the older call can return after the newer one.
    monkeypatch.setattr(client.app.state.voice_previews, "_voice_lock", None)
    voice, _ = ready_voice(client)
    en_1 = PREVIEW_SENTENCES["en"]["EN-1"]
    gate = models.hold(en_1)
    _test(client, voice.id, text_id="EN-1")
    assert models.blocked.wait(10)
    _test(client, voice.id, text_id="EN-2")
    _wait_for(lambda: 'data-state="done"' in _state(client))
    gate.set()
    client.app.state.await_voice_preview()
    assert models.calls == [en_1, EN_2]
    assert _stored_tests(client) == [(voice.id, EN_2)]
    state = _state(client)
    assert 'data-state="done"' in state and 'data-run="2"' in state


class GatedStorage:
    """The voices store, whose write of `watched` waits until `release` is set."""

    def __init__(self, real, watched: bytes):
        self.real = real
        self.watched = watched
        self.entered = threading.Event()
        self.release = threading.Event()

    def write(self, key, data):
        if data == self.watched:
            self.entered.set()
            assert self.release.wait(10)
        self.real.write(key, data)

    def __getattr__(self, name):
        return getattr(self.real, name)


def test_a_cancel_during_the_publication_waits_for_it_and_the_test_stays(client, models,
                                                                        monkeypatch):
    voice, _ = ready_voice(client)
    gated = GatedStorage(ctx(client).voices, _audio_for(EN_2))
    monkeypatch.setattr(ctx(client), "voices", gated)
    _test(client, voice.id)
    assert gated.entered.wait(10)
    replies = {}
    canceller = threading.Thread(
        target=lambda: replies.update(cancel=client.post("/voices/preview/cancel")))
    canceller.start()
    canceller.join(0.3)
    assert canceller.is_alive()            # the publication holds the job lock
    gated.release.set()
    canceller.join(10)
    assert 'data-state="idle"' in replies["cancel"].text
    assert 'data-completed="1"' in replies["cancel"].text
    client.app.state.await_voice_preview()
    assert _stored_tests(client) == [(voice.id, EN_2)]


def test_a_test_never_changes_casting_or_the_takes(client, models):
    voice, _ = ready_voice(client)
    c = ctx(client)
    tables = ("memory_narrators", "voice_examples", "voice_models", "voice_film_examples")
    before = [c.conn.execute(f"SELECT * FROM {table}").fetchall() for table in tables]
    _test(client, voice.id)
    client.app.state.await_voice_preview()
    after = [c.conn.execute(f"SELECT * FROM {table}").fetchall() for table in tables]
    assert [list(map(tuple, part)) for part in after] == [list(map(tuple, part)) for part in before]
    assert rows(client, "voice_tests") == 1


def test_every_fragment_carries_the_epoch_and_the_last_stored_run(client, models):
    voice, _ = ready_voice(client)
    previews_job = client.app.state.voice_previews
    assert re.fullmatch(r"[0-9a-f]{32}", previews_job.epoch)
    assert previews_job.snapshot()["completed"] == 0
    assert "example_id" not in previews_job.snapshot()
    response = client.get("/voices/preview")
    assert response.headers["cache-control"] == "no-store"
    assert f'data-epoch="{previews_job.epoch}" data-completed="0"' in response.text
    _test(client, voice.id)
    client.app.state.await_voice_preview()
    assert previews_job.completed == 1
    assert previews_job.snapshot()["voice_id"] == voice.id
    assert previews_job.snapshot()["language"] == "en"
    assert 'data-completed="1"' in _state(client)
    assert 'data-completed="1"' in leaf(client, voice.id)
    assert 'data-completed="1"' in client.post("/voices/preview/cancel").text


def test_two_apps_have_two_epochs(settings):
    epochs = []
    for _ in range(2):
        with _client(settings) as test_client:
            epochs.append(test_client.app.state.voice_previews.epoch)
    assert epochs[0] != epochs[1]


def _list_certificate(body: str) -> tuple[str, int]:
    found = re.search(r'<div id="voices"[^>]*?data-epoch="([0-9a-f]+)"\s+data-completed="(\d+)"',
                      body)
    assert found, "no certificate on #voices"
    return found.group(1), int(found.group(2))


def test_every_list_body_carries_the_test_certificate(client, models):
    voice, example = ready_voice(client)
    _test(client, voice.id)
    client.app.state.await_voice_preview()
    bodies = {
        "page": leaf(client, voice.id),
        "list": body_of(client, voice.id),
        "rename": client.post(f"/voices/{voice.id}/rename", data={"name": "Gran"}).text,
        "add example": post_example(client, voice.id, clip(freq=330.0)).text,
        "delete example": client.post(
            f"/voices/{voice.id}/examples/{example.id}/delete").text,
        "delete voice": client.post(f"/voices/{voice.id}/delete").text,
    }
    epoch = client.app.state.voice_previews.epoch
    assert {name: _list_certificate(body) for name, body in bodies.items()} == {
        name: (epoch, 1) for name in bodies}


def _store_a_test_on(monkeypatch, job, *, before_the_read: bool):
    """`get_voice`, with a test stored by another thread just before or after its first
    read."""
    real = voices_api.get_voice
    reads = []

    def stored() -> None:
        with job.lock:
            job.completed += 1

    def reading(conn, owner_id, voice_id):
        first = not reads
        reads.append(voice_id)
        if first and before_the_read:
            stored()
        voice = real(conn, owner_id, voice_id)
        if first and not before_the_read:
            stored()
        return voice

    monkeypatch.setattr(voices_api, "get_voice", reading)


def test_the_certificate_is_read_before_the_list(client, monkeypatch):
    voice, _ = voice_with_example(client)
    job = client.app.state.voice_previews
    _store_a_test_on(monkeypatch, job, before_the_read=True)
    assert _list_certificate(body_of(client, voice.id)) == (job.epoch, 0)


def test_a_test_stored_after_the_list_read_leaves_the_page_on_one_snapshot(client, monkeypatch):
    voice, _ = voice_with_example(client)
    job = client.app.state.voice_previews
    _store_a_test_on(monkeypatch, job, before_the_read=False)
    body = leaf(client, voice.id)
    assert job.completed == 1
    assert _list_certificate(body) == (job.epoch, 0)
    assert re.search(r'<div id="voice-preview-job"[^>]*?data-completed="0"', body)


def _script() -> str:
    return pathlib.Path("web/static/voices.js").read_text()


def test_the_script_polls_the_state_and_posts_the_test_text():
    source = _script()
    assert "const STATUS_URL = '/voices/preview';" in source
    assert "fetch(STATUS_URL, { cache: 'no-store' })" in _js_function(source, "function pollPreview")
    generate = _js_function(source, "function generatePreview")
    assert "mutate(`/voices/${target.voiceId}/preview`, body)" in generate
    for field in ("'language'", "'text_kind'", "'text_id'", "'text'"):
        assert f"body.append({field}," in generate
    assert ("'The app restarted, so this test stopped. Generate it again.'"
            in _js_function(source, "function showRestarted"))


PURE = ("function previewVerdict", "function completionKey", "async function refreshLoop",
        "function previewSync")

# The page's own list code, run inside each harness against a fake DOM and fetch.
DOM = ("function certOf", "function replaceList", "function pollWhileDeleting",
       "async function detailOf", "async function send", "async function refreshBody")

HARNESS = r"""
const flush = () => new Promise((resolve) => setImmediate(resolve));
const frag = (epoch, run, state, completed = 0) => ({ epoch, run, state, completed });
const POLL_MS = 2000;
// A #voices body: "NAME@epoch/completed", plus "+deleting" while a row is deleted; a body
// with no "@" has no certificate.
function element(text) {
  const [head, flag] = text.split('+');
  const [name, cert] = head.split('@');
  const dataset = {};
  if (cert) { [dataset.epoch, dataset.completed] = cert.split('/'); }
  return { name, dataset, deleting: flag === 'deleting' };
}
function harness(initial, page = `PAGE@${initial.epoch}/${initial.completed}`) {
  const h = { typing: false, replaced: [], marked: [], applied: [], restarted: 0, errors: [],
              timers: [], polls: [], list: [], status: [], posts: [], texts: [], sent: 0 };
  h.current = element(page);
  const onGrid = () => false;
  const leafUrl = () => '/voices/1';
  const carryInputs = () => {};
  const placeRecorder = () => {};
  const showError = (text) => h.errors.push(text);
  const clearError = () => {};
  const typingInList = () => h.typing;
  const decorate = () => {};
  const markHearing = () => {};
  const hearingBusyHere = () => false;
  const pollHearing = () => {};
  const listOf = (html) => element(html);
  const document = {
    getElementById: (id) => (id !== 'voices' ? null : {
      contains: () => false,
      replaceWith: (next) => { h.current = next; h.replaced.push(next.name); },
    }),
    querySelector: () => (h.current.deleting ? {} : null),
  };
  const held = (queue) => new Promise((resolve, reject) => queue.push({ resolve, reject }));
  const fetch = (url, options = {}) => {
    if (options.method !== 'POST') { return held(h.list); }
    h.sent += 1;
    return held(h.posts);
  };
  const mounted = {
    alive: () => true,
    fetch: (url, options) => fetch(url, options),
    schedule: (run) => { h.polls.push(run); return h.polls.length; },
  };
  let pollTimer = null;
  let bodyTimer = null;
  /* DOM */
  const sync = previewSync({
    fetchList: () => fetch('/voices/1/list', { cache: 'no-store' }),
    replaceList: (html) => replaceList(html, sync),
    typing: typingInList,
    schedule: (run) => h.timers.push(run),
    apply: (f) => h.applied.push(`${f.epoch}/${f.run}/${f.state}`),
    poll: () => {},
    stopTimers: () => {},
    fetchStatus: () => held(h.status),
    restarted: () => { h.restarted += 1; },
    refreshed: (key) => h.marked.push(`${key.epoch}/${key.completed}`),
  }, initial, certOf(h.current));
  const reply = (text, ok) => ({ ok, status: ok ? 200 : 409, text: async () => text,
                                  json: async () => ({ detail: text }) });
  h.sync = sync;
  h.send = (url) => send(url, null);
  h.pollWhileDeleting = pollWhileDeleting;
  h.refreshBody = () => refreshBody(mounted);
  h.show = (f) => sync.show(f, sync.generation());
  h.runTimers = async () => { h.timers.splice(0).forEach((run) => run()); await flush(); };
  h.runPolls = async () => { h.polls.splice(0).forEach((run) => run()); await flush(); };
  h.body = async (text, ok = true, at = 0) => { h.list.splice(at, 1)[0].resolve(reply(text, ok)); await flush(); };
  h.down = async () => { h.list.shift().reject(new Error('down')); await flush(); };
  h.reply = async (f) => { h.status.shift().resolve(f); await flush(); };
  h.statusDown = async () => { h.status.shift().reject(new Error('down')); await flush(); };
  h.post = async (text, ok = true) => { h.posts.shift().resolve(reply(text, ok)); await flush(); };
  // A POST reply whose body still decodes; `h.decode(text)` ends it.
  h.postHeld = async () => {
    const response = { ok: true, status: 200, text: () => held(h.texts) };
    h.posts.shift().resolve(response); await flush();
  };
  h.decode = async (text) => { h.texts.shift().resolve(text); await flush(); };
  h.out = () => ({ replaced: h.replaced, marked: h.marked, applied: h.applied,
                   restarted: h.restarted, fetches: h.list.length, statuses: h.status.length,
                   sent: h.sent, current: h.current.name });
  return h;
}
"""


def _node(scenario: str) -> dict:
    source = _script()
    functions = "\n".join(f"{_js_function(source, head)}\n}}" for head in PURE)
    dom = "\n".join(f"{_js_function(source, head)}\n}}" for head in DOM)
    harness = HARNESS.replace("/* DOM */", dom)
    script = f"{functions}\n{harness}\n(async () => {{\n{scenario}\n}})();"
    done = subprocess.run(["node", "-e", script], capture_output=True, text=True, timeout=30,
                          check=False)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


@needs_node
def test_verdicts_compare_runs_only_within_one_epoch():
    out = _node("""
      const seen = { epoch: 'e1', run: 2, state: 'done' };
      const retired = new Set(['e0']);
      console.log(JSON.stringify({
        retired: previewVerdict(seen, retired, frag('e0', 9, 'done')),
        unknown: previewVerdict(seen, retired, frag('e2', 1, 'idle')),
        lower: previewVerdict(seen, retired, frag('e1', 1, 'done')),
        runningAfterDone: previewVerdict(seen, retired, frag('e1', 2, 'running')),
        runningAfterFailed: previewVerdict({ ...seen, state: 'failed' }, retired, frag('e1', 2, 'running')),
        newer: previewVerdict(seen, retired, frag('e1', 3, 'running')),
        none: completionKey(frag('e1', 1, 'idle', 0)),
        key: completionKey(frag('e1', 1, 'done', 1)),
      }));
    """)
    assert out == {"retired": "ignore", "unknown": "reconcile", "lower": "ignore",
                   "runningAfterDone": "ignore", "runningAfterFailed": "ignore",
                   "newer": "apply", "none": None, "key": {"epoch": "e1", "completed": 1}}


@needs_node
def test_a_completion_in_each_epoch_refreshes_once():
    out = _node("""
      const h = harness(frag('e1', 0, 'idle'));
      h.show(frag('e1', 1, 'done', 1)); await flush();
      await h.body('L1@e1/1');
      h.show(frag('e2', 1, 'done', 1)); await flush();
      await h.reply(frag('e2', 1, 'done', 1));
      while (h.list.length) { await h.body('L2@e2/1'); }
      console.log(JSON.stringify(h.out()));
    """)
    assert out["replaced"] == ["L1", "L2"]
    assert out["marked"] == ["e1/1", "e2/1"]


@needs_node
def test_an_old_epoch_is_ignored_and_an_unseen_one_is_never_applied():
    out = _node("""
      const h = harness(frag('e1', 1, 'running'));
      const first = h.show(frag('e2', 0, 'idle'));
      await h.reply(frag('e2', 0, 'idle'));
      const old = h.show(frag('e1', 2, 'done', 2));
      const third = h.show(frag('e3', 0, 'idle'));
      console.log(JSON.stringify({ first, old, third, ...h.out() }));
    """)
    assert (out["first"], out["old"], out["third"]) == ("reconcile", "ignore", "reconcile")
    assert out["applied"] == ["e2/0/idle"]
    assert out["restarted"] == 1          # the panel showed running; the new process is idle


@needs_node
def test_a_stale_refresh_never_replaces_the_list():
    out = _node("""
      const result = {};
      // A typing retry after a Cancel.
      let h = harness(frag('e1', 0, 'idle'));
      h.typing = true;
      h.show(frag('e1', 1, 'done', 1)); await flush();
      h.sync.bump(); h.typing = false; await h.runTimers();
      result.cancel = h.out();
      // A delayed list body after a new run.
      h = harness(frag('e1', 0, 'idle'));
      h.show(frag('e1', 1, 'done', 1)); await flush();
      h.sync.bump();
      await h.body('late@e1/1');
      result.generate = h.out();
      // A delayed list body after a restart.
      h = harness(frag('e1', 0, 'idle'));
      h.show(frag('e1', 1, 'done', 1)); await flush();
      h.show(frag('e2', 0, 'idle')); await flush();
      await h.body('late@e1/1');
      result.restart = h.out();
      console.log(JSON.stringify(result));
    """)
    for case in ("cancel", "generate", "restart"):
        assert out[case]["replaced"] == [] and out[case]["marked"] == [], case


@needs_node
def test_a_failed_refresh_retries_only_while_it_is_current():
    out = _node("""
      const h = harness(frag('e1', 0, 'idle'));
      h.show(frag('e1', 1, 'done', 1)); await flush();
      await h.down();
      const retries = h.timers.length;
      await h.runTimers();
      const refetched = h.list.length;
      await h.body('busy', false);
      h.sync.bump();
      await h.runTimers();
      console.log(JSON.stringify({ retries, refetched, ...h.out() }));
    """)
    assert out["retries"] == 1 and out["refetched"] == 1
    assert out["fetches"] == 0 and out["replaced"] == []


@needs_node
def test_a_cancel_after_the_publication_refreshes_once():
    out = _node("""
      const h = harness(frag('e1', 1, 'running'));
      const at = h.sync.bump();
      h.sync.show(frag('e1', 2, 'idle', 1), at); await flush();
      await h.body('L@e1/1');
      h.show(frag('e1', 2, 'idle', 1)); await flush();
      console.log(JSON.stringify(h.out()));
    """)
    assert out["replaced"] == ["L"] and out["marked"] == ["e1/1"] and out["fetches"] == 0


@needs_node
def test_a_refused_generate_rearms_a_deferred_completion():
    out = _node("""
      const h = harness(frag('e1', 0, 'idle'));
      h.typing = true;
      h.show(frag('e1', 1, 'done', 1)); await flush();
      const at = h.sync.bump();               // Generate, answered 400
      h.sync.show(frag('e1', 1, 'done', 1), at); await flush();
      h.typing = false;                       // the blur
      await h.runTimers();
      while (h.list.length) { await h.body('L@e1/1'); }
      await h.runTimers();
      console.log(JSON.stringify(h.out()));
    """)
    assert out["replaced"] == ["L"] and out["marked"] == ["e1/1"]


@needs_node
def test_only_the_restart_sync_refresh_replaces_after_an_epoch_change():
    out = _node("""
      const h = harness(frag('e1', 0, 'idle'));
      h.show(frag('e1', 1, 'done', 1)); await flush();
      h.show(frag('e2', 0, 'idle', 0)); await flush();
      await h.reply(frag('e2', 0, 'idle', 0));
      await h.body('OLD@e1/1');
      await h.body('SYNC@e2/0');
      h.show(frag('e2', 1, 'done', 1)); await flush();
      await h.body('NEW@e2/1');
      console.log(JSON.stringify(h.out()));
    """)
    assert out["replaced"] == ["SYNC", "NEW"]
    assert out["marked"] == ["e2/0", "e2/1"]


@needs_node
def test_an_old_rename_reply_never_replaces_the_list_that_holds_a_new_test():
    out = _node("""
      const h = harness(frag('e1', 1, 'running'));
      const sent = h.send('/voices/1/rename'); await flush();
      h.show(frag('e1', 1, 'done', 1)); await flush();
      await h.body('NEW@e1/1');
      await h.post('OLD@e1/0');
      const ok = await sent;
      const afterOld = [...h.replaced];
      await h.reply(frag('e1', 1, 'done', 1));   // the repair: the state, then the list
      await h.body('RENAMED@e1/1');
      console.log(JSON.stringify({ ok, afterOld, ...h.out() }));
    """)
    assert out["ok"] is True and out["afterOld"] == ["NEW"]
    assert out["replaced"] == ["NEW", "RENAMED"] and out["marked"] == ["e1/1"]
    assert (out["sent"], out["fetches"], out["statuses"]) == (1, 0, 0)


@needs_node
def test_an_old_deletion_poll_never_replaces_the_list_that_holds_a_new_test():
    out = _node("""
      const h = harness(frag('e1', 1, 'running'), 'PAGE@e1/0+deleting');
      h.pollWhileDeleting();
      await h.runPolls();                          // the 2 s tick: its GET is held
      h.show(frag('e1', 1, 'done', 1)); await flush();
      await h.body('NEW@e1/1+deleting', true, 1);  // the refresh answers first
      await h.body('OLD@e1/0+deleting');           // then the poll, read before the test
      console.log(JSON.stringify({ polls: h.polls.length, ...h.out() }));
    """)
    assert out["replaced"] == ["NEW"] and out["current"] == "NEW"
    assert out["polls"] == 1                       # the accepted NEW keeps polling


@needs_node
def test_a_completion_while_a_reply_decodes_keeps_the_new_list():
    out = _node("""
      const h = harness(frag('e1', 1, 'running'));
      const sent = h.send('/voices/1/rename'); await flush();
      await h.postHeld();
      h.show(frag('e1', 1, 'done', 1)); await flush();
      await h.body('NEW@e1/1');
      await h.decode('OLD@e1/0');
      const ok = await sent;
      console.log(JSON.stringify({ ok, ...h.out() }));
    """)
    assert out["ok"] is True and out["replaced"] == ["NEW"] and out["statuses"] == 1


@needs_node
def test_a_refused_reply_still_succeeds_and_is_never_sent_again():
    # A reply read after a test this page has not seen yet: refused, then repaired by GETs.
    out = _node("""
      const h = harness(frag('e1', 1, 'running'));
      const sent = h.send('/voices/1/examples'); await flush();
      await h.post('ADDED@e1/1');
      const ok = await sent;
      const before = [...h.replaced];
      await h.reply(frag('e1', 1, 'done', 1));
      while (h.list.length) { await h.body('FRESH@e1/1'); }
      await h.runTimers();
      console.log(JSON.stringify({ ok, before, ...h.out() }));
    """)
    assert out["ok"] is True and out["before"] == []
    assert out["replaced"] == ["FRESH"] and out["marked"] == ["e1/1"]
    assert (out["sent"], out["fetches"], out["statuses"]) == (1, 0, 0)


@needs_node
def test_a_refused_delete_reply_still_reaches_a_fresh_deleting_list():
    out = _node("""
      const h = harness(frag('e1', 1, 'running'));
      const sent = h.send('/voices/1/delete'); await flush();
      h.show(frag('e1', 1, 'done', 1)); await flush();
      await h.body('NEW@e1/1');
      await h.post('DELETING@e1/0+deleting');
      const ok = await sent;
      await h.reply(frag('e1', 1, 'done', 1));
      await h.body('DELETING@e1/1+deleting');
      const polls = h.polls.length;
      await h.runPolls();
      console.log(JSON.stringify({ ok, polls, ...h.out() }));
    """)
    assert out["ok"] is True and out["replaced"] == ["NEW", "DELETING"]
    assert out["polls"] == 1 and out["fetches"] == 1   # the deleting list polls again
    assert out["sent"] == 1


@needs_node
def test_the_repair_waits_for_typing_and_retries_a_failed_read():
    out = _node("""
      const h = harness(frag('e1', 1, 'done', 1));
      h.typing = true;
      const sent = h.send('/voices/1/rename'); await flush();
      await h.post('BARE');                         // no certificate: refused
      await sent;
      const typing = { timers: h.timers.length, statuses: h.status.length };
      h.typing = false; await h.runTimers();
      await h.statusDown(); await h.runTimers();    // the status read fails
      await h.reply(frag('e1', 1, 'done', 1));
      await h.body('BUSY@e1/1', false); await h.runTimers();   // a non-ok list
      await h.reply(frag('e1', 1, 'done', 1));
      await h.down(); await h.runTimers();          // the list read fails
      await h.reply(frag('e1', 1, 'done', 1));
      await h.body('FRESH@e1/1');
      await h.runTimers();
      console.log(JSON.stringify({ typing, ...h.out() }));
    """)
    assert out["typing"] == {"timers": 1, "statuses": 0}
    assert out["replaced"] == ["FRESH"]
    assert (out["sent"], out["fetches"], out["statuses"]) == (1, 0, 0)


@needs_node
def test_a_retired_epoch_is_refused_and_an_unknown_one_reconciles_without_applying():
    out = _node("""
      const h = harness(frag('e1', 0, 'idle'));
      h.show(frag('e2', 0, 'idle')); await flush();
      await h.reply(frag('e2', 0, 'idle'));        // e2 adopted, e1 retired
      await h.body('SYNC@e2/0');
      let sent = h.send('/voices/1/rename'); await flush();
      await h.post('OLD@e1/0'); await sent;
      const retired = { replaced: [...h.replaced], statuses: h.status.length };
      await h.reply(frag('e2', 0, 'idle'));
      await h.body('FRESH@e2/0');
      sent = h.send('/voices/1/rename'); await flush();
      await h.post('NEXT@e3/0'); await sent;
      const unknown = { replaced: [...h.replaced], applied: [...h.applied], statuses: h.status.length };
      await h.reply(frag('e3', 0, 'idle'));
      await h.body('ADOPTED@e3/0');
      console.log(JSON.stringify({ retired, unknown, ...h.out() }));
    """)
    assert out["retired"] == {"replaced": ["SYNC"], "statuses": 1}
    assert out["unknown"] == {"replaced": ["SYNC", "FRESH"], "applied": ["e2/0/idle", "e2/0/idle"],
                              "statuses": 1}
    assert out["replaced"] == ["SYNC", "FRESH", "ADOPTED"]
    assert out["applied"][-1] == "e3/0/idle" and out["sent"] == 2


@needs_node
def test_a_held_repair_status_from_before_a_cancel_is_dropped():
    out = _node("""
      const h = harness(frag('e1', 1, 'done', 1));
      const sent = h.send('/voices/1/rename'); await flush();
      await h.post('BARE'); await sent;             // the repair's status read is held
      h.sync.bump();                                // Cancel: a new owner reads again
      const reads = h.status.length;
      await h.reply(frag('e1', 2, 'running', 1));   // the held pre-Cancel read
      const afterOld = { applied: [...h.applied], fetches: h.list.length };
      await h.reply(frag('e1', 3, 'idle', 1));
      await h.body('FRESH@e1/1');
      console.log(JSON.stringify({ reads, afterOld, ...h.out() }));
    """)
    assert out["reads"] == 2
    assert out["afterOld"] == {"applied": [], "fetches": 0}
    assert out["applied"] == ["e1/3/idle"] and out["replaced"] == ["FRESH"]


@needs_node
def test_an_old_repair_neither_erases_the_new_owner_nor_starts_a_second_one():
    out = _node("""
      const h = harness(frag('e1', 1, 'done', 1));
      let sent = h.send('/voices/1/rename'); await flush();
      await h.post('BARE'); await sent;
      await h.reply(frag('e1', 1, 'done', 1));     // the old repair's list read is held
      h.sync.bump();                                // the new owner reads the state
      await h.body('LATE@e1/1');                    // the old repair ends: no replace
      sent = h.send('/voices/1/rename'); await flush();
      await h.post('BARE'); await sent;             // refused again: the owner holds on
      const reads = h.status.length;
      await h.reply(frag('e1', 1, 'done', 1));
      await h.body('FRESH@e1/1');
      console.log(JSON.stringify({ reads, ...h.out() }));
    """)
    assert out["reads"] == 1
    assert out["replaced"] == ["FRESH"] and (out["fetches"], out["statuses"]) == (0, 0)


def test_the_old_address_answers_308_to_voices(client):
    response = client.get("/organize/voices", follow_redirects=False)
    assert response.status_code == 308
    assert response.headers["location"] == "/voices"
    voice, example = voice_with_example(client)
    for path in ("/organize/voices", f"/organize/voices/{voice.id}/rename",
                 f"/organize/voices/{voice.id}/delete",
                 f"/organize/voices/{voice.id}/examples/{example.id}/delete"):
        assert client.post(path, data={"name": "x"}).status_code in (404, 405), path
    assert client.get("/organize/voices/list").status_code == 404
    assert client.get("/organize/voices/preview").status_code == 404


def test_the_literal_paths_answer_their_fragments_not_422(client):
    assert client.get("/voices/list").text.lstrip().startswith('<div id="voice-cards"')
    state = client.get("/voices/preview")
    assert state.status_code == 200 and state.headers["cache-control"] == "no-store"
    assert state.text.lstrip().startswith('<div id="voice-preview-job"')
    assert client.post("/voices/preview/cancel").status_code == 200


def test_a_foreign_voice_is_404_on_every_voice_route(client):
    c = ctx(client)
    foreign = create_voice(c.conn, 2, "Theirs")
    theirs = add_example(c.conn, c.voices, 2, foreign.id, language="en", source="uploaded",
                         data=clip(freq=440.0))
    v, e = foreign.id, theirs.id
    gets = [f"/voices/{v}", f"/voices/{v}/list", f"/voices/{v}/examples/{e}/audio",
            f"/voices/{v}/preview/audio"]
    posts = [f"/voices/{v}/rename", f"/voices/{v}/delete", f"/voices/{v}/examples/{e}/delete",
             f"/voices/{v}/preview"]
    assert {path: client.get(path).status_code for path in gets} == dict.fromkeys(gets, 404)
    assert {path: client.post(path, data={"name": "x", "language": "en",
                                         "example_id": str(e)}).status_code
            for path in posts} == dict.fromkeys(posts, 404)
    assert post_example(client, v).status_code == 404
    assert rows(client, "voice_deletions") == 0


def test_the_grid_is_top_level_with_new_voice_before_plain_link_cards(client):
    c = ctx(client)
    voice = create_voice(c.conn, OWNER, "Grandma")
    for freq in (220.0, 330.0):
        add_example(c.conn, c.voices, OWNER, voice.id, language="ru", source="uploaded",
                    data=clip(freq=freq))
    add_example(c.conn, c.voices, OWNER, voice.id, language="en", source="uploaded",
                data=clip(25.0, freq=440.0))
    empty = create_voice(c.conn, OWNER, "Dad")
    body = client.get("/voices").text
    assert "voices-close" not in body and "ctx=" not in body
    assert body.index("voices-new") < body.index('id="voice-cards"')
    card = re.search(rf'<a class="voice-card" href="/voices/{voice.id}"[^>]*>.*?</a>', body,
                     re.DOTALL).group(0)
    assert "hx-" not in card
    lines = re.findall(r'<span class="voice-card-(?:line|ready)[^"]*">(.*?)</span>', card)
    assert lines == ["English · 0:25 · 1 take", "Russian · 0:12 · 2 takes", "Not ready for films yet"]
    assert "<audio" not in body
    dad = re.search(rf'<a class="voice-card" href="/voices/{empty.id}"[^>]*>.*?</a>', body,
                    re.DOTALL).group(0)
    assert "No takes yet" in dad


def test_a_voice_being_deleted_is_a_card_without_a_link(queued):
    voice, _ = voice_with_example(queued)
    queued.post(f"/voices/{voice.id}/delete")
    body = queued.get("/voices/list").text
    assert f'href="/voices/{voice.id}"' not in body
    pending = re.search(r'<div class="voice-card voice-pending"[^>]*data-deleting>.*?</div>', body,
                        re.DOTALL).group(0)
    assert "Grandma" in pending and "deleting…" in pending


def _voices_end(body: str) -> int:
    """Where the #voices <div> of a page closes."""
    start = body.index('<div id="voices"')
    depth = 0
    for tag in re.finditer(r"<(/?)div\b", body[start:]):
        depth += -1 if tag.group(1) else 1
        if depth == 0:
            return start + tag.end()
    raise AssertionError("#voices is not closed")


def test_the_leaf_header_menu_tabs_and_outside_parts(client):
    voice, _ = voice_with_example(client)
    body = leaf(client, voice.id)
    head = re.search(r'<header class="voice-head">.*?</header>', body, re.DOTALL).group(0)
    assert '<a class="voices-up" href="/voices">Voices</a>' in head
    assert '<span class="voice-name">Grandma</span>' in head
    assert 'class="voices-close"' in head
    menu = re.search(r'<details class="voice-menu">.*?</details>', head, re.DOTALL).group(0)
    assert "voice-rename" in menu and "Rename…" in menu
    assert ('data-confirm="Delete «Grandma» and its examples? Films already narrated with this '
            'voice keep their sound until you regenerate them."') in menu
    assert "Delete voice…" in menu
    tabs = re.findall(r'<button type="button" role="tab" class="voice-tab" data-tab="(\w+)"\s+'
                      r'aria-selected="(\w+)">', body)
    assert tabs == [("recordings", "true"), ("test", "false"), ("faces", "false")]
    end = _voices_end(body)
    for outside in ('id="voice-recorder"', 'id="voice-preview"'):
        assert body.index(outside) > end, outside
    assert "data-from-grid" not in body


@pytest.mark.parametrize(("tab", "shown"), [
    (None, "recordings"), ("model", "recordings"), ("test", "test"), ("nope", "recordings")])
def test_the_server_renders_the_tab_of_the_url(client, tab, shown):
    voice, _ = voice_with_example(client)
    body = leaf(client, voice.id, **({"tab": tab} if tab else {}))
    assert re.search(rf'data-tab="{shown}"\s+aria-selected="true"', body)
    parts = dict(re.findall(r'data-tab-part="(\w+)"( hidden)?>', body))
    assert parts.pop(shown) == ""
    assert set(parts.values()) == {" hidden"}


def test_a_pending_voice_leaf_shows_deleting_and_no_tabs(queued):
    voice, _ = voice_with_example(queued)
    queued.post(f"/voices/{voice.id}/delete")
    body = leaf(queued, voice.id)
    assert "deleting…" in body and "data-deleting" in body
    assert 'role="tab"' not in body and "voice-menu" not in body


def _node_run(script: str) -> dict:
    done = subprocess.run(["node", "-e", script], capture_output=True, text=True, timeout=30,
                          check=False)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def _pure(*heads: str) -> str:
    source = _script()
    return "\n".join(f"{_js_function(source, head)}\n}}" for head in heads)


@needs_node
def test_came_from_grid_tab_from_and_leave_needs_confirm():
    out = _node_run(_pure("function cameFromGrid", "function tabFrom", "function leaveNeedsConfirm")
                    + """
      const TABS = ['recordings', 'test'];
      console.log(JSON.stringify({
        grid: cameFromGrid('http://h/voices', 'http://h'),
        leaf: cameFromGrid('http://h/voices/3', 'http://h'),
        foreign: cameFromGrid('http://x/voices', 'http://h'),
        none: cameFromGrid('', 'http://h'),
        test: tabFrom('?tab=test'), model: tabFrom('?x=1&tab=model'),
        unknown: tabFrom('?tab=x'), empty: tabFrom(''),
        live: leaveNeedsConfirm('live'), take: leaveNeedsConfirm('take'),
        prepared: leaveNeedsConfirm('prepared'), idle: leaveNeedsConfirm('idle'),
      }));
    """)
    assert out == {"grid": True, "leaf": False, "foreign": False, "none": False,
                   "test": "test", "model": "recordings", "unknown": "recordings",
                   "empty": "recordings", "live": True, "take": True, "prepared": False,
                   "idle": False}


LIFECYCLE = r"""
const flush = () => new Promise((resolve) => setImmediate(resolve));
function node(name) {
  const attrs = new Set();
  return { name, hasAttribute: (a) => attrs.has(a), setAttribute: (a) => attrs.add(a),
           removeAttribute: (a) => attrs.delete(a) };
}
const h = { page: node('fresh'), timers: new Map(), next: 1, fetches: [], replies: [],
            starts: [], stops: 0, tracks: [] };
const life = lifecycle({
  page: () => h.page,
  setTimeout: (run, ms) => { const id = h.next++; h.timers.set(id, { run, page: h.page.name }); return id; },
  clearTimeout: (id) => { h.timers.delete(id); },
  fetch: () => new Promise((resolve) => h.fetches.push(resolve)),
  start: (session, page) => {
    h.starts.push(page.name);
    session.schedule(() => {}, 2000);   // the one poller
    const track = { stopped: false, stop() { this.stopped = true; } };
    h.tracks.push(track);
    session.hold({ getTracks: () => [track] });
    session.fetch('/voices/preview').then(() => h.replies.push(page.name));
  },
  stop: () => { h.stops += 1; },
});
const pollers = () => [...h.timers.values()].map((timer) => timer.page);
"""


@needs_node
def test_the_lifecycle_mounts_once_and_unmount_stops_everything():
    out = _node_run(_pure("function lifecycle") + LIFECYCLE + """
      (async () => {
        const result = {};
        life.mount(); life.mount();
        life.pageshow({ persisted: false });
        result.fresh = { starts: [...h.starts], pollers: pollers() };
        life.unmount();
        result.left = { pollers: pollers(), stops: h.stops,
                        tracks: h.tracks.map((track) => track.stopped) };
        h.fetches.forEach((resolve) => resolve({ ok: true })); await flush();
        result.late = h.replies.length;
        console.log(JSON.stringify(result));
      })();
    """)
    assert out["fresh"] == {"starts": ["fresh"], "pollers": ["fresh"]}
    assert out["left"] == {"pollers": [], "stops": 1, "tracks": [True]}
    assert out["late"] == 0                       # a reply after unmount changes nothing


@needs_node
def test_native_back_forward_and_the_htmx_restore_each_end_with_one_poller():
    out = _node_run(_pure("function lifecycle") + LIFECYCLE + """
      (async () => {
        const result = {};
        life.mount();
        // Native Back away, then Forward to the page from bfcache: no script runs again.
        life.unmount();                        // pagehide
        life.pageshow({ persisted: true });
        result.back = pollers();
        life.unmount();
        life.pageshow({ persisted: true });
        result.forward = pollers();
        // htmx: the snapshot, a boosted nav, then the restore swaps NEW nodes in.
        life.unmount();                        // htmx:beforeHistorySave
        h.page = node('restored');
        life.mount();                          // htmx:historyRestore
        life.mount();                          // htmx:afterSettle
        result.htmx = pollers();
        h.fetches.forEach((resolve) => resolve({ ok: true })); await flush();
        result.replies = h.replies;
        result.starts = h.starts;
        console.log(JSON.stringify(result));
      })();
    """)
    assert out["back"] == ["fresh"] and out["forward"] == ["fresh"]
    assert out["htmx"] == ["restored"]
    assert out["replies"] == ["restored"]          # only the current session's reply lands
    assert out["starts"] == ["fresh", "fresh", "fresh", "restored"]


@needs_node
def test_the_script_wires_once_and_captures_no_node_at_evaluation():
    out = _node_run(r"""
      const fs = require('fs');
      const vm = require('vm');
      const source = fs.readFileSync('web/static/voices.js', 'utf8');
      const calls = { document: {}, window: {}, ids: [] };
      const count = (bucket) => (type) => { bucket[type] = (bucket[type] || 0) + 1; };
      const document = {
        addEventListener: count(calls.document),
        getElementById: (id) => { calls.ids.push(id); return null; },
        querySelector: () => null,
      };
      const window = { addEventListener: count(calls.window) };
      const context = vm.createContext({ window, document, location: { search: '', origin: 'http://h' },
                                         history: {}, navigator: {}, URL, URLSearchParams });
      vm.runInContext(source, context);
      vm.runInContext(source, context);
      console.log(JSON.stringify(calls));
    """)
    assert out["document"] == dict.fromkeys(
        ["submit", "click", "input", "change", "htmx:confirm", "keydown", "htmx:historyRestore",
         "htmx:afterSettle", "htmx:beforeHistorySave"], 1)
    assert out["window"] == {"pageshow": 1, "pagehide": 1}
    # The only look-up at evaluation is the page itself; every other node is found at event time.
    assert set(out["ids"]) == {"voices-page"}


def test_a_nav_click_asks_before_it_discards_a_recording_and_close_knows_the_grid():
    source = _script()
    confirm = source[source.index("document.addEventListener('htmx:confirm'"):]
    confirm = confirm[:confirm.index("\n  });\n")]
    assert "source.closest('nav.nav')" in confirm and "source.closest('#settings-open')" in confirm
    assert confirm.index("leaveNeedsConfirm(recorderState())") < confirm.index("event.preventDefault()")
    assert confirm.index("confirm(DISCARD)") < confirm.index("life.unmount()") < confirm.index(
        "event.detail.issueRequest(true)")
    assert "const DISCARD = 'Discard the recording you have not saved?';" in source
    close = _js_function(source, "function close()")
    assert close.index("leaveNeedsConfirm(recorderState()) && !confirm(DISCARD)") < close.index(
        "history.back()")
    assert "page.hasAttribute('data-from-grid')" in close
    assert "location.replace('/voices')" in close
    assert "history.length" not in source
    assert "if (window.__voicesWired) { return; }" in source
    mount = _js_function(source, "function mountLeaf")
    assert "cameFromGrid(document.referrer, location.origin)" in mount
    assert "page.setAttribute('data-from-grid', '')" in mount


def test_a_tab_switch_replaces_the_url_and_the_recorder_survives_a_swap():
    source = _script()
    tab = _js_function(source, "function showTab")
    assert "history.replaceState(history.state, '', url)" in tab
    assert "pushState" not in source
    swap = _js_function(source, "function replaceList")
    assert swap.index("recorder.remove()") < swap.index("current.replaceWith(next)") < swap.index(
        "placeRecorder(next, recorder)")
    prepare = _js_function(source, "function prepare")
    assert "placeRecorder(document.getElementById('voices'), recorderNode())" in prepare


OWNED = (*PURE, "function lifecycle", "function certOf", "function fragmentOf",
         "function replaceList", "function pollWhileDeleting", "async function detailOf",
         "async function send", "function pollCards", "async function refreshCards",
         "async function fetchStatus", "function stopPreviewTimers", "function showPreviewState",
         "function pollPreview", "async function mutate", "function mountLeaf",
         "function start(session", "function stop()")

# Leaf 1 reaches a held reply body; the test then leaves it for leaf 2 (the same epoch,
# completed and generation), the grid or no page, and lets the old body land.
PAGE = r"""
const flush = async () => { for (let i = 0; i < 10; i += 1) { await new Promise((resolve) => setImmediate(resolve)); } };
const POLL_MS = 2000;
const STATUS_URL = '/voices/preview';
const h = { page: null, list: null, job: null, writes: [], timers: new Map(), next: 1,
            fetches: [], bodies: [], crashes: [] };
process.on('unhandledRejection', (error) => h.crashes.push(String(error)));
let mounted = null;
let sync = null;
let pollTimer = null;
let cardsTimer = null;
let previewTimer = null;
let hearingTimer = null;
const document = {
  referrer: '',
  getElementById: (id) => (id === 'voices' ? h.list : null),
  querySelector: (selector) => (selector === '#voices [data-deleting]' && h.list && h.list.deleting ? {} : null),
};
const location = { origin: 'http://h', search: '' };
// "NAME@epoch/completed", plus "+deleting"; "epoch/run/state/completed" for a job.
function listNode(text) {
  const [head, flag] = text.split('+');
  const [name, cert] = head.split('@');
  const dataset = {};
  if (cert) { [dataset.epoch, dataset.completed] = cert.split('/'); }
  return { name, dataset, deleting: flag === 'deleting', contains: () => false,
           replaceWith(next) { h.writes.push(`list:${next.name}`); h.list = next; } };
}
function jobNode(text) {
  const [epoch, run, state, completed] = text.split('/');
  return { dataset: { epoch, run, state, completed, voiceId: '1', exampleId: '1' } };
}
const pageNode = () => h.page;
const onGrid = () => !!h.page && !h.page.dataset.voiceId;
const leafUrl = () => `/voices/${h.page.dataset.voiceId}`;
const cameFromGrid = () => false;
const tabFrom = () => 'recordings';
const showTab = () => {};
const decorate = () => {};
const markOtherVoice = () => {};
const selectTest = () => {};
const carryInputs = () => {};
const placeRecorder = () => {};
const resetRecorder = () => {};
const typingInList = () => false;
const listOf = (html) => listNode(html);
const parseJob = (html) => (html.startsWith('JOB:') ? fragmentOf(jobNode(html.slice(4))) : null);
const jobBox = () => h.job;
const previewBusy = () => !!h.job && h.job.dataset.state === 'running';
const showError = (text) => h.writes.push(`error:${text}`);
const clearError = () => {};
const replaceCards = (html) => h.writes.push(`cards:${html}`);
const applyJob = (fragment) => h.writes.push(`job:${fragment.epoch}/${fragment.run}/${fragment.state}`);
const showRestarted = () => h.writes.push('restarted');
const reconnectHearing = () => {};
const markHearing = () => {};
const hearingBusyHere = () => false;
const pollHearing = () => {};
/* OWNED */
const life = lifecycle({
  page: () => h.page,
  setTimeout: (run) => { const id = h.next++; h.timers.set(id, run); return id; },
  clearTimeout: (id) => { h.timers.delete(id); },
  fetch: (url, options = {}) => new Promise((resolve, reject) => h.fetches.push({ url, resolve, reject })),
  start,
  stop,
});
function show(voiceId, { deleting = false, job = null } = {}) {
  const attrs = new Set();
  h.page = { dataset: voiceId ? { voiceId: String(voiceId), epoch: 'e1', run: '0', state: 'idle', completed: '0' } : {},
             hasAttribute: (name) => attrs.has(name), setAttribute: (name) => attrs.add(name),
             removeAttribute: (name) => attrs.delete(name) };
  h.list = voiceId ? listNode(`L${voiceId}@e1/0${deleting ? '+deleting' : ''}`) : null;
  h.job = job ? jobNode(`e1/0/${job}/0`) : null;
  life.mount();
}
async function fire() {
  const runs = [...h.timers.values()];
  h.timers.clear();
  runs.forEach((run) => run());
  await flush();
}
const held = () => new Promise((resolve, reject) => h.bodies.push({ resolve, reject }));
// Answers the newest open request to `url` with headers only: its body stays held.
async function headers(url, ok = true) {
  const call = [...h.fetches].reverse().find((request) => !request.answered && request.url === url);
  call.answered = true;
  call.resolve({ ok, status: ok ? 200 : 409, headers: {}, text: () => held(), json: () => held() });
  await flush();
}
async function body(value) { h.bodies.shift().resolve(value); await flush(); }
const SITES = {
  deletingPoll: { leaf: { deleting: true }, late: 'OLD@e1/0+deleting', async arm() {
    await fire(); await headers('/voices/1/list');
  } },
  sendDetail: { leaf: {}, late: { detail: 'taken' }, async arm() {
    send('/voices/1/rename', null); await flush(); await headers('/voices/1/rename', false);
  } },
  sendSuccess: { leaf: {}, late: 'OLD@e1/0', async arm() {
    send('/voices/1/rename', null); await flush(); await headers('/voices/1/rename');
  } },
  mutateBody: { leaf: {}, late: 'JOB:e1/5/running/0', async arm() {
    mutate('/voices/preview/cancel'); await flush(); await headers('/voices/preview/cancel');
  } },
  mutateDetail: { leaf: {}, late: { detail: 'busy' }, async arm() {
    mutate('/voices/preview/cancel'); await flush(); await headers('/voices/preview/cancel', false);
  } },
  pollPreview: { leaf: { job: 'running' }, late: 'JOB:e1/5/running/0', async arm() {
    pollPreview(); await fire(); await headers(STATUS_URL);
  } },
  reconcile: { leaf: {}, late: 'JOB:e2/0/idle/0', async arm() {
    sync.show({ epoch: 'e2', run: 0, state: 'idle', completed: 0 }, sync.generation()); await flush();
    await headers(STATUS_URL);
  } },
  repair: { leaf: {}, late: 'OLD@e1/0', async arm() {
    send('/voices/1/rename', null); await flush();
    await headers('/voices/1/rename'); await body('BARE');      // refused: the repair starts
    await headers(STATUS_URL); await body('JOB:e1/0/idle/0');  // then it reads the list
    await headers('/voices/1/list');
  } },
};
async function land(name, destination, outcome) {
  const site = SITES[name];
  show(1, site.leaf); await flush();
  await site.arm();
  const bodies = h.bodies.length;
  const generation = sync.generation();
  if (destination !== 'alive') {
    life.unmount();
    if (destination === 'leaf') {
      show(2, site.leaf);
      while (sync.generation() < generation) { sync.bump(); }
    } else if (destination === 'grid') {
      show(null);
    } else {
      h.page = null; h.list = null; h.job = null;
    }
    await flush();
  }
  const writes = h.writes.length;
  const fetches = h.fetches.length;
  const timers = [...h.timers.keys()].join();
  const last = h.bodies.shift();
  if (outcome === 'resolve') { last.resolve(site.late); } else { last.reject(new Error('cut')); }
  await flush();
  return { bodies, writes: h.writes.slice(writes), fetches: h.fetches.slice(fetches).map((call) => call.url),
           timers: [...h.timers.keys()].join() !== timers, crashes: h.crashes };
}
"""


def _land(site: str, destination: str, outcome: str) -> dict:
    script = PAGE.replace("/* OWNED */", _pure(*OWNED))
    return _node_run(f"{script}\n(async () => {{ console.log(JSON.stringify("
                     f"await land('{site}', '{destination}', '{outcome}'))); }})();")


@needs_node
@pytest.mark.parametrize("outcome", ["resolve", "reject"])
@pytest.mark.parametrize("destination", ["leaf", "grid", "none"])
@pytest.mark.parametrize("site", ["deletingPoll", "sendDetail", "sendSuccess", "mutateBody",
                                  "mutateDetail", "pollPreview", "reconcile", "repair"])
def test_a_body_that_lands_after_unmount_changes_nothing(site, destination, outcome):
    out = _land(site, destination, outcome)
    assert out == {"bodies": 1, "writes": [], "fetches": [], "timers": False, "crashes": []}


@needs_node
def test_a_body_that_lands_while_its_owner_lives_swaps_the_list():
    out = _land("deletingPoll", "alive", "resolve")
    assert out == {"bodies": 1, "writes": ["list:OLD"], "fetches": [], "timers": True, "crashes": []}


TEST_TAB = r"""
function option(value, lang, words) {
  return { value, dataset: lang ? { lang, words } : {}, hidden: false, disabled: false };
}
function select(options, value) {
  const node = { options, selectedIndex: Math.max(0, options.findIndex((o) => o.value === value)) };
  Object.defineProperty(node, 'value', {
    get: () => node.options[node.selectedIndex].value,
    set: (v) => { node.selectedIndex = node.options.findIndex((o) => o.value === v); },
  });
  return node;
}
function language_(ready, value, take) {
  return { value, dataset: ready && take ? { take } : {}, hidden: false, disabled: false };
}
function page(ready, language = 'en', text = 'custom', typed = 'Typed.') {
  const row = { hidden: false };
  const texts = select([
    option('custom'), option('take:4', 'en', 'My words.'), option('take:5', 'en', 'More words.'),
    option('take:6', 'ru', 'Мои слова.'),
  ], text);
  texts.closest = () => row;
  const parts = {
    '.test-language': select([language_(ready, 'en', '3'), language_(ready, 'ru', '9'),
                              language_(ready, 'de', '')], language),
    '.test-text': texts,
  };
  const list = { dataset: ready ? { testReady: '1' } : {},
                 querySelector: (selector) => parts[selector] || null };
  const box = { shownHtml: undefined, replaceChildren() {} };
  const panel = {
    '.preview-count': { hidden: true, textContent: '' },
    '.preview-custom': { hidden: true, value: typed, maxLength: 300 },
    '.preview-generate': { disabled: false },
    '.preview-through': { textContent: '' },
    '.preview-compare': box,
  };
  const nodes = { voices: list, 'voice-preview': { querySelector: (selector) => panel[selector] } };
  return { parts, panel, nodes, row };
}
let current = null;
const document = { getElementById: (id) => current.nodes[id] || null,
                   querySelector: (selector) => current.parts[selector.replace('#voices ', '')] || null };
const pageNode = () => ({ dataset: { voiceId: '7' } });
const posted = [];
class FormData { constructor() { this.fields = []; } append(k, v) { this.fields.push([k, v]); } }
const mutate = (url, body) => posted.push({ url, fields: body.fields });
function view(p) {
  syncTest();
  return {
    shown: p.parts['.test-text'].options.filter((o) => o.dataset.lang && !o.hidden).map((o) => o.value),
    value: p.parts['.test-text'].value,
    row: !p.row.hidden,
    words: p.panel['.preview-custom'].value,
    box: !p.panel['.preview-custom'].hidden,
    count: p.panel['.preview-count'].textContent,
    generate: !p.panel['.preview-generate'].disabled,
    through: p.panel['.preview-through'].textContent,
    target: testTarget(),
  };
}
"""


def _test_tab(scenario: str) -> dict:
    script = (_pure("function testTarget", "function syncTest", "function editTest",
                    "function selectTest", "function generatePreview") + TEST_TAB + scenario)
    return _node_run(script)


@needs_node
def test_sync_test_lists_the_takes_of_the_language_and_the_box_is_always_there():
    out = _test_tab("""
      const result = {};
      current = page(true); result.own = view(current);
      current.parts['.test-language'].value = 'ru'; result.russian = view(current);
      current = page(true, 'de'); result.none = view(current);
      console.log(JSON.stringify(result));
    """)
    assert out["own"]["shown"] == ["take:4", "take:5"] and out["own"]["row"] and out["own"]["box"]
    assert out["own"]["target"] == {"voiceId": "7", "language": "en", "kind": "custom",
                                    "textId": "", "words": "", "ready": True}
    assert (out["own"]["words"], out["own"]["count"]) == ("Typed.", "6 / 300")
    assert out["own"]["generate"] and out["own"]["through"] == "from take #3 only"
    assert (out["russian"]["shown"], out["russian"]["words"]) == (["take:6"], "Typed.")
    assert (out["none"]["row"], out["none"]["box"], out["none"]["generate"]) == (False, True, True)


@needs_node
def test_a_take_fills_the_box_and_an_edit_makes_it_own_text():
    out = _test_tab("""
      const result = {};
      current = page(true, 'en', 'custom', 'Thrown away.');
      current.parts['.test-text'].value = 'take:4'; result.take = view(current);
      current.panel['.preview-custom'].value = 'My words, edited.'; editTest();
      result.edited = view(current);
      current.parts['.test-text'].value = 'take:5'; view(current);
      current.parts['.test-language'].value = 'ru'; result.moved = view(current);
      console.log(JSON.stringify(result));
    """)
    assert (out["take"]["words"], out["take"]["target"]["kind"], out["take"]["target"]["textId"]) == (
        "My words.", "take", "4")
    assert out["take"]["count"] == "Words of take #4. Edit them to speak your own text."
    assert (out["edited"]["value"], out["edited"]["words"]) == ("custom", "My words, edited.")
    assert out["edited"]["count"] == "17 / 300"
    assert (out["moved"]["value"], out["moved"]["words"], out["moved"]["generate"]) == (
        "custom", "", False)


@needs_node
def test_generate_needs_a_text_within_the_limit_but_a_long_take_is_fine():
    out = _test_tab("""
      const long = 'x'.repeat(301);
      const result = {};
      current = page(true, 'en', 'custom', '   '); result.blank = view(current);
      current = page(true, 'en', 'custom', long); result.over = view(current);
      current = page(true, 'en', 'take:4');
      current.parts['.test-text'].options[1].dataset.words = long; result.take = view(current);
      console.log(JSON.stringify(result));
    """)
    assert (out["blank"]["generate"], out["blank"]["count"]) == (False, "0 / 300")
    assert (out["over"]["generate"], out["over"]["count"]) == (False, "301 / 300")
    assert out["take"]["generate"] is True


@needs_node
def test_generate_is_disabled_without_a_ready_language_and_posts_the_person_test():
    out = _test_tab("""
      current = page(false); const blocked = view(current); generatePreview();
      const none = posted.length;
      current = page(true, 'en', 'take:4'); view(current); generatePreview();
      console.log(JSON.stringify({ blocked, none, posted }));
    """)
    assert out["blocked"]["generate"] is False and out["blocked"]["through"] == ""
    assert out["none"] == 0
    assert out["posted"] == [{"url": "/voices/7/preview", "fields": [
        ["language", "en"], ["text_kind", "take"], ["text_id", "4"], ["text", "My words."]]}]


@needs_node
def test_a_swap_keeps_the_test_language_and_text():
    out = _node_run(_pure("function carryInputs") + r"""
      const node = (value, options) => ({ value, options: options.map((v) => ({ value: v })) });
      const make = (language, text) => {
        const parts = { '.voice-language': node('en', ['en', 'ru']),
                        '.test-language': node(language, ['en', 'ru']),
                        '.test-text': node(text, ['custom', 'take:4']) };
        return { querySelector: (s) => parts[s] || null, parts };
      };
      const from = make('ru', 'take:4');
      const to = make('en', 'custom');
      carryInputs(from, to);
      console.log(JSON.stringify([to.parts['.test-language'].value, to.parts['.test-text'].value]));
    """)
    assert out == ["ru", "take:4"]


@needs_node
def test_a_kit_rows_record_binds_its_language_and_text():
    out = _node_run(_pure("function chosen") + r"""
      const add = { dataset: { voiceId: '7' }, querySelector: () => ({ value: 'en' }) };
      const list = { querySelector: (s) => (s === '.voice-add' ? add : null) };
      const row = { dataset: { lang: 'ru', promptId: 'ru-slow-1' } };
      const button = (inRow) => ({ closest: (s) => (s === '#voices' ? list : s === '.kit-row' && inRow ? row : null) });
      console.log(JSON.stringify({ kit: chosen(button(true)), add: chosen(button(false)) }));
    """)
    assert out["kit"] == {"voiceId": "7", "language": "ru", "promptId": "ru-slow-1", "kit": True}
    assert out["add"] == {"voiceId": "7", "language": "en"}


def test_the_script_has_no_film_take_or_words_code_and_opens_the_recorder_at_its_row():
    source = _script()
    for gone in (".example-preview", ".film-take-select", ".test-take", "film-example",
                 "'transcript'", "selectTake", "/examples/${"):
        assert gone not in source, gone
    save = _js_function(source, "async function saveTake")
    assert [field for field in re.findall(r"body\.append\('(\w+)'", save)] == [
        "language", "source", "prompt_id", "file"]
    place = _js_function(source, "function placeRecorder")
    assert place.index("held.kit") < place.index("row.append(recorder)") < place.index(".voice-add")
    prepare = _js_function(source, "function prepare")
    assert "recorderPart('.recorder-next').hidden = !!prepared.kit;" in prepare
    assert "recorderPart('.recorder-hint').hidden = !promptId;" in _js_function(source, "function showPrompt")
    carry = _js_function(source, "function carryInputs")
    assert "['.voice-language', '.test-language', '.test-text']" in carry
    click = source[source.index("document.addEventListener('click'"):]
    assert "voice-bake" not in click and "showTab('model'" not in click
    mount = _js_function(source, "function mountLeaf")
    assert "selectTest(job.dataset.language);" in mount


RECORDING = ("async function startRecording", "function togglePause", "function stopRecording",
             "function resetRecorder", "function discardTake", "function recordAgain")

# The real recorder code against a fake microphone, MediaRecorder and clock. `advance(ms)`
# runs the 200 ms tick; `finish()` is the browser's late dataavailable + stop events.
RECORDER = r"""
const flush = () => new Promise((resolve) => setImmediate(resolve));
const MAX_SECONDS = 30;
let now = 0;
const performance = { now: () => now };
const intervals = new Map();
let nextInterval = 1;
const setInterval = (run) => { intervals.set(nextInterval, run); nextInterval += 1; return nextInterval - 1; };
const clearInterval = (id) => { intervals.delete(id); };
function advance(ms) {
  for (let step = 0; step < ms; step += 200) { now += 200; [...intervals.values()].forEach((run) => run()); }
}
const recorders = [];
class MediaRecorder {
  constructor(stream) { this.stream = stream; this.state = 'inactive'; this.mimeType = 'audio/webm'; this.calls = []; recorders.push(this); }
  start() { this.state = 'recording'; this.calls.push('start'); }
  pause() { if (this.state !== 'recording') { throw new Error('InvalidStateError'); } this.state = 'paused'; this.calls.push('pause'); }
  resume() { if (this.state !== 'paused') { throw new Error('InvalidStateError'); } this.state = 'recording'; this.calls.push('resume'); }
  stop() { if (this.state === 'inactive') { throw new Error('InvalidStateError'); } this.state = 'inactive'; this.calls.push('stop'); }
  finish() { this.ondataavailable({ data: { size: 3 } }); this.onstop(); }
}
class Blob { constructor(parts, options) { this.parts = parts; this.type = options.type; } }
const mics = [];
const navigator = { mediaDevices: { getUserMedia: () => new Promise((resolve, reject) => mics.push({ resolve, reject })) } };
const streams = [];
function allow() {
  const stream = { stopped: 0 };
  stream.getTracks = () => [{ stop: () => { stream.stopped += 1; } }];
  streams.push(stream);
  mics.shift().resolve(stream);
  return flush();
}
const audios = [0, 1].map(() => ({ paused: false, pause() { this.paused = true; } }));
const document = { querySelectorAll: (selector) => (selector === 'audio' ? audios : []), getElementById: () => null };
const parts = {
  '.recorder-timer': { textContent: '' },
  '.recorder-status': { textContent: '' },
  '.recorder-pause': { textContent: 'Pause', disabled: true },
  '.recorder-live': { hidden: true },
};
const recorderPart = (selector) => parts[selector];
const recorderNode = () => ({});
const h = { takes: [], closed: 0, errors: [] };
const openPanel = () => {};
const closePanel = () => { h.closed += 1; prepared = null; };
const clearError = () => {};
const showError = (text) => h.errors.push(text);
const mounted = { hold: () => {} };
const makeTake = (blob, source) => h.takes.push({ chunks: blob.parts.length, source });
let prepared = null;
let live = null;
let take = null;
/* CLOCK */
/* OWNED */
async function begin() {
  prepared = { voiceId: '7', language: 'en', promptId: 'en-1', kit: true };
  startRecording();
  await flush();
}
const recorder = () => recorders[recorders.length - 1];
const view = () => ({
  timer: parts['.recorder-timer'].textContent, button: parts['.recorder-pause'].textContent,
  disabled: parts['.recorder-pause'].disabled, status: parts['.recorder-status'].textContent,
  state: live && live.recorder ? live.recorder.state : null, live: !!live,
});
"""


def _recorder(scenario: str) -> dict:
    source = _script()
    clock = next(line for line in source.splitlines() if line.strip().startswith("const clock ="))
    script = RECORDER.replace("/* CLOCK */", clock).replace("/* OWNED */", _pure(*RECORDING))
    return _node_run(f"{script}\n(async () => {{\n{scenario}\n}})();")


@needs_node
def test_record_again_drops_the_take_and_records_the_same_text_at_once():
    out = _recorder("""
      await begin(); await allow(); advance(6000);
      stopRecording(false); recorder().finish();
      take = { voiceId: '7', language: 'en', promptId: 'en-1', kit: true,
               source: 'recorded', wav: {}, url: 'blob:first' };
      audios.forEach((audio) => { audio.paused = false; });
      recordAgain();
      const retake = { take, target: live.target, audio: audios.map((audio) => audio.paused) };
      await allow(); advance(6000);
      stopRecording(false); recorder().finish();
      take = null;
      recordAgain();
      console.log(JSON.stringify({ retake, takes: h.takes.length, recorders: recorders.length,
                                   mics: mics.length, live: !!live }));
    """)
    assert out["retake"] == {
        "take": None, "audio": [True, True],
        "target": {"voiceId": "7", "language": "en", "promptId": "en-1", "kit": True}}
    assert (out["takes"], out["recorders"], out["mics"], out["live"]) == (2, 2, 0, False)


def test_only_a_recording_offers_record_again():
    source = _script()
    assert ("recorderPart('.recorder-again').hidden = source !== 'recorded';"
            in _js_function(source, "async function makeTake"))
    click = source[source.index("document.addEventListener('click'"):]
    assert "if (event.target.closest('.recorder-again')) { recordAgain(); return; }" in click


@needs_node
def test_pauses_freeze_the_timer_and_the_cycles_make_one_take():
    out = _recorder("""
      const seen = {};
      await begin(); await allow();
      advance(3000); togglePause(); seen.paused = view();
      advance(10000); seen.frozen = view();
      togglePause(); seen.resumed = view();
      advance(2000); togglePause(); advance(5000); togglePause(); advance(1000);
      seen.again = view();
      stopRecording(false); recorder().finish();
      console.log(JSON.stringify({ seen, calls: recorder().calls, recorders: recorders.length,
                                   takes: h.takes, live: !!live, intervals: intervals.size }));
    """)
    paused = "Paused — the pause is not recorded."
    assert out["seen"]["paused"] == {"timer": "0:03", "button": "Resume", "disabled": False,
                                     "status": paused, "state": "paused", "live": True}
    assert out["seen"]["frozen"]["timer"] == "0:03"
    assert out["seen"]["resumed"] == {"timer": "0:03", "button": "Pause", "disabled": False,
                                      "status": "", "state": "recording", "live": True}
    assert out["seen"]["again"]["timer"] == "0:06"
    assert out["calls"] == ["start", "pause", "resume", "pause", "resume", "stop"]
    assert out["recorders"] == 1
    assert out["takes"] == [{"chunks": 1, "source": "recorded"}]
    assert (out["live"], out["intervals"]) == (False, 0)


@needs_node
def test_the_30_s_limit_counts_recorded_time_only():
    out = _recorder("""
      await begin(); await allow();
      advance(20000); togglePause(); advance(60000);
      const paused = recorder().calls.slice();
      togglePause(); advance(9800);
      const before = recorder().calls.slice();
      advance(200);
      console.log(JSON.stringify({ paused, before, after: recorder().calls, view: view() }));
    """)
    assert out["paused"] == ["start", "pause"]
    assert out["before"] == ["start", "pause", "resume"]
    assert out["after"] == ["start", "pause", "resume", "stop"]
    assert (out["view"]["timer"], out["view"]["disabled"], out["view"]["state"]) == ("0:30", True, "inactive")


@needs_node
def test_resume_pauses_every_audio_on_the_page_again():
    out = _recorder("""
      await begin(); await allow();
      togglePause();
      audios.forEach((audio) => { audio.paused = false; });
      togglePause();
      console.log(JSON.stringify(audios.map((audio) => audio.paused)));
    """)
    assert out == [True, True]


@needs_node
def test_stop_esc_and_unmount_while_paused_work_as_without_a_pause():
    out = _recorder("""
      const result = {};
      await begin(); await allow(); advance(6000); togglePause();
      stopRecording(false); recorder().finish();
      result.stop = { takes: h.takes.length, closed: h.closed, live: !!live, stopped: streams[0].stopped };
      await begin(); await allow(); advance(6000); togglePause();
      stopRecording(true); recorder().finish();
      result.esc = { takes: h.takes.length, closed: h.closed, live: !!live, stopped: streams[1].stopped };
      await begin(); await allow(); advance(6000); togglePause();
      resetRecorder();
      result.unmount = { state: recorder().state, live: !!live, intervals: intervals.size,
                         stopped: streams[2].stopped > 0 };
      recorder().finish();
      result.late = { takes: h.takes.length };
      console.log(JSON.stringify(result));
    """)
    assert out["stop"] == {"takes": 1, "closed": 0, "live": False, "stopped": 1}
    assert out["esc"] == {"takes": 1, "closed": 1, "live": False, "stopped": 1}
    assert out["unmount"] == {"state": "inactive", "live": False, "intervals": 0, "stopped": True}
    assert out["late"] == {"takes": 1}


@needs_node
def test_pause_does_nothing_while_waiting_for_the_microphone_or_after_stop():
    out = _recorder("""
      const result = {};
      togglePause();
      result.idle = view();
      await begin();
      togglePause();
      result.waiting = { ...view(), recorders: recorders.length };
      await allow(); togglePause();
      stopRecording(false);
      togglePause();
      result.stopped = { ...view(), calls: recorder().calls.slice() };
      recorder().finish();
      await begin();
      result.next = view();
      await allow();
      result.nextLive = view();
      console.log(JSON.stringify(result));
    """)
    assert out["idle"]["live"] is False
    assert out["waiting"] == {"timer": "0:00", "button": "Pause", "disabled": True, "status":
                              "Waiting for the microphone…", "state": None, "live": True,
                              "recorders": 0}
    assert out["stopped"]["calls"] == ["start", "pause", "stop"]
    assert (out["stopped"]["disabled"], out["stopped"]["state"]) == (True, "inactive")
    # A new recording after a paused one starts with Pause again, disabled until the microphone.
    assert (out["next"]["button"], out["next"]["disabled"]) == ("Pause", True)
    assert (out["nextLive"]["button"], out["nextLive"]["disabled"], out["nextLive"]["state"]) == (
        "Pause", False, "recording")


def test_the_pause_button_sits_between_the_timer_and_stop_and_is_wired(client):
    voice, _ = voice_with_example(client)
    body = leaf(client, voice.id)
    live = body[body.index('<div class="recorder-live" hidden>'):]
    live = live[:live.index("</div>")]
    assert live.index("recorder-timer") < live.index(
        '<button type="button" class="recorder-pause" disabled>Pause</button>') < live.index(
        '<button type="button" class="recorder-stop">Stop</button>')
    click = _script()[_script().index("document.addEventListener('click'"):]
    assert "if (event.target.closest('.recorder-pause')) { togglePause(); return; }" in click


class GatedASR(HeardModels):
    """`HeardModels` whose `transcribe` waits for `release` once it is `entered`;
    `error` is raised by every call."""

    def __init__(self) -> None:
        super().__init__()
        self.entered, self.release = threading.Event(), threading.Event()
        self.error: Exception | None = None

    def transcribe(self, audio: bytes, language: str) -> dict:
        if self.error is not None:
            raise self.error
        self.entered.set()
        assert self.release.wait(10)
        return super().transcribe(audio, language)


@pytest.fixture
def asr(monkeypatch):
    fake = GatedASR()
    monkeypatch.setattr(Settings, "build_models_client", lambda self: fake)
    yield fake
    fake.release.set()   # never leave a hearing thread waiting


def _hearing(client, voice_id) -> str:
    return client.get("/voices/hearing", params={"voice": voice_id}).text


def _cancel_hearing(client, voice_id) -> str:
    return client.post("/voices/hearing/cancel", params={"voice": voice_id}).text


def _words_state(client, voice_id) -> list[str]:
    return [take.words_state for take in sv.get_voice(ctx(client).conn, OWNER, voice_id).examples]


def _no_hear_button(body: str) -> bool:
    return 'class="voice-hear"' not in body


def test_a_saved_take_is_heard_and_becomes_the_take_films_speak_from(client, heard_models):
    voice = create_voice(ctx(client).conn, OWNER, "Grandma")
    assert post_example(client, voice.id).status_code == 200
    client.app.state.await_voice_hearing()
    assert heard_models.heard == [hashlib.sha256(clip()).hexdigest()]
    [take] = sv.get_voice(ctx(client).conn, OWNER, voice.id).examples
    assert (take.words_state, take.transcript) == ("heard", WORDS)
    assert f"English · films and tests speak from take #{take.id}" in _ready_lines(body_of(client, voice.id))
    assert 'data-state="idle"' in _hearing(client, voice.id)


def test_the_job_line_shows_the_stage_while_it_hears_and_cancel_stops_it(client, asr):
    voice = create_voice(ctx(client).conn, OWNER, "Grandma")
    reply = post_example(client, voice.id).text
    assert re.search(r'<div id="voice-hearing-job"[^>]*data-state="running"', reply)
    assert asr.entered.wait(10)
    response = client.get("/voices/hearing", params={"voice": voice.id})
    assert response.headers["cache-control"] == "no-store"
    assert "hearing the words · take 1 of 1 · 0%" in response.text
    assert 'class="voice-hearing-cancel">Cancel' in response.text
    body = body_of(client, voice.id)
    job = body[body.index('<div id="voice-hearing-job"'):]
    assert " hidden" not in job[:job.index(">")]
    assert _no_hear_button(body)                      # no Hear the words while it hears
    assert 'data-state="idle"' in _cancel_hearing(client, voice.id)
    asr.release.set()
    client.app.state.await_voice_hearing()
    assert 'data-state="idle"' in _hearing(client, voice.id)


def test_the_job_line_of_another_voice_is_hidden_on_this_voice(client, asr):
    heard, _ = voice_with_example(client, "Anna")
    other, _ = voice_with_example(client, "Boris", data=clip(freq=330.0))
    _hear_all(client, other.id)
    assert client.post(f"/voices/{heard.id}/hear").status_code == 200
    assert asr.entered.wait(10)
    assert _hearing(client, other.id).count('data-state="idle"') == 1
    assert 'class="voice-hearing-cancel"' not in body_of(client, other.id)


def test_a_take_of_another_voice_waits_its_turn_and_is_heard_after(client, asr):
    anna = create_voice(ctx(client).conn, OWNER, "Anna")
    boris = create_voice(ctx(client).conn, OWNER, "Boris")
    post_example(client, anna.id)
    assert asr.entered.wait(10)
    post_example(client, boris.id, clip(freq=330.0))
    assert client.app.state.voice_hearings.hears(boris.id)
    assert _no_hear_button(body_of(client, boris.id))
    asr.release.set()
    client.app.state.await_voice_hearing()
    assert _words_state(client, anna.id) == ["heard"] and _words_state(client, boris.id) == ["heard"]


def test_a_failed_hearing_shows_its_reason_until_close(client, asr):
    asr.error = http_error(503, "no ASR")
    voice = create_voice(ctx(client).conn, OWNER, "Grandma")
    post_example(client, voice.id)
    client.app.state.await_voice_hearing()
    job = _hearing(client, voice.id)
    assert 'data-state="failed"' in job and f'data-voice-id="{voice.id}"' in job
    assert "Hearing the words of a take is not offered on this device yet." in job
    assert 'class="voice-hearing-dismiss">Close' in job
    assert _words_state(client, voice.id) == ["waiting"]
    assert f'<button type="button" class="voice-hear" data-voice-id="{voice.id}">' in body_of(
        client, voice.id)
    assert 'data-state="idle"' in _cancel_hearing(client, voice.id)


def test_the_page_of_a_queued_voice_follows_its_work_to_the_end(client, asr):
    anna = create_voice(ctx(client).conn, OWNER, "Anna")
    boris = create_voice(ctx(client).conn, OWNER, "Boris")
    post_example(client, anna.id)
    assert asr.entered.wait(10)
    post_example(client, boris.id, clip(freq=330.0))
    page = leaf(client, boris.id)
    job = page[page.index('<div id="voice-hearing-job"'):]
    job = job[:job.index("</div>")]
    assert 'data-state="queued"' in job and f'data-voice-id="{boris.id}"' in job
    assert " hidden" not in job[:job.index(">")]
    assert "waiting to hear the words, after another voice" in job
    assert 'class="voice-hearing-cancel">Cancel' in job
    assert "words not heard yet" in page
    asr.release.set()
    client.app.state.await_voice_hearing()
    assert 'data-state="idle"' in _hearing(client, boris.id)
    assert _words_state(client, boris.id) == ["heard"]
    [take] = sv.get_voice(ctx(client).conn, OWNER, boris.id).examples
    body = body_of(client, boris.id)
    assert f"English · films and tests speak from take #{take.id}" in _ready_lines(body)
    assert "words not heard yet" not in body and 'data-test-ready="1"' in body


def test_the_page_of_a_queued_voice_shows_it_heard_after_the_one_before(client, asr):
    anna = create_voice(ctx(client).conn, OWNER, "Anna")
    boris = create_voice(ctx(client).conn, OWNER, "Boris")
    post_example(client, anna.id)
    assert asr.entered.wait(10)
    post_example(client, boris.id, clip(freq=330.0))
    hearings = client.app.state.voice_hearings
    assert hearings.view(boris.id)["state"] == "queued"
    held, asr.release = asr.release, threading.Event()
    asr.entered.clear()
    held.set()
    assert asr.entered.wait(10)            # Boris is heard now
    view = _hearing(client, boris.id)
    assert 'data-state="running"' in view and "hearing the words · take 1 of 1" in view
    assert 'data-state="idle"' in _hearing(client, anna.id)
    asr.release.set()
    client.app.state.await_voice_hearing()
    assert 'data-state="idle"' in _hearing(client, boris.id)


def test_a_cancel_of_a_queued_voice_leaves_the_voice_heard_now(client, asr):
    anna = create_voice(ctx(client).conn, OWNER, "Anna")
    boris = create_voice(ctx(client).conn, OWNER, "Boris")
    post_example(client, anna.id)
    assert asr.entered.wait(10)
    post_example(client, boris.id, clip(freq=330.0))
    assert 'data-state="idle"' in _cancel_hearing(client, boris.id)
    assert 'data-state="running"' in _hearing(client, anna.id)
    asr.release.set()
    client.app.state.await_voice_hearing()
    assert _words_state(client, anna.id) == ["heard"] and _words_state(client, boris.id) == ["waiting"]


class FailingFor(HeardModels):
    """`HeardModels` that refuses the takes of the clips in `refused`."""

    def __init__(self, refused: set[str]) -> None:
        super().__init__()
        self.refused = refused

    def transcribe(self, audio: bytes, language: str) -> dict:
        if hashlib.sha256(audio).hexdigest() in self.refused:
            raise RuntimeError("the clip broke the model")
        return super().transcribe(audio, language)


def test_a_failure_of_one_voice_never_drops_the_work_of_another(client, asr, monkeypatch):
    anna = create_voice(ctx(client).conn, OWNER, "Anna")
    boris = create_voice(ctx(client).conn, OWNER, "Boris")
    post_example(client, anna.id)                   # holds the job until Boris is queued
    assert asr.entered.wait(10)
    failing = FailingFor({hashlib.sha256(clip(freq=330.0)).hexdigest()})
    monkeypatch.setattr(Settings, "build_models_client", lambda self: failing)
    post_example(client, boris.id, clip(freq=330.0))
    carl = create_voice(ctx(client).conn, OWNER, "Carl")
    post_example(client, carl.id, clip(freq=440.0))
    asr.release.set()
    client.app.state.await_voice_hearing()
    assert _words_state(client, anna.id) == ["heard"] and _words_state(client, carl.id) == ["heard"]
    assert _words_state(client, boris.id) == ["waiting"]
    failed = _hearing(client, boris.id)
    [take] = sv.get_voice(ctx(client).conn, OWNER, boris.id).examples
    assert 'data-state="failed"' in failed
    assert f"Could not hear the words of take #{take.id} (the clip broke the model)." in failed
    assert 'data-state="idle"' in _hearing(client, carl.id)
    failing.refused.clear()
    assert 'class="voice-hear"' in body_of(client, boris.id)
    client.post(f"/voices/{boris.id}/hear")
    client.app.state.await_voice_hearing()
    assert _words_state(client, boris.id) == ["heard"]
    assert 'data-state="idle"' in _hearing(client, boris.id)


def test_hear_the_words_hears_the_waiting_takes(client, heard_models):
    voice, _ = voice_with_example(client)
    response = client.post(f"/voices/{voice.id}/hear")
    assert response.status_code == 200
    assert response.text.lstrip().startswith('<div id="voice-hearing-job"')
    client.app.state.await_voice_hearing()
    assert _words_state(client, voice.id) == ["heard"]


def test_hear_the_words_is_refused_for_an_unknown_or_pending_voice(queued):
    assert queued.post("/voices/99/hear").status_code == 404
    voice, _ = voice_with_example(queued)
    queued.post(f"/voices/{voice.id}/delete")
    response = queued.post(f"/voices/{voice.id}/hear")
    assert (response.status_code, response.json()["detail"]) == (409, "This voice is being deleted.")


def test_no_take_is_heard_on_a_device_without_speech_recognition(client, heard_models,
                                                                  monkeypatch):
    monkeypatch.setattr(voices_api, "transcriber_for", lambda profile: None)
    voice = create_voice(ctx(client).conn, OWNER, "Grandma")
    assert post_example(client, voice.id).status_code == 200
    response = client.post(f"/voices/{voice.id}/hear")
    client.app.state.await_voice_hearing()
    assert (response.status_code, response.json()["detail"]) == (
        409, "Hearing the words of a take is not offered on this device yet.")
    assert heard_models.heard == []


def test_on_cloud_a_saved_take_is_heard_and_a_test_speaks_from_its_words(settings, models):
    settings.profile = "cloud"
    with _client(settings) as cloud:
        voice = create_voice(ctx(cloud).conn, OWNER, "Grandma")
        assert post_example(cloud, voice.id).status_code == 200
        cloud.app.state.await_voice_hearing()
        found = sv.film_take(ctx(cloud).conn, OWNER, voice.id, "en")
        assert found.problem is None and found.take.transcript == WORDS
        assert _test(cloud, voice.id).status_code == 200
        cloud.app.state.await_voice_preview()
        assert _stored_tests(cloud) == [(voice.id, EN_2)]
    reference = models.references[0][sv.token(voice.id)]
    assert reference["sha256"] == found.take.content_hash and reference["transcript"] == WORDS
    assert "adapter" not in reference


class FailOnceGatedASR(HeardModels):
    """`HeardModels` whose first `transcribe` waits for `release` once it is `entered`,
    then fails; every later call hears the take."""

    def __init__(self) -> None:
        super().__init__()
        self.entered, self.release = threading.Event(), threading.Event()
        self.calls = 0

    def transcribe(self, audio: bytes, language: str) -> dict:
        self.calls += 1
        if self.calls == 1:
            self.entered.set()
            assert self.release.wait(10)
            raise RuntimeError("the first pass broke")
        return super().transcribe(audio, language)


def test_a_failed_pass_of_a_voice_queued_again_shows_no_failure_and_hears_both_takes(
        client, monkeypatch):
    fake = FailOnceGatedASR()
    monkeypatch.setattr(Settings, "build_models_client", lambda self: fake)
    voice = create_voice(ctx(client).conn, OWNER, "Grandma")
    post_example(client, voice.id)
    assert fake.entered.wait(10)
    post_example(client, voice.id, clip(freq=330.0))
    hearings = client.app.state.voice_hearings
    states = []
    fake.release.set()
    while hearings.working:
        states.append(hearings.view(voice.id)["state"])
        time.sleep(0.01)
    client.app.state.await_voice_hearing()
    states.append(hearings.view(voice.id)["state"])
    assert "failed" not in states and states[-1] == "idle"
    assert _words_state(client, voice.id) == ["heard", "heard"]


def test_a_hearing_that_ends_between_the_reads_of_the_body_leaves_the_page_following_it(
        client, asr, monkeypatch):
    voice = create_voice(ctx(client).conn, OWNER, "Grandma")
    post_example(client, voice.id)
    assert asr.entered.wait(10)
    read = voices_api.get_voice

    def read_then_finish(conn, owner_id, voice_id):
        found = read(conn, owner_id, voice_id)
        asr.release.set()
        client.app.state.await_voice_hearing()
        return found

    monkeypatch.setattr(voices_api, "get_voice", read_then_finish)
    body = body_of(client, voice.id)
    job = re.search(r'<div id="voice-hearing-job"[^>]*>', body).group(0)
    assert _words_state(client, voice.id) == ["heard"]
    assert 'data-state="running"' in job or "words not heard yet" not in body


def test_a_voice_deleted_while_its_body_is_read_says_so_in_its_header(queued, monkeypatch):
    voice, _ = voice_with_example(queued)
    read = voices_api.get_voice

    def read_then_delete(conn, owner_id, voice_id):
        found = read(conn, owner_id, voice_id)
        request_voice_deletion(conn, owner_id, voice_id)
        return found

    monkeypatch.setattr(voices_api, "get_voice", read_then_delete)
    response = queued.get(f"/voices/{voice.id}/list")
    assert response.status_code == 200
    assert "English · the voice is being deleted" in _ready_lines(response.text)


class BakeService(FakeService):
    """The fake models service of `tests/test_voice_bake.py` as the web job meets it.
    `bake_current` answers `answers` first, one per call, then `current`. A cancel of a
    bake this app did not start ends it (`foreign_cancels`). `log` keeps the order of the
    calls; `on_cancel` runs inside `bake_cancel`; each bake starts from the first stage."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current: dict | None = None
        self.answers: list = []
        self.log: list[str] = []
        self.foreign_cancels: list[str] = []
        self.on_cancel = None

    def bake_current(self):
        self.current_calls += 1
        self.log.append("current")
        return self.answers.pop(0) if self.answers else self.current

    def transcribe(self, audio, language):
        self.log.append("transcribe")
        return super().transcribe(audio, language)

    def bake_start(self, clips, *, tag, config=None):
        self.log.append("start")
        self.cancels, self.step = [], 0
        return super().bake_start(clips, tag=tag, config=config)

    def bake_cancel(self, bake_id):
        self.log.append(f"cancel:{bake_id}")
        if self.on_cancel is not None:
            self.on_cancel(bake_id)
        if bake_id != BAKE_ID:
            self.foreign_cancels.append(bake_id)
            self.current = None
            return dict(FOREIGN, state="cancelled")
        return super().bake_cancel(bake_id)

    def bake_delete(self, bake_id):
        self.log.append(f"delete:{bake_id}")
        return super().bake_delete(bake_id)

    def app_bake(self, conn, voice_id: int, bake_key: str, state: str = "running") -> dict:
        """The service's status of a bake this app started for the run `bake_key`."""
        self.tag = {"source": "app", "owner_id": OWNER, "voice_id": voice_id, "bake_key": bake_key}
        self.dataset = sv.read_bake_run(conn, OWNER, bake_key).dataset_hash
        return self._status(state, "training" if state == "running" else "done",
                            0.4 if state == "running" else 1.0)


@pytest.fixture
def service(monkeypatch):
    fake = BakeService()
    monkeypatch.setattr(Settings, "build_models_client", lambda self: fake)
    monkeypatch.setattr(voice_bake, "POLL_SECONDS", 0.0)
    return fake


@pytest.fixture
def baker(settings, service):
    """The app with the fake service from its start: its start-up reattach has run."""
    with _client(settings) as test_client:
        test_client.app.state.await_voice_bake()
        yield test_client
        test_client.app.state.await_voice_bake()


def _settle(client):
    client.app.state.await_voice_bake()


def _runs(client) -> list[dict]:
    return [dict(row) for row in ctx(client).conn.execute(
        "SELECT bake_key, voice_id, cancel_requested_at FROM voice_bake_runs")]


def _model_rows(client) -> list[tuple[int, int]]:
    return [tuple(row) for row in ctx(client).conn.execute(
        "SELECT voice_id, version FROM voice_models ORDER BY voice_id")]


def _seed_run(client, voice_id) -> str:
    """The run row a bake of this app writes before it starts: heard takes, a snapshot."""
    c = ctx(client)
    for take in sv.takes_to_hear(c.conn, OWNER, voice_id):
        sv.set_heard_words(c.conn, OWNER, voice_id, take.id, text="Hello there.", model=ASR)
    return sv.start_bake_run(c.conn, OWNER, voice_id, sv.bake_snapshot(c.conn, OWNER, voice_id))


def test_the_app_starts_no_bake_and_its_bake_routes_are_gone(baker, service):
    voice, _ = voice_with_example(baker)
    _hear_all(baker, voice.id)
    gone = [baker.post(f"/voices/{voice.id}/bake"),
            baker.post(f"/voices/{voice.id}/bake", data={"replace": "yes"}),
            baker.get("/voices/bake"), baker.post("/voices/bake/cancel")]
    assert [response.status_code for response in gone] == [404, 404, 422, 404]
    _settle(baker)
    assert service.starts == [] and _runs(baker) == [] and _model_rows(baker) == []


def test_reattach_adopts_this_apps_running_bake_and_leaves_the_bake_check_alone(baker, service,
                                                                               settings):
    voice, _ = voice_with_example(baker)
    bake_key = _seed_run(baker, voice.id)
    service.current = service.app_bake(ctx(baker).conn, voice.id, bake_key)
    with _client(settings) as restarted:
        restarted.app.state.await_voice_bake()
        assert restarted.app.state.voice_bakes.snapshot()["state"] == "done"
    assert _model_rows(baker) == [(voice.id, 1)]
    assert service.cancels == [] and service.deletes == [BAKE_ID] and _runs(baker) == []
    service.current, service.log = dict(FOREIGN), []
    with _client(settings) as restarted:
        restarted.app.state.await_voice_bake()
        assert restarted.app.state.voice_bakes.snapshot()["state"] == "idle"
    assert service.log == ["current"]                    # never adopted, cancelled or deleted


def test_a_restart_after_a_cancel_cancels_and_deletes_the_bake_and_stores_nothing(baker, service,
                                                                                  settings):
    voice, _ = voice_with_example(baker)
    bake_key = _seed_run(baker, voice.id)
    assert sv.request_bake_cancel(ctx(baker).conn, bake_key)
    service.current = service.app_bake(ctx(baker).conn, voice.id, bake_key)
    service.end_after_cancel = "done"                    # also a `done` end stores nothing
    with _client(settings) as restarted:
        restarted.app.state.await_voice_bake()
        assert restarted.app.state.voice_bakes.snapshot()["state"] == "idle"
    assert service.cancels == [BAKE_ID] and service.deletes == [BAKE_ID]
    assert _model_rows(baker) == [] and _runs(baker) == []


def test_a_restart_after_replace_drops_the_old_run_and_stores_the_new_one_once(baker, service,
                                                                               settings):
    anna, _ = voice_with_example(baker, "Anna")
    boris, _ = voice_with_example(baker, "Boris", data=clip(freq=330.0))
    old = _seed_run(baker, anna.id)
    assert sv.request_bake_cancel(ctx(baker).conn, old)
    service.current = service.app_bake(ctx(baker).conn, anna.id, old)
    with _client(settings) as restarted:
        restarted.app.state.await_voice_bake()
    assert _model_rows(baker) == [] and service.deletes == [BAKE_ID]
    new = _seed_run(baker, boris.id)
    service.current = service.app_bake(ctx(baker).conn, boris.id, new, state="done")
    with _client(settings) as restarted:
        restarted.app.state.await_voice_bake()
    assert _model_rows(baker) == [(boris.id, 1)]


def test_a_bake_of_this_app_with_no_run_row_is_cancelled_and_deleted(baker, service, settings):
    voice, _ = voice_with_example(baker)
    service.tag = {"source": "app", "owner_id": OWNER, "voice_id": voice.id, "bake_key": "gone"}
    service.current = service._status("running", "training", 0.4)
    with _client(settings) as restarted:
        restarted.app.state.await_voice_bake()
    assert service.cancels == [BAKE_ID] and service.deletes == [BAKE_ID]
    assert _model_rows(baker) == []


@needs_node
def test_a_hearing_of_this_voice_that_starts_ends_or_is_replaced_refreshes_the_body():
    out = _node_run(_pure("function hearingChanged") + r"""
      const job = (state, run, voiceId) => ({ state, run, voiceId });
      console.log(JSON.stringify({
        started: hearingChanged(job('idle', '0', ''), job('running', '1', '7'), '7'),
        queued: hearingChanged(job('idle', '0', ''), job('queued', '3', '7'), '7'),
        queued_ticked: hearingChanged(job('queued', '3', '7'), job('queued', '4', '7'), '7'),
        queued_ended: hearingChanged(job('queued', '3', '7'), job('idle', '5', '7'), '7'),
        ticked: hearingChanged(job('running', '1', '7'), job('running', '1', '7'), '7'),
        ended: hearingChanged(job('running', '1', '7'), job('idle', '1', '7'), '7'),
        failed: hearingChanged(job('running', '1', '7'), job('failed', '1', '7'), '7'),
        replaced: hearingChanged(job('running', '1', '7'), job('running', '2', '7'), '7'),
        moved: hearingChanged(job('running', '1', '7'), job('running', '2', '8'), '7'),
        other: hearingChanged(job('idle', '0', ''), job('running', '1', '8'), '7'),
      }));
    """)
    assert out == {"started": True, "queued": True, "queued_ticked": False, "queued_ended": True,
                   "ticked": False, "ended": True, "failed": True, "replaced": True, "moved": True,
                   "other": False}


@needs_node
def test_typing_during_the_hearing_refresh_keeps_the_name_and_the_refresh_lands_after():
    out = _node("""
      const h = harness(frag('e1', 1, 'done', 1));
      h.typing = true;
      h.refreshBody(); await flush();
      await h.runPolls();
      const typing = { fetches: h.list.length, polls: h.polls.length, current: h.current.name };
      h.typing = false; await h.runPolls();
      await h.body('HEARD@e1/1');
      const heard = h.current.name;
      h.refreshBody(); await flush();
      h.typing = true;
      await h.body('READ@e1/1');
      const read = { current: h.current.name, polls: h.polls.length };
      h.typing = false; await h.runPolls();
      await h.body('LATER@e1/1');
      console.log(JSON.stringify({ typing, heard, read, polls: h.polls.length, ...h.out() }));
    """)
    assert out["typing"] == {"fetches": 0, "polls": 1, "current": "PAGE"}
    assert out["heard"] == "HEARD"
    assert out["read"] == {"current": "HEARD", "polls": 1}
    assert out["replaced"] == ["HEARD", "LATER"]
    assert (out["polls"], out["fetches"]) == (0, 0)


@needs_node
def test_a_failed_hearing_refresh_is_read_again_until_a_body_is_accepted():
    out = _node("""
      const h = harness(frag('e1', 1, 'done', 1));
      h.refreshBody(); await flush();
      await h.down();
      const down = h.polls.length;
      await h.runPolls();
      await h.body('BUSY@e1/1', false);
      const busy = h.polls.length;
      await h.runPolls();
      await h.body('HEARD@e1/1');
      console.log(JSON.stringify({ down, busy, polls: h.polls.length, ...h.out() }));
    """)
    assert (out["down"], out["busy"]) == (1, 1)
    assert out["replaced"] == ["HEARD"]
    assert (out["polls"], out["fetches"]) == (0, 0)


def test_the_script_wires_the_hearing_and_its_poller_follows_t36():
    source = _script()
    assert "const HEARING_URL = '/voices/hearing';" in source
    for gone in ("BAKE_URL", "voice-bake", "startBake", "cancelBake", "'model'"):
        assert gone not in source, gone
    poll = _js_function(source, "function pollHearing")
    assert poll.index("const own = mounted;") < poll.index("await readHearing(own)")
    assert "if (own.alive() && hearingBusyHere()) { pollHearing(); }" in poll
    read = _js_function(source, "async function readHearing")
    assert "own.fetch(`${HEARING_URL}?voice=${own.page.dataset.voiceId}`," in read
    busy = source[source.index("const hearingBusyHere"):]
    assert "(job.state === 'running' || job.state === 'queued')" in busy[:busy.index("};")]
    assert read.index("await response.text()") < read.index("if (own.alive()) { applyHearing(own, html); }")
    for head in ("async function postHearing", "async function refreshBody"):
        body = _js_function(source, head)
        first_await = body.index("await ")
        assert "own" in body[:first_await], head
        assert "own.alive()" in body[first_await:], head
    click = source[source.index("document.addEventListener('click'"):]
    assert "postHearing(`/voices/${hear.dataset.voiceId}/hear`)" in click
    assert "event.target.closest('.voice-hearing-cancel, .voice-hearing-dismiss')" in click
    assert "postHearing(`/voices/hearing/cancel?voice=${pageNode().dataset.voiceId}`);" in click
    assert "reconnectHearing(own);" in _js_function(source, "function mountLeaf")
    swap = _js_function(source, "function replaceList")
    assert "if (hearingBusyHere()) { pollHearing(); }" in swap
    assert "#voice-cards [data-deleting]" in _js_function(source, "function pollCards")
