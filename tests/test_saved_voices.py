"""Saved voices (design §6, §11.1): the store, the WAV check, the write-ahead file
writes and the deletion outbox (`feed/saved_voices.py`, PLAN §2.6-§2.7, D21, D22, D33)."""

import dataclasses
import hashlib
import importlib
import inspect
import io
import json
import math
import re
import sqlite3
import struct
import threading
import wave

import pytest

from db.connection import _ADDED_COLUMNS, SCHEMA_VERSION, connect, migrate
from feed import saved_voices as sv
from feed.saved_voices import (
    RECORDING_PROMPTS,
    NarratorChoice,
    VoiceDeleting,
    VoiceError,
    VoiceNameTaken,
    add_example,
    conditionings,
    create_voice,
    example_for,
    get_voice,
    list_voices,
    pending_voice_deletions,
    process_voice_deletions,
    read_example_audio,
    read_narrator,
    rename_voice,
    request_example_deletion,
    request_voice_deletion,
    set_narrator,
    token,
)
from models import catalog
from storage.keys import content_key
from storage.local import LocalStorage

OWNER = 1


def make_wav(seconds=6.0, *, rate=48000, channels=1, width=2, freq=220.0, amp=0.2) -> bytes:
    frames = round(seconds * rate)
    peak = (1 << (8 * width - 1)) - 1
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(width)
        handle.setframerate(rate)
        out = bytearray()
        for i in range(frames):
            value = int(amp * peak * math.sin(2 * math.pi * freq * i / rate))
            if width == 1:
                sample = bytes([value + 128])  # 8-bit WAV is unsigned
            else:
                sample = value.to_bytes(width, "little", signed=True)
            out += sample * channels
        handle.writeframes(bytes(out))
    return buffer.getvalue()


def hand_header(format_tag: int, *, width=4, rate=48000, frames=48000) -> bytes:
    """A WAV whose `fmt ` chunk names a non-PCM format (3 = float, 2 = ADPCM)."""
    data = b"\x00" * (frames * width)
    fmt = struct.pack("<HHIIHH", format_tag, 1, rate, rate * width, width, width * 8)
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt
    body += b"data" + struct.pack("<I", len(data)) + data
    return b"RIFF" + struct.pack("<I", len(body)) + body


CLIP = make_wav(6.0)


@pytest.fixture
def store(tmp_path):
    return LocalStorage(tmp_path / "voices")


def add(conn, store, voice_id, data=CLIP, *, language="en", source="uploaded", owner=OWNER):
    return add_example(conn, store, owner, voice_id, language=language, source=source,
                       data=data)


def make_group(conn, owner=OWNER) -> int:
    return conn.execute(
        "INSERT INTO groups(owner_id, kind, name, status, created_at)"
        " VALUES (?, 'memory', 'Beach', 'ready', '2026-01-01')",
        (owner,),
    ).lastrowid


def rows(conn, table) -> int:
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def seed_conditioning(conn, store, example_id, blob) -> str:
    """A legacy `voice_conditionings` row and its file, as the one-take films left them."""
    key = content_key(hashlib.sha256(blob).hexdigest(), ".cond")
    store.write(key, blob)
    conn.execute(
        "INSERT INTO voice_conditionings(example_id, model_key, revision, storage_key, created_at)"
        " VALUES (?, 'voxcpm2', 'rev1', ?, '2026-01-01')",
        (example_id, key),
    )
    return key


class CountingStorage:
    """A LocalStorage that counts writes and deletes, and can fail on demand."""

    def __init__(self, inner, *, fail_delete=(), delete_then_raise=(), on_write=None):
        self.inner = inner
        self.writes = 0
        self.deletes: list[str] = []
        self.fail_delete = set(fail_delete)
        self.delete_then_raise = set(delete_then_raise)
        self.on_write = on_write

    def write(self, key, data):
        self.writes += 1
        self.inner.write(key, data)
        if self.on_write is not None:
            self.on_write()

    def delete(self, key):
        if key in self.fail_delete:
            self.fail_delete.discard(key)
            raise OSError("disk says no")
        self.inner.delete(key)
        self.deletes.append(key)
        if key in self.delete_then_raise:
            self.delete_then_raise.discard(key)
            raise OSError("died right after the unlink")

    def __getattr__(self, name):
        return getattr(self.inner, name)


def test_tables_exist_after_migrate(conn):
    assert SCHEMA_VERSION == 15
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 15
    names = {r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"saved_voices", "voice_examples", "voice_conditionings", "memory_narrators",
            "voice_deletions", "voice_file_deletions"} <= names


def test_create_and_rename(conn):
    voice = create_voice(conn, OWNER, "  Grandma  ")
    assert voice.name == "Grandma" and voice.examples == () and not voice.deleting
    renamed = rename_voice(conn, OWNER, voice.id, "Nana")
    assert renamed.name == "Nana"
    assert rename_voice(conn, OWNER, voice.id, "NANA").name == "NANA"  # its own name, recased
    assert [v.name for v in list_voices(conn, OWNER)] == ["NANA"]


@pytest.mark.parametrize("first, second", [("Grandma", "grandma"), ("Бабушка", "БАБУШКА")])
def test_duplicate_name_by_casefold(conn, first, second):
    voice = create_voice(conn, OWNER, first)
    other = create_voice(conn, OWNER, "Other")
    with pytest.raises(VoiceNameTaken):
        create_voice(conn, OWNER, second)
    with pytest.raises(VoiceNameTaken):
        rename_voice(conn, OWNER, other.id, second)
    assert get_voice(conn, OWNER, voice.id).name == first
    assert not conn.in_transaction


@pytest.mark.parametrize("name", ["", "   ", "x" * (sv.NAME_MAX + 1)])
def test_blank_or_too_long_name(conn, name):
    with pytest.raises(VoiceError) as raised:
        create_voice(conn, OWNER, name)
    assert not isinstance(raised.value, VoiceNameTaken)
    assert create_voice(conn, OWNER, "x" * sv.NAME_MAX).name == "x" * sv.NAME_MAX


def test_owner_scoping(conn, store):
    mine = create_voice(conn, OWNER, "Grandma")
    theirs = create_voice(conn, 2, "Grandma")  # the same name is fine for another owner
    add(conn, store, mine.id)
    assert [v.id for v in list_voices(conn, OWNER)] == [mine.id]
    assert get_voice(conn, OWNER, theirs.id) is None
    with pytest.raises(VoiceDeleting):
        rename_voice(conn, OWNER, theirs.id, "Mine now")
    with pytest.raises(VoiceDeleting):
        add(conn, store, theirs.id)
    assert example_for(conn, 2, mine.id, "en") is None
    assert request_voice_deletion(conn, OWNER, theirs.id) is False
    example = get_voice(conn, OWNER, mine.id).examples[0]
    assert request_example_deletion(conn, 2, mine.id, example.id) is False
    assert rows(conn, "voice_deletions") == 0


def test_token():
    assert token(7) == "saved:7"


def test_a_stored_take_holds_no_words_until_it_is_heard(conn, store):
    voice = create_voice(conn, OWNER, "Grandma")
    recorded = add(conn, store, voice.id, language="ru", source="recorded")
    uploaded = add(conn, store, voice.id, make_wav(7.0), language="ru", source="uploaded")
    assert recorded.transcript is None and uploaded.transcript is None
    assert recorded.sample_rate == 48000 and recorded.seconds == pytest.approx(6.0)
    stored = conn.execute("SELECT consent_at, transcript FROM voice_examples WHERE id = ?",
                          (recorded.id,)).fetchone()
    assert stored["consent_at"] and stored["transcript"] is None


def test_stored_file_is_the_posted_bytes(conn, store):
    voice = create_voice(conn, OWNER, "Grandma")
    example = add(conn, store, voice.id)
    assert example.content_hash == hashlib.sha256(CLIP).hexdigest()
    assert store.read(example.storage_key) == CLIP
    assert example.storage_key.endswith(f"{example.content_hash}.wav")
    assert read_example_audio(store, example) == CLIP
    assert rows(conn, "voice_file_deletions") == 0  # the write-ahead row is gone


@pytest.mark.parametrize("seconds", [5.0, 30.0])
def test_length_bounds_accepted(conn, store, seconds):
    voice = create_voice(conn, OWNER, "Grandma")
    assert add(conn, store, voice.id, make_wav(seconds)).seconds == pytest.approx(seconds)


@pytest.mark.parametrize("seconds, text", [(4.99, "at least"), (30.01, "at most")])
def test_length_bounds_refused(conn, store, seconds, text):
    voice = create_voice(conn, OWNER, "Grandma")
    with pytest.raises(VoiceError, match=text):
        add(conn, store, voice.id, make_wav(seconds))


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
def test_wrong_format_names_the_format_found(conn, store, data, text):
    voice = create_voice(conn, OWNER, "Grandma")
    with pytest.raises(VoiceError, match="must be mono 16-bit PCM at 48000 Hz") as raised:
        add(conn, store, voice.id, data)
    assert text in str(raised.value)
    assert rows(conn, "voice_examples") == 0 and rows(conn, "voice_file_deletions") == 0


@pytest.mark.parametrize(
    "data",
    [
        hand_header(3),          # IEEE float
        hand_header(2, width=2), # ADPCM (compressed)
        b"this is not audio at all, just some garbage bytes",
        b"",
    ],
    ids=["float", "compressed", "garbage", "empty"],
)
def test_not_a_pcm_wav(conn, store, data):
    voice = create_voice(conn, OWNER, "Grandma")
    with pytest.raises(VoiceError):
        add(conn, store, voice.id, data)
    assert rows(conn, "voice_examples") == 0


def test_zero_frames(conn, store):
    voice = create_voice(conn, OWNER, "Grandma")
    with pytest.raises(VoiceError, match="no audio"):
        add(conn, store, voice.id, make_wav(0.0))


def test_truncated_data_chunk(conn, store):
    voice = create_voice(conn, OWNER, "Grandma")
    with pytest.raises(VoiceError, match="cut short"):
        add(conn, store, voice.id, make_wav(6.0)[:-1000])


def test_silent_clip(conn, store):
    voice = create_voice(conn, OWNER, "Grandma")
    with pytest.raises(VoiceError, match="silent"):
        add(conn, store, voice.id, make_wav(6.0, amp=0.0))


def test_other_add_example_checks(conn, store, monkeypatch):
    voice = create_voice(conn, OWNER, "Grandma")
    with pytest.raises(VoiceError, match="language"):
        add(conn, store, voice.id, language="de")
    with pytest.raises(VoiceError, match="source"):
        add(conn, store, voice.id, source="cloned")
    monkeypatch.setattr(sv, "MAX_BYTES", len(CLIP) - 1)
    with pytest.raises(VoiceError, match="larger"):
        add(conn, store, voice.id)
    monkeypatch.undo()
    add(conn, store, voice.id)
    with pytest.raises(VoiceError, match="already has"):
        add(conn, store, voice.id)
    assert rows(conn, "voice_examples") == 1


def test_example_for_is_the_newest_in_the_language(conn, store):
    voice = create_voice(conn, OWNER, "Grandma")
    add(conn, store, voice.id, make_wav(6.0), language="en")
    newer = add(conn, store, voice.id, make_wav(7.0), language="en")
    russian = add(conn, store, voice.id, make_wav(8.0), language="ru")
    assert example_for(conn, OWNER, voice.id, "en").id == newer.id
    assert example_for(conn, OWNER, voice.id, "ru").id == russian.id


def test_narrator_set_read_clear(conn):
    group = make_group(conn)
    voice = create_voice(conn, OWNER, "Grandma")
    assert read_narrator(conn, group, "en") is None
    set_narrator(conn, group, "en", voice.id)
    assert read_narrator(conn, group, "en") == NarratorChoice(voice.id)
    assert read_narrator(conn, group, "ru") is None
    set_narrator(conn, group, "en", None)
    assert read_narrator(conn, group, "en") is None


def test_narrator_of_another_owner_or_pending_is_refused(conn):
    group = make_group(conn)
    foreign = create_voice(conn, 2, "Theirs")
    with pytest.raises(VoiceDeleting):
        set_narrator(conn, group, "en", foreign.id)
    voice = create_voice(conn, OWNER, "Grandma")
    request_voice_deletion(conn, OWNER, voice.id)
    with pytest.raises(VoiceDeleting):
        set_narrator(conn, group, "en", voice.id)
    assert read_narrator(conn, group, "en") is None and not conn.in_transaction


def test_group_delete_cascades_the_narrator(conn):
    group = make_group(conn)
    voice = create_voice(conn, OWNER, "Grandma")
    set_narrator(conn, group, "en", voice.id)
    conn.execute("DELETE FROM groups WHERE id = ?", (group,))
    assert rows(conn, "memory_narrators") == 0


def test_requests_only_enqueue(conn, store):
    voice = create_voice(conn, OWNER, "Grandma")
    example = add(conn, store, voice.id)
    other = add(conn, store, voice.id, make_wav(7.0))
    assert request_example_deletion(conn, OWNER, voice.id, example.id) is True
    assert request_example_deletion(conn, OWNER, voice.id, example.id) is True
    assert rows(conn, "voice_deletions") == 1
    assert request_voice_deletion(conn, OWNER, voice.id) is True
    assert request_voice_deletion(conn, OWNER, voice.id) is True
    assert rows(conn, "voice_deletions") == 2
    # An example of a pending voice is covered already: True, nothing inserted.
    assert request_example_deletion(conn, OWNER, voice.id, other.id) is True
    assert rows(conn, "voice_deletions") == 2
    assert rows(conn, "saved_voices") == 1 and rows(conn, "voice_examples") == 2
    assert store.exists(example.storage_key) and store.exists(other.storage_key)


def test_unknown_targets_are_refused(conn, store):
    voice = create_voice(conn, OWNER, "Grandma")
    example = add(conn, store, voice.id)
    second = create_voice(conn, OWNER, "Other")
    assert request_voice_deletion(conn, OWNER, 999) is False
    assert request_example_deletion(conn, OWNER, voice.id, 999) is False
    assert request_example_deletion(conn, OWNER, second.id, example.id) is False  # wrong voice
    assert rows(conn, "voice_deletions") == 0 and not conn.in_transaction


def test_pending_voice_rejects_mutations_and_casts(conn, store):
    voice = create_voice(conn, OWNER, "Grandma")
    add(conn, store, voice.id)
    request_voice_deletion(conn, OWNER, voice.id)
    listed = list_voices(conn, OWNER)
    assert listed[0].deleting and all(e.deleting for e in listed[0].examples)
    with pytest.raises(VoiceDeleting):
        rename_voice(conn, OWNER, voice.id, "New")
    with pytest.raises(VoiceDeleting):
        add(conn, store, voice.id, make_wav(7.0))
    with pytest.raises(VoiceDeleting):
        create_voice(conn, OWNER, "GRANDMA")
    assert example_for(conn, OWNER, voice.id, "en") is None


def test_pending_example_falls_back_to_an_older_one(conn, store):
    voice = create_voice(conn, OWNER, "Grandma")
    older = add(conn, store, voice.id, make_wav(6.0))
    newer = add(conn, store, voice.id, make_wav(7.0))
    request_example_deletion(conn, OWNER, voice.id, newer.id)
    examples = {e.id: e.deleting for e in get_voice(conn, OWNER, voice.id).examples}
    assert examples == {older.id: False, newer.id: True}
    assert not get_voice(conn, OWNER, voice.id).deleting
    assert example_for(conn, OWNER, voice.id, "en").id == older.id
    with pytest.raises(VoiceDeleting):
        add(conn, store, voice.id, make_wav(7.0))  # the same bytes as the pending one


def test_voice_cascade_removes_rows_and_files(conn, store):
    group = make_group(conn)
    voice = create_voice(conn, OWNER, "Grandma")
    example = add(conn, store, voice.id)
    blob_key = seed_conditioning(conn, store, example.id, b"blob")
    set_narrator(conn, group, "en", voice.id)
    request_voice_deletion(conn, OWNER, voice.id)
    assert process_voice_deletions(conn, store, OWNER) == 0
    for table in ("saved_voices", "voice_examples", "voice_conditionings", "voice_deletions",
                  "voice_file_deletions"):
        assert rows(conn, table) == 0, table
    assert read_narrator(conn, group, "en") == NarratorChoice(None)
    assert not store.exists(example.storage_key) and not store.exists(blob_key)
    assert get_voice(conn, OWNER, voice.id) is None


def test_example_cascade_removes_only_that_example(conn, store):
    voice = create_voice(conn, OWNER, "Grandma")
    gone = add(conn, store, voice.id, make_wav(6.0))
    kept = add(conn, store, voice.id, make_wav(7.0))
    seed_conditioning(conn, store, gone.id, b"gone-blob")
    seed_conditioning(conn, store, kept.id, b"kept-blob")
    request_example_deletion(conn, OWNER, voice.id, gone.id)
    assert process_voice_deletions(conn, store, OWNER) == 0
    assert [e.id for e in get_voice(conn, OWNER, voice.id).examples] == [kept.id]
    assert not store.exists(gone.storage_key) and store.exists(kept.storage_key)
    assert conditionings(conn, store, kept.id)[0]["blob"] == b"kept-blob"
    assert rows(conn, "voice_conditionings") == 1


def test_a_wav_shared_with_another_voice_stays(conn, store):
    first = create_voice(conn, OWNER, "Ann")
    second = create_voice(conn, OWNER, "Bob")
    example = add(conn, store, first.id)
    counting = CountingStorage(store)
    add(conn, counting, second.id)  # the same bytes → the same key
    assert counting.writes == 0     # a valid shared key is not rewritten
    request_voice_deletion(conn, OWNER, first.id)
    assert process_voice_deletions(conn, store, OWNER) == 0
    assert store.read(example.storage_key) == CLIP
    assert rows(conn, "voice_file_deletions") == 0


def test_a_clip_added_again_before_the_file_step_stays(conn, store):
    voice = create_voice(conn, OWNER, "Ann")
    example = add(conn, store, voice.id)
    request_voice_deletion(conn, OWNER, voice.id)
    sv._cascade_intents(conn, OWNER)
    assert rows(conn, "voice_file_deletions") == 1 and store.exists(example.storage_key)
    again = create_voice(conn, OWNER, "Ann again")
    add(conn, store, again.id)
    assert process_voice_deletions(conn, store, OWNER) == 0
    assert store.read(example.storage_key) == CLIP
    assert rows(conn, "voice_file_deletions") == 0 and rows(conn, "voice_deletions") == 0


def test_crash_before_unlink_is_retried(conn, store):
    voice = create_voice(conn, OWNER, "Ann")
    example = add(conn, store, voice.id)
    request_voice_deletion(conn, OWNER, voice.id)
    failing = CountingStorage(store, fail_delete={example.storage_key})
    assert process_voice_deletions(conn, failing, OWNER) > 0
    assert rows(conn, "saved_voices") == 0 and rows(conn, "voice_examples") == 0
    assert rows(conn, "voice_file_deletions") == 1 and rows(conn, "voice_deletions") == 1
    assert store.exists(example.storage_key)
    assert process_voice_deletions(conn, failing, OWNER) == 0
    assert not store.exists(example.storage_key) and rows(conn, "voice_deletions") == 0


def test_crash_after_unlink_is_acknowledged_next_run(conn, store):
    voice = create_voice(conn, OWNER, "Ann")
    example = add(conn, store, voice.id)
    request_voice_deletion(conn, OWNER, voice.id)
    dying = CountingStorage(store, delete_then_raise={example.storage_key})
    assert process_voice_deletions(conn, dying, OWNER) > 0
    assert not store.exists(example.storage_key)
    assert rows(conn, "voice_file_deletions") == 1
    assert process_voice_deletions(conn, dying, OWNER) == 0
    assert rows(conn, "voice_deletions") == 0 and rows(conn, "voice_file_deletions") == 0


def test_one_bad_key_never_stops_the_pass(conn, store):
    voices = [create_voice(conn, OWNER, f"V{i}") for i in range(3)]
    examples = [add(conn, store, v.id, make_wav(6.0 + i)) for i, v in enumerate(voices)]
    for voice in voices:
        request_voice_deletion(conn, OWNER, voice.id)
    bad = examples[1].storage_key
    failing = CountingStorage(store, fail_delete={bad})
    assert process_voice_deletions(conn, failing, OWNER) == 2  # the bad key + its intent
    assert store.exists(bad)
    assert not store.exists(examples[0].storage_key) and not store.exists(examples[2].storage_key)
    assert process_voice_deletions(conn, failing, OWNER) == 0


def test_two_drainers_remove_every_key_once(conn, store, settings):
    voices = [create_voice(conn, OWNER, f"V{i}") for i in range(6)]
    for i, voice in enumerate(voices):
        add(conn, store, voice.id, make_wav(5.0 + i))
        request_voice_deletion(conn, OWNER, voice.id)
    counting = CountingStorage(store)
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def drain():
        own = connect(settings.db_path)
        try:
            barrier.wait(timeout=10)
            process_voice_deletions(own, counting, OWNER)
        except BaseException as exc:  # noqa: BLE001 — collected for the assert
            errors.append(exc)
        finally:
            own.close()

    threads = [threading.Thread(target=drain) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert sorted(counting.deletes) == sorted(set(counting.deletes)) and len(counting.deletes) == 6
    assert pending_voice_deletions(conn) == 0
    assert list(store.iter_keys()) == []


def test_a_failed_insert_rolls_back_and_the_drain_removes_the_file(conn, store):
    voice = create_voice(conn, OWNER, "Ann")
    conn.execute(
        "CREATE TEMP TRIGGER boom BEFORE INSERT ON main.voice_examples"
        " BEGIN SELECT RAISE(ABORT, 'boom'); END"
    )
    with pytest.raises(sqlite3.IntegrityError):
        add(conn, store, voice.id)
    key = conn.execute("SELECT storage_key, deletion_id FROM voice_file_deletions").fetchone()
    assert key["deletion_id"] is None and store.exists(key["storage_key"])
    assert rows(conn, "voice_examples") == 0 and not conn.in_transaction
    assert process_voice_deletions(conn, store, OWNER) == 0
    assert not store.exists(key["storage_key"])


def test_process_death_mid_write_leaves_the_write_ahead_row(conn, store, settings):
    voice = create_voice(conn, OWNER, "Ann")
    writer = connect(settings.db_path)
    dying = CountingStorage(store, on_write=writer.close)
    with pytest.raises(sqlite3.ProgrammingError):
        add(writer, dying, voice.id)
    key = conn.execute("SELECT storage_key, deletion_id FROM voice_file_deletions").fetchone()
    assert key["deletion_id"] is None and store.exists(key["storage_key"])
    assert rows(conn, "voice_examples") == 0
    assert process_voice_deletions(conn, store, OWNER) == 0
    assert not store.exists(key["storage_key"])


def test_a_drain_between_write_ahead_and_lock_is_survived(conn, store, settings, monkeypatch):
    voice = create_voice(conn, OWNER, "Ann")
    original = sv._write_ahead
    calls = []

    def write_ahead_then_drain(c, key):
        row_id = original(c, key)
        calls.append(row_id)
        if len(calls) == 1:
            other = connect(settings.db_path)
            try:
                process_voice_deletions(other, store, OWNER)
            finally:
                other.close()
        return row_id

    monkeypatch.setattr(sv, "_write_ahead", write_ahead_then_drain)
    example = add(conn, store, voice.id)
    assert len(calls) == 2  # the first row was acknowledged; a second one was committed
    assert store.read(example.storage_key) == CLIP
    assert rows(conn, "voice_examples") == 1 and rows(conn, "voice_file_deletions") == 0


def use_plain_integer_pk(conn):
    """The table as a DB migrated before AUTOINCREMENT has it: `CREATE TABLE IF NOT
    EXISTS` keeps it, and SQLite reuses the highest id once its row is deleted."""
    conn.execute("DROP TABLE voice_file_deletions")
    conn.execute(
        "CREATE TABLE voice_file_deletions (id INTEGER PRIMARY KEY, storage_key TEXT NOT NULL,"
        " deletion_id INTEGER REFERENCES voice_deletions(id), requested_at TEXT NOT NULL,"
        " UNIQUE(storage_key, deletion_id))"
    )


class DiesAfterWrite(Exception):
    pass


@pytest.mark.parametrize("schema", ["plain", "autoincrement"])
def test_a_reused_write_ahead_id_never_acks_another_writers_row(
    conn, store, settings, monkeypatch, schema
):
    """A's write-ahead row is drained; B's row takes the freed id, B writes its file and
    dies before publication. A must retry, never take B's row for its own."""
    if schema == "plain":
        use_plain_integer_pk(conn)
    voice = create_voice(conn, OWNER, "Ann")

    def write(c, storage, n):
        return add(c, storage, voice.id, make_wav(7.0 + n))

    original = sv._write_ahead
    ahead: list[tuple[str, str, int]] = []

    def write_ahead_then_race(c, key):
        row_id = original(c, key)
        first = not ahead
        ahead.append(("A" if c is conn else "B", key, row_id))
        if first:
            other = connect(settings.db_path)
            try:
                process_voice_deletions(other, store, OWNER)  # acknowledges A's row
                with pytest.raises(DiesAfterWrite):
                    write(other, CountingStorage(store, on_write=dies), 2)
            finally:
                other.close()
        return row_id

    def dies():
        raise DiesAfterWrite

    monkeypatch.setattr(sv, "_write_ahead", write_ahead_then_race)
    a_result = write(conn, store, 1)
    (_, a_key, a_first), (_, b_key, b_id), (who, retry_key, _) = ahead
    assert (b_id == a_first) is (schema == "plain")  # the collision this test is about
    assert (who, retry_key) == ("A", a_key)  # A retried with a fresh write-ahead row
    assert [tuple(r) for r in conn.execute("SELECT storage_key FROM voice_file_deletions")] == [
        (b_key,)
    ]  # B keeps its cleanup row
    assert store.exists(b_key)
    assert process_voice_deletions(conn, store, OWNER) == 0
    assert not store.exists(b_key) and store.exists(a_key)
    assert store.read(a_result.storage_key) == make_wav(8.0)


def test_voice_ids_are_never_reused(conn, store):
    create_voice(conn, OWNER, "Ann")
    newest = create_voice(conn, OWNER, "Bob")
    request_voice_deletion(conn, OWNER, newest.id)
    assert process_voice_deletions(conn, store, OWNER) == 0
    fresh = create_voice(conn, OWNER, "Cid")
    assert fresh.id > newest.id
    assert request_voice_deletion(conn, OWNER, fresh.id) is True
    assert rows(conn, "voice_deletions") == 1  # its own intent, not swallowed


def test_example_ids_are_never_reused(conn, store):
    voice = create_voice(conn, OWNER, "Ann")
    add(conn, store, voice.id, make_wav(6.0))
    newest = add(conn, store, voice.id, make_wav(7.0))
    request_example_deletion(conn, OWNER, voice.id, newest.id)
    assert process_voice_deletions(conn, store, OWNER) == 0
    fresh = add(conn, store, voice.id, make_wav(8.0))
    assert fresh.id > newest.id
    assert request_example_deletion(conn, OWNER, voice.id, fresh.id) is True
    assert rows(conn, "voice_deletions") == 1


def test_file_deletion_ids_are_never_reused(conn, store):
    first = sv._write_ahead(conn, "aa/first.wav")
    assert process_voice_deletions(conn, store, OWNER) == 0
    assert rows(conn, "voice_file_deletions") == 0
    assert sv._write_ahead(conn, "bb/second.wav") > first


def test_each_language_starts_with_its_four_reading_texts_with_ids_that_name_it():
    assert {language: list(prompts)[:4] for language, prompts in RECORDING_PROMPTS.items()} == {
        "en": ["en-kitchen", "en-market", "en-rain", "en-train"],
        "ru": ["ru-kitchen", "ru-market", "ru-rain", "ru-train"],
    }


def test_a_recording_keeps_its_prompt_id_and_never_its_text_as_words(conn, store):
    voice = create_voice(conn, OWNER, "Grandma")
    example = add_example(conn, store, OWNER, voice.id, language="en", source="recorded",
                          data=CLIP, prompt_id="en-rain")
    assert (example.kit_id, example.transcript) == ("en-rain", None)
    stored = conn.execute("SELECT kit_id, transcript FROM voice_examples WHERE id = ?", (example.id,)).fetchone()
    assert (stored["kit_id"], stored["transcript"]) == ("en-rain", None)


@pytest.mark.parametrize("prompt_id", ["en-nowhere", "ru-rain", ""])
def test_an_unknown_or_other_language_prompt_id_is_refused(conn, store, prompt_id):
    voice = create_voice(conn, OWNER, "Grandma")
    with pytest.raises(VoiceError, match=f"Unknown reading text for English: '{prompt_id}'"):
        add_example(conn, store, OWNER, voice.id, language="en", source="recorded",
                    data=CLIP, prompt_id=prompt_id)
    assert rows(conn, "voice_examples") == 0


def test_the_prompt_id_is_checked_before_the_clip(conn, store):
    voice = create_voice(conn, OWNER, "Grandma")
    with pytest.raises(VoiceError, match="reading text"):
        add_example(conn, store, OWNER, voice.id, language="en", source="recorded",
                    data=b"not a wav", prompt_id="en-nowhere")


def test_an_upload_ignores_a_posted_prompt_id(conn, store):
    voice = create_voice(conn, OWNER, "Grandma")
    example = add_example(conn, store, OWNER, voice.id, language="en", source="uploaded",
                          data=CLIP, prompt_id="ru-rain")
    assert (example.kit_id, example.transcript) == (None, None)


OLD_NARRATORS_DDL = """CREATE TABLE memory_narrators (
  group_id INTEGER NOT NULL REFERENCES groups(id) ON DELETE CASCADE,
  language TEXT NOT NULL,
  voice_id INTEGER REFERENCES saved_voices(id) ON DELETE SET NULL,
  PRIMARY KEY (group_id, language)
)"""


@pytest.fixture
def old_table(tmp_path):
    """A DB at the current version whose `memory_narrators` predates `preset_speaker`,
    holding one row; then migrated, as the user's DB is on the next start."""
    db = connect(tmp_path / "old.db")
    migrate(db)
    db.execute("DROP TABLE memory_narrators")
    db.execute(OLD_NARRATORS_DDL)
    group = make_group(db)
    voice = create_voice(db, OWNER, "Grandma")
    db.execute("INSERT INTO memory_narrators(group_id, language, voice_id) VALUES (?, 'en', ?)",
               (group, voice.id))
    migrate(db)
    yield db, group, voice.id
    db.close()


def both_set(conn) -> int:
    return conn.execute(
        "SELECT COUNT(*) FROM memory_narrators WHERE voice_id IS NOT NULL AND preset_speaker IS NOT NULL"
    ).fetchone()[0]


def test_a_fresh_table_refuses_a_row_with_both_narrators(conn):
    group = make_group(conn)
    voice = create_voice(conn, OWNER, "Grandma")
    columns = {r["name"] for r in conn.execute("PRAGMA table_info(memory_narrators)")}
    assert "preset_speaker" in columns
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO memory_narrators(group_id, language, voice_id, preset_speaker)"
                     " VALUES (?, 'en', ?, 'af_heart')", (group, voice.id))


def test_migrate_adds_the_preset_column_to_an_old_table_and_keeps_its_row(old_table):
    db, group, voice_id = old_table
    columns = {r["name"] for r in db.execute("PRAGMA table_info(memory_narrators)")}
    assert "preset_speaker" in columns
    assert read_narrator(db, group, "en") == NarratorChoice(voice_id, None)


def test_a_preset_narrator_round_trips_and_replaces_a_saved_one(conn):
    group = make_group(conn)
    voice = create_voice(conn, OWNER, "Grandma")
    set_narrator(conn, group, "en", preset_speaker="af_heart")
    assert read_narrator(conn, group, "en") == NarratorChoice(None, "af_heart")
    set_narrator(conn, group, "en", voice.id)
    assert read_narrator(conn, group, "en") == NarratorChoice(voice.id)
    set_narrator(conn, group, "en", preset_speaker="af_bella")
    assert read_narrator(conn, group, "en") == NarratorChoice(None, "af_bella")
    set_narrator(conn, group, "en", None)
    assert read_narrator(conn, group, "en") is None


def test_a_saved_and_a_preset_narrator_at_once_is_refused(conn):
    group = make_group(conn)
    voice = create_voice(conn, OWNER, "Grandma")
    set_narrator(conn, group, "en", preset_speaker="af_heart")
    with pytest.raises(VoiceError, match="not both"):
        set_narrator(conn, group, "en", voice.id, preset_speaker="af_bella")
    assert read_narrator(conn, group, "en") == NarratorChoice(None, "af_heart")
    assert not conn.in_transaction


@pytest.mark.parametrize("blank", ["", "   "])
def test_a_blank_preset_narrator_is_refused(conn, blank):
    group = make_group(conn)
    with pytest.raises(VoiceError):
        set_narrator(conn, group, "en", preset_speaker=blank)
    assert read_narrator(conn, group, "en") is None


def test_the_old_table_never_holds_both_narrators(old_table):
    db, group, voice_id = old_table
    set_narrator(db, group, "en", preset_speaker="af_heart")
    assert read_narrator(db, group, "en") == NarratorChoice(None, "af_heart")
    set_narrator(db, group, "en", voice_id)
    assert read_narrator(db, group, "en") == NarratorChoice(voice_id)
    with pytest.raises(VoiceError):
        set_narrator(db, group, "en", voice_id, preset_speaker="af_bella")
    assert both_set(db) == 0


def test_draining_a_saved_voice_leaves_a_preset_narrator_alone(conn, store):
    by_voice, by_preset = make_group(conn), make_group(conn)
    voice = create_voice(conn, OWNER, "Grandma")
    add(conn, store, voice.id)
    set_narrator(conn, by_voice, "en", voice.id)
    set_narrator(conn, by_preset, "en", preset_speaker="af_heart")
    request_voice_deletion(conn, OWNER, voice.id)
    assert process_voice_deletions(conn, store, OWNER) == 0
    assert read_narrator(conn, by_voice, "en") == NarratorChoice(None, None)
    assert read_narrator(conn, by_preset, "en") == NarratorChoice(None, "af_heart")


OTHER_OWNER = 2
TEST_AUDIO = make_wav(2.0, freq=330.0)
TEST_SECONDS = 2.0  # 96000 frames at 48 kHz


def film_rows(conn) -> list[tuple]:
    return [tuple(r) for r in conn.execute(
        "SELECT voice_id, language, example_id FROM voice_film_examples ORDER BY voice_id, language"
    )]


def seed_pin(conn, voice_id, example_id) -> None:
    conn.execute(
        "INSERT INTO voice_film_examples(voice_id, language, example_id, chosen_at)"
        " VALUES (?, 'en', ?, '2026-01-01')",
        (voice_id, example_id),
    )


def seed_preview(conn, store, example_id, audio=TEST_AUDIO) -> str:
    key = content_key(hashlib.sha256(audio).hexdigest(), ".wav")
    store.write(key, audio)
    conn.execute(
        "INSERT INTO voice_previews(example_id, text_kind, text_id, text, model_key, revision,"
        " conditioning, elapsed_seconds, seconds, storage_key, created_at)"
        " VALUES (?, 'custom', NULL, 'Hello there.', 'voxcpm2', 'rev1', 'built', 3.5, ?, ?, '2026-01-01')",
        (example_id, TEST_SECONDS, key),
    )
    return key


def test_phase4_tables_exist_after_migrate(conn):
    assert SCHEMA_VERSION == 15
    names = {r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"voice_film_examples", "voice_previews"} <= names


def test_migrate_adds_the_phase4_tables_to_a_phase3_db_and_keeps_its_rows(tmp_path):
    db = connect(tmp_path / "phase3.db")
    migrate(db)
    db.execute("DROP TABLE voice_film_examples")
    db.execute("DROP TABLE voice_previews")
    voice = create_voice(db, OWNER, "Grandma")
    migrate(db)
    names = {r["name"] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"voice_film_examples", "voice_previews"} <= names
    assert [v.id for v in list_voices(db, OWNER)] == [voice.id]
    db.close()


@pytest.mark.parametrize("text_kind, conditioning", [("other", "built"), ("custom", "x")])
def test_a_preview_outside_its_checks_is_refused(conn, store, text_kind, conditioning):
    voice = create_voice(conn, OWNER, "Grandma")
    example = add(conn, store, voice.id)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO voice_previews(example_id, text_kind, text, model_key, conditioning,"
            " elapsed_seconds, seconds, storage_key, created_at)"
            " VALUES (?, ?, 'Hi', 'voxcpm2', ?, 1.0, 1.0, 'k.wav', '2026-01-01')",
            (example.id, text_kind, conditioning),
        )


def test_a_drained_pinned_example_leaves_a_tombstone_row(conn, store):
    voice = create_voice(conn, OWNER, "Grandma")
    chosen = add(conn, store, voice.id, make_wav(6.0))
    seed_pin(conn, voice.id, chosen.id)
    request_example_deletion(conn, OWNER, voice.id, chosen.id)
    assert process_voice_deletions(conn, store, OWNER) == 0
    assert film_rows(conn) == [(voice.id, "en", None)]


def test_a_drained_voice_leaves_no_film_choice(conn, store):
    voice = create_voice(conn, OWNER, "Grandma")
    example = add(conn, store, voice.id)
    seed_pin(conn, voice.id, example.id)
    request_voice_deletion(conn, OWNER, voice.id)
    assert process_voice_deletions(conn, store, OWNER) == 0
    assert film_rows(conn) == []


def test_reference_for_is_the_speak_reference_of_the_example(conn, store):
    voice = create_voice(conn, OWNER, "Grandma")
    example = add(conn, store, voice.id, source="recorded")
    seed_conditioning(conn, store, example.id, b"blob-1")
    assert sv.reference_for(conn, store, example) == {
        "audio": CLIP,
        "sha256": hashlib.sha256(CLIP).hexdigest(),
        "language": "en",
        "transcript": None,
        "conditioning": [{"model": "voxcpm2", "revision": "rev1", "blob": b"blob-1"}],
    }


@pytest.mark.parametrize("kind", ["voice", "example"])
def test_a_delete_removes_the_test_wav(conn, store, kind):
    voice = create_voice(conn, OWNER, "Grandma")
    example = add(conn, store, voice.id)
    key = seed_preview(conn, store, example.id)
    if kind == "voice":
        request_voice_deletion(conn, OWNER, voice.id)
    else:
        request_example_deletion(conn, OWNER, voice.id, example.id)
    assert process_voice_deletions(conn, store, OWNER) == 0
    assert rows(conn, "voice_previews") == 0
    assert not store.exists(key)


def test_a_test_wav_equal_to_another_examples_wav_stays(conn, store):
    holder = create_voice(conn, OWNER, "Ann")
    held = add(conn, store, holder.id)
    tested = create_voice(conn, OWNER, "Bob")
    example = add(conn, store, tested.id, make_wav(7.0))
    assert seed_preview(conn, store, example.id, CLIP) == held.storage_key
    request_example_deletion(conn, OWNER, tested.id, example.id)
    assert process_voice_deletions(conn, store, OWNER) == 0
    assert store.read(held.storage_key) == CLIP


def test_a_key_held_only_by_a_preview_is_kept(conn, store):
    voice = create_voice(conn, OWNER, "Grandma")
    example = add(conn, store, voice.id)
    key = seed_preview(conn, store, example.id)
    conn.execute(
        "INSERT INTO voice_file_deletions(storage_key, deletion_id, requested_at) VALUES (?, NULL, ?)",
        (key, "2026-01-01"),
    )
    assert process_voice_deletions(conn, store, OWNER) == 0
    assert store.read(key) == TEST_AUDIO


ENTRY = catalog.get("voice", "voxcpm2")
REVISION = ENTRY.source.revision
LORA_JSON = json.dumps(ENTRY.adapter.lora, sort_keys=True)
ADAPTER = b"adapter-safetensors-v1"
ADAPTER_2 = b"adapter-safetensors-v2"
WORDS = "The words I said in this take."
ASR = "whisper-large-v3-turbo@41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
VOICE_CHECK_WORDS = {"lake", "fog", "dog", "julia", "july", "cold", "road", "hills", "valley", "porch", "stars"}
IMPORTERS = ("feed.build", "feed.voice_preview", "feed.voice_hearing", "web.app", "web.voices_api")


def take(conn, store, voice_id, data=CLIP, *, language="en", words=WORDS, owner=OWNER):
    """An uploaded take whose words ASR heard (`heard`)."""
    added = add_example(conn, store, owner, voice_id, language=language, source="uploaded",
                        data=data)
    return sv.set_heard_words(conn, owner, voice_id, added.id, text=words, model=ASR)


def unusable_take(conn, store, voice_id, data, *, text="♪ ♪"):
    """An uploaded take whose heard words are not usable (`unusable`)."""
    added = add(conn, store, voice_id, data)
    return sv.set_heard_words(conn, OWNER, voice_id, added.id, text=text, model=ASR)


def fingerprint(content_hash, language, words) -> str:
    return hashlib.sha256(f"{content_hash}\t{language}\t{words}".encode()).hexdigest()


def done_status(dataset_hash, **changes) -> dict:
    """A `/voice/bakes` status at `done`, in the shape `modelsvc/voice_bakes.py` snapshots."""
    return {"id": "bake-1", "state": "done", "step": 4, "steps": 4, "loss_last": 1.25,
            "model": "voxcpm2", "revision": REVISION, "lora": LORA_JSON, "config": {"steps": 4},
            "dataset_sha256": dataset_hash, "clips": 1, "seconds": 6.0} | changes


def start(conn, voice_id) -> str:
    return sv.start_bake_run(conn, OWNER, voice_id, sv.bake_snapshot(conn, OWNER, voice_id))


def store_run(conn, store, voice_id, bake_key, data=ADAPTER, **changes):
    run = sv.read_bake_run(conn, OWNER, bake_key)
    return sv.save_voice_model(conn, store, OWNER, voice_id, bake_key=bake_key, data=data,
                               status=done_status(run.dataset_hash, **changes),
                               takes=run.takes, dataset_hash=run.dataset_hash)


def bake(conn, store, voice_id, data=ADAPTER):
    return store_run(conn, store, voice_id, start(conn, voice_id), data)


def columns(conn, table) -> set[str]:
    return {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}


def store_voice_test(conn, store, voice_id, prompt, audio=TEST_AUDIO, *, text_kind="custom",
                     example_id=None):
    return sv.save_test(
        conn, store, OWNER, voice_id, language="en", text_kind=text_kind,
        text_id="EN-1" if text_kind == "sentence" else None, example_id=example_id,
        text="Hello there.", prompt=prompt, model="voxcpm2", revision=REVISION,
        elapsed_seconds=3.5, audio=audio,
    )


def test_slice_b_tables_and_columns_exist_after_migrate(conn):
    assert SCHEMA_VERSION == 15
    names = {r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"voice_bake_runs", "voice_models", "voice_model_takes", "voice_tests"} <= names
    assert {"language", "seconds", "fingerprint"} <= columns(conn, "voice_model_takes")
    assert "cancel_requested_at" in columns(conn, "voice_bake_runs")
    assert {"transcript_confirmed_at", "kit_id", "transcript_model", "transcript_error"} <= columns(
        conn, "voice_examples")
    assert "last_model_version" not in columns(conn, "saved_voices")


OLD_EXAMPLES_DDL = """CREATE TABLE voice_examples (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  voice_id INTEGER NOT NULL REFERENCES saved_voices(id) ON DELETE CASCADE,
  language TEXT NOT NULL, source TEXT NOT NULL, transcript TEXT, storage_key TEXT NOT NULL,
  content_hash TEXT NOT NULL, seconds REAL NOT NULL, sample_rate INTEGER NOT NULL,
  consent_at TEXT NOT NULL, created_at TEXT NOT NULL,
  UNIQUE(voice_id, content_hash)
)"""


def test_migrate_adds_the_take_columns_to_an_old_table_and_keeps_its_row(tmp_path):
    db = connect(tmp_path / "old.db")
    migrate(db)
    db.execute("DROP TABLE voice_examples")
    db.execute(OLD_EXAMPLES_DDL)
    voice = create_voice(db, OWNER, "Grandma")
    db.execute(
        "INSERT INTO voice_examples(voice_id, language, source, transcript, storage_key, content_hash,"
        " seconds, sample_rate, consent_at, created_at)"
        " VALUES (?, 'en', 'recorded', 'Hi.', 'aa/x.wav', 'x', 6.0, 48000, '2026-01-01', '2026-01-01')",
        (voice.id,),
    )
    migrate(db)
    added = {"transcript_confirmed_at", "kit_id", "transcript_model", "transcript_error"}
    assert added <= columns(db, "voice_examples")
    assert {(table, column) for table, column, _type in _ADDED_COLUMNS if table == "voice_examples"} == {
        ("voice_examples", column) for column in added}
    (example,) = get_voice(db, OWNER, voice.id).examples
    assert (example.transcript, example.transcript_model, example.transcript_error, example.kit_id) == (
        "Hi.", None, None, None)
    assert example.words_state == "waiting"
    db.close()


def test_migrate_adds_the_measure_to_an_old_voice_tests_table_and_keeps_its_row(tmp_path):
    db = connect(tmp_path / "old.db")
    migrate(db)
    voice = create_voice(db, OWNER, "Grandma")
    db.execute("DROP TABLE voice_take_embeddings")
    db.execute("ALTER TABLE voice_tests DROP COLUMN likeness_json")
    db.execute("ALTER TABLE voice_tests DROP COLUMN likeness_error")
    db.execute(
        "INSERT INTO voice_tests(voice_id, language, text_kind, text, model_version, model_sha256, model_key,"
        " revision, elapsed_seconds, seconds, storage_key, created_at)"
        " VALUES (?, 'en', 'custom', 'Hi.', 1, 'sha', 'voxcpm2', 'rev', 1.0, 2.0, 'aa/t.wav', '2026-01-01')",
        (voice.id,),
    )
    migrate(db)
    assert {"likeness_json", "likeness_error"} <= columns(db, "voice_tests")
    assert {"example_id", "model_key", "revision", "pipeline", "vector"} <= columns(db, "voice_take_embeddings")
    test = sv.voice_test(db, OWNER, voice.id)
    assert (test.likeness, test.likeness_error) == (None, None)
    db.close()


OLD_VOICE_TESTS_DDL = """CREATE TABLE voice_tests (
  voice_id INTEGER PRIMARY KEY REFERENCES saved_voices(id) ON DELETE CASCADE,
  language TEXT NOT NULL,
  text_kind TEXT NOT NULL CHECK (text_kind IN ('sentence', 'take', 'custom')),
  text_id TEXT,
  example_id INTEGER REFERENCES voice_examples(id) ON DELETE SET NULL,
  text TEXT NOT NULL,
  model_version INTEGER NOT NULL,
  model_sha256 TEXT NOT NULL,
  model_key TEXT NOT NULL,
  revision TEXT NOT NULL,
  elapsed_seconds REAL NOT NULL,
  seconds REAL NOT NULL,
  storage_key TEXT NOT NULL,
  created_at TEXT NOT NULL,
  likeness_json TEXT,
  likeness_error TEXT
)"""


def test_migrate_adds_the_take_that_spoke_to_an_old_voice_tests_table(tmp_path, store):
    db = connect(tmp_path / "old.db")
    migrate(db)
    voice = create_voice(db, OWNER, "Grandma")
    db.execute("DROP TABLE voice_tests")
    db.execute(OLD_VOICE_TESTS_DDL)
    db.execute(
        "INSERT INTO voice_tests(voice_id, language, text_kind, text, model_version, model_sha256, model_key,"
        " revision, elapsed_seconds, seconds, storage_key, created_at)"
        " VALUES (?, 'en', 'custom', 'Hi.', 3, 'sha', 'voxcpm2', 'rev', 1.0, 2.0, 'aa/t.wav', '2026-01-01')",
        (voice.id,),
    )
    migrate(db)
    assert "prompt_example_id" in columns(db, "voice_tests")
    old = sv.voice_test(db, OWNER, voice.id)
    assert (old.model_version, old.prompt_example_id) == (3, None)
    prompt = take(db, store, voice.id)
    test = store_voice_test(db, store, voice.id, prompt)
    assert sv.voice_test(db, OWNER, voice.id) == test and test.prompt_example_id == prompt.id
    request_example_deletion(db, OWNER, voice.id, prompt.id)
    assert process_voice_deletions(db, store, OWNER) == 0
    assert sv.voice_test(db, OWNER, voice.id).prompt_example_id is None
    db.close()


def test_migrate_keeps_the_one_take_rows_and_their_files(conn, store):
    voice = create_voice(conn, OWNER, "Grandma")
    example = add(conn, store, voice.id)
    keys = [seed_conditioning(conn, store, example.id, b"blob-1"), seed_preview(conn, store, example.id)]
    seed_pin(conn, voice.id, example.id)
    migrate(conn)
    assert [rows(conn, t) for t in ("voice_conditionings", "voice_film_examples", "voice_previews")] == [1, 1, 1]
    assert all(store.exists(key) for key in keys)


def test_the_kit_holds_26_texts_per_language_with_the_reading_texts_first():
    for language, kit in sv.RECORDING_KIT.items():
        ids = [text.id for text in kit]
        assert len(kit) == 26 and len(set(ids)) == 26
        assert all(text_id.startswith(f"{language}-") for text_id in ids)
        assert ids[:4] == [f"{language}-{name}" for name in ("kitchen", "market", "rain", "train")]
        assert {text.style for text in kit[:4]} == {"normal"}
        assert all(text.style in sv.KIT_STYLES for text in kit)
        styles = [text.style for text in kit[4:]]
        assert sorted(set(styles)) == sorted(set(sv.KIT_STYLES) - {"normal"})
        assert all(styles.count(style) == 2 for style in styles)
        assert RECORDING_PROMPTS[language] == {text.id: text.text for text in kit}
    assert RECORDING_PROMPTS["en"]["en-kitchen"].startswith("The quick morning light")


def test_no_kit_text_holds_a_word_of_the_voice_check_sentences():
    for text in sv.RECORDING_KIT["en"]:
        assert not set(re.findall(r"[a-z]+", text.text.lower())) & VOICE_CHECK_WORDS, text.id


def test_every_added_kit_text_has_80_to_140_characters():
    for kit in sv.RECORDING_KIT.values():
        for text in kit[4:]:
            assert 80 <= len(text.text) <= 140, text.id


def test_add_example_takes_no_words():
    assert "transcript" not in inspect.signature(add_example).parameters


@pytest.mark.parametrize("source, prompt_id, kit_id", [
    ("recorded", "ru-slow-1", "ru-slow-1"), ("recorded", None, None), ("uploaded", "ru-slow-1", None),
    ("uploaded", None, None)])
def test_a_new_take_keeps_only_the_kit_text_it_was_shown_and_waits_for_asr(
        conn, store, source, prompt_id, kit_id):
    voice = create_voice(conn, OWNER, "Grandma")
    example = add_example(conn, store, OWNER, voice.id, language="ru", source=source, data=CLIP, prompt_id=prompt_id)
    assert (example.kit_id, example.transcript, example.transcript_model, example.transcript_error) == (
        kit_id, None, None, None)
    assert example.words_state == "waiting"
    assert get_voice(conn, OWNER, voice.id).examples == (example,)


def test_a_recording_with_an_unknown_prompt_id_is_refused(conn, store):
    voice = create_voice(conn, OWNER, "Grandma")
    with pytest.raises(VoiceError, match="Unknown reading text"):
        add_example(conn, store, OWNER, voice.id, language="ru", source="recorded", data=CLIP, prompt_id="ru-nowhere")
    assert rows(conn, "voice_examples") == 0


@pytest.mark.parametrize("text, expected", [
    ("  Мои   слова. ", ("Мои слова.", None)),
    ("1 2 3 4 5", ("1 2 3 4 5", None)),
    ("", (None, sv.NO_WORDS)), ("   ", (None, sv.NO_WORDS)), ("♪ ♪", (None, sv.NO_WORDS)),
    ("...", (None, sv.NO_WORDS)),
    ("x" * 1000, ("x" * 1000, None)),
    ("x" * 1001, (None, sv.TOO_LONG)),
])
def test_heard_words_keeps_letters_or_digits_up_to_the_limit(text, expected):
    assert sv.heard_words(text) == expected


def test_the_reasons_never_claim_that_no_speech_was_heard():
    assert sv.NO_WORDS == "the speech recognition found no words in it"
    assert sv.TOO_LONG == "it holds more than 1000 characters of words"
    assert "speech was heard" not in sv.NO_WORDS + sv.TOO_LONG


def test_not_used_text_names_each_unusable_take_with_its_own_reason(conn, store):
    voice = create_voice(conn, OWNER, "Grandma")
    heard = take(conn, store, voice.id)
    silent = unusable_take(conn, store, voice.id, make_wav(7.0))
    long = unusable_take(conn, store, voice.id, make_wav(8.0), text="x" * 1001)
    waiting = add(conn, store, voice.id, make_wav(9.0))
    assert sv.not_used_text([long, waiting, heard, silent]) == (
        f"take #{silent.id} — {sv.NO_WORDS}; take #{long.id} — {sv.TOO_LONG}")
    assert sv.not_used_text([heard, waiting]) == ""


def test_take_to_hear_is_the_take_only_while_it_waits(conn, store):
    voice = create_voice(conn, OWNER, "Grandma")
    waiting = add(conn, store, voice.id)
    assert sv.take_to_hear(conn, OWNER, voice.id, waiting.id) == waiting
    assert sv.take_to_hear(conn, OTHER_OWNER, voice.id, waiting.id) is None
    heard = take(conn, store, voice.id, make_wav(7.0))
    assert sv.take_to_hear(conn, OWNER, voice.id, heard.id) is None
    pending = add(conn, store, voice.id, make_wav(8.0))
    drained = add(conn, store, voice.id, make_wav(9.0))
    request_example_deletion(conn, OWNER, voice.id, pending.id)
    request_example_deletion(conn, OWNER, voice.id, drained.id)
    assert sv.take_to_hear(conn, OWNER, voice.id, pending.id) is None
    assert process_voice_deletions(conn, store, OWNER) == 0
    assert sv.take_to_hear(conn, OWNER, voice.id, drained.id) is None
    request_voice_deletion(conn, OWNER, voice.id)
    assert sv.take_to_hear(conn, OWNER, voice.id, waiting.id) is None


def test_set_heard_words_stores_the_words_or_the_reason_once(conn, store):
    voice = create_voice(conn, OWNER, "Grandma")
    first = add(conn, store, voice.id)
    second = add(conn, store, voice.id, make_wav(7.0))
    heard = sv.set_heard_words(conn, OWNER, voice.id, first.id, text="  Мои   слова. ", model=ASR)
    assert (heard.transcript, heard.transcript_model, heard.transcript_error) == ("Мои слова.", ASR, None)
    assert heard.words_state == "heard"
    unusable = sv.set_heard_words(conn, OWNER, voice.id, second.id, text="♪", model=ASR)
    assert (unusable.transcript, unusable.transcript_model, unusable.transcript_error) == (None, ASR, sv.NO_WORDS)
    assert unusable.words_state == "unusable"
    assert get_voice(conn, OWNER, voice.id).examples == (heard, unusable)
    for example_id in (first.id, second.id):
        assert sv.set_heard_words(conn, OWNER, voice.id, example_id, text="Other words.", model="other@1") is None
    assert get_voice(conn, OWNER, voice.id).examples == (heard, unusable)
    assert not conn.in_transaction


def test_set_heard_words_writes_nothing_for_a_pending_or_foreign_take(conn, store):
    voice = create_voice(conn, OWNER, "Grandma")
    pending = add(conn, store, voice.id)
    request_example_deletion(conn, OWNER, voice.id, pending.id)
    other = create_voice(conn, OWNER, "Ann")
    of_pending_voice = add(conn, store, other.id, make_wav(7.0))
    request_voice_deletion(conn, OWNER, other.id)
    stranger = create_voice(conn, OTHER_OWNER, "Grandma")
    of_other_owner = add(conn, store, stranger.id, make_wav(8.0), owner=OTHER_OWNER)
    assert sv.set_heard_words(conn, OWNER, voice.id, pending.id, text=WORDS, model=ASR) is None
    assert sv.set_heard_words(conn, OWNER, other.id, of_pending_voice.id, text=WORDS, model=ASR) is None
    assert sv.set_heard_words(conn, OWNER, stranger.id, of_other_owner.id, text=WORDS, model=ASR) is None
    assert sv.set_heard_words(conn, OWNER, voice.id, of_pending_voice.id, text=WORDS, model=ASR) is None
    assert conn.execute("SELECT COUNT(*) FROM voice_examples WHERE transcript_model IS NOT NULL").fetchone()[0] == 0
    assert not conn.in_transaction


def test_a_take_stored_before_asr_is_heard_and_its_shown_text_replaced(conn, store):
    voice = create_voice(conn, OWNER, "Grandma")
    legacy = add(conn, store, voice.id, source="recorded")
    shown = RECORDING_PROMPTS["en"]["en-kitchen"]
    conn.execute("UPDATE voice_examples SET transcript = ? WHERE id = ?", (shown, legacy.id))
    (stored,) = get_voice(conn, OWNER, voice.id).examples
    assert (stored.transcript, stored.words_state) == (shown, "waiting")
    assert sv.takes_to_hear(conn, OWNER, voice.id) == [stored]
    assert sv.bake_snapshot(conn, OWNER, voice.id).takes == ()
    heard = sv.set_heard_words(conn, OWNER, voice.id, legacy.id, text=WORDS, model=ASR)
    assert (heard.transcript, heard.words_state) == (WORDS, "heard")


def test_takes_to_hear_lists_the_waiting_takes_that_are_not_pending_by_id(conn, store):
    voice = create_voice(conn, OWNER, "Grandma")
    first = add(conn, store, voice.id)
    take(conn, store, voice.id, make_wav(7.0))
    unusable_take(conn, store, voice.id, make_wav(8.0))
    pending = add(conn, store, voice.id, make_wav(9.0))
    last = add(conn, store, voice.id, make_wav(10.0), language="ru", source="recorded")
    request_example_deletion(conn, OWNER, voice.id, pending.id)
    assert [t.id for t in sv.takes_to_hear(conn, OWNER, voice.id)] == [first.id, last.id]
    assert sv.takes_to_hear(conn, OTHER_OWNER, voice.id) == []
    request_voice_deletion(conn, OWNER, voice.id)
    assert sv.takes_to_hear(conn, OWNER, voice.id) == []


def test_the_confirmed_words_api_is_gone():
    assert not hasattr(sv, "set_transcript")
    names = {field.name for field in dataclasses.fields(sv.VoiceExample)}
    assert "transcript_confirmed_at" not in names
    assert not hasattr(sv.VoiceExample, "unresolved")
    assert [field.name for field in dataclasses.fields(sv.ModelChanges)] == ["added", "deleted", "unusable"]
    assert [field.name for field in dataclasses.fields(sv.BakeSnapshot)] == [
        "takes", "waiting", "unusable", "seconds", "by_language", "dataset_hash"]


def test_bake_snapshot_sends_the_heard_takes_and_lists_the_others_by_state(conn, store):
    voice = create_voice(conn, OWNER, "Grandma")
    heard = take(conn, store, voice.id)
    russian = take(conn, store, voice.id, make_wav(7.0), language="ru", words="Слова.")
    waiting = add(conn, store, voice.id, make_wav(8.0))
    recorded = add(conn, store, voice.id, make_wav(9.0), source="recorded")
    unusable = unusable_take(conn, store, voice.id, make_wav(10.0))
    pending = take(conn, store, voice.id, make_wav(11.0))
    request_example_deletion(conn, OWNER, voice.id, pending.id)
    snapshot = sv.bake_snapshot(conn, OWNER, voice.id)
    assert [t.id for t in snapshot.takes] == [heard.id, russian.id]
    assert [t.id for t in snapshot.waiting] == [waiting.id, recorded.id]
    assert [t.id for t in snapshot.unusable] == [unusable.id]
    assert snapshot.seconds == pytest.approx(13.0)
    assert snapshot.by_language == pytest.approx({"en": 6.0, "ru": 7.0})
    assert snapshot.dataset_hash == sv.snapshot_hash([heard, russian])


def test_snapshot_hash_is_the_services_dataset_hash_ignores_order_and_changes_with_words(conn, store):
    voice = create_voice(conn, OWNER, "Grandma")
    first = take(conn, store, voice.id)
    second = take(conn, store, voice.id, make_wav(7.0), language="ru", words="Слова.")
    lines = sorted([f"{first.content_hash}\ten\t{WORDS}", f"{second.content_hash}\tru\tСлова."])
    assert sv.snapshot_hash([first, second]) == hashlib.sha256("\n".join(lines).encode()).hexdigest()
    assert sv.snapshot_hash([second, first]) == sv.snapshot_hash([first, second])
    other = take(conn, store, create_voice(conn, OWNER, "Ann").id, words="Other words.")
    assert other.content_hash == first.content_hash
    assert sv.snapshot_hash([other, second]) != sv.snapshot_hash([first, second])


def test_a_bake_run_gives_back_its_snapshot_after_the_takes_change(conn, store):
    voice = create_voice(conn, OWNER, "Grandma")
    first = take(conn, store, voice.id)
    recorded = add_example(conn, store, OWNER, voice.id, language="ru", source="recorded",
                           data=make_wav(7.0), prompt_id="ru-slow-1")
    second = sv.set_heard_words(conn, OWNER, voice.id, recorded.id, text="Слова.", model=ASR)
    snapshot = sv.bake_snapshot(conn, OWNER, voice.id)
    bake_key = sv.start_bake_run(conn, OWNER, voice.id, snapshot)
    conn.execute("UPDATE voice_examples SET transcript = 'Changed later.' WHERE id = ?", (first.id,))
    request_example_deletion(conn, OWNER, voice.id, second.id)
    assert process_voice_deletions(conn, store, OWNER) == 0
    take(conn, store, voice.id, make_wav(8.0))
    run = sv.read_bake_run(conn, OWNER, bake_key)
    assert run == sv.BakeRun(voice.id, (
        sv.SnapshotTake(first.id, first.content_hash, "en", 6.0, WORDS, None,
                        fingerprint(first.content_hash, "en", WORDS)),
        sv.SnapshotTake(second.id, second.content_hash, "ru", 7.0, "Слова.", "ru-slow-1",
                        fingerprint(second.content_hash, "ru", "Слова.")),
    ), snapshot.dataset_hash, None)
    voice_id, takes, dataset_hash, _cancel = run
    assert (voice_id, len(takes), dataset_hash) == (voice.id, 2, snapshot.dataset_hash)
    assert sv.read_bake_run(conn, OTHER_OWNER, bake_key) is None
    sv.end_bake_run(conn, bake_key)
    assert sv.read_bake_run(conn, OWNER, bake_key) is None


def test_start_bake_run_on_a_pending_voice_is_refused(conn, store):
    voice = create_voice(conn, OWNER, "Grandma")
    take(conn, store, voice.id)
    snapshot = sv.bake_snapshot(conn, OWNER, voice.id)
    request_voice_deletion(conn, OWNER, voice.id)
    with pytest.raises(VoiceDeleting):
        sv.start_bake_run(conn, OWNER, voice.id, snapshot)
    assert rows(conn, "voice_bake_runs") == 0 and not conn.in_transaction


def test_save_voice_model_stores_v1_then_v2_in_place(conn, store):
    voice = create_voice(conn, OWNER, "Grandma")
    first = take(conn, store, voice.id)
    v1 = bake(conn, store, voice.id)
    assert (v1.version, v1.model_key, v1.revision, v1.lora_json) == (1, "voxcpm2", REVISION, LORA_JSON)
    assert (v1.sha256, v1.bytes, v1.takes, v1.seconds) == (hashlib.sha256(ADAPTER).hexdigest(), len(ADAPTER), 1, 6.0)
    assert json.loads(v1.config_json) == {"steps": 4} and v1.metrics["loss_last"] == 1.25
    assert sv.read_voice_model(store, v1) == ADAPTER
    assert v1.storage_key.endswith(".safetensors")
    assert rows(conn, "voice_bake_runs") == 0 and rows(conn, "voice_file_deletions") == 0
    second = take(conn, store, voice.id, make_wav(7.0), language="ru", words="Слова.")
    v2 = bake(conn, store, voice.id, ADAPTER_2)
    assert v2.version == 2 and v2.takes == 2 and v2.seconds == pytest.approx(13.0)
    assert rows(conn, "voice_models") == 1
    assert [tuple(r) for r in conn.execute(
        "SELECT example_id, language, seconds, fingerprint FROM voice_model_takes ORDER BY example_id")] == [
        (first.id, "en", 6.0, fingerprint(first.content_hash, "en", WORDS)),
        (second.id, "ru", 7.0, fingerprint(second.content_hash, "ru", "Слова.")),
    ]
    queued = conn.execute("SELECT storage_key, deletion_id FROM voice_file_deletions").fetchall()
    assert [(r["storage_key"], r["deletion_id"]) for r in queued] == [(v1.storage_key, None)]
    assert process_voice_deletions(conn, store, OWNER) == 0
    assert not store.exists(v1.storage_key) and sv.read_voice_model(store, v2) == ADAPTER_2


def test_the_same_bake_is_stored_once(conn, tmp_path):
    counting = CountingStorage(LocalStorage(tmp_path / "counted"))
    voice = create_voice(conn, OWNER, "Grandma")
    take(conn, counting, voice.id)
    bake_key = start(conn, voice.id)
    run = sv.read_bake_run(conn, OWNER, bake_key)
    first = store_run(conn, counting, voice.id, bake_key)
    writes = counting.writes
    again = sv.save_voice_model(conn, counting, OWNER, voice.id, bake_key=bake_key, data=ADAPTER,
                                status=done_status(run.dataset_hash), takes=run.takes,
                                dataset_hash=run.dataset_hash)
    assert again == first and again.version == 1
    assert counting.writes == writes and rows(conn, "voice_models") == 1


def test_a_model_of_other_takes_is_refused(conn, store):
    voice = create_voice(conn, OWNER, "Grandma")
    take(conn, store, voice.id)
    bake_key = start(conn, voice.id)
    with pytest.raises(VoiceError, match="does not match the takes"):
        store_run(conn, store, voice.id, bake_key, dataset_sha256="other")
    assert rows(conn, "voice_models") == 0 and rows(conn, "voice_bake_runs") == 1


@pytest.mark.parametrize("drained", [False, True])
def test_a_take_deleted_during_the_bake_does_not_drop_the_model(conn, store, drained):
    voice = create_voice(conn, OWNER, "Grandma")
    kept = take(conn, store, voice.id)
    gone = take(conn, store, voice.id, make_wav(7.0))
    bake(conn, store, voice.id)
    bake_key = start(conn, voice.id)
    request_example_deletion(conn, OWNER, voice.id, gone.id)
    if drained:
        assert process_voice_deletions(conn, store, OWNER) == 0
    model = store_run(conn, store, voice.id, bake_key, ADAPTER_2)
    assert model.version == 2 and model.takes == 2
    trained = [r[0] for r in conn.execute("SELECT example_id FROM voice_model_takes ORDER BY example_id")]
    assert trained == [kept.id, gone.id]


def test_a_heard_take_keeps_its_words_during_a_bake(conn, store):
    voice = create_voice(conn, OWNER, "Grandma")
    first = take(conn, store, voice.id)
    bake_key = start(conn, voice.id)
    assert sv.set_heard_words(conn, OWNER, voice.id, first.id, text="Other words.", model=ASR) is None
    row = conn.execute("SELECT transcript, transcript_model, transcript_error FROM voice_examples WHERE id = ?",
                       (first.id,)).fetchone()
    assert tuple(row) == (WORDS, ASR, None)
    assert store_run(conn, store, voice.id, bake_key).version == 1
    assert [r[0] for r in conn.execute("SELECT fingerprint FROM voice_model_takes")] == [
        fingerprint(first.content_hash, "en", WORDS)]
    assert sv.voice_model_for(conn, OWNER, voice.id).changes == sv.ModelChanges(0, 0, 0)


def test_a_pending_voice_stores_no_model(conn, store):
    voice = create_voice(conn, OWNER, "Grandma")
    take(conn, store, voice.id)
    bake_key = start(conn, voice.id)
    request_voice_deletion(conn, OWNER, voice.id)
    assert store_run(conn, store, voice.id, bake_key) is None
    assert rows(conn, "voice_models") == 0
    assert process_voice_deletions(conn, store, OWNER) == 0
    assert not store.exists(sv.content_key(hashlib.sha256(ADAPTER).hexdigest(), ".safetensors"))


def test_voice_model_for_a_voice_never_baked(conn, store):
    voice = create_voice(conn, OWNER, "Grandma")
    take(conn, store, voice.id)
    add(conn, store, voice.id, make_wav(7.0))
    unusable_take(conn, store, voice.id, make_wav(8.0))
    pending = add(conn, store, voice.id, make_wav(9.0))
    request_example_deletion(conn, OWNER, voice.id, pending.id)
    assert sv.voice_model_for(conn, OWNER, voice.id) == sv.VoiceModelState(
        None, False, "none", {}, sv.ModelChanges(added=2, deleted=0, unusable=1))


def test_voice_model_for_counts_each_change_since_the_model_once(conn, store):
    voice = create_voice(conn, OWNER, "Grandma")
    take(conn, store, voice.id)
    deleted = take(conn, store, voice.id, make_wav(7.0))
    take(conn, store, voice.id, make_wav(8.0), language="ru", words="Слова.")
    model = bake(conn, store, voice.id)
    state = sv.voice_model_for(conn, OWNER, voice.id)
    assert state == sv.VoiceModelState(model, True, None, {"en": (2, 13.0), "ru": (1, 8.0)},
                                       sv.ModelChanges(0, 0, 0))
    take(conn, store, voice.id, make_wav(9.0))
    assert sv.voice_model_for(conn, OWNER, voice.id).changes == sv.ModelChanges(1, 0, 0)
    add(conn, store, voice.id, make_wav(10.0))
    assert sv.voice_model_for(conn, OWNER, voice.id).changes == sv.ModelChanges(2, 0, 0)
    unusable_take(conn, store, voice.id, make_wav(11.0))
    assert sv.voice_model_for(conn, OWNER, voice.id).changes == sv.ModelChanges(2, 0, 1)
    request_example_deletion(conn, OWNER, voice.id, deleted.id)
    assert sv.voice_model_for(conn, OWNER, voice.id).changes == sv.ModelChanges(2, 1, 1)
    assert process_voice_deletions(conn, store, OWNER) == 0
    state = sv.voice_model_for(conn, OWNER, voice.id)
    assert state.changes == sv.ModelChanges(2, 1, 1)
    assert state.usable and state.languages == {"en": (2, 13.0), "ru": (1, 8.0)}


@pytest.mark.parametrize("column, value", [("revision", "another-commit"), ("lora_json", "{}"),
                                           ("model_key", "kokoro-gone")])
def test_a_model_for_another_revision_or_lora_shape_is_not_usable(conn, store, column, value):
    voice = create_voice(conn, OWNER, "Grandma")
    take(conn, store, voice.id)
    bake(conn, store, voice.id)
    conn.execute(f"UPDATE voice_models SET {column} = ?", (value,))
    state = sv.voice_model_for(conn, OWNER, voice.id)
    assert (state.usable, state.problem) == (False, "revision") and state.model is not None
    assert sv.stored_model(conn, OWNER, voice.id, "en") == sv.StoredModel(None, "revision")


def test_stored_model_is_the_usable_model_that_learned_the_language(conn, store):
    baked = create_voice(conn, OWNER, "Ann")
    take(conn, store, baked.id, language="ru", words="Слова.")
    model = bake(conn, store, baked.id)
    unbaked = create_voice(conn, OWNER, "Bob")
    take(conn, store, unbaked.id, make_wav(7.0))
    assert sv.stored_model(conn, OWNER, baked.id, "ru") == sv.StoredModel(model, None)
    assert sv.stored_model(conn, OWNER, baked.id, "en") == sv.StoredModel(None, "language")
    assert sv.stored_model(conn, OWNER, unbaked.id, "en") == sv.StoredModel(None, "none")
    assert sv.stored_model(conn, OTHER_OWNER, baked.id, "ru") == sv.StoredModel(None, "deleting")
    request_voice_deletion(conn, OWNER, baked.id)
    assert sv.stored_model(conn, OWNER, baked.id, "ru") == sv.StoredModel(None, "deleting")


def test_film_take_is_the_newest_heard_take_of_the_language(conn, store):
    voice = create_voice(conn, OWNER, "Grandma")
    take(conn, store, voice.id)
    second = take(conn, store, voice.id, make_wav(7.0))
    assert sv.film_take(conn, OWNER, voice.id, "en") == sv.FilmTake(second, None)
    add(conn, store, voice.id, make_wav(8.0))
    unusable_take(conn, store, voice.id, make_wav(9.0))
    assert sv.film_take(conn, OWNER, voice.id, "en") == sv.FilmTake(second, None)


def test_film_take_never_speaks_from_a_pending_take(conn, store):
    voice = create_voice(conn, OWNER, "Grandma")
    first = take(conn, store, voice.id)
    second = take(conn, store, voice.id, make_wav(7.0))
    request_example_deletion(conn, OWNER, voice.id, second.id)
    assert sv.film_take(conn, OWNER, voice.id, "en") == sv.FilmTake(first, None)


def test_film_take_names_why_a_language_is_not_ready(conn, store):
    voice = create_voice(conn, OWNER, "Grandma")
    assert sv.film_take(conn, OWNER, voice.id, "ru") == sv.FilmTake(None, "language")
    unusable_take(conn, store, voice.id, make_wav(6.5))
    assert sv.film_take(conn, OWNER, voice.id, "en") == sv.FilmTake(None, "unusable")
    add(conn, store, voice.id, make_wav(7.0))
    assert sv.film_take(conn, OWNER, voice.id, "en") == sv.FilmTake(None, "waiting")
    assert sv.film_take(conn, OTHER_OWNER, voice.id, "en") == sv.FilmTake(None, "deleting")
    request_voice_deletion(conn, OWNER, voice.id)
    assert sv.film_take(conn, OWNER, voice.id, "en") == sv.FilmTake(None, "deleting")


def test_film_take_needs_no_voice_model(conn, store):
    voice = create_voice(conn, OWNER, "Grandma")
    heard = take(conn, store, voice.id, language="ru", words="Слова.")
    assert rows(conn, "voice_models") == 0
    assert sv.film_take(conn, OWNER, voice.id, "ru") == sv.FilmTake(heard, None)


def test_castable_takes_are_the_ready_voices_with_their_film_take(conn, store):
    ann = create_voice(conn, OWNER, "Ann")
    ann_take = take(conn, store, ann.id, language="ru", words="Слова.")
    bob = create_voice(conn, OWNER, "Bob")
    add(conn, store, bob.id, make_wav(7.0), language="ru")
    assert [(v.id, t) for v, t in sv.castable_takes(conn, OWNER, "ru")] == [(ann.id, ann_take)]
    assert sv.castable_takes(conn, OWNER, "en") == []
    assert sv.castable_takes(conn, OTHER_OWNER, "ru") == []
    request_voice_deletion(conn, OWNER, ann.id)
    assert sv.castable_takes(conn, OWNER, "ru") == []


def test_model_reference_is_the_adapter_alone(conn, store):
    voice = create_voice(conn, OWNER, "Grandma")
    take(conn, store, voice.id)
    model = bake(conn, store, voice.id)
    assert sv.model_reference(model, ADAPTER, "en") == {
        "language": "en", "adapter": {"blob": ADAPTER, "sha256": hashlib.sha256(ADAPTER).hexdigest()},
        "use_clip": False,
    }


def test_save_test_stores_the_take_that_spoke_and_a_second_replaces_it(conn, store):
    voice = create_voice(conn, OWNER, "Grandma")
    prompt = take(conn, store, voice.id)
    first = store_voice_test(conn, store, voice.id, prompt, text_kind="sentence")
    assert first == sv.VoiceTest(
        voice_id=voice.id, language="en", text_kind="sentence", text_id="EN-1", example_id=None,
        text="Hello there.", model_version=0, model_sha256="", model_key="voxcpm2",
        revision=REVISION, elapsed_seconds=3.5, seconds=TEST_SECONDS,
        storage_key=first.storage_key, created_at=first.created_at, prompt_example_id=prompt.id,
    )
    assert sv.voice_test(conn, OWNER, voice.id) == first
    assert sv.voice_test(conn, OTHER_OWNER, voice.id) is None
    assert sv.read_test_audio(store, first) == TEST_AUDIO
    second = store_voice_test(conn, store, voice.id, prompt, make_wav(3.0, freq=330.0))
    assert rows(conn, "voice_tests") == 1 and sv.voice_test(conn, OWNER, voice.id) == second
    queued = conn.execute("SELECT storage_key, deletion_id FROM voice_file_deletions").fetchall()
    assert [(r["storage_key"], r["deletion_id"]) for r in queued] == [(first.storage_key, None)]
    assert process_voice_deletions(conn, store, OWNER) == 0
    assert not store.exists(first.storage_key) and store.exists(second.storage_key)


def test_save_test_refuses_a_pending_take_or_a_pending_voice(conn, store):
    voice = create_voice(conn, OWNER, "Grandma")
    prompt = take(conn, store, voice.id)
    other = take(conn, store, voice.id, make_wav(7.0))
    request_example_deletion(conn, OWNER, voice.id, other.id)
    assert store_voice_test(conn, store, voice.id, other) is None
    request_voice_deletion(conn, OWNER, voice.id)
    assert store_voice_test(conn, store, voice.id, prompt) is None
    assert rows(conn, "voice_tests") == 0


def test_a_test_keeps_its_row_when_the_take_that_spoke_is_deleted(conn, store):
    voice = create_voice(conn, OWNER, "Grandma")
    take(conn, store, voice.id)
    prompt = take(conn, store, voice.id, make_wav(7.0))
    test = store_voice_test(conn, store, voice.id, prompt)
    request_example_deletion(conn, OWNER, voice.id, prompt.id)
    assert process_voice_deletions(conn, store, OWNER) == 0
    kept = sv.voice_test(conn, OWNER, voice.id)
    assert kept.prompt_example_id is None and kept.storage_key == test.storage_key


def test_a_take_test_keeps_its_row_when_its_take_is_deleted(conn, store):
    voice = create_voice(conn, OWNER, "Grandma")
    spoken = take(conn, store, voice.id)
    cascaded = take(conn, store, voice.id, make_wav(7.0))
    prompt = take(conn, store, voice.id, make_wav(8.0))
    request_example_deletion(conn, OWNER, voice.id, cascaded.id)
    assert process_voice_deletions(conn, store, OWNER) == 0
    gone = store_voice_test(conn, store, voice.id, prompt, text_kind="take", example_id=cascaded.id)
    assert gone.example_id is None
    stored = store_voice_test(conn, store, voice.id, prompt, make_wav(3.0, freq=330.0),
                              text_kind="take", example_id=spoken.id)
    assert stored.example_id == spoken.id
    request_example_deletion(conn, OWNER, voice.id, spoken.id)
    assert process_voice_deletions(conn, store, OWNER) == 0
    kept = sv.voice_test(conn, OWNER, voice.id)
    assert kept.example_id is None and kept.storage_key == stored.storage_key
    assert store.exists(stored.storage_key)


def test_a_voice_delete_removes_its_model_and_its_test(conn, store):
    voice = create_voice(conn, OWNER, "Grandma")
    voice_take = take(conn, store, voice.id)
    model = bake(conn, store, voice.id)
    test = store_voice_test(conn, store, voice.id, voice_take)
    start(conn, voice.id)
    request_voice_deletion(conn, OWNER, voice.id)
    assert process_voice_deletions(conn, store, OWNER) == 0
    assert [rows(conn, t) for t in ("voice_models", "voice_model_takes", "voice_tests", "voice_bake_runs")] == [0] * 4
    assert not store.exists(model.storage_key) and not store.exists(test.storage_key)


def test_an_example_delete_keeps_the_model_and_what_it_learned(conn, store):
    voice = create_voice(conn, OWNER, "Grandma")
    trained = take(conn, store, voice.id)
    model = bake(conn, store, voice.id)
    request_example_deletion(conn, OWNER, voice.id, trained.id)
    assert process_voice_deletions(conn, store, OWNER) == 0
    assert sv.voice_model_for(conn, OWNER, voice.id).model == model
    assert [r[0] for r in conn.execute("SELECT example_id FROM voice_model_takes")] == [trained.id]
    assert sv.read_voice_model(store, model) == ADAPTER


def test_a_key_held_only_by_a_model_or_a_test_is_kept(conn, store):
    voice = create_voice(conn, OWNER, "Grandma")
    voice_take = take(conn, store, voice.id)
    model = bake(conn, store, voice.id)
    test = store_voice_test(conn, store, voice.id, voice_take)
    for key in (model.storage_key, test.storage_key):
        conn.execute("INSERT INTO voice_file_deletions(storage_key, deletion_id, requested_at)"
                     " VALUES (?, NULL, '2026-01-01')", (key,))
    assert process_voice_deletions(conn, store, OWNER) == 0
    assert sv.read_voice_model(store, model) == ADAPTER and sv.read_test_audio(store, test) == TEST_AUDIO


def test_the_one_take_api_is_gone_and_the_check_script_names_remain():
    for name in ("film_example", "set_film_example", "FilmExample", "castable", "save_conditioning",
                 "save_preview", "previews", "read_preview_audio", "VoicePreview", "film_model",
                 "castable_models", "FilmModel"):
        assert not hasattr(sv, name), name
    for name in ("example_for", "conditionings", "reference_for"):
        assert hasattr(sv, name), name


@pytest.mark.parametrize("module", IMPORTERS)
def test_every_importer_of_the_store_still_loads(module):
    importlib.import_module(module)


def test_a_cancel_request_is_stored_once_with_an_open_run(conn, store):
    voice = create_voice(conn, OWNER, "Grandma")
    take(conn, store, voice.id)
    bake_key = start(conn, voice.id)
    assert sv.read_bake_run(conn, OWNER, bake_key).cancel_requested_at is None
    assert sv.request_bake_cancel(conn, bake_key) is True
    asked = sv.read_bake_run(conn, OWNER, bake_key).cancel_requested_at
    assert asked is not None
    assert sv.request_bake_cancel(conn, bake_key) is True
    assert sv.read_bake_run(conn, OWNER, bake_key).cancel_requested_at == asked
    sv.end_bake_run(conn, bake_key)
    assert sv.request_bake_cancel(conn, bake_key) is False
    assert not conn.in_transaction


def test_a_cancelled_run_stores_no_model(conn, store):
    voice = create_voice(conn, OWNER, "Grandma")
    take(conn, store, voice.id)
    bake_key = start(conn, voice.id)
    sv.request_bake_cancel(conn, bake_key)
    with pytest.raises(sv.BakeCancelled):
        store_run(conn, store, voice.id, bake_key)
    assert rows(conn, "voice_models") == 0 and rows(conn, "voice_bake_runs") == 1
    assert rows(conn, "voice_file_deletions") == 1  # the write-ahead row, for the drain
    key = sv.content_key(hashlib.sha256(ADAPTER).hexdigest(), ".safetensors")
    assert not store.exists(key)
    assert process_voice_deletions(conn, store, OWNER) == 0


def test_a_run_that_already_ended_stores_no_model(conn, store):
    voice = create_voice(conn, OWNER, "Grandma")
    take(conn, store, voice.id)
    bake_key = start(conn, voice.id)
    run = sv.read_bake_run(conn, OWNER, bake_key)
    sv.end_bake_run(conn, bake_key)
    with pytest.raises(sv.BakeCancelled):
        sv.save_voice_model(conn, store, OWNER, voice.id, bake_key=bake_key, data=ADAPTER,
                            status=done_status(run.dataset_hash), takes=run.takes,
                            dataset_hash=run.dataset_hash)
    assert rows(conn, "voice_models") == 0


def test_a_cancel_after_the_store_finds_no_run_and_the_model_stays(conn, store):
    voice = create_voice(conn, OWNER, "Grandma")
    take(conn, store, voice.id)
    bake_key = start(conn, voice.id)
    model = store_run(conn, store, voice.id, bake_key)
    assert sv.request_bake_cancel(conn, bake_key) is False
    assert sv.voice_model_for(conn, OWNER, voice.id).model == model


def test_deleting_a_person_returns_their_face_groups_to_faces_to_name(conn, store):
    from ingest.people import groups_to_name
    from tests.factories import add_face, add_face_group, add_photo

    photo = add_photo(conn, content_hash="fa" * 32)
    person = create_voice(conn, OWNER, "Grandma")
    group = add_face_group(conn, owner_id=OWNER, person_id=person.id)
    for _ in range(2):
        add_face(conn, photo_id=photo, group_id=group, vector=(1.0, 0.0))

    request_voice_deletion(conn, OWNER, person.id)
    assert process_voice_deletions(conn, store, OWNER) == 0

    waiting = groups_to_name(conn, OWNER)
    assert [(g.id, g.person_id, g.face_count) for g in waiting] == [(group, None, 2)]
