"""The speaker similarity of a voice test (design §11.1), through the real `/voice/embed`
surface over `FakeBackend`: no real model."""

import hashlib
import math
import struct

import pytest
from fastapi.testclient import TestClient

from feed.saved_voices import (
    add_example,
    create_voice,
    get_voice,
    process_voice_deletions,
    request_example_deletion,
    set_heard_words,
)
from feed.voice_likeness import (
    LikenessError,
    Meter,
    measure_test,
    measured_older_takes,
)
from inference.models_client import ModelsClient
from modelsvc.app import create_models_app
from modelsvc.backends.fake import FakeBackend
from storage.local import LocalStorage
from tests.test_feed_build import FAKE_ASR, clip

OWNER = 1
DIM = 192
TEST_CLIP = clip(330.0)
ROOT_HALF = 1 / math.sqrt(2)


def axes(**values: float) -> list[float]:
    """A 192-value vector: `x` and `y` name its first two values."""
    return [values.get("x", 0.0), values.get("y", 0.0)] + [0.0] * (DIM - 2)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Recording(ModelsClient):
    def __init__(self, backend: FakeBackend) -> None:
        super().__init__("http://modelsvc", transport=TestClient(create_models_app(backend))._transport)
        self.embedded: list[bytes] = []

    def embed_voice(self, audio, *, timeout=600.0):
        self.embedded.append(audio)
        return super().embed_voice(audio, timeout=timeout)


@pytest.fixture
def backend():
    return FakeBackend()


@pytest.fixture
def client(backend):
    return Recording(backend)


@pytest.fixture
def voices(tmp_path):
    return LocalStorage(tmp_path / "voices")


@pytest.fixture
def voice_id(conn):
    return create_voice(conn, OWNER, "me").id


def take(conn, voices, backend, voice_id, data, vector=None, *, language="en"):
    if vector is not None:
        backend.speaker_vectors[sha(data)] = vector
    return add_example(conn, voices, OWNER, voice_id, language=language, source="uploaded", data=data)


def measure(conn, client, voices, voice_id, *, language="en", audio=TEST_CLIP, seen=None):
    def before_embedding(stage, done):
        if seen is not None:
            seen.append((stage, done))

    return measure_test(conn, client, voices, OWNER, voice_id, language=language, audio=audio,
                        before_embedding=before_embedding)


def test_the_score_is_the_cosine_with_the_centroid_of_all_the_takes(conn, voices, backend, client, voice_id):
    take(conn, voices, backend, voice_id, clip(220.0), axes(x=1.0))
    take(conn, voices, backend, voice_id, clip(240.0), axes(y=1.0))
    backend.speaker_vectors[sha(TEST_CLIP)] = axes(x=1.0)
    assert measure(conn, client, voices, voice_id)["cosine"] == pytest.approx(ROOT_HALF, abs=1e-6)


def test_each_take_weighs_the_same_whatever_its_raw_length(conn, voices, backend, client, voice_id):
    take(conn, voices, backend, voice_id, clip(220.0), axes(x=50.0))
    take(conn, voices, backend, voice_id, clip(240.0), axes(y=0.5))
    backend.speaker_vectors[sha(TEST_CLIP)] = axes(y=7.0)
    assert measure(conn, client, voices, voice_id)["cosine"] == pytest.approx(ROOT_HALF, abs=1e-6)


def test_the_own_range_is_leave_one_out(conn, voices, backend, client, voice_id):
    first = take(conn, voices, backend, voice_id, clip(220.0), axes(x=1.0))
    second = take(conn, voices, backend, voice_id, clip(240.0), axes(y=1.0))
    third = take(conn, voices, backend, voice_id, clip(260.0), axes(x=1.0, y=1.0))
    likeness = measure(conn, client, voices, voice_id)
    # (1, 0) against the centroid of (0, 1) and (1, 1)/√2 is cos 67.5°.
    assert likeness["own"] == pytest.approx({"min": 0.382683, "median": 0.382683, "max": 1.0}, abs=1e-6)
    assert {row["example_id"]: row["own_cosine"] for row in likeness["takes"]} == pytest.approx(
        {first.id: 0.382683, second.id: 0.382683, third.id: 1.0}, abs=1e-6)


def test_under_two_takes_there_is_no_own_range_and_the_score_stays(conn, voices, backend, client, voice_id):
    take(conn, voices, backend, voice_id, clip(220.0), axes(x=1.0))
    backend.speaker_vectors[sha(TEST_CLIP)] = axes(x=1.0, y=1.0)
    likeness = measure(conn, client, voices, voice_id)
    assert likeness["own"] is None
    assert likeness["takes_count"] == 1
    assert likeness["cosine"] == pytest.approx(ROOT_HALF, abs=1e-6)


def test_the_set_is_the_takes_of_the_tests_language_that_are_not_deleting(conn, voices, backend, client, voice_id):
    english = take(conn, voices, backend, voice_id, clip(220.0))
    leaving = take(conn, voices, backend, voice_id, clip(240.0))
    take(conn, voices, backend, voice_id, clip(260.0), language="ru")
    request_example_deletion(conn, OWNER, voice_id, leaving.id)
    likeness = measure(conn, client, voices, voice_id)
    assert [row["example_id"] for row in likeness["takes"]] == [english.id]
    assert [row["content_hash"] for row in likeness["takes"]] == [english.content_hash]


def test_a_take_with_a_transcript_error_stays_in_the_set(conn, voices, backend, client, voice_id):
    example = take(conn, voices, backend, voice_id, clip(220.0))
    set_heard_words(conn, OWNER, voice_id, example.id, text=" ", model=FAKE_ASR)
    assert get_voice(conn, OWNER, voice_id).examples[0].transcript_error is not None
    assert measure(conn, client, voices, voice_id)["takes_count"] == 1


def test_a_refused_take_is_skipped_and_named(conn, voices, backend, client, voice_id):
    kept = take(conn, voices, backend, voice_id, clip(220.0))
    refused = take(conn, voices, backend, voice_id, clip(240.0))
    backend.speaker_refusals[sha(clip(240.0))] = "too little voiced audio"
    likeness = measure(conn, client, voices, voice_id)
    assert [row["example_id"] for row in likeness["takes"]] == [kept.id]
    assert likeness["skipped"] == [{"example_id": refused.id, "reason": "too little voiced audio"}]


def test_a_take_whose_file_is_gone_is_skipped_and_named(conn, voices, backend, client, voice_id):
    take(conn, voices, backend, voice_id, clip(220.0))
    gone = take(conn, voices, backend, voice_id, clip(240.0))
    voices.delete(gone.storage_key)
    likeness = measure(conn, client, voices, voice_id)
    assert [entry["example_id"] for entry in likeness["skipped"]] == [gone.id]
    assert likeness["takes_count"] == 1


def test_no_measurable_take_is_a_named_failure_never_a_zero(conn, voices, backend, client, voice_id):
    take(conn, voices, backend, voice_id, clip(220.0))
    backend.speaker_refusals[sha(clip(220.0))] = "too little voiced audio"
    with pytest.raises(LikenessError, match="no take"):
        measure(conn, client, voices, voice_id)


def test_a_refused_test_clip_embeds_no_take(conn, voices, backend, client, voice_id):
    take(conn, voices, backend, voice_id, clip(220.0))
    backend.speaker_refusals[sha(TEST_CLIP)] = "too little voiced audio"
    with pytest.raises(Exception, match="422"):
        measure(conn, client, voices, voice_id)
    assert client.embedded == [TEST_CLIP]


def test_provenance_names_the_evaluator_and_the_exact_takes(conn, voices, backend, client, voice_id):
    first = take(conn, voices, backend, voice_id, clip(220.0))
    second = take(conn, voices, backend, voice_id, clip(240.0))
    likeness = measure(conn, client, voices, voice_id)
    assert (likeness["evaluator"], likeness["pipeline"], likeness["method"]) == ("fake-speaker@fake", 1, 1)
    assert likeness["takes_count"] == 2
    expected = "\n".join(sorted((first.content_hash, second.content_hash))).encode()
    assert likeness["reference_hash"] == hashlib.sha256(expected).hexdigest()


def test_a_take_vector_is_stored_as_192_little_endian_float32(conn, voices, backend, client, voice_id):
    example = take(conn, voices, backend, voice_id, clip(220.0), axes(x=0.25, y=-0.5))
    measure(conn, client, voices, voice_id)
    row = conn.execute("SELECT * FROM voice_take_embeddings").fetchone()
    assert (row["example_id"], row["model_key"], row["revision"], row["pipeline"]) == (
        example.id, "fake-speaker", "fake", 1)
    assert len(row["vector"]) == DIM * 4
    assert struct.unpack(f"<{DIM}f", row["vector"])[:3] == (0.25, -0.5, 0.0)


def test_a_later_measure_embeds_only_the_new_takes_and_scores_the_same(conn, voices, backend, client, voice_id):
    take(conn, voices, backend, voice_id, clip(220.0), axes(x=1.0))
    first = measure(conn, client, voices, voice_id)
    client.embedded.clear()
    assert measure(conn, client, voices, voice_id) == first
    assert client.embedded == [TEST_CLIP]
    take(conn, voices, backend, voice_id, clip(240.0), axes(y=1.0))
    client.embedded.clear()
    measure(conn, client, voices, voice_id)
    assert client.embedded == [TEST_CLIP, clip(240.0)]


def test_a_vector_of_another_pipeline_is_never_used(conn, voices, backend, client, voice_id):
    example = take(conn, voices, backend, voice_id, clip(220.0), axes(x=1.0))
    conn.execute(
        "INSERT INTO voice_take_embeddings(example_id, model_key, revision, pipeline, vector, created_at)"
        " VALUES (?, 'fake-speaker', 'fake', 0, ?, '2026-01-01')",
        (example.id, struct.pack(f"<{DIM}f", *axes(y=1.0))))
    backend.speaker_vectors[sha(TEST_CLIP)] = axes(x=1.0)
    assert measure(conn, client, voices, voice_id)["cosine"] == pytest.approx(1.0)


def test_two_evaluators_in_one_measure_end_it(voices):
    class Swapped:
        replies = iter([{"vector": axes(x=1.0), "model": "m", "revision": "r", "pipeline": 1},
                        {"vector": axes(x=1.0), "model": "m", "revision": "r", "pipeline": 2}])

        def embed_voice(self, _audio):
            return next(self.replies)

    meter = Meter(Swapped())
    meter.embed(b"first")
    with pytest.raises(LikenessError, match="changed"):
        meter.embed(b"second")


def test_the_stage_names_the_take_and_the_fraction_rises(conn, voices, backend, client, voice_id):
    for freq in (220.0, 240.0):
        take(conn, voices, backend, voice_id, clip(freq))
    seen: list = []
    measure(conn, client, voices, voice_id, seen=seen)
    assert seen == [("measuring the voice", 0.0), ("measuring the voice · take 1 of 2", 0.0),
                    ("measuring the voice · take 2 of 2", 0.5)]


def test_a_stop_before_a_call_makes_no_more_calls(conn, voices, backend, client, voice_id):
    for freq in (220.0, 240.0):
        take(conn, voices, backend, voice_id, clip(freq))

    class Stop(Exception):
        pass

    def stop_at_the_second_take(stage, _done):
        if stage.endswith("take 2 of 2"):
            raise Stop

    with pytest.raises(Stop):
        measure_test(conn, client, voices, OWNER, voice_id, language="en", audio=TEST_CLIP,
                     before_embedding=stop_at_the_second_take)
    assert client.embedded == [TEST_CLIP, clip(220.0)]


def test_a_deleted_take_takes_its_vector_with_it(conn, voices, backend, client, voice_id):
    example = take(conn, voices, backend, voice_id, clip(220.0))
    measure(conn, client, voices, voice_id)
    assert conn.execute("SELECT COUNT(*) FROM voice_take_embeddings").fetchone()[0] == 1
    request_example_deletion(conn, OWNER, voice_id, example.id)
    process_voice_deletions(conn, voices, OWNER)
    assert conn.execute("SELECT COUNT(*) FROM voice_take_embeddings").fetchone()[0] == 0


def test_a_take_drained_during_the_measure_leaves_no_vector_and_no_error(conn, voices, backend, client, voice_id):
    example = take(conn, voices, backend, voice_id, clip(220.0))
    original = client.embed_voice

    def drained_during_the_call(audio, **kwargs):
        reply = original(audio, **kwargs)
        if audio != TEST_CLIP:
            conn.execute("DELETE FROM voice_examples WHERE id = ?", (example.id,))
        return reply

    client.embed_voice = drained_during_the_call
    assert measure(conn, client, voices, voice_id)["takes_count"] == 1
    assert conn.execute("SELECT COUNT(*) FROM voice_take_embeddings").fetchone()[0] == 0


def test_older_takes_shows_only_when_the_measured_set_changed(conn, voices, backend, client, voice_id):
    first = take(conn, voices, backend, voice_id, clip(220.0))
    refused = take(conn, voices, backend, voice_id, clip(240.0))
    backend.speaker_refusals[sha(clip(240.0))] = "too little voiced audio"
    likeness = measure(conn, client, voices, voice_id)

    def older() -> bool:
        return measured_older_takes(likeness, get_voice(conn, OWNER, voice_id), "en")

    assert not older()
    request_example_deletion(conn, OWNER, voice_id, refused.id)
    assert not older()
    take(conn, voices, backend, voice_id, clip(260.0), language="ru")
    assert not older()
    added = take(conn, voices, backend, voice_id, clip(280.0))
    assert older()
    request_example_deletion(conn, OWNER, voice_id, added.id)
    assert not older()
    request_example_deletion(conn, OWNER, voice_id, first.id)
    assert older()
