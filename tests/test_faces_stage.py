"""The `faces` stage (design §8): every photo gets a job, the worker stores the faces the
models service finds, a re-run replaces them, an outage defers the job, and a face that
joins a named group is that person at once. A fake models client; no model."""

import io
import struct

import httpx
import pytest
from PIL import Image

from feed.saved_voices import create_voice
from inference.models_client import DetectedFace
from ingest.faces import FaceStageParams, backfill_faces, faces_handler
from ingest.jobs import enqueue, reprocess_one
from ingest.people import people_in_photos
from ingest.pipeline import drain_pass
from ingest.vocab import load_vocab
from ingest.worker import drain
from storage.local import LocalStorage
from tests.factories import add_face, add_face_group, add_photo
from web.app import VOCAB_PATH
from web.deps import build_context

MODEL = "buffalo-l@d09715916a0778919a770c343533641e250b8699"
PARAMS = FaceStageParams(match_threshold=0.5, input_edge=640)
ORIENTATION = 0x0112


def _basis(index: int) -> tuple[float, ...]:
    return tuple(1.0 if i == index else 0.0 for i in range(512))


def _detected(box, index: int, score: float = 0.9) -> DetectedFace:
    return DetectedFace(box=box, score=score, vector=_basis(index), model=MODEL)


class _FaceClient:
    """`ModelsClient.detect_faces` alone: answers its replies in order, or raises."""

    def __init__(self, *replies: list[DetectedFace], error: Exception | None = None) -> None:
        self.replies = list(replies)
        self.error = error
        self.images: list[bytes] = []

    def detect_faces(self, image: bytes, *, timeout: float = 600.0) -> list[DetectedFace]:
        self.images.append(image)
        if self.error is not None:
            raise self.error
        return self.replies.pop(0)


def _jpeg(size=(800, 600), orientation: int | None = None) -> bytes:
    image = Image.new("RGB", size, "gray")
    exif = image.getexif()
    if orientation is not None:
        exif[ORIENTATION] = orientation
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", exif=exif)
    return buffer.getvalue()


@pytest.fixture
def originals(tmp_path):
    return LocalStorage(tmp_path / "orig")


def _photo(conn, originals, content_hash="aa" * 32, data=None) -> int:
    photo_id = add_photo(conn, content_hash=content_hash)
    key = conn.execute("SELECT storage_key FROM photos WHERE id = ?", (photo_id,)).fetchone()[0]
    originals.write(key, data or _jpeg())
    return photo_id


def _run(conn, originals, client, photo_id) -> None:
    enqueue(conn, photo_id, "faces")
    drain(conn, {"faces": faces_handler(originals, client, PARAMS)}, ("faces",))


def _faces(conn, photo_id):
    return conn.execute(
        "SELECT box_x, box_y, box_w, box_h, score, vector, model, group_id FROM faces"
        " WHERE photo_id = ? ORDER BY box_x", (photo_id,)).fetchall()


def _job(conn, photo_id):
    return conn.execute("SELECT status, error FROM jobs WHERE photo_id = ? AND stage = 'faces'",
                        (photo_id,)).fetchone()


def test_backfill_queues_a_faces_job_for_every_photo_without_one(conn):
    first = add_photo(conn, content_hash="a1")
    second = add_photo(conn, content_hash="a2")
    enqueue(conn, first, "faces")
    conn.execute("UPDATE jobs SET status = 'done' WHERE photo_id = ?", (first,))

    assert backfill_faces(conn) == 1
    assert _job(conn, first)["status"] == "done"
    second_job = _job(conn, second)
    assert second_job is not None and second_job["status"] == "pending"


def test_the_stage_stores_each_face_with_its_box_score_vector_and_model(conn, originals):
    photo = _photo(conn, originals)
    client = _FaceClient([_detected((0.1, 0.2, 0.3, 0.4), 0, 0.91),
                          _detected((0.6, 0.1, 0.2, 0.2), 1, 0.82)])

    _run(conn, originals, client, photo)

    rows = _faces(conn, photo)
    assert [v for r in rows for v in (r["box_x"], r["box_y"], r["box_w"], r["box_h"])] == (
        pytest.approx([0.1, 0.2, 0.3, 0.4, 0.6, 0.1, 0.2, 0.2]))
    assert [r["score"] for r in rows] == pytest.approx([0.91, 0.82])
    assert [struct.unpack("<512f", r["vector"]) for r in rows] == [_basis(0), _basis(1)]
    assert {r["model"] for r in rows} == {MODEL}
    assert all(r["group_id"] is not None for r in rows)
    assert _job(conn, photo)["status"] == "done"


def test_a_rerun_replaces_the_faces_of_the_photo(conn, originals):
    photo = _photo(conn, originals)
    client = _FaceClient([_detected((0.1, 0.1, 0.2, 0.2), 0), _detected((0.5, 0.1, 0.2, 0.2), 1)],
                         [_detected((0.7, 0.3, 0.1, 0.1), 2)])
    _run(conn, originals, client, photo)

    reprocess_one(conn, photo, "faces")
    drain(conn, {"faces": faces_handler(originals, client, PARAMS)}, ("faces",))

    rows = _faces(conn, photo)
    assert len(client.images) == 2
    assert [v for r in rows for v in (r["box_x"], r["box_y"])] == pytest.approx([0.7, 0.3])


def test_the_photo_is_sent_upright_and_shrunk_to_the_input_edge(conn, originals):
    # EXIF orientation 6: stored 1280x960, shown rotated a quarter turn as 960x1280.
    photo = _photo(conn, originals, data=_jpeg((1280, 960), orientation=6))
    client = _FaceClient([])

    _run(conn, originals, client, photo)

    assert len(client.images) == 1
    assert Image.open(io.BytesIO(client.images[0])).size == (480, 640)


def test_a_photo_smaller_than_the_input_edge_is_sent_at_its_own_size(conn, originals):
    photo = _photo(conn, originals, data=_jpeg((320, 240)))
    client = _FaceClient([])

    _run(conn, originals, client, photo)

    assert len(client.images) == 1
    assert Image.open(io.BytesIO(client.images[0])).size == (320, 240)


def test_a_models_service_outage_leaves_the_job_pending_with_no_faces(conn, originals):
    photo = _photo(conn, originals)
    client = _FaceClient(error=httpx.ConnectError("connection refused"))

    _run(conn, originals, client, photo)

    job = _job(conn, photo)
    assert len(client.images) == 1
    assert (job["status"], bool(job["error"])) == ("pending", True)
    assert _faces(conn, photo) == []


def test_a_drain_pass_with_the_models_service_down_defers_the_faces_job(settings):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    ctx = build_context(settings)
    photo = _photo(ctx.conn, ctx.originals, content_hash="cd" * 32)
    enqueue(ctx.conn, photo, "thumbnail")

    drain_pass(ctx, load_vocab(VOCAB_PATH))

    job = _job(ctx.conn, photo)
    assert job is not None and job["status"] == "pending"
    assert _faces(ctx.conn, photo) == []


def test_a_new_face_that_matches_a_named_group_is_that_person_at_once(conn, originals):
    known = add_photo(conn, content_hash="k1")
    ann = create_voice(conn, 1, "Ann").id
    group = add_face_group(conn, person_id=ann)
    add_face(conn, photo_id=known, group_id=group, vector=_basis(0))
    add_face(conn, photo_id=known, group_id=group, vector=_basis(0))
    photo = _photo(conn, originals)
    client = _FaceClient([_detected((0.2, 0.2, 0.3, 0.3), 0)])

    _run(conn, originals, client, photo)

    assert [r["group_id"] for r in _faces(conn, photo)] == [group]
    assert people_in_photos(conn, 1, [photo]) == {photo: ["Ann"]}
