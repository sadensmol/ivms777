"""The `faces` stage across drain passes (design §8, AC4): a pass stores the faces the
models service finds, an outage of any length keeps the job pending with no attempt
counted, and a refused image or an unreadable original still fails the job. The models
service is a fake HTTP transport under the real `ModelsClient`; no model."""

import base64
import io
import json

import httpx
import pytest
from PIL import Image

from config import Settings
from inference.models_client import ModelsClient
from ingest.faces import FacesDeferred, FaceStageParams, faces_handler
from ingest.jobs import MAX_ATTEMPTS, enqueue
from ingest.pipeline import drain_pass
from ingest.vocab import load_vocab
from ingest.worker import drain
from tests.factories import add_photo
from web.app import VOCAB_PATH
from web.deps import build_context

REVISION = "d09715916a0778919a770c343533641e250b8699"
PARAMS = FaceStageParams(match_threshold=0.5, input_edge=640)
OUTAGE_PASSES = MAX_ATTEMPTS + 1
FACE_BOX = [0.1, 0.2, 0.3, 0.4]
FACE_SCORE = 0.93
FACE_VECTOR = [1.0] + [0.0] * 511


class _ModelsService:
    """`POST /faces/detect` in the state the test sets; every other route answers 503."""

    def __init__(self) -> None:
        self.state = "up"
        self.face_calls = 0
        self.poison: str | None = None
        self.poison_calls = 0

    def _poisoned(self, request: httpx.Request) -> httpx.Response:
        self.poison_calls += 1
        if self.poison == "read-timeout":
            raise httpx.ReadTimeout("timed out", request=request)
        return httpx.Response(500, json={"detail": "a face vector is not finite"})

    def handle(self, request: httpx.Request) -> httpx.Response:
        if request.url.path != "/faces/detect":
            return httpx.Response(503, json={"detail": "not in this test"})
        self.face_calls += 1
        if self.poison is not None and _is_poison(request):
            return self._poisoned(request)
        if self.state == "refused":
            return httpx.Response(422, json={"detail": "cannot identify image file"})
        if self.state == "connect-error":
            raise httpx.ConnectError("connection refused", request=request)
        if self.state == "read-timeout":
            raise httpx.ReadTimeout("timed out", request=request)
        if self.state == "server-error":
            return httpx.Response(503, json={"detail": "busy"})
        face = {"box": FACE_BOX, "score": FACE_SCORE, "vector": FACE_VECTOR}
        return httpx.Response(
            200, json={"model": "buffalo-l", "revision": REVISION, "faces": [face]}
        )

    def client(self) -> ModelsClient:
        return ModelsClient("http://models.test", transport=httpx.MockTransport(self.handle))


def _jpeg(color: str = "gray") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (800, 600), color).save(buffer, format="JPEG")
    return buffer.getvalue()


def _is_poison(request: httpx.Request) -> bool:
    image = base64.b64decode(json.loads(request.content)["image"])
    red, green, blue = Image.open(io.BytesIO(image)).convert("RGB").getpixel((0, 0))
    return red > 200 and green < 60 and blue < 60


@pytest.fixture
def service(monkeypatch) -> _ModelsService:
    fake = _ModelsService()
    monkeypatch.setattr(Settings, "build_models_client", lambda self: fake.client())
    return fake


@pytest.fixture
def ctx(settings):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return build_context(settings)


def _photo(ctx, content_hash: str, data: bytes | None = None) -> int:
    photo_id = add_photo(ctx.conn, content_hash=content_hash)
    key = ctx.conn.execute("SELECT storage_key FROM photos WHERE id = ?", (photo_id,)).fetchone()[0]
    ctx.originals.write(key, data if data is not None else _jpeg())
    return photo_id


def _pass(ctx) -> None:
    drain_pass(ctx, load_vocab(VOCAB_PATH))


def _job(conn, photo_id):
    return conn.execute(
        "SELECT status, attempts, error FROM jobs WHERE photo_id = ? AND stage = 'faces'",
        (photo_id,),
    ).fetchone()


def _faces(conn, photo_id):
    return conn.execute(
        "SELECT box_x, box_y, box_w, box_h, score, model, group_id FROM faces WHERE photo_id = ?",
        (photo_id,),
    ).fetchall()


def test_a_drain_pass_stores_the_faces_the_models_service_finds(ctx, service):
    photo = _photo(ctx, "e1" * 32)

    _pass(ctx)

    rows = _faces(ctx.conn, photo)
    assert len(rows) == 1
    face = rows[0]
    assert [face["box_x"], face["box_y"], face["box_w"], face["box_h"]] == pytest.approx(FACE_BOX)
    assert face["score"] == pytest.approx(FACE_SCORE)
    assert face["model"] == f"buffalo-l@{REVISION}"
    assert face["group_id"] is not None
    assert _job(ctx.conn, photo)["status"] == "done"
    assert service.face_calls == 1


@pytest.mark.parametrize("outage", ["connect-error", "server-error"])
def test_an_outage_longer_than_the_attempts_keeps_the_jobs_pending_until_the_service_is_up(
    ctx, service, outage
):
    photos = [_photo(ctx, "f1" * 32), _photo(ctx, "f2" * 32)]
    service.state = outage

    for passes in range(1, OUTAGE_PASSES + 1):
        _pass(ctx)
        assert service.face_calls == passes
    jobs = [_job(ctx.conn, photo) for photo in photos]
    assert [(job["status"], job["attempts"]) for job in jobs] == [("pending", 0)] * 2
    assert all(_faces(ctx.conn, photo) == [] for photo in photos)

    service.state = "up"
    _pass(ctx)

    assert [_job(ctx.conn, photo)["status"] for photo in photos] == ["done", "done"]
    assert all(len(_faces(ctx.conn, photo)) == 1 for photo in photos)


def test_a_refused_image_fails_the_job_after_its_attempts(ctx, service):
    photo = _photo(ctx, "a7" * 32)
    service.state = "refused"

    for _ in range(MAX_ATTEMPTS):
        _pass(ctx)

    job = _job(ctx.conn, photo)
    assert (job["status"], job["attempts"]) == ("failed", MAX_ATTEMPTS)
    assert service.face_calls == MAX_ATTEMPTS


def test_an_unreadable_original_fails_the_job_after_its_attempts(ctx, service):
    photo = _photo(ctx, "b8" * 32, data=b"not an image")

    for _ in range(MAX_ATTEMPTS):
        _pass(ctx)

    job = _job(ctx.conn, photo)
    assert (job["status"], job["attempts"]) == ("failed", MAX_ATTEMPTS)
    assert service.face_calls == 0


@pytest.mark.parametrize("outage", ["connect-error", "server-error"])
def test_an_outage_stops_the_stage_with_faces_deferred(ctx, service, outage):
    photo = _photo(ctx, "c9" * 32)
    enqueue(ctx.conn, photo, "faces")
    service.state = outage

    with pytest.raises(FacesDeferred):
        drain(
            ctx.conn, {"faces": faces_handler(ctx.originals, service.client(), PARAMS)}, ("faces",)
        )

    job = _job(ctx.conn, photo)
    assert (job["status"], job["attempts"], bool(job["error"])) == ("pending", 0, True)


@pytest.mark.parametrize("poison", ["server-error", "read-timeout"])
def test_a_poison_photo_fails_after_its_attempts_and_never_stops_a_good_photo(ctx, service, poison):
    bad = _photo(ctx, "d1" * 32, data=_jpeg("red"))
    good = _photo(ctx, "d2" * 32)
    service.poison = poison

    _pass(ctx)

    assert _job(ctx.conn, good)["status"] == "done"
    assert len(_faces(ctx.conn, good)) == 1

    for _ in range(MAX_ATTEMPTS - 1):
        _pass(ctx)

    job = _job(ctx.conn, bad)
    assert (job["status"], job["attempts"]) == ("failed", MAX_ATTEMPTS)
    assert service.poison_calls == MAX_ATTEMPTS
    assert _faces(ctx.conn, bad) == []
    assert _job(ctx.conn, good)["status"] == "done"
