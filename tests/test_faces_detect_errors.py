"""`POST /faces/detect` (design §5.1): absent face weights are an outage of the service
(503, the worker defers the job), and any other model error belongs to the image (500,
the worker fails the job). The child is a fake that raises what `TorchWorker` raises."""

import base64
import sys
import types

import pytest
from fastapi.testclient import TestClient

from embedding.face_model import FaceModel
from modelsvc.app import create_models_app
from modelsvc.backends.composite import CompositeBackend

REVISION = "d09715916a0778919a770c343533641e250b8699"
IMAGE = b"\x89PNG-one-upright-photo"


class _FailingChild:
    def __init__(self, error: str) -> None:
        self.error = error

    def call(self, method, *args):
        raise RuntimeError(self.error)


class _Slots:
    def __init__(self, child) -> None:
        self.child = child

    def has_unit(self, unit):
        return unit == "faces"

    def worker(self, unit):
        return self.child


def _detect(child) -> object:
    backend = CompositeBackend(embed=None, slots=_Slots(child))
    with TestClient(create_models_app(backend)) as tc:
        return tc.post("/faces/detect", json={"image": base64.b64encode(IMAGE).decode()})


class _NotInCache(Exception):
    pass


def _weights_missing_error(monkeypatch) -> str:
    def download(repo, revision, local_files_only):
        raise _NotInCache

    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = download
    errors = types.ModuleType("huggingface_hub.errors")
    errors.LocalEntryNotFoundError = _NotInCache
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", errors)
    with pytest.raises(RuntimeError) as raised:
        FaceModel("repo", "cuda", REVISION)
    return f"faces failed to load: RuntimeError: {raised.value}"


def test_absent_face_weights_are_a_503(monkeypatch):
    resp = _detect(_FailingChild(_weights_missing_error(monkeypatch)))

    assert resp.status_code == 503
    assert "make models" in resp.json()["detail"]


@pytest.mark.parametrize(
    "error",
    [
        "RuntimeError: a face vector is not finite or has a zero norm",
        "faces worker died during detect",
    ],
)
def test_another_face_model_error_is_a_500(error):
    resp = _detect(_FailingChild(error))

    assert resp.status_code == 500
    assert resp.json()["detail"] == error
