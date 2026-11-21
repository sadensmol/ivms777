"""Edges of the face feature around its core: vectors the clustering cannot compare, the
match-threshold override, a profile with no face model, and a detect body that is not
base64 (design §4.1, §5.1, §8)."""

import base64
import hashlib

from fastapi.testclient import TestClient

from config import Settings
from ingest.faces import GroupMean, nearest_group
from models.catalog import face_model_for
from modelsvc.app import create_models_app
from modelsvc.backends.fake import FakeBackend

THRESHOLD = 0.5


def test_a_zero_vector_joins_no_group():
    groups = [GroupMean(group_id=1, mean=(1.0, 0.0))]
    assert nearest_group((0.0, 0.0), groups, THRESHOLD) is None


def test_a_group_of_another_vector_size_or_a_zero_mean_is_skipped():
    groups = [
        GroupMean(group_id=1, mean=(1.0, 0.0, 0.0)),
        GroupMean(group_id=2, mean=(0.0, 0.0)),
        GroupMean(group_id=3, mean=(0.8, 0.6)),
    ]
    assert nearest_group((1.0, 0.0), groups, THRESHOLD) == 3


def test_the_env_threshold_overrides_the_declared_one(monkeypatch, tmp_path):
    monkeypatch.setenv("IVMS777_FACE_MATCH_THRESHOLD", "0.62")
    assert Settings(profile="mac", data_dir=tmp_path).face_threshold() == 0.62


def test_a_profile_with_no_face_model_has_none():
    assert face_model_for("no-such-profile") is None


def test_a_detect_body_that_is_not_base64_is_a_422():
    with TestClient(create_models_app(FakeBackend())) as tc:
        resp = tc.post("/faces/detect", json={"image": "%%% not base64 %%%"})
    assert resp.status_code == 422


def test_the_fake_backend_answers_the_faces_declared_for_an_image():
    image = b"one photo"
    backend = FakeBackend()
    face = {"box": [0.1, 0.1, 0.2, 0.2], "score": 0.9, "vector": [1.0] + [0.0] * 511}
    backend.faces[hashlib.sha256(image).hexdigest()] = [face]
    with TestClient(create_models_app(backend)) as tc:
        resp = tc.post("/faces/detect", json={"image": base64.b64encode(image).decode()})
    assert resp.status_code == 200
    assert resp.json()["faces"] == [face]


def test_the_fake_backend_refuses_an_image_it_was_told_to():
    reason = "the image cannot be decoded"
    image = b"broken"
    backend = FakeBackend()
    backend.face_refusals[hashlib.sha256(image).hexdigest()] = reason
    with TestClient(create_models_app(backend)) as tc:
        resp = tc.post("/faces/detect", json={"image": base64.b64encode(image).decode()})
    assert resp.status_code == 422
    assert reason in resp.json()["detail"]
