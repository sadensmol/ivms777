"""The slot control surface on the `models` service (design §5.1, §4.1):
`GET /models/catalog`, `PUT /models/slots`, `POST /models/download`, `/embed/spec`.
"""

from fastapi.testclient import TestClient

from models import catalog
from modelsvc.app import create_models_app
from modelsvc.backends.fake import FakeBackend


def _client(**kw) -> TestClient:
    return TestClient(create_models_app(FakeBackend(**kw)))


def test_catalog_lists_the_profiles_entries_with_one_current_per_slot():
    c = _client(profile="jetson")
    body = c.get("/models/catalog").json()
    assert body["profile"] == "jetson"
    assert set(body["slots"]) == set(catalog.SLOTS)
    for slot in catalog.SLOTS:
        entries = [e for e in body["entries"] if e["slot"] == slot]
        assert entries, slot
        current = [e for e in entries if e["current"]]
        assert len(current) == 1
        assert current[0]["key"] == body["slots"][slot]


def test_catalog_entries_carry_what_the_popup_shows():
    c = _client(profile="jetson")
    entry = next(
        e for e in c.get("/models/catalog").json()["entries"] if e["key"] == "siglip2-so400m-384"
    )
    assert entry["display"]
    assert entry["size_mb"] > 0
    assert entry["cost_mb"] == 3400
    assert entry["cost_measured"] is True
    assert entry["dim"] == 1152
    assert entry["preprocess"] == {"input_px": 384, "resample": "bilinear", "mode": "squash"}
    assert entry["download"]["state"] in ("absent", "ready", "downloading", "error")
    assert entry["switchable"] is True


def test_catalog_hides_entries_this_profile_cannot_run():
    keys = {e["key"] for e in _client(profile="jetson").get("/models/catalog").json()["entries"]}
    assert "qwen3-vl-8b" not in keys  # mac only


def test_cloud_entries_are_not_switchable():
    body = _client(profile="cloud").get("/models/catalog").json()
    assert all(e["switchable"] is False for e in body["entries"])


def test_put_slots_switches_and_bumps_the_generation():
    c = _client(profile="jetson")
    before = c.get("/models").json()["generation"]
    resp = c.put("/models/slots", json={"slots": {"caption": "qwen3-vl-4b"}})
    assert resp.status_code == 200
    assert resp.json()["slots"]["caption"] == "qwen3-vl-4b"
    assert resp.json()["generation"] == before + 1
    assert c.get("/models").json()["slots"]["caption"] == "qwen3-vl-4b"


def test_put_slots_is_idempotent():
    c = _client(profile="jetson")
    c.put("/models/slots", json={"slots": {"caption": "qwen3-vl-4b"}})
    generation = c.get("/models").json()["generation"]
    c.put("/models/slots", json={"slots": {"caption": "qwen3-vl-4b"}})
    assert c.get("/models").json()["generation"] == generation


def test_put_slots_rejects_an_unknown_key_without_changing_anything():
    c = _client(profile="jetson")
    before = c.get("/models").json()
    resp = c.put("/models/slots", json={"slots": {"caption": "no-such-model"}})
    assert resp.status_code == 400
    assert c.get("/models").json() == before


def test_put_slots_rejects_a_model_this_profile_cannot_run():
    c = _client(profile="jetson")
    assert c.put("/models/slots", json={"slots": {"caption": "qwen3-vl-8b"}}).status_code == 400


def test_download_starts_and_shows_up_in_the_catalog():
    c = _client(profile="jetson")
    resp = c.post("/models/download", json={"slot": "caption", "key": "qwen3-vl-4b"})
    assert resp.status_code == 200
    assert resp.json()["state"] in ("downloading", "ready")
    entry = next(
        e for e in c.get("/models/catalog").json()["entries"] if e["key"] == "qwen3-vl-4b"
    )
    assert entry["download"]["state"] in ("downloading", "ready")


def test_download_rejects_an_unknown_model():
    c = _client(profile="jetson")
    assert c.post("/models/download", json={"slot": "caption", "key": "nope"}).status_code == 400


def test_embed_spec_carries_calibration_preprocess_and_generation():
    c = _client(profile="jetson")
    body = c.get("/embed/spec").json()
    assert body["logit_scale"] == 10.0 and body["logit_bias"] == -5.0
    assert body["preprocess"] == {"input_px": 384, "resample": "bilinear", "mode": "squash"}
    assert body["generation"] == c.get("/models").json()["generation"]


def test_embed_spec_follows_a_slot_switch():
    c = _client(profile="jetson")
    c.put("/models/slots", json={"slots": {"image_embed": "siglip2-so400m-512"}})
    body = c.get("/embed/spec").json()
    assert body["preprocess"]["input_px"] == 512
    assert body["generation"] == c.get("/models").json()["generation"]


def test_the_calibration_only_endpoint_is_gone():
    # One round trip carries calibration AND preprocessing since plan 21.
    assert _client().get("/embed/calibration").status_code == 404
