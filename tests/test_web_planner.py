import json

import pytest
from fastapi.testclient import TestClient

from config import Settings
from embedding.fakes import FakeEmbedder
from embedding.store import write_vector
from inference.fakes import FakeInferenceClient
from tests.factories import add_photo
from web.app import create_app


@pytest.fixture
def planner_client(settings, monkeypatch):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    spec = json.dumps({"semantic": "beach", "facets": {"time_of_day": ["night"]}})
    monkeypatch.setattr(
        Settings, "build_inference_client",
        lambda self: (FakeInferenceClient(responses=[spec]), "fake"),
    )
    app = create_app(settings)
    conn = app.state.context.conn
    fe = FakeEmbedder()
    pid = add_photo(conn, photo_id=1, content_hash="beach", thumb_key="beach.jpg")
    write_vector(conn, pid, fe.embed_texts(["beach"])[0])
    conn.execute(
        "INSERT INTO photo_facets(photo_id, key, value_text) VALUES (1, 'time_of_day', 'night')"
    )
    with TestClient(app) as tc:
        yield tc


def test_query_redirects_to_materialized_params(planner_client):
    r = planner_client.get("/library?q=moody+beach+at+night", follow_redirects=False)
    assert r.status_code == 303
    location = r.headers["location"]
    assert "planned=1" in location and "f_time_of_day=night" in location and "q=beach" in location


def test_planned_page_shows_a_removable_chip(planner_client):
    body = planner_client.get("/library?q=beach&f_time_of_day=night&planned=1").text
    assert "time_of_day" in body and "night" in body      # the chip label
    assert 'class="chip' in body                            # rendered as a chip
    assert "/thumb/1" in body                               # the matching photo survives the filter


def test_planner_failure_still_searches(settings, monkeypatch):
    # No monkeypatch of inference -> the default empty fake raises -> fallback.
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    app = create_app(settings)
    conn = app.state.context.conn
    fe = FakeEmbedder()
    add_photo(conn, photo_id=1, content_hash="beach", thumb_key="beach.jpg")
    write_vector(conn, 1, fe.embed_texts(["beach"])[0])
    with TestClient(app) as tc:
        assert "/thumb/1" in tc.get("/library?q=beach").text   # follows redirect, still finds it
