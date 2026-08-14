import json

import pytest
from fastapi.testclient import TestClient

from tests.factories import add_photo
from web.app import create_app


@pytest.fixture
def client(settings):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    app = create_app(settings)
    conn = app.state.context.conn
    for pid in (1, 2, 3):
        add_photo(conn, photo_id=pid, content_hash=f"h{pid}", thumb_key=f"{pid}.jpg",
                  caption="x", shot_at=f"2025-07-12T1{pid}:00:00")
    with TestClient(app) as test_client:
        yield test_client


def test_memories_is_in_the_dropdown(client):
    assert "Memories" in client.get("/organize?by=memories").text


def test_empty_state_prompts_a_first_build(client):
    body = client.get("/organize?by=memories").text
    assert "Rebuild memories" in body  # the build control is present


def test_progress_endpoint_reports_percent_while_building(client):
    client.app.state.memories_building = True
    client.app.state.memories_progress = {"done": 3, "total": 10}
    body = client.get("/organize/memories/progress").text
    assert "3/10 (30%)" in body


def test_progress_endpoint_refreshes_the_page_when_done(client):
    client.app.state.memories_building = False
    response = client.get("/organize/memories/progress")
    assert response.headers.get("HX-Refresh") == "true"


def test_rebuild_builds_and_shows_the_memory(client):
    client.app.state.queue_inference([json.dumps(
        {"action": "answer", "keep": True, "title": "Beach day",
         "description": "An afternoon by the water.", "drop_photo_ids": []}
    )])
    client.post("/organize/memories/rebuild", follow_redirects=False)
    client.app.state.await_build()  # join the background build thread
    assert "Beach day" in client.get("/organize?by=memories").text
