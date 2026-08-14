import pytest
from fastapi.testclient import TestClient

from ingest.jobs import stage_counts
from tests.factories import add_photo
from web.app import create_app


@pytest.fixture
def client(settings):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    app = create_app(settings)
    conn = app.state.context.conn
    add_photo(conn, photo_id=1, content_hash="a", thumb_key="1.jpg")
    add_photo(conn, photo_id=2, content_hash="b", thumb_key="2.jpg")
    with TestClient(app) as test_client:
        yield test_client


def test_reprocess_control_is_a_single_full_pipeline_button(client):
    body = client.get("/upload").text
    assert "/reprocess" in body
    assert 'value="thumbnail"' in body          # one button, from the first stage
    assert body.count('type="submit" name="from_stage"') == 1


def test_reprocess_all_requeues_every_stage_for_every_photo(client):
    response = client.post("/reprocess", data={"from_stage": "thumbnail"}, follow_redirects=False)
    assert response.status_code == 303
    conn = client.app.state.context.conn
    for stage in ("thumbnail", "embed", "taxonomy", "caption"):
        assert stage_counts(conn, stage)["pending"] == 2   # whole pipeline, both photos


def test_reprocess_unknown_stage_is_clamped_not_an_error(client):
    response = client.post("/reprocess", data={"from_stage": "bogus"}, follow_redirects=False)
    assert response.status_code == 303
