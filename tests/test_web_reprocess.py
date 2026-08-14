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


def test_upload_offers_reprocess_and_a_separate_recaption_button(client):
    body = client.get("/upload").text
    assert 'value="thumbnail"' in body            # the everyday reprocess
    assert 'value="caption"' in body              # the separate re-caption button
    assert "Re-caption all photos" in body
    assert 'name="to_stage" value="taxonomy"' in body  # everyday reprocess stops before captions


def test_reprocess_all_rebuilds_up_to_taxonomy_but_not_captions(client):
    response = client.post(
        "/reprocess", data={"from_stage": "thumbnail", "to_stage": "taxonomy"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    conn = client.app.state.context.conn
    for stage in ("thumbnail", "embed", "taxonomy"):
        assert stage_counts(conn, stage)["pending"] == 2
    assert stage_counts(conn, "caption")["pending"] == 0   # captions untouched


def test_recaption_requeues_only_the_caption_stage(client):
    response = client.post("/reprocess", data={"from_stage": "caption"}, follow_redirects=False)
    assert response.status_code == 303
    conn = client.app.state.context.conn
    assert stage_counts(conn, "caption")["pending"] == 2
    for stage in ("thumbnail", "embed", "taxonomy"):
        assert stage_counts(conn, stage)["pending"] == 0


def test_reprocess_unknown_stage_is_clamped_not_an_error(client):
    response = client.post("/reprocess", data={"from_stage": "bogus"}, follow_redirects=False)
    assert response.status_code == 303
