import pytest
from fastapi.testclient import TestClient

from tests.factories import add_photo
from web.app import create_app


@pytest.fixture
def client(settings):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client


def test_upload_page_offers_a_directory_picker(client):
    body = client.get("/upload").text
    assert "webkitdirectory" in body
    assert "/static/upload.js" in body


def test_upload_page_shows_stage_progress(client):
    assert "thumbnail" in client.get("/upload").text


def test_progress_fragment_lists_failed_files_by_path(client):
    conn = client.app.state.context.conn
    photo_id = add_photo(conn, content_hash="ab" * 32, sources=("Pictures/bad.jpg",))
    conn.execute(
        "INSERT INTO jobs(photo_id, stage, status, error, updated_at)"
        " VALUES (?, 'thumbnail', 'failed', 'cannot open', '2026-01-01T00:00:00')",
        (photo_id,),
    )
    body = client.get("/upload/progress").text
    assert "Pictures/bad.jpg" in body
    assert "cannot open" in body


def test_static_assets_are_served(client):
    for asset in ("/static/upload.js", "/static/hash-worker.js"):
        assert client.get(asset).status_code == 200
