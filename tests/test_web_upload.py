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


def test_progress_shows_per_stage_throughput(client):
    # Two captions finished 1s apart -> 1.0/s, rendered on the caption row.
    conn = client.app.state.context.conn
    for pid, when in ((1, "2026-01-01T00:00:00"), (2, "2026-01-01T00:00:01")):
        add_photo(conn, photo_id=pid, content_hash=f"h{pid}", thumb_key=f"{pid}.jpg")
        conn.execute(
            "INSERT INTO jobs(photo_id, stage, status, updated_at)"
            " VALUES (?, 'caption', 'done', ?)",
            (pid, when),
        )
    assert "1.0/s" in client.get("/upload/progress").text


def test_progress_shows_slow_stage_in_seconds_never_per_minute(client):
    # A slow stage (two captions 60s apart -> 0.0167/s) must still read per SECOND —
    # never "/min" or "/hour". The label keeps decimals so it is not a bare 0.0/s.
    conn = client.app.state.context.conn
    for pid, when in ((1, "2026-01-01T00:00:00"), (2, "2026-01-01T00:01:00")):
        add_photo(conn, photo_id=pid, content_hash=f"h{pid}", thumb_key=f"{pid}.jpg")
        conn.execute(
            "INSERT INTO jobs(photo_id, stage, status, updated_at)"
            " VALUES (?, 'caption', 'done', ?)",
            (pid, when),
        )
    body = client.get("/upload/progress").text
    assert "0.017/s" in body
    assert "/min" not in body and "/hour" not in body


def test_upload_page_remembers_the_current_folder_across_restarts(client):
    conn = client.app.state.context.conn
    # A completed upload from a folder — the persisted `uploads` row is what a
    # restart still has (the browser file picker cannot remember a selection).
    conn.execute(
        "INSERT INTO uploads(owner_id, root_label, started_at, files_sent)"
        " VALUES (1, 'Summer 2024', '2026-01-01T00:00:00', 3)"
    )
    body = client.get("/upload").text
    assert "Summer 2024" in body and "Delete from library" in body  # listed, deletable


def test_delete_folder_endpoint_enqueues_an_outbox_deletion(client):
    conn = client.app.state.context.conn
    conn.execute(
        "INSERT INTO uploads(owner_id, root_label, started_at, files_sent)"
        " VALUES (1, 'Old trip', '2026-01-01T00:00:00', 5)"
    )
    response = client.post(
        "/upload/folder/delete", data={"root_label": "Old trip"}, follow_redirects=False
    )
    assert response.status_code == 303
    assert conn.execute(
        "SELECT 1 FROM folder_deletions WHERE root_label = 'Old trip'"
    ).fetchone() is not None


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
