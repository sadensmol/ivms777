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


def test_upload_offers_a_reprocess_button_per_stage(client):
    body = client.get("/upload").text
    # Every stage gets its own single-stage reprocess (from == to), so re-tagging
    # never rebuilds thumbnails and re-embedding never re-captions.
    for stage in ("thumbnail", "embed", "taxonomy", "caption"):
        assert f'name="from_stage" value="{stage}"' in body
        assert f'name="to_stage" value="{stage}"' in body
    # The caption button confirms first — it alone re-runs the slow vision model.
    # The confirm never quotes a duration in minutes/hours (all UI time is seconds).
    assert "onsubmit=\"return confirm(" in body
    assert "runs the captioner over the whole library" in body
    assert "hours" not in body


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


def test_photo_reprocess_requeues_only_that_photos_stage(client):
    from ingest.jobs import complete, enqueue

    conn = client.app.state.context.conn
    for pid in (1, 2):
        enqueue(conn, pid, "taxonomy")
        complete(conn, pid, "taxonomy")  # both done
    response = client.post(
        "/photo/1/reprocess", data={"stage": "taxonomy", "back": "ctx=library"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert response.headers["location"] == "/photo/1?ctx=library"  # returns to the photo in place
    status = lambda pid: conn.execute(
        "SELECT status FROM jobs WHERE photo_id = ? AND stage = 'taxonomy'", (pid,)
    ).fetchone()["status"]
    assert status(1) == "pending"   # only this photo requeued
    assert status(2) == "done"


def test_photo_reprocess_ignores_static_stages(client):
    # thumbnails/embeddings are static — the endpoint must not requeue them.
    response = client.post(
        "/photo/1/reprocess", data={"stage": "thumbnail"}, follow_redirects=False
    )
    assert response.status_code == 303
    conn = client.app.state.context.conn
    assert conn.execute(
        "SELECT 1 FROM jobs WHERE photo_id = 1 AND stage = 'thumbnail'"
    ).fetchone() is None


def test_photo_reprocess_unknown_photo_is_404(client):
    response = client.post(
        "/photo/999/reprocess", data={"stage": "taxonomy"}, follow_redirects=False
    )
    assert response.status_code == 404
