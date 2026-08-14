import pytest
from fastapi.testclient import TestClient

from ingest.receive import receive
from ingest.worker import drain, thumbnail_handler
from tests.factories import add_upload
from tests.fixtures import jpeg_bytes, sha
from web.app import create_app


@pytest.fixture
def client(settings):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    app = create_app(settings)
    ctx = app.state.context
    upload_id = add_upload(ctx.conn)
    for index in range(3):
        data = jpeg_bytes(color=["red", "green", "blue"][index])
        receive(
            ctx.conn, ctx.originals, owner_id=settings.owner_id, upload_id=upload_id,
            rel_path=f"photo{index}.jpg", declared_hash=sha(data), data=data,
        )
    drain(ctx.conn, {"thumbnail": thumbnail_handler(ctx.originals, ctx.derived, 320, 1600)})
    with TestClient(app) as test_client:
        yield test_client


def test_root_redirects_to_library(client):
    response = client.get("/", follow_redirects=False)
    assert response.status_code in (302, 307)
    assert response.headers["location"] == "/library"


def test_library_page_renders_a_tile_per_photo(client):
    response = client.get("/library")
    assert response.status_code == 200
    assert response.text.count('class="tile"') == 3


def test_grid_page_returns_a_fragment_not_a_full_document(client):
    response = client.get("/library/page?offset=0")
    assert response.status_code == 200
    assert "<html" not in response.text
    assert 'class="tile"' in response.text


def test_grid_page_respects_offset(client):
    response = client.get("/library/page?offset=3")
    assert 'class="tile"' not in response.text


def test_thumbnail_is_served_as_jpeg(client):
    photo_id = client.app.state.context.conn.execute(
        "SELECT id FROM photos LIMIT 1"
    ).fetchone()["id"]
    response = client.get(f"/thumb/{photo_id}")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content[:2] == b"\xff\xd8"


def test_unknown_thumbnail_is_404(client):
    assert client.get("/thumb/99999").status_code == 404


def test_a_photo_from_several_paths_shows_a_badge_and_one_tile(client):
    ctx = client.app.state.context
    upload_id = add_upload(ctx.conn)
    data = jpeg_bytes(color="teal")
    for rel_path in ("Pictures/a.jpg", "Desktop/a copy.jpg"):
        receive(
            ctx.conn, ctx.originals, owner_id=ctx.settings.owner_id, upload_id=upload_id,
            rel_path=rel_path, declared_hash=sha(data), data=data,
        )
    drain(ctx.conn, {"thumbnail": thumbnail_handler(ctx.originals, ctx.derived, 320, 1600)})

    response = client.get("/library")
    # 3 originals + 1 new distinct photo = 4 tiles, and the new one wears ×2.
    assert response.text.count('class="tile"') == 4
    assert "×2" in response.text
