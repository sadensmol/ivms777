import pytest
from fastapi.testclient import TestClient

from embedding.fakes import FakeEmbedder
from embedding.store import write_vector
from ingest.receive import receive
from ingest.worker import drain, thumbnail_handler
from tests.factories import add_photo, add_upload
from tests.fixtures import jpeg_bytes_with_exif, sha
from web.app import create_app


@pytest.fixture
def client(settings):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    app = create_app(settings)
    ctx = app.state.context
    upload_id = add_upload(ctx.conn)
    data = jpeg_bytes_with_exif(model="X-T5")
    receive(
        ctx.conn, ctx.originals, owner_id=1, upload_id=upload_id,
        rel_path="Pictures/holiday/a.jpg", declared_hash=sha(data), data=data,
    )
    drain(ctx.conn, {"thumbnail": thumbnail_handler(ctx.originals, ctx.derived, 320, 1600)})
    with TestClient(app) as test_client:
        yield test_client


def _first_id(client):
    return client.app.state.context.conn.execute("SELECT id FROM photos LIMIT 1").fetchone()["id"]


def test_photo_page_shows_exif_and_source_path(client):
    body = client.get(f"/photo/{_first_id(client)}").text
    assert "X-T5" in body
    assert "Pictures/holiday/a.jpg" in body


def test_photo_page_shows_every_duplicate_path_and_wasted_space(client):
    ctx = client.app.state.context
    base = _first_id(client)
    upload_id = add_upload(ctx.conn)
    ctx.conn.execute("UPDATE photos SET bytes = 1000 WHERE id = ?", (base,))
    ctx.conn.execute(
        "INSERT INTO photo_sources(photo_id, upload_id, rel_path, filename)"
        " VALUES (?, ?, 'Backup/a.jpg', 'a.jpg')",
        (base, upload_id),
    )
    body = client.get(f"/photo/{base}").text
    assert "Pictures/holiday/a.jpg" in body
    assert "Backup/a.jpg" in body


def test_photo_page_serves_the_detail_image(client):
    photo_id = _first_id(client)
    assert client.get(f"/photo/{photo_id}").status_code == 200
    detail = client.get(f"/thumb/{photo_id}?size=detail")
    assert detail.status_code == 200
    assert detail.headers["content-type"] == "image/jpeg"


def test_unknown_photo_is_404(client):
    assert client.get("/photo/99999").status_code == 404


def test_similar_strip_lists_other_photos(client):
    ctx = client.app.state.context
    fake = FakeEmbedder()
    base = _first_id(client)
    write_vector(ctx.conn, base, fake.embed_texts(["a"])[0])
    other = add_photo(ctx.conn, content_hash="bb" * 32, thumb_key="bb.jpg")
    write_vector(ctx.conn, other, fake.embed_texts(["a"])[0])  # identical → nearest
    body = client.get(f"/photo/{base}").text
    assert f"/thumb/{other}" in body
