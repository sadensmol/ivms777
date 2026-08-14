import pytest
from fastapi.testclient import TestClient

from embedding.fakes import FakeEmbedder
from embedding.store import write_vector
from tests.factories import add_photo
from web.app import create_app


@pytest.fixture
def client(settings):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    app = create_app(settings)
    fake = FakeEmbedder()
    for pid, word in ((1, "beach"), (2, "keyboard")):
        add_photo(app.state.context.conn, photo_id=pid, content_hash=word, thumb_key=f"{word}.jpg")
        write_vector(app.state.context.conn, pid, fake.embed_texts([word])[0])
    with TestClient(app) as test_client:
        yield test_client


def test_search_box_is_on_the_library_page(client):
    assert 'name="q"' in client.get("/library").text


def test_search_returns_the_matching_photo_first(client):
    body = client.get("/library?q=beach").text
    assert "/thumb/1" in body
    assert body.index("/thumb/1") < body.index("/thumb/2")


def test_keyword_match_surfaces_via_fusion(client):
    # A caption-only match (no close vector) still appears, proving keyword feeds fusion.
    conn = client.app.state.context.conn
    pid = add_photo(conn, content_hash="zz" * 32, thumb_key="zz.jpg", caption="Zermatt ski trip")
    write_vector(conn, pid, FakeEmbedder().embed_texts(["unrelated subject"])[0])
    conn.execute(
        "INSERT INTO photo_fts(rowid, caption, tags_text) VALUES (?, 'Zermatt ski trip', '')", (pid,)
    )
    assert f"/thumb/{pid}" in client.get("/library?q=Zermatt").text


def test_duplicates_toggle_is_offered(client):
    assert "dupes=1" in client.get("/library").text


def test_duplicates_filter_shows_only_multi_source_photos(client):
    conn = client.app.state.context.conn
    dup = add_photo(conn, content_hash="dd" * 32, thumb_key="dd.jpg",
                    sources=("Pictures/a.jpg", "Backup/a.jpg"))
    body = client.get("/library?dupes=1").text
    assert f"/thumb/{dup}" in body
    assert "/thumb/1" not in body  # the beach photo has a single source
