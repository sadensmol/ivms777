import pytest
from fastapi.testclient import TestClient

from tests.factories import add_photo
from web.app import create_app


@pytest.fixture
def client(settings):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    app = create_app(settings)
    conn = app.state.context.conn
    add_photo(conn, photo_id=1, content_hash="h1", thumb_key="1.jpg",
              shot_at="2025-07-12T09:00:00", camera="X-T5")
    add_photo(conn, photo_id=2, content_hash="h2", thumb_key="2.jpg",
              shot_at="2025-07-12T09:10:00", camera="X-T5")
    with TestClient(app) as test_client:
        yield test_client


def test_organize_tab_is_in_the_nav(client):
    assert 'href="/organize"' in client.get("/library").text


def test_organize_offers_all_four_types_in_the_dropdown(client):
    body = client.get("/organize").text
    for label in ("By date", "By similarity", "By camera", "By place"):
        assert label in body


def test_default_organization_is_by_date_with_a_described_album(client):
    body = client.get("/organize").text
    assert "12 Jul 2025" in body
    assert "2 photos" in body  # the album description


def test_choosing_an_organization_type_shows_its_albums(client):
    body = client.get("/organize?by=camera").text
    assert "X-T5" in body
    assert "taken with X-T5" in body


def test_unknown_type_falls_back_to_default(client):
    assert client.get("/organize?by=nonsense").status_code == 200
