import pytest
from fastapi.testclient import TestClient

from albums.memory_store import Memory, replace_memories
from tests.factories import add_photo
from web.app import create_app


@pytest.fixture
def client(settings):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    app = create_app(settings)
    conn = app.state.context.conn
    for pid in (1, 2, 3, 4):
        add_photo(conn, photo_id=pid, content_hash=f"h{pid}", thumb_key=f"{pid}.jpg",
                  caption="x", shot_at=f"2025-07-1{pid}T10:00:00")
    replace_memories(conn, 1, [
        Memory("Beach day", "An afternoon by the water.", [1, 2, 3], "sig"),
    ])
    with TestClient(app) as test_client:
        yield test_client


def test_photo_in_a_memory_shows_the_memory_first_and_pages_within_it(client):
    body = client.get("/photo/2?ctx=album:memories::memory-0").text
    assert "Beach day" in body                        # the memory's title, shown first
    assert "An afternoon by the water." in body       # its description
    assert "2 / 3" in body                            # position within the memory
    assert 'href="/photo/1?ctx=album%3Amemories%3A%3Amemory-0"' in body  # prev, in-collection
    assert 'href="/photo/3?ctx=album%3Amemories%3A%3Amemory-0"' in body  # next, in-collection
    assert 'href="/organize?by=memories"' in body     # close returns to the memory grid


def test_last_photo_in_a_memory_has_no_next_into_the_library(client):
    body = client.get("/photo/3?ctx=album:memories::memory-0").text
    assert "photo-nav next" not in body               # 3 is last in the memory
    assert "photo-nav prev" in body                   # but has a previous within it


def test_photo_in_the_library_pages_across_the_library(client):
    body = client.get("/photo/2?ctx=library").text
    assert "In</span> <strong>Library</strong>" in body
    # library default order is capture-date DESC (4,3,2,1); around 2 -> prev 3, next 1
    assert 'href="/photo/3?ctx=library"' in body
    assert 'href="/photo/1?ctx=library"' in body
