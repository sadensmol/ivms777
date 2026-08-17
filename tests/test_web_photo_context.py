import pytest
from fastapi.testclient import TestClient

from albums.memory_store import Memory, replace_memories
from chat.history import add_message, current_session
from embedding.fakes import FakeEmbedder
from embedding.store import write_vector
from tests.factories import add_photo
from web.app import create_app


def test_chat_history_citations_carry_the_chat_ctx(settings):
    # A cited thumbnail must open the photo as a leaf of the CHAT grid (§13.1),
    # so close returns to the conversation — not the library.
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    app = create_app(settings)
    conn = app.state.context.conn
    add_photo(conn, photo_id=2, content_hash="h2", thumb_key="2.jpg", caption="a dog")
    add_message(conn, current_session(conn, 1), "find a dog", "here [photo:2]", [2])
    with TestClient(app) as tc:
        body = tc.get("/chat").text
    assert 'href="/photo/2?ctx=chat"' in body


def test_cited_photo_from_chat_closes_back_to_the_conversation(settings):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    app = create_app(settings)
    conn = app.state.context.conn
    add_photo(conn, photo_id=2, content_hash="h2", thumb_key="2.jpg", caption="a dog")
    add_message(conn, current_session(conn, 1), "find a dog", "here [photo:2]", [2])
    with TestClient(app) as tc:
        body = tc.get("/photo/2?ctx=chat").text
    assert 'class="photo-close" href="/chat"' in body   # close goes UP to the conversation
    assert "In</span> <strong>Chat</strong>" in body    # the grid identity is the chat


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


def test_photo_in_a_memory_shows_a_collage_of_all_its_members(client):
    body = client.get("/photo/2?ctx=album:memories::memory-0").text
    assert 'class="collection-collage"' in body
    for pid in (1, 2, 3):                              # every member thumbnail present
        assert f'src="/thumb/{pid}"' in body
    assert "collage-item current" in body             # the open photo (2) is highlighted
    # collage links carry the memory ctx forward and page in place
    assert 'href="/photo/1?ctx=album%3Amemories%3A%3Amemory-0"' in body


def test_memory_shown_in_chat_pages_within_it_but_closes_to_chat(client):
    # A memory opened from chat is the SAME grid as in Organize — it pages within
    # the memory's photos and shows the whole-memory collage — but "close" returns
    # UP to the conversation, not to /organize (§10, §13.1).
    body = client.get("/photo/2?ctx=chat-memory:memory-0").text
    assert "Beach day" in body                          # the memory identity, shown first
    assert "2 / 3" in body                              # position within the memory
    assert 'href="/photo/1?ctx=chat-memory%3Amemory-0"' in body  # prev stays in the memory
    assert 'href="/photo/3?ctx=chat-memory%3Amemory-0"' in body  # next stays in the memory
    assert 'class="photo-close" href="/chat"' in body   # close goes back to the conversation
    assert 'class="collection-collage"' in body         # bounded set → whole-memory collage


def test_memory_shown_in_chat_excludes_members_from_similar(settings):
    # Same member-exclusion as the Organize memory leaf: a photo already in the
    # collage is not repeated in the "similar" strip (§13).
    from embedding.fakes import FakeEmbedder
    from embedding.store import write_vector
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    app = create_app(settings)
    conn = app.state.context.conn
    for pid in (1, 2, 3, 4):
        add_photo(conn, photo_id=pid, content_hash=f"h{pid}", thumb_key=f"{pid}.jpg", caption="x")
    write_vector(conn, 1, FakeEmbedder().embed_texts(["dog"])[0])
    conn.execute("INSERT OR IGNORE INTO tags(dimension, label) VALUES ('subject', 'dog')")
    tid = conn.execute(
        "SELECT id FROM tags WHERE dimension = 'subject' AND label = 'dog'"
    ).fetchone()["id"]
    for pid in (1, 2, 4):
        conn.execute(
            "INSERT INTO photo_tags(photo_id, tag_id, score, source) VALUES (?, ?, 1.0, 'siglip')",
            (pid, tid),
        )
    replace_memories(conn, 1, [Memory("Beach day", "desc", [1, 2, 3], "sig")])
    with TestClient(app) as tc:
        body = tc.get("/photo/1/similar?ctx=chat-memory:memory-0").text
    assert 'href="/photo/4?ctx=similar:1"' in body      # non-member similar stays
    assert 'href="/photo/2?ctx=similar:1"' not in body  # member is not repeated


def test_stale_chat_memory_ctx_falls_back_to_the_library(client):
    # If the memory no longer exists (rebuilt away), the leaf degrades to the
    # library rather than 500ing (§13.1 fallback).
    body = client.get("/photo/2?ctx=chat-memory:memory-99").text
    assert "In</span> <strong>Library</strong>" in body


def test_library_photo_has_no_member_collage(client):
    # The library/search collection is unbounded, so no whole-collection collage.
    body = client.get("/photo/2?ctx=library").text
    assert 'class="collection-collage"' not in body


def test_similar_strip_excludes_photos_already_in_the_memory(settings):
    # A photo that is both "similar" and a memory member must not repeat in the
    # similar strip — the collage already shows it (§13).
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    app = create_app(settings)
    conn = app.state.context.conn
    for pid in (1, 2, 3, 4):
        add_photo(conn, photo_id=pid, content_hash=f"h{pid}", thumb_key=f"{pid}.jpg", caption="x")
    write_vector(conn, 1, FakeEmbedder().embed_texts(["dog"])[0])  # origin needs a vector
    conn.execute("INSERT OR IGNORE INTO tags(dimension, label) VALUES ('subject', 'dog')")
    tid = conn.execute(
        "SELECT id FROM tags WHERE dimension = 'subject' AND label = 'dog'"
    ).fetchone()["id"]
    for pid in (1, 2, 4):  # 2 is a member, 4 is not — both share the subject 'dog'
        conn.execute(
            "INSERT INTO photo_tags(photo_id, tag_id, score, source) VALUES (?, ?, 1.0, 'siglip')",
            (pid, tid),
        )
    replace_memories(conn, 1, [Memory("Beach day", "desc", [1, 2, 3], "sig")])
    with TestClient(app) as tc:
        # The strip loads via the async fragment route now, carrying ctx forward
        # so the member-exclusion matches the main /photo route exactly.
        body = tc.get("/photo/1/similar?ctx=album:memories::memory-0").text
    assert 'href="/photo/4?ctx=similar:1"' in body      # non-member similar stays
    assert 'href="/photo/2?ctx=similar:1"' not in body  # member is not repeated as "similar"


def test_static_assets_are_cache_busted(client):
    # app.css / chat.js carry a ?v=<mtime> so a browser never serves a stale copy
    # after a rebuild (this is why CSS/JS changes "didn't show" before).
    body = client.get("/photo/1?ctx=library").text
    assert "/static/app.css?v=" in body


def test_photo_in_the_library_pages_across_the_library(client):
    body = client.get("/photo/2?ctx=library").text
    assert "In</span> <strong>Library</strong>" in body
    # library default order is capture-date DESC (4,3,2,1); around 2 -> prev 3, next 1
    assert 'href="/photo/3?ctx=library"' in body
    assert 'href="/photo/1?ctx=library"' in body
