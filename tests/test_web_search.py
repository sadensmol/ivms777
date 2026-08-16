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


def test_library_search_releases_the_search_lease(client):
    # SigLIP-embed work in the library read-path is coordinated (§8.1 FIX I1):
    # a search request takes the SEARCH lease for the query and releases it
    # before the response is returned, never leaving SigLIP resident uncoordinated.
    from models import lease_store as ls

    client.get("/library?q=beach")
    assert ls.read_lease(client.app.state.context.conn) is None


def test_facet_filter_narrows_search_results(client):
    # Candidate generation now goes through search/retriever.py's candidates(), but
    # the hard EXIF-facet narrowing (_filter_where) must still apply exactly: a
    # facet-mismatched photo is removed even though it matched the text query.
    conn = client.app.state.context.conn
    matching = add_photo(conn, content_hash="cc" * 32, thumb_key="cc.jpg")
    write_vector(conn, matching, FakeEmbedder().embed_texts(["beach"])[0])
    conn.execute(
        "INSERT INTO photo_facets(photo_id, key, value_text) VALUES (?, 'camera_model', 'X-T5')",
        (matching,),
    )
    # planned=1 skips the planner redirect (§9.1), which would otherwise rebuild
    # the param set from a planner spec and drop the explicit f_camera_model.
    body = client.get("/library?q=beach&f_camera_model=X-T5&planned=1").text
    assert f"/thumb/{matching}" in body
    assert "/thumb/1" not in body  # photo 1 also matches "beach" but has no camera_model
