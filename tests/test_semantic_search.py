import pytest

from embedding.fakes import FakeEmbedder
from embedding.store import write_vector
from search.semantic import search_photos, similar_photos
from tests.factories import add_photo


@pytest.fixture
def library(conn):
    # Give each photo the fake's embedding for a distinct word, so a text query
    # for that word retrieves it exactly. This tests the plumbing; real ranking
    # quality is the slow SigLIP test.
    fake = FakeEmbedder()
    for pid, word in ((1, "beach"), (2, "keyboard"), (3, "mountain")):
        add_photo(conn, photo_id=pid, content_hash=word, thumb_key=f"{word}.jpg")
        write_vector(conn, pid, fake.embed_texts([word])[0])
    return conn


def test_search_finds_the_photo_whose_vector_matches_the_query(library):
    ids = search_photos(library, FakeEmbedder(), owner_id=1, query="beach", k=3)
    assert ids[0] == 1


def test_search_returns_at_most_k(library):
    assert len(search_photos(library, FakeEmbedder(), owner_id=1, query="beach", k=2)) == 2


def test_empty_query_returns_nothing(library):
    assert search_photos(library, FakeEmbedder(), owner_id=1, query="   ", k=3) == []


def test_similar_excludes_the_photo_itself(library):
    ids = similar_photos(library, owner_id=1, photo_id=1, k=5)
    assert 1 not in ids


def test_similar_of_an_unembedded_photo_is_empty(library):
    add_photo(library, photo_id=99, content_hash="novec")
    assert similar_photos(library, owner_id=1, photo_id=99, k=5) == []
