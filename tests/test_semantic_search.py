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


def _tag(conn, photo_id, dimension, label, score, source="siglip"):
    conn.execute(
        "INSERT INTO tags(dimension, label) VALUES (?, ?)"
        " ON CONFLICT(dimension, label) DO NOTHING",
        (dimension, label),
    )
    tag_id = conn.execute(
        "SELECT id FROM tags WHERE dimension = ? AND label = ?", (dimension, label)
    ).fetchone()["id"]
    conn.execute(
        "INSERT INTO photo_tags(photo_id, tag_id, score, source) VALUES (?, ?, ?, ?)",
        (photo_id, tag_id, score, source),
    )


def _caption(conn, photo_id, text):
    conn.execute("UPDATE photos SET caption = ? WHERE id = ?", (text, photo_id))
    conn.execute("DELETE FROM photo_fts WHERE rowid = ?", (photo_id,))
    conn.execute(
        "INSERT INTO photo_fts(rowid, caption, tags_text) VALUES (?, ?, '')",
        (photo_id, text),
    )


def _ids(results):
    return [r["id"] for r in results]


def test_similar_requires_a_content_match_style_tags_alone_do_not_qualify(library):
    # 3 shares the SUBJECT (content) — it's similar. 2 shares only a style tag
    # (vibe), which must NOT make it "similar" on its own.
    _tag(library, 1, "subject", "dog", 0.9)
    _tag(library, 3, "subject", "dog", 0.8)
    _tag(library, 1, "vibe", "cozy", 0.9)
    _tag(library, 2, "vibe", "cozy", 0.9)
    by_id = {r["id"]: r for r in similar_photos(library, owner_id=1, photo_id=1, k=5)}
    assert 3 in by_id and 2 not in by_id            # subject qualifies, style-only doesn't
    assert ("subject", "dog") in by_id[3]["tags"]
    assert any("dog" in r["text"] for r in by_id[3]["reasons"])
    assert all(0 <= r["pct"] <= 100 for r in by_id[3]["reasons"])


def test_style_tags_rerank_content_matches(library):
    # A 4th untagged photo keeps subject=dog distinctive (idf > 0). Both 2 and 3
    # share the subject (content) and qualify; 3 also shares a style tag, so it
    # ranks above 2.
    add_photo(library, photo_id=4, content_hash="dd" * 32)
    for pid in (1, 2, 3):
        _tag(library, pid, "subject", "dog", 0.9)
    _tag(library, 1, "vibe", "cozy", 0.9)
    _tag(library, 3, "vibe", "cozy", 0.9)
    ids = _ids(similar_photos(library, owner_id=1, photo_id=1, k=5))
    assert ids[0] == 3 and 2 in ids                 # 3 (subject+style) ranks above 2 (subject)


def test_reasons_are_capped_at_three(library):
    for dim, label in (("subject", "dog"), ("vibe", "cozy"), ("setting", "beach"),
                       ("light", "night"), ("emotion", "joyful")):
        _tag(library, 1, dim, label, 0.9)
        _tag(library, 3, dim, label, 0.9)
    reasons = {r["id"]: r for r in similar_photos(library, owner_id=1, photo_id=1, k=5)}[3]["reasons"]
    assert len(reasons) == 3                         # only the 3 strongest shown
    assert reasons[0]["text"] == "subject: dog"      # most relevant (importance) leads


def test_caption_meaning_connects_photos_tags_missed(library):
    # Neither photo shares a subject tag, but their captions MEAN the same thing
    # (near-identical caption embeddings) — the caption source links them where
    # single-label subject tagging could not.
    from embedding.store import write_caption_vector

    vec = [1.0, 0.0, 0.0, 0.0]
    write_caption_vector(library, 1, vec)
    write_caption_vector(library, 3, vec)  # same meaning as #1
    results = similar_photos(library, owner_id=1, photo_id=1, k=5)
    by_id = {r["id"]: r for r in results}
    assert 3 in by_id and 2 not in by_id            # 2 has no caption vector
    assert any("caption" in r["text"] for r in by_id[3]["reasons"])


def test_similar_of_a_one_off_photo_is_empty(library):
    # No shared tags, no caption, near-orthogonal fake vectors -> nothing similar.
    assert similar_photos(library, owner_id=1, photo_id=1, k=5) == []


def test_similar_of_an_unembedded_photo_is_empty(library):
    # No embedding -> the photo can't be compared to anything yet (tier 0).
    add_photo(library, photo_id=99, content_hash="novec")
    assert similar_photos(library, owner_id=1, photo_id=99, k=5) == []


def test_similarity_breakdown_explains_the_match(library):
    from search.semantic import similarity_breakdown

    _tag(library, 1, "subject", "dog", 0.9)
    _tag(library, 3, "subject", "dog", 0.6)
    _tag(library, 1, "quality", "sharp", 1.0)  # weight 0 -> excluded
    _tag(library, 3, "quality", "sharp", 1.0)
    rows = similarity_breakdown(library, owner_id=1, origin_id=1, current_id=3)
    params = [r["param"] for r in rows]
    assert "subject: dog" in params
    assert "quality: sharp" not in params          # zero-weight dimension is excluded
    assert rows[0]["param"] == "subject: dog"      # sorted by contribution, subject leads
    dog = next(r for r in rows if r["param"] == "subject: dog")
    assert dog["origin"] == "90%" and dog["current"] == "60%"
    assert dog["match"] == 60                        # match is the weaker confidence
    assert any(r["param"] == "visual (image)" for r in rows)  # visual always compared
