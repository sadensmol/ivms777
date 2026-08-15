"""Tests for the retriever core (§9.2, plan 12 task 2) — the one pipeline every
feature (search, similar, chat, memory) is meant to call. `candidates()` is the
fast phase (KNN / semantic+keyword fusion, no scoring); `refine()` is the graceful
additive scoring phase; `retrieve() = refine(candidates())`.
"""

from embedding.fakes import FakeEmbedder
from embedding.store import write_caption_vector, write_vector
from inference.fakes import FakeInferenceClient
from search import retriever
from search.retriever import Query, candidates, retrieve
from tests.factories import add_photo


def _photo(conn, embedder, pid, caption, *, with_caption_vec=True):
    add_photo(conn, photo_id=pid, content_hash=f"h{pid}", thumb_key=f"{pid}.jpg", caption=caption)
    write_vector(conn, pid, embedder.embed_texts([caption])[0])
    if with_caption_vec:
        write_caption_vector(conn, pid, FakeInferenceClient().embed("fake", [caption])[0])


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


def _facet(conn, photo_id, key, value_text):
    conn.execute(
        "INSERT INTO photo_facets(photo_id, key, value_text) VALUES (?, ?, ?)",
        (photo_id, key, value_text),
    )


# --- (a) text query ranks a matching photo above a non-matching one ----------

def test_text_query_ranks_matching_photo_first_with_reasons(conn):
    embedder = FakeEmbedder()
    _photo(conn, embedder, 1, "a dog on a beach")
    _photo(conn, embedder, 2, "a plate of pasta")
    client = FakeInferenceClient()
    query = Query(text="a dog on a beach", k=8)

    results = retrieve(conn, embedder, client, 1, query, caption_model="fake")

    assert results[0]["id"] == 1
    assert results[0]["reasons"]


# --- (b) missing caption_vec drops only that contribution, not the candidate --

def test_missing_caption_vector_is_ranked_not_dropped(conn):
    embedder = FakeEmbedder()
    _photo(conn, embedder, 1, "a dog on a beach", with_caption_vec=False)
    client = FakeInferenceClient()
    query = Query(text="a dog on a beach", k=8)

    results = retrieve(conn, embedder, client, 1, query, caption_model="fake")

    assert [r["id"] for r in results] == [1]  # image/fused-rank contribution keeps it


# --- (c) a hard EXIF facet removes non-matching photos before scoring ---------

def test_hard_facet_filter_removes_non_matching_photos(conn):
    embedder = FakeEmbedder()
    _photo(conn, embedder, 1, "a dog on a beach")
    _photo(conn, embedder, 2, "a dog on a beach")
    _facet(conn, 1, "camera_model", "X-T5")
    _facet(conn, 2, "camera_model", "iPhone")
    client = FakeInferenceClient()
    query = Query(text="a dog on a beach", k=8, hard_filters={"f_camera_model": "X-T5"})

    results = retrieve(conn, embedder, client, 1, query, caption_model="fake")

    assert [r["id"] for r in results] == [1]


# --- (d) a soft tag on an untagged library does not empty the result ----------

def test_soft_tag_with_no_matching_taxonomy_does_not_empty_result(conn):
    embedder = FakeEmbedder()
    _photo(conn, embedder, 1, "a dog on a beach")  # no tags at all
    client = FakeInferenceClient()
    query = Query(text="a dog on a beach", k=8, soft_tags={"subject": ["dog"]})

    results = retrieve(conn, embedder, client, 1, query, caption_model="fake")

    assert [r["id"] for r in results] == [1]


def test_soft_tag_match_boosts_a_tagged_candidate(conn):
    embedder = FakeEmbedder()
    _photo(conn, embedder, 1, "a dog on a beach")
    _photo(conn, embedder, 2, "a dog on a beach")
    _tag(conn, 1, "subject", "dog", 0.9)
    client = FakeInferenceClient()
    query = Query(text="a dog on a beach", k=8, soft_tags={"subject": ["dog"]})

    results = retrieve(conn, embedder, client, 1, query, caption_model="fake")
    by_id = {r["id"]: r for r in results}

    assert by_id[1]["score"] > by_id[2]["score"]


# --- (e) floor cuts only when set ----------------------------------------------

def test_floor_cuts_only_when_set(conn):
    embedder = FakeEmbedder()
    _photo(conn, embedder, 1, "a dog on a beach", with_caption_vec=False)
    _photo(conn, embedder, 2, "a plate of pasta", with_caption_vec=False)
    client = FakeInferenceClient()

    unfloored = retrieve(
        conn, embedder, client, 1, Query(text="a dog on a beach", k=8), caption_model="fake"
    )
    assert {r["id"] for r in unfloored} == {1, 2}  # None = rank everyone, no cut

    floored = retrieve(
        conn, embedder, client, 1,
        Query(text="a dog on a beach", k=8, floor=0.8), caption_model="fake",
    )
    assert [r["id"] for r in floored] == [1]  # only the strong match clears 0.8


# --- (f) seed perf guarantee: candidates() never touches text-query machinery -

def test_seed_candidates_never_touch_text_query_machinery(conn, monkeypatch):
    embedder = FakeEmbedder()
    add_photo(conn, photo_id=1, content_hash="h1", thumb_key="1.jpg")
    write_vector(conn, 1, embedder.embed_texts(["seed"])[0])
    add_photo(conn, photo_id=2, content_hash="h2", thumb_key="2.jpg")
    write_vector(conn, 2, embedder.embed_texts(["other"])[0])

    def _boom(*args, **kwargs):
        raise AssertionError("seed candidates() must never touch text-query machinery")

    # candidates() takes no `client` at all (see search/retriever.py), so
    # client.embed structurally cannot be reached from a seed query; here we
    # additionally prove the text encoder and the fused search primitives (the
    # other text-query machinery) are never touched either.
    monkeypatch.setattr(embedder, "embed_texts", _boom)
    monkeypatch.setattr(retriever, "search_photos", _boom)
    monkeypatch.setattr(retriever, "keyword_search", _boom)

    ids = candidates(conn, embedder, 1, Query(seed_photo_id=1, k=8))

    assert ids == [2]
