"""Tests for the retriever core (§9.2, plan 12 task 2) — the one pipeline every
feature (search, similar, chat, memory) is meant to call. `candidates()` is the
fast phase (KNN / semantic+keyword fusion, no scoring); `refine()` is the graceful
additive scoring phase; `retrieve() = refine(candidates())`.
"""

from embedding.fakes import FakeEmbedder
from embedding.store import write_caption_vector, write_vector
from inference.fakes import FakeInferenceClient
from search import retriever, signals
from search.retriever import Query, candidates, retrieve
from search.signals import CAPTION_GATE
from tests.factories import add_photo


def _photo(conn, embedder, pid, caption, *, with_caption_vec=True):
    add_photo(conn, photo_id=pid, content_hash=f"h{pid}", thumb_key=f"{pid}.jpg", caption=caption)
    write_vector(conn, pid, embedder.embed_texts([caption])[0])
    if with_caption_vec:
        # caption_vec lives in the dedicated text-embed space (the client's caption
        # model — "fake" here), the SAME space retrieve() embeds the query in (§4/§9).
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
        Query(text="a dog on a beach", k=8, floor=0.5), caption_model="fake",
    )
    assert [r["id"] for r in floored] == [1]  # only the strong match clears the floor


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


# --- (g) card reasons lead with what DROVE the match (same rule as the why-similar table)

def test_reasons_lead_with_what_drove_the_match_not_the_biggest_percent():
    from search.scoring import score_candidates

    # A big percentage is not a big reason. `quality: sharp` agrees at 100% on almost
    # every pair in the library and drives ~0.01 — sorting reasons by match % put it
    # at the top of 403 result cards.
    contribs = [
        {"text": "quality: sharp", "pct": 100, "evidence": 0.012,
         "tag": ("quality", "sharp")},
        {"text": "subject: dog", "pct": 88, "evidence": 0.62, "content": True,
         "tag": ("subject", "dog")},
        {"text": "light: low light", "pct": 95, "evidence": 0.04,
         "tag": ("light", "low light")},
    ]
    [result] = score_candidates({7: contribs})

    assert [r["text"] for r in result["reasons"]] == [
        "subject: dog", "light: low light", "quality: sharp"
    ]


# --- (h) every signal is gated above its measured noise floor -----------------

def test_a_caption_at_the_noise_floor_is_absent_not_weak():
    from search.scoring import caption_contribution, tag_contribution

    # Two RANDOM captions in the real library sit at cosine 0.62. The old bar was
    # 0.60 — BELOW that median — so 64% of all pairs cleared it and the caption
    # scored pure chance. Above its real gate the signal is simply absent.
    assert caption_contribution(0.62) is None
    assert caption_contribution(0.74) is None

    # Crossing the gate is immediately worth half the caption's weight, so admitting
    # a pair actually buys something.
    at_gate = caption_contribution(CAPTION_GATE)
    assert at_gate["pct"] == 50
    tag = tag_contribution("subject", "dog", agreement=0.85, idf=0.35,
                           dimension_weight=signals.TAG_TIERS["subject"].weight)
    strong = caption_contribution(0.9)
    assert strong["evidence"] > at_gate["evidence"] > 0
    assert strong["evidence"] > tag["evidence"]  # a caption that truly means the same leads


def test_style_facets_alone_never_qualify_a_pair():
    from search.scoring import caption_contribution, score_candidates, tag_contribution

    look = tag_contribution("palette", "pastel", agreement=0.9, idf=0.5,
                            dimension_weight=signals.TAG_TIERS["look"].weight)
    where = tag_contribution("setting", "indoor", agreement=0.9, idf=0.5,
                             dimension_weight=signals.TAG_TIERS["where"].weight)
    # However many style/scene facets agree, none of them is CONTENT.
    assert score_candidates({1: [look, where]}) == []
    # One real content signal qualifies the same candidate.
    assert [r["id"] for r in score_candidates({1: [caption_contribution(0.9), look]})] == [1]


def test_a_half_believed_subject_tag_cannot_qualify_a_pair():
    from search.scoring import score_candidates, tag_contribution

    subject_weight = signals.TAG_TIERS["subject"].weight
    # This is the real failure: a girl by a Christmas tree carried `subject: toy` at
    # 0.58 (runner-up `person` at 0.31) and was called similar to a teddy bear.
    guess = tag_contribution("subject", "toy", agreement=0.58, idf=0.41,
                             dimension_weight=subject_weight)
    believed = tag_contribution("subject", "dog", agreement=0.85, idf=0.5,
                                dimension_weight=subject_weight)
    style = tag_contribution("palette", "cool", agreement=0.9, idf=0.5,
                             dimension_weight=signals.TAG_TIERS["look"].weight)

    assert guess is None                                         # below the subject gate
    assert score_candidates({1: [style]}) == []                  # style alone is not content
    assert [r["id"] for r in score_candidates({1: [believed, style]})] == [1]


def test_the_score_is_a_bounded_zero_to_one():
    from search.scoring import caption_contribution, image_contribution, score_candidates

    [result] = score_candidates({1: [image_contribution(0.99), caption_contribution(0.99)]})
    assert 0.0 < result["score"] < 1.0


def test_score_min_drops_results_nothing_really_supports(conn):
    embedder = FakeEmbedder()
    _photo(conn, embedder, 1, "a dog on a beach")
    _photo(conn, embedder, 2, "a plate of pasta")
    client = FakeInferenceClient()
    query = Query(text="a dog on a beach", k=8)

    unfloored = retrieve(conn, embedder, client, 1, query, caption_model="fake")
    weakest = min(r["score"] for r in unfloored)

    floored = Query(text="a dog on a beach", k=8, floor=weakest + 0.01)
    kept = retrieve(conn, embedder, client, 1, floored, caption_model="fake")
    assert len(kept) < len(unfloored)
    assert all(r["score"] >= weakest for r in kept)


def test_caption_below_the_bar_is_absent_from_the_breakdown(conn):
    from search.semantic import similarity_breakdown

    embedder = FakeEmbedder()
    _photo(conn, embedder, 1, "a dog on a beach")
    _photo(conn, embedder, 2, "a plate of pasta")

    params = [
        r["param"]
        for r in similarity_breakdown(conn, 1, 1, 2, caption_min=0.99)
    ]
    assert "caption (meaning)" not in params  # no contribution -> no row (§9.3)
