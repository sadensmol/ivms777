from embedding.store import write_caption_vector
from embedding.vectors import l2_normalize
from inference.fakes import FakeInferenceClient
from search.rerank import rerank, rerank_llm
from tests.factories import add_photo


def _cap(conn, pid, text):
    add_photo(conn, photo_id=pid, content_hash=f"h{pid}", thumb_key=f"{pid}.jpg", caption=text)
    # caption vectors live in the inference-client embed space (caption_embed_model).
    write_caption_vector(conn, pid, l2_normalize(FakeInferenceClient().embed("fake", [text])[0]))


def _qvec(text):
    return l2_normalize(FakeInferenceClient().embed("fake", [text])[0])


def test_matches_rank_above_non_matches(conn):
    _cap(conn, 1, "a dog on a beach")
    _cap(conn, 2, "a plate of pasta")
    ranked = rerank(conn, _qvec("a dog on a beach"), [1, 2], floor=0.0)
    assert ranked[0][0] == 1


def test_floor_drops_everything_when_nothing_matches(conn):
    _cap(conn, 1, "a plate of pasta")
    assert rerank(conn, _qvec("a dog on a beach"), [1], floor=0.9) == []


def test_photo_without_caption_vector_is_kept(conn):
    # A candidate with no caption vector has an UNAVAILABLE signal, not a bad one:
    # it must survive rerank (the agent revalidates it) instead of being floored.
    add_photo(conn, photo_id=3, content_hash="h3", thumb_key="3.jpg", caption=None)
    assert [pid for pid, _ in rerank(conn, _qvec("anything"), [3], floor=0.01)] == [3]


def test_scored_match_ranks_above_kept_unscored(conn):
    # Photos WITH a caption vector are ordered by caption-meaning cosine; photos
    # WITHOUT one are kept but sink below the scored matches, in input order.
    _cap(conn, 1, "a dog on a beach")
    add_photo(conn, photo_id=2, content_hash="h2", thumb_key="2.jpg", caption=None)
    ranked = [pid for pid, _ in rerank(conn, _qvec("a dog on a beach"), [2, 1], floor=0.3)]
    assert ranked == [1, 2]


def test_llm_rerank_keeps_only_high_scores(conn):
    for pid in (1, 2):
        add_photo(conn, photo_id=pid, content_hash=f"h{pid}", thumb_key=f"{pid}.jpg", caption="x")
    client = FakeInferenceClient(['{"1": 3, "2": 0}'])
    assert rerank_llm(client, "m", conn, "a dog", [1, 2], keep_min=2) == [1]
