from search.fusion import reciprocal_rank_fusion
from search.keyword import keyword_search
from tests.factories import add_photo


def _fts(conn, pid, caption, tags):
    add_photo(conn, photo_id=pid, content_hash=f"h{pid}", thumb_key=f"{pid}.jpg", caption=caption)
    conn.execute(
        "INSERT INTO photo_fts(rowid, caption, tags_text) VALUES (?, ?, ?)", (pid, caption, tags)
    )


def test_keyword_matches_caption_words(conn):
    _fts(conn, 1, "a birthday cake with candles", "food indoor")
    _fts(conn, 2, "a dog on a beach", "pet beach")
    assert keyword_search(conn, owner_id=1, query="birthday", k=10) == [1]


def test_keyword_matches_tag_text(conn):
    _fts(conn, 1, "no words here", "beach summer")
    assert 1 in keyword_search(conn, owner_id=1, query="beach", k=10)


def test_keyword_is_owner_scoped(conn):
    add_photo(conn, photo_id=1, owner_id=1, content_hash="a", thumb_key="a.jpg", caption="cake")
    add_photo(conn, photo_id=2, owner_id=2, content_hash="b", thumb_key="b.jpg", caption="cake")
    conn.execute("INSERT INTO photo_fts(rowid, caption, tags_text) VALUES (1, 'cake', '')")
    conn.execute("INSERT INTO photo_fts(rowid, caption, tags_text) VALUES (2, 'cake', '')")
    assert keyword_search(conn, owner_id=1, query="cake", k=10) == [1]


def test_keyword_empty_query_returns_nothing(conn):
    _fts(conn, 1, "a cake", "food")
    assert keyword_search(conn, owner_id=1, query="  ", k=10) == []


def test_fusion_prefers_items_ranked_well_by_both():
    fused = reciprocal_rank_fusion([[1, 2, 3], [2, 1, 4]])
    assert fused[0] in (1, 2)
    assert set(fused) == {1, 2, 3, 4}


def test_fusion_ignores_empty_rankings():
    assert reciprocal_rank_fusion([[], [5, 6]]) == [5, 6]
