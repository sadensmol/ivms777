from chat.context import build_context
from tests.factories import add_photo


def test_block_carries_id_caption_and_facts(conn):
    pid = add_photo(
        conn, content_hash="a" * 64, thumb_key="a.jpg",
        caption="A dog on a beach", shot_at="2025-06-14T18:30:00", camera="Canon R6",
    )
    conn.execute(
        "INSERT INTO photo_facets(photo_id, key, value_text) VALUES (?, 'lens', '50mm')",
        (pid,),
    )
    block = build_context(conn, [pid])
    assert f"[photo:{pid}]" in block
    assert "A dog on a beach" in block
    assert "Canon R6" in block
    assert "50mm" in block


def test_empty_list_is_a_no_match_sentinel(conn):
    assert build_context(conn, []) == "No photos matched."
