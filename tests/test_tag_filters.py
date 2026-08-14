from pathlib import Path

from ingest.vocab import load_vocab, seed_tags
from search.tags import parse_tag_filters, tag_sidebar, tag_where
from tests.factories import add_photo

VOCAB = load_vocab(Path("vocab.yaml"))


def _tagged(conn, pid, pairs):
    add_photo(conn, photo_id=pid, content_hash=f"h{pid}", thumb_key=f"{pid}.jpg")
    ids = {(r["dimension"], r["label"]): r["id"] for r in conn.execute("SELECT * FROM tags")}
    for dim, label in pairs:
        conn.execute(
            "INSERT INTO photo_tags(photo_id, tag_id, score, source) VALUES (?, ?, 0.9, 'siglip')",
            (pid, ids[(dim, label)]),
        )


def test_parse_reads_t_prefixed_params():
    assert parse_tag_filters({"t_setting": "beach,forest", "q": "x"}) == {"setting": ["beach", "forest"]}


def test_tag_where_filters_by_label(conn):
    seed_tags(conn, VOCAB)
    _tagged(conn, 1, [("setting", "beach")])
    _tagged(conn, 2, [("setting", "forest")])
    where, params = tag_where({"setting": ["beach"]}, score_min=0.2)
    rows = conn.execute(
        "SELECT p.id FROM photos p WHERE p.owner_id = ?" + where, (1, *params)
    ).fetchall()
    assert [r["id"] for r in rows] == [1]


def test_tag_where_ands_across_dimensions(conn):
    seed_tags(conn, VOCAB)
    _tagged(conn, 1, [("setting", "beach"), ("vibe", "serene")])
    _tagged(conn, 2, [("setting", "beach")])
    where, params = tag_where({"setting": ["beach"], "vibe": ["serene"]}, score_min=0.2)
    rows = conn.execute(
        "SELECT p.id FROM photos p WHERE p.owner_id = ?" + where, (1, *params)
    ).fetchall()
    assert [r["id"] for r in rows] == [1]


def test_tag_where_ors_within_a_dimension(conn):
    seed_tags(conn, VOCAB)
    _tagged(conn, 1, [("setting", "beach")])
    _tagged(conn, 2, [("setting", "forest")])
    where, params = tag_where({"setting": ["beach", "forest"]}, score_min=0.2)
    rows = conn.execute(
        "SELECT p.id FROM photos p WHERE p.owner_id = ?" + where + " ORDER BY p.id", (1, *params)
    ).fetchall()
    assert [r["id"] for r in rows] == [1, 2]


def test_sidebar_counts_labels(conn):
    seed_tags(conn, VOCAB)
    _tagged(conn, 1, [("setting", "beach")])
    _tagged(conn, 2, [("setting", "beach")])
    groups = tag_sidebar(conn, owner_id=1, dimensions=["setting"], score_min=0.2, limit=12)
    setting = next(g for g in groups if g["dimension"] == "setting")
    assert ("beach", 2) in [(v["label"], v["count"]) for v in setting["values"]]
