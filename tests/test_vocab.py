from pathlib import Path

from ingest.vocab import load_vocab, seed_tags, tag_id_map

VOCAB = Path("vocab.yaml")


def test_vocab_has_the_ten_dimensions():
    vocab = load_vocab(VOCAB)
    assert set(vocab.dimensions) == {
        "subject", "setting", "vibe", "emotion", "light",
        "season_weather", "composition", "palette", "occasion", "quality",
    }
    assert "beach" in vocab.dimensions["setting"]


def test_seed_tags_is_idempotent(conn):
    vocab = load_vocab(VOCAB)
    seed_tags(conn, vocab)
    first = conn.execute("SELECT count(*) AS n FROM tags").fetchone()["n"]
    seed_tags(conn, vocab)  # second run adds nothing
    assert conn.execute("SELECT count(*) AS n FROM tags").fetchone()["n"] == first
    assert first == sum(len(v) for v in vocab.dimensions.values())


def test_tag_id_map_keys_by_dimension_and_label(conn):
    vocab = load_vocab(VOCAB)
    seed_tags(conn, vocab)
    ids = tag_id_map(conn)
    assert ("setting", "beach") in ids
    assert isinstance(ids[("setting", "beach")], int)


def test_thresholds_default_when_unspecified():
    vocab = load_vocab(VOCAB)
    assert 0.0 < vocab.threshold("subject") <= 1.0
