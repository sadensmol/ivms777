from pathlib import Path

from embedding.fakes import FakeEmbedder
from embedding.store import write_vector
from embedding.vectors import siglip_probability
from ingest.jobs import enqueue, stage_counts
from ingest.taxonomy import label_prompt, select_dimension_tags, taxonomy_handler
from ingest.vocab import load_vocab, seed_tags
from ingest.worker import drain
from storage.local import LocalStorage
from tests.factories import add_photo
from tests.fixtures import make_jpeg

VOCAB = load_vocab(Path("vocab.yaml"))


def test_select_dimension_tags_picks_argmax_despite_tiny_absolute_probs():
    # Real SigLIP2 calibration: scale~110, bias~-16, best cosine ~0.08. The old
    # absolute-sigmoid floor (>=0.2) tagged nothing because every prob is ~1e-4.
    scale, bias = 110.0, -16.0
    cosines = {"pet": 0.078, "food": 0.046, "document": 0.045}
    assert all(siglip_probability(c, scale, bias) < 0.01 for c in cosines.values())

    logits = [(label, c * scale + bias) for label, c in cosines.items()]
    kept = select_dimension_tags(logits, max_per_dim=3, select_ratio=0.5)

    assert kept[0][0] == "pet"            # the argmax is selected, floor be damned
    assert 0.0 < kept[0][1] <= 1.0        # a real softmax probability
    assert abs(sum(p for _, p in select_dimension_tags(
        logits, max_per_dim=3, select_ratio=0.0)) - 1.0) < 1e-9  # a distribution


def test_select_dimension_tags_keeps_runners_up_within_ratio_and_caps():
    logits = [("a", 5.0), ("b", 4.9), ("c", 4.8), ("d", -5.0)]
    kept = select_dimension_tags(logits, max_per_dim=2, select_ratio=0.5)
    assert [label for label, _ in kept] == ["a", "b"]   # capped at max_per_dim
    kept3 = select_dimension_tags(logits, max_per_dim=5, select_ratio=0.5)
    assert [label for label, _ in kept3] == ["a", "b", "c"]  # d falls below ratio


def test_select_dimension_tags_empty_is_empty():
    assert select_dimension_tags([], max_per_dim=3, select_ratio=0.5) == []


def _photo_with_vector(conn, derived, pid, label_dim, label):
    make_jpeg(derived.local_path(f"{pid}_320.jpg"))  # thumbnail the pixel stats read
    add_photo(conn, photo_id=pid, content_hash=f"{pid:02x}" * 32, thumb_key=f"{pid}_320.jpg")
    # vector == the label's prompt embedding -> that label scores 1.0
    write_vector(conn, pid, FakeEmbedder().embed_texts([label_prompt(label_dim, label)])[0])


def test_backfill_enqueues_taxonomy_for_embedded_photos(conn):
    from ingest.taxonomy import backfill_taxonomy

    add_photo(conn, photo_id=1, content_hash="a", thumb_key="1.jpg", embedding_model="fake")
    add_photo(conn, photo_id=2, content_hash="b", thumb_key="2.jpg")  # not embedded -> skipped
    assert backfill_taxonomy(conn) == 1
    assert stage_counts(conn, "taxonomy")["pending"] == 1
    assert backfill_taxonomy(conn) == 0  # idempotent: the job already exists


def test_taxonomy_assigns_the_matching_label(conn, tmp_path):
    derived = LocalStorage(tmp_path / "thumbs")
    seed_tags(conn, VOCAB)
    _photo_with_vector(conn, derived, 1, "setting", "beach")
    enqueue(conn, 1, "taxonomy")

    drain(conn, {"taxonomy": taxonomy_handler(derived, FakeEmbedder(), VOCAB)})

    rows = conn.execute(
        "SELECT t.dimension, t.label, pt.source FROM photo_tags pt"
        " JOIN tags t ON t.id = pt.tag_id WHERE pt.photo_id = 1"
    ).fetchall()
    pairs = {(r["dimension"], r["label"], r["source"]) for r in rows}
    assert ("setting", "beach", "siglip") in pairs
    assert stage_counts(conn, "taxonomy")["done"] == 1


def test_siglip_scores_are_probabilities_and_the_match_wins_its_dimension(conn, tmp_path):
    derived = LocalStorage(tmp_path / "thumbs")
    seed_tags(conn, VOCAB)
    _photo_with_vector(conn, derived, 1, "setting", "beach")
    enqueue(conn, 1, "taxonomy")
    drain(conn, {"taxonomy": taxonomy_handler(derived, FakeEmbedder(), VOCAB)})
    rows = conn.execute(
        "SELECT t.label, pt.score FROM photo_tags pt JOIN tags t ON t.id = pt.tag_id"
        " WHERE pt.photo_id = 1 AND pt.source = 'siglip' AND t.dimension = 'setting'"
    ).fetchall()
    scored = {r["label"]: r["score"] for r in rows}
    assert set(scored) == {"beach"}          # beach dominates the setting softmax
    assert 0.0 < scored["beach"] <= 1.0      # a softmax probability, not a raw cosine


def test_every_dimension_gets_a_siglip_tag(conn, tmp_path):
    # Regression for the "only 2-3 classes" bug: softmax-per-dimension always
    # yields the best label for every dimension, so a photo is never left with
    # only pixel tags.
    derived = LocalStorage(tmp_path / "thumbs")
    seed_tags(conn, VOCAB)
    _photo_with_vector(conn, derived, 1, "setting", "beach")
    enqueue(conn, 1, "taxonomy")
    drain(conn, {"taxonomy": taxonomy_handler(derived, FakeEmbedder(), VOCAB)})
    tagged = {
        r["dimension"] for r in conn.execute(
            "SELECT DISTINCT t.dimension FROM photo_tags pt JOIN tags t ON t.id = pt.tag_id"
            " WHERE pt.photo_id = 1 AND pt.source = 'siglip'"
        )
    }
    assert tagged == set(VOCAB.dimensions)


def test_taxonomy_writes_pixel_tags_for_palette(conn, tmp_path):
    derived = LocalStorage(tmp_path / "thumbs")
    seed_tags(conn, VOCAB)
    _photo_with_vector(conn, derived, 1, "setting", "beach")
    enqueue(conn, 1, "taxonomy")
    drain(conn, {"taxonomy": taxonomy_handler(derived, FakeEmbedder(), VOCAB)})
    sources = {
        r["source"] for r in conn.execute(
            "SELECT DISTINCT pt.source FROM photo_tags pt WHERE pt.photo_id = 1"
        )
    }
    assert "pixel" in sources  # palette/quality came from pixel stats


def test_taxonomy_reindexes_fts_with_tag_labels(conn, tmp_path):
    derived = LocalStorage(tmp_path / "thumbs")
    seed_tags(conn, VOCAB)
    _photo_with_vector(conn, derived, 1, "setting", "beach")
    enqueue(conn, 1, "taxonomy")
    drain(conn, {"taxonomy": taxonomy_handler(derived, FakeEmbedder(), VOCAB)})
    hit = conn.execute("SELECT rowid FROM photo_fts WHERE photo_fts MATCH 'beach'").fetchone()
    assert hit is not None and hit["rowid"] == 1
