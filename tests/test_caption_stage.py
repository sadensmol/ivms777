import json
from pathlib import Path

from inference.fakes import FakeInferenceClient
from ingest.caption import caption_handler
from ingest.jobs import enqueue, stage_counts
from ingest.thumbs import thumb_key
from ingest.vocab import load_vocab, seed_tags
from ingest.worker import drain
from storage.local import LocalStorage
from tests.factories import add_photo
from tests.fixtures import make_jpeg

VOCAB = load_vocab(Path("vocab.yaml"))
DIMS = list(VOCAB.dimensions)


def _photo_with_detail_thumb(conn, derived, pid, hash_hex, detail_px=1600):
    make_jpeg(derived.local_path(thumb_key(hash_hex, detail_px)))  # the image the VLM reads
    add_photo(conn, photo_id=pid, content_hash=hash_hex, thumb_key=thumb_key(hash_hex, 320))


def test_caption_stage_writes_caption_title_description_and_vlm_tags(conn, tmp_path):
    derived = LocalStorage(tmp_path / "thumbs")
    seed_tags(conn, VOCAB)
    _photo_with_detail_thumb(conn, derived, 1, "aa" * 32)
    client = FakeInferenceClient([json.dumps({
        "caption": "a dog on a beach", "title": "Beach day",
        "description": "A dog runs on the sand.",
        "tags": {"subject": ["pet"], "setting": ["beach"]},
    })])
    enqueue(conn, 1, "caption")

    drain(conn, {"caption": caption_handler(derived, client, "fake-vlm", DIMS, 1600)})

    row = conn.execute(
        "SELECT caption, caption_model, ai_title, ai_description FROM photos WHERE id = 1"
    ).fetchone()
    assert row["caption"] == "a dog on a beach"
    assert row["ai_title"] == "Beach day"
    assert "sand" in row["ai_description"]
    assert row["caption_model"] == "fake-vlm"
    vlm = {
        (r["dimension"], r["label"]) for r in conn.execute(
            "SELECT t.dimension, t.label FROM photo_tags pt JOIN tags t ON t.id = pt.tag_id"
            " WHERE pt.photo_id = 1 AND pt.source = 'vlm'"
        )
    }
    assert ("subject", "pet") in vlm and ("setting", "beach") in vlm
    assert conn.execute("SELECT rowid FROM photo_fts WHERE photo_fts MATCH 'beach'").fetchone() is not None
    assert stage_counts(conn, "caption")["done"] == 1


def test_caption_makes_a_photo_findable_by_keyword(conn, tmp_path):
    from search.keyword import keyword_search

    derived = LocalStorage(tmp_path / "thumbs")
    seed_tags(conn, VOCAB)
    _photo_with_detail_thumb(conn, derived, 1, "cc" * 32)
    client = FakeInferenceClient([json.dumps({
        "caption": "a slice of Neapolitan pizza", "title": "Pizza night",
        "description": "A cheesy slice on a plate.", "tags": {},
    })])
    enqueue(conn, 1, "caption")
    drain(conn, {"caption": caption_handler(derived, client, "fake-vlm", DIMS, 1600)})
    assert keyword_search(conn, owner_id=1, query="Neapolitan", k=10) == [1]


def test_caption_skips_a_photo_with_no_thumbnail(conn, tmp_path):
    derived = LocalStorage(tmp_path / "thumbs")
    seed_tags(conn, VOCAB)
    add_photo(conn, photo_id=1, content_hash="dd" * 32, thumb_key=None)  # no thumbnail
    client = FakeInferenceClient([])  # must never be called
    enqueue(conn, 1, "caption")

    drain(conn, {"caption": caption_handler(derived, client, "fake-vlm", DIMS, 1600)})

    assert conn.execute("SELECT caption FROM photos WHERE id = 1").fetchone()["caption"] is None
    assert stage_counts(conn, "caption")["done"] == 1  # skipped cleanly, not a crash/failure
    assert client.calls == []


def test_caption_falls_back_to_the_grid_thumb_when_the_detail_is_missing(conn, tmp_path):
    derived = LocalStorage(tmp_path / "thumbs")
    seed_tags(conn, VOCAB)
    hash_hex = "ee" * 32
    make_jpeg(derived.local_path(thumb_key(hash_hex, 320)))  # only the grid thumb exists
    add_photo(conn, photo_id=1, content_hash=hash_hex, thumb_key=thumb_key(hash_hex, 320))
    client = FakeInferenceClient([json.dumps(
        {"caption": "c", "title": "t", "description": "d", "tags": {}}
    )])
    enqueue(conn, 1, "caption")

    drain(conn, {"caption": caption_handler(derived, client, "fake-vlm", DIMS, 1600)})

    assert conn.execute("SELECT caption FROM photos WHERE id = 1").fetchone()["caption"] == "c"


def test_invalid_json_leaves_the_photo_uncaptioned(conn, tmp_path):
    derived = LocalStorage(tmp_path / "thumbs")
    seed_tags(conn, VOCAB)
    _photo_with_detail_thumb(conn, derived, 1, "bb" * 32)
    client = FakeInferenceClient(["not json at all"])
    enqueue(conn, 1, "caption")

    drain(conn, {"caption": caption_handler(derived, client, "fake-vlm", DIMS, 1600)})

    assert conn.execute("SELECT caption FROM photos WHERE id = 1").fetchone()["caption"] is None
    assert stage_counts(conn, "caption")["done"] == 0  # not marked done; retried later
