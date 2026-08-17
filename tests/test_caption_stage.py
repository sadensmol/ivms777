from pathlib import Path

from fastapi.testclient import TestClient

from inference.models_client import ModelsClient
from ingest.caption import caption_handler
from ingest.jobs import enqueue, stage_counts
from ingest.thumbs import thumb_key
from ingest.vocab import load_vocab, seed_tags
from ingest.worker import drain
from modelsvc.app import create_models_app
from storage.local import LocalStorage
from tests.factories import add_photo
from tests.fixtures import make_jpeg

VOCAB = load_vocab(Path("vocab.yaml"))
DIMS = list(VOCAB.dimensions)


class FakeCaptionBackend:
    """Minimal `ModelBackend` double (design §5.1): the caption stage is now a
    thin HTTP client of the `models` service, so it is proven here the same
    way as tests/test_models_client.py — round-tripped over a real
    `create_models_app`, no local captioner. Only `caption` is exercised."""

    def __init__(self, response: dict | None = None, *, error: Exception | None = None):
        self.response = response
        self.error = error
        self.calls: list[tuple[bytes, list[str]]] = []

    def caption(self, image: bytes, dimensions: list[str]) -> dict:
        self.calls.append((image, dimensions))
        if self.error is not None:
            raise self.error
        return self.response


def _models_client(backend) -> ModelsClient:
    app = create_models_app(backend)
    transport = TestClient(app)._transport
    return ModelsClient("http://modelsvc", transport=transport)


def _photo_with_detail_thumb(conn, derived, pid, hash_hex, detail_px=1600):
    make_jpeg(derived.local_path(thumb_key(hash_hex, detail_px)))  # the image the VLM reads
    add_photo(conn, photo_id=pid, content_hash=hash_hex, thumb_key=thumb_key(hash_hex, 320))


def test_caption_stage_writes_caption_title_description_and_vlm_tags(conn, tmp_path):
    derived = LocalStorage(tmp_path / "thumbs")
    seed_tags(conn, VOCAB)
    _photo_with_detail_thumb(conn, derived, 1, "aa" * 32)
    backend = FakeCaptionBackend({
        "caption": "a dog on a beach",
        "title": "Beach day",
        "description": "A dog runs on the sand.",
        "tags": {"subject": ["pet"], "setting": ["beach"]},
        "model": "fake-vlm",
    })
    models_client = _models_client(backend)
    enqueue(conn, 1, "caption")

    drain(conn, {"caption": caption_handler(derived, models_client, DIMS, 1600)})

    row = conn.execute(
        "SELECT caption, caption_model, ai_title, ai_description FROM photos WHERE id = 1"
    ).fetchone()
    assert row["caption"] == "a dog on a beach"
    assert row["ai_title"] == "Beach day"
    assert "sand" in row["ai_description"]
    assert row["caption_model"] == "fake-vlm"  # the ACTUAL model the service used
    vlm = {
        (r["dimension"], r["label"]) for r in conn.execute(
            "SELECT t.dimension, t.label FROM photo_tags pt JOIN tags t ON t.id = pt.tag_id"
            " WHERE pt.photo_id = 1 AND pt.source = 'vlm'"
        )
    }
    assert ("subject", "pet") in vlm and ("setting", "beach") in vlm
    assert conn.execute("SELECT rowid FROM photo_fts WHERE photo_fts MATCH 'beach'").fetchone() is not None
    # The caption stage writes TEXT only; the §9 vector is embedded separately in the
    # SigLIP group (backfill_caption_vectors), so it is NOT set by this stage.
    assert conn.execute("SELECT caption_vec FROM photos WHERE id = 1").fetchone()["caption_vec"] is None
    assert stage_counts(conn, "caption")["done"] == 1


def test_backfill_caption_vectors_embeds_only_captioned_photos(conn):
    # The §9 caption-vector embed is its OWN step (pipeline group 2c), decoupled from
    # the caption stage: it embeds every captioned photo that has no vector yet, via
    # the dedicated text embedder (nomic in prod, fake here), and leaves un-captioned
    # photos alone.
    from embedding.store import read_caption_vector
    from inference.fakes import FakeInferenceClient
    from ingest.caption import backfill_caption_vectors

    add_photo(conn, photo_id=1, content_hash="a" * 64, thumb_key="1.jpg", caption="a dog on a beach")
    add_photo(conn, photo_id=2, content_hash="b" * 64, thumb_key="2.jpg")  # no caption

    n = backfill_caption_vectors(conn, FakeInferenceClient(), "fake")

    assert n == 1
    assert read_caption_vector(conn, 1) is not None
    assert read_caption_vector(conn, 2) is None
    # Idempotent: a second pass finds nothing pending.
    assert backfill_caption_vectors(conn, FakeInferenceClient(), "fake") == 0


def test_caption_makes_a_photo_findable_by_keyword(conn, tmp_path):
    from search.keyword import keyword_search

    derived = LocalStorage(tmp_path / "thumbs")
    seed_tags(conn, VOCAB)
    _photo_with_detail_thumb(conn, derived, 1, "cc" * 32)
    backend = FakeCaptionBackend({
        "caption": "a slice of Neapolitan pizza",
        "title": "Pizza night",
        "description": "A cheesy slice on a plate.",
        "tags": {},
        "model": "fake-vlm",
    })
    models_client = _models_client(backend)
    enqueue(conn, 1, "caption")
    drain(conn, {"caption": caption_handler(derived, models_client, DIMS, 1600)})
    assert keyword_search(conn, owner_id=1, query="Neapolitan", k=10) == [1]


def test_caption_skips_a_photo_with_no_thumbnail(conn, tmp_path):
    derived = LocalStorage(tmp_path / "thumbs")
    seed_tags(conn, VOCAB)
    add_photo(conn, photo_id=1, content_hash="dd" * 32, thumb_key=None)  # no thumbnail
    backend = FakeCaptionBackend()  # must never be called
    models_client = _models_client(backend)
    enqueue(conn, 1, "caption")

    drain(conn, {"caption": caption_handler(derived, models_client, DIMS, 1600)})

    assert conn.execute("SELECT caption FROM photos WHERE id = 1").fetchone()["caption"] is None
    assert stage_counts(conn, "caption")["done"] == 1  # skipped cleanly, not a crash/failure
    assert backend.calls == []


def test_caption_falls_back_to_the_grid_thumb_when_the_detail_is_missing(conn, tmp_path):
    derived = LocalStorage(tmp_path / "thumbs")
    seed_tags(conn, VOCAB)
    hash_hex = "ee" * 32
    make_jpeg(derived.local_path(thumb_key(hash_hex, 320)))  # only the grid thumb exists
    add_photo(conn, photo_id=1, content_hash=hash_hex, thumb_key=thumb_key(hash_hex, 320))
    backend = FakeCaptionBackend({
        "caption": "c", "title": "t", "description": "d", "tags": {}, "model": "fake-vlm",
    })
    models_client = _models_client(backend)
    enqueue(conn, 1, "caption")

    drain(conn, {"caption": caption_handler(derived, models_client, DIMS, 1600)})

    assert conn.execute("SELECT caption FROM photos WHERE id = 1").fetchone()["caption"] == "c"


def test_models_client_caption_error_leaves_the_photo_uncaptioned(conn, tmp_path):
    # A failed `/caption` call (the service's captioner errored) must not write a
    # half row — the job is failed, not done, and retried later.
    derived = LocalStorage(tmp_path / "thumbs")
    seed_tags(conn, VOCAB)
    _photo_with_detail_thumb(conn, derived, 1, "bb" * 32)
    backend = FakeCaptionBackend(error=ValueError("bad output from the VLM"))
    models_client = _models_client(backend)
    enqueue(conn, 1, "caption")

    drain(conn, {"caption": caption_handler(derived, models_client, DIMS, 1600)})

    assert conn.execute("SELECT caption FROM photos WHERE id = 1").fetchone()["caption"] is None
    assert stage_counts(conn, "caption")["done"] == 0  # not marked done; retried later
