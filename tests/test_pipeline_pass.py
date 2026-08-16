"""The drain pass must be resilient to an unavailable embedder/inference backend.

Regression for the jetson failure where the app/worker could not init CUDA
(RuntimeError 801): building the SigLIP embedder eagerly at the top of the drain
blocked the GPU-free thumbnail stage too, so uploaded photos never got a
`thumb_key` and never appeared in the library (§8: stages are independent)."""

import pytest

from ingest.pipeline import drain_pass
from ingest.vocab import load_vocab
from storage.keys import content_key
from tests.fixtures import jpeg_bytes
from web.app import VOCAB_PATH
from web.deps import build_context


@pytest.fixture
def ctx(settings):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return build_context(settings)


def _uploaded_photo(ctx, content_hash="ab" * 32):
    # A received-but-unprocessed photo: bytes stored, row inserted, thumbnail job
    # pending — exactly the state after /api/upload/finish, before any drain.
    ctx.originals.write(content_key(content_hash, ".jpg"), jpeg_bytes())
    pid = ctx.conn.execute(
        "INSERT INTO photos(owner_id, content_hash, storage_key, created_at, updated_at)"
        " VALUES (1, ?, ?, '2026-01-01', '2026-01-01')",
        (content_hash, content_key(content_hash, ".jpg")),
    ).lastrowid
    ctx.conn.execute(
        "INSERT INTO jobs(photo_id, stage, status, updated_at)"
        " VALUES (?, 'thumbnail', 'pending', '2026-01-01')",
        (pid,),
    )
    return pid


def test_thumbnail_stage_runs_even_when_the_embedder_is_down(ctx, monkeypatch):
    from config import Settings
    pid = _uploaded_photo(ctx)

    def boom(self):
        raise RuntimeError("Unexpected error from cudaGetDeviceCount ... Error 801")

    monkeypatch.setattr(Settings, "build_embedder", boom)
    drain_pass(ctx, load_vocab(VOCAB_PATH))  # must not raise
    thumb = ctx.conn.execute("SELECT thumb_key FROM photos WHERE id = ?", (pid,)).fetchone()
    assert thumb["thumb_key"] is not None    # photo is now visible in the library
    status = ctx.conn.execute(
        "SELECT status FROM jobs WHERE photo_id = ? AND stage = 'thumbnail'", (pid,)
    ).fetchone()["status"]
    assert status == "done"


def test_full_pass_processes_when_the_embedder_is_available(ctx):
    # With the fake embedder (settings default in tests), the whole pass runs and
    # the photo is both thumbnailed and embedded.
    pid = _uploaded_photo(ctx)
    drain_pass(ctx, load_vocab(VOCAB_PATH))
    assert ctx.conn.execute(
        "SELECT thumb_key FROM photos WHERE id = ?", (pid,)
    ).fetchone()["thumb_key"] is not None
    embedded = ctx.conn.execute(
        "SELECT status FROM jobs WHERE photo_id = ? AND stage = 'embed'", (pid,)
    ).fetchone()
    assert embedded["status"] == "done"


def test_siglip_is_released_before_the_caption_stage(ctx, monkeypatch):
    # On the real-embedder path, drain_pass must free SigLIP *before* captioning so
    # the Ollama vision model gets the GPU (design §8). Flip use_fake_embedder off to
    # exercise the release gate; stub the caption handler to record ordering (the
    # fake inference client can't answer real caption calls).
    from config import Settings
    from embedding.fakes import FakeEmbedder
    from ingest import pipeline

    pid = _uploaded_photo(ctx)
    ctx.settings.use_fake_embedder = False
    monkeypatch.setattr(Settings, "build_embedder", lambda self: (FakeEmbedder(), "fake"))

    order: list[str] = []
    monkeypatch.setattr(
        "embedding.siglip.release_siglip_embedder", lambda: order.append("release")
    )

    def fake_caption_handler(*_args, **_kwargs):
        def handle(_conn, _photo_id):
            order.append("caption")
        return handle

    monkeypatch.setattr(pipeline, "caption_handler", fake_caption_handler)

    drain_pass(ctx, load_vocab(VOCAB_PATH))

    assert "release" in order and "caption" in order
    assert order.index("release") < order.index("caption")  # freed before captioning
    caption = ctx.conn.execute(
        "SELECT status FROM jobs WHERE photo_id = ? AND stage = 'caption'", (pid,)
    ).fetchone()
    assert caption["status"] == "done"


def test_siglip_is_not_released_when_no_caption_work_is_pending(ctx, monkeypatch):
    # An idle poll (nothing to caption) must not reload SigLIP needlessly.
    from config import Settings
    from embedding.fakes import FakeEmbedder

    _uploaded_photo(ctx)
    ctx.settings.use_fake_embedder = False
    monkeypatch.setattr(Settings, "build_embedder", lambda self: (FakeEmbedder(), "fake"))
    # Pre-caption everything so backfill_captions queues nothing.
    ctx.conn.execute("UPDATE photos SET caption = 'x' WHERE caption IS NULL")
    released: list[bool] = []
    monkeypatch.setattr(
        "embedding.siglip.release_siglip_embedder", lambda: released.append(True)
    )

    drain_pass(ctx, load_vocab(VOCAB_PATH))

    assert released == []  # nothing to caption → SigLIP stays resident
