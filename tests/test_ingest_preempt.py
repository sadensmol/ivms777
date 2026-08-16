# tests/test_ingest_preempt.py — uses the `conn` fixture from tests/conftest.py
from pathlib import Path

import pytest

from ingest import jobs
from ingest.caption import caption_handler
from ingest.thumbs import thumb_key
from ingest.vocab import load_vocab, seed_tags
from ingest.worker import Preempted, drain
from inference.client import InferenceCancelled
from inference.fakes import FakeInferenceClient
from storage.local import LocalStorage
from tests.factories import add_photo
from tests.fixtures import make_jpeg

VOCAB = load_vocab(Path("vocab.yaml"))


def test_drain_stops_and_requeues_on_preempt(conn):
    # two photos, each with a pending 'embed' job (same pattern as tests/test_jobs.py)
    for pid, h in ((1, "a"), (2, "b")):
        add_photo(conn, photo_id=pid, content_hash=h, thumb_key=f"{h}.jpg")
        jobs.enqueue(conn, pid, "embed")
    handled = []

    def handler(conn, photo_id):
        handled.append(photo_id)

    # preempt fires immediately → nothing handled, both jobs still pending
    with pytest.raises(Preempted):
        drain(conn, {"embed": handler}, should_preempt=lambda: True)
    assert handled == []
    assert jobs.stage_counts(conn, "embed")["pending"] == 2


def test_caption_aborts_in_flight_and_requeues_the_photo_on_preempt(conn, tmp_path):
    # The force-preempt (§8.1): an interactive request flips preempt WHILE a caption
    # is being generated. The caption stage must abort the in-flight VLM call, yield
    # (Preempted), and leave the photo PENDING — not failed (no burnt retry), not
    # done — so chat gets the models within a chunk, not a whole caption.
    seed_tags(conn, VOCAB)
    derived = LocalStorage(tmp_path / "thumbs")
    h = "aa" * 32
    make_jpeg(derived.local_path(thumb_key(h, 1600)))
    add_photo(conn, photo_id=1, content_hash=h, thumb_key=thumb_key(h, 320))
    jobs.enqueue(conn, 1, "caption")

    class _PreemptingCaptioner:
        """A `Captioner` whose in-flight call is cancelled (design §4/§8.1) —
        mirrors what OllamaCaptioner/VLMCaptioner do when should_preempt fires
        mid-call: raise InferenceCancelled, never return a result."""

        caption_model = "fake-vlm"

        def caption(self, image, dimensions, *, should_preempt=lambda: False):
            raise InferenceCancelled(self.caption_model)

    captioner = _PreemptingCaptioner()
    embed_client = FakeInferenceClient([])

    handler = caption_handler(
        derived, captioner, embed_client, "fake-embed", list(VOCAB.dimensions), 1600,
        should_preempt=lambda: True,          # preempt is already pending mid-caption
    )
    with pytest.raises(Preempted):
        drain(conn, {"caption": handler})     # drain's own pre-claim gate stays off

    counts = jobs.stage_counts(conn, "caption")
    assert counts["pending"] == 1             # requeued for the next pass
    assert counts["done"] == 0 and counts["failed"] == 0
    assert conn.execute("SELECT caption FROM photos WHERE id = 1").fetchone()["caption"] is None
