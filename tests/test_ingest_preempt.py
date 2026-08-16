# tests/test_ingest_preempt.py — uses the `conn` fixture from tests/conftest.py
import pytest

from ingest import jobs
from ingest.worker import Preempted, drain
from tests.factories import add_photo

# Caption-stage preemption (aborting an in-flight `/caption` call mid-VLM-call)
# moved to the `models` service itself (design §5.1/§8.1, plan 15 task 3/5) — the
# caption stage is now a thin HTTP client with no local captioner to cancel, so
# there is nothing here for the stage to map to `Preempted` any more. See
# tests/test_caption_stage.py for the stage's own tests.


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
