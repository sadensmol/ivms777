import time
from pathlib import Path

from config import get_settings
from ingest.jobs import requeue_stalled
from ingest.pipeline import drain_pass
from ingest.vocab import load_vocab, seed_tags
from models.lease_store import preempt_requested, release as release_lease
from web.deps import build_context

POLL_SECONDS = 10
VOCAB_PATH = Path(__file__).resolve().parent.parent / "vocab.yaml"


def main() -> None:
    context = build_context(get_settings())
    vocab = load_vocab(VOCAB_PATH)
    seed_tags(context.conn, vocab)
    # Reclaim jobs a previous worker was mid-processing when it was killed (a crash,
    # or the dev auto-reloader) — otherwise they sit 'running' forever (§8).
    requeue_stalled(context.conn)
    # Likewise drop any model lease our previous (now-dead) self still holds: a
    # fresh worker process holds nothing, so a lingering 'worker' row is an orphan
    # from a crash/reload. Clearing it immediately means chat need not wait out the
    # stale-reclaim window after a worker restart (design §8.1). Only our own holder
    # is dropped — a real 'app'/CHAT lease is left untouched.
    release_lease(context.conn, "worker")
    # The embedder/inference backend is built lazily INSIDE each pass (drain_pass),
    # not eagerly here: on a backend outage (e.g. the container cannot init CUDA on
    # jetson) the worker must keep producing thumbnails so photos still appear in
    # the library — it must NOT crash on startup, which left the whole library
    # unprocessed (§8).
    client, _ = context.settings.build_inference_client()
    coordinator = context.make_coordinator(client, "worker")
    while True:
        drain_pass(
            context, vocab, coordinator,
            should_preempt=lambda: preempt_requested(context.conn),
        )
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
