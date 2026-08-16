import time
from pathlib import Path

from config import get_settings
from ingest.jobs import requeue_stalled
from ingest.pipeline import drain_pass
from ingest.vocab import load_vocab, seed_tags
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
    # The embedder/inference backend is built lazily INSIDE each pass (drain_pass),
    # not eagerly here: on a backend outage (e.g. the container cannot init CUDA on
    # jetson) the worker must keep producing thumbnails so photos still appear in
    # the library — it must NOT crash on startup, which left the whole library
    # unprocessed (§8).
    while True:
        drain_pass(context, vocab)
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
