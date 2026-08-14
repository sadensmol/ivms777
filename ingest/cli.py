import time

from config import get_settings
from ingest.embed import backfill_embeds, embed_handler
from ingest.worker import drain, thumbnail_handler
from web.deps import build_context

POLL_SECONDS = 10


def main() -> None:
    context = build_context(get_settings())
    settings = context.settings
    embedder, model_name = settings.build_embedder()
    handlers = {
        "thumbnail": thumbnail_handler(
            context.originals, context.derived,
            settings.thumb_grid_px, settings.thumb_detail_px,
        ),
        "embed": embed_handler(context.originals, embedder, model_name),
    }
    while True:
        backfill_embeds(context.conn)
        drain(context.conn, handlers)
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
