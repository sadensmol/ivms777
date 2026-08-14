import time
from pathlib import Path

from config import get_settings
from ingest.caption import (
    backfill_caption_vectors,
    backfill_captions,
    caption_handler,
)
from ingest.embed import backfill_embeds, embed_handler
from ingest.facets import backfill_place_facets
from ingest.folders import process_folder_deletions
from ingest.taxonomy import backfill_taxonomy, taxonomy_handler
from ingest.thumbs import backfill_thumbnails
from ingest.vocab import load_vocab, seed_tags
from ingest.worker import drain, thumbnail_handler
from web.deps import build_context

POLL_SECONDS = 10
VOCAB_PATH = Path(__file__).resolve().parent.parent / "vocab.yaml"


def main() -> None:
    context = build_context(get_settings())
    settings = context.settings
    embedder, model_name = settings.build_embedder()
    client, caption_model = settings.build_inference_client()
    vocab = load_vocab(VOCAB_PATH)
    seed_tags(context.conn, vocab)
    handlers = {
        "thumbnail": thumbnail_handler(
            context.originals, context.derived,
            settings.thumb_grid_px, settings.thumb_detail_px,
        ),
        "embed": embed_handler(context.originals, embedder, model_name),
        "taxonomy": taxonomy_handler(context.derived, embedder, vocab),
        "caption": caption_handler(
            context.derived, client, caption_model, settings.caption_embed_model,
            list(vocab.dimensions), settings.thumb_detail_px,
        ),
    }
    while True:
        backfill_thumbnails(context.conn)
        backfill_embeds(context.conn)
        backfill_taxonomy(context.conn)
        backfill_place_facets(context.conn)
        backfill_captions(context.conn)
        backfill_caption_vectors(context.conn, client, settings.caption_embed_model)
        process_folder_deletions(
            context.conn, context.originals, context.derived, settings.owner_id,
            settings.thumb_grid_px, settings.thumb_detail_px,
        )
        drain(context.conn, handlers)
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
