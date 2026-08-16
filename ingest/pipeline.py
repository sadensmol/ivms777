"""One full ingest drain pass, shared by the worker loop (`ingest.cli`) and the
app's best-effort inline drain after an upload (`web.app`).

The pass is split into two groups by dependency, so a failure of the heavy
backend never blocks the light stage (§8, stages are independent):

1. **GPU/inference-free** — thumbnails, EXIF place facets, folder deletions. These
   are what make an uploaded photo *appear in the library* (the grid shows photos
   with a `thumb_key`), so they must run even when the embedder or inference
   backend is unavailable.
2. **Embedder/inference** — embed, taxonomy, caption. If the SigLIP embedder cannot
   be built (e.g. the app/worker container cannot init CUDA on jetson — the
   observed `RuntimeError 801`), these are skipped THIS pass and retried on the
   next one, never blocking group 1. Search/tags/captions fill in once the backend
   is healthy; the photos are already browsable in the grid.
"""

import logging

from ingest.caption import backfill_caption_vectors, backfill_captions, caption_handler
from ingest.embed import backfill_embeds, embed_handler
from ingest.facets import backfill_place_facets
from ingest.folders import process_folder_deletions
from ingest.jobs import stage_counts
from ingest.taxonomy import backfill_taxonomy, taxonomy_handler
from ingest.thumbs import backfill_thumbnails
from ingest.worker import drain, thumbnail_handler

logger = logging.getLogger("ivms777.pipeline")


def drain_pass(context, vocab) -> None:
    """Run one drain pass over `context`'s library. Never raises for a backend
    outage: the embedder/inference stages are guarded so the GPU-free thumbnail
    stage always runs (see module docstring)."""
    settings = context.settings
    conn = context.conn

    # Group 1 — no GPU, no inference backend. Makes photos visible in the library.
    backfill_thumbnails(conn)
    backfill_place_facets(conn)
    process_folder_deletions(
        conn, context.originals, context.derived, settings.owner_id,
        settings.thumb_grid_px, settings.thumb_detail_px,
    )
    drain(conn, {
        "thumbnail": thumbnail_handler(
            context.originals, context.derived,
            settings.thumb_grid_px, settings.thumb_detail_px,
        ),
    })

    # Group 2 — needs the embedder / inference backend. Skip (retry next pass) if the
    # embedder can't be built, so a GPU outage never blocks group 1.
    try:
        embedder, model_name = settings.build_embedder()
        client, caption_model = settings.build_inference_client()
    except Exception:  # a backend outage defers model stages, never crashes the pass
        logger.exception(
            "embedder/inference unavailable this pass; thumbnails done, "
            "embed/taxonomy/caption deferred to a later pass",
        )
        return

    # Group 2a — SigLIP stages. Both use the in-process embedder (GPU).
    backfill_embeds(conn)
    backfill_taxonomy(conn)
    drain(conn, {
        "embed": embed_handler(context.originals, embedder, model_name),
        "taxonomy": taxonomy_handler(context.derived, embedder, vocab),
    })

    # Queue caption jobs now (needs no SigLIP) so the release gate below can see
    # whether any caption work is actually pending this pass.
    backfill_captions(conn)

    # Free SigLIP's GPU allocation before captioning. On the 8 GB unified-memory
    # Jetson a resident SigLIP pins torch's CUDA allocator, so Ollama can't get GPU
    # buffers for the vision captioner and silently falls back to CPU (design §8 —
    # SigLIP is unloaded before the captioner runs). Only when caption work is
    # actually pending, so an idle worker poll doesn't needlessly reload SigLIP.
    if not settings.use_fake_embedder and stage_counts(conn, "caption")["pending"]:
        del embedder
        from embedding.siglip import release_siglip_embedder

        release_siglip_embedder()

    # Group 2b — caption stage. Uses the Ollama client, not SigLIP.
    backfill_caption_vectors(conn, client, settings.caption_embed_model)
    drain(conn, {
        "caption": caption_handler(
            context.derived, client, caption_model, settings.caption_embed_model,
            list(vocab.dimensions), settings.thumb_detail_px,
        ),
    })
