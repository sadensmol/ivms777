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
from ingest.worker import Preempted, drain, thumbnail_handler

logger = logging.getLogger("ivms777.pipeline")


def drain_pass(context, vocab, should_preempt=lambda: False) -> None:
    """Run one drain pass over `context`'s library. Never raises for a backend
    outage: the embedder/inference stages are guarded so the GPU-free thumbnail
    stage always runs (see module docstring).

    Model load/evict and GPU serialization are owned by the `models` service's
    conveyor (plan 18): every embed/taxonomy/caption call goes through its
    scheduler over HTTP, so the pipeline itself takes no lease and needs no
    cross-process coordinator. `should_preempt` is retained as a soft cancel hook
    for the worker; it defaults to no-op."""
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

    # Group 2a — SigLIP stages (embed, taxonomy), ONLY if pending. Queueing needs
    # no model, so backfill unconditionally; only build the embedder (an HTTP shim
    # over the `models` service) when there is actually embed/taxonomy work to do.
    # The service's conveyor loads/evicts SigLIP and serializes the GPU (plan 18).
    backfill_embeds(conn)
    backfill_taxonomy(conn)
    if stage_counts(conn, "embed")["pending"] or stage_counts(conn, "taxonomy")["pending"]:
        try:
            embedder, model_name = settings.build_embedder()
            drain(conn, {
                "embed": embed_handler(context.originals, embedder, model_name),
                "taxonomy": taxonomy_handler(context.derived, embedder, vocab),
            }, should_preempt=should_preempt)
        except Preempted:
            # Preempted mid-pass → defer to the next poll.
            return
        except Exception:  # a backend outage defers model stages, never crashes the pass
            logger.exception(
                "embed/taxonomy unavailable this pass; thumbnails done, "
                "embed/taxonomy/caption deferred to a later pass",
            )
            return

    # Group 2b — caption stage, ONLY if pending. Detached from the model coordinator
    # (design §5.1, plan 15 task 3): captioning runs inside the `models` service,
    # reached over HTTP through `models_client` — no local captioner, no
    # INGEST_CAPTION lease, no in-worker VLM load. The stage writes caption TEXT only;
    # its §9 vector is embedded in group 2c below.
    backfill_captions(conn)
    if stage_counts(conn, "caption")["pending"]:
        try:
            models_client = settings.build_models_client()
            drain(conn, {
                "caption": caption_handler(
                    context.derived, models_client,
                    settings.thumb_detail_px,
                    should_preempt=should_preempt,
                ),
            }, should_preempt=should_preempt)
        except Preempted:
            return
        except Exception:
            logger.exception("caption deferred this pass")
            return

    # Group 2c — §9 caption-vector embed: one batched pass over the dedicated text
    # embedder (`nomic-embed-text` via Ollama, design §4). It is its own step because
    # the text embedder is a different model/backend from SigLIP and the caption VLM,
    # so it shares no residency slot with them — a caption written in group 2b gets
    # its vector here or on the next drain. Best-effort: `backfill_caption_vectors`
    # swallows an embedder outage, so this never blocks the pass.
    if conn.execute(
        "SELECT 1 FROM photos WHERE caption IS NOT NULL AND caption_vec IS NULL LIMIT 1"
    ).fetchone() is not None:
        try:
            client, _ = settings.build_inference_client()
            backfill_caption_vectors(conn, client, settings.caption_embed_model)
        except Exception:  # never crash the pass on the §9 enhancement
            logger.exception("caption-vector embed deferred this pass")
