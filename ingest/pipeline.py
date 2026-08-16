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

import contextlib
import logging

from ingest.caption import backfill_caption_vectors, backfill_captions, caption_handler
from ingest.embed import backfill_embeds, embed_handler
from ingest.facets import backfill_place_facets
from ingest.folders import process_folder_deletions
from ingest.jobs import stage_counts
from ingest.taxonomy import backfill_taxonomy, taxonomy_handler
from ingest.thumbs import backfill_thumbnails
from ingest.worker import Preempted, drain, thumbnail_handler
from models.coordinator import LeaseBusyError

logger = logging.getLogger("ivms777.pipeline")


@contextlib.contextmanager
def _lease(coordinator, workload):
    """No-op when `coordinator is None` (tests, the app's inline drain that
    doesn't contend) — otherwise takes the model lease for `workload` (§8.1)."""
    if coordinator is None:
        yield
    else:
        with coordinator.require(workload):
            yield


def drain_pass(context, vocab, coordinator=None, should_preempt=lambda: False) -> None:
    """Run one drain pass over `context`'s library. Never raises for a backend
    outage: the embedder/inference stages are guarded so the GPU-free thumbnail
    stage always runs (see module docstring).

    `coordinator` and `should_preempt` are the worker's model-lease + hard-preempt
    hooks (§8.1); both default to no-op so the app's best-effort inline drain
    behaves exactly as before — no lease, no preemption."""
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
    # no model, so backfill unconditionally; only take the INGEST_EMBED lease
    # (and load SigLIP) when there is actually embed/taxonomy work to do — an
    # idle poll must never load+unload a model for nothing.
    backfill_embeds(conn)
    backfill_taxonomy(conn)
    if stage_counts(conn, "embed")["pending"] or stage_counts(conn, "taxonomy")["pending"]:
        try:
            with _lease(coordinator, "INGEST_EMBED"):
                embedder, model_name = settings.build_embedder()
                drain(conn, {
                    "embed": embed_handler(context.originals, embedder, model_name),
                    "taxonomy": taxonomy_handler(context.derived, embedder, vocab),
                }, should_preempt=should_preempt)
        except (Preempted, LeaseBusyError):
            # Preempted mid-pass, or interactive work holds the lease → defer to
            # the next poll. The lease exit (or never having taken one) already
            # released SigLIP, so there is nothing left to clean up here.
            return
        except Exception:  # a backend outage defers model stages, never crashes the pass
            logger.exception(
                "embed/taxonomy unavailable this pass; thumbnails done, "
                "embed/taxonomy/caption deferred to a later pass",
            )
            return

    # Group 2b — caption stage, ONLY if pending. Same reasoning: queue for free,
    # only take the INGEST_CAPTION lease when there is caption work to do.
    backfill_captions(conn)
    if stage_counts(conn, "caption")["pending"]:
        try:
            with _lease(coordinator, "INGEST_CAPTION"):
                # SigLIP is already released (the INGEST_EMBED lease exit above
                # frees it), so Ollama gets the GPU for the vision captioner
                # instead of silently falling back to CPU (§8).
                client, _ = settings.build_inference_client()
                # SHARED INSTANCE (design §4/§8.1): when a coordinator is present it
                # already LOADED `coordinator.captioner` on lease entry (the
                # INGEST_CAPTION resource) — this stage must call THAT instance, not
                # build a second one, or the coordinator loads instance A onto the
                # GPU while the stage calls unloaded instance B. With
                # `coordinator=None` (the app's best-effort inline drain), there is
                # no lease/load step, so a fresh captioner is built here instead:
                # `OllamaCaptioner` needs no explicit load (Ollama auto-loads on the
                # `complete` call), so mac's inline drain is unaffected; an
                # in-process `VLMCaptioner` has no weights loaded, so `caption()`
                # raises "before load()", the job is marked failed, and the WORKER
                # (which always has a coordinator) re-runs it on its next pass —
                # acceptable best-effort degradation. Do NOT lazy-load the VLM here
                # outside a lease — that would bypass the RAM budget.
                captioner = coordinator.captioner if coordinator is not None else settings.build_captioner(client)
                backfill_caption_vectors(conn, client, settings.caption_embed_model)
                drain(conn, {
                    "caption": caption_handler(
                        context.derived, captioner, client, settings.caption_embed_model,
                        list(vocab.dimensions), settings.thumb_detail_px,
                        should_preempt=should_preempt,   # abort an in-flight caption (§8.1)
                    ),
                }, should_preempt=should_preempt)
        except (Preempted, LeaseBusyError):
            return
        except Exception:
            logger.exception("caption deferred this pass")
            return
