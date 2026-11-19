"""One full ingest drain pass, shared by the worker loop (`ingest.cli`) and the
app's best-effort inline drain after an upload (`web.app`).

The pass is split into two groups by dependency, so a failure of the heavy
backend never blocks the light stage (§8, stages are independent):

1. **GPU/inference-free** — thumbnails, EXIF place facets, folder deletions,
   saved-voice deletions. These are what make an uploaded photo *appear in the
   library* (the grid shows photos with a `thumb_key`), so they must run even when
   the embedder or inference backend is unavailable.
2. **Embedder/inference** — embed, taxonomy, caption. If the SigLIP embedder cannot
   be built (e.g. the app/worker container cannot init CUDA on jetson — the
   observed `RuntimeError 801`), these are skipped THIS pass and retried on the
   next one, never blocking group 1. Search/tags/captions fill in once the backend
   is healthy; the photos are already browsable in the grid.
3. **Faces** — the `faces` stage, last and guarded on its own (§8): an outage
   defers it to a later pass and never fails the pass.
"""

import logging

from feed.saved_voices import process_voice_deletions
from ingest.caption import (
    backfill_caption_embeds,
    backfill_caption_vectors,
    backfill_captions,
    caption_handler,
)
from ingest.embed import backfill_embeds, embed_handler, relabel_legacy_embeds
from ingest.faces import FacesDeferred, FaceStageParams, backfill_faces, faces_handler
from ingest.facets import backfill_place_facets
from ingest.folders import process_folder_deletions
from ingest.jobs import stage_counts
from ingest.taxonomy import backfill_taxonomy, taxonomy_handler
from ingest.thumbs import backfill_thumbnails
from ingest.worker import Preempted, drain, thumbnail_handler
from models.catalog import face_model_for

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
    # The saved-voice deletion outbox (D33). Never raises for one row (it logs it).
    process_voice_deletions(conn, context.voices, settings.owner_id)
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
    # Naming a model costs no model: repairing the legacy stamps must happen on a
    # pass that has no embed work left, so it is here and not inside the branch.
    relabel_legacy_embeds(conn, settings.embed_model_key(conn))
    if stage_counts(conn, "embed")["pending"] or stage_counts(conn, "taxonomy")["pending"]:
        try:
            # `conn` so the stamped `embedding_model` names the model the user
            # actually selected, not the env override / profile default (§4.1).
            embedder, model_name = settings.build_embedder(conn)
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
    backfill_caption_embeds(conn)
    if stage_counts(conn, "caption_embed")["pending"]:
        try:
            client, _ = settings.build_inference_client()
            # `conn` so the captions are embedded under the model the user actually
            # selected, not the env override / profile default (§4.1) — the name
            # picks the task prefix, so a stale one silently degrades the vectors.
            backfill_caption_vectors(conn, client, settings.caption_embed_model_key(conn))
        except Exception:  # never crash the pass on the §9 enhancement
            logger.exception("caption-vector embed deferred this pass")

    # Group 3 — the `faces` stage (§8), last and guarded on its own, ONLY if pending. The
    # face model is reached through the `models` service; the worker stores the faces and
    # groups them. A models-service outage returns the claimed job to pending with no
    # attempt counted and stops the stage until the next pass, never failing it.
    backfill_faces(conn)
    if not stage_counts(conn, "faces")["pending"]:
        return
    face_model = face_model_for(settings.profile)
    if face_model is None or face_model.input_edge is None:
        logger.info("faces deferred: profile %s offers no face model", settings.profile)
        return
    try:
        drain(conn, {
            "faces": faces_handler(
                context.originals, settings.build_models_client(),
                FaceStageParams(match_threshold=settings.face_threshold(),
                                input_edge=face_model.input_edge),
            ),
        }, ("faces",), should_preempt=should_preempt)
    except FacesDeferred as error:
        logger.info("faces deferred: models service unavailable: %s", error)
    except Preempted:
        return
    except Exception:
        logger.exception("faces deferred this pass")
