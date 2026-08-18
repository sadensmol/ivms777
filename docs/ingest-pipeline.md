# Ingest pipeline — stages, residency, reprocessing

The exact per-stage mechanics, the in-process residency guard, the reprocess
endpoints, self-healing backfills, failure handling, and the manifest gate. Design
§8 carries the stage sequence and the draining invariant; this file is the detail.
Section references point into `docs/design.md`.

## Stages

Stages run per photo, each recorded in `jobs`. The worker drains pending jobs stage
by stage. Killing the container and restarting resumes exactly where it stopped.

0. **receive** — `app` accepts an uploaded file, verifies its SHA-256 against the
   hash the client declared, stores the original under a content-addressed key,
   reads EXIF, and inserts the `photos` row plus its `photo_sources` row. A hash
   that already exists adds only the source row and queues nothing. This stage runs
   in `app`, synchronously with the request, and is the only work not driven by the
   job queue — everything after it is.
1. **facets** — derive queryable facets from the stored EXIF (§6.2).
2. **thumbnail** — two sizes (320 px grid, 1600 px detail) written to storage. HEIC
   via `pillow-heif`.
3. **embed** — SigLIP 2 image embedding, batched, written to `photo_vec`.
4. **taxonomy** — SigLIP zero-shot scoring against `vocab.yaml`, plus pixel
   statistics. Fast, runs immediately after embed.
5. **caption** — the `worker` calls the `models` service's `POST /caption` (§5.1)
   with one photo's thumbnail; the service's `CaptionBackend` (`OpenAICaptioner`,
   §4) POSTs an OpenAI `/v1/chat/completions` request with the image to
   `llama-server` (mac/jetson) / vLLM (cloud) and returns a caption sentence
   (title/description), written straight to the DB, then rebuilds the FTS row. It
   writes **no tags** — tags are owned entirely by the SigLIP taxonomy stage (§7).
   Slowest stage, runs last.

The caption's §9 text embedding is a **separate step, not part of the caption
stage** (pipeline group 2c): `backfill_caption_vectors` embeds every captioned
photo that still has no vector, in ONE batch, with the dedicated text embedder
(`nomic-embed-text-v1.5`, in-process in the `models` service, §4) into
`photos.caption_vec`. It is its own step (a different model from SigLIP and from the
caption call) so it is never interleaved per-photo with captioning. A caption
written this pass gets its vector this pass or the next; a library captioned before
the column existed backfills the same way — no re-caption needed.

**Stages are drained in order across the whole library, not per photo.** Every
photo is embedded and scored before any photo is captioned. This keeps the Jetson
profile viable: the SigLIP-using stages (`embed`, `taxonomy`) call `/embed/image`
and `/tag`, and the `caption` stage calls `/caption` — two calls that **never hold
the GPU at once** inside the `models` service, so 8 GB never has to hold both SigLIP
and the captioner. Draining embed/taxonomy first means search and "show similar"
work across the entire collection within minutes of the upload finishing, while
captions fill in over the following hours.

## In-process residency — SigLIP ensure-loaded

All model work lives in the one `models` service (§5.1), so deciding which heavy
model is loaded right now is an **in-process** concern —
`modelsvc/residency.py::Residency` — not a cross-process DB lease. The earlier
`model_lease` table and `models/coordinator.py::ModelCoordinator` (a DB row,
heartbeat thread, and stale-reclaim logic that coordinated the separate `app` and
`worker` processes, back when each loaded its own SigLIP) are gone.
`models/coordinator.py` keeps only a torch-free no-op stub — `RefusedError` and
`LeaseBusyError` (kept only so existing `except (...)` clauses need no edit) and
`NoopCoordinator.require()`, a nullcontext — so `app`/`worker` call sites
(`ctx.make_coordinator(...).require("CHAT")`, `.require("SEARCH")`,
`.require("MEMORY_REBUILD")`, `_lease(coordinator, "INGEST_EMBED")`, §10/§11) need
no edit even though there is nothing left for them to coordinate: `app` and `worker`
hold no models, so `require()` now loads nothing, refuses nothing, and never raises.

Since plan 16 removed the in-process caption VLM (captioning is now a remote OpenAI
call to `llama-server`), **SigLIP is the only heavy in-process model** in the
service. Nothing contends with it for the GPU, so residency is no longer a
swap/preempt arbiter — it is a plain **ensure-loaded** guard. A sub-backend never
loads its model directly; it wraps its call in `residency.use(name)` and the manager
loads it once:

```python
with residency.use("siglip", priority=HIGH):
    ...            # SigLIP is loaded once, then reused
```

`SiglipBackend` wraps every `/embed/image`, `/embed/text`, `/tag`, and
`/embed/calibration` call in `use("siglip", priority=HIGH)` — search, chat, memory
rebuild, and ingest embed/taxonomy all look identical from here, an HTTP request
against the same endpoint, so there is no per-caller workload taxonomy the way the
old coordinator needed one. `priority` is retained as a readability hint only; with
a single registered model there is nothing to prioritise against, so `use()` loads
once and never evicts or preempts. Captioning is not registered with `Residency` at
all — it is a remote call. Neither is the in-process `nomic` text embedder (tiny,
~0.3 GB, loaded once by `TextBackend`).

Because captioning left the GPU-sharing picture, the whole caption-preemption path
is gone too: no `should_preempt`, no `CaptionPreempted`/503, no
`ModelsCaptionPreempted`. A caption is a plain HTTP call that either returns or
errors (and retries like any stage). This removes the `StoppingCriteria` +
`threading.Condition` machinery the two-model swap used to need.

What is resident (`residency.resident()` — `["siglip"]` once loaded, **plus the
llama-server / vLLM text model**, e.g. `gemma4-E2B`, a separate always-on process
serving both text and captioning, so it is reported in EVERY state — the bar shows
it during captioning/idle too, not only chat/planning), the models-process RAM, the
**GPU load**, and the **current in-flight op** (`active` — `embedding` / `tagging` /
`captioning` / `planning` / `chat`, or `null` when idle; tracked by
`modelsvc/activity.py` since the service is the one place that sees every call) are
all reported by `CompositeBackend.resources()` on `GET /resources`, which
`/api/resources` proxies for the **resource bar (§13)**. The `active` step is ground
truth of what the GPU is doing, not a declared intent.

## Failure handling & degradation

Failed jobs retry up to 3 times with the error recorded, then stay `failed` and are
listed in the UI. One bad file never stalls the queue.

**A backend outage never blocks the GPU-free stages, and never fails an upload.** A
drain pass (`ingest/pipeline.py::drain_pass`, shared by the `worker` loop and the
app's inline drain) runs in two groups: first the **GPU/inference-free** stages —
thumbnail, EXIF place facets, folder deletions — then the **embedder/inference**
stages (embed, taxonomy, caption, and the batched caption_embed). The embedder (`RemoteEmbedder`, an HTTP shim over
the `models` service, §5.1) is built **inside the pass**, not eagerly at process
start; if a call through it fails — e.g. the `models` service is unreachable, or its
own CUDA init fails on jetson (the previously observed `RuntimeError 801`, now inside
that service rather than the worker) — the model group is skipped for that pass and
retried on the next, while thumbnails still run. So an uploaded photo **appears in
the library** (the grid shows photos with a `thumb_key`) even while the GPU is down,
and the `worker` keeps looping instead of crashing on startup. For the same reason
the **upload receipt is decoupled from processing**: `/api/upload/finish` records
the upload and returns even if its best-effort inline drain fails — the bytes are
stored, the jobs are queued, and the `worker` drains them regardless. The receipt
succeeding is not a claim that processing is done.

A file rejected at **receive** — hash mismatch, unreadable image, unsupported
format — never becomes a `photos` row. It is counted in `uploads.files_failed` and
reported to the client, which lists it on the upload screen so a failed transfer is
visible rather than silently missing.

## Reprocessing

Originals are kept (§3.2b), so the derived state — thumbnails, embeddings, tags,
captions — can be rebuilt without re-uploading. `POST /reprocess` resets a **range**
of stages (`from_stage` through an optional `to_stage`, inclusive) to `pending` for
the owner's photos; the `worker` re-runs them in `STAGES` order on its next poll.
The `/upload` UI puts a **Reprocess** button on **every stage row** — `thumbnail`,
`embed`, `taxonomy`, `caption`, `caption_embed` — and each re-runs **only that one stage**
(`from=to=stage`). So re-tagging after a `vocab.yaml` change never rebuilds
thumbnails, and re-embedding never re-captions; each stage is re-run in isolation.
The `caption` button is styled as a destructive action and confirms first ("can take
hours"), since it alone re-runs the slow vision model. The endpoint still accepts any
range, so a multi-stage re-run — `from=embed to=taxonomy` for a new embedding model —
is one POST away.

**Per-photo reprocess.** `POST /photo/{id}/reprocess` re-runs a single model stage
for one photo — `taxonomy` (re-tag) or `caption` (re-caption) — resetting just that
photo's job to `pending`. It is exposed on the `/photo` page (§13) so a single bad
tagging or caption can be redone without touching the rest of the library. Only the
model-derived stages are offered; `thumbnail` and `embed` are static (the bytes
never change) and are not re-runnable per photo. It redirects back to the photo with
its collection query intact.

Every stage handler is idempotent — it overwrites its own output — so a reprocess is
safe to trigger at any time. **Self-healing backfills** run automatically each
drain: they queue `thumbnail` for a photo still without one, `embed` for one missing
a vector, `taxonomy` for an embedded-but-untagged photo, `caption` for a
thumbnailed-but-uncaptioned one, and `caption_embed` for a captioned photo with no
caption vector — so a library predating a stage, or a photo whose
thumbnail once failed, heals with no manual action. A photo that genuinely can't be
thumbnailed is skipped by the later stages rather than crashing them.

## The manifest gate

Stage 2 must not run against a half-processed library, so `/api/manifest` reports
the collection as `complete` only when no `pending` or `running` job rows remain for
the owner. It still serves a manifest while incomplete, marked as such, and
`ivms777-sync` refuses to `apply` one unless `--allow-incomplete` is passed.
