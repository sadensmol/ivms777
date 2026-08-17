# Roadmap — phases & future work

Delivery phases and the future-work backlog. This is direction, not current
state — `docs/design.md` describes what the app *is* today; this file describes
the order it was built in and what may come next. Section references (`§3.2`,
`§9.1`, …) point into `docs/design.md`.

## Phases

| Phase | Delivers |
|---|---|
| 0 | Skeleton, config and profiles, compose files, SQLite schema with `sqlite-vec` and FTS5, storage and inference interfaces, fakes, test harness |
| 1 | Upload — client-side hashing worker, probe endpoint, receive stage, full EXIF capture and facet derivation, thumbnails, `/upload` progress, `/library` grid with EXIF facet filters and sorting, `/duplicates`, caption model bake-off script |
| 2 | SigLIP embeddings, taxonomy scoring, semantic + facet + keyword + fusion search, similar photos, `/photo` detail |
| 3 | Captioning stage against the inference service, captions in the UI; the caption stage also emits a per-photo **AI title + description**, and `/photo` renders the full AI panel (title, description, caption, tags) |
| 4 | Query planner, parsed-filter chips, caption vocabulary mining with tag suggestions |
| 5 | Memories organizer — agentic RAG builder, persisted `groups(kind='memory')`, `/organize?by=memories` with rebuild (plan 07, done) |
| 6 | Ask-your-library chat with streaming and citations |
| 7 | Stage 2 — layouts, `/api/manifest`, `/export` preview, and the `ivms777-sync` CLI with plan, apply, undo, and verify |

Each phase leaves a working, useful application.

Phase 7 is last because the manifest is richer the more the library knows —
the `date-tags` layout needs taxonomy from phase 2 and captions from phase 3.
But it depends on nothing after phase 3, so it can be pulled forward if
reorganizing the disk matters more than chat does.

## Future work

- Authentication, signup, and per-user quotas for public access (v02).
- Face detection and person clustering.
- Object storage backend behind the existing `Storage` interface.
- Optional XMP sidecar export so other tools see the tags.
- Offline reverse geocoding place names — the "By place" organizer names albums
  by city and a `place_city`/`place_country` sidebar facet filters the library by
  place (plan 08, done). Future: sub-city neighbourhoods and user-editable labels.
- Agentic RAG + reranking for chat retrieval (plan 10, **done**). Chat routes the
  question through the query planner, ranks candidates by caption-meaning cosine
  (dedicated text embedder `nomic-embed-text`, §4) and takes the **top-k** — no fixed
  floor — then runs a bounded verify-before-answer agent loop that returns the verified
  matches or an honest "nothing found", the documented interactive exception to §9.1's
  "one call" rule (see §10). Still future here: multi-turn conversational memory and a
  learned reranker model.
- Postgres and pgvector if concurrent writes become a real constraint.
- Video support.
- A watch mode for `ivms777-sync` that uploads new files as they appear.
- **MCP server exposing the organized library, read-only (plan 11).** The
  counterpart to stage 2: instead of exporting a change plan to reshape the disk,
  expose the *organized* library over the Model Context Protocol so an external
  agent (Claude Desktop, a local agent) reads it live — `search`, ask-your-library,
  list memories/albums, get a photo with its metadata, get the export plan as a
  resource. Read-only and single-owner over stdio (no auth, matching §3.2); it
  goes through the app's read layer only (`app` serves reads, §5) and never writes
  disk or DB, so "source folders are sacred" (§3.2c) still holds. Hosted,
  multi-tenant MCP with per-owner tokens waits on the auth work above (v02).
- User-defined layouts, expressed as a path template over facets and tags.
- **Chat degradation now covers empty results, not only exceptions (done).** §10's
  rerank keeps a candidate whose `caption_vec` is not computed yet (signal-unavailable,
  not floored), so a partially-processed library no longer answers a false "no photos
  matching". An honest "no sources" now means the candidate pool truly has no caption
  match. The soft `_narrow` predicate this bullet originally shipped was superseded by
  plan 12's `hard_filters`/`soft_tags` split (§9.2) — see §10 steps 2–3.
- **One graceful retriever core shared by search, similar, chat, and memory
  (done, plan 12).** Retrieval used to be duplicated and inconsistent: similar
  degraded gracefully by scoring additively; chat's `_narrow` and rerank floor were
  hard gates that wiped everything when one signal was absent; `/library` search and
  memory each rolled their own fusion/retrieval calls. All four now sit on **one
  core**, `search/retriever.py` (§9.2): `candidates()` (fast, no LLM) then `refine()`
  (hard EXIF/date filter → additive graceful scoring, `search/scoring.py`). `/library`
  search and `/photo` similar call it directly, no agent, no per-click latency
  (§9.1) — `similar_photos` is now a thin wrapper over `refine(candidates())`. Chat's
  `retrieve()` (§10) calls the core's `candidates()` + hard-filter, then ranks the
  caption vectors itself and takes the top-k (no floor) rather than `refine()`, for the
  reason §10 explains (the core's fused rank is unconditional content, wrong for a chat
  seed; honest-empty is the agent's job); its outer
  "degrade, never crash" fallback (`chat/retrieve.py`) now also routes through the core's
  `candidates()`, so the last stray fusion is gone — fusion lives in exactly one place.
  Memory's event-composition `similar` tool (§11) also routes through the core. `/photo` (task
  3b, **done**) now paints instantly and loads the similar strip asynchronously via
  `GET /photo/{id}/similar` (§9.2, §13) — first paint never waits on the full-library
  scan. Still open: splitting that async fragment itself into **phase-1 KNN paint,
  phase-2 `refine()` swap** (§9.2) — today the fragment runs the whole
  `refine(candidates())` in one call; the finer two-stage split remains a follow-up.
