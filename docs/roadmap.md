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
| 7 | Stage 2 — the **Organize files…** button, `/organize/plan` preview, `/api/plan`, and the `ivms777-sync` CLI with apply, undo, and verify (plan 22) |

Each phase leaves a working, useful application.

Phase 7 is last because the plan is richer the more the library knows — the
organizer selected in Organize *is* the layout, so exporting memories needs the
agent from phase 5. But it depends on nothing after phase 3 (date, camera, and
place organizers export on their own), so it can be pulled forward if reorganizing
the disk matters more than chat does.

## Future work

- Authentication, signup, and per-user quotas for public access (v02).
- **Face detection and person clustering (done, plan 27).** The `faces` stage finds
  the faces of each photo with InsightFace `buffalo_l` in the models service, groups
  them by nearest group mean, and the owner names each group as a person (a saved
  voice) on `/voices` (design §4.1, §8, §13). The photo page and the film shot brief
  show the names. Still future: search, library filters and chat answers by person; a
  person page with a grid of all their photos (its own `ctx`, design §13.1); picking a
  film narrator from the people in a memory; and a re-cluster that keeps the owner's
  fixes.
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
- **Faster-than-realtime export.** Downloading a film is DONE (design §11.2), by
  recording the player's own canvas and audio bus with `MediaRecorder` — which
  keeps one renderer instead of two, but records in real time. Beating that means
  rendering frames on a virtual clock and muxing them against separately-rendered
  audio, i.e. shipping a muxer into the page. Worth it only once someone is
  exporting films long enough to mind the wait.

  The ffmpeg route was considered and rejected rather than deferred: the timeline
  JSON is a complete render script, but every one of the eight shot effects is
  GLSL, and a filter graph reproducing them would be a SECOND renderer that has to
  agree with the first frame for frame. That drift costs more than the wait does.
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
- A browser-side apply for stage 2, wherever the File System Access API is actually
  usable (Chromium desktop over HTTPS/localhost only — see design §3.2b).
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
