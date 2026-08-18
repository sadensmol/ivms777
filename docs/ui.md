# UI — the app shell and per-route detail

The persistent shell, the resource bar, and what every route renders. Design §13
carries the shell concept, the nav order, and (authoritatively) the §13.1 navigation
model; this file is the per-route detail. Section references point into
`docs/design.md`; the navigation contract every drill-down obeys is design §13.1.

## App shell & resource bar

- **Persistent app shell** — the nav is a **full-width fixed header**; the content
  region **below it is the only scroll container** (the window never scrolls), so the
  page scrollbar starts under the menu and never crosses it. The four nav links
  (Upload/Library/Chat/Organize) are **HTMX-boosted** and swap **only `<main>`**, so
  the nav and its resource-bar polling stay resident across top-level navigation —
  the menu never re-renders and the resource bar never blanks. Boost is **scoped to
  these links only**: every grid↔leaf drill-down (photo, similar, prev/next) is a
  full navigation, so §13.1's bfcache-based `history.back()` is unaffected.
- **Resource bar** — live status at the **right of the nav row** on **every** page.
  It polls `GET /api/resources` (~2 s) and always renders the same shape:

  `RAM 7.1/7.4 GB · CPU 5% 51°C · GPU 99% 58°C · captioning · gemma4-E2B +vision`

- **The machine metrics are read by `app` itself, never proxied**
  (`models/resources.py::snapshot`). RAM and CPU come from `psutil`; GPU load from
  the kernel counter `/sys/devices/platform/gpu.0/load` on jetson (per mille — the
  same number `tegrastats` prints as `GR3D_FREQ`), `nvidia-smi` on cloud, `ioreg`
  (`AGXAccelerator` → `Device Utilization %`, no sudo) on mac; temperatures from the
  thermal zones `/sys/devices/virtual/thermal/thermal_zone*` (millidegrees — the
  same source as `tegrastats`' `cpu@…C`/`gpu@…C`). **None of this needs a CUDA
  context, a driver library or `runtime: nvidia`**, so reading it in `app` does not
  violate the one-model-process rule (§5.1) — verified on the board, the thin `app`
  container reads both paths with no extra mount. Sysfs rather than `tegrastats`
  because no container ships that binary. The consequence that matters: the bar
  **keeps showing RAM/CPU/GPU/temperature while the `models` service is down,
  restarting, or still loading** — which is exactly when you want to look at it.
- **CPU and GPU are ALWAYS shown**, load and temperature together so each number
  has an owner and the bar never changes shape as the box goes idle. An idle GPU
  reads `GPU 0%`, not a missing field. A genuinely absent sensor renders `—`
  rather than vanishing, because a field that disappears looks like a bug. Only
  `cpu-thermal` and `gpu-thermal` are shown (`tj-thermal`, the junction temperature
  throttling is keyed to, rides in the payload); the soc/cv zones are not — two
  numbers the user can act on, not nine. A zone that reads empty (the Orin's cv
  zones do) is **skipped, never reported as 0 °C**. Temperatures earn their place
  because sustained captioning holds the GPU at ~99% and is the hottest thing the
  board does (§3.1).
- **Only the model info comes over HTTP** — which models are resident and the op in
  flight, from the `models` service's own `GET /resources` (§5.1), the two things
  it alone knows. When it is unreachable those degrade to "no model info" and every
  other field is unaffected.

## Routes

- `/upload` — leads with the **folder list**: every folder in the library (§3.2c)
  with its photo count and a confirm-guarded **Delete from library** button (a folder
  mid-deletion shows "deleting…"). Below it, a **directory-only** picker adds a new
  folder; watch client-side hashing, the transfer, then live processing progress per
  stage with counts, **per-stage throughput** (`done/sec`, measured from recent
  `jobs.updated_at` so it is derived — not stored — and the last speed **survives a
  restart**), and failed files (Web Worker for hashing/upload, HTMX polling for
  processing). Every stage row carries its own **Reprocess** button that re-runs just
  that stage over the already-uploaded library without re-uploading — `thumbnail`,
  `embed`, `taxonomy` (re-tag), `caption embed` (the §9 caption vector, shown with a
  space), and a confirm-guarded `caption` (re-caption, the slow
  one); the worker drains the reset jobs.
- `/export` — choose a layout, preview the folder tree it would produce, and download
  the manifest. Shows whether the collection is fully processed, and the exact
  `ivms777-sync` command line to run next.
- `/library` — infinite-scroll thumbnail grid. Hover shows caption and top tags; an
  `×N` badge marks photos with exact duplicates. Left sidebar has two filter groups
  with counts: model-derived tags per dimension, and EXIF facets (camera, lens, ISO
  and aperture ranges, year, time of day, orientation). A sort control offers capture
  date or any numeric facet. Filters and sort **apply on change** — no Apply button —
  swapping only the grid via HTMX so the sidebar and its scroll position stay put; a
  single **Clear all filters** button at the top resets them. Top bar has the search
  box and parsed-filter chips.
- `/photo/{id}` — a full-screen view of one photo, always shown **inside the
  collection it was opened from** (the library with its filters/search/sort, an
  Organize album, or a memory). `‹`/`›` buttons and the ← / → arrow keys page to the
  previous/next photo **within that collection**, in its order (owner-scoped; the
  arrows are absent at the ends) — never leaking into another album or the wider
  library. The collection travels in a `ctx` URL parameter, which the route resolves
  to the ordered id list. The panel leads with the **collection's identity** — its
  title, description, and `N / M` position — shown on every photo of it, and only then
  the photo's own data. Closing returns to the collection's top-level grid (the
  album/memory/filtered library) with state and scroll intact via the browser's
  history; every in-photo nav (paging, similar) uses replace, so close always lands on
  the grid, never on a prior photo. The per-photo panel carries everything known about
  it: an **AI-written title and description**, the caption, tags grouped by dimension
  with scores and source badges (the "AI data"), the full EXIF panel — including GPS
  **coordinates**, which live here as a technical detail and nowhere else — every
  local path the file arrived from with the wasted-space total when there is more than
  one, and a "similar photos" strip. The photo itself and everything above render on
  first paint; the similar strip is the one expensive part (a full-library scan), so
  it loads **asynchronously** — the page ships a placeholder that fires
  `GET /photo/{id}/similar` on load and swaps in the finished strip (§9.2) — so opening
  a photo is never delayed by it. When the photo was opened within an album or memory,
  a **collage of that whole collection** — every photo in it, the current one
  highlighted — sits between the photo and the similar strip; each thumbnail opens in
  place (`replace`, §13.1) so paging stays within the collection, and the collage uses
  the **same tile size as the similar strip** so the two read as one gallery. The
  similar strip then **excludes any photo already in that collection**, so a member
  never appears twice. Each similar thumbnail is **enlarged and labelled with why it
  matched** — its top-3 reasons (shared tags / caption meaning / "looks alike" / "same
  time & place") with confidence percentages, one per line, **sorted by what actually
  drove the match**, not by the biggest percentage, overlaid on the image — so
  similarity is never a black box (§9). Opening a similar photo opens in
  a **"Similar to <this photo>"** layer (`ctx=similar:<id>`) that shows the base
  photo's thumbnail (clickable, to jump back to it) and pages within this photo's
  similar set. **A photo is always exactly one level below a grid** (§13.1): every
  photo→photo move — prev/next, opening a similar, the origin thumbnail — **replaces**
  history, so it stays `[grid, photo]` and **close always goes up to the grid** (the
  library for `library`/`q`/`similar:*`, the album for `album:*`), never replaying the
  chain of photos visited. The layer panel leads with the two photos' **own words —
  `Base` then `This`**, each with that photo's AI title, description, and caption, in
  the same order as the comparison table's columns, so *what* is being compared is read
  before *how* it scored (in this layer the leaf's own title/description/caption block
  is not repeated below the table). Then the **"Why similar — base vs this"** table:
  every facet that ACTUALLY SCORED (tag, caption meaning, visual, moment), both photos'
  values side by side and their match %, **sorted by how much each drove the match** —
  so a weak match is visibly weak rather than a mystery. Facets below their gate get no
  row at all: they contributed nothing, so the panel must not claim them. Sorting by
  match % instead is what used to headline two unrelated photos with `light: low light
  69%` — a big percentage is not a big reason. The panel also
  offers **per-photo reprocess** — *Re-tag* and a confirm-guarded *Re-caption* — that
  re-run just this photo's model stages (§8); thumbnails and embeddings are static and
  are not offered. The AI title/description and tags fill in with the caption and
  planner phases; until then the panel shows EXIF, sources, and embedding status. This
  is where duplicate paths are seen, since there is no separate duplicates screen.
- `/organize` — a dropdown of organization principles (date, memories, camera, place)
  over a list of album cards, each with a cover, title, description, and a strip of
  its photos. `date` shows a grain sub-selector (day / month / year, default month).
  Live principles recompute on selection; `memories` reads stored rows and offers a
  "Rebuild memories" control that queues the background build, showing a live
  `done/total (%)` indicator (HTMX polling) while it runs and reloading to the finished
  albums when it completes. The **last-opened organizer and grain are remembered** (a
  per-owner cookie), so loading `/organize` from the nav returns to the view you last
  used — memories, place, or a specific date grain — rather than snapping back to the
  default date view ("never lose the user's place").
- `/chat` — a normal chat view: a running **conversation history** of questions and
  their grounded answers, a text **input** at the bottom, a **processing indicator**
  while the model works, streamed answer tokens, and inline thumbnail citations.
  A `thought for … · HH:MM` line sits **above** each answer, in the order it
  happened: it is the **wait before the answer** — request in to first token out,
  covering routing, retrieval and any model swap — and deliberately **excludes
  streaming time**, since once tokens are arriving the user is reading, not waiting,
  and a long answer is not a slow one. The server announces it on its own `thinking`
  SSE event as the first token goes out, so it prints as the answer starts rather
  than after it ends, and it is persisted with the turn so a reload shows the same
  number. The log **follows the stream to the bottom** and scrolls once more when
  the turn finishes.
  History is **persisted** (`chat_sessions`/`chat_messages`, §6) and re-rendered
  server-side on load, so it survives navigation and restarts; a **New session** button
  starts a fresh conversation. Each question is grounded independently — retrieval runs
  per question, and the persisted history is the transcript, not multi-turn model
  memory. A cited thumbnail opens the photo as a **leaf of the chat grid** (`ctx=chat`,
  §13.1), so closing it returns to the conversation, not the library. A **"show me a
  memory"** answer additionally renders that memory as the **same Organize memory
  card** below the reply; opening a photo from it pages **within the memory**
  (`ctx=chat-memory:<key>`) and closes back to the conversation (§10, §13.1).

## Nav order

The nav order is **Upload → Library → Chat → Organize**: bring photos in, browse and
search them, ask about them to understand the collection, then — last, once you know
what you have — group and reorganize them. Organize is the final stage of the process
(it feeds stage 2, the on-disk reorg); chat is a review-and-understand tool, so it
sits before it.
