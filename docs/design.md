# Photo Library Organizer — Design

Status: approved design, ready for implementation planning
Date: 2026-08-13

## 1. Goal

A local Python service with a simple web UI. Point it at a folder of photos.
It classifies every photo, writes a description and tags, and lets you search,
filter, find similar photos, browse suggested groups, and ask questions about
the collection in plain language.

All inference runs on hardware you control. Photos never go to a third-party
API.

## 2. Non-goals (v1)

- Face detection and person clustering. Deferred to v2.
- Authentication, signup, per-user quotas, browser upload. See section 3.2.
- Editing, rating, or deleting photos.
- Cloud model APIs of any kind.
- Video files.

## 3. Constraints and targets

### 3.1 Deploy profiles

The same code and the same `docker-compose.yml` run in three places. The only
differences are which inference service is active and which model name is in
config.

| Profile | Inference | Caption model | Planner / chat model | Embed device |
|---|---|---|---|---|
| `mac` | Ollama on the **host** | `gemma4:26b-a4b` | `gemma4:e4b` | `cpu` |
| `jetson` | Ollama in a container | `qwen3-vl:4b` | `qwen3-vl:4b` | `cuda` |
| `cloud` | vLLM in a container, `--gpus all` | `gemma4:26b-a4b` | `gemma4:e4b` | `cuda` |

**Why Ollama runs on the host under `mac`.** Docker Desktop on macOS boots a
Linux VM, and Apple exposes no GPU to Linux guests — there is no Metal in a
container, and no configuration changes that. Apple's own `container` project
does not support GPU passthrough either. Containerised inference on a Mac falls
back to CPU and runs 3-6x slower. Ollama installed natively gets full Metal, and
the containerised app reaches it at `host.docker.internal:11434`.

On Linux this problem does not exist. Under `jetson` and `cloud` everything,
including inference, runs in containers with real GPU access via the NVIDIA
container runtime.

**Why Ollama and not vLLM on Mac and Jetson.** vLLM's Metal backend does not
support vision models, which is the entire workload here, and Docker's own
benchmarks put it 1.2-1.3x behind llama.cpp on Apple Silicon with far more
variance. On Jetson, vLLM's aarch64 build is a maintenance burden. vLLM earns
its place only under `cloud`, where continuous batching genuinely helps with
concurrent users.

**Jetson sizing.** The Orin Nano Super has 8 GB shared between CPU and GPU at
102 GB/s. After the OS, roughly 6 GB is usable, so the 26B A4B model does not
fit and a 4B-class model is used instead. This works only because pipeline
stages are sequential — SigLIP and the captioner are never resident at the same
time (see section 8). Expect roughly 6-8 s/photo, so about 8-11 hours for 5,000
photos at 25 W.

The default is `qwen3-vl:4b`, with `gemma4:e4b` as the alternate. Published
scores do not settle the choice: Qwen3-VL 4B's 67.4 is MMMU while Gemma 4 E4B's
52.6 is MMMU-Pro, a harder benchmark, so the two numbers are not comparable.
The phase 1 bake-off decides it on real photos.

### 3.2 Multi-tenancy

v1 ships as a single-user product with multi-user bones. There is **no auth, no
login, no user table, no admin role, no signup, no upload flow, and no quotas**.
There is one implicit owner ingesting from a local folder, and `owner_id` is a
constant.

But three things are cheap now and expensive to retrofit, so they are in from
the start:

- `owner_id` on every user-scoped row and in every query.
- Photo bytes reached through a `Storage` interface (local filesystem now,
  object storage later).
- Inference reached through one HTTP client, so swapping Ollama for vLLM is a
  config change.

When public multi-user access arrives it adds auth, upload, and quotas on top of
this. It does not require a data migration or a rewrite of every query.

### 3.3 Scale

- Collection: 1,000-5,000 photos for the first user.
- First full index may run overnight. It must be resumable and observable.
- Original files are never modified, moved, or renamed.

## 4. Models

| Role | Model | Where it runs |
|---|---|---|
| Image and text embeddings, zero-shot tags | SigLIP 2 `so400m-patch14-384` | in-process in the worker (`transformers`) |
| Captions and structured tags | Gemma 4, size per profile | inference service over HTTP |
| Query planning, chat answers | Gemma 4 E4B | inference service over HTTP |

Gemma 4 is the default family on `mac` and `cloud`; the Jetson profile runs a
4B-class model that may be Qwen3-VL. Caption and planner prompts therefore live
in a per-model template registry in `inference/prompts.py`, keyed by model name,
with a shared JSON schema all templates must satisfy. Adding a model means
adding a template, not touching the pipeline.

Gemma 3 is dominated on every axis — Gemma 4 12B beats Gemma 3 27B by roughly 20
MMMU-Pro points at a third of the memory — and is not used.

SigLIP 2 outperforms OpenCLIP on zero-shot and retrieval. It stays in-process
rather than behind the inference service because neither Ollama nor vLLM exposes
an image-embedding endpoint. On `mac` it runs on CPU inside the container:
roughly 1 s/photo, so about 1.5 hours one-time for 5,000 photos, and about 30 ms
per search query. That one-time cost is the price of containerising the app on a
Mac.

**Model download.** Caption and planner models are pulled by the inference
service on first use (`ollama pull`, or vLLM's Hugging Face fetch) — nothing to
script. SigLIP is fetched from Hugging Face into a mounted cache volume by a
startup step that checks free disk first and reports progress. Both caches
survive container restarts.

**Bake-off gate.** Phase 1 includes a script that runs candidate caption models
over the same 50 real photos and reports seconds per photo, memory high-water
mark, and side-by-side captions. On `mac` that is `gemma4:26b-a4b` against
`gemma4:12b`; on `jetson`, `qwen3-vl:4b` against `gemma4:e4b`. Because published
benchmarks for these pairs are either close or not directly comparable, the
bake-off is how the default is actually chosen. The winner becomes the default
in config, not in code.

## 5. Architecture

Three containers, one SQLite file.

```
                    ┌─────────────────────────────────┐
      browser ◄────►│ app        FastAPI + Jinja+HTMX │
                    │            /library /photo      │
                    │            /groups /chat /index │
                    └────────┬───────────────┬────────┘
                             │               │
                             │        ┌──────▼──────────────┐
                             │        │ inference           │
                             │        │ ollama | vllm       │
                             │        │ (host under `mac`)  │
                             │        └──────▲──────────────┘
                             │               │
                    ┌────────▼───────────────┴────────┐
                    │ worker                          │
                    │ scan, thumbs, SigLIP, taxonomy, │
                    │ caption, groups                 │
                    └────────┬────────────────────────┘
                             │
                    ┌────────▼────────────────────────┐
                    │ SQLite (WAL) + sqlite-vec + FTS5│
                    │ on a named volume               │
                    └─────────────────────────────────┘
```

`app` serves the UI and read queries. `worker` owns the ingest pipeline and is
the primary writer. Both open the same SQLite file in WAL mode with a busy
timeout — SQLite permits one writer at a time, and `app`'s writes are rare
(accepting a group, editing vocabulary), so contention stays negligible at this
scale. If public traffic ever makes that false, the fix is Postgres, and the
repository layer is the only thing that changes.

Vector search uses `sqlite-vec`, which ships prebuilt wheels for arm64 and
x86_64, so the same code works on Mac, Jetson, and cloud. At 5,000 photos a
brute-force scan would also be fine; `sqlite-vec` is there so growth to 100k+
needs no new component.

## 6. Data model

```sql
photos (
  id              INTEGER PRIMARY KEY,
  owner_id        INTEGER NOT NULL,
  path            TEXT NOT NULL,      -- storage key, not necessarily a fs path
  content_hash    TEXT NOT NULL,      -- sha256 of file bytes, detects moves
  phash           TEXT,               -- perceptual hash, near-duplicate groups
  bytes           INTEGER,
  width           INTEGER,
  height          INTEGER,
  mtime           REAL,
  shot_at         TEXT,               -- EXIF DateTimeOriginal, ISO-8601
  camera          TEXT,
  lens            TEXT,
  gps_lat         REAL,
  gps_lon         REAL,
  thumb_key       TEXT,
  caption         TEXT,
  caption_model   TEXT,
  embedding_model TEXT,
  exif_json       TEXT,               -- full EXIF as captured, for reference
  missing_since   TEXT,               -- file vanished; row and tags survive
  duplicate_of    INTEGER REFERENCES photos(id) ON DELETE SET NULL,
  created_at      TEXT NOT NULL,
  updated_at      TEXT NOT NULL,
  UNIQUE(owner_id, path)
);
CREATE INDEX photos_owner_hash ON photos(owner_id, content_hash);
CREATE INDEX photos_owner_shot ON photos(owner_id, shot_at);
CREATE INDEX photos_duplicate_of ON photos(duplicate_of);

-- EXIF-derived facets: exact, queryable, never model-guessed
photo_facets (
  photo_id   INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
  key        TEXT NOT NULL,           -- camera_make, iso, year, time_of_day, ...
  value_text TEXT,                    -- set for categorical facets
  value_num  REAL,                    -- set for numeric facets, enables ranges
  PRIMARY KEY (photo_id, key)
);
CREATE INDEX photo_facets_lookup ON photo_facets(key, value_text);
CREATE INDEX photo_facets_range  ON photo_facets(key, value_num);

-- sqlite-vec virtual table; rowid joins to photos.id
CREATE VIRTUAL TABLE photo_vec USING vec0(
  embedding float[1152]
);

tags (
  id        INTEGER PRIMARY KEY,
  dimension TEXT NOT NULL,
  label     TEXT NOT NULL,
  UNIQUE(dimension, label)
);

photo_tags (
  photo_id INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
  tag_id   INTEGER NOT NULL REFERENCES tags(id),
  score    REAL NOT NULL,             -- 0..1
  source   TEXT NOT NULL,             -- siglip | vlm | exif | pixel | user
  PRIMARY KEY (photo_id, tag_id, source)
);
CREATE INDEX photo_tags_tag ON photo_tags(tag_id);

jobs (
  photo_id   INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
  stage      TEXT NOT NULL,           -- thumbnail | embed | taxonomy | caption
  status     TEXT NOT NULL,           -- pending | running | done | failed
  attempts   INTEGER NOT NULL DEFAULT 0,
  error      TEXT,
  updated_at TEXT NOT NULL,
  PRIMARY KEY (photo_id, stage)
);
CREATE INDEX jobs_pending ON jobs(stage, status);

groups (
  id         INTEGER PRIMARY KEY,
  owner_id   INTEGER NOT NULL,
  kind       TEXT NOT NULL,           -- event | cluster | duplicate
  name       TEXT NOT NULL,
  params     TEXT,                    -- JSON, how it was generated
  status     TEXT NOT NULL,           -- suggested | accepted | dismissed
  created_at TEXT NOT NULL
);

group_photos (
  group_id INTEGER NOT NULL REFERENCES groups(id) ON DELETE CASCADE,
  photo_id INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
  rank     REAL,
  PRIMARY KEY (group_id, photo_id)
);

scans (
  id          INTEGER PRIMARY KEY,
  owner_id    INTEGER NOT NULL,
  root_path   TEXT NOT NULL,
  started_at  TEXT NOT NULL,
  finished_at TEXT,
  files_seen  INTEGER DEFAULT 0,
  files_new   INTEGER DEFAULT 0
);
```

Plus an FTS5 virtual table `photo_fts(caption, tags_text)` kept in sync by
triggers.

`tags` is a shared vocabulary and deliberately has no `owner_id`; ownership
comes from the joined photo. Every user-scoped query filters on `owner_id`, and
a repository-layer helper makes omitting it awkward.

Storing every tag with a `score` and a `source` lets the UI show why a tag is
present, and lets thresholds be tuned per dimension without re-running models.

## 6.1 Exact duplicates

The same image bytes often sit in several folders under different names. Those
are detected by `content_hash` and collapsed: one photo is **canonical**, every
other copy is a row with `duplicate_of` pointing at it.

Only canonical rows get job rows. An image stored in five places is embedded,
scored, and captioned once. Duplicates are hidden from the grid behind an `×N`
badge, and a `/duplicates` screen lists every path with the disk space the
redundant copies occupy. The screen is read-only — it reports, it never deletes.

Distinguishing a move from a duplicate requires knowing the whole live file set,
so scanning is two passes: hash everything, then reconcile. Same hash with the
original gone is a move; same hash with the original still present is a
duplicate. If a canonical file is deleted while a duplicate survives, the
surviving copy is promoted to canonical so its content still gets processed.

This is exact, byte-level matching. Visually similar but not identical shots are
near-duplicates and are handled by perceptual hashing in section 11.

## 6.2 EXIF facets — external filters

Model output is a guess. EXIF is a fact. The two are kept in separate tables so
the UI can be honest about which is which, and so a filter on "shot at ISO 3200"
is never diluted by a model's opinion.

Every photo's full EXIF is stored verbatim in `exif_json`. From it, a fixed set
of **facets** is derived into `photo_facets`, each either categorical
(`value_text`) or numeric (`value_num`, so ranges work):

| Group | Facets |
|---|---|
| Camera | `camera_make`, `camera_model`, `lens`, `software` |
| Exposure | `iso`, `aperture`, `shutter_speed`, `focal_length`, `exposure_bias`, `flash`, `exposure_program`, `metering_mode`, `white_balance` |
| Time | `year`, `month`, `weekday`, `hour`, `time_of_day` (night/dawn/morning/afternoon/evening), `is_weekend` |
| Place | `has_gps`, `gps_lat`, `gps_lon` |
| Image | `megapixels`, `orientation`, `aspect` (portrait/landscape/square) |

Facets are used four ways:

- **Filtering** — sidebar controls on `/library`, with counts. Categorical
  facets are checkboxes; numeric facets are ranges.
- **Sorting** — any numeric facet is a sort key, in addition to capture date.
- **Search** — facet predicates narrow the candidate set before semantic and
  keyword ranking (section 9).
- **Chat** — the query planner emits facet predicates, and retrieved photos
  carry their facts into the answer context, so "what lens did I use most in
  Italy?" is answered from data rather than from captions.

Facets are also a cheap quality signal the models never see: a photo at ISO
12800 with a 1/8 s shutter is probably a noisy handheld night shot, and that is
knowable without inference.

`time_of_day` uses local clock hour from EXIF, which is what a photographer
means by "evening shots". It is not recomputed from GPS and UTC.

## 7. Taxonomy

Ten dimensions, defined in an editable `vocab.yaml`. Users add or remove labels
without touching code. Re-scoring a changed dimension only re-runs the SigLIP
stage, which is cheap.

| Dimension | Example labels |
|---|---|
| `subject` | portrait, group of people, pet, food, architecture, nature, vehicle, document, artwork |
| `setting` | indoor, outdoor, beach, mountain, forest, city street, restaurant, home, office, water, snow |
| `vibe` | cozy, energetic, serene, moody, festive, nostalgic, dramatic, minimal, chaotic, romantic |
| `emotion` | joyful, sad, tense, affectionate, playful, contemplative, neutral |
| `light` | golden hour, blue hour, night, harsh midday, overcast, backlit, neon, candlelit |
| `season_weather` | summer, autumn, winter, spring, rain, snow, fog, clear sky |
| `composition` | close-up, wide shot, aerial, shallow depth of field, symmetry, silhouette, leading lines |
| `palette` | warm, cool, pastel, vivid, monochrome, dark, bright, high contrast |
| `occasion` | birthday, wedding, travel, hike, concert, holiday, everyday, work |
| `quality` | sharp, blurry, noisy, overexposed, underexposed |

Sources per dimension:

- `palette` and `quality` come from cheap pixel statistics first (mean
  saturation, Laplacian variance, histogram clipping). SigLIP refines them.
- All ten get SigLIP zero-shot scores against prompt templates
  (`"a photo with a {label} mood"`), sigmoid-scored, thresholded per dimension.
- Gemma 4 returns a JSON object with a caption plus its own picks from the same
  vocabulary, stored with `source='vlm'`.
- `shot_at`, `camera`, and GPS come from EXIF with `source='exif'`.

Thresholds start at sensible defaults and are tuned against a small hand-labeled
dev set of ~100 photos built during phase 2.

### 7.1 Vocabulary mining

The starting vocabulary cannot anticipate one particular library. A batch job
reads every caption, extracts recurring noun phrases, and drops any that an
existing label already covers (cosine similarity between SigLIP text embeddings
above a threshold). What remains is ranked by frequency and offered in the UI as
"suggested tags", each with the photos that triggered it.

Accepting a suggestion appends the label to `vocab.yaml` and queues a re-run of
the taxonomy stage for that dimension only. That stage is SigLIP-only, so a new
label costs seconds across the whole library, not hours. This is how the tag
vocabulary grows into the collection instead of being guessed up front.

## 8. Ingest pipeline

Stages run per photo, each recorded in `jobs`. The worker drains pending jobs
stage by stage. Killing the container and restarting resumes exactly where it
stopped.

1. **discover** — walk the root, skip non-images, read EXIF, hash file bytes.
   A matching `content_hash` at a new path is a move, not a new photo. A file
   that disappears gets `missing_since` set; its tags and embedding survive in
   case it comes back.
2. **thumbnail** — two sizes (320 px grid, 1600 px detail) written to storage.
   HEIC via `pillow-heif`.
3. **embed** — SigLIP 2 image embedding, batched, written to `photo_vec`.
4. **taxonomy** — SigLIP zero-shot scoring against `vocab.yaml`, plus pixel
   statistics. Fast, runs immediately after embed.
5. **caption** — Gemma 4 produces a caption sentence and a JSON tag object over
   HTTP. Slowest stage by two orders of magnitude, runs last.

**Stages are drained in order across the whole library, not per photo.** Every
photo is embedded and scored before any photo is captioned. This is what keeps
the Jetson profile viable: SigLIP is unloaded before the captioner is ever
asked for anything, so 8 GB never has to hold both. It also means search and
"show similar" work across the entire collection within minutes of starting a
scan, while captions fill in over the following hours.

Failed jobs retry up to 3 times with the error recorded, then stay `failed` and
are listed in the UI. One bad file never stalls the queue.

## 9. Retrieval

Four mechanisms, composed:

- **Semantic** — query text through the SigLIP text encoder, KNN via
  `sqlite-vec`. Handles "dogs playing in snow" with no matching caption.
- **Tag facets** — model-derived tag filters from the sidebar, plain SQL over
  `photo_tags` with a score threshold. AND across dimensions, OR within a
  dimension.
- **EXIF facets** — exact filters over `photo_facets` (section 6.2):
  categorical equality and numeric ranges. Applied before any ranking, since
  they are cheap and exact.
- **Keyword** — FTS5 BM25 over captions and tag text. Catches proper nouns,
  OCR'd text, and exact words embeddings smear over.
- **Fusion** — semantic and keyword results merged by reciprocal rank fusion
  (`score = sum 1/(60 + rank)`). Facet filters apply first, narrowing the
  candidate set, then fusion ranks what survives.

**Similar photos** — KNN against the clicked photo's embedding, with
near-duplicates (cosine > 0.98) collapsed behind a "show N near-duplicates"
toggle so the strip is not ten copies of one frame.

### 9.1 Query planner

Gemma 4 E4B converts a natural-language query into a `QuerySpec` in one call:

```
"moody shots of the dog at the beach last summer, shot wide open at night"
  -> {"semantic": "dog on a beach",
      "date_from": "2025-06-01", "date_to": "2025-08-31",
      "tags": {"vibe": ["moody"], "setting": ["beach"]},
      "facets": {"time_of_day": ["night"], "aperture": {"lte": 2.0}}}
```

The `facets` block maps directly onto `photo_facets` — categorical keys take a
list of accepted values, numeric keys take `gte`/`lte` bounds. Because these are
exact, a wrong facet guess is visible and removable as a chip rather than
silently skewing the ranking.

A single structured-output call, not an agent loop. A multi-step tool-calling
agent would cost seconds per step to query one SQLite table, which is a bad
trade.

The planner is strictly an enhancement. If it fails, times out, or returns
invalid JSON, the raw query goes straight to semantic + keyword fusion. The UI
shows the parsed filters as removable chips, so the interpretation is always
visible and correctable.

## 10. Ask-your-library chat

1. Question goes through the same planner to get a `QuerySpec`.
2. Retrieval returns the top 30 photos.
3. Context assembly builds a compact block per photo: id, date, caption, top
   tags, and its EXIF facts — camera, lens, ISO, aperture, shutter, focal
   length, coordinates when present. About 60 tokens each, ~2k tokens total.
   Including the facts means questions like "what lens did I use most on that
   trip?" are answered from data, not inferred from captions.
4. Gemma 4 answers, citing photos as `[photo:123]`.
5. The UI streams tokens over SSE and renders each citation inline as a
   clickable thumbnail.

The answer is grounded only in retrieved captions and tags. When retrieval
returns nothing above a relevance floor, the model is instructed to say so
rather than invent an answer. Captions are model-generated and imperfect, so the
chat view always shows its sources — the thumbnails are the evidence.

Chat and indexing share one inference service. Interactive requests use the
planner model, which is small and stays loaded, so a question during indexing
does not evict the captioner.

## 11. Suggested groups

Three generators, all cheap, all producing `suggested` rows the user accepts or
dismisses:

- **Events** — cluster `shot_at` by time gaps (new event after a gap > 6 h, or
  > 1 day for sparse periods). The most intuitive grouping, and it needs no
  model at all. Named from the dominant tags and date range, e.g. "Beach trip,
  12-14 July 2025".
- **Visual clusters** — HDBSCAN over SigLIP embeddings. Named by the most
  distinctive tags in the cluster relative to the whole library (TF-IDF style),
  so a cluster is "kitchen, cooking, warm" rather than "outdoor, sharp".
- **Near-duplicates** — perceptual hash buckets refined by cosine > 0.98.
  Presented as burst sets with the sharpest frame marked as the pick.

Groups regenerate on demand, not automatically. Dismissed suggestions stay
dismissed.

## 12. UI

- `/index` — pick a folder, start a scan, live progress per stage with counts,
  throughput, ETA, and a list of failed files. HTMX polling.
- `/library` — infinite-scroll thumbnail grid. Hover shows caption and top
  tags; an `×N` badge marks photos with exact duplicates. Left sidebar has two
  filter groups with counts: model-derived tags per dimension, and EXIF facets
  (camera, lens, ISO and aperture ranges, year, time of day, orientation). A
  sort control offers capture date or any numeric facet. Top bar has the search
  box and parsed-filter chips.
- `/photo/{id}` — large image, caption, tags grouped by dimension with scores
  and source badges, full EXIF panel, duplicate paths when any, and a "similar
  photos" strip.
- `/duplicates` — exact-duplicate sets with every path and the wasted disk
  space. Read-only.
- `/groups` — suggestion cards with a cover mosaic, accept/dismiss buttons,
  click through to a filtered grid.
- `/chat` — question box, streamed answer, inline thumbnail citations.

## 13. Layout

```
photolens/
  config.py            # pydantic settings, profile selection
  db/
    schema.sql
    migrations.py
    repo.py            # owner-scoped query helpers
  storage/
    base.py            # Storage protocol
    local.py
  inference/
    client.py          # OpenAI-compatible HTTP client (Ollama and vLLM)
    prompts.py         # caption, planner, chat templates
    fakes.py
  embedding/
    siglip.py
    fakes.py
  ingest/
    scanner.py         # walk, hash, duplicate detection
    exif.py            # full EXIF capture
    facets.py          # EXIF -> queryable facets
    thumbs.py
    taxonomy.py        # zero-shot + pixel stats
    caption.py
    worker.py          # job queue driver
  search/
    semantic.py
    facets.py
    keyword.py
    fusion.py
    planner.py
  groups/
    events.py
    clusters.py
    duplicates.py
    naming.py
  chat/
    context.py
    answer.py
  web/
    app.py
    routes/
    templates/
    static/
  vocab.yaml
compose.yaml
compose.mac.yaml       # profile overrides
compose.jetson.yaml
compose.cloud.yaml
tests/
docs/
```

Both Ollama and vLLM expose an OpenAI-compatible API, so one HTTP client covers
every profile. Only `base_url` and the model name differ.

## 14. Testing

- Inference and embedding sit behind protocols with deterministic fakes. The
  fake embedder returns a hash-derived unit vector, so similarity is
  reproducible. The whole pipeline, search, grouping, and chat context assembly
  test in milliseconds with no model weights and no network.
- Fixture images are generated with PIL at test time, including EXIF, so the
  repository carries no binary test data.
- Integration tests run the full pipeline over ~20 generated images against a
  temporary SQLite file with `sqlite-vec` loaded.
- Repository tests assert that every user-scoped query filters on `owner_id`,
  including a test that a second owner's photos never appear in the first
  owner's results.
- One optional, explicitly-marked test loads the real SigLIP model and asserts
  that a picture of a beach ranks above a picture of a keyboard for the query
  "beach". Skipped by default.
- Route tests use FastAPI's `TestClient` and assert on rendered HTML fragments.

## 15. Phases

| Phase | Delivers |
|---|---|
| 0 | Skeleton, config and profiles, compose files, SQLite schema with `sqlite-vec` and FTS5, storage and inference interfaces, fakes, test harness |
| 1 | Scan, hash, exact-duplicate collapsing, full EXIF capture and facet derivation, thumbnails, `/index` progress, `/library` grid with EXIF facet filters and sorting, `/duplicates`, caption model bake-off script |
| 2 | SigLIP embeddings, taxonomy scoring, semantic + facet + keyword + fusion search, similar photos, `/photo` detail |
| 3 | Captioning stage against the inference service, captions in the UI |
| 4 | Query planner, parsed-filter chips, caption vocabulary mining with tag suggestions |
| 5 | Event, cluster, and duplicate groups, `/groups` |
| 6 | Ask-your-library chat with streaming and citations |

Each phase leaves a working, useful application.

## 16. Risks

| Risk | Mitigation |
|---|---|
| SQLite single-writer contention under real multi-user load | WAL plus busy timeout is ample at v1 scale; the repository layer is the only thing that changes if Postgres becomes necessary |
| SigLIP on CPU makes the `mac` embed pass slow | ~1.5 h one-time for 5,000 photos, and it is a background stage; `cuda` on the other profiles |
| Jetson 8 GB cannot hold SigLIP and the captioner together | Stages drain library-wide in order, so the two are never resident at once |
| SigLIP zero-shot scores are poorly calibrated across dimensions | Per-dimension thresholds tuned against a ~100-photo hand-labeled dev set in phase 2 |
| Overnight indexing fails silently partway | Per-photo, per-stage job rows; resume on restart; failed files surfaced in the UI |
| HEIC and RAW files fail to open | `pillow-heif` for HEIC; RAW files are skipped in v1 and logged, not silently dropped |
| Captions are wrong and mislead chat answers | Chat always renders source thumbnails; captions display their model name |
| `sqlite-vec` behaves differently across arm64 and x86_64 | Integration tests run the real extension; it ships prebuilt wheels for both |

## 17. Future work

- Authentication, signup, browser upload, and per-user quotas for public access.
- Face detection and person clustering.
- Object storage backend behind the existing `Storage` interface.
- Optional XMP sidecar export so other tools see the tags.
- Offline reverse geocoding for place names.
- Postgres and pgvector if concurrent writes become a real constraint.
- Video support.
