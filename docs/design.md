# Photo Library Organizer — Design

Status: approved design, ready for implementation planning
Date: 2026-08-13

## 1. Goal

A Python service with a simple web UI, deployed on a GPU box you control. It
works in two stages.

**Stage 1 — understand, in the cloud.** You select photos in the browser and
upload them. The service classifies every photo, writes a description and tags,
and lets you search, filter, find similar photos, browse suggested groups, and
ask questions about the collection in plain language.

**Stage 2 — reorganize, on your machine.** When the collection is fully
processed, `ivms777-sync` — a small standalone CLI that runs on the machine
holding the photos — downloads the collection state and applies it to disk: it
sorts files into a chosen folder layout, renames them consistently, and sweeps
redundant copies aside. It plans first and shows you every operation before it
touches anything.

The two stages share nothing but a manifest. The cloud never reaches into your
filesystem; the CLI never needs your library uploaded a second time.

All inference runs on hardware you control. Photos never go to a third-party
API.

## 2. Non-goals (v1)

- Face detection and person clustering. Deferred to v2.
- Authentication, signup, and per-user quotas. Deferred to v02; see section 3.2.
- Editing or rating photos in the cloud UI. The only thing that ever writes to
  your disk is `ivms777-sync`, and only on explicit confirmation.
- Cloud model APIs of any kind.
- Video files.

## 3. Constraints and targets

### 3.1 Deploy profiles

The same code and the same `docker-compose.yml` run in three places. The only
differences are which inference service is active and which model name is in
config. Ingest is identical everywhere — photos always arrive by upload
(section 3.2b), so no profile depends on a host bind mount.

| Profile | Inference | Caption model | Planner / chat model | Embed device |
|---|---|---|---|---|
| `mac` | Ollama on the **host** | `qwen2.5vl:7b` | `qwen2.5:3b` | `cpu` |
| `jetson` | Ollama in a container | `qwen2.5vl:3b` | `qwen2.5:3b` | `cuda` |
| `cloud` | vLLM in a container, `--gpus all` | `qwen2.5vl:7b` | `qwen2.5:3b` | `cuda` |

The caption model must be **vision-capable**. The tags above are the shipping
defaults in `config.py` — real, currently-pullable Ollama models — overridable
with `IVMS777_CAPTION_MODEL` / `IVMS777_PLANNER_MODEL`. The "Gemma 4" family named
in the rationale below (§4) is the intended target once it is available on Ollama;
until then Qwen2.5-VL is the working default.

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
login, no user table, no admin role, no signup, and no quotas**. There is one
implicit owner, and `owner_id` is a constant. Deploy it behind a private URL or
a reverse proxy that requires a password.

But three things are cheap now and expensive to retrofit, so they are in from
the start:

- `owner_id` on every user-scoped row and in every query.
- Photo bytes reached through a `Storage` interface (local filesystem now,
  object storage later).
- Inference reached through one HTTP client, so swapping Ollama for vLLM is a
  config change.

v02 adds accounts, signup, and quotas on top of this. Because upload is already
the only ingest path and every row is already owner-scoped, that is an additive
change: no data migration, no rewrite of every query.

### 3.2b Getting photos in — upload

Photos arrive by browser upload. There is no host bind mount, no server-side
folder picker, and no path the server walks. The server never sees your
filesystem; it sees the bytes you chose to send.

The upload screen accepts a whole directory (`<input webkitdirectory>`) or a
drag-and-drop selection, and runs in three steps:

1. **Hash locally.** A Web Worker computes each file's SHA-256, one file at a
   time, off the main thread. WebCrypto has no incremental digest, so a file is
   read whole — handling them one at a time bounds memory to the largest single
   photo rather than the size of the selection. Nothing is sent yet.
2. **Ask what is new.** The client POSTs each hash *with the path it came from*
   to `/api/upload/probe`, which answers with the subset whose bytes the server
   has never seen. Paths for content it already holds are recorded on the spot —
   that is what keeps duplicate detection correct when no bytes move.
   Re-uploading a folder you already sent transfers nothing.
3. **Send the new bytes.** Files upload in bounded concurrent batches with a
   per-file progress row. The relative path from the selected root travels
   alongside each file and is recorded, because that path is what stage 2 needs
   to find the file again on disk.

Interruptions are cheap: closing the tab loses only in-flight files, and
restarting the upload re-probes and resumes with what is still missing. The
server verifies the hash of every received file before storing it, so a
truncated transfer is rejected rather than indexed.

Originals are kept. They are written through the `Storage` interface under a
content-addressed key and are needed for full-resolution viewing and for
re-processing the library when a better model arrives.

**Why upload rather than a host mount.** A mounted folder is faster and copies
nothing, but it only works when the app runs on the same machine as the photos.
That constraint defeats the point of a hosted GPU. Upload costs disk and a first
transfer; in exchange, ingest is one code path on every profile, the container
boundary needs no holes, and the same deployment serves a user who is nowhere
near it.

**Why not have the browser reorganize files directly.** The File System Access
API can write to a user-picked directory, but only on Chromium desktop — Firefox
has no directory picker, Safari is read-only, and mobile has neither. Stage 2 is
a CLI instead, which works on every OS and has no browser to disagree with.

### 3.3 Scale

- Collection: 1,000-5,000 photos for the first user.
- First upload of 5,000 photos is a background transfer measured in tens of
  minutes; the first full index may run overnight. Both must be resumable and
  observable.
- The cloud never modifies an uploaded original. Only `ivms777-sync` writes to
  your disk, only after you confirm a plan, and every operation it performs is
  journaled and reversible (section 12).

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

Three containers and one SQLite file in the cloud. One standalone binary on your
machine.

```
  ── your machine ─────────────┐   ── the GPU box ───────────────────────────┐
                               │                                             │
   photos on disk              │       ┌─────────────────────────────────┐   │
        │                      │       │ app        FastAPI + Jinja+HTMX │   │
        │  ┌──────────┐ upload │       │            /upload /library     │   │
        └─►│ browser  │────────┼──────►│            /photo /organize /chat │ │
           └──────────┘        │       └────────┬───────────────┬────────┘   │
        ▲                      │                │               │            │
        │                      │                │        ┌──────▼─────────┐  │
        │  ┌────────────────┐  │ manifest       │        │ inference      │  │
        └──┤ ivms777-sync │◄─┼────────────────┤        │ ollama | vllm  │  │
           │  plan / apply  │  │  GET /api/     │        │ (host on `mac`)│  │
           └────────────────┘  │    manifest    │        └──────▲─────────┘  │
                               │       ┌────────▼──────────────┴─────────┐   │
                               │       │ worker                          │   │
                               │       │ thumbs, SigLIP, taxonomy,       │   │
                               │       │ caption, groups                 │   │
                               │       └────────┬────────────────────────┘   │
                               │                │                            │
                               │  ┌─────────────▼───────┐  ┌──────────────┐  │
                               │  │ SQLite (WAL)        │  │ Storage      │  │
                               │  │ + sqlite-vec + FTS5 │  │ originals +  │  │
                               │  │ on a named volume   │  │ thumbnails   │  │
                               │  └─────────────────────┘  └──────────────┘  │
                               └─────────────────────────────────────────────┘
```

`app` serves the UI, read queries, upload receipt, and the manifest endpoint.
`worker` owns the ingest pipeline and is the primary writer. Both open the same
SQLite file in WAL mode with a busy timeout — SQLite permits one writer at a
time, and `app`'s writes are small and short (recording a received upload,
accepting a group, editing vocabulary), so contention stays negligible at this
scale. If public traffic ever makes that false, the fix is Postgres, and the
repository layer is the only thing that changes.

`ivms777-sync` is not part of the deployment. It talks to `app` over HTTP,
reads one endpoint, and is the only component that ever writes to your disk.

Vector search uses `sqlite-vec`, which ships prebuilt wheels for arm64 and
x86_64, so the same code works on Mac, Jetson, and cloud. At 5,000 photos a
brute-force scan would also be fine; `sqlite-vec` is there so growth to 100k+
needs no new component.

## 6. Data model

```sql
-- one row per distinct image, keyed by its bytes
photos (
  id              INTEGER PRIMARY KEY,
  owner_id        INTEGER NOT NULL,
  content_hash    TEXT NOT NULL,      -- sha256 of file bytes; the identity
  storage_key     TEXT NOT NULL,      -- where the original lives in Storage
  phash           TEXT,               -- perceptual hash, near-duplicate groups
  bytes           INTEGER,
  width           INTEGER,
  height          INTEGER,
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
  created_at      TEXT NOT NULL,
  updated_at      TEXT NOT NULL,
  UNIQUE(owner_id, content_hash)
);
CREATE INDEX photos_owner_shot ON photos(owner_id, shot_at);

-- every local path these bytes arrived from; >1 row means a duplicate on disk
photo_sources (
  id          INTEGER PRIMARY KEY,
  photo_id    INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
  upload_id   INTEGER NOT NULL REFERENCES uploads(id) ON DELETE CASCADE,
  rel_path    TEXT NOT NULL,          -- path relative to the selected root
  filename    TEXT NOT NULL,
  mtime       REAL,
  UNIQUE(photo_id, rel_path)
);
CREATE INDEX photo_sources_photo ON photo_sources(photo_id);

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
  id          INTEGER PRIMARY KEY,
  owner_id    INTEGER NOT NULL,
  kind        TEXT NOT NULL,          -- event | cluster | duplicate | memory
  name        TEXT NOT NULL,          -- AI-written title for a memory
  description TEXT,                   -- AI-written story for a memory
  params      TEXT,                   -- JSON, how it was generated; carries the
                                      -- library signature a memory was built from
  status      TEXT NOT NULL,          -- suggested | accepted | dismissed
  created_at  TEXT NOT NULL
);

group_photos (
  group_id INTEGER NOT NULL REFERENCES groups(id) ON DELETE CASCADE,
  photo_id INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
  rank     REAL,
  PRIMARY KEY (group_id, photo_id)
);

uploads (
  id            INTEGER PRIMARY KEY,
  owner_id      INTEGER NOT NULL,
  root_label    TEXT NOT NULL,         -- the folder name the user picked
  started_at    TEXT NOT NULL,
  finished_at   TEXT,
  files_offered INTEGER DEFAULT 0,     -- hashes probed
  files_sent    INTEGER DEFAULT 0,     -- bytes actually transferred
  files_failed  INTEGER DEFAULT 0
);

-- Persisted chat transcript (§10). A session groups a conversation; "New
-- session" starts a fresh one. The current session is the owner's latest.
chat_sessions (
  id         INTEGER PRIMARY KEY,
  owner_id   INTEGER NOT NULL,
  created_at TEXT NOT NULL
);

chat_messages (
  id         INTEGER PRIMARY KEY,
  session_id INTEGER NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
  question   TEXT NOT NULL,
  answer     TEXT NOT NULL,
  sources    TEXT,                     -- JSON array of cited photo ids
  created_at TEXT NOT NULL
);
```

Plus an FTS5 virtual table `photo_fts(caption, tags_text)`, its rows keyed by
`photos.id` and refreshed by the taxonomy and caption stages. Because `tags_text`
is derived from many `photo_tags` rows, the stage rebuilds the row explicitly
(delete-then-insert) rather than through per-row triggers.

`tags` is a shared vocabulary and deliberately has no `owner_id`; ownership
comes from the joined photo. Every user-scoped query filters on `owner_id`, and
a repository-layer helper makes omitting it awkward.

Storing every tag with a `score` and a `source` lets the UI show why a tag is
present, and lets thresholds be tuned per dimension without re-running models.

## 6.1 Exact duplicates

The same image bytes often sit in several folders under different names. Because
a photo is identified by its `content_hash`, this needs no detection pass at
all: the second copy simply adds a row to `photo_sources` against the photo that
already exists. An image stored in five places is one `photos` row with five
sources, so it is embedded, scored, and captioned exactly once, and its bytes
are stored once.

This is what makes upload the cheaper ingest model in practice. The client
probes hashes before sending (section 3.2b), so redundant copies cost one hash
comparison each and no transfer.

There is no separate duplicates screen. A photo with more than one source wears
an `×N` badge in the grid, and its detail page lists every path the bytes were
found at, with the disk space the redundant copies occupy on **your** machine.
Reclaiming that space is stage 2's job, and only after you approve a plan
(section 12).

The earlier design detected duplicates by walking a mounted filesystem, which
required a two-pass scan to tell a move from a copy. Nothing walks a filesystem
now, so that distinction disappears: the client sends what exists, and paths
that stop appearing in later uploads are simply stale sources, marked rather
than deleted.

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
| Place | `has_gps`, `gps_lat`, `gps_lon`, `place_city`, `place_country` (reverse-geocoded, §11) |
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

0. **receive** — `app` accepts an uploaded file, verifies its SHA-256 against
   the hash the client declared, stores the original under a content-addressed
   key, reads EXIF, and inserts the `photos` row plus its `photo_sources` row.
   A hash that already exists adds only the source row and queues nothing. This
   stage runs in `app`, synchronously with the request, and is the only work not
   driven by the job queue — everything after it is.
1. **facets** — derive queryable facets from the stored EXIF (section 6.2).
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
"show similar" work across the entire collection within minutes of the upload
finishing, while captions fill in over the following hours.

Failed jobs retry up to 3 times with the error recorded, then stay `failed` and
are listed in the UI. One bad file never stalls the queue.

A file rejected at **receive** — hash mismatch, unreadable image, unsupported
format — never becomes a `photos` row. It is counted in `uploads.files_failed`
and reported to the client, which lists it on the upload screen so a failed
transfer is visible rather than silently missing.

**Reprocessing.** Originals are kept (§3.2b), so the derived state — thumbnails,
embeddings, tags, captions — can be rebuilt without re-uploading. `POST
/reprocess` resets a **range** of stages (`from_stage` through an optional
`to_stage`, inclusive) to `pending` for the owner's photos; the `worker` re-runs
them in `STAGES` order on its next poll. The `/upload` UI exposes **two** buttons:

- **Reprocess all photos** — `from=thumbnail` bounded `to=taxonomy`: rebuilds
  thumbnails, embeddings, and tags but **not captions**. Captioning is the slow
  stage (hours), and an already-captioned image never needs re-captioning because
  the bytes are static — so the everyday reprocess deliberately stops before it.
- **Re-caption all photos** — `from=caption`, styled as a destructive action and
  guarded by a confirm ("can take hours"). Only needed after switching the caption
  model. This is the one path that re-runs the vision model over the whole library.

The endpoint accepts any range, so a narrower re-run — `from=taxonomy` after a
`vocab.yaml` change, `from=embed` for a new embedding model — is one POST away.
Every stage handler is idempotent — it
overwrites its own output — so a reprocess is safe to trigger at any time.
Self-healing backfills run automatically each drain: they queue `thumbnail` for a
photo still without one, `embed` for one missing a vector, `taxonomy` for an
embedded-but-untagged photo, and `caption` for a thumbnailed-but-uncaptioned one —
so a library predating a stage, or a photo whose thumbnail once failed, heals with
no manual action. A photo that genuinely can't be thumbnailed is skipped by the
later stages rather than crashing them.

**The manifest gate.** Stage 2 must not run against a half-processed library, so
`/api/manifest` reports the collection as `complete` only when no `pending` or
`running` job rows remain for the owner. It still serves a manifest while
incomplete, marked as such, and `ivms777-sync` refuses to `apply` one unless
`--allow-incomplete` is passed.

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

Concretely, a free-text query is planned once and its predicates are
materialized into the same filter params the sidebar uses (`f_`/`n_`/`t_`, plus
`date_from`/`date_to` over `shot_at`); the chips are those params, so removing a
chip simply drops a predicate and re-runs the ordinary filtered search — the
planner does not run again until a new query is typed.

A single structured-output call, not an agent loop. A multi-step tool-calling
agent would cost seconds per step to query one SQLite table, which is a bad
trade. This applies to *interactive* retrieval only. The Memories organizer
(section 11) does run an agent loop, because it is a batch, offline job whose
per-step latency is paid once at build time, not on every keystroke.

The planner is strictly an enhancement. If it fails, times out, or returns
invalid JSON, the raw query goes straight to semantic + keyword fusion. The UI
shows the parsed filters as removable chips, so the interpretation is always
visible and correctable.

## 10. Ask-your-library chat

1. The question drives retrieval directly through semantic + keyword fusion
   (§9) — the same path interactive search uses. The query planner (§9.1) is a
   future enhancement; when it lands the question will pass through it first to
   get a `QuerySpec`. Until then the raw question is retrieved as-is.
2. Retrieval returns the top 30 photos.
3. Context assembly builds a compact block per photo: id, date, caption, top
   tags, and its EXIF facts — camera, lens, ISO, aperture, shutter, focal
   length, coordinates when present. About 60 tokens each, ~2k tokens total.
   Including the facts means questions like "what lens did I use most on that
   trip?" are answered from data, not inferred from captions.
4. Gemma 4 answers, citing photos as `[photo:123]`.
5. The UI streams tokens over SSE and renders each citation inline as a
   clickable thumbnail.

**Off-topic guard.** Before retrieving anything, a one-word classifier decides
whether the question is actually about the photo collection. A question that is
not — general advice, trivia, "should I walk or drive?" — skips retrieval
entirely and gets a short "I can only answer questions about your photos" reply,
so unrelated questions never dump the library as false evidence. Only questions
that pass the gate reach retrieval.

The answer is grounded only in retrieved captions and tags. When retrieval
returns nothing above a relevance floor, the model is instructed to say so
rather than invent an answer. Captions are model-generated and imperfect, so the
chat view always shows its sources — the thumbnails are the evidence.

The view reads like a normal chat: each question and its grounded answer are kept
on the page as a running conversation, a processing indicator shows while the
model works, and the input stays pinned at the bottom.

**History is persisted.** Each answered turn (question, full answer, cited photo
ids) is written to `chat_messages` under the owner's current `chat_sessions` row
(§6). On load, `/chat` renders the current session's turns server-side as static
history, so switching away and back — or restarting the app — keeps the
conversation. A **New session** button opens a fresh, empty session; older
sessions stay in the database. Persistence is the visible transcript only: each
question is still answered independently against freshly retrieved photos, not
against prior turns — there is no multi-turn model memory.

Chat and indexing share one inference service. The chat route calls the planner
model directly, which is small and stays loaded, so a question during indexing
does not evict the captioner.

## 11. Organize — albums by principle

The **Organize** tab groups the library into albums by a principle the user
picks from a dropdown. Most organizers recompute their albums live — the grouping
is a view over the current data, so it is always up to date, with no
accept/dismiss step and nothing persisted. **Memories** is the one exception: it
runs an agent, is far too slow to recompute per view, so it is built in the
background and stored (below).

An **organizer** is a function from the library to a list of albums. Each album
has a title, a description, a cover, and its photos. The organizers live in
`albums/` behind an `Organizer` protocol, so a new principle is one module and
one dropdown entry.

**By date.** A `grain` sub-selector picks the calendar bucket — `day`, `month`
(default), or `year`. It is a query parameter on `/organize` alongside the
organizer name and needs only EXIF, no model. Title is the period ("14 Jun 2025",
"June 2025", "2025"); description is the photo count and dominant camera — "140
photos, mostly Canon EOS R6."

Day, month, and year are the **same axis at three zooms** — a real hierarchy. There
is deliberately no "events" grouping here: time-gap clustering is a different
principle, not a zoom level of the calendar, so it never sits in this sub-selector.
It lives on only as an internal seeding step for Memories (below).

**By camera / device.** Group by EXIF camera model. Exact, trivial.

**By place.** Group photos by where they were taken and title each album with a
**real place name** — "Kyiv", "Rome", "Toronto" — from offline reverse geocoding
of the GPS, **never raw coordinates**. A bundled GeoNames dataset (`reverse_geocoder`)
resolves each point on the box, with no network, so one city is one album. Only
photos carrying GPS appear. Coordinates are a technical detail, shown only on
`/photo` (§13) and never in Organize — bare lat/long is not a place a person
recognizes. The library also filters by place: the facets stage reverse-geocodes
GPS into `place_city`/`place_country` facets (§6.2), shown as a "Place" group in
the `/library` sidebar.

**Memories.** The interesting one, and the reason the Organize tab needs the LLM.
It composes the library into named, described *memories* — "Family night in
Ontario", "A family having fun, 22 Nov 1999" — not sets of look-alike photos. It
replaces the earlier "By similarity (visual clusters)" organizer, which grouped by
raw SigLIP cosine and produced palette-alike blobs with no meaning. It runs as a
background job in three steps:

1. **Seed** — form candidate clusters cheaply, no model. Start from capture-time
   gaps (a gap > 6 h starts a new run), then split or merge by GPS bucket (~1 km)
   and SigLIP similarity. A candidate is a time-and-place-contiguous run of related
   photos — the natural boundary of a memory.
2. **Curate with an agent** — for each candidate, the planner model (Gemma 4 E4B,
   which stays loaded) reads the photos' captions, tags, and EXIF facts (date,
   place, camera). It may pull bounded extra context through a small tool set —
   similar photos, facet lookups, photos near in time — then judges whether the
   candidate is one coherent memory, may split or merge it, and writes a **title
   and a story description grounded in that data**. Retrieval-augmented and
   agentic, but capped at a few tool calls per candidate so cost stays bounded.
   This is the batch agent loop section 9.1 permits.
3. **Persist** — accepted memories are written to `groups` (`kind='memory'`, with
   `name` and `description`) and `group_photos` (cover is the lowest-`rank` row).
   `/organize?by=memories` then reads stored rows and renders instantly, with no
   LLM on the page load.

Memories need captions (phase 3) and the planner (phase 4), so the organizer lands
in phase 5. It depends on nothing interactive; it is pure offline enrichment.

**Rebuilding.** The build job records the library signature it built from — the
owner's photo count and newest `updated_at` — in each memory's `params`. A rebuild
is triggered manually ("Rebuild memories") or when the library changes, and is
skipped when the current signature already matches, so opening the tab never
silently re-runs the agent. The other organizers, being live, need no such guard.

The `groups`/`group_photos` tables — reserved and unused in the earlier design —
now back Memories. They remain available for a future "save this album" action on
the live organizers.

## 12. Stage 2 — the local sync tool

Stage 1 learns what the photos are. Stage 2 acts on it, on the machine that
holds them. The whole contract between the two is one JSON document.

### 12.1 The manifest

`GET /api/manifest?layout=date` returns, for every photo the owner has, its
content hash, the path the chosen layout says it belongs at, and every local
path it was uploaded from.

```json
{
  "manifest_version": 1,
  "generated_at": "2026-08-13T09:12:44Z",
  "layout": "date",
  "complete": true,
  "photo_count": 4812,
  "files": [
    {
      "hash": "9f2c1a…",
      "target": "2024/2024-06 June/2024-06-14_183012_IMG_4471.jpg",
      "sources": ["Pictures/iphone dump/IMG_4471.jpg",
                  "Desktop/to sort/IMG_4471 copy.jpg"],
      "bytes": 3841122
    }
  ]
}
```

`complete` is false while any job row is still pending or running (section 8).
`sources` carries every path this content arrived from; the first entry is the
copy the plan will keep, chosen as the shallowest path and then the
lexicographically smallest, so the result is stable across runs.

The manifest is derived state. Regenerating it with a different layout produces
a different `target` for every file and nothing else changes.

### 12.2 Layouts

A layout is a pure function from a photo's facts to a relative path. It sees
EXIF facets, tags, captions, and group membership, and it may use none of them.

```python
class Layout(Protocol):
    name: str
    def target(self, photo: PhotoView) -> PurePosixPath: ...
```

Three ship in v1. `date` is the default.

**`date`** — a year/month tree, filenames prefixed with capture time.
Depends only on EXIF, so it is completely stable: re-running it after new
captions or a retrained model produces byte-identical output.

```
2024/2024-06 June/2024-06-14_183012_IMG_4471.jpg
2025/2025-01 January/2025-01-03_101533_DSC_0088.jpg
_undated/9f2c1a3e_scan012.jpg
```

**`date-tags`** — the same tree holds every real file, plus an `_albums/`
directory of symlinks grouped by the strongest tags. One copy of the bytes,
many ways in. Where symlinks are unavailable the tool reports it and writes the
date tree alone rather than duplicating files.

```
2024/2024-06 June/2024-06-14_183012_IMG_4471.jpg
_albums/beach/2024-06-14_183012_IMG_4471.jpg -> ../../2024/2024-06 June/…
```

**`flat`** — one directory, every file named by capture time. For people who
search rather than browse and want no tree at all.

Photos with no usable capture date go to `_undated/`, named by a short hash
prefix so the name is stable. When two photos would land on the same path, the
later one gains an `_<hash8>` suffix; the choice is deterministic, so a re-run
does not shuffle names.

Layouts live server-side in `ivms777/organize/`. Adding one is a new module
and a new option on `/export` — the CLI needs no change, because it only
executes paths it is handed.

### 12.3 The CLI

```
ivms777-sync plan   --url https://photos.example --root ~/Pictures \
                      --layout date -o plan.json
ivms777-sync apply  plan.json
ivms777-sync undo   .ivms777-sync/journal-20260813T091244Z.jsonl
ivms777-sync verify --url https://photos.example --root ~/Pictures
```

**`plan`** fetches the manifest, walks `--root`, hashes every file it finds, and
matches by hash — never by path or filename, so a library reorganized since
upload still matches perfectly. It writes `plan.json` and prints a summary:

```
  4,812 photos in manifest
  4,796 matched on disk
     16 in manifest but not found locally      (left alone)
    241 files on disk not in manifest          (left alone)

  3,104 to move          e.g. Pictures/iphone dump/IMG_4471.jpg
                           -> 2024/2024-06 June/2024-06-14_183012_IMG_4471.jpg
  1,692 already in place
    387 redundant copies -> _duplicates/       (reclaims 4.1 GB)
      0 conflicts

  nothing has been changed. run: ivms777-sync apply plan.json
```

**`apply`** executes that plan and nothing else. It re-hashes each file
immediately before touching it and skips any that changed since planning.

**`undo`** replays the journal backwards, returning every file to where it was.

**`verify`** hashes the root and reports how it differs from the manifest,
changing nothing. It is `plan` without the output file.

### 12.4 Safety

The tool moves other people's photographs, so its defaults are paranoid.

- **Nothing is deleted, ever.** Redundant copies move to `_duplicates/` under
  their original relative path. Reclaiming the space is a folder the user
  deletes when they are satisfied.
- **Nothing outside the manifest is touched.** Files the manifest does not know
  are counted and reported, never moved.
- **Every operation is journaled before it runs.** `.ivms777-sync/journal-<ts>.jsonl`
  gets one record per operation with its status updated after. A crash mid-run
  leaves a journal that `undo` can replay.
- **Moves prefer `os.rename`.** Within one filesystem a move is atomic. Across
  filesystems it is copy, fsync, verify the hash, then unlink — the original
  goes only after the copy is proven good.
- **Conflicts stop the plan, not the apply.** If a target path is occupied by a
  file that belongs elsewhere, `plan` orders the moves so the occupant leaves
  first, routing through a temporary name when the moves form a cycle. A
  conflict it cannot order is reported and that file is skipped.
- **Plans expire.** A plan records the manifest's `generated_at` and the root it
  was built against. `apply` refuses a plan built for a different root, and
  warns when the manifest has since changed.

## 13. UI

- `/upload` — pick a folder or drop files, watch client-side hashing, then the
  transfer, then live processing progress per stage with counts, throughput,
  ETA, and a list of failed files. Hashing and upload progress come from the
  Web Worker; processing progress is HTMX polling. Two reprocess buttons re-run
  work over the already-uploaded library without re-uploading: **Reprocess all
  photos** (thumbnails, embeddings, tags — captions kept, since images are static)
  and a separate, confirm-guarded **Re-caption all photos** for when the caption
  model changes; the worker drains the reset jobs.
- `/export` — choose a layout, preview the folder tree it would produce, and
  download the manifest. Shows whether the collection is fully processed, and
  the exact `ivms777-sync` command line to run next.
- `/library` — infinite-scroll thumbnail grid. Hover shows caption and top
  tags; an `×N` badge marks photos with exact duplicates. Left sidebar has two
  filter groups with counts: model-derived tags per dimension, and EXIF facets
  (camera, lens, ISO and aperture ranges, year, time of day, orientation). A
  sort control offers capture date or any numeric facet. Filters and sort
  **apply on change** — no Apply button — swapping only the grid via HTMX so the
  sidebar and its scroll position stay put; a single **Clear all filters** button
  at the top resets them. Top bar has the search box and parsed-filter chips.
- `/photo/{id}` — a full-screen view of one photo, always shown **inside the
  collection it was opened from** (the library with its filters/search/sort, an
  Organize album, or a memory). `‹`/`›` buttons and the ← / → arrow keys page to
  the previous/next photo **within that collection**, in its order (owner-scoped;
  the arrows are absent at the ends) — never leaking into another album or the
  wider library. The collection travels in a `ctx` URL parameter, which the route
  resolves to the ordered id list. The panel leads with the **collection's
  identity** — its title, description, and `N / M` position — shown on every photo
  of it, and only then the photo's own data. Closing returns to the collection's
  top-level grid (the album/memory/filtered library) with state and scroll intact
  via the browser's history; every in-photo nav (paging, similar) uses replace, so
  close always lands on the grid, never on a prior photo. The per-photo panel
  carries everything known about it: an **AI-written title and
  description**, the caption, tags grouped by dimension with scores and source
  badges (the "AI data"), the full EXIF panel — including GPS **coordinates**,
  which live here as a technical detail and nowhere else — every local path the
  file arrived from with the wasted-space total when there is more than one, and a
  "similar photos" strip. The AI title/description and tags fill in with the caption and
  planner phases (3–4); until then the panel shows EXIF, sources, and embedding
  status. This is where duplicate paths are seen, since there is no separate
  duplicates screen.
- `/organize` — a dropdown of organization principles (date, memories, camera,
  place) over a list of album cards, each with a cover, title, description, and a
  strip of its photos. `date` shows a grain sub-selector (day / month / year,
  default month). Live principles recompute on selection; `memories` reads stored rows and
  offers a "Rebuild memories" control that queues the background build, showing a
  live `done/total (%)` indicator (HTMX polling) while it runs and reloading to the
  finished albums when it completes.
- `/chat` — a normal chat view: a running **conversation history** of questions
  and their grounded answers, a text **input** at the bottom, a **processing
  indicator** while the model works, streamed answer tokens, and inline thumbnail
  citations. History is **persisted** (`chat_sessions`/`chat_messages`, §6) and
  re-rendered server-side on load, so it survives navigation and restarts; a **New
  session** button starts a fresh conversation. Each question is grounded
  independently — retrieval runs per question, and the persisted history is the
  transcript, not multi-turn model memory.

The nav order is **Upload → Library → Chat → Organize**: bring photos in, browse
and search them, ask about them to understand the collection, then — last, once
you know what you have — group and reorganize them. Organize is the final stage
of the process (it feeds stage 2, the on-disk reorg); chat is a
review-and-understand tool, so it sits before it.

## 14. Code layout

The source uses a **flat layout**: each top-level package sits at the repo root,
so imports are `from web.app import ...`, `from ingest.receive import ...`. There
is no wrapping package directory — the repo is `ivms777`, the code is the root.

```
config.py              # pydantic settings, profile selection
db/
  schema.sql
  connection.py        # connect + versioned migrate
storage/
  base.py              # Storage protocol
  local.py
  keys.py              # content-addressed storage keys
inference/
  client.py            # OpenAI-compatible HTTP client (Ollama and vLLM)
  prompts.py           # caption, planner, chat templates
  fakes.py
embedding/
  base.py              # Embedder protocol, EMBED_DIM
  vectors.py           # (de)serialize + L2-normalize
  store.py             # photo_vec read/write + KNN
  siglip.py            # real SigLIP 2 (torch, runtime only)
  fakes.py             # deterministic hash-derived vectors for tests
ingest/
  receive.py           # verify hash, store original, create photo + source rows
  exif.py              # full EXIF capture
  facets.py            # EXIF -> queryable facets
  geocode.py           # offline reverse geocoding: GPS -> city/country

  thumbs.py
  embed.py             # embed stage + backfill
  taxonomy.py          # zero-shot + pixel stats           (plan 04)
  caption.py           #                                    (plan 05)
  worker.py            # job queue driver
  cli.py               # worker entrypoint (python -m ingest.cli)
organize/              # stage 2, server side                (plan 09)
  base.py              # Layout protocol
  date.py / date_tags.py / flat.py
  manifest.py
search/
  semantic.py          # text->vector KNN, similar photos
  facets.py            # EXIF facet filters + sidebar counts
  tags.py              # model-tag filters + sidebar counts   (plan 04)
  keyword.py / fusion.py                                       (plan 04)
  planner.py                                                   (plan 06)
albums/                # Organize tab — grouping into described albums
  base.py              # Album, Organizer protocol
  by_date.py           # day / month / year grains
  by_camera.py / by_place.py
  memories.py          # organizer: reads stored memory groups        (plan 07)
  memories_build.py    # agentic RAG builder + owner-level worker job  (plan 07)
  registry.py
chat/                  #                                     (plan 06)
  retrieve.py          # off-topic gate + question -> top photo ids via fusion
  context.py           # photo ids -> compact grounding block
  history.py           # persisted chat sessions/messages + answer HTML render
web/
  app.py
  deps.py              # AppContext
  upload_api.py        # /api/upload/*
  templates/
  static/
    upload.js          # picker, batching, progress
    hash-worker.js     # Web Worker: SHA-256, one file at a time
vocab.yaml                                                    (plan 04)
ivms777_sync/          # stage 2 — separate package, separate entry point (plan 09)
  cli.py               # plan | apply | undo | verify
  client.py / scan.py / plan.py / apply.py / journal.py
compose.yaml
compose.mac.yaml       # profile overrides
compose.jetson.yaml
compose.cloud.yaml
tests/
docs/
```

Both Ollama and vLLM expose an OpenAI-compatible API, so one HTTP client covers
every profile. Only `base_url` and the model name differ.

`ivms777_sync` imports nothing from `ivms777`. It has no database, no
Pillow, no models — only the standard library plus `httpx` — so it installs on a
user's machine in seconds and runs anywhere Python does. The layouts in
`ivms777/organize/` run server-side to build the manifest; the CLI only
executes the paths it is given.

## 15. Testing

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
- Upload is tested end to end through `TestClient`: probe returns only unknown
  hashes, a body whose bytes do not match the declared hash is rejected, the
  same file sent twice creates one `photos` row and two `photo_sources` rows.
- Layouts are pure functions and test as such — a `PhotoView` in, a path out —
  including collision suffixes, undated photos, and characters illegal on
  Windows.
- `ivms777_sync` tests build a real directory tree in `tmp_path` and run
  `plan` and `apply` against a manifest fixture, then assert the tree matches
  the expected layout exactly. Every such test also runs `undo` and asserts the
  tree is byte-identical to how it started.
- Failure injection covers the paths that can lose data: a crash between
  journal write and rename, a target that already exists, a file modified
  between plan and apply, and a cross-filesystem move whose copy is truncated.

## 16. Phases

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

## 17. Risks

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
| Uploading 5,000 photos is slow and a tab close loses the transfer | Hashes are probed before bytes are sent, so a restart resumes with only what is missing; nothing already received is re-sent |
| Hashing thousands of files freezes the browser tab | Hashing runs in a Web Worker, one file at a time, never on the main thread |
| Storing every original fills the disk | Free space is checked before an upload is accepted and the upload is refused with a clear message rather than failing halfway |
| `ivms777-sync` corrupts or loses photos | Every operation is journaled before it runs and reversed by `undo`; moves are same-filesystem renames where possible and copy-verify-unlink otherwise; nothing is ever deleted, only moved to `_duplicates/` |
| The library changed on disk since upload, so the plan is stale | `plan` matches by content hash, not path; files whose hash is unknown to the manifest are reported and left untouched |
| Stage 2 runs against a half-processed library | The manifest carries a `complete` flag; `apply` refuses an incomplete manifest without `--allow-incomplete` |

## 18. Future work

- Authentication, signup, and per-user quotas for public access (v02).
- Face detection and person clustering.
- Object storage backend behind the existing `Storage` interface.
- Optional XMP sidecar export so other tools see the tags.
- Offline reverse geocoding place names — the "By place" organizer names albums
  by city and a `place_city`/`place_country` sidebar facet filters the library by
  place (plan 08, done). Future: sub-city neighbourhoods and user-editable labels.
- Agentic RAG + reranking for chat retrieval (plan 10). Chat is one-shot fusion
  today and dumps loosely-related photos with a sometimes-confabulated answer; the
  future phase reranks candidates against the question, applies a relevance floor
  (honest "nothing found" instead of 30 neighbours), reuses the query planner, and
  adds a bounded verify-before-answer agent loop — the documented interactive
  exception to §9.1's "one call" rule.
- Postgres and pgvector if concurrent writes become a real constraint.
- Video support.
- A watch mode for `ivms777-sync` that uploads new files as they appear.
- User-defined layouts, expressed as a path template over facets and tags.
