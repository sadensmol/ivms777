# Code layout

The source tree. Design §14 carries the layout decisions (flat packages, the
standalone sync package, one HTTP client for every profile); this file is the map.

The source uses a **flat layout**: each top-level package sits at the repo root, so
imports are `from web.app import ...`, `from ingest.receive import ...`. There is no
wrapping package directory — the repo is `ivms777`, the code is the root.

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
  client.py            # OpenAI-compatible HTTP client (llama-server and vLLM)
  prompts.py           # caption, planner, chat templates
  fakes.py
embedding/
  base.py              # Embedder protocol, EMBED_DIM
  vectors.py           # (de)serialize + L2-normalize
  store.py             # photo_vec read/write + KNN
  siglip.py            # real SigLIP 2 (torch, runtime only)
  caption_text.py      # caption-meaning text embedding (task prefixes) (§9)
  text_embedder.py     # in-process nomic text embedder (torch, models-service only)
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
  retrieve.py          # off-topic gate + fusion fallback (via core candidates)
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

Both `llama-server` (mac/jetson) and vLLM (cloud) expose an OpenAI-compatible API,
so one HTTP client covers every profile. Only `base_url` and the model name differ.

`ivms777_sync` imports nothing from `ivms777`. It has no database, no Pillow, no
models — only the standard library plus `httpx` — so it installs on a user's machine
in seconds and runs anywhere Python does. The layouts in `ivms777/organize/` run
server-side to build the manifest; the CLI only executes the paths it is given.
