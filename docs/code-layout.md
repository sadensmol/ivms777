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
  wav.py               # WAV encode/measure — numpy + stdlib, torch-free on purpose
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
organize/              # stage 2, server side                (plan 22)
  plan.py              # Album list -> plan document (targets, albums, description.txt)
  names.py             # album title -> safe, deterministic folder segment
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
feed/                  # memory film — offline prepare, thin client (plans 23, 24)
  signals.py           # PURE SQL: per-shot orientation, tags, place, vector
  director.py          # PURE: signals -> Edit (moves, transitions, grade, pacing)
  story.py             # the ONE model call: narration, per-shot lines, music prompt
  language.py          # the feed_language setting (en | ru)
  store.py             # memory_presentations(group_id, language) + audio pruning
  build.py             # the prepare job: signals -> direct -> story -> audio -> row
audio/                 # models-service ONLY — hosts the feed's two models (plan 23)
  kokoro_voice.py      # Kokoro-82M TTS (English) in a TorchWorker child (torch)
  silero_voice.py      # Silero v4_ru TTS (Russian) in a TorchWorker child (torch)
  stable_audio.py      # Stable Audio Open 1.0 in a TorchWorker child (torch)
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
    feed-gl.js         # memory film player: WebGL2 moves, transitions, grade (plan 24)
    feed.css           # the player's own stylesheet — it does not load app.css
vocab.yaml                                                    (plan 04)
ivms777_sync/          # stage 2 — separate package, separate entry point (plan 22)
  cli.py               # apply | undo | verify
  plan.py              # read + validate the downloaded plan document
  scan.py / match.py / apply.py / journal.py / client.py
compose.yaml           # jetson/cloud only — mac is host-native (design §3.1)
compose.jetson.yaml    # profile overrides
compose.cloud.yaml
tests/
docs/
```

Both `llama-server` (mac/jetson) and vLLM (cloud) expose an OpenAI-compatible API,
so one HTTP client covers every profile. Only `base_url` and the model name differ.

`feed/` and `audio/` sit on opposite sides of the one-model-process line (design
§5.1). `feed/` is a thin client — no torch, covered by
`tests/test_one_model_process_gate.py` — while `audio/` is models-service-only and,
like `modelsvc/`, exempt from that gate. `storage/wav.py` is the only module both
import, which is exactly why it is numpy + stdlib and nothing more.

`ivms777_sync` imports nothing from `ivms777`. It has no database, no Pillow, no
models — only the standard library plus `httpx` — so it installs on a user's machine
in seconds and runs anywhere Python does. `organize/` runs server-side, turning the
albums an `albums/` organizer already produced into the plan document; the CLI only
executes the paths it is handed, and `apply` needs no network at all.
