# ivms777

Local photo library organizer. Classifies, searches, and groups a photo folder
using local models — nothing is sent to a third-party API.

See `docs/design.md` for the design and `docs/plans/` for implementation plans.

## Run on a Mac

```bash
make up          # everything: Ollama + models on the host, then the stack → http://localhost:8000
make worker-logs # watch embedding / tagging / captioning progress
make down        # stop the stack
make help        # all targets (restart, dev, clean, …)
```

`make up` does it all in one command: it makes sure **Ollama is running on the
host** with the caption + planner models (Docker Desktop on macOS has no GPU
passthrough, so the LLMs run natively for Metal), then builds and starts the
containerised app, worker, and database. SigLIP runs on CPU inside the container.

The **first** `make up` pulls the vision model (~6 GB) once — later runs are fast.
It needs Ollama installed (`brew install ollama`); everything else it handles.

Open http://localhost:8000/upload, pick one or more folders of photos, and wait.
The browser hashes each file locally, uploads only what the server does not
already have, and skips videos. Nothing on your disk is touched — the originals
are copied into the app's own storage on the `/data` volume.

## Browse, search, and organize

Once photos are indexed, `/library` is a searchable grid. Every photo is embedded
with SigLIP and tagged across ten dimensions (subject, setting, vibe, light,
palette, and more) by a `taxonomy` job stage that runs after embedding; `palette`
and `quality` also get cheap pixel-statistic tags. The left sidebar filters by
those model tags and by exact EXIF facets, both with live counts.

Search blends three signals: SigLIP semantic similarity, FTS5 keyword match over
captions and tag text, and reciprocal-rank fusion of the two — so "dogs in snow"
finds the look while a proper noun finds the word. `/photo` shows every tag with
its score and source; `‹`/`›` and the arrow keys page through the library.
`/organize` groups the library by date (day/month/year), camera, or place.

**Captions** add a sentence, an AI-written title/description, and model-chosen tags
per photo — filling the `/photo` AI panel and feeding search. The vision model they
need is pulled and started by `make up` automatically; the `caption` stage runs
after tagging. Override the model with `IVMS777_CAPTION_MODEL`.

The query planner, chat, and Memories are later phases — see `docs/plans/`.

## Run on a Jetson Orin Nano

Fully containerised — JetPack ships the NVIDIA container runtime.

```bash
docker compose -f compose.yaml -f compose.jetson.yaml up --build -d
docker compose exec inference ollama pull qwen3-vl:4b
```

## Run on a cloud GPU box

```bash
export VLLM_MODEL=google/gemma-4-26b-a4b-it
docker compose -f compose.yaml -f compose.cloud.yaml up --build -d
```

## Develop

Tests run natively for a fast loop; the app itself always runs in containers.

```bash
uv sync
uv run pytest
uv run ruff check .
```

### Hot reload

Add `compose.dev.yaml` last. It bind-mounts the source and restarts the app
(uvicorn `--reload`) and the worker (`watchfiles`) on every edit. Templates and
CSS need no restart at all.

```bash
docker compose -f compose.yaml -f compose.mac.yaml -f compose.dev.yaml up -d
```

Polling is forced on, because inotify events do not cross Docker's bind mount
on macOS.

## Compare caption models

Picks the default caption model for a profile by measuring both on your own
photos rather than trusting published benchmarks.

```bash
uv run python -m scripts.bakeoff \
  --models gemma4:26b-a4b gemma4:12b \
  --library /path/to/photos --count 50
```
