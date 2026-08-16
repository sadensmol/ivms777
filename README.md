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
`/organize` groups the library by date (day/month/year), **Memories**, camera, or
place.

**Captions** add a sentence, an AI-written title/description, and model-chosen tags
per photo — filling the `/photo` AI panel and feeding search. The vision model they
need is pulled and started by `make up` automatically; the `caption` stage runs
after tagging. Override the model with `IVMS777_CAPTION_MODEL`.

**Search** is planner-backed: a free-text query is turned into date/facet/tag
filters (shown as removable chips) before ranking. **Chat** (`/chat`) answers
questions grounded in your photos, with persisted history and a New-session
button. **Memories** (`/organize?by=memories`) is built on demand — hit *Rebuild
memories* and a background agent groups the library into named, described albums
("Family night in Ontario"); it needs captions (the caption stage) and the planner
model present first, and re-opening the tab is instant. Agentic RAG + reranking for
chat retrieval is the next planned phase — see `docs/plans/10-chat-agentic-rerank.md`.

## Run on a Jetson Orin Nano (8 GB)

Fully containerised — the NVIDIA container runtime hands the Orin iGPU to
containers, so the image builds and the GPU is reached **on the Jetson** (you
cannot build the aarch64 image on a Mac). Requires **JetPack 7** (L4T r39,
CUDA 13.2): the `app`/`worker` build from `Dockerfile.jetson`, which is
`python:3.12-slim` with `torch`/`torchvision` from the CUDA-13.2 index
(`cu132`) so SigLIP runs on the GPU — no jetson-containers, no `autotag`, no
manual pinning. `make run-jetson` builds and starts all three containers. Two
steps: get the code onto the Jetson, then start it there.

**1. Get the code onto the Jetson.** Either clone it:

```bash
ssh jetson
git clone <your-repo> ivms777 && cd ivms777
```

…or, to test uncommitted local changes, `rsync` your working tree from your Mac:

```bash
rsync -av --exclude .venv --exclude .git --exclude __pycache__ \
  ~/work/sadensmol/ivms777/ jetson:~/ivms777/
```

**2. Start it (on the Jetson).**

```bash
make run-jetson   # build + start the stack, pull the caption + planner models → http://<jetson>:8000
```

Then open `http://<jetson>:8000/upload`.

`make run-jetson` builds and starts the containerised `inference` (Ollama),
`app`, and `worker`, waits for Ollama, then pulls the models. The **first** build
compiles the CUDA image and is slow (tens of minutes) — later runs are cached.
8 GB is shared
between CPU and GPU, so the models default to a small vision captioner
(`qwen2.5vl:3b`) and a 3B planner (`qwen2.5:3b`) — the `config.py` jetson
defaults. Override either on the command line:

```bash
make run-jetson JETSON_CAPTION_MODEL=gemma4:e4b
```

The chosen tags are passed to the `app`/`worker` containers **and** pulled into
the in-container Ollama, so what runs matches what was pulled. Watch progress:
`docker compose -f compose.yaml -f compose.jetson.yaml logs -f worker`.

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
