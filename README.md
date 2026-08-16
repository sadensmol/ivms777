# ivms777

Local photo library organizer. Classifies, searches, and groups a photo folder
using local models — nothing is sent to a third-party API.

See `docs/design.md` for the design and `docs/plans/` for implementation plans.

## Run on a Mac

```bash
make up   # Ollama on the host, then models + worker + app NATIVELY → http://localhost:8000
make help # all targets
```

`make up` does it all in one command: it makes sure **Ollama is running on the
host** with the caption + planner models (Docker Desktop on macOS has no GPU
passthrough, so the LLMs run natively for Metal), then runs the `models`
service, `worker`, and `app` as plain **host processes — no containers on
mac**. `app`/`worker` are thin HTTP clients with no torch; SigLIP and the
caption VLM live only in the `models` service (design §5.1). Ctrl-C stops all
three; data lives under `$HOME/.ivms777`.

The **first** `make up` installs the `models` extra (torch/transformers) and
pulls the vision model (~6 GB) once — later runs are fast. It needs Ollama
installed (`brew install ollama`); everything else it handles.

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
CUDA 13.2). Only the `models` service needs GPU/torch: it builds from
`Dockerfile.models.jetson`, which is `python:3.12-slim` with
`torch`/`torchvision` from the CUDA-13.2 index (`cu132`) so SigLIP and the
in-process caption VLM run on the GPU — no jetson-containers, no `autotag`, no
manual pinning. `app`/`worker` build from the plain `Dockerfile` (no GPU, no
torch — thin HTTP clients of `models`, design §5.1). `make run-jetson` builds
and starts all four containers (`inference`, `models`, `worker`, `app`). Two
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
make run-jetson   # build + start the stack, pull the planner model → http://<jetson>:8000
```

Then open `http://<jetson>:8000/upload`.

`make run-jetson` builds and starts the containerised `inference` (Ollama),
`models`, `worker`, and `app`, waits for Ollama, then pulls the planner model
(the caption VLM is not an Ollama tag on jetson — it loads in-process inside
`models`, see below). The **first** build compiles the CUDA image and is slow
(tens of minutes) — later runs are cached. 8 GB is shared
between CPU and GPU, so the models default to a small vision captioner
(`qwen2.5vl:3b`) and a 3B planner (`qwen2.5:3b`) — the `config.py` jetson
defaults. Override either on the command line:

```bash
make run-jetson JETSON_CAPTION_MODEL=gemma4:e4b
```

The planner tag is passed to the `app`/`worker`/`models` containers **and**
pulled into the in-container Ollama, so what runs matches what was pulled. The
caption tag goes to `models` only — on jetson it names a Hugging Face
in-process model, not an Ollama pull. Watch progress:
`docker compose -f compose.yaml -f compose.jetson.yaml logs -f worker`.

## Run on a cloud GPU box

```bash
export VLLM_MODEL=google/gemma-4-26b-a4b-it
docker compose -f compose.yaml -f compose.cloud.yaml up --build -d
```

## Develop

Tests run natively for a fast loop. On mac `make up` also runs natively (no
containers, see above); jetson and cloud always run containerised.

```bash
uv sync                # app/worker/CLI stay torch-free
uv run pytest
uv run ruff check .
```

`uv sync --extra models` additionally installs torch/transformers, needed only
to run the `models` service itself or its `slow`-marked tests
(`uv run pytest -m slow`).

### Hot reload (containerised — jetson/cloud, or as an alternative to native `make up` on mac)

Add `compose.dev.yaml` last. It bind-mounts the source and restarts `app` and
`models` (uvicorn `--reload`) and `worker` (`watchfiles`) on every edit.
Templates and CSS need no restart at all.

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
