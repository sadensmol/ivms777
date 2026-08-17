# ivms777

Local photo library organizer. Classifies, searches, and groups a photo folder
using local models — nothing is sent to a third-party API.

See `docs/design.md` for the design and `docs/plans/` for implementation plans.

## Run on a Mac

```bash
brew install cmake   # one time — needed to build llama.cpp
make up              # host llama-server (Metal), then models + worker + app NATIVELY → http://localhost:8000
make help            # all targets
```

`make up` does it all in one command: it makes sure a **host-native
`llama-server` (Metal) is running** with one gemma4-E2B GGUF that serves **both
text (planner/chat) and vision (caption)** — Docker Desktop on macOS has no GPU
passthrough, so the LLM runs natively for Metal — then runs the `models` service,
`worker`, and `app` as plain **host processes — no containers on mac**.
`app`/`worker` are thin HTTP clients with no torch; SigLIP and the caption-text
embedder live only in the `models` service, and captioning is an OpenAI call to
`llama-server` (design §5.1). Ctrl-C stops the three app processes; the host
`llama-server` keeps running (`make llama-stop` to stop it). Data lives under
`$HOME/.ivms777`.

The **first** `make up` builds `llama.cpp` with Metal and downloads the gemma
GGUF (~2 GB) once, and installs the `models` extra (torch/transformers) — later
runs are fast. There is **no Ollama** any more.

### Build llama.cpp for Mac (Metal)

`make up` (via `make llama-mac`) builds and starts `llama-server` for you into
`$HOME/.llama/llama.cpp` — a persistent cache **outside** the library dir, so it
is built once and `make clean` never removes it (`make llama-rebuild` forces a
from-scratch rebuild + GGUF re-download). To do it by hand instead:

```bash
brew install cmake
git clone https://github.com/ggml-org/llama.cpp
cmake -S llama.cpp -B llama.cpp/build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build llama.cpp/build --target llama-server -j
# download gemma-4-E2B-it-Q4_K_M.gguf + mmproj-F16.gguf (unsloth/gemma-4-E2B-it-GGUF)
llama.cpp/build/bin/llama-server -m gemma-4-E2B-it-Q4_K_M.gguf --mmproj mmproj-F16.gguf \
  -ngl 99 --flash-attn on --jinja --chat-template-kwargs '{"enable_thinking":false}' \
  -c 4096 --host 0.0.0.0 --port 8080
```

`make up` expects `llama-server` on `:8080`. `--chat-template-kwargs
'{"enable_thinking":false}'` and `--jinja` are **required** for gemma4 to caption
instead of dumping chain-of-thought (design §4).

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
per photo — filling the `/photo` AI panel and feeding search. The same gemma4-E2B
`llama-server` that answers chat also captions (vision on the GPU); the `caption`
stage runs after tagging. Override the model with `IVMS777_CAPTION_MODEL`.

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
containers, so the images build and the GPU is reached **on the Jetson** (you
cannot build the aarch64 images on a Mac). Requires **JetPack 7** (L4T r39,
CUDA 13.2). Inference is a containerised `llama-server` compiled for the Orin
**sm_87** GPU — one gemma4-E2B GGUF serving **text + vision** (design §3.1, plan
16); there is **no Ollama** and **no in-process caption VLM**. The `llama-server`
binary is **built once** and reused: the `inference` service runs the prebuilt
binary from the `llamacpp` volume (no recompile on later `make run-jetson`), and
`Dockerfile.llamacpp.jetson` is the reproducible from-scratch builder for a fresh
board. The `models` service builds from
`Dockerfile.models.jetson` (`python:3.12-slim` + `torch`/`torchvision` from the
CUDA-13.2 `cu132` index) so SigLIP and the caption-text embedder run on the GPU —
no jetson-containers, no `autotag`. `app`/`worker` build from the plain
`Dockerfile` (no GPU, no torch — thin HTTP clients of `models`, design §5.1).
`make run-jetson` sets MAXN_SUPER, then builds and starts all four containers
(`inference`, `models`, `worker`, `app`). Two steps: get the code onto the Jetson,
then start it there.

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
make run-jetson   # set MAXN, build + start the stack → http://<jetson>:8000
```

Then open `http://<jetson>:8000/upload`.

`make run-jetson` sets **MAXN_SUPER** (`nvpmodel -m 2 && jetson_clocks` — needs
sudo; ~8× faster than the 25 W default), then builds and starts the containerised
`inference` (llama-server), `models`, `worker`, and `app`. The **first** build
compiles the CUDA `llama-server` image and is slow (tens of minutes), and the
`inference` container downloads the gemma GGUF (~2 GB) into a named volume on
first start — later runs are cached. 8 GB is shared between CPU and GPU, so both
the caption and planner roles default to the single **gemma4-E2B** GGUF (the
`config.py` jetson defaults). Override on the command line:

```bash
make run-jetson JETSON_CAPTION_MODEL=gemma4-E2B
```

> **MAXN does not survive reboot.** `nvpmodel -m 2` persists, but `jetson_clocks`
> does not — install a small boot service to re-apply it, or rerun `make
> run-jetson` after a reboot (design §3.1).

Watch progress:
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
