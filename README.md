# ivms777

Local photo library organizer. Classifies, searches, and groups a photo folder
using local models — nothing is sent to a third-party API.

See `docs/design.md` for the design and `docs/plans/` for implementation plans.

## Run on a Mac

Docker Desktop on macOS cannot reach the Apple GPU, so Ollama runs natively on
the host. Everything else — app, worker, database — runs in containers.

```bash
brew install ollama
ollama serve &
ollama pull gemma4:26b-a4b
ollama pull gemma4:e4b

docker compose -f compose.yaml -f compose.mac.yaml up --build -d
```

Open http://localhost:8000/upload, pick one or more folders of photos, and wait.
The browser hashes each file locally, uploads only what the server does not
already have, and skips videos. Nothing on your disk is touched — the originals
are copied into the app's own storage on the `/data` volume.

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
