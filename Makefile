# ivms777 — local dev on macOS.
#
# `make up` runs everything NATIVELY — no containers. It makes sure a host-native
# `llama-server` (Metal) is built and running with the gemma4-E2B GGUF (text +
# vision), then launches the models service, the worker, and the app as plain host
# processes against $HOME/.ivms777. SigLIP + the caption-text embedder run on the
# host inside the models service on the Apple GPU (`embed_device=mps`, the mac
# profile default) — NOTHING on mac runs on the CPU (design §3.1). `llama-server`
# is on the host for the same reason: Docker Desktop on macOS has no Metal
# passthrough, so a containerised model could only fall back to the CPU. app/worker
# are thin clients and reach the native models service at IVMS777_MODELS_BASE_URL
# (design §5.1).
#
# There is no `make down`: native `up` runs in the foreground and Ctrl-C stops the
# three app processes (the host llama-server keeps running — `make llama-stop`).
# The compose.*.yaml files describe the deployed stack for jetson/cloud ONLY;
# there is no containerised mac path.
#
# First run builds llama.cpp and downloads the gemma GGUF (~2 GB) once, into
# ~/.llama (NOT the library dir), so `make clean` keeps them and `make up` never
# rebuilds. `make llama-rebuild` is the only thing that wipes that cache.

# --- host llama-server (mac) -------------------------------------------------
# The llama.cpp checkout+build AND the gemma GGUF live OUTSIDE the library dir,
# under $(LLAMA_HOME) (default ~/.llama). They are built/downloaded ONCE and
# `make clean` NEVER touches them, so `make up` never rebuilds. Only the explicit
# `make llama-rebuild` wipes and re-creates them.
# NOTE: no inline `# comments` on these assignments — GNU Make keeps the spaces
# before the `#` as part of the value, which silently created dirs named
# "llama.cpp   " / "models   ". Keep values bare.
LLAMA_HOME   ?= $(HOME)/.llama
LLAMA_SRC    := $(LLAMA_HOME)/llama.cpp
LLAMA_BIN    := $(LLAMA_SRC)/build/bin/llama-server
LLAMA_MODELS ?= $(LLAMA_HOME)/models
LLAMA_GGUF   ?= gemma-4-E2B-it-Q4_K_M.gguf
LLAMA_MMPROJ ?= mmproj-F16.gguf
LLAMA_REPO   ?= unsloth/gemma-4-E2B-it-GGUF
# NOT 8080, and the models service is NOT on 9000. Both are among the most contended
# ports on a developer's machine — on this one an OrbStack container publishes 8080 and
# a SigNoz ClickHouse publishes 9000, so `make up` failed to bind llama-server and, more
# quietly, the models service died on startup and the whole stack looked broken. Two
# ports nobody else wants cost nothing and save an afternoon.
LLAMA_PORT   ?= 8088
# KV context (`-c`) for the host server, and the value `app`/`worker` are told the
# server has (IVMS777_LLM_CTX → `settings.llm_ctx`, which the story pass budgets its
# prompt against). 4096 was too small for a real film: a 40-shot memory's story prompt
# is ~6 300 tokens and llama-server rejected the whole call ("request (6306 tokens)
# exceeds the available context size (4096 tokens)"), which surfaced as a memory that
# would not narrate. MEASURED on this mac (32 GB): after one generation the server is
# 4.11 GiB resident at -c 4096, 4.18 at 16384 and 4.28 at 32768 — gemma's sliding
# window keeps the KV cache tiny, so the whole increase costs ~0.17 GiB.
LLAMA_CTX    ?= 32768

# Jetson (8 GB Orin Nano, JetPack 7): fully containerised. Since plan 16 ONE
# gemma4-E2B GGUF is served by a containerised sm_87 CUDA llama-server for text +
# vision; there is no Ollama and no in-process VLM (docs/design.md §3.1). Both
# model names default to gemma4-E2B (config.py jetson defaults); override on the
# command line, e.g. `make run-jetson IVMS777_CAPTION_MODEL=gemma4-E2B`.
JETSON_COMPOSE       := -f compose.yaml -f compose.jetson.yaml
JETSON_CAPTION_MODEL ?= gemma4-E2B
JETSON_PLANNER_MODEL ?= gemma4-E2B

# The models service's port, for the same reason.
MODELS_PORT  ?= 9100

.DEFAULT_GOAL := help
.PHONY: up models set-maxn-jetson run-jetson stop-jetson clean-jetson test lint llama-mac llama-stop llama-rebuild clean help

# The models service reloads too (--reload), scoped to the packages it actually
# imports. Without it, every change to a backend, a slot rule or an audio child needed
# a manual restart, and a stale service looks exactly like a broken feature — it is
# what made Russian narration "not work" long after the code was right. The watch list
# is scoped rather than the whole repo so an edit to web/ or feed/ does not drop the
# resident models mid-generation.
models: ## Pre-fetch the SELECTED slots' weights into the shared cache (idempotent)
	@echo "  syncing the models extra (torch/transformers) for the native models service…"
	@uv sync --extra models
	@uv run --no-sync python -m scripts.fetch_models

up: llama-mac models ## Ensure host llama-server + weights, then run models + worker + app NATIVELY → http://localhost:8000
	@mkdir -p "$$HOME/.ivms777"
	@echo "  data $$HOME/.ivms777 · inference localhost:$(LLAMA_PORT) · models localhost:$(MODELS_PORT) · app http://localhost:8000 (Ctrl-C stops all)"
	@IVMS777_PROFILE="$${IVMS777_PROFILE:-mac}" \
	 IVMS777_DATA_DIR="$${IVMS777_DATA_DIR:-$$HOME/.ivms777}" \
	 IVMS777_MODELS_BASE_URL="$${IVMS777_MODELS_BASE_URL:-http://localhost:$(MODELS_PORT)}" \
	 IVMS777_LLM_CTX="$${IVMS777_LLM_CTX:-$(LLAMA_CTX)}" \
	 bash -c 'IVMS777_INFERENCE_BASE_URL="$${IVMS777_INFERENCE_BASE_URL:-http://localhost:$(LLAMA_PORT)/v1}" \
	   uv run --no-sync uvicorn modelsvc.asgi:app_factory --factory --host 0.0.0.0 --port $(MODELS_PORT) \
	     --reload --reload-dir modelsvc --reload-dir audio --reload-dir models \
	     --reload-dir embedding --reload-dir captioning --reload-dir inference & M=$$!; \
	   uv run --no-sync watchfiles "python -m ingest.cli" ingest embedding search inference albums storage db models web config.py vocab.yaml & W=$$!; \
	   ( for i in $$(seq 1 120); do \
	       curl -sf -o /dev/null "http://localhost:8000/" && { \
	         printf "\n  ┌────────────────────────────────────────────┐\n"; \
	         printf   "  │  ivms777 is up — open the web page at:     │\n"; \
	         printf   "  │  \033[1;36mhttp://localhost:8000\033[0m                     │\n"; \
	         printf   "  └────────────────────────────────────────────┘\n\n"; \
	         break; }; \
	       sleep 1; \
	     done ) & R=$$!; \
	   trap "kill $$M $$W $$R 2>/dev/null" EXIT INT TERM; \
	   uv run --no-sync uvicorn web.app:app_factory --factory --host 0.0.0.0 --port 8000 --reload'

llama-mac: ## Build (if needed) + start the host-native llama-server (Metal) on :$(LLAMA_PORT)
	@command -v cmake >/dev/null || { echo "  cmake not found — run: brew install cmake"; exit 1; }
	@if [ ! -x "$(LLAMA_BIN)" ]; then \
	  echo "  building llama.cpp (Metal) — one time; cached in $(LLAMA_SRC) (survives 'make clean')…"; \
	  [ -d "$(LLAMA_SRC)/.git" ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp "$(LLAMA_SRC)"; \
	  cmake -S "$(LLAMA_SRC)" -B "$(LLAMA_SRC)/build" -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF; \
	  cmake --build "$(LLAMA_SRC)/build" --target llama-server -j; \
	else \
	  echo "  reusing cached llama-server ($(LLAMA_BIN)) — no build."; \
	fi
	@for PORT in $(LLAMA_PORT) $(MODELS_PORT); do \
	  HOLDER=$$(lsof -nP -iTCP:$$PORT -sTCP:LISTEN 2>/dev/null | awk 'NR==2 {print $$1" (pid "$$2")"}'); \
	  if [ -n "$$HOLDER" ] && ! echo "$$HOLDER" | grep -q llama-server; then \
	    echo "  port $$PORT is already held by $$HOLDER."; \
	    echo "  Free it, or run: make up LLAMA_PORT=<free> MODELS_PORT=<free>"; \
	    exit 1; \
	  fi; \
	done
	@mkdir -p "$(LLAMA_MODELS)"
	@[ -s "$(LLAMA_MODELS)/$(LLAMA_GGUF)" ]   || { echo "  downloading $(LLAMA_GGUF) (~2 GB, one time)…"; curl -fL --progress-bar -o "$(LLAMA_MODELS)/$(LLAMA_GGUF)"   "https://huggingface.co/$(LLAMA_REPO)/resolve/main/$(LLAMA_GGUF)"; }
	@[ -s "$(LLAMA_MODELS)/$(LLAMA_MMPROJ)" ] || { echo "  downloading $(LLAMA_MMPROJ)…"; curl -fL --progress-bar -o "$(LLAMA_MODELS)/$(LLAMA_MMPROJ)" "https://huggingface.co/$(LLAMA_REPO)/resolve/main/$(LLAMA_MMPROJ)"; }
	@RUNNING=$$(python3 -c 'import json,urllib.request; print(json.load(urllib.request.urlopen("http://localhost:$(LLAMA_PORT)/props", timeout=2))["default_generation_settings"]["n_ctx"])' 2>/dev/null); \
	 if [ "$$RUNNING" = "$(LLAMA_CTX)" ]; then \
	  echo "  llama-server already up on :$(LLAMA_PORT) (context $$RUNNING)"; \
	else \
	  if [ -n "$$RUNNING" ]; then \
	    echo "  llama-server on :$(LLAMA_PORT) has context $$RUNNING, want $(LLAMA_CTX) — restarting it…"; \
	  fi; \
	  pkill -f "llama-server .*--port $(LLAMA_PORT)" >/dev/null 2>&1 || true; \
	  for i in $$(seq 1 40); do lsof -nP -iTCP:$(LLAMA_PORT) -sTCP:LISTEN >/dev/null 2>&1 || break; sleep 0.5; done; \
	  echo "  starting llama-server on :$(LLAMA_PORT) (context $(LLAMA_CTX), log: /tmp/ivms777-llama.log)…"; \
	  nohup "$(LLAMA_BIN)" -m "$(LLAMA_MODELS)/$(LLAMA_GGUF)" --mmproj "$(LLAMA_MODELS)/$(LLAMA_MMPROJ)" \
	    -ngl 99 --flash-attn on --jinja --chat-template-kwargs '{"enable_thinking":false}' \
	    -c $(LLAMA_CTX) --host 0.0.0.0 --port $(LLAMA_PORT) >/tmp/ivms777-llama.log 2>&1 & \
	  for i in $$(seq 1 60); do curl -sf "http://localhost:$(LLAMA_PORT)/health" >/dev/null 2>&1 && break; sleep 1; done; \
	  STARTED=$$(python3 -c 'import json,urllib.request; print(json.load(urllib.request.urlopen("http://localhost:$(LLAMA_PORT)/props", timeout=2))["default_generation_settings"]["n_ctx"])' 2>/dev/null); \
	  [ "$$STARTED" = "$(LLAMA_CTX)" ] || { echo "  llama-server is not serving context $(LLAMA_CTX) (got '$$STARTED') — see /tmp/ivms777-llama.log"; exit 1; }; \
	  echo "  llama-server ready (context $$STARTED)."; \
	fi

llama-stop: ## Stop the host-native llama-server started by `make llama-mac`
	@pkill -f "llama-server .*--port $(LLAMA_PORT)" && echo "  llama-server stopped." || echo "  no llama-server running on :$(LLAMA_PORT)."

llama-rebuild: ## Force a from-scratch llama.cpp build + GGUF re-download (wipes $(LLAMA_HOME) — the ONLY thing that deletes the cache)
	@echo "  removing $(LLAMA_HOME) (llama.cpp build + GGUF cache)…"; rm -rf "$(LLAMA_HOME)"
	@$(MAKE) llama-mac

# nvpmodel persists the mode ITSELF — it records `pmode:0002` in
# /var/lib/nvpmodel/status and re-applies it at boot (the `PM_CONFIG DEFAULT` in
# /etc/nvpmodel.conf is only the fallback for when no status file exists), so the
# power mode already survives a reboot. `jetson_clocks` does NOT: it writes the
# clock rails directly and they reset on boot. The unit below is what closes that
# gap; it runs AFTER nvpmodel.service so the clocks are pinned on top of the
# restored power mode, never underneath it.
set-maxn-jetson: ## Run ON THE JETSON (needs sudo): set MAXN_SUPER power mode + pin max clocks, both surviving reboot (docs/design.md §3.1)
	@echo "  setting MAXN_SUPER power mode (nvpmodel -m 2)…"
	sudo nvpmodel -m 2
	@echo "  pinning max clocks (jetson_clocks)…"
	sudo jetson_clocks
	@echo "  installing jetson_clocks boot unit (jetson_clocks alone does NOT survive reboot)…"
	@printf '%s\n' \
	  '[Unit]' \
	  'Description=Pin max Jetson clocks at boot (ivms777)' \
	  'After=nvpmodel.service' \
	  'Wants=nvpmodel.service' \
	  '' \
	  '[Service]' \
	  'Type=oneshot' \
	  'ExecStart=/usr/bin/jetson_clocks' \
	  'RemainAfterExit=yes' \
	  '' \
	  '[Install]' \
	  'WantedBy=multi-user.target' \
	  | sudo tee /etc/systemd/system/ivms777-jetson-clocks.service > /dev/null
	sudo systemctl daemon-reload
	sudo systemctl enable ivms777-jetson-clocks.service
	@echo "  done. Power mode AND max clocks now survive reboot."
	@echo "  NOTE: MAXN_SUPER is the hottest setting. 'sudo nvpmodel -m 0' (15W) or '-m 1' (25W)"
	@echo "        run cooler; sustained captioning holds the GPU at ~99%. Undo the clock pin with:"
	@echo "        sudo systemctl disable --now ivms777-jetson-clocks.service"

run-jetson: ## Run ON THE JETSON: build + start the containerised stack → http://<jetson>:8000 (run 'make set-maxn-jetson' once for max speed)
	@echo "  models     : gemma4-E2B on an sm_87 cu132 llama-server the models container SPAWNS (text + vision, GPU); SigLIP + nomic in-process · caption=$(JETSON_CAPTION_MODEL) planner=$(JETSON_PLANNER_MODEL)"; \
	 IVMS777_CAPTION_MODEL="$(JETSON_CAPTION_MODEL)" \
	 IVMS777_PLANNER_MODEL="$(JETSON_PLANNER_MODEL)" \
	 docker compose $(JETSON_COMPOSE) up --build -d
	@echo "  waiting for the models service (first build compiles cu132 llama.cpp — slow once, then cached)…"
	@for i in $$(seq 1 120); do \
	  docker compose $(JETSON_COMPOSE) exec -T models curl -sf http://localhost:9000/models >/dev/null 2>&1 && break; sleep 3; done
	@echo "  note       : gemma loads ON DEMAND (first chat/caption, or POST /models/gemma/ensure) and idle-unloads after IVMS777_LLM_IDLE_TTL_S (plan 18)."
	@gpu=$$(docker compose $(JETSON_COMPOSE) exec -T models uv run --no-sync python -c "import torch,numpy;print(torch.cuda.is_available(),numpy.__version__)" 2>/dev/null || true); \
	 echo "  preflight   : cuda numpy = $${gpu:-<models not ready>}"; \
	 case "$$gpu" in \
	   True\ *)  echo "  preflight   : GPU OK — SigLIP + caption-text embedder run on cuda." ;; \
	   False\ *) echo "  ⚠️  GPU UNAVAILABLE — SigLIP/chat WILL 500. Fix the nvidia container runtime (docs/design.md §3.1), then rerun 'make run-jetson'." ;; \
	   *)        echo "  ⚠️  could not verify GPU — check: docker compose $(JETSON_COMPOSE) logs models" ;; \
	 esac
	@ip=$$(hostname -I 2>/dev/null | awk '{print $$1}'); \
	 echo "  ivms777 (jetson) up."; \
	 echo "    on this Jetson : http://localhost:8000"; \
	 echo "    from your Mac  : http://$${ip:-<jetson-ip>}:8000   (or http://$$(hostname).local:8000)"; \
	 echo "    logs           : docker compose $(JETSON_COMPOSE) logs -f worker"

stop-jetson: ## Run ON THE JETSON: stop the containerised stack (data + models kept)
	docker compose $(JETSON_COMPOSE) down
	@echo "  ivms777 (jetson) stopped. Data and the GGUF cache survive (named volumes)."

clean-jetson: ## DANGER: wipe Jetson library (db + uploads + derived); code + models kept
	docker compose $(JETSON_COMPOSE) down
	@vols=$$(docker volume ls -q -f label=com.docker.compose.volume=ivms777-data); \
	 if [ -n "$$vols" ]; then \
	   docker volume rm $$vols && echo "  removed data volume: $$vols"; \
	 else echo "  no ivms777-data volume found — nothing to wipe"; fi
	@echo "  ivms777 (jetson) library wiped. SigLIP + GGUF caches kept."
	@echo "  Run 'make run-jetson' to start fresh (a new empty db is created)."

test: ## Run the test suite natively
	uv run pytest -q

lint: ## Run ruff
	uv run ruff check .

clean: ## DANGER: wipe the LIBRARY only (db + uploads + thumbs). Keeps the llama.cpp build + GGUF cache in $(LLAMA_HOME).
	@rm -rf "$$HOME/.ivms777/ivms777.db" "$$HOME/.ivms777/ivms777.db-wal" "$$HOME/.ivms777/ivms777.db-shm" \
	        "$$HOME/.ivms777/originals" "$$HOME/.ivms777/thumbs"
	@echo "  library wiped (db + originals + thumbs). Kept: llama.cpp build + GGUF cache in $(LLAMA_HOME) (use 'make llama-rebuild' to wipe those)."

help: ## List targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-13s\033[0m %s\n", $$1, $$2}'
