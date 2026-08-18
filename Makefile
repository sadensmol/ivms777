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
LLAMA_PORT   ?= 8080

# Jetson (8 GB Orin Nano, JetPack 7): fully containerised. Since plan 16 ONE
# gemma4-E2B GGUF is served by a containerised sm_87 CUDA llama-server for text +
# vision; there is no Ollama and no in-process VLM (docs/design.md §3.1). Both
# model names default to gemma4-E2B (config.py jetson defaults); override on the
# command line, e.g. `make run-jetson IVMS777_CAPTION_MODEL=gemma4-E2B`.
JETSON_COMPOSE       := -f compose.yaml -f compose.jetson.yaml
JETSON_CAPTION_MODEL ?= gemma4-E2B
JETSON_PLANNER_MODEL ?= gemma4-E2B

.DEFAULT_GOAL := help
.PHONY: up set-maxn-jetson run-jetson stop-jetson clean-jetson test lint llama-mac llama-stop llama-rebuild clean help

up: llama-mac ## Ensure host llama-server, then run models + worker + app NATIVELY → http://localhost:8000
	@mkdir -p "$$HOME/.ivms777"
	@echo "  syncing the models extra (torch/transformers) for the native models service…"
	@uv sync --extra models
	@echo "  data $$HOME/.ivms777 · inference localhost:$(LLAMA_PORT) · models localhost:9000 · app http://localhost:8000 (Ctrl-C stops all)"
	@IVMS777_PROFILE="$${IVMS777_PROFILE:-mac}" \
	 IVMS777_DATA_DIR="$${IVMS777_DATA_DIR:-$$HOME/.ivms777}" \
	 IVMS777_MODELS_BASE_URL="$${IVMS777_MODELS_BASE_URL:-http://localhost:9000}" \
	 bash -c 'IVMS777_INFERENCE_BASE_URL="$${IVMS777_INFERENCE_BASE_URL:-http://localhost:$(LLAMA_PORT)/v1}" \
	   uv run --no-sync uvicorn modelsvc.asgi:app_factory --factory --host 0.0.0.0 --port 9000 & M=$$!; \
	   uv run --no-sync watchfiles "python -m ingest.cli" ingest embedding search inference albums storage db models web config.py vocab.yaml & W=$$!; \
	   trap "kill $$M $$W 2>/dev/null" EXIT INT TERM; \
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
	@mkdir -p "$(LLAMA_MODELS)"
	@[ -s "$(LLAMA_MODELS)/$(LLAMA_GGUF)" ]   || { echo "  downloading $(LLAMA_GGUF) (~2 GB, one time)…"; curl -fL --progress-bar -o "$(LLAMA_MODELS)/$(LLAMA_GGUF)"   "https://huggingface.co/$(LLAMA_REPO)/resolve/main/$(LLAMA_GGUF)"; }
	@[ -s "$(LLAMA_MODELS)/$(LLAMA_MMPROJ)" ] || { echo "  downloading $(LLAMA_MMPROJ)…"; curl -fL --progress-bar -o "$(LLAMA_MODELS)/$(LLAMA_MMPROJ)" "https://huggingface.co/$(LLAMA_REPO)/resolve/main/$(LLAMA_MMPROJ)"; }
	@if curl -sf "http://localhost:$(LLAMA_PORT)/health" >/dev/null 2>&1; then \
	  echo "  llama-server already up on :$(LLAMA_PORT)"; \
	else \
	  echo "  starting llama-server on :$(LLAMA_PORT) (log: /tmp/ivms777-llama.log)…"; \
	  nohup "$(LLAMA_BIN)" -m "$(LLAMA_MODELS)/$(LLAMA_GGUF)" --mmproj "$(LLAMA_MODELS)/$(LLAMA_MMPROJ)" \
	    -ngl 99 --flash-attn on --jinja --chat-template-kwargs '{"enable_thinking":false}' \
	    -c 4096 --host 0.0.0.0 --port $(LLAMA_PORT) >/tmp/ivms777-llama.log 2>&1 & \
	  for i in $$(seq 1 60); do curl -sf "http://localhost:$(LLAMA_PORT)/health" >/dev/null 2>&1 && break; sleep 1; done; \
	  curl -sf "http://localhost:$(LLAMA_PORT)/health" >/dev/null 2>&1 || { echo "  llama-server did not come up — see /tmp/ivms777-llama.log"; exit 1; }; \
	  echo "  llama-server ready."; \
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
