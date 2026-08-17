# ivms777 — local dev on macOS.
#
# `make up` runs everything NATIVELY — no containers. It makes sure a host-native
# `llama-server` (Metal) is built and running with the gemma4-E2B GGUF (text +
# vision), then launches the models service, the worker, and the app as plain host
# processes against $HOME/.ivms777. SigLIP + the caption-text embedder run on the
# host inside the models service (CPU by default; set IVMS777_EMBED_DEVICE=mps for
# Metal on SigLIP); `llama-server` is on the host because Docker Desktop on macOS
# has no GPU passthrough. app/worker are thin clients and reach the native models
# service at IVMS777_MODELS_BASE_URL (design §5.1).
#
# There is no `make down`: native `up` runs in the foreground and Ctrl-C stops the
# three app processes (the host llama-server keeps running — `make llama-stop`).
# The compose.*.yaml files still describe the deployed stack for jetson/cloud.
#
# First run builds llama.cpp and downloads the gemma GGUF (~2 GB) once.

# --- host llama-server (mac) -------------------------------------------------
LLAMA_DIR    ?= $(HOME)/.ivms777/llama.cpp        # checkout + build tree
LLAMA_BIN    := $(LLAMA_DIR)/build/bin/llama-server
LLAMA_MODELS ?= $(HOME)/.ivms777/models           # GGUF cache
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
.PHONY: up run-jetson stop-jetson clean-jetson test lint llama-mac llama-stop clean help

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
	  echo "  building llama.cpp (Metal) — one time…"; \
	  [ -d "$(LLAMA_DIR)/.git" ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp "$(LLAMA_DIR)"; \
	  cmake -S "$(LLAMA_DIR)" -B "$(LLAMA_DIR)/build" -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF; \
	  cmake --build "$(LLAMA_DIR)/build" --target llama-server -j; \
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

run-jetson: ## Run ON THE JETSON: set MAXN, build + start the containerised stack → http://<jetson>:8000
	@echo "  power mode : setting MAXN_SUPER (nvpmodel -m 2 + jetson_clocks — needs sudo)…"
	@sudo nvpmodel -m 2 && sudo jetson_clocks || echo "  ⚠️  could not set MAXN — inference will be ~8× slower (docs/design.md §3.1)"
	@echo "  models     : gemma4-E2B on llama-server (text + vision, GPU) · caption=$(JETSON_CAPTION_MODEL) planner=$(JETSON_PLANNER_MODEL)"; \
	 IVMS777_CAPTION_MODEL="$(JETSON_CAPTION_MODEL)" \
	 IVMS777_PLANNER_MODEL="$(JETSON_PLANNER_MODEL)" \
	 docker compose $(JETSON_COMPOSE) up --build -d
	@echo "  waiting for the inference (llama-server) container… (first run downloads the GGUF ~2 GB; the sm_87 binary is REUSED from the llamacpp volume — no recompile)"
	@for i in $$(seq 1 120); do \
	  curl -sf http://localhost:8080/health >/dev/null 2>&1 && break; sleep 3; done
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
	@echo "  NOTE: jetson_clocks does NOT survive reboot — install a boot unit to re-apply MAXN (docs/design.md §3.1)."

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

clean: ## DANGER: DELETE $HOME/.ivms777 (removes uploaded photos + index)
	rm -rf "$$HOME/.ivms777"

help: ## List targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-13s\033[0m %s\n", $$1, $$2}'
