# ivms777 — local dev on macOS.
#
# `make up` runs everything NATIVELY — no containers. It makes sure host Ollama
# is running with the caption/planner models, then launches the app and the
# worker as plain host processes against $HOME/.ivms777. SigLIP runs on the host
# (CPU by default; set IVMS777_EMBED_DEVICE=mps for Metal); Ollama is already on
# the host because Docker Desktop on macOS has no GPU passthrough.
#
# There is no `make down`: native `up` runs in the foreground and Ctrl-C stops
# both processes. The compose.*.yaml files still describe the deployed stack for
# jetson/cloud — just run `docker compose` directly there.
#
# First run pulls the vision model (~6 GB) once — later runs are fast.

OLLAMA_MODELS := qwen2.5vl:7b qwen2.5:3b

# Jetson (8 GB Orin Nano, JetPack 7): fully containerised. 8 GB is shared
# CPU+GPU, so only a 4B-class vision captioner + a 3B planner fit alongside SigLIP
# (docs/design.md §3.1). These are the config.py jetson defaults; override on the
# command line, e.g. `make run-jetson JETSON_CAPTION_MODEL=gemma4:e4b`.
JETSON_COMPOSE       := -f compose.yaml -f compose.jetson.yaml
# JETSON_CAPTION_MODEL is NOT an Ollama tag on jetson — captioning runs in-process
# (transformers 4-bit VLM, config.py caption_model_id) and is not `ollama pull`ed;
# the var is still passed through as IVMS777_CAPTION_MODEL for config/display
# (docs/design.md §3.1/§4). Only the planner is pulled into Ollama below.
JETSON_CAPTION_MODEL ?= qwen2.5vl:3b
JETSON_PLANNER_MODEL ?= qwen2.5:3b
JETSON_MODELS        := $(JETSON_PLANNER_MODEL)

.DEFAULT_GOAL := help
.PHONY: up run-jetson stop-jetson clean-jetson test lint ollama clean help

up: ollama ## Ensure Ollama + models, then run app + worker NATIVELY → http://localhost:8000
	@mkdir -p "$$HOME/.ivms777"
	@echo "  data $$HOME/.ivms777 · inference localhost:11434 · app http://localhost:8000 (Ctrl-C stops both)"
	@IVMS777_PROFILE="$${IVMS777_PROFILE:-mac}" \
	 IVMS777_DATA_DIR="$${IVMS777_DATA_DIR:-$$HOME/.ivms777}" \
	 IVMS777_INFERENCE_BASE_URL="$${IVMS777_INFERENCE_BASE_URL:-http://localhost:11434/v1}" \
	 bash -c 'uv run watchfiles "python -m ingest.cli" ingest embedding search inference albums storage db models web config.py vocab.yaml & W=$$!; trap "kill $$W 2>/dev/null" EXIT INT TERM; \
	   uv run uvicorn web.app:app_factory --factory --host 0.0.0.0 --port 8000 --reload'

run-jetson: ## Run ON THE JETSON: build + start the containerised stack, pull its (text) model → http://<jetson>:8000
	@echo "  models     : caption=$(JETSON_CAPTION_MODEL) (in-process, Hugging Face) planner=$(JETSON_PLANNER_MODEL) (Ollama)"; \
	 IVMS777_CAPTION_MODEL="$(JETSON_CAPTION_MODEL)" \
	 IVMS777_PLANNER_MODEL="$(JETSON_PLANNER_MODEL)" \
	 docker compose $(JETSON_COMPOSE) up --build -d
	@echo "  waiting for the inference container…"
	@for i in $$(seq 1 30); do \
	  docker compose $(JETSON_COMPOSE) exec -T inference ollama list >/dev/null 2>&1 && break; sleep 2; done
	@# Only the planner is an Ollama tag on jetson — pull it. The in-process caption
	@# VLM (transformers 4-bit) fetches itself from Hugging Face into the mounted HF
	@# cache on first caption call; there is nothing to `ollama pull` for it.
	@for m in $(JETSON_MODELS); do \
	  echo "  pulling $$m…"; \
	  docker compose $(JETSON_COMPOSE) exec -T inference ollama pull "$$m"; \
	done
	@gpu=$$(docker compose $(JETSON_COMPOSE) exec -T app uv run --no-sync python -c "import torch,numpy;print(torch.cuda.is_available(),numpy.__version__)" 2>/dev/null || true); \
	 echo "  preflight   : cuda numpy = $${gpu:-<app not ready>}"; \
	 case "$$gpu" in \
	   True\ *)  echo "  preflight   : GPU OK — SigLIP will run on cuda." ;; \
	   False\ *) echo "  ⚠️  GPU UNAVAILABLE — SigLIP/chat WILL 500. Fix the nvidia container runtime (docs/design.md §3.1), then rerun 'make run-jetson'." ;; \
	   *)        echo "  ⚠️  could not verify GPU — check: docker compose $(JETSON_COMPOSE) logs app" ;; \
	 esac
	@ip=$$(hostname -I 2>/dev/null | awk '{print $$1}'); \
	 echo "  ivms777 (jetson) up."; \
	 echo "    on this Jetson : http://localhost:8000"; \
	 echo "    from your Mac  : http://$${ip:-<jetson-ip>}:8000   (or http://$$(hostname).local:8000)"; \
	 echo "    logs           : docker compose $(JETSON_COMPOSE) logs -f worker"

stop-jetson: ## Run ON THE JETSON: stop the containerised stack (data + models kept)
	docker compose $(JETSON_COMPOSE) down
	@echo "  ivms777 (jetson) stopped. Data and pulled models survive (named volumes)."

clean-jetson: ## DANGER: wipe Jetson library (db + uploads + derived); code + models kept
	docker compose $(JETSON_COMPOSE) down
	@vols=$$(docker volume ls -q -f label=com.docker.compose.volume=ivms777-data); \
	 if [ -n "$$vols" ]; then \
	   docker volume rm $$vols && echo "  removed data volume: $$vols"; \
	 else echo "  no ivms777-data volume found — nothing to wipe"; fi
	@echo "  ivms777 (jetson) library wiped. SigLIP + Ollama model caches kept."
	@echo "  Run 'make run-jetson' to start fresh (a new empty db is created)."

ollama: ## Ensure host Ollama is running and the caption/planner models are pulled
	@command -v ollama >/dev/null || { echo "  Ollama not installed — run: brew install ollama"; exit 1; }
	@ollama list >/dev/null 2>&1 || { \
	  echo "  starting Ollama…"; \
	  brew services start ollama >/dev/null 2>&1 || (nohup ollama serve >/tmp/ivms777-ollama.log 2>&1 &) ; \
	  for i in $$(seq 1 15); do ollama list >/dev/null 2>&1 && break; sleep 1; done; }
	@ollama list >/dev/null 2>&1 || { echo "  Ollama did not start — run 'ollama serve' and retry"; exit 1; }
	@for m in $(OLLAMA_MODELS); do \
	  ollama list | awk 'NR>1{print $$1}' | grep -qx "$$m" \
	    || { echo "  pulling $$m (first run — large)…"; ollama pull "$$m"; }; \
	done
	@echo "  Ollama ready: $(OLLAMA_MODELS)"

test: ## Run the test suite natively
	uv run pytest -q

lint: ## Run ruff
	uv run ruff check .

clean: ## DANGER: DELETE $HOME/.ivms777 (removes uploaded photos + index)
	rm -rf "$$HOME/.ivms777"

help: ## List targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-13s\033[0m %s\n", $$1, $$2}'
