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

.DEFAULT_GOAL := help
.PHONY: up test lint ollama clean help

up: ollama ## Ensure Ollama + models, then run app + worker NATIVELY → http://localhost:8000
	@mkdir -p "$$HOME/.ivms777"
	@echo "  data $$HOME/.ivms777 · inference localhost:11434 · app http://localhost:8000 (Ctrl-C stops both)"
	@IVMS777_PROFILE="$${IVMS777_PROFILE:-mac}" \
	 IVMS777_DATA_DIR="$${IVMS777_DATA_DIR:-$$HOME/.ivms777}" \
	 IVMS777_INFERENCE_BASE_URL="$${IVMS777_INFERENCE_BASE_URL:-http://localhost:11434/v1}" \
	 bash -c 'uv run watchfiles "python -m ingest.cli" ingest embedding search inference albums storage db config.py vocab.yaml & W=$$!; trap "kill $$W 2>/dev/null" EXIT INT TERM; \
	   uv run uvicorn web.app:app_factory --factory --host 0.0.0.0 --port 8000 --reload'

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
