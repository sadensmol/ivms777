# ivms777 — local dev on macOS.
#
# `make up` does everything: it makes sure host Ollama is running with the
# caption/planner models, then builds and starts the containerised stack (app +
# worker + data volume). SigLIP runs on CPU inside the container; Ollama runs on
# the host because Docker Desktop on macOS has no GPU passthrough.
#
# First `make up` pulls the vision model (~6 GB) once — later runs are fast.

COMPOSE       := docker compose -f compose.yaml -f compose.mac.yaml
COMPOSE_DEV   := $(COMPOSE) -f compose.dev.yaml
OLLAMA_MODELS := qwen2.5vl:7b qwen2.5:3b

.DEFAULT_GOAL := help
.PHONY: up down restart logs worker-logs ps dev test lint ollama clean help

up: ollama ## Ensure Ollama + models, then build & start the stack → http://localhost:8000
	$(COMPOSE) up --build -d
	@echo ""
	@echo "  ivms777 is up → http://localhost:8000"
	@echo "  the worker is indexing + captioning in the background; watch: make worker-logs"

down: ## Stop the stack (the host Ollama service is left running)
	$(COMPOSE) down

restart: down up ## Rebuild and restart everything (Ollama + stack)

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

logs: ## Follow app + worker logs
	$(COMPOSE) logs -f app worker

worker-logs: ## Follow the worker only (indexing / tagging / captioning progress)
	$(COMPOSE) logs -f worker

ps: ## Show container status
	$(COMPOSE) ps

dev: ollama ## Start with hot reload (bind-mounts source, restarts on edit)
	$(COMPOSE_DEV) up --build -d
	@echo "  dev (hot reload) → http://localhost:8000"

test: ## Run the test suite natively
	uv run pytest -q

lint: ## Run ruff
	uv run ruff check .

clean: ## DANGER: stop and DELETE the data volume (removes uploaded photos + index)
	$(COMPOSE) down -v

help: ## List targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-13s\033[0m %s\n", $$1, $$2}'
