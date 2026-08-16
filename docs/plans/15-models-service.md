# The `models` Service — One Inference Gateway (Implementation Plan)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Make **exactly one process** load and run every model/LLM/AI-library. Extract all inference into a `models` service; turn `app`/`worker`/CLI into thin HTTP clients that never import torch/transformers and never call Ollama directly. This fixes the 8 GB Jetson OOM (duplicated SigLIP + two torch contexts) and enforces CLAUDE.md § "One model process".

**Architecture:** `docs/design.md` §5 + **§5.1** (canonical). One `models` FastAPI service is the only torch/transformers importer and the only Ollama client. It loads SigLIP (CPU mac / CUDA jetson) + the caption VLM (jetson) once, coordinates residency **in-process** (SigLIP xor VLM; interactive preempts caption), proxies text to Ollama, and exposes `/embed/image`, `/embed/text`, `/tag`, `/caption`, `/plan`, `/chat`, `/embed/caption-text`, `/resources`. `app`/`worker` call it over HTTP.

**Single implementation, both platforms:** identical code on mac and jetson; only the service's backends differ by profile (SigLIP device; caption = host-Ollama vs in-process VLM; text always Ollama).

**Supersedes:** the cross-process `model_lease` + `ModelCoordinator` (plans 13/14) — deleted; residency is now in-process in the service. The caption adapters (plan 14) move *into* the service. SigLIP (`embedding/siglip.py`) moves *behind* the service.

**Tech Stack:** Python 3.12, FastAPI, `uv`, torch/transformers/SigLIP/bitsandbytes (ONLY in the models-service image), Ollama, SQLite, httpx.

## Global Constraints

- **HARD RULE (CLAUDE.md § "One model process"):** only the `models` service imports torch/transformers/bitsandbytes or loads a model, and it is the only client of Ollama. `app`/`worker`/CLI/tests-of-those must not import those libs. A grep gate enforces it (Task 9).
- **One implementation, both platforms** — mac (Ollama host, SigLIP CPU) and jetson (Ollama container + in-process VLM, SigLIP CUDA) run the same code; differences are config only. Mac must keep working end-to-end.
- **design.md is the source of truth** — §5/§5.1 already describe the target; §4 updated; §8.1 carries a superseded pointer and is fully rewritten in Task 8.
- No git commits (user commits). Tests in `tests/`, `uv run pytest`. New modules need tests.
- Tests must not load real models (board-only) — the service's model backends are faked/injected in unit tests; a `use_fake_*` path returns deterministic vectors/captions.
- Packaging: the `models` service package ships in its own image; `app`/`worker` images DROP torch/transformers.

---

### Task 1: `models` service skeleton + a thin client, both behind fakes

Stand up the FastAPI app and the client `app`/`worker` will use, with a fake backend so everything downstream can be built/tested without real models.

**Files:** create `modelsvc/__init__.py`, `modelsvc/app.py` (FastAPI factory + routes), `modelsvc/backends/base.py` (a `ModelBackend` protocol: `embed_image/embed_text/tag/caption/plan/chat/embed_caption_text/resources`), `modelsvc/backends/fake.py`; create `inference/models_client.py` (`ModelsClient` — httpx calls to the service, matching the backend protocol); tests `tests/test_modelsvc_api.py`, `tests/test_models_client.py`.

**Interfaces:**
- `ModelsClient(base_url)` methods mirror the endpoints and return plain Python (lists of floats, dicts, strings). Used by `app`/`worker`.
- `create_models_app(settings)` → FastAPI app; routes call the configured `ModelBackend`.
- Config: `Settings.models_base_url` (per profile, e.g. `http://models:9000`), `Settings.build_models_client()`.

- [ ] Step 1: failing test — `create_models_app(fake settings)` + TestClient: `POST /embed/text {"texts":["cat"]}` returns a vector; `POST /caption` returns `{caption,title,description,tags}`; `GET /resources` returns memory+resident. `ModelsClient` against the TestClient transport round-trips each.
- [ ] Steps 2-5: implement the FastAPI routes over a `ModelBackend`, the `FakeBackend` (deterministic), the `ModelsClient`, and `config.build_models_client()`. Run tests; full suite green.

---

### Task 2: SigLIP backend + `/embed/image` `/embed/text` `/tag`; app/worker call the service

Move SigLIP behind the service; replace in-process `build_embedder()` in `app`/`worker`.

**Files:** create `modelsvc/backends/siglip_backend.py` (wraps `embedding/siglip.py` — the ONLY place it's imported now); add `modelsvc/backends/__init__.py::build_backend(settings) -> ModelBackend` (config→backend factory: `use_fake` → `FakeBackend`, else the real SigLIP backend — the spec'd entrypoint from profile config to a running app, resolving the Task-1 DI-by-backend ruling; the real service main calls `create_models_app(build_backend(settings))`); modify `modelsvc/app.py` if needed; modify the retrieval/ingest call sites (`search/semantic.py`, `search/retriever.py`, `ingest/embed.py`, `ingest/taxonomy.py`, and chat/search query-embed in `chat/agent.py`/`web/app.py`) to call `ModelsClient` instead of a local embedder; tests.

**Interfaces:** `Embedder`-shaped methods now served: `embed_image(bytes|list)`, `embed_text(list[str])`, `tag(image, vocab)`. The retriever/ingest keep their signatures but receive a `ModelsClient`-backed embedder shim (an adapter implementing the existing `Embedder` protocol by calling HTTP) so the large retrieval/scoring code is untouched.

- [ ] Steps: introduce `inference/remote_embedder.py::RemoteEmbedder(ModelsClient)` implementing the existing `embedding.base.Embedder` protocol via HTTP; `settings.build_embedder()` returns it (fake path unchanged). The SigLIP backend in the service uses the real `get_siglip_embedder`. Assert `app`/`worker` no longer import torch (grep). Tests + suite green.

---

### Task 3: caption backend + `/caption`; worker caption stage calls the service

Move the plan-14 caption adapters into the service.

**Files:** move `captioning/` usage behind `modelsvc/backends/caption_backend.py` (it owns `OllamaCaptioner`/`VLMCaptioner`); modify `ingest/caption.py::caption_handler` to call `ModelsClient.caption(image, dimensions)` instead of a local captioner; modify `ingest/pipeline.py` (drop the local captioner build); tests.

- [ ] Steps: `/caption` runs the profile's captioner inside the service (mac→Ollama proxy, jetson→in-process VLM). `caption_handler` becomes a thin HTTP call (still maps a cancelled/failed caption to a retry; preemption is now the service's concern — see Task 5). Mac behavior identical (same Ollama call, just one hop away). Tests + suite.

---

### Task 4: text proxy `/plan` `/chat` `/embed/caption-text`; remove direct Ollama use

The service becomes the only Ollama client.

**Files:** create `modelsvc/backends/text_backend.py` (wraps the existing `OpenAICompatClient` → Ollama); route `/plan`, `/chat` (streaming), `/embed/caption-text`; modify `chat/agent.py`, `web/app.py` (chat stream), `chat/retrieve.py` (is_photo_question), `ingest/caption.py` (`_embed_caption`) to call `ModelsClient` instead of `OpenAICompatClient`; tests.

- [ ] Steps: `ModelsClient` gains `plan()`, `chat_stream()` (SSE passthrough), `embed_caption_text()`. `app` uses it; `build_inference_client()` is removed from `app`/`worker` (only the service builds the Ollama client). Streaming chat must still stream token-by-token through the service. Tests + suite; mac chat unchanged.

---

### Task 5: in-process residency manager (replaces the coordinator)

Inside the service: keep one heavy in-process model resident at a time (SigLIP xor VLM) within a RAM budget; interactive embed preempts an in-flight caption.

**Files:** create `modelsvc/residency.py` (an in-process manager: `with residency.use("siglip"|"caption"): ...` — loads the needed model, frees the other, priority so `/embed/*` preempts `/caption`); wire into the backends; `/resources` reports it; tests (fake models, assert eviction + preempt ordering).

- [ ] Steps: a `threading`/async lock + priority; on jetson, loading the VLM frees SigLIP and vice-versa (`release_siglip_embedder`/drop VLM + `empty_cache`). Text (Ollama) is a separate backend, unaffected. Tests assert: a caption in flight yields to an embed request; only one heavy model resident.

---

### Task 6: compose + Dockerfiles (new service; app/worker images drop torch)

**Files:** `compose.yaml` (+ `compose.mac.yaml`/`compose.jetson.yaml`/`compose.cloud.yaml`): add the `models` service (jetson: `Dockerfile.jetson`-style cu132 image + gcc/bitsandbytes/accelerate + GPU; mac: SigLIP CPU, reaches host Ollama; cloud: GPU). `app`/`worker` now build from a **slim** image with NO torch/transformers. `pyproject.toml`: split deps — core (app/worker: fastapi, httpx, sqlite-vec, pillow…) vs a `models` optional group (torch, transformers, bitsandbytes…). `Dockerfile` (app/worker) drops the heavy deps; the models image installs them.

- [ ] Steps: define the service, its ports, GPU flags (jetson/cloud), HF cache mount; set `models_base_url` per profile; ensure `app`/`worker` images no longer install torch (big slim-down). Verify compose config parses.

---

### Task 7: delete the cross-process coordinator + `model_lease`

**Files:** remove `models/coordinator.py`, `models/workloads.py` lease bits, `models/lease_store.py`, the `model_lease` table (`db/schema.sql`), and all wiring (`web/deps.py::make_coordinator`, `ingest/cli.py`, `ingest/pipeline.py` `_lease`, `web/app.py` `require(...)`). Preemption/residency now live in the service (Task 5). Keep `inference/client.py::warm/evict` only if the text backend still needs them (else drop). Update/remove the now-obsolete coordinator tests.

- [ ] Steps: excise the lease; `app`/`worker` hold no models so they need no coordinator. Ensure chat/search/ingest still work through `ModelsClient`. Suite green.

---

### Task 8: docs reconciliation (§8/§8.1 + resource bar §13)

**Files:** `docs/design.md` — rewrite §8.1 (remove the `ModelCoordinator`/`model_lease` description; point to §5.1's in-process residency); §8 caption/embed stages now say "call the `models` service"; §13 resource bar reads the service's `/resources`; check §3.1 for consistency. `models/resources.py` → `/api/resources` proxies `ModelsClient.resources()`.

- [ ] Steps: doc matches the shipped code; no residual "in-process SigLIP in app/worker" or "model_lease" claims.

---

### Task 9: enforcement gate + full verification

- [ ] Grep gate (a test): assert no `import torch`/`transformers`/`bitsandbytes` and no direct Ollama base-url use anywhere under `app`/`worker`/`ingest`/`chat`/`search`/`web` (only `modelsvc/` may). Fail the build otherwise.
- [ ] `uv run pytest -q` green; `ruff check` clean.

---

### Task 10: on-board verification (board-only)

- [ ] `make run-jetson`: `models` service loads SigLIP+VLM once; `app`/`worker` are thin. Confirm via `docker stats` that only the `models` container holds a large RSS, total well under 8 GB, captions complete on GPU, chat/search work, HTTP is fast. Confirm mac still works (host Ollama + SigLIP CPU in the models service).

---

## Notes
- This removes plan-13's cross-process lease and folds plan-14's caption adapters into the service. Those plans' code is superseded, not lost — the adapters and the VLM captioner move into `modelsvc/backends/`.
- Streaming chat through the service (Task 4) is the trickiest seam — keep SSE passthrough token-by-token.
- The RAM win: one torch process + one heavy model resident (≤2.7 GB) + Ollama text + thin app/worker ≈ well under 8 GB on jetson.
