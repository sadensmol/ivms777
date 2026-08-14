# Photo Library Organizer — Plan 05: Captions and per-photo AI title/description

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every photo gets a **caption** (a sentence describing it), a short **AI-written title and description**, and VLM-chosen **tags** — produced by a vision model over the inference service. The `/photo` AI panel fills in (§13), captions feed keyword search, and the FTS index carries real sentences. This is spec phase 3 and the last prerequisite for Memories (plan 07).

**Architecture:** A `caption` job stage — the slowest, so it drains **last**, after `taxonomy` (§8) — sends each photo's detail thumbnail to the inference service (Ollama on `mac`/`jetson`, vLLM on `cloud`) through the existing OpenAI-compatible `InferenceClient`, asking for one JSON object: `{caption, title, description, tags{dimension:[label]}}`. The result is written to `photos.caption`, `photos.caption_model`, new `photos.ai_title`/`photos.ai_description`, and `photo_tags(source='vlm')`; then the FTS row is rebuilt so the caption is searchable. Prompts live in a per-model registry (`inference/prompts.py`) keyed by model name, all satisfying one JSON schema — adding a model is adding a template, not touching the pipeline (§4).

**Tech Stack:** Python 3.12, the existing `inference/client.py` (`OpenAICompatClient`, `encode_image`) and `inference/fakes.py` (`FakeInferenceClient`), SQLite, FastAPI, Jinja2. No new runtime services beyond the inference backend that already ships per profile.

**Spec:** `docs/design.md` — §4 (caption model per profile, prompt registry, one JSON schema), §7 (`source='vlm'` tags), §8 (caption stage last), §10 (captions ground chat), §13 (`/photo` AI title/description + caption; captions in the grid hover), §16 phase 3.

**Builds on:** the job queue + FTS reindex (plan 04), the inference client + fakes (phase 0). Captions add to the FTS `caption` column that `reindex_fts` already writes (currently empty).

**Covers:** the caption stage, per-photo AI title/description, VLM tags, the `/photo` AI panel, and captions in keyword search. **Deferred:** caption vocabulary mining (§7.1, plan 06); the query planner and chat (plans 06, 08).

## ⚠️ External dependency — read first

- The caption stage needs a **running vision model** on the inference backend. On `mac` that is **Ollama on the host** (`make ollama` won't pull a real caption model yet — see below); the containers reach it at `host.docker.internal:11434`.
- **The design's `caption_model` values (`gemma4:26b-a4b`, `gemma4:e4b`) are placeholders — no such Ollama tags exist.** Point `caption_model` at a **real** vision model before running this stage. Good Mac choices: `qwen2.5vl:7b`, `llava:13b`, `moondream`, or `gemma3:4b`/`gemma3:12b` (Gemma 3 is multimodal). Task 6 reconciles the design's model table with whatever is chosen.
- Everything in this plan is **unit-tested offline with `FakeInferenceClient`** — no model, no network. Only the task-6 hand-run needs Ollama.

## Global Constraints

- Python 3.12. Dependencies via `uv` with a committed `uv.lock`.
- **Never run `git commit`/`git add`.** The user commits. Every task ends at a checkpoint.
- Every user-scoped query filters on `owner_id`.
- Tests never hit the network or a real model — the caption stage is driven by `FakeInferenceClient`.
- The full fast suite passes at the end of every task: `uv run pytest -q`; `uv run ruff check .` clean.
- The `caption` stage name already exists in `ingest/jobs.STAGES`; this plan supplies its handler, drained after `taxonomy`.

---

### Task 1: Prompt registry, JSON schema, and the caption columns

**Files:**
- Create: `inference/prompts.py`
- Create: `tests/test_prompts.py`
- Modify: `db/schema.sql`, `db/connection.py` (add `ai_title`, `ai_description`; bump `SCHEMA_VERSION` to 3)

**Interfaces:**
- Produces:
  - `inference.prompts.CAPTION_SCHEMA` — the JSON schema every caption response must satisfy: `caption` (str), `title` (str), `description` (str), `tags` (object of dimension → list[str]).
  - `inference.prompts.caption_messages(model: str, image_data_uri: str, dimensions: list[str]) -> list[ChatMessage]` — the chat messages for a model, from a per-model template registry with a sensible default; embeds the vocabulary dimensions so the VLM picks from the same taxonomy.

- [ ] **Step 1: Failing test** — `CAPTION_SCHEMA` has the four required keys; `caption_messages("qwen2.5vl:7b", uri, ["subject","vibe"])` returns a system+user message pair whose user content includes the image and asks for JSON; an unknown model falls back to the default template (no `KeyError`).
- [ ] **Step 2:** run → `ModuleNotFoundError`.
- [ ] **Step 3:** write `inference/prompts.py` — a `_TEMPLATES` dict keyed by model with a `_DEFAULT`, each producing messages that (a) show the image (`{"type":"image_url","image_url":{"url":image_data_uri}}`), (b) instruct "reply with ONLY JSON matching this shape", (c) list the dimensions. Keep the grounding rule: describe only what is visible; no invented people/places.
- [ ] **Step 4:** add `ai_title TEXT` and `ai_description TEXT` to `photos` in `schema.sql`; bump `SCHEMA_VERSION = 3`; in `migrate()` add the guarded v2→v3 `ALTER TABLE photos ADD COLUMN` for each (mirrors the plan-07 `groups.description` migration pattern — check `PRAGMA table_info` first).
- [ ] **Step 5–6:** run the prompt tests (PASS), whole suite (PASS). Checkpoint — report the schema, the registry shape, and the two new columns.

*(If plan 07's `groups.description` migration already took v3, use the next free version and fold both `ALTER`s into one hop.)*

---

### Task 2: The caption stage

**Files:**
- Create: `ingest/caption.py`
- Create: `tests/test_caption_stage.py`

**Interfaces:**
- Consumes: `InferenceClient` (`complete` with `json_schema`), `encode_image` (`inference/client.py`), the detail thumbnail via `Storage`, the vocabulary dimensions.
- Produces: `ingest.caption.caption_handler(derived, client, model, dimensions) -> StageHandler` — reads the detail thumbnail, calls the model for the JSON object, and writes `caption`, `caption_model`, `ai_title`, `ai_description`, `photo_tags(source='vlm')`, then `reindex_fts` (reuse `ingest.taxonomy.reindex_fts`). Unusable/invalid JSON fails the job (retried per the queue), never writes half a row.

- [ ] **Step 1: Failing test** — with a `FakeInferenceClient` queued to return `{"caption":"a dog on a beach","title":"Beach day","description":"A dog runs on the sand.","tags":{"subject":["pet"],"setting":["beach"]}}`, draining `caption` sets `photos.caption`/`ai_title`/`ai_description`/`caption_model`, inserts `vlm` tags for pet+beach, and makes `photo_fts MATCH 'beach'` hit. A response that isn't valid JSON leaves the photo uncaptioned and the job failed (not a crash).
- [ ] **Step 2:** run → `ModuleNotFoundError`.
- [ ] **Step 3:** write `ingest/caption.py`. Load the detail thumbnail bytes, `encode_image` → data URI, `caption_messages(...)`, `client.complete(model, messages, json_schema=CAPTION_SCHEMA)`, `json.loads`, then write the row + tags + FTS in one go. Map `tags` to `photo_tags` via `tag_id_map`, `source='vlm'`, score 1.0 (the VLM asserts, it does not score); skip labels not in the vocabulary.
- [ ] **Step 4–5:** run the stage tests (PASS), whole suite (PASS). Checkpoint — report the write set and the JSON-failure behaviour.

---

### Task 3: Wire the stage, backfill, and the inference client factory

**Files:**
- Modify: `config.py` (add `build_inference_client()`), `ingest/receive.py` (enqueue `caption`), `web/app.py` and `ingest/cli.py` (drain `caption` + backfill)
- Modify: `tests/test_config.py`, `tests/test_receive.py`

**Interfaces:**
- Produces:
  - `config.Settings.build_inference_client() -> (InferenceClient, caption_model)` — real `OpenAICompatClient(inference_base_url)` normally; a fake path (`use_fake_inference=True`, default in the test `settings` fixture) returns a `FakeInferenceClient` so the suite never needs a backend.
  - `ingest.caption.backfill_captions(conn) -> int` — enqueue `caption` for every photo without one (mirrors `backfill_taxonomy`).

- [ ] **Step 1: Failing test** — `receive` enqueues `caption` (`stage_counts(... 'caption')['pending'] == 1`); `settings.build_inference_client()` returns a fake when `use_fake_inference=True`.
- [ ] **Step 2–3:** add the factory to `config.py` (local imports so no backend is contacted unless real). Add `use_fake_inference: bool = False` and set it `True` in the `tests/conftest.py` `settings` fixture (alongside `use_fake_embedder`). Enqueue `caption` in `receive.py` after `taxonomy`. In `web/app.py` `drain_now` and `ingest/cli.py`, build the client + caption handler and add `"caption": caption_handler(...)`, plus `backfill_captions(...)` alongside the other backfills. Because `STAGES` order is `thumbnail, embed, taxonomy, caption`, captions drain last — the §8 ordering the Jetson needs.
- [ ] **Step 4–5:** run tests (PASS), whole suite (PASS). Checkpoint — report the drain order and the fake-by-default switch.

---

### Task 4: The `/photo` AI panel

**Files:**
- Modify: `web/app.py` (`photo_detail` passes `ai_title`/`ai_description`), `web/templates/photo.html`, `tests/test_web_photo.py`

- [ ] **Step 1: Failing test** — a photo with `caption`, `ai_title`, `ai_description` set renders all three on `/photo`; the panel shows the caption model name.
- [ ] **Step 2–3:** the route already selects `*`; pass `ai_title`/`ai_description` to the template. Replace the "AI title/description fill in with the caption phase" placeholder in `photo.html` with the real fields (title as a heading, description as a paragraph, caption below), keeping the existing tags block (now including `vlm` tags with their source badge).
- [ ] **Step 4–5:** run tests (PASS), whole suite (PASS). Checkpoint.

---

### Task 5: Captions in search and the grid

Confirm the sentences captions add actually help retrieval — the grid hover already shows `photo.caption`; keyword/fusion already read `photo_fts.caption` (populated now).

**Files:**
- Modify: `tests/test_web_search.py`

- [ ] **Step 1: Failing/again test** — a photo whose caption contains a distinctive word (no matching tag) is found by `/library?q=<word>` via keyword→fusion. (This asserts captions reach FTS through the caption stage, end to end with the fake.)
- [ ] **Step 2–3:** it should already pass once the caption stage writes FTS; if not, fix `reindex_fts`/wiring. No new UI — the hover caption and search box already exist.
- [ ] **Step 4:** whole suite + `ruff` (PASS, clean). Checkpoint.

---

### Task 6: Real model, model-name reconciliation, and docs

- [ ] **Step 1: Pick and pull a real vision model.** On the host: `ollama pull qwen2.5vl:7b` (or another from the list above). Set `IVMS777_CAPTION_MODEL` (or the profile default) to it.
- [ ] **Step 2: Caption the library.** Bring the stack up, run **Re-caption** (`POST /reprocess?from=caption`, or let `backfill_captions` run), and watch the `caption` stage drain on `/upload`. Then confirm on `/photo`: caption, AI title, AI description, and `vlm` tags all present and *accurate to the image* (no invented content); and that `/library` search finds a word that appears only in a caption.
- [ ] **Step 3: Reconcile the design.** Update `docs/design.md` §3.1/§4's caption-model table to the real model(s) chosen (replace the `gemma4:*` placeholders, or clearly mark them "when available" with the working default named). Update `README.md` (captions need a vision model via `make ollama` + the chosen tag). Update §16 phase 3 as delivered.
- [ ] **Step 4: Checkpoint** — report the model used, caption quality spot-checks, and per-photo latency. Stop. Do not commit.

---

## What plan 05 delivers

Open any photo and read a real **caption**, an AI **title** and **description**, and the model's own **tags** — all written on your box by a vision model, never sent to a third party. Those sentences flow into search, so typing a word that only appears in a caption finds the photo. Captions drain last, one resumable stage after tagging, so search and tags are useful long before the slow captioner catches up. And with captions in place, the last dependency for **Memories** (plan 07) is met.

**Not yet:** caption-driven vocabulary mining (§7.1), the query planner and parsed-filter chips (plan 06), and chat (plan 08).

## Following plans

| Plan | Spec phase | Delivers |
|---|---|---|
| 06 | 4 | Query planner, parsed-filter chips, caption vocabulary mining |
| 07 | 5 | Memories — agentic, persisted albums (`docs/plans/07-memories.md`) |
| 08 | — | Place-name sidebar filter (`docs/plans/08-place-names.md`, organizer + filter done) |
| later | 6 | Ask-your-library chat with streaming and citations |
| later | 7 | Stage 2 — the `ivms777-sync` CLI (layouts, manifest, plan/apply/undo) |
