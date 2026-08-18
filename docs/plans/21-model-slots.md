# Model slots — a settings popup that switches models while the app runs — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the user open a ⚙ popup from the nav, see which model each of the
four slots uses right now, which candidates are already downloaded, download one
(with progress), and switch to it while the app runs — with the library re-indexed
automatically when the switch invalidates stored data.

**Architecture:** A **catalog** (`models/catalog.py`, plain data, no torch) lists
every selectable model with everything its consumers need: source, download size,
resident cost, embedding width, preprocessing contract, prompt template. The four
**slots** (`image_embed`, `text_embed`, `caption`, `planner`) each hold one catalog
key. `app` owns the choice in `app_settings` and resolves stored → env → profile
default. The `models` service builds its `ModelRegistry` specs *from* the resolved
slots instead of hardcoding four names, exposes `GET /models/catalog` /
`PUT /models/slots` / `POST /models/download`, and reports a `generation` so `app`
can re-push after a restart. Switching a slot writes the choice, rebuilds
`photo_vec` when the embedding width changes, and calls `jobs.reprocess()` over the
stages that switch invalidates.

**Tech Stack:** Python 3.12, FastAPI, HTMX + a `<dialog>`, SQLite, `huggingface_hub`
(already present via `transformers`), pytest, `uv`.

**Spec:** [`docs/design.md`](../design.md) §4.1 (model slots — the catalog and
runtime switching), §5.1 (the models service HTTP surface), §8.1 (residency and
cost), §6 (`app_settings`, `photo_vec` width), §13 (the settings overlay);
[`docs/models.md`](../models.md) (catalog contents, switch mechanics, downloads);
[`docs/ui.md`](../ui.md) (the popup's exact behaviour).

## Global Constraints

- **One model process** (CLAUDE.md, §5.1). The catalog is *data* — `app` may
  import it. Nothing in this plan makes `app`/`worker` import torch, load a model,
  or talk to a backend directly. Downloads run **inside the `models` service**,
  because it owns the model files.
- **The `models` parent stays torch-free** (§8.1). `huggingface_hub` is not torch;
  a download must not import `transformers`.
- **Residency units are named by slot**: `image_embed`, `text_embed`, `llm`,
  `llm_vision`. This renames today's `siglip` / `nomic` / `gemma` / `gemma-vision`
  everywhere (registry, composite "needs" lists, cost map, tests, `/resources`
  consumers).
- **Cost figures are the Jetson ones** — the binding profile (§8.1). Every catalog
  entry must satisfy `cost_mb + headroom_mb <= ram_budget_mb` for each profile it
  is offered on, or it is not offered there. An entry whose cost is an estimate
  carries `cost_measured=False` and says so in the UI.
- **Never migrate vectors between encoders.** A width change drops `photo_vec` and
  requeues the embed stage. Two encoders' vectors are not comparable.
- `uv run` for everything. Tests live in `tests/`. New module ⇒ new tests.
- The design docs are already updated (§4.1 etc.); this plan implements them. If an
  implementation detail contradicts the doc, fix the doc in the same task.

---

### Task 1: The catalog — `models/catalog.py`

**Files:**
- Create: `models/catalog.py`
- Test: `tests/test_catalog.py`
- Modify: `tests/test_config.py` (budget test runs over the catalog)

**Interfaces:**

```python
SLOTS: tuple[str, ...] = ("image_embed", "text_embed", "caption", "planner")

# Which ingest stages a switch of this slot invalidates (inclusive range).
INVALIDATES: dict[str, tuple[str, str] | None] = {
    "image_embed": ("embed", "taxonomy"),
    "text_embed": ("caption_embed", "caption_embed"),
    "caption": ("caption", "caption_embed"),
    "planner": None,
}

@dataclass(frozen=True)
class Preprocess:
    input_px: int
    resample: str          # "bilinear" | "bicubic"
    mode: str              # "squash" (stretch to input_px²) | "native" (fit inside, aspect kept)

@dataclass(frozen=True)
class HfSource:
    repo: str

@dataclass(frozen=True)
class GgufSource:
    repo: str
    file: str
    mmproj_repo: str | None = None
    mmproj_file: str | None = None

@dataclass(frozen=True)
class ModelEntry:
    key: str
    slot: str
    display: str
    source: HfSource | GgufSource
    size_mb: int                 # download size, for the UI
    cost_mb: int                 # resident cost the governor budgets against (jetson figure)
    cost_measured: bool          # False ⇒ estimate, shown as unverified
    profiles: tuple[str, ...]
    dim: int | None = None       # embedder slots only
    preprocess: Preprocess | None = None   # image_embed only
    prompt_template: str | None = None     # caption/planner only
    note: str | None = None

CATALOG: tuple[ModelEntry, ...]

def get(key: str) -> ModelEntry                       # KeyError on unknown
def entries_for(slot: str, profile: str) -> list[ModelEntry]
def default_key(slot: str, profile: str) -> str
def is_switchable(slot: str, profile: str) -> bool    # False on cloud
```

- [ ] **Step 1: Write the failing tests** (`tests/test_catalog.py`)

Cover, with no mocks:

- `default_key(slot, profile)` resolves for all 4 slots × 3 profiles, and each
  default is in `entries_for(slot, profile)`.
- keys are unique; every entry's `slot` is in `SLOTS`; every `profiles` value is a
  real profile.
- **field completeness by slot**: `image_embed` ⇒ `dim` and `preprocess` set,
  `prompt_template` unset; `text_embed` ⇒ `dim` set, `preprocess` unset;
  `caption`/`planner` ⇒ `prompt_template` set, `dim`/`preprocess` unset.
- every `caption`/`planner` entry's `prompt_template` is accepted by
  `inference.prompts.caption_messages(model=…)` — today `_SYSTEM_BY_MODEL` is empty
  and every model falls back to `_DEFAULT_SYSTEM`, so the assertion is "it resolves
  to a non-empty system prompt", not "it has a bespoke one".
- `preprocess.mode` ∈ {"squash", "native"}; `resample` ∈ {"bilinear", "bicubic"}.
- the defaults reproduce today's behaviour exactly: `image_embed` default is
  `siglip2-so400m-384` with `Preprocess(384, "bilinear", "squash")` and `dim=1152`;
  `caption`/`planner` defaults are the same gemma GGUF on mac/jetson.
- `is_switchable(slot, "cloud") is False` for every slot.

- [ ] **Step 2: Write `models/catalog.py`** — data only, no imports beyond
`dataclasses`/`typing`. Ships:

| Slot | Entries (default first) |
|---|---|
| `image_embed` | `siglip2-so400m-384` (1152, squash 384, measured 3400) · `siglip2-so400m-512` (1152, squash 512, estimate) |
| `text_embed` | `nomic-1.5` (768, measured 2200) · `embeddinggemma-300m` (768, estimate) · `qwen3-embed-0.6b` (1024, estimate) |
| `caption` | `gemma4-E2B` (measured 4300) · `qwen3-vl-4b` (estimate) · `qwen3-vl-8b` (mac only, estimate) |
| `planner` | `gemma4-E2B` (measured 3800) · `qwen3-4b-2507` (estimate) · `qwen3-vl-8b` (mac only, estimate) |

Cloud entries stay `qwen2.5vl:7b` / `qwen2.5:3b`, `profiles=("cloud",)`.

- [ ] **Step 3: Move the budget test onto the catalog**

`tests/test_config.py::test_every_model_fits_its_profile_budget` currently walks
`settings.model_cost_mb`. Rewrite it to assert, for every profile and every entry
offered on it: `entry.cost_mb + headroom_mb <= ram_budget_mb`. An entry that fails
must be removed from that profile's `profiles`, never "fixed" by raising the budget.

- [ ] **Step 4: Run `uv run pytest tests/test_catalog.py tests/test_config.py`**

---

### Task 2: `app_settings` and slot resolution

**Files:**
- Modify: `db/schema.sql`, `db/connection.py` (`SCHEMA_VERSION` 9 → 10)
- Create: `db/settings.py`
- Create: `models/slots.py`
- Modify: `config.py` (env overrides are catalog keys)
- Test: `tests/test_app_settings.py`, `tests/test_slots.py`

**Interfaces:**

```python
# db/settings.py
def get_setting(conn, owner_id: int, key: str) -> str | None
def set_setting(conn, owner_id: int, key: str, value: str) -> None
def all_settings(conn, owner_id: int) -> dict[str, str]

# models/slots.py
SETTING_PREFIX = "model_slot."
def resolve(conn, settings) -> dict[str, ModelEntry]     # stored → env → profile default
def resolve_key(conn, settings, slot: str) -> str
```

- [ ] **Step 1: Failing tests**

`tests/test_app_settings.py` — round-trip a setting; upsert overwrites and bumps
`updated_at`; settings are owner-scoped; a fresh DB has none.

`tests/test_slots.py` — resolution order: with nothing stored, `resolve` returns
the profile defaults; an env override (`IVMS777_CAPTION_MODEL=qwen3-vl-4b`) beats
the default; a stored value beats the env override; a stored key that is not in the
catalog (or not offered on this profile) falls back to the default **and does not
raise**, so a downgraded install still boots.

- [ ] **Step 2: Schema**

Add to `db/schema.sql` (matching `docs/data-model.md`):

```sql
CREATE TABLE IF NOT EXISTS app_settings (
  owner_id   INTEGER NOT NULL,
  key        TEXT NOT NULL,
  value      TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  PRIMARY KEY (owner_id, key)
);
```

Bump `SCHEMA_VERSION` to 10. No `_ADDED_COLUMNS` entry is needed — the whole table
is new and `CREATE TABLE IF NOT EXISTS` covers both fresh and existing DBs.

- [ ] **Step 3: Implement `db/settings.py` + `models/slots.py`**

`config.Settings.caption_model` / `planner_model` keep their names but are now
documented as **catalog keys**; a value that is not a catalog key is passed through
only on `cloud` (config-only slots), everywhere else it is ignored with a log line.

- [ ] **Step 4: `uv run pytest tests/test_app_settings.py tests/test_slots.py`**

---

### Task 3: The service builds its registry from slots

**Files:**
- Create: `modelsvc/slots.py`
- Modify: `modelsvc/backends/__init__.py`, `modelsvc/backends/composite.py`,
  `modelsvc/llm_process.py`, `config.py` (`model_cost_mb` becomes an override map)
- Test: `tests/test_modelsvc_slots.py`; update every test that names the four
  residency units — `tests/test_modelsvc_backends.py`, `tests/test_registry.py`,
  `tests/test_governor.py`, `tests/test_models_control_api.py`,
  `tests/test_inference_residency.py`, `tests/test_ingest_preempt.py`,
  `tests/test_llm_process.py`

**Interfaces:**

```python
class SlotManager:
    def __init__(self, settings, registry, *, llm, torch_worker_factory): ...
    def apply(self, slots: dict[str, str]) -> None   # switch; evicts what changed
    def state(self) -> dict                          # {"slots": {...}, "generation": int}
    def entry(self, slot: str) -> ModelEntry
    generation: int                                  # bumps on every accepted apply
```

- [ ] **Step 1: Failing tests** (`tests/test_modelsvc_slots.py`, fakes only — no
torch, no llama-server)

- registering from the default slots produces exactly the four units
  `image_embed`, `text_embed`, `llm`, `llm_vision`, with `cost_mb` taken from the
  catalog entry;
- `apply({"image_embed": "siglip2-so400m-512"})` **unloads** the resident
  `image_embed` (its `free` is called) and re-registers the unit with the new
  entry's cost, and bumps `generation`;
- applying the same slots again is a no-op: no unload, no generation bump;
- applying an unknown key raises and changes nothing (generation unmoved);
- `llm` and `llm_vision` stay mutually exclusive after a switch;
- the GGUF path handed to `llama-server` comes from the entry, not a constant.

- [ ] **Step 2: Rename the residency units**

`siglip`→`image_embed`, `nomic`→`text_embed`, `gemma`→`llm`,
`gemma-vision`→`llm_vision` in `backends/__init__.py`, `composite.py` (the
`_run([...])` needs lists, `reap_idle`, `text_embed_needs`), `config.model_cost_mb`
keys and every test that names them. `config.model_cost_mb` keeps its env name
(`IVMS777_MODEL_COST_MB`) but becomes an **override keyed by catalog key**; the
catalog supplies the default, so an unmeasured entry can be corrected on the board
without a code change.

- [ ] **Step 3: Implement `SlotManager`** and make `build_backend` use it:
`build_backend(settings)` resolves the profile defaults (the service has no DB) and
calls `SlotManager.apply` once. `llm_process.build_llm_process` takes the resolved
`caption`/`planner` entries instead of hardcoding `gemma-4-E2B-it-Q4_K_M.gguf` and
`settings.mmproj_name`; `vision_args` comes from the `caption` entry's mmproj.
Different GGUFs for the two slots ⇒ a restart on mode change (already the shape
`SubprocessLlm.load(vision=…)` implements).

- [ ] **Step 4: `uv run pytest tests/` — the rename must leave every existing test
green**, including `tests/test_one_model_process_gate.py`: `models/catalog.py` is
imported by `app`, so it must stay free of torch and of any model-loading import.

---

### Task 4: Downloads — `modelsvc/downloads.py`

**Files:**
- Create: `modelsvc/downloads.py`
- Test: `tests/test_downloads.py`

**Interfaces:**

```python
class Downloads:
    def __init__(self, model_dir: Path, *, hf_fetch=..., http_fetch=...): ...
    def status(self, entry: ModelEntry) -> dict
        # {"state": "absent"|"downloading"|"ready"|"error",
        #  "bytes": int, "total": int, "error": str | None}
    def start(self, entry: ModelEntry) -> None   # idempotent; background thread
    def is_present(self, entry: ModelEntry) -> bool
```

- [ ] **Step 1: Failing tests** — inject fake fetchers, no network:

- a GGUF entry whose file is already in `model_dir` reports `ready` without
  fetching (a manually placed GGUF counts — presence is a path probe, not a flag);
- `start` reports `downloading` with rising `bytes`, then `ready`, and the file
  lands at the path `llama-server` is pointed at;
- a GGUF entry with an mmproj fetches **both**, and is `ready` only when both are
  present;
- a failing fetch ends `error` with the message, and a second `start` retries;
- `start` twice while in flight spawns one fetch, not two;
- an HF entry probes the HF cache and calls `snapshot_download` once.

- [ ] **Step 2: Implement.** GGUF: `httpx.stream("GET", …)` to a `.part` file, then
atomic rename — a killed download must never leave a truncated file that the path
probe would call `ready`. HF: `huggingface_hub.snapshot_download` (imported inside
the function so the parent's import graph stays light and torch-free), progress from
its callback; when it reports none, fall back to `bytes = total` on completion so the
UI still terminates.

- [ ] **Step 3: `uv run pytest tests/test_downloads.py`**

---

### Task 5: The service's HTTP surface

**Files:**
- Modify: `modelsvc/app.py`, `modelsvc/backends/base.py`,
  `modelsvc/backends/composite.py`, `modelsvc/backends/fake.py`
- Test: `tests/test_modelsvc_api.py` (extend)

**Endpoints** (design §5.1):

- `GET /models/catalog` → `{"generation": n, "slots": {...}, "entries": [{…entry…, "current": bool, "download": {…status…}}]}`
- `PUT /models/slots` `{"slots": {"caption": "qwen3-vl-4b"}}` → applies, returns the new state; 400 on an unknown/not-offered key.
- `POST /models/download` `{"key": …}` → starts, returns the status immediately.
- `GET /embed/spec` → `{"logit_scale", "logit_bias", "preprocess": {...}, "generation"}` — **replaces** `GET /embed/calibration`.
- `GET /models` gains `slots` + `generation`.

- [ ] **Step 1: Failing tests** with `TestClient` over `FakeBackend`: each endpoint's
shape; `PUT` with a bad key is a 400 and leaves the generation unmoved; `/embed/spec`
carries the current `image_embed` entry's preprocess; `/models/catalog` marks exactly
one entry per slot `current`.

- [ ] **Step 2: Implement**, extending the `ModelBackend` protocol with
`catalog()`, `set_slots()`, `download()`, `embed_spec()` and giving `FakeBackend` a
usable in-memory implementation (the app tests depend on it).

- [ ] **Step 3: Delete `/embed/calibration`** and its client method — one caller
(`RemoteEmbedder`), replaced in Task 6.

- [ ] **Step 4: `uv run pytest tests/test_models_service.py`**

---

### Task 6: Preprocessing follows the model

**Files:**
- Modify: `inference/models_client.py`, `inference/remote_embedder.py`
- Test: `tests/test_remote_embedder.py` (extend), `tests/test_models_client.py`

- [ ] **Step 1: Failing tests**

- `mode: "squash"`, `input_px: 384` ⇒ the bytes posted decode to exactly 384×384
  (today's behaviour, now data-driven);
- `mode: "squash"`, `input_px: 512` ⇒ 512×512;
- `mode: "native"`, `input_px: 512` ⇒ the longest side is 512 and the **aspect ratio
  is preserved** (a 3000×2000 original becomes 512×341);
- the spec is fetched **once** for many `embed_images` calls;
- a changed `generation` on the next response invalidates the cache and the new
  `input_px` is used;
- `logit_scale`/`logit_bias` still come from the same call (no second round trip).

- [ ] **Step 2: Implement.** Delete `_SIGLIP_INPUT_PX`; keep its comment's substance
(why the client resizes at all: PNG-encoding a full original cost 5.7 s and 15 MB
per photo on the Jetson) attached to the new spec-driven path, and move the NaFlex
caveat to `mode: "native"`, which now handles it instead of warning about it.

- [ ] **Step 3: `uv run pytest tests/test_remote_embedder.py tests/test_models_client.py`**

---

### Task 7: The switch — vector table, reprocess, one transaction

**Files:**
- Create: `db/vectors.py`
- Modify: `models/slots.py`, `web/deps.py` (startup self-heal)
- Test: `tests/test_vec_dim.py`, `tests/test_slot_switch.py`

**Interfaces:**

```python
# db/vectors.py
def vec_dim(conn) -> int              # parsed from sqlite_master's declared float[N]
def ensure_vec_dim(conn, dim: int) -> bool   # drop+recreate when different; True if rebuilt

# models/slots.py
@dataclass(frozen=True)
class SwitchResult:
    slot: str
    key: str
    photos_requeued: int
    stages: tuple[str, ...]
    vectors_dropped: bool

def preview(conn, settings, slot: str, key: str) -> SwitchResult   # counts only, writes nothing
def switch(conn, settings, slot: str, key: str) -> SwitchResult
```

- [ ] **Step 1: Failing tests**

`tests/test_vec_dim.py` — `vec_dim` reads 1152 from a fresh DB; `ensure_vec_dim`
with the same width is a no-op and **keeps existing vectors**; with a different
width it drops every row and the new declared width is readable back.

`tests/test_slot_switch.py` —

- switching `image_embed` to a same-width model requeues `embed`+`taxonomy` for
  every owner photo and does **not** touch `photo_vec`'s schema, but the stale
  vectors are still requeued (they will be overwritten);
- switching `image_embed` to a different-width model **also** rebuilds `photo_vec`
  (rows gone) — and the two happen in one transaction: an injected failure in
  `reprocess` leaves the old slot value *and* the old vectors intact;
- `text_embed` requeues only `caption_embed`;
- `caption` requeues `caption`+`caption_embed`;
- `planner` requeues nothing and writes only the setting;
- switching to the already-active key is a no-op returning `photos_requeued=0`;
- an unknown key, or one not offered on this profile, raises and writes nothing;
- `preview` returns the same counts as `switch` but leaves the DB untouched.

- [ ] **Step 2: Implement.** Wrap the write in `BEGIN IMMEDIATE` … `COMMIT`
explicitly — the connection is `isolation_level=None` (autocommit), so without it
the three writes are three transactions and a mid-way failure leaves the library in
a state the doc says it is never in.

- [ ] **Step 3: Startup self-heal.** In `web/deps.build_context`, after `migrate`,
call `ensure_vec_dim(conn, resolve(conn, settings)["image_embed"].dim)` and, when it
rebuilds, `reprocess(conn, owner_id, "embed", "taxonomy")` — so a DB whose width
disagrees with the selected model (e.g. the process died mid-switch) repairs itself
instead of failing every KNN query.

- [ ] **Step 4: `uv run pytest tests/test_vec_dim.py tests/test_slot_switch.py`**

---

### Task 8: `app` routes — `/settings/models`

**Files:**
- Modify: `web/app.py`, `inference/models_client.py`
- Test: `tests/test_web_settings.py`

**Routes:**

- `GET /settings/models` → the dialog body (HTML fragment): the four slots, their
  entries, current selection, download state, and — when `?select=<slot>:<key>` is
  present — the consequence line from `preview()` and an enabled **Switch**.
- `POST /settings/models` `slot`, `key` → `switch()`, then `PUT /models/slots`;
  re-renders the fragment. A `models`-service error rolls nothing back (the DB is
  the source of truth) but is shown in the fragment, and the next resources poll
  re-pushes.
- `POST /settings/models/download` `key` → proxies `POST /models/download`,
  re-renders the fragment.

- [ ] **Step 1: Failing tests** (`TestClient`, fake `ModelsClient`)

- the fragment marks exactly one entry per slot as current, and it is the resolved
  one (stored beats default);
- an entry the service reports as present renders "on disk"; one downloading
  renders its percentage; one absent renders **Download**;
- `?select=` on a different key renders the consequence line with the real photo
  count, and `Switch` is **disabled** when the model is not on disk;
- `POST` switches the slot, calls `PUT /models/slots` exactly once with the full
  slot map, and the re-rendered fragment shows the new current entry;
- `POST` with an unknown key is a 400 and changes nothing;
- on `cloud`, every slot renders read-only and `POST` is refused;
- the resources poll re-pushes slots when the service reports a `generation` this
  `app` did not set, and does **not** re-push when it matches.

- [ ] **Step 2: Implement.** Add `catalog()`, `set_slots()`, `download()`,
`embed_spec()` to `ModelsClient`. The re-push lives in the existing
`GET /api/resources` handler — it already talks to the service every ~2 s, so no new
polling loop appears.

- [ ] **Step 3: `uv run pytest tests/test_web_settings.py`**

---

### Task 9: The popup itself

**Files:**
- Modify: `web/templates/base.html`, `web/static/app.css`
- Create: `web/templates/_settings_models.html`, `web/static/settings.js`
- Test: `tests/test_web_settings.py` (extend with markup assertions)

- [ ] **Step 1: Failing tests** — the nav contains a `⚙` button targeting the
dialog; the dialog is a `<dialog>` (not a route) and the button is **not** an
`<a href>`, so no history entry and no URL change (design §13.1); the fragment
polls (`hx-trigger="every 1s"`) **only** while a download is in flight.

- [ ] **Step 2: Implement.**

`base.html`: a `<button id="settings-open">` right after `#resbar`, plus an empty
`<dialog id="settings">` whose body is loaded by HTMX from `/settings/models` on
first open. `settings.js`: `showModal()`/`close()`, Esc and backdrop click close it,
focus returns to the ⚙ button. Radios post `?select=` to re-render (HTMX
`hx-get` + `hx-target`), so the consequence line is server-rendered — no duplicated
logic in JS.

`app.css`: dialog sizing, per-slot sections, the progress bar, the "estimated" and
"unverified cost" badges. Follow the existing token/spacing conventions in
`app.css` rather than introducing new ones.

- [ ] **Step 3: `uv run pytest tests/ && uv run ruff check .`**

---

### Task 10: Docs and board verification

- [ ] **Step 1: Re-read the design against the code** — §4.1, §5.1 (incl. the §5
mermaid), §6, §8.1, §13, `docs/models.md`, `docs/data-model.md`, `docs/ui.md`.
Anything the implementation did differently gets fixed **in the doc** now.

- [ ] **Step 2: Mac smoke test**

```bash
make llama-mac && uv run pytest tests/ && make up
```

Open ⚙: all four slots list their entries, the defaults are current, gemma and
SigLIP show "on disk". Switch `text_embed` to `embeddinggemma-300m`: it downloads
with visible progress, the confirm line names `caption_embed` and the photo count,
and after Switch the `/upload` progress rows show `caption_embed` requeued.

- [ ] **Step 3: Jetson verification** (the profile that can actually fail)

```bash
ssh lockbox@192.168.100.8 'docker logs -f ivms777-models-1'
```

- switch `caption` to `qwen3-vl-4b`, watch `llama-server` restart with the new
  GGUF, and confirm a caption returns real text;
- `free -m` before/after the switch: the outgoing model's RAM comes back (the
  eviction is a process kill, §8.1);
- **measure** each newly used entry's cost the way §8.1 requires — the drop in
  `psutil.virtual_memory().available` across a load — and replace its estimate in
  `models/catalog.py`, flipping `cost_measured=True`;
- switch `image_embed` to `siglip2-so400m-512`, confirm `photo_vec` was **not**
  rebuilt (same width 1152) and that embeds re-run at 512 with the new spec.

- [ ] **Step 4: Report the measured numbers, then stop** — the user commits.

## Out of scope (deliberate)

- Free-form model ids — see §4.1 for why the catalog is closed.
- Download cancel/resume, and deleting downloaded weights from the UI.
- Switching slots on `cloud` (vLLM serves one model per container; read-only).
- Keeping two indexes so search stays hot during a re-embed.
- Per-slot device overrides (CPU vs CUDA) — still `embed_device` for all.
