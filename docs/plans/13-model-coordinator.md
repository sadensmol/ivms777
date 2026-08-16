# Model Coordinator + Resource Bar Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make one component decide which models are resident from a declared *workload*, so at most one workload's models occupy RAM at a time, and surface it in a live resource bar.

**Architecture:** A `ModelCoordinator` (single decision point) maps a `Workload` to a model-set, guards it against a per-profile RAM budget, takes a cross-process **lease row in the shared SQLite DB**, and reconciles residency (in-process SigLIP load/release; Ollama LLM warm/evict). Interactive workloads (chat, memory rebuild) hard-preempt background ingest: the worker checks the lease at stage boundaries + before the slow caption call, requeues the in-flight job, releases models, and yields. A `/api/resources` endpoint + a base-template top strip make RAM/CPU/lease observable.

**Tech Stack:** Python 3.12, `uv`, FastAPI + Jinja/HTMX, SQLite (WAL), `transformers`/`torch` (SigLIP in-process), Ollama over the OpenAI-compat HTTP client, `psutil`.

**Spec:** `docs/design.md` §3.1, §5, §8 + **§8.1** (the coordinator — canonical), §10, §11, §13.

## Global Constraints

- Python `>=3.12`; dependencies via `uv` (`uv run`, `uv add`). Tests in `tests/`, run with `uv run pytest`.
- **`docs/design.md` is the source of truth** — already updated for this plan (§8.1). Do not let code diverge from it; if an interface here must change, change §8.1 in the same commit.
- **Profile-agnostic:** the coordinator runs identically on mac/jetson/cloud; only `ram_budget_mb` differs per profile (`config.py::PROFILE_DEFAULTS`).
- **Source folders are sacred / worker owns writes** (CLAUDE.md): this change adds no disk writes; it only gates model loads and adds one DB table.
- Every new module needs tests (CLAUDE.md). TDD: failing test first for each behavioural unit.
- SQLite is shared by `app` and `worker`; the lease is the ONLY cross-process channel — no new service, socket, or file.

---

### Task 1: `model_lease` table + `LeaseStore`

The cross-process primitive. A single-row-per-holder lease with priority + a `preempt_requested` flag, in the shared DB.

**Files:**
- Modify: `db/schema.sql` (append the `model_lease` table)
- Create: `models/__init__.py` (empty)
- Create: `models/lease_store.py`
- Test: `tests/test_lease_store.py`

**Interfaces:**
- Produces:
  - `WorkloadName = Literal["CHAT", "MEMORY_REBUILD", "INGEST_EMBED", "INGEST_CAPTION"]`
  - `INTERACTIVE: frozenset[WorkloadName]` = `{"CHAT", "MEMORY_REBUILD"}`
  - `class Lease(TypedDict): id:int; holder:str; workload:str; priority:int; heartbeat:str; preempt_requested:int`
  - `read_lease(conn) -> Lease | None`
  - `try_acquire(conn, holder:str, workload:WorkloadName, priority:int) -> bool` — inserts the single lease row iff none held; returns success.
  - `release(conn, holder:str) -> None`
  - `request_preempt(conn) -> None` — sets `preempt_requested=1` on the held row.
  - `preempt_requested(conn) -> bool`
  - `heartbeat(conn, holder:str) -> None` — bumps `heartbeat` to `CURRENT_TIMESTAMP`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_lease_store.py
import sqlite3
import pathlib

from models import lease_store as ls


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    schema = pathlib.Path("db/schema.sql").read_text()
    conn.executescript(schema)
    return conn


def test_single_holder_excludes_others():
    conn = _conn()
    assert ls.try_acquire(conn, holder="app", workload="CHAT", priority=10) is True
    # a second holder cannot acquire while one is held
    assert ls.try_acquire(conn, holder="worker", workload="INGEST_EMBED", priority=1) is False
    lease = ls.read_lease(conn)
    assert lease["holder"] == "app" and lease["workload"] == "CHAT"


def test_release_frees_the_lease():
    conn = _conn()
    ls.try_acquire(conn, holder="worker", workload="INGEST_CAPTION", priority=1)
    ls.release(conn, holder="worker")
    assert ls.read_lease(conn) is None
    assert ls.try_acquire(conn, holder="app", workload="CHAT", priority=10) is True


def test_preempt_flag_roundtrip():
    conn = _conn()
    ls.try_acquire(conn, holder="worker", workload="INGEST_EMBED", priority=1)
    assert ls.preempt_requested(conn) is False
    ls.request_preempt(conn)
    assert ls.preempt_requested(conn) is True
    # releasing clears the row (and its flag) so the next holder starts clean
    ls.release(conn, holder="worker")
    assert ls.preempt_requested(conn) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_lease_store.py -v`
Expected: FAIL — `ModuleNotFoundError: models.lease_store` / no `model_lease` table.

- [ ] **Step 3: Add the schema table**

Append to `db/schema.sql`:

```sql
-- One model lease at a time across the app + worker processes (design §8.1).
-- A single held row (id is always 1); absence of the row means "idle".
CREATE TABLE IF NOT EXISTS model_lease (
    id                 INTEGER PRIMARY KEY CHECK (id = 1),
    holder             TEXT    NOT NULL,   -- process tag: 'app' | 'worker'
    workload           TEXT    NOT NULL,
    priority           INTEGER NOT NULL,   -- higher wins; interactive > background
    heartbeat          TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    preempt_requested  INTEGER NOT NULL DEFAULT 0
);
```

- [ ] **Step 4: Implement `models/lease_store.py`**

```python
# models/lease_store.py
"""The cross-process model lease (design §8.1). A single row (id=1) means held;
no row means idle. app and worker are separate processes; this shared-DB row is
their only coordination channel."""
import sqlite3
from typing import Literal, TypedDict

WorkloadName = Literal["CHAT", "MEMORY_REBUILD", "INGEST_EMBED", "INGEST_CAPTION"]
INTERACTIVE: frozenset[str] = frozenset({"CHAT", "MEMORY_REBUILD"})


class Lease(TypedDict):
    id: int
    holder: str
    workload: str
    priority: int
    heartbeat: str
    preempt_requested: int


def read_lease(conn: sqlite3.Connection) -> Lease | None:
    row = conn.execute("SELECT * FROM model_lease WHERE id = 1").fetchone()
    return dict(row) if row is not None else None  # type: ignore[return-value]


def try_acquire(conn: sqlite3.Connection, holder: str, workload: WorkloadName, priority: int) -> bool:
    """Insert the single lease row iff none is held. Returns whether we now hold it."""
    with conn:  # atomic: INSERT OR IGNORE on the id=1 PK
        cur = conn.execute(
            "INSERT OR IGNORE INTO model_lease (id, holder, workload, priority, preempt_requested)"
            " VALUES (1, ?, ?, ?, 0)",
            (holder, workload, priority),
        )
    return cur.rowcount == 1


def release(conn: sqlite3.Connection, holder: str) -> None:
    with conn:
        conn.execute("DELETE FROM model_lease WHERE id = 1 AND holder = ?", (holder,))


def request_preempt(conn: sqlite3.Connection) -> None:
    with conn:
        conn.execute("UPDATE model_lease SET preempt_requested = 1 WHERE id = 1")


def preempt_requested(conn: sqlite3.Connection) -> bool:
    row = conn.execute("SELECT preempt_requested FROM model_lease WHERE id = 1").fetchone()
    return bool(row["preempt_requested"]) if row is not None else False


def heartbeat(conn: sqlite3.Connection, holder: str) -> None:
    with conn:
        conn.execute(
            "UPDATE model_lease SET heartbeat = CURRENT_TIMESTAMP WHERE id = 1 AND holder = ?",
            (holder,),
        )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_lease_store.py -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add db/schema.sql models/__init__.py models/lease_store.py tests/test_lease_store.py
git commit -m "feat(models): model_lease table + LeaseStore (design §8.1)"
```

---

### Task 2: Workload declarations + RAM budget guard

Pure config: workload→model-set, per-model footprint, per-profile budget, and a `fits()` guard. No residency side effects yet.

**Files:**
- Modify: `config.py` (add `ram_budget_mb` to `PROFILE_DEFAULTS` + a `Settings.ram_budget_mb` field)
- Create: `models/workloads.py`
- Test: `tests/test_workloads.py`

**Interfaces:**
- Consumes: `WorkloadName` from `models.lease_store`.
- Produces:
  - `PRIORITY: dict[WorkloadName, int]` (interactive=10, background=1)
  - `def model_set(workload: WorkloadName, *, planner_model: str, caption_model: str) -> frozenset[str]` — returns the concrete model tags the workload needs. SigLIP is the sentinel string `"siglip"`.
  - `FOOTPRINT_MB: dict[str, int]` — resident-MB estimate per model tag (`"siglip"` + LLM tags); initial estimates, tuned via the resource bar.
  - `def footprint_mb(models: frozenset[str]) -> int`
  - `def fits(models: frozenset[str], budget_mb: int) -> bool`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_workloads.py
from models import workloads as w


def test_chat_needs_siglip_and_planner():
    s = w.model_set("CHAT", planner_model="qwen2.5:3b", caption_model="qwen2.5vl:3b")
    assert s == frozenset({"siglip", "qwen2.5:3b"})


def test_ingest_caption_needs_only_the_vision_llm():
    s = w.model_set("INGEST_CAPTION", planner_model="qwen2.5:3b", caption_model="qwen2.5vl:3b")
    assert s == frozenset({"qwen2.5vl:3b"})


def test_interactive_outranks_background():
    assert w.PRIORITY["CHAT"] > w.PRIORITY["INGEST_EMBED"]


def test_fits_rejects_over_budget():
    big = frozenset({"siglip", "qwen2.5:3b"})
    assert w.fits(big, budget_mb=100) is False           # tiny budget → refuse
    assert w.fits(big, budget_mb=99_999) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_workloads.py -v`
Expected: FAIL — `ModuleNotFoundError: models.workloads`.

- [ ] **Step 3: Implement `models/workloads.py`**

```python
# models/workloads.py
"""Workload → model-set declaration + RAM budget guard (design §8.1). Adding a
workload is a table entry here, not new load/unload logic."""
from models.lease_store import WorkloadName

SIGLIP = "siglip"  # sentinel for the in-process SigLIP model

PRIORITY: dict[WorkloadName, int] = {
    "CHAT": 10, "MEMORY_REBUILD": 10, "INGEST_EMBED": 1, "INGEST_CAPTION": 1,
}

# Resident-MB estimates. Initial values; tune against the resource bar (§13).
# SigLIP so400m ~1.6 GB on GPU; qwen2.5:3b ~2.2 GB (Q4); qwen2.5vl:3b ~3.3 GB.
FOOTPRINT_MB: dict[str, int] = {
    SIGLIP: 1600, "qwen2.5:3b": 2200, "qwen2.5vl:3b": 3300, "qwen2.5vl:7b": 6000,
}
_FALLBACK_LLM_MB = 3000  # unknown LLM tag → conservative estimate


def model_set(workload: WorkloadName, *, planner_model: str, caption_model: str) -> frozenset[str]:
    if workload in ("CHAT", "MEMORY_REBUILD"):
        return frozenset({SIGLIP, planner_model})
    if workload == "INGEST_EMBED":
        return frozenset({SIGLIP})
    if workload == "INGEST_CAPTION":
        return frozenset({caption_model})
    raise ValueError(f"unknown workload: {workload}")


def footprint_mb(models: frozenset[str]) -> int:
    return sum(FOOTPRINT_MB.get(m, _FALLBACK_LLM_MB) for m in models)


def fits(models: frozenset[str], budget_mb: int) -> bool:
    return footprint_mb(models) <= budget_mb
```

- [ ] **Step 4: Add the budget to config**

In `config.py::PROFILE_DEFAULTS`, add `"ram_budget_mb"` to each profile:
- `mac`: `24000`, `jetson`: `6000`, `cloud`: `60000`.

Add the field to `Settings` (after `embed_model_name`):

```python
    # Usable RAM budget for the model coordinator (design §8.1). A workload whose
    # model-set exceeds this is refused, not loaded. Per-profile default below.
    ram_budget_mb: int = 6000
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_workloads.py -v`
Expected: PASS (4 tests).

- [ ] **Step 6: Commit**

```bash
git add config.py models/workloads.py tests/test_workloads.py
git commit -m "feat(models): workload model-sets + per-profile RAM budget guard (§8.1)"
```

---

### Task 3: Ollama warm/evict on the inference client

The coordinator needs to make Ollama hold exactly the right LLM. Ollama controls residency via `keep_alive`: a normal request warms a model; `keep_alive: 0` evicts it.

**Files:**
- Modify: `inference/client.py` (add `warm` + `evict` to the Protocol and `OpenAICompatClient`; the fake in `inference/fakes.py` gets no-ops)
- Modify: `inference/fakes.py` (add no-op `warm`/`evict`)
- Test: `tests/test_inference_residency.py`

**Interfaces:**
- Produces (on `InferenceClient`):
  - `def warm(self, model: str, *, timeout: float = 120.0) -> None` — a zero-token generate that loads the model resident.
  - `def evict(self, model: str, *, timeout: float = 30.0) -> None` — a `keep_alive: 0` generate that unloads it.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_inference_residency.py
import httpx

from inference.client import OpenAICompatClient


def test_evict_sends_keep_alive_zero():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["json"] = __import__("json").loads(request.content)
        return httpx.Response(200, json={"done": True})

    client = OpenAICompatClient("http://x/v1", transport=httpx.MockTransport(handler))
    client.evict("qwen2.5vl:3b")
    assert seen["url"].endswith("/api/generate")
    assert seen["json"]["model"] == "qwen2.5vl:3b"
    assert seen["json"]["keep_alive"] == 0


def test_warm_requests_the_model_resident():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["json"] = __import__("json").loads(request.content)
        return httpx.Response(200, json={"done": True})

    client = OpenAICompatClient("http://x/v1", transport=httpx.MockTransport(handler))
    client.warm("qwen2.5:3b")
    assert seen["json"]["model"] == "qwen2.5:3b"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_inference_residency.py -v`
Expected: FAIL — `AttributeError: 'OpenAICompatClient' object has no attribute 'evict'`.

- [ ] **Step 3: Implement warm/evict**

In `inference/client.py`, add to the `InferenceClient` Protocol:

```python
    def warm(self, model: str, *, timeout: float = 120.0) -> None: ...
    def evict(self, model: str, *, timeout: float = 30.0) -> None: ...
```

And to `OpenAICompatClient` (note: `/api/generate` is Ollama-native, one level above the `/v1` base — derive the host root by stripping a trailing `/v1`):

```python
    def _native_url(self, path: str) -> str:
        base = str(self._client.base_url)
        root = base[: -len("/v1")] if base.endswith("/v1") else base
        return root.rstrip("/") + path

    def warm(self, model: str, *, timeout: float = 120.0) -> None:
        # An empty prompt loads the model without generating tokens.
        self._client.post(
            self._native_url("/api/generate"),
            json={"model": model, "prompt": "", "stream": False},
            timeout=timeout,
        ).raise_for_status()

    def evict(self, model: str, *, timeout: float = 30.0) -> None:
        # keep_alive: 0 unloads the model immediately (Ollama-native semantics).
        self._client.post(
            self._native_url("/api/generate"),
            json={"model": model, "prompt": "", "keep_alive": 0, "stream": False},
            timeout=timeout,
        ).raise_for_status()
```

Add no-op `warm`/`evict` to the fake in `inference/fakes.py`:

```python
    def warm(self, model, *, timeout=120.0): return None
    def evict(self, model, *, timeout=30.0): return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_inference_residency.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add inference/client.py inference/fakes.py tests/test_inference_residency.py
git commit -m "feat(inference): warm/evict for Ollama model residency (§8.1)"
```

---

### Task 4: `ModelCoordinator.require()` — lease + residency + preempt-wait

The single decision point. A context manager that guards the budget, acquires the lease (preempting a lower-priority holder), reconciles residency on enter, and releases on exit.

**Files:**
- Create: `models/coordinator.py`
- Test: `tests/test_model_coordinator.py`

**Interfaces:**
- Consumes: `models.lease_store` (Task 1), `models.workloads` (Task 2), `inference.client` warm/evict (Task 3), `embedding.siglip.release_siglip_embedder` (exists).
- Produces:
  - `class RefusedError(RuntimeError)` — raised when a set exceeds the budget.
  - `class ModelCoordinator` with:
    - `__init__(self, conn, client, *, holder: str, budget_mb: int, planner_model: str, caption_model: str, load_siglip=..., release_siglip=..., sleep=time.sleep, now=time.monotonic)`
    - `require(self, workload: WorkloadName) -> contextmanager` — acquire→reconcile→(yield)→release.
    - `resident: frozenset[str]` — currently-loaded set (this process's view).
- Notes: SigLIP load/release are injected so tests avoid torch. `load_siglip`/`release_siglip` default to real functions.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_model_coordinator.py
import sqlite3
import pathlib
import pytest

from models import lease_store as ls
from models.coordinator import ModelCoordinator, RefusedError


class FakeClient:
    def __init__(self): self.warmed, self.evicted = [], []
    def warm(self, model, *, timeout=120.0): self.warmed.append(model)
    def evict(self, model, *, timeout=30.0): self.evicted.append(model)


def _conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(pathlib.Path("db/schema.sql").read_text())
    return conn


def _coord(conn, client, holder="app", budget=99_999, loaded=None):
    return ModelCoordinator(
        conn, client, holder=holder, budget_mb=budget,
        planner_model="qwen2.5:3b", caption_model="qwen2.5vl:3b",
        load_siglip=lambda: loaded.append("siglip") if loaded is not None else None,
        release_siglip=lambda: loaded.append("release") if loaded is not None else None,
        sleep=lambda s: None,
    )


def test_require_loads_declared_set_and_evicts_the_rest():
    conn, client, loaded = _conn(), FakeClient(), []
    coord = _coord(conn, client, loaded=loaded)
    with coord.require("CHAT"):
        assert "siglip" in loaded          # SigLIP loaded in-process
        assert client.warmed == ["qwen2.5:3b"]
        assert ls.read_lease(conn)["workload"] == "CHAT"
    assert ls.read_lease(conn) is None      # released on exit


def test_over_budget_is_refused_not_loaded():
    conn, client = _conn(), FakeClient()
    coord = _coord(conn, client, budget=100)   # nothing fits
    with pytest.raises(RefusedError):
        with coord.require("CHAT"):
            pass
    assert client.warmed == []
    assert ls.read_lease(conn) is None


def test_interactive_preempts_a_background_holder():
    conn, client = _conn(), FakeClient()
    # worker holds a background lease
    ls.try_acquire(conn, holder="worker", workload="INGEST_CAPTION", priority=1)
    coord = _coord(conn, client, holder="app")

    # app's CHAT request should flag preempt, then (once worker releases) acquire.
    calls = {"n": 0}
    def fake_sleep(_):
        calls["n"] += 1
        if calls["n"] == 1:
            assert ls.preempt_requested(conn) is True   # flag was set
            ls.release(conn, holder="worker")           # simulate worker yielding
    coord._sleep = fake_sleep

    with coord.require("CHAT"):
        assert ls.read_lease(conn)["holder"] == "app"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_model_coordinator.py -v`
Expected: FAIL — `ModuleNotFoundError: models.coordinator`.

- [ ] **Step 3: Implement `models/coordinator.py`**

```python
# models/coordinator.py
"""The single model-residency decision point (design §8.1). Callers declare a
workload; the coordinator guards the RAM budget, takes the cross-process lease,
loads exactly the declared model-set (evicting the rest), and releases on exit."""
import contextlib
import time

from embedding.siglip import get_siglip_embedder, release_siglip_embedder
from models import lease_store as ls
from models import workloads as wl
from models.lease_store import INTERACTIVE, WorkloadName

_PREEMPT_POLL_S = 0.25
_PREEMPT_TIMEOUT_S = 30.0


class RefusedError(RuntimeError):
    """A workload's model-set does not fit the RAM budget (design §8.1)."""


class ModelCoordinator:
    def __init__(self, conn, client, *, holder, budget_mb, planner_model, caption_model,
                 load_siglip=None, release_siglip=None, sleep=time.sleep, now=time.monotonic):
        self._conn = conn
        self._client = client
        self._holder = holder
        self._budget_mb = budget_mb
        self._planner_model = planner_model
        self._caption_model = caption_model
        self._load_siglip = load_siglip
        self._release_siglip = release_siglip or release_siglip_embedder
        self._sleep = sleep
        self._now = now
        self.resident: frozenset[str] = frozenset()

    def _do_load_siglip(self) -> None:
        if self._load_siglip is not None:
            self._load_siglip()
        else:
            get_siglip_embedder("siglip2-so400m-patch14-384", "cuda")  # real load; device via settings in caller

    @contextlib.contextmanager
    def require(self, workload: WorkloadName):
        want = wl.model_set(workload, planner_model=self._planner_model, caption_model=self._caption_model)
        if not wl.fits(want, self._budget_mb):
            raise RefusedError(
                f"{workload} needs {wl.footprint_mb(want)}MB > budget {self._budget_mb}MB: {sorted(want)}"
            )
        self._acquire(workload)
        try:
            self._reconcile(want)
            yield
        finally:
            self._release(want)

    def _acquire(self, workload: WorkloadName) -> None:
        priority = wl.PRIORITY[workload]
        if ls.try_acquire(self._conn, self._holder, workload, priority):
            return
        # Someone holds it. If we outrank them, ask them to yield and wait.
        deadline = self._now() + _PREEMPT_TIMEOUT_S
        if workload in INTERACTIVE:
            ls.request_preempt(self._conn)
        while self._now() < deadline:
            self._sleep(_PREEMPT_POLL_S)
            if ls.try_acquire(self._conn, self._holder, workload, priority):
                return
            if workload in INTERACTIVE:
                ls.request_preempt(self._conn)
        raise TimeoutError(f"could not acquire model lease for {workload} within {_PREEMPT_TIMEOUT_S}s")

    def _reconcile(self, want: frozenset[str]) -> None:
        # Evict LLMs we no longer want; release SigLIP if it's leaving.
        for model in self.resident - want:
            if model == wl.SIGLIP:
                self._release_siglip()
            else:
                self._client.evict(model)
        # Load LLMs we now want; load SigLIP if entering.
        for model in want - self.resident:
            if model == wl.SIGLIP:
                self._do_load_siglip()
            else:
                self._client.warm(model)
        self.resident = want

    def _release(self, want: frozenset[str]) -> None:
        for model in want:
            if model == wl.SIGLIP:
                self._release_siglip()
            else:
                self._client.evict(model)
        self.resident = frozenset()
        ls.release(self._conn, self._holder)
```

> **Note on `_sleep`:** the test overrides `coord._sleep`; keep `self._sleep` as the attribute name (not a closure) so it is patchable.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_model_coordinator.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add models/coordinator.py tests/test_model_coordinator.py
git commit -m "feat(models): ModelCoordinator.require — lease + residency + preempt (§8.1)"
```

---

### Task 5: Run chat + memory rebuild under `require()`

Wire the coordinator into the app. `chat_stream` currently builds SigLIP eagerly at the top (`web/app.py:819`) — the crashing line — and runs even for count questions that need no embedder. Move all model use inside `require(CHAT)`; the memory rebuild thread inside `require(MEMORY_REBUILD)`.

**Files:**
- Modify: `web/deps.py` (build one `ModelCoordinator` on the app/worker context — add a `coordinator` attribute)
- Modify: `web/app.py` (`chat_stream`: wrap retrieval+stream in `with ctx.coordinator.require("CHAT")`; drop the eager `build_embedder()` at the top — build the embedder inside the block only if retrieval needs it)
- Modify: `albums/memory_store.py` (rebuild path wrapped in `require("MEMORY_REBUILD")`)
- Test: `tests/test_web_chat.py` (extend: a count question answers with the fake embedder never crashing; assert lease released after)

**Interfaces:**
- Consumes: `ModelCoordinator` (Task 4).
- Produces: `ctx.coordinator: ModelCoordinator` available to routes; `holder="app"` in `app`, `holder="worker"` in the worker.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_web_chat.py  (add)
def test_count_question_answers_and_releases_lease(client_and_ctx):
    client, ctx = client_and_ctx                       # existing fixture (fake embedder + fake inference)
    resp = client.get("/chat/stream", params={"q": "how many images in my library?"})
    body = resp.text
    assert "data:" in body                              # at least one delta streamed — not "(no answer)"
    from models import lease_store as ls
    assert ls.read_lease(ctx.conn) is None              # CHAT lease released after the turn
```

> If `client_and_ctx` does not exist, adapt to the file's existing fixture that builds the app with fakes; the assertion is: the stream yields ≥1 `data:` event and the lease is released.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_web_chat.py -k count_question -v`
Expected: FAIL — either no `coordinator` on ctx, or the stream path unchanged.

- [ ] **Step 3: Add the coordinator to the context**

In `web/deps.py`, where the context/settings are assembled, construct and attach:

```python
from models.coordinator import ModelCoordinator

# holder is 'app' in the web process, 'worker' in the ingest process; pass it in
# from the caller (build_context(..., holder="app")).
coordinator = ModelCoordinator(
    conn, client, holder=holder,
    budget_mb=settings.ram_budget_mb,
    planner_model=settings.planner_model or "fake",
    caption_model=settings.caption_model or "fake",
    load_siglip=lambda: settings.build_embedder(),   # real SigLIP load on the configured device
    sleep=__import__("time").sleep,
)
```

Attach it to the context object returned by `build_context` (`ctx.coordinator = coordinator`). Thread a `holder: str = "app"` parameter through `build_context`.

- [ ] **Step 4: Wrap `chat_stream`**

In `web/app.py::chat_stream`, remove the eager top-of-handler `embedder, _ = ctx.settings.build_embedder()` (line ~819). Inside `events()`, wrap the retrieval+stream body:

```python
        def events():
            with ctx.coordinator.require("CHAT"):
                embedder, _ = ctx.settings.build_embedder()   # now inside the lease
                # ... existing gate → agent_retrieve → context → client.stream ...
```

Everything from `is_photo_question(...)` through the final `yield _done(...)` moves inside the `with`. The off-topic early-return stays inside too (it still holds the lease briefly — fine).

- [ ] **Step 5: Wrap the memory rebuild**

In `albums/memory_store.py`, the background rebuild entry (the function the `/organize/memories/rebuild` thread runs) wraps its model-using work:

```python
    with coordinator.require("MEMORY_REBUILD"):
        # existing pool → compose → reconcile → persist
```

Pass the coordinator into the rebuild function from the route (`web/app.py:419`).

- [ ] **Step 6: Run the chat tests**

Run: `uv run pytest tests/test_web_chat.py -v`
Expected: PASS (existing + new).

- [ ] **Step 7: Commit**

```bash
git add web/deps.py web/app.py albums/memory_store.py tests/test_web_chat.py
git commit -m "feat(chat,memory): run under ModelCoordinator.require; drop eager SigLIP build (§8.1,§10,§11)"
```

---

### Task 6: Ingest under `require()` + hard preemption

`drain_pass` runs embed/taxonomy under `INGEST_EMBED` and caption under `INGEST_CAPTION`. The `drain` loop checks a `should_preempt` callback between photos; on preempt it requeues the in-flight job and raises `Preempted`, which `drain_pass` catches to release + return. The worker loop passes the callback and backs off while an interactive lease waits.

**Files:**
- Modify: `ingest/worker.py` (`drain` accepts `should_preempt`; add `Preempted` exception + requeue-on-preempt)
- Modify: `ingest/pipeline.py` (`drain_pass` takes the coordinator; wraps groups in `require`; passes `should_preempt`)
- Modify: `ingest/cli.py` (build worker-holder coordinator; `should_preempt = lambda: preempt_requested(conn)`)
- Modify: `ingest/jobs.py` if needed (a `requeue(conn, photo_id, stage)` helper → set job back to `pending`)
- Test: `tests/test_ingest_preempt.py`

**Interfaces:**
- Consumes: `models.lease_store.preempt_requested`, `ModelCoordinator` (Task 4).
- Produces:
  - `class Preempted(Exception)` in `ingest/worker.py`
  - `drain(conn, handlers, stages=STAGES, *, should_preempt=lambda: False)` — checks the callback before claiming each photo; on True, requeues nothing in-flight (it claims one at a time) and raises `Preempted`.
  - `ingest/jobs.py::requeue(conn, photo_id, stage)` — sets that job row back to `pending`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ingest_preempt.py
import sqlite3, pathlib
from ingest.worker import drain, Preempted
from ingest import jobs


def _conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(pathlib.Path("db/schema.sql").read_text())
    return conn


def test_drain_stops_and_requeues_on_preempt():
    conn = _conn()
    # seed two photos with a pending 'embed' job each (use existing jobs helpers)
    ids = jobs.seed_two_pending_embed(conn)   # test helper; or insert rows directly per schema
    handled = []

    def handler(conn, photo_id):
        handled.append(photo_id)

    # preempt fires immediately → nothing handled, both jobs still pending
    try:
        drain(conn, {"embed": handler}, should_preempt=lambda: True)
        assert False, "expected Preempted"
    except Preempted:
        pass
    assert handled == []
    assert jobs.stage_counts(conn, "embed")["pending"] == 2
```

> If a `seed_two_pending_embed` helper is inconvenient, insert two `photos` rows + two `jobs` rows (`stage='embed'`, `status='pending'`) directly per `db/schema.sql`. The assertion is: with `should_preempt` true, `drain` handles nothing, raises `Preempted`, and leaves both jobs `pending`.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ingest_preempt.py -v`
Expected: FAIL — `ImportError: cannot import name 'Preempted'`.

- [ ] **Step 3: Add preemption to `drain`**

In `ingest/worker.py`:

```python
class Preempted(Exception):
    """An interactive workload asked ingest to yield the models (design §8.1)."""


def drain(conn, handlers, stages=STAGES, *, should_preempt=lambda: False):
    completed: dict[str, int] = {}
    for stage in stages:
        handler = handlers.get(stage)
        if handler is None:
            continue
        done = 0
        attempted: set[int] = set()
        while True:
            if should_preempt():
                raise Preempted()                      # yield BEFORE claiming another photo
            photo_id = claim_next(conn, stage, exclude=attempted)
            if photo_id is None:
                break
            attempted.add(photo_id)
            try:
                handler(conn, photo_id)
            except Exception as error:  # noqa: BLE001
                fail(conn, photo_id, stage, str(error))
            else:
                complete(conn, photo_id, stage)
                done += 1
        completed[stage] = done
    return completed
```

(Because `drain` claims one photo at a time and checks before each claim, no in-flight job needs requeuing — a claimed photo either completes or fails. The check point "before the slow caption call" is satisfied by checking before claiming each caption photo.)

- [ ] **Step 4: Wrap `drain_pass` groups in leases + pass the callback**

In `ingest/pipeline.py::drain_pass`, add a `coordinator` parameter and a `should_preempt` callback; wrap group 2a and 2b:

```python
def drain_pass(context, vocab, coordinator=None, should_preempt=lambda: False) -> None:
    ...
    # Group 1 (no models) unchanged.
    ...
    # Group 2a — SigLIP stages under INGEST_EMBED.
    try:
        with _lease(coordinator, "INGEST_EMBED"):
            embedder, model_name = settings.build_embedder()
            backfill_embeds(conn); backfill_taxonomy(conn)
            drain(conn, {"embed": embed_handler(context.originals, embedder, model_name),
                         "taxonomy": taxonomy_handler(context.derived, embedder, vocab)},
                  should_preempt=should_preempt)
    except Preempted:
        return                                          # yielded; models released by the lease exit
    except Exception:
        logger.exception("embed/taxonomy deferred this pass"); return

    backfill_captions(conn)

    # Group 2b — caption stage under INGEST_CAPTION (its own lease; SigLIP already released).
    try:
        with _lease(coordinator, "INGEST_CAPTION"):
            client, caption_model = settings.build_inference_client()
            backfill_caption_vectors(conn, client, settings.caption_embed_model)
            drain(conn, {"caption": caption_handler(context.derived, client, caption_model,
                         settings.caption_embed_model, list(vocab.dimensions), settings.thumb_detail_px)},
                  should_preempt=should_preempt)
    except Preempted:
        return
    except Exception:
        logger.exception("caption deferred this pass"); return
```

Add a small helper so `drain_pass` still works with `coordinator=None` (tests, mac inline drain that doesn't contend):

```python
import contextlib

@contextlib.contextmanager
def _lease(coordinator, workload):
    if coordinator is None:
        yield
    else:
        with coordinator.require(workload):
            yield
```

Remove the now-redundant manual `release_siglip_embedder()` block — the `INGEST_EMBED` lease exit releases SigLIP.

- [ ] **Step 5: Pass the callback from the worker loop**

In `ingest/cli.py`, build a worker-holder coordinator and pass the preempt check:

```python
from models.lease_store import preempt_requested
...
    context = build_context(get_settings(), holder="worker")
    ...
    while True:
        drain_pass(context, vocab, context.coordinator,
                   should_preempt=lambda: preempt_requested(context.conn))
        time.sleep(POLL_SECONDS)
```

- [ ] **Step 6: Add `jobs.requeue` (used by self-healing / clarity)**

In `ingest/jobs.py`:

```python
def requeue(conn: sqlite3.Connection, photo_id: int, stage: str) -> None:
    """Reset a job to pending so a later pass re-drains it (design §8.1 preemption)."""
    with conn:
        conn.execute(
            "UPDATE jobs SET status = 'pending' WHERE photo_id = ? AND stage = ?",
            (photo_id, stage),
        )
```

- [ ] **Step 7: Run the ingest tests**

Run: `uv run pytest tests/test_ingest_preempt.py tests/test_jobs.py -v`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add ingest/worker.py ingest/pipeline.py ingest/cli.py ingest/jobs.py tests/test_ingest_preempt.py
git commit -m "feat(ingest): run stages under model leases with hard preemption (§8.1)"
```

---

### Task 7: `/api/resources` + resource bar

The observability layer: a psutil snapshot + the current lease, an endpoint, and a top strip on every page.

**Files:**
- Create: `models/resources.py`
- Modify: `pyproject.toml` (add `psutil`)
- Modify: `web/app.py` (add `GET /api/resources`)
- Modify: `web/templates/base.html` (top strip + include the poller)
- Create: `web/static/resources.js`
- Modify: `web/static/app.css` (bar styles)
- Test: `tests/test_resources_api.py`

**Interfaces:**
- Consumes: `models.lease_store.read_lease`, `models.workloads.model_set/footprint_mb`.
- Produces:
  - `models/resources.py::snapshot(conn, *, planner_model, caption_model) -> dict` with keys: `ram_used_mb`, `ram_total_mb`, `cpu_pct`, `workload` (str|None), `models` (list[str]), `budget_used_mb`.
  - `GET /api/resources -> JSONResponse` of that snapshot.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_resources_api.py
def test_resources_endpoint_reports_ram_cpu_and_idle_lease(app_client):
    resp = app_client.get("/api/resources")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ram_total_mb"] > 0
    assert "cpu_pct" in data
    assert data["workload"] is None          # idle: no lease held in a fresh app


def test_resources_reflects_a_held_lease(app_client, ctx):
    from models import lease_store as ls
    ls.try_acquire(ctx.conn, holder="worker", workload="INGEST_CAPTION", priority=1)
    data = app_client.get("/api/resources").json()
    assert data["workload"] == "INGEST_CAPTION"
    assert "qwen2.5vl:3b" in data["models"] or data["models"]  # caption model listed
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_resources_api.py -v`
Expected: FAIL — no `/api/resources` route.

- [ ] **Step 3: Add psutil**

Run: `uv add psutil` (adds to `pyproject.toml` + lock).

- [ ] **Step 4: Implement `models/resources.py`**

```python
# models/resources.py
"""Live resource + lease snapshot for the resource bar (design §13, §8.1)."""
import psutil

from models import lease_store as ls
from models import workloads as wl


def snapshot(conn, *, planner_model: str, caption_model: str) -> dict:
    vm = psutil.virtual_memory()
    lease = ls.read_lease(conn)
    workload = lease["workload"] if lease else None
    models: list[str] = []
    budget_used = 0
    if workload:
        want = wl.model_set(workload, planner_model=planner_model, caption_model=caption_model)
        models = sorted(want)
        budget_used = wl.footprint_mb(want)
    return {
        "ram_used_mb": (vm.total - vm.available) // (1024 * 1024),
        "ram_total_mb": vm.total // (1024 * 1024),
        "cpu_pct": psutil.cpu_percent(interval=None),
        "workload": workload,
        "models": models,
        "budget_used_mb": budget_used,
    }
```

- [ ] **Step 5: Add the route**

In `web/app.py`:

```python
from fastapi.responses import JSONResponse
from models.resources import snapshot

    @app.get("/api/resources")
    def resources() -> JSONResponse:
        ctx = context()
        return JSONResponse(snapshot(
            ctx.conn,
            planner_model=ctx.settings.planner_model or "fake",
            caption_model=ctx.settings.caption_model or "fake",
        ))
```

- [ ] **Step 6: Add the top strip + poller**

In `web/templates/base.html`, inside `<body>` above `<nav>`:

```html
    <div id="resbar" class="resbar" aria-live="polite">…</div>
    <script src="/static/resources.js?v={{ static_v('resources.js') }}" defer></script>
```

Create `web/static/resources.js`:

```javascript
async function tick() {
  try {
    const r = await fetch("/api/resources");
    const d = await r.json();
    const gb = (mb) => (mb / 1024).toFixed(1);
    const lease = d.workload
      ? `${d.workload.toLowerCase()} · ${d.models.join("+")} · ${gb(d.budget_used_mb)} GB`
      : "idle";
    document.getElementById("resbar").textContent =
      `RAM ${gb(d.ram_used_mb)}/${gb(d.ram_total_mb)} GB · CPU ${Math.round(d.cpu_pct)}% · ${lease}`;
  } catch (_) { /* best-effort */ }
}
tick();
setInterval(tick, 2000);
```

Add to `web/static/app.css`:

```css
.resbar { position: sticky; top: 0; z-index: 50; font: 12px/1.6 ui-monospace, monospace;
  color: #aaa; background: #111; padding: 2px 10px; border-bottom: 1px solid #222; }
```

- [ ] **Step 7: Run the tests**

Run: `uv run pytest tests/test_resources_api.py -v`
Expected: PASS (2 tests).

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml uv.lock models/resources.py web/app.py web/templates/base.html web/static/resources.js web/static/app.css tests/test_resources_api.py
git commit -m "feat(ui): /api/resources + live RAM/CPU/lease resource bar (§13)"
```

---

### Task 8: Full-suite verification + Jetson deploy check

**Files:** none (verification only).

- [ ] **Step 1: Run the whole suite**

Run: `uv run pytest -q`
Expected: all green. Fix any fallout in the wired modules (`test_web_chat.py`, `test_jobs.py`, `test_memory_store.py`).

- [ ] **Step 2: Lint**

Run: `make lint` (or the project's configured linter). Expected: clean.

- [ ] **Step 3: Deploy to the Jetson and measure (the spike)**

On `lockbox-nv` (ask permission + creds per the jetson skill): `make run-jetson`, open `/chat`, ask "how many images in my library?" and a photo-content question. Watch the **resource bar** and `docker stats` / `ollama ps`:
- Confirm chat answers (no `(no answer)`).
- Confirm the captioner is **not** resident while chat holds the lease.
- Record `ram_used/total` for `CHAT` — verify SigLIP + qwen2.5:3b fit under the 6 GB budget.
- **Open verification item (from §8.1):** does SigLIP `.to('cuda')` init now that the GPU is idle under the lease? If it still throws the `NVML_SUCCESS` assert, set the jetson profile's `app` embed to CPU (config `embed_device` for the app process) and re-measure — record which, and reconcile §3.1/§8.1 if the resolution differs from the doc.

- [ ] **Step 4: Update design.md if reality differed**

If the deploy showed the model-sets don't fit 6 GB, or SigLIP must run on CPU in `app`, update `FOOTPRINT_MB`, the profile `ram_budget_mb`, and the relevant §8.1/§3.1 prose in the same commit — doc stays the source of truth.

---

## Self-Review

**Spec coverage (design §8.1 + §3.1/§5/§10/§11/§13):**
- Single decision point `require(workload)` → Task 4. ✓
- Workload→model-set table + priorities → Task 2. ✓
- RAM guard / refuse-when-over → Task 2 (`fits`) + Task 4 (`RefusedError`). ✓
- Cross-process DB lease → Task 1. ✓
- Residency reconcile (SigLIP in-process; Ollama warm/evict, MAX_LOADED_MODELS=1) → Task 3 + Task 4. (`OLLAMA_MAX_LOADED_MODELS=1` is a compose env, add to `compose.jetson.yaml` inference service in Task 3's commit or note it — **add here:** set it on the `inference` service.) ✓
- Hard preemption (check at boundaries + before caption; requeue; yield) → Task 6. ✓
- Chat + memory under lease; drop eager SigLIP crash → Task 5. ✓
- Resource bar (RAM/CPU/lease) + `/api/resources`, psutil, all profiles → Task 7. ✓
- Profile-agnostic budget → Task 2 (config). ✓

**Placeholder scan:** one intentional test-helper flexibility noted in Task 6 Step 1 (seed rows directly if no helper) — the assertion is concrete. No "TODO"/"handle edge cases".

**Type consistency:** `WorkloadName` string literals are identical across lease_store/workloads/coordinator/resources. `model_set(workload, *, planner_model, caption_model)` signature matches every call site. `require(workload)` returns a context manager everywhere. `snapshot(conn, *, planner_model, caption_model)` matches the route call.

**One correction folded in:** set `OLLAMA_MAX_LOADED_MODELS=1` on the `inference` service in `compose.jetson.yaml` (and note it belongs on any GPU profile) — the coordinator assumes Ollama holds one model at a time. Add this to Task 3's commit.
