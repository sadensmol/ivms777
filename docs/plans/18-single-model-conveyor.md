# Single Model Conveyor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the `models` service the single control plane for every model — one scheduler (semaphores + priority) plus a memory governor that loads/evicts SigLIP, nomic, and the gemma `llama-server` process against a measured RAM budget, so the 8 GB Jetson never over-commits and mac/cloud stay parallel.

**Architecture:** All model control moves *inside* the `models` service. A `Scheduler` gate serializes GPU work (concurrency = 1 on Jetson, N on mac/cloud) and orders waiters by priority (interactive chat/search ahead of batch ingest). A `MemoryGovernor` over a `ModelRegistry` ensures the models an op needs are resident, evicting others (real `free()` calls) to fit `ram_budget_mb` + measured free RAM. gemma is no longer a peer container: the models service **supervises `llama-server` as a child process** (`SubprocessLlm`) on mac/jetson — `unload` = terminate (frees ~2 GB), `load` = spawn + health-wait — and points at a remote vLLM on cloud (`RemoteLlm`). gemma stays a separate *process*, so no `llama_cpp`/heavy lib enters our Python and the one-model-process gate is untouched. Ingest becomes worker-only (the app's inline drain is removed) so exactly one process drives the pipeline.

**Tech Stack:** Python 3.12, `uv`, FastAPI (sync handlers on the anyio threadpool → threading-based gate, not asyncio), httpx, psutil, `pytest`, Docker Compose, source-built `sm_87` / Metal `llama-server`.

**Spec:** `docs/design.md` — the project's source of truth. This plan implements and rewrites **§3.1** (deploy profiles), **§4** (models), **§5** (architecture mermaid), **§5.1** (the models service), **§8.1** (residency → the conveyor). Per `CLAUDE.md`, every task that changes described behaviour edits the owning design section **in the same commit** as the code; Task 10 is the final coherence pass.

## Global Constraints

- **One model process (hard rule).** Only `modelsvc/` (+ the three `_EXEMPT_FILES` in `tests/test_one_model_process_gate.py`: `embedding/siglip.py`, `embedding/text_embedder.py`, `inference/client.py`) may import `torch`/`transformers`/`bitsandbytes` or touch `inference_base_url`. New conveyor code lives in `modelsvc/` and uses only `subprocess`/`httpx`/`psutil` — **never** import `llama_cpp` or any model lib. gemma stays a child *process*, reached over localhost HTTP. `app`/`worker`/CLI stay thin clients.
- **Profile is `IVMS777_PROFILE`** (`mac`/`jetson`/`cloud`); defaults filled by `Settings._apply_profile_defaults`. Env prefix `IVMS777_`.
- **Jetson real budget ≈ 5.8 GB** (§3.1); `ram_budget_mb` = jetson `6000`, mac `24000`, cloud `60000`. Jetson unified LPDDR5 → `psutil.virtual_memory()` reflects GPU allocations, so measured free RAM is the real signal.
- **gemma `llama-server` flags are mandatory** (§3.1/§4): `-ngl 99 --flash-attn on --jinja --chat-template-kwargs '{"enable_thinking":false}' -c 4096`. Preserve them verbatim when the models service spawns the process.
- **No git commits by the tooling** — the human commits. Steps end at a staged, tested state; the "Commit" step lists the `git add`/message for the human to run.
- **uv only:** run everything with `uv run` (`uv run pytest -q`, `uv run ruff check .`).
- **Out of scope / pre-existing:** the caption `dimensions`/`tags` mismatch (dossier §14) is NOT fixed here. Caption changes are limited to *routing captions through the scheduler*. Tests that would otherwise depend on caption schema use `FakeBackend`.

---

## File Structure

**New files (all under `modelsvc/`, the single control plane):**
- `modelsvc/registry.py` — `ModelSpec`, `ModelRegistry`: the resident-set manager with **real** `load`/`free` per model, thread-safe. Replaces `modelsvc/residency.py`.
- `modelsvc/governor.py` — `MemoryGovernor`: budget + measured-free + LRU/priority eviction over the registry.
- `modelsvc/scheduler.py` — `Scheduler`, `Priority`: the single gate (concurrency cap + priority ordering) that every model op passes through; wraps the governor.
- `modelsvc/llm_process.py` — `LlmProcess` protocol, `SubprocessLlm` (spawn/kill `llama-server`), `RemoteLlm` (cloud, no-op lifecycle).
- `tests/test_registry.py`, `tests/test_governor.py`, `tests/test_scheduler.py`, `tests/test_llm_process.py`, `tests/test_models_control_api.py`.

**Modified files:**
- `config.py` — conveyor settings (costs, concurrency, llm spawn command, idle TTL).
- `modelsvc/backends/__init__.py` — build registry+governor+scheduler+LlmProcess, inject into backends.
- `modelsvc/backends/siglip_backend.py`, `text_backend.py`, `caption_backend.py`, `composite.py` — acquire through the scheduler; report registry state.
- `modelsvc/app.py` — `GET /models`, `POST /models/{name}/ensure|unload`; `resources()` reports budget + free.
- `embedding/siglip.py`, `embedding/text_embedder.py` — expose `free`/`release` hooks the registry calls.
- `ingest/pipeline.py`, `ingest/cli.py`, `web/app.py`, `web/deps.py` — worker-only ingest; remove the dead `NoopCoordinator`/`require()`/`_lease` plumbing.
- `models/coordinator.py` — deleted (its no-op is fully replaced by the scheduler).
- `compose.yaml`, `compose.jetson.yaml`, `compose.mac.yaml`, `Dockerfile.models.jetson`, `Makefile`, `README.md` — models container owns `llama-server`; drop the standalone `inference` service.
- `docs/design.md` — §3.1, §4, §5, §5.1, §8.1.
- Tests updated: `test_residency.py` (→ `test_registry.py`), `test_modelsvc_backends.py`, `test_modelsvc_api.py`, `test_pipeline_pass.py`, `test_web_chat.py`, `test_coordinator_noop.py` (deleted), `test_config.py`.

---

## Task 1: Conveyor configuration

**Files:**
- Modify: `config.py:15-45` (`PROFILE_DEFAULTS`), `config.py:48-92` (`Settings` fields)
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: new `Settings` attrs consumed by later tasks — `gpu_concurrency: int`, `model_cost_mb: dict[str, int]`, `llm_managed: bool`, `llm_command: list[str]`, `llm_health_url: str`, `llm_idle_ttl_s: int | None`. `ram_budget_mb` stays but is now **read** by the governor (Task 3).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py — append
def test_conveyor_profile_defaults():
    from config import Settings
    jetson = Settings(profile="jetson")
    assert jetson.gpu_concurrency == 1
    assert jetson.llm_managed is True
    assert jetson.llm_idle_ttl_s == 120
    assert jetson.model_cost_mb["siglip"] == 1600
    assert jetson.model_cost_mb["gemma"] == 2200
    assert jetson.model_cost_mb["nomic"] == 300

    mac = Settings(profile="mac")
    assert mac.gpu_concurrency == 3
    assert mac.llm_managed is True
    assert mac.llm_idle_ttl_s is None          # 32 GB: never idle-unload

    cloud = Settings(profile="cloud")
    assert cloud.gpu_concurrency == 4
    assert cloud.llm_managed is False           # remote vLLM, not supervised

def test_conveyor_env_override():
    from config import Settings
    s = Settings(profile="jetson", gpu_concurrency=2)
    assert s.gpu_concurrency == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py::test_conveyor_profile_defaults -v`
Expected: FAIL — `AttributeError: 'Settings' object has no attribute 'gpu_concurrency'`.

- [ ] **Step 3: Add the fields and per-profile defaults**

In `config.py` `PROFILE_DEFAULTS`, add to each profile dict:
```python
# mac
"gpu_concurrency": 3, "llm_managed": True, "llm_idle_ttl_s": None,
# jetson
"gpu_concurrency": 1, "llm_managed": True, "llm_idle_ttl_s": 120,
# cloud
"gpu_concurrency": 4, "llm_managed": False, "llm_idle_ttl_s": None,
```

In `Settings` (after line 92) add fields (defaults `None`, filled by the existing `_apply_profile_defaults` validator which copies any `PROFILE_DEFAULTS[profile]` key whose attr is `None`):
```python
gpu_concurrency: int | None = None
llm_managed: bool | None = None
llm_idle_ttl_s: int | None = None
model_cost_mb: dict[str, int] = Field(
    default_factory=lambda: {"siglip": 1600, "nomic": 300, "gemma": 2200}
)
# how the models service launches llama-server (mac/jetson). {gguf}/{mmproj} are
# resolved from data_dir at build time in Task 5; kept as a plain list here.
llm_bin: str | None = None          # path to llama-server binary; None → "llama-server" on PATH
llm_port: int = 8080
```
Add a computed property for the health URL:
```python
@property
def llm_health_url(self) -> str:
    return f"http://localhost:{self.llm_port}/health"
```
Note: `_apply_profile_defaults` iterates `PROFILE_DEFAULTS[self.profile].items()`; `bool`/`int` values of `None` are filled, so add the three new keys there. `model_cost_mb` has a real default (not profile-specific) and is overridable via `IVMS777_MODEL_COST_MB` JSON.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS (including the existing `ram_budget_mb` assertions at `test_config.py:50-51`).

- [ ] **Step 5: Commit**

```bash
git add config.py tests/test_config.py
git commit -m "feat(config): conveyor settings — concurrency, model costs, llm supervision"
```

---

## Task 2: ModelRegistry — resident set with real load/free

**Files:**
- Create: `modelsvc/registry.py`
- Test: `tests/test_registry.py`
- (Later) Delete: `modelsvc/residency.py` + rename `tests/test_residency.py` (done in Task 6 once callers move.)

**Interfaces:**
- Produces: `ModelSpec(name: str, load: Callable[[], None], free: Callable[[], None], cost_mb: int)`; `ModelRegistry` with `register(spec)`, `ensure(name) -> None` (loads once if absent, records LRU touch), `unload(name) -> None` (calls `free`, forgets it), `resident() -> list[str]` (LRU order, oldest first), `is_resident(name) -> bool`, `cost_mb(name) -> int`, `touch(name)`. Thread-safe (`threading.RLock`). Consumed by `MemoryGovernor` (Task 3).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_registry.py
import threading
import pytest
from modelsvc.registry import ModelSpec, ModelRegistry


def _spec(name, log, cost=100):
    return ModelSpec(
        name=name,
        load=lambda: log.append(("load", name)),
        free=lambda: log.append(("free", name)),
        cost_mb=cost,
    )


def test_ensure_loads_once():
    log = []
    r = ModelRegistry()
    r.register(_spec("siglip", log))
    r.ensure("siglip")
    r.ensure("siglip")
    assert log == [("load", "siglip")]
    assert r.is_resident("siglip")


def test_unload_calls_free_and_forgets():
    log = []
    r = ModelRegistry()
    r.register(_spec("siglip", log))
    r.ensure("siglip")
    r.unload("siglip")
    assert log == [("load", "siglip"), ("free", "siglip")]
    assert not r.is_resident("siglip")


def test_resident_is_lru_oldest_first():
    log = []
    r = ModelRegistry()
    for n in ("a", "b", "c"):
        r.register(_spec(n, log))
    r.ensure("a"); r.ensure("b"); r.ensure("c")
    r.touch("a")                       # a becomes most-recent
    assert r.resident() == ["b", "c", "a"]


def test_unknown_model_raises():
    r = ModelRegistry()
    with pytest.raises(KeyError):
        r.ensure("nope")


def test_ensure_is_thread_safe():
    log = []
    r = ModelRegistry()
    r.register(_spec("siglip", log))
    barrier = threading.Barrier(8)

    def worker():
        barrier.wait()
        r.ensure("siglip")

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert log.count(("load", "siglip")) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_registry.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'modelsvc.registry'`.

- [ ] **Step 3: Write minimal implementation**

```python
# modelsvc/registry.py
"""The resident-set manager: the one place a model is loaded or freed.

Replaces the old ensure-loaded-only ``Residency``. Each model registers a real
``load``/``free`` pair and a rough resident cost; the ``MemoryGovernor`` drives
``ensure``/``unload`` against a budget. Thread-safe: the models service runs
sync handlers on the anyio threadpool, so several requests touch this at once.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class ModelSpec:
    name: str
    load: Callable[[], None]
    free: Callable[[], None]
    cost_mb: int


class ModelRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, ModelSpec] = {}
        self._loaded: "OrderedDict[str, None]" = OrderedDict()  # LRU: oldest first
        self._lock = threading.RLock()

    def register(self, spec: ModelSpec) -> None:
        with self._lock:
            self._specs[spec.name] = spec

    def ensure(self, name: str) -> None:
        with self._lock:
            spec = self._specs[name]                # KeyError if unknown
            if name in self._loaded:
                self._loaded.move_to_end(name)
                return
            spec.load()
            self._loaded[name] = None

    def unload(self, name: str) -> None:
        with self._lock:
            if name not in self._loaded:
                return
            self._specs[name].free()
            del self._loaded[name]

    def touch(self, name: str) -> None:
        with self._lock:
            if name in self._loaded:
                self._loaded.move_to_end(name)

    def is_resident(self, name: str) -> bool:
        with self._lock:
            return name in self._loaded

    def resident(self) -> list[str]:
        with self._lock:
            return list(self._loaded.keys())

    def cost_mb(self, name: str) -> int:
        with self._lock:
            return self._specs[name].cost_mb
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_registry.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add modelsvc/registry.py tests/test_registry.py
git commit -m "feat(modelsvc): ModelRegistry with real load/free and LRU resident set"
```

---

## Task 3: MemoryGovernor — budget + measured free + eviction

**Files:**
- Create: `modelsvc/governor.py`
- Test: `tests/test_governor.py`

**Interfaces:**
- Consumes: `ModelRegistry` (Task 2); a `measure_free_mb: Callable[[], float]` (injected; real one wraps `psutil.virtual_memory().available / 1e6`); `budget_mb: int`; `headroom_mb: int = 512`.
- Produces: `MemoryGovernor.acquire(needed: list[str], *, pinned: frozenset[str] = frozenset()) -> None` — makes every model in `needed` resident, evicting non-needed, non-pinned residents (LRU, oldest first) until the set fits `min(budget, measured_free + evictable)`. Raises `InsufficientMemory` if it cannot fit even after evicting everything evictable. `MemoryGovernor.state() -> GovernorState` (resident list, used_mb estimate, free_mb, budget_mb).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_governor.py
import pytest
from modelsvc.registry import ModelSpec, ModelRegistry
from modelsvc.governor import MemoryGovernor, InsufficientMemory


def _registry(log, costs):
    r = ModelRegistry()
    for name, cost in costs.items():
        r.register(ModelSpec(name, (lambda n=name: log.append(("load", n))),
                             (lambda n=name: log.append(("free", n))), cost))
    return r


def test_acquire_loads_when_it_fits():
    log = []
    r = _registry(log, {"siglip": 1600})
    gov = MemoryGovernor(r, measure_free_mb=lambda: 4000.0, budget_mb=6000)
    gov.acquire(["siglip"])
    assert ("load", "siglip") in log
    assert r.is_resident("siglip")


def test_acquire_evicts_lru_to_fit_budget():
    # budget 6000, siglip(1600)+gemma(2200)+nomic(300) resident = 4100.
    # need to load a hypothetical big(3000) -> must evict LRU until it fits.
    log = []
    r = _registry(log, {"siglip": 1600, "gemma": 2200, "nomic": 300, "big": 3000})
    gov = MemoryGovernor(r, measure_free_mb=lambda: 10000.0, budget_mb=6000, headroom_mb=0)
    for n in ("nomic", "siglip", "gemma"):     # LRU order: nomic oldest
        gov.acquire([n])
    log.clear()
    gov.acquire(["big"])
    # evicts oldest first (nomic, then siglip) until 3000 + remaining <= 6000
    assert ("free", "nomic") in log
    assert ("free", "siglip") in log
    assert ("free", "gemma") not in log        # gemma newest, kept if it now fits
    assert r.is_resident("big")


def test_acquire_never_evicts_needed_or_pinned():
    log = []
    r = _registry(log, {"siglip": 1600, "gemma": 2200})
    gov = MemoryGovernor(r, measure_free_mb=lambda: 0.0, budget_mb=3000, headroom_mb=0)
    gov.acquire(["gemma"])
    log.clear()
    # need siglip while gemma pinned; 1600+2200=3800 > 3000 but gemma pinned -> raise
    with pytest.raises(InsufficientMemory):
        gov.acquire(["siglip"], pinned=frozenset({"gemma"}))
    assert ("free", "gemma") not in log


def test_acquire_uses_measured_free_not_just_budget():
    # budget says 6000 but the box only reports 500 MB free -> must evict to fit reality.
    log = []
    r = _registry(log, {"gemma": 2200, "siglip": 1600})
    gov = MemoryGovernor(r, measure_free_mb=lambda: 500.0, budget_mb=6000, headroom_mb=0)
    gov.acquire(["gemma"])
    log.clear()
    gov.acquire(["siglip"])                    # 500 free < 1600 -> evict gemma
    assert ("free", "gemma") in log
    assert r.is_resident("siglip")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_governor.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'modelsvc.governor'`.

- [ ] **Step 3: Write minimal implementation**

```python
# modelsvc/governor.py
"""Budget-and-measurement gate over the ModelRegistry.

``acquire`` makes the requested models resident, evicting non-needed / non-pinned
residents (LRU, oldest first) until the set fits the tighter of (a) the configured
``budget_mb`` and (b) the *measured* free RAM + what eviction would give back. On
the unified-memory Jetson, measured free RAM already reflects GPU allocations, so
(b) is the real signal; ``budget_mb`` is the conservative ceiling.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable

from modelsvc.registry import ModelRegistry


class InsufficientMemory(RuntimeError):
    pass


@dataclass(frozen=True)
class GovernorState:
    resident: list[str]
    used_mb: int
    free_mb: float
    budget_mb: int


class MemoryGovernor:
    def __init__(self, registry: ModelRegistry, *, measure_free_mb: Callable[[], float],
                 budget_mb: int, headroom_mb: int = 512) -> None:
        self._r = registry
        self._measure = measure_free_mb
        self._budget = budget_mb
        self._headroom = headroom_mb
        self._lock = threading.RLock()

    def _resident_cost(self, names: list[str]) -> int:
        return sum(self._r.cost_mb(n) for n in names)

    def acquire(self, needed: list[str], *, pinned: frozenset[str] = frozenset()) -> None:
        with self._lock:
            keep = set(needed) | set(pinned)
            missing = [n for n in needed if not self._r.is_resident(n)]
            add_cost = sum(self._r.cost_mb(n) for n in missing)

            # Evict LRU (oldest first) that are neither needed nor pinned, until the
            # projected resident cost fits the budget AND measured free covers the adds.
            for name in self._r.resident():
                free_now = self._measure()
                projected = self._resident_cost(self._r.resident()) + add_cost
                fits_budget = projected + self._headroom <= self._budget
                fits_real = free_now >= add_cost + self._headroom
                if fits_budget and fits_real:
                    break
                if name in keep:
                    continue
                self._r.unload(name)

            free_now = self._measure()
            projected = self._resident_cost(self._r.resident()) + add_cost
            if projected + self._headroom > self._budget or free_now < add_cost + self._headroom:
                raise InsufficientMemory(
                    f"cannot fit {needed}: budget={self._budget} "
                    f"projected={projected} free={free_now:.0f} pinned={sorted(pinned)}"
                )

            for n in needed:
                self._r.ensure(n)               # loads missing, LRU-touches present

    def state(self) -> GovernorState:
        with self._lock:
            res = self._r.resident()
            return GovernorState(
                resident=res,
                used_mb=self._resident_cost(res),
                free_mb=self._measure(),
                budget_mb=self._budget,
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_governor.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add modelsvc/governor.py tests/test_governor.py
git commit -m "feat(modelsvc): MemoryGovernor — budget + measured-free eviction"
```

---

## Task 4: Scheduler — the single gate (concurrency + priority)

**Files:**
- Create: `modelsvc/scheduler.py`
- Test: `tests/test_scheduler.py`

**Interfaces:**
- Consumes: `MemoryGovernor` (Task 3); `concurrency: int` (from `settings.gpu_concurrency`).
- Produces: `class Priority(IntEnum): BATCH = 1; INTERACTIVE = 2`; `Scheduler.run(needed, priority, fn, *, pinned=frozenset())` — acquires a slot (semaphore sized to `concurrency`), grants slots **highest-priority first** among waiters, then `governor.acquire(needed, pinned=pinned)`, then calls `fn()` and returns its result, releasing the slot in `finally`. `Scheduler.slots` (int). Threading-based (handlers are sync `def`). Consumed by every backend op (Task 6).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scheduler.py
import threading
import time
from modelsvc.scheduler import Scheduler, Priority


class _StubGov:
    def __init__(self): self.acquired = []
    def acquire(self, needed, *, pinned=frozenset()): self.acquired.append(list(needed))


def test_run_calls_fn_through_governor():
    gov = _StubGov()
    s = Scheduler(gov, concurrency=1)
    out = s.run(["siglip"], Priority.INTERACTIVE, lambda: 42)
    assert out == 42
    assert gov.acquired == [["siglip"]]


def test_concurrency_cap_is_enforced():
    gov = _StubGov()
    s = Scheduler(gov, concurrency=1)
    live = []
    peak = [0]
    lock = threading.Lock()

    def body():
        with lock:
            live.append(1); peak[0] = max(peak[0], len(live))
        time.sleep(0.05)
        with lock:
            live.pop()

    threads = [threading.Thread(target=lambda: s.run(["m"], Priority.BATCH, body)) for _ in range(4)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert peak[0] == 1                          # never more than 1 at once


def test_interactive_preempts_batch_in_queue():
    gov = _StubGov()
    s = Scheduler(gov, concurrency=1)
    order = []
    started = threading.Event()

    def hog():
        started.set(); time.sleep(0.1); order.append("hog")

    hog_t = threading.Thread(target=lambda: s.run(["m"], Priority.BATCH, hog))
    hog_t.start(); started.wait()
    # queue a BATCH then an INTERACTIVE while the hog holds the only slot
    b = threading.Thread(target=lambda: s.run(["m"], Priority.BATCH, lambda: order.append("batch")))
    i = threading.Thread(target=lambda: s.run(["m"], Priority.INTERACTIVE, lambda: order.append("inter")))
    b.start(); time.sleep(0.01); i.start()
    for t in (hog_t, b, i): t.join()
    assert order == ["hog", "inter", "batch"]    # interactive jumps the queued batch
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_scheduler.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'modelsvc.scheduler'`.

- [ ] **Step 3: Write minimal implementation**

```python
# modelsvc/scheduler.py
"""The single entry gate for all GPU/model work in the models service.

Every op passes through ``run``: it takes one of ``concurrency`` slots (1 on
Jetson → serialized GPU; N on mac/cloud), and when slots are contended the
highest-priority waiter goes first, so an interactive chat never waits behind a
batch of ingest captions. Once admitted, it asks the governor to make the needed
models resident, then runs the op. Threading, not asyncio: the FastAPI handlers
are sync and run on the anyio threadpool.
"""
from __future__ import annotations

import heapq
import itertools
import threading
from enum import IntEnum
from typing import Callable, TypeVar

from modelsvc.governor import MemoryGovernor

T = TypeVar("T")


class Priority(IntEnum):
    BATCH = 1          # ingest
    INTERACTIVE = 2    # chat / search


class Scheduler:
    def __init__(self, governor: MemoryGovernor, *, concurrency: int) -> None:
        self._gov = governor
        self.slots = concurrency
        self._free = concurrency
        self._cv = threading.Condition()
        self._waiters: list[tuple[int, int, threading.Event]] = []  # max-heap by -priority
        self._seq = itertools.count()

    def _admit(self, priority: Priority) -> None:
        ev = threading.Event()
        with self._cv:
            if self._free > 0 and not self._waiters:
                self._free -= 1
                return
            # -priority so higher priority pops first; seq breaks ties FIFO
            heapq.heappush(self._waiters, (-int(priority), next(self._seq), ev))
        ev.wait()

    def _release(self) -> None:
        with self._cv:
            if self._waiters:
                _, _, ev = heapq.heappop(self._waiters)
                ev.set()                        # hand the slot directly to the winner
            else:
                self._free += 1

    def run(self, needed: list[str], priority: Priority, fn: Callable[[], T],
            *, pinned: frozenset[str] = frozenset()) -> T:
        self._admit(priority)
        try:
            self._gov.acquire(needed, pinned=pinned)
            return fn()
        finally:
            self._release()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_scheduler.py -v`
Expected: PASS. (If `test_interactive_preempts_batch` is timing-flaky in CI, keep the sleeps as written — they are generous — but do not weaken the assertion.)

- [ ] **Step 5: Commit**

```bash
git add modelsvc/scheduler.py tests/test_scheduler.py
git commit -m "feat(modelsvc): Scheduler gate — concurrency cap + priority admission"
```

---

## Task 5: LlmProcess — supervise llama-server (mac/jetson) / remote (cloud)

**Files:**
- Create: `modelsvc/llm_process.py`
- Test: `tests/test_llm_process.py`

**Interfaces:**
- Consumes: `settings` (Task 1: `llm_managed`, `llm_bin`, `llm_port`, `llm_health_url`, gemma flags, `data_dir`).
- Produces: `LlmProcess` protocol — `load() -> None`, `free() -> None`, `is_loaded() -> bool`. `SubprocessLlm(command, health_url, ready_timeout_s=120, probe=<httpx get>)` — `load` spawns via `subprocess.Popen` (scoped env), polls `health_url` until 200 or timeout (kills + raises on timeout); `free` `terminate()` then `kill()` after grace. `RemoteLlm(health_url)` — `load`/`free` no-ops; `is_loaded()` probes health. `build_llm_process(settings) -> LlmProcess`. gemma registers into the registry (Task 6) with `load=llm.load, free=llm.free, cost_mb=model_cost_mb["gemma"]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_llm_process.py
import sys
import time
import pytest
from modelsvc.llm_process import SubprocessLlm, RemoteLlm


def test_subprocess_loads_waits_healthy_and_frees():
    calls = {"n": 0}
    def probe(url):                              # healthy on the 2nd poll
        calls["n"] += 1
        return calls["n"] >= 2
    # a trivial child that just sleeps so terminate() has something to kill
    cmd = [sys.executable, "-c", "import time; time.sleep(30)"]
    llm = SubprocessLlm(cmd, health_url="http://x/health", ready_timeout_s=5,
                        probe=probe, poll_interval_s=0.01)
    llm.load()
    assert llm.is_loaded()
    assert calls["n"] >= 2
    llm.free()
    assert not llm.is_loaded()


def test_subprocess_raises_and_cleans_up_on_unhealthy():
    cmd = [sys.executable, "-c", "import time; time.sleep(30)"]
    llm = SubprocessLlm(cmd, health_url="http://x/health", ready_timeout_s=0.05,
                        probe=lambda url: False, poll_interval_s=0.01)
    with pytest.raises(TimeoutError):
        llm.load()
    assert not llm.is_loaded()                   # child was killed


def test_remote_llm_lifecycle_is_noop():
    llm = RemoteLlm(health_url="http://x/health", probe=lambda url: True)
    llm.load(); llm.free()                       # no raise, no process
    assert llm.is_loaded() is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_llm_process.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'modelsvc.llm_process'`.

- [ ] **Step 3: Write minimal implementation**

```python
# modelsvc/llm_process.py
"""gemma's lifecycle, owned by the models service.

mac/jetson: ``SubprocessLlm`` spawns the local ``llama-server`` as a child process
(Metal binary on mac, sm_87 CUDA binary on jetson) and kills it to free ~2 GB —
this is how the governor "unloads" gemma. cloud: ``RemoteLlm`` points at a remote
vLLM and never supervises it. No ``llama_cpp`` import — gemma stays a separate
process reached over localhost HTTP, so the one-model-process gate is untouched.
"""
from __future__ import annotations

import os
import subprocess
import time
from typing import Callable, Protocol

import httpx


def _http_probe(url: str) -> bool:
    try:
        return httpx.get(url, timeout=1.0).status_code == 200
    except Exception:
        return False


class LlmProcess(Protocol):
    def load(self) -> None: ...
    def free(self) -> None: ...
    def is_loaded(self) -> bool: ...


class SubprocessLlm:
    def __init__(self, command: list[str], *, health_url: str, ready_timeout_s: float = 120.0,
                 poll_interval_s: float = 0.5, env: dict[str, str] | None = None,
                 probe: Callable[[str], bool] = _http_probe) -> None:
        self._cmd = command
        self._health = health_url
        self._timeout = ready_timeout_s
        self._poll = poll_interval_s
        self._env = env
        self._probe = probe
        self._proc: subprocess.Popen | None = None

    def load(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        # scoped env so llama-server's CUDA libs never clash with torch's in-process
        run_env = {**os.environ, **(self._env or {})}
        self._proc = subprocess.Popen(self._cmd, env=run_env)
        deadline = time.monotonic() + self._timeout
        while time.monotonic() < deadline:
            if self._probe(self._health):
                return
            if self._proc.poll() is not None:
                raise RuntimeError(f"llama-server exited early: rc={self._proc.returncode}")
            time.sleep(self._poll)
        self.free()
        raise TimeoutError(f"llama-server not healthy within {self._timeout}s")

    def free(self) -> None:
        p, self._proc = self._proc, None
        if p is None:
            return
        p.terminate()
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait(timeout=5)

    def is_loaded(self) -> bool:
        return self._proc is not None and self._proc.poll() is None


class RemoteLlm:
    def __init__(self, *, health_url: str, probe: Callable[[str], bool] = _http_probe) -> None:
        self._health = health_url
        self._probe = probe

    def load(self) -> None:  # remote vLLM is always up; nothing to supervise
        return None

    def free(self) -> None:
        return None

    def is_loaded(self) -> bool:
        return True
```

Also add `build_llm_process(settings)` (used by Task 6):
```python
def build_llm_process(settings) -> LlmProcess:
    if not settings.llm_managed:
        return RemoteLlm(health_url=settings.llm_health_url)
    gguf = settings.data_dir / "models" / "gemma-4-E2B-it-Q4_K_M.gguf"
    mmproj = settings.data_dir / "models" / "mmproj-F16.gguf"
    cmd = [
        settings.llm_bin or "llama-server",
        "-m", str(gguf), "--mmproj", str(mmproj),
        "-ngl", "99", "--flash-attn", "on", "--jinja",
        "--chat-template-kwargs", '{"enable_thinking":false}',
        "-c", "4096", "--host", "0.0.0.0", "--port", str(settings.llm_port),
    ]
    return SubprocessLlm(cmd, health_url=settings.llm_health_url)
```
(GGUF paths mirror `scripts/llama-server-entrypoint.sh` and Makefile `LLAMA_GGUF`/`LLAMA_MMPROJ`. On mac/jetson the entrypoint/Makefile still fetch the GGUFs into `data_dir/models`; Task 9 wires the volume + `llm_bin`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_llm_process.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add modelsvc/llm_process.py tests/test_llm_process.py
git commit -m "feat(modelsvc): LlmProcess — supervise llama-server (mac/jetson) / remote (cloud)"
```

---

## Task 6: Wire backends through the conveyor; retire Residency

**Files:**
- Modify: `modelsvc/backends/__init__.py:19-50` (`build_backend`), `modelsvc/backends/siglip_backend.py`, `modelsvc/backends/text_backend.py`, `modelsvc/backends/composite.py`
- Modify: `embedding/siglip.py` (expose `release`/`free` hook), `embedding/text_embedder.py` (expose a `free` hook)
- Delete: `modelsvc/residency.py`; Modify/rename `tests/test_residency.py` → `tests/test_registry.py` already covers it — delete `test_residency.py`.
- Test: `tests/test_modelsvc_backends.py` (update)

**Interfaces:**
- Consumes: `ModelRegistry`, `MemoryGovernor`, `Scheduler`, `Priority`, `build_llm_process` (Tasks 2–5); `settings.gpu_concurrency`, `settings.ram_budget_mb`, `settings.model_cost_mb`.
- Produces: `build_backend` constructs one `ModelRegistry` with **three** specs registered — `siglip` (load=`get_siglip_embedder`, free=`release_siglip_embedder`), `nomic` (load/free the text embedder), `gemma` (load/free = `llm.load`/`llm.free`) — a `MemoryGovernor` (measure = psutil available MB), a `Scheduler(gov, concurrency=settings.gpu_concurrency)`, injected into `CompositeBackend`. Backends call `scheduler.run(needed, priority, fn)`. `CompositeBackend.resources()` reports `registry.resident()` + governor `budget_mb`/`free_mb`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_modelsvc_backends.py — replace the residency-specific assertions
def test_build_backend_wires_conveyor():
    from config import Settings
    from modelsvc.backends import build_backend
    from modelsvc.scheduler import Scheduler
    b = build_backend(Settings(profile="jetson"))
    assert isinstance(b._scheduler, Scheduler)
    assert b._scheduler.slots == 1                       # jetson concurrency
    # all three models registered
    names = set(b._registry._specs)
    assert {"siglip", "nomic", "gemma"} <= names


def test_embed_image_runs_through_scheduler(monkeypatch):
    from config import Settings
    from modelsvc.backends import build_backend
    b = build_backend(Settings(profile="mac", use_fake_embedder=False))
    seen = {}
    real = b._scheduler.run
    def spy(needed, priority, fn, **kw):
        seen["needed"] = needed
        return real(needed, priority, fn, **kw)
    monkeypatch.setattr(b._scheduler, "run", spy)
    # siglip load is stubbed so no torch: register a fake spec
    b._registry._specs["siglip"] = b._registry._specs["siglip"].__class__(
        "siglip", lambda: None, lambda: None, 1600)
    b.embed_image([])                                    # empty batch: no real inference
    assert seen["needed"] == ["siglip"]
```

Update the existing `test_modelsvc_backends.py` assertions that referenced `residency`/`resident()==["siglip"]`/`HIGH` to the registry equivalents (`b._registry.resident()`), and keep the "caption + text share ONE inference client" assertion (`backend._caption._captioner._client is backend._text._client`) — that wiring is unchanged.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_modelsvc_backends.py -v`
Expected: FAIL — `AttributeError: 'CompositeBackend' object has no attribute '_scheduler'`.

- [ ] **Step 3: Implement the wiring**

`embedding/siglip.py` — confirm `release_siglip_embedder()` exists (dossier: it does, lines 64-79) and is import-safe; no change needed beyond ensuring it clears the `lru_cache`. `embedding/text_embedder.py` — add a `release_text_embedder()` mirroring it:
```python
def release_text_embedder() -> None:
    get_text_embedder.cache_clear()
    import gc; gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache(); torch.cuda.ipc_collect()
    except Exception:
        pass
```
(`embedding/text_embedder.py` is already an `_EXEMPT_FILES` member of the gate, so importing torch here is allowed.)

`modelsvc/backends/__init__.py::build_backend` — replace the `Residency` block:
```python
def build_backend(settings) -> ModelBackend:
    if settings.use_fake_embedder:
        return FakeBackend()
    import psutil
    from inference.client import OpenAICompatClient
    from modelsvc.registry import ModelSpec, ModelRegistry
    from modelsvc.governor import MemoryGovernor
    from modelsvc.scheduler import Scheduler
    from modelsvc.llm_process import build_llm_process
    from embedding.siglip import get_siglip_embedder, release_siglip_embedder
    from embedding.text_embedder import get_text_embedder, release_text_embedder

    inf = OpenAICompatClient(settings.inference_base_url or "")
    llm = build_llm_process(settings)

    registry = ModelRegistry()
    registry.register(ModelSpec(
        "siglip",
        load=lambda: get_siglip_embedder(settings.embed_model_name, settings.embed_device),
        free=release_siglip_embedder, cost_mb=settings.model_cost_mb["siglip"]))
    text_embed_model = None if settings.profile == "cloud" else settings.text_embed_model
    if text_embed_model is not None:
        registry.register(ModelSpec(
            "nomic",
            load=lambda: get_text_embedder(text_embed_model, settings.embed_device),
            free=release_text_embedder, cost_mb=settings.model_cost_mb["nomic"]))
    registry.register(ModelSpec(
        "gemma", load=llm.load, free=llm.free, cost_mb=settings.model_cost_mb["gemma"]))

    gov = MemoryGovernor(
        registry, measure_free_mb=lambda: psutil.virtual_memory().available / 1e6,
        budget_mb=settings.ram_budget_mb)
    scheduler = Scheduler(gov, concurrency=settings.gpu_concurrency)

    embed = SiglipBackend(settings.embed_model_name, settings.embed_device, scheduler=scheduler)
    caption = build_caption_backend(settings, inf)
    return CompositeBackend(
        embed=embed, caption=caption,
        text=TextBackend(inf, text_embed_model=text_embed_model,
                         device=settings.embed_device, model_name=settings.planner_model,
                         scheduler=scheduler),
        registry=registry, governor=gov, scheduler=scheduler)
```

`modelsvc/backends/siglip_backend.py` — replace `residency.use("siglip", HIGH)` with `scheduler.run(["siglip"], Priority.INTERACTIVE, fn)`. Constructor takes `scheduler=` instead of `residency=`. Each op (`embed_image`/`embed_text`/`tag`/`calibration`) wraps its body in `self._scheduler.run(["siglip"], priority, lambda: <body>)`. Ingest embed/taxonomy come in over the same HTTP endpoints as chat/search, so priority is uniform here — use `Priority.INTERACTIVE` (search/chat) as the default; the batch/interactive split is enforced where the *caller* sets it is not visible to SigLIP, so keep INTERACTIVE (SigLIP ops are short). gemma-bound ops (caption/plan/chat) carry the real batch-vs-interactive split (below).

`modelsvc/backends/text_backend.py` — inject `scheduler`. `text_embed` wraps the nomic path in `scheduler.run(["nomic"], Priority.INTERACTIVE, fn)` when `_text_embed_model` set (cloud path unchanged). `text_complete`/`text_stream` wrap in `scheduler.run(["gemma"], priority, fn)` — `text_complete` (planner) uses `Priority.INTERACTIVE`; captioning (a separate op) is BATCH. Because both planner and caption reach gemma via the shared `OpenAICompatClient`, the scheduler ensures gemma is resident before the HTTP call.

`modelsvc/backends/caption_backend.py` / `composite.py` — `CompositeBackend.caption` wraps in `scheduler.run(["gemma"], Priority.BATCH, fn)`. Add `registry`/`governor`/`scheduler` to `CompositeBackend.__init__`; `resources()` returns `registry.resident()` for `resident` and adds `budget_mb`, `free_mb` from `governor.state()` (extend `ResourcesResponse` in Task 7).

Delete `modelsvc/residency.py` and `tests/test_residency.py` (superseded by `tests/test_registry.py`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_modelsvc_backends.py tests/test_registry.py tests/test_activity.py -v`
Expected: PASS. Then full targeted run: `uv run pytest tests/test_modelsvc_api.py -v` (caption/text over `FakeBackend` — unaffected).

- [ ] **Step 5: Commit**

```bash
git add modelsvc/backends embedding/siglip.py embedding/text_embedder.py tests/test_modelsvc_backends.py tests/test_registry.py
git rm modelsvc/residency.py tests/test_residency.py
git commit -m "feat(modelsvc): route all backend ops through the scheduler; retire Residency"
```

---

## Task 7: Control API — GET /models, POST /models/{name}/ensure|unload

**Files:**
- Modify: `modelsvc/app.py` (add routes + extend `ResourcesResponse`), `modelsvc/backends/base.py` (Protocol), `modelsvc/backends/fake.py` (fake impls), `modelsvc/backends/composite.py` (`models_state`/`ensure`/`unload`)
- Modify: `inference/models_client.py` (client methods for the new endpoints)
- Test: `tests/test_models_control_api.py`, update `tests/test_modelsvc_api.py`

**Interfaces:**
- Produces: `GET /models` → `ModelsStateResponse{resident: list[str], budget_mb: int, free_mb: float, used_mb: int, active: str | None}`; `POST /models/{name}/ensure` → `{}` (scheduler.run to make it resident); `POST /models/{name}/unload` → `{}` (governor/registry unload). `ModelsClient.models_state()`, `ModelsClient.model_ensure(name)`, `ModelsClient.model_unload(name)`. `CompositeBackend.models_state()`, `.model_ensure(name)`, `.model_unload(name)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_models_control_api.py
from starlette.testclient import TestClient
from modelsvc.app import create_models_app
from modelsvc.backends.fake import FakeBackend


def test_models_state_endpoint():
    app = create_models_app(FakeBackend())
    c = TestClient(app)
    r = c.get("/models")
    assert r.status_code == 200
    body = r.json()
    assert "resident" in body and "budget_mb" in body and "free_mb" in body


def test_ensure_and_unload_endpoints():
    app = create_models_app(FakeBackend())
    c = TestClient(app)
    assert c.post("/models/siglip/ensure").status_code == 200
    assert c.post("/models/siglip/unload").status_code == 200
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_models_control_api.py -v`
Expected: FAIL — `404` on `/models` (route absent).

- [ ] **Step 3: Implement**

`modelsvc/backends/base.py` — add to the `ModelBackend` Protocol: `models_state() -> dict`, `model_ensure(name: str) -> None`, `model_unload(name: str) -> None`.

`modelsvc/backends/fake.py::FakeBackend` — add trivial impls: `models_state()` → `{"resident": [], "budget_mb": 0, "free_mb": 0.0, "used_mb": 0, "active": None}`; `model_ensure`/`model_unload` → `None`.

`modelsvc/backends/composite.py`:
```python
def models_state(self) -> dict:
    st = self._governor.state()
    return {"resident": st.resident, "budget_mb": st.budget_mb, "free_mb": st.free_mb,
            "used_mb": st.used_mb, "active": self._activity.current()}

def model_ensure(self, name: str) -> None:
    self._scheduler.run([name], Priority.INTERACTIVE, lambda: None)

def model_unload(self, name: str) -> None:
    self._registry.unload(name)
```

`modelsvc/app.py` — add:
```python
class ModelsStateResponse(BaseModel):
    resident: list[str]
    budget_mb: int
    free_mb: float
    used_mb: int
    active: str | None = None

@app.get("/models", response_model=ModelsStateResponse)
def models_state() -> ModelsStateResponse:
    return ModelsStateResponse(**backend.models_state())

@app.post("/models/{name}/ensure")
def model_ensure(name: str) -> dict:
    backend.model_ensure(name); return {}

@app.post("/models/{name}/unload")
def model_unload(name: str) -> dict:
    backend.model_unload(name); return {}
```

`inference/models_client.py` — add `models_state()` (`GET /models`), `model_ensure(name)` (`POST /models/{name}/ensure`), `model_unload(name)` (`POST /models/{name}/unload`), mirroring the existing `resources()` shape (`inference/models_client.py:96`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_models_control_api.py tests/test_modelsvc_api.py tests/test_models_client.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add modelsvc/app.py modelsvc/backends inference/models_client.py tests/test_models_control_api.py tests/test_modelsvc_api.py tests/test_models_client.py
git commit -m "feat(modelsvc): control API — /models state + ensure/unload"
```

---

## Task 8: Ingest worker-only; remove dead coordinator plumbing

**Files:**
- Modify: `web/app.py:162-171` (`drain_now`), `web/app.py:173` (`register_upload_api`), and the four `require(...)` sites (`:204,:464,:515,:889`)
- Modify: `ingest/pipeline.py:34-132` (drop `_lease`/coordinator params), `ingest/cli.py:14-37` (drop coordinator)
- Modify: `web/deps.py:44-52` (remove `make_coordinator`)
- Delete: `models/coordinator.py`, `tests/test_coordinator_noop.py`
- Test: update `tests/test_pipeline_pass.py`, `tests/test_web_chat.py`

**Interfaces:**
- Produces: `drain_pass(context, vocab, should_preempt=lambda: False)` (no `coordinator` param); ingest driven only by `ingest/cli.py`'s loop. Upload finish enqueues jobs and returns without draining. The models `Scheduler` (Task 4) is now the sole serializer, so app-side leases are gone.

- [ ] **Step 1: Write/adjust the failing test**

```python
# tests/test_pipeline_pass.py — update signature usage
def test_drain_pass_has_no_coordinator_param():
    import inspect
    from ingest.pipeline import drain_pass
    assert "coordinator" not in inspect.signature(drain_pass).parameters
```

```python
# tests/test_web_chat.py — the chat stream must not reference a coordinator
def test_chat_stream_has_no_coordinator(monkeypatch):
    import web.app as wa
    src = inspect_source(wa.chat_stream)   # helper: inspect.getsource
    assert "require(" not in src and "make_coordinator" not in src
```
(Keep the existing chat behavioural assertions in `test_web_chat.py`; only remove the ones asserting lease/timeout semantics — those are gone.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_pipeline_pass.py::test_drain_pass_has_no_coordinator_param -v`
Expected: FAIL — `drain_pass` still has a `coordinator` parameter.

- [ ] **Step 3: Implement the removal**

`web/app.py::drain_now` (162-171) — change to enqueue-only. Options: keep `drain_now` as a best-effort thumbnail pass? No — worker-only means the app does not drain. Replace the body so upload finish just returns (jobs already recorded by `register_upload_api`); the worker's `drain_pass` handles everything. Concretely, delete the `drain_pass(context(), vocab)` call and pass a no-op (or drop the `drain_now` arg from `register_upload_api` if the upload API can enqueue without it — verify `register_upload_api` signature in `web/upload_api.py`).

`ingest/pipeline.py` — remove `_lease` (34-42), drop the `coordinator` param from `drain_pass`, and delete the `with _lease(coordinator, "INGEST_EMBED"):` wrapper (keep the body). Remove `from models.coordinator import LeaseBusyError` (29) and the `LeaseBusyError` except clauses (now only `Preempted`/`Exception`).

`ingest/cli.py` (14-37) — remove `coordinator = context.make_coordinator(...)` and pass just `drain_pass(context, vocab, should_preempt=lambda: False)`.

`web/app.py` — at each `require(...)` site (`:204,:464,:515,:889`) remove the `coordinator = ctx.make_coordinator(...)` line and the `with coordinator.require(...):` wrapper, keeping the wrapped body. Remove the now-dead `except (TimeoutError, RefusedError, LeaseBusyError)` guard around the chat body (`:930`).

`web/deps.py` — delete `make_coordinator` (44-52) and its `NoopCoordinator` import.

Delete `models/coordinator.py` and `tests/test_coordinator_noop.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_pipeline_pass.py tests/test_web_chat.py tests/test_one_model_process_gate.py -v`
Expected: PASS. The one-model-process gate still passes (no new torch imports in thin clients; `models/coordinator.py` removal does not affect the scanned dirs).

- [ ] **Step 5: Commit**

```bash
git add web/app.py web/deps.py ingest/pipeline.py ingest/cli.py tests/test_pipeline_pass.py tests/test_web_chat.py
git rm models/coordinator.py tests/test_coordinator_noop.py
git commit -m "refactor: ingest worker-only; remove dead NoopCoordinator/lease plumbing"
```

---

## Task 9: Infra — models container owns llama-server; drop the inference service

**Files:**
- Modify: `compose.jetson.yaml` (remove `inference` service; mount `llamacpp`+`llama-models` into `models`; set `llm_bin`/paths/`LD_LIBRARY_PATH`), `compose.mac.yaml`, `compose.yaml`
- Modify: `Dockerfile.models.jetson` (ensure runtime CUDA libs for the spawned binary reachable), `Makefile` (`up`, `run-jetson`, `llama-mac`), `README.md`
- Verification: manual/compose commands (no pytest)

**Interfaces:**
- Produces: on jetson the `models` container mounts the external `llamacpp` volume (the prebuilt `sm_87` `llama-server` + `libggml-cuda.so`) and `llama-models` volume (GGUFs), with `IVMS777_LLM_BIN=/llamacpp/llama.cpp/build/bin/llama-server` and a spawn env that puts that dir on `LD_LIBRARY_PATH` for the child only. On mac, `make up` sets `IVMS777_LLM_BIN=$(LLAMA_BIN)` and the models service spawns the Metal binary; the standalone `llama-mac` *run* step is dropped (the build+download step stays).

- [ ] **Step 1: Jetson compose — fold llama-server into models**

In `compose.jetson.yaml`: **delete** the `inference` service block (19-52). On the `models` service (74-87) add:
```yaml
    volumes:
      - llamacpp:/llamacpp
      - llama-models:/data/models
      - ivms777-data:/data
    environment:
      # ...existing...
      IVMS777_LLM_BIN: /llamacpp/llama.cpp/build/bin/llama-server
      IVMS777_LLM_MANAGED: "true"
      IVMS777_INFERENCE_BASE_URL: http://localhost:8080/v1
    ports:
      - "8080:8080"
```
Remove `depends_on: [inference]`. `app`/`worker` drop `depends_on: [inference]` too (keep `depends_on: [models]`). Keep the `llamacpp` (external) and `llama-models` volume declarations.

The models container's `Popen` env for the child (set in `build_llm_process`/`SubprocessLlm.env`) must include `LD_LIBRARY_PATH=/llamacpp/llama.cpp/build/bin` so the cu130-built `llama-server` finds `libggml-cuda.so`, and rely on the nvidia runtime (already `NVIDIA_VISIBLE_DEVICES=all`, `NVIDIA_DRIVER_CAPABILITIES=all`) for `libcuda`. Add that env in Task 5's `build_llm_process` via `SubprocessLlm(..., env={"LD_LIBRARY_PATH": os.environ.get("IVMS777_LLM_LDPATH", "/llamacpp/llama.cpp/build/bin")})` — add `llm_ldpath` to config or read the env directly. Keep it out of the container-wide env so it never shadows torch's cu132 libs.

- [ ] **Step 2: Mac Makefile — models service spawns the Metal binary**

`Makefile` `up` (39-51): keep the `llama-mac` *build+download* dependency but remove the process *launch* (or split `llama-mac` into `llama-mac-build` + drop the `nohup llama-server` run). Add `IVMS777_LLM_BIN=$(LLAMA_BIN)` and `IVMS777_LLM_MANAGED=true` to the `up` env block; the models `uvicorn` will spawn/kill llama-server itself. `IVMS777_INFERENCE_BASE_URL` stays `http://localhost:$(LLAMA_PORT)/v1`.

`run-jetson` (79-101): drop the `inference` build/wait; wait on `http://localhost:8080/health` still works (now served by the models-container child). Keep MAXN.

- [ ] **Step 3: README + design**

`README.md`: update `## Run on a Mac` and `## Run on a Jetson` — the models service now owns `llama-server`; there is no separate `inference` container; unload/reload is automatic under memory pressure. Keep the Metal/sm_87 build recipes (the binary is still built the same way; only *who launches it* changed).

- [ ] **Step 4: Verify (no pytest — infra)**

```bash
# mac
make up            # models service should spawn llama-server; check:
curl -s localhost:8080/health && curl -s localhost:9000/models | jq
# jetson (on device)
make run-jetson
docker compose -f compose.yaml -f compose.jetson.yaml ps    # no 'inference' service
docker compose ... exec -T models sh -c 'curl -s localhost:8080/health'
curl -s http://<jetson>:9000/models    # resident set + budget
```
Expected: `/models` shows `gemma` resident after a caption/chat; after `llm_idle_ttl_s` of SigLIP-only work, `gemma` is unloaded (freeing ~2 GB) and reloaded on the next caption.

- [ ] **Step 5: Commit**

```bash
git add compose.yaml compose.jetson.yaml compose.mac.yaml Dockerfile.models.jetson Makefile README.md
git commit -m "infra: models service supervises llama-server; drop the standalone inference service"
```

---

## Task 10: Idle-TTL unload + design.md coherence pass

**Files:**
- Modify: `modelsvc/backends/composite.py` or `modelsvc/scheduler.py` — a lightweight idle-unload of `gemma` after `settings.llm_idle_ttl_s`
- Modify: `docs/design.md` §3.1, §4, §5 (mermaid), §5.1, §8.1
- Test: `tests/test_scheduler.py` (idle-unload), design self-review

**Interfaces:**
- Produces: after each gemma op completes, record a monotonic timestamp; a cheap check on each subsequent op (or a small daemon thread) unloads `gemma` when idle longer than `llm_idle_ttl_s` and no op needs it. `None` TTL (mac/cloud) disables it.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scheduler.py — append
def test_idle_unload_frees_gemma_after_ttl():
    from modelsvc.governor import MemoryGovernor
    from modelsvc.registry import ModelSpec, ModelRegistry
    log = []
    r = ModelRegistry()
    r.register(ModelSpec("gemma", lambda: log.append("load"), lambda: log.append("free"), 2200))
    gov = MemoryGovernor(r, measure_free_mb=lambda: 9999.0, budget_mb=6000)
    from modelsvc.scheduler import Scheduler, Priority
    now = [1000.0]
    s = Scheduler(gov, concurrency=1, idle_ttl_s=120, clock=lambda: now[0])
    s.run(["gemma"], Priority.BATCH, lambda: None)   # loads + stamps
    now[0] += 200                                     # exceed TTL
    s.reap_idle(["gemma"])                            # explicit reap (also called per-op)
    assert log == ["load", "free"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_scheduler.py::test_idle_unload_frees_gemma_after_ttl -v`
Expected: FAIL — `Scheduler.__init__` has no `idle_ttl_s`/`clock`, no `reap_idle`.

- [ ] **Step 3: Implement**

Extend `Scheduler.__init__(..., idle_ttl_s: int | None = None, clock=time.monotonic)`; record `self._last_use: dict[str, float]` stamped in `run` after `fn()` for each name in `needed`; add:
```python
def reap_idle(self, candidates: list[str]) -> None:
    if self._idle_ttl_s is None:
        return
    now = self._clock()
    for name in candidates:
        last = self._last_use.get(name)
        if last is not None and now - last >= self._idle_ttl_s:
            self._gov._r.unload(name)         # via registry; safe if already gone
            self._last_use.pop(name, None)
```
Call `reap_idle(["gemma"])` at the end of every non-gemma op (cheap, no lock contention) — e.g. in `CompositeBackend.embed_image`/`embed_text`/`text_embed` after the scheduler returns. Wire `idle_ttl_s=settings.llm_idle_ttl_s` in `build_backend` (Task 6's `Scheduler(...)` construction).

- [ ] **Step 4: Design coherence pass — rewrite the owning sections**

Edit `docs/design.md`:
- **§3.1** — profile table row for Inference: mac/jetson now "one `gemma4-E2B` GGUF on `llama-server`, **supervised as a child process by the `models` service** (spawn/kill for memory)". Remove the standalone `inference` container from the jetson description; note the models container mounts the `llamacpp`+`llama-models` volumes. Keep the sizing/MAXN/GPU-vision facts.
- **§4** — add the conveyor paragraph: all model control is in the `models` service; a scheduler (concurrency + priority) + governor (budget + measured free) load/evict SigLIP, nomic, gemma; gemma stays a separate process supervised by the service.
- **§5 mermaid** — remove the separate `infer` box on jetson; show `models` owning a `llama-server` child process; add the scheduler/governor as the models service's internals; `app`/`worker` still call `models` over HTTP.
- **§5.1** — rewrite: the models service is the single control plane — HTTP surface gains `GET /models`, `POST /models/{name}/ensure|unload`; residency replaced by registry+governor+scheduler+LlmProcess.
- **§8.1** — replace "ensure-loaded, nothing to evict" with the conveyor: budget-driven load/evict + priority; gemma unloads on idle TTL / under pressure; ingest is worker-only.

Run the writing-plans self-review checklist against the spec sections: confirm every §-change has a task, no placeholders remain, and method names match across tasks (`registry.ensure/unload/resident`, `governor.acquire/state`, `scheduler.run/reap_idle`, `llm.load/free/is_loaded`).

- [ ] **Step 5: Commit**

```bash
git add modelsvc/scheduler.py modelsvc/backends/composite.py docs/design.md tests/test_scheduler.py
git commit -m "feat(modelsvc): idle-TTL gemma unload; rewrite design §3.1/§4/§5/§5.1/§8.1 for the conveyor"
```

---

## Verification (whole plan)

- [ ] `uv run pytest -q` — full suite green (note the pre-existing caption `dimensions`/`tags` inconsistency from dossier §14 may already be red on `main`; do not let this plan depend on it — if those tests were red before, they stay out of scope).
- [ ] `uv run ruff check .` — clean.
- [ ] `uv run pytest tests/test_one_model_process_gate.py -v` — the gate still passes; no `llama_cpp`/torch leaked into thin clients.
- [ ] On the Jetson: a chat during a running caption batch is served promptly (interactive priority), and `GET /models` shows gemma unloading after idle then reloading for the next caption.
