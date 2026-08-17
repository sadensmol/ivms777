# Torch models in a killable child process — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make evicting SigLIP/nomic actually return their RAM to the OS, so
`gemma-vision` fits after an embed/taxonomy batch and captioning stops failing on
the 8 GB Jetson.

**Architecture:** The `models` service stops importing torch in its own process.
SigLIP and nomic each move into a **supervised child process** (`TorchWorker`,
`multiprocessing` "spawn") that the registry starts and **kills** — the same shape
`llama-server` already has (`modelsvc/llm_process.py`). `free()` becomes
`terminate()`, which is the only operation that returns a CUDA context's memory.
The child hosts one plain object and answers `(method, args)` messages over a
`Pipe`; `alive` is `Process.is_alive`, which plugs straight into the liveness probe
`ModelRegistry.ensure()` already re-checks.

**Tech Stack:** Python 3.12, `multiprocessing` (spawn context), pytest, `uv`.

**Spec:** [`docs/design.md`](../design.md) §5.1 (the one model process), §8.1
(in-process residency — the memory governor), §3.1 (deploy profiles).

## Why (measured on the board, JetPack 7.2, 8 GB unified)

| after | process anon RSS | CUDA device-free |
|---|---|---|
| CUDA context only | 442 MB | 3945 MB |
| SigLIP loaded + one batch | 5251 MB | 534 MB |
| SigLIP released (`cache_clear` + `gc` + `empty_cache` + `ipc_collect`) | **2712 MB** | **2196 MB** |
| + `_cuda_clearCublasWorkspaces` + `malloc_trim(0)` | 2646 MB | 2475 MB |

`torch.cuda.memory_reserved()` is 20 MB after release — torch has let go, the CUDA
driver has not. gemma-vision needs ~4.4 GB, so after one embed batch it can never
load: `llama-server` aborts (`rc=-6`) at model load or at image decode
(`cudaMalloc failed: out of memory`, 98 MiB), and captions fail until the container
is restarted. Reproduced end-to-end: embed 16 photos → 3/3 captions 500.

Rejected alternatives, with measurements:
- **SigLIP on CPU** (no CUDA context, residue 0.7 GB): **54× slower** — 9.64 img/s
  GPU vs 0.18 img/s CPU. Not viable.
- **`--parallel 1`** on llama-server: saves ~20 MB. Noise.
- **`-ub 256`** to shrink the vision compute buffer: breaks captioning outright —
  `GGML_ASSERT(cparams.causal_attn || cparams.n_ubatch >= n_tokens_all)`.
- **In-process deep release** (cuBLAS workspaces, `malloc_trim`): ~50 MB. Noise.

## Global Constraints

- **One model process** (CLAUDE.md, §5.1): only the `models` service owns model
  work. A `TorchWorker` child is *part of* that service, exactly as the
  `llama-server` child is — `app`/`worker`/CLI stay thin HTTP clients and never
  import torch. No model is ever loaded twice.
- **The parent must stay torch-free.** After this plan, nothing the `models`
  service imports in its own interpreter pulls `torch`/`transformers`. That is
  what keeps a CUDA context out of the parent for good.
- **GPU-only for gemma** (§8.1): never "fix" memory pressure by moving gemma
  layers to the CPU.
- `uv run` for everything. Tests live in `tests/`. New module ⇒ new tests.
- Python 3.12; no new third-party dependency — `multiprocessing` is stdlib.
- Spawn (not fork): a forked child inherits the parent's CUDA state and would
  defeat the whole point.

---

### Task 1: `TorchWorker` — a supervised, killable model host

**Files:**
- Create: `modelsvc/torch_process.py`
- Create: `tests/worker_fixtures.py`
- Test: `tests/test_torch_process.py`

**Interfaces:**
- Consumes: nothing (stdlib only).
- Produces:
  - `TorchWorker(target: str, args: tuple = (), *, warm: str | None = None, ready_timeout_s: float = 300.0)`
    where `target` is `"module.path:CallableName"` resolved **in the child**.
  - `TorchWorker.start() -> None` (idempotent while alive)
  - `TorchWorker.call(method: str, *args) -> Any`
  - `TorchWorker.stop() -> None`
  - `TorchWorker.is_alive() -> bool`

- [ ] **Step 1: Write the fixture the child will host**

Create `tests/worker_fixtures.py` — a plain, importable, torch-free object so the
worker's process semantics can be tested for real (no mocks, no torch):

```python
"""Objects hosted inside a `TorchWorker` child in tests (importable by `spawn`)."""

import os


class Doubler:
    def __init__(self, offset: int = 0) -> None:
        self._offset = offset
        self._warmed = False

    def warm(self) -> None:
        self._warmed = True

    def double(self, values: list[int]) -> list[int]:
        return [v * 2 + self._offset for v in values]

    def warmed(self) -> bool:
        return self._warmed

    def pid(self) -> int:
        return os.getpid()

    def boom(self) -> None:
        raise ValueError("nope")

    def die(self) -> None:
        os._exit(1)  # simulates llama-server-style SIGABRT: no reply ever comes
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_torch_process.py`:

```python
import os

import pytest

from modelsvc.torch_process import TorchWorker


def _worker(**kw):
    return TorchWorker("tests.worker_fixtures:Doubler", (1,), **kw)


def test_calls_run_in_a_separate_process():
    w = _worker()
    w.start()
    try:
        assert w.call("double", [1, 2]) == [3, 5]
        assert w.call("pid") != os.getpid()
    finally:
        w.stop()


def test_stop_ends_the_child_and_is_alive_reports_it():
    w = _worker()
    w.start()
    assert w.is_alive()
    w.stop()
    assert not w.is_alive()


def test_start_is_idempotent_while_alive():
    w = _worker()
    w.start()
    try:
        first = w.call("pid")
        w.start()
        assert w.call("pid") == first
    finally:
        w.stop()


def test_warm_runs_at_start_when_asked():
    w = _worker(warm="warm")
    w.start()
    try:
        assert w.call("warmed") is True
    finally:
        w.stop()


def test_error_inside_the_child_surfaces_as_runtime_error():
    w = _worker()
    w.start()
    try:
        with pytest.raises(RuntimeError, match="ValueError: nope"):
            w.call("boom")
        assert w.is_alive()  # a raising method must not kill the host
    finally:
        w.stop()


def test_a_child_that_dies_mid_call_raises_and_reports_dead():
    w = _worker()
    w.start()
    try:
        with pytest.raises(RuntimeError, match="died"):
            w.call("die")
        assert not w.is_alive()
    finally:
        w.stop()
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_torch_process.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'modelsvc.torch_process'`.

- [ ] **Step 4: Write the implementation**

Create `modelsvc/torch_process.py`:

```python
"""A model hosted in a child process the registry can KILL.

Freeing a torch model in-process does not return its memory: on the Jetson the
CUDA driver keeps ~2.7 GB of the process's anonymous RSS after `empty_cache()`
(measured, design §8.1), which is exactly the RAM gemma then cannot have. Ending
the process is the only operation that gives it back, so SigLIP and nomic live in
a supervised child — the same shape `llama-server` already has
(`modelsvc/llm_process.py`), and `free()` is `terminate()`.

`spawn`, never `fork`: a forked child inherits the parent's CUDA state, which
would defeat the point. The parent therefore never imports torch at all.
"""

from __future__ import annotations

import multiprocessing as mp
import threading
from importlib import import_module
from typing import Any

_CTX = mp.get_context("spawn")


def _resolve(target: str):
    module_path, _, attr = target.partition(":")
    return getattr(import_module(module_path), attr)


def _child_main(conn, target: str, args: tuple, warm: str | None) -> None:
    """Build the hosted object, then answer `(method, args)` messages forever."""
    try:
        obj = _resolve(target)(*args)
        if warm is not None:
            getattr(obj, warm)()
    except BaseException as exc:  # noqa: BLE001 - report, then die quietly
        conn.send(("err", f"{type(exc).__name__}: {exc}"))
        return
    conn.send(("ok", None))  # ready handshake
    while True:
        try:
            message = conn.recv()
        except EOFError:
            return
        if message is None:
            return
        method, call_args = message
        try:
            conn.send(("ok", getattr(obj, method)(*call_args)))
        except BaseException as exc:  # noqa: BLE001 - a bad call must not kill the host
            conn.send(("err", f"{type(exc).__name__}: {exc}"))


class TorchWorker:
    def __init__(
        self,
        target: str,
        args: tuple = (),
        *,
        warm: str | None = None,
        ready_timeout_s: float = 300.0,
    ) -> None:
        self._target = target
        self._args = args
        self._warm = warm
        self._ready_timeout = ready_timeout_s
        self._proc: mp.process.BaseProcess | None = None
        self._conn = None
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            if self._proc is not None and self._proc.is_alive():
                return
            self._reap()
            parent, child = _CTX.Pipe()
            proc = _CTX.Process(
                target=_child_main,
                args=(child, self._target, self._args, self._warm),
                daemon=True,
            )
            proc.start()
            child.close()  # the parent's copy of the child end, or EOF never fires
            self._proc, self._conn = proc, parent
            if not parent.poll(self._ready_timeout):
                self._kill()
                raise TimeoutError(f"{self._target} not ready in {self._ready_timeout}s")
            kind, payload = parent.recv()
            if kind != "ok":
                self._kill()
                raise RuntimeError(f"{self._target} failed to load: {payload}")

    def call(self, method: str, *args: Any) -> Any:
        with self._lock:
            if self._conn is None or self._proc is None or not self._proc.is_alive():
                raise RuntimeError(f"{self._target} worker is not running")
            try:
                self._conn.send((method, args))
                kind, payload = self._conn.recv()
            except (EOFError, BrokenPipeError, ConnectionResetError):
                self._kill()
                raise RuntimeError(f"{self._target} worker died during {method}") from None
            if kind != "ok":
                raise RuntimeError(payload)
            return payload

    def stop(self) -> None:
        with self._lock:
            self._kill()

    def is_alive(self) -> bool:
        proc = self._proc
        return proc is not None and proc.is_alive()

    def _kill(self) -> None:
        proc, conn, self._proc, self._conn = self._proc, self._conn, None, None
        if conn is not None:
            conn.close()
        if proc is None:
            return
        proc.terminate()
        proc.join(10)
        if proc.is_alive():
            proc.kill()
            proc.join(5)

    def _reap(self) -> None:
        if self._proc is not None:
            self._proc.join(0)
            self._proc = None
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_torch_process.py -v`
Expected: PASS (6 tests).

- [ ] **Step 6: Run the whole suite**

Run: `uv run pytest -q`
Expected: all green, output pristine.

- [ ] **Step 7: Commit**

The user commits. Stop here and report; never run `git add`/`git commit`.

---

### Task 2: SigLIP through the worker

**Files:**
- Modify: `embedding/siglip.py` (add a bytes-in method so the child does the decode)
- Modify: `modelsvc/backends/siglip_backend.py` (take a worker, not an lru_cache)
- Test: `tests/test_modelsvc_backends.py`

**Interfaces:**
- Consumes: `TorchWorker` from Task 1.
- Produces:
  - `SiglipEmbedder.embed_image_bytes(images: list[bytes]) -> list[list[float]]`
  - `SiglipEmbedder.calibration() -> dict` (`{"logit_scale": float, "logit_bias": float}`)
  - `SiglipBackend(worker)` — `embed_image`/`embed_text`/`calibration` all route
    through `worker.call(...)`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_modelsvc_backends.py` (a fake worker records the calls — the
real child is covered by Task 1's process tests):

```python
class _RecordingWorker:
    def __init__(self):
        self.calls = []

    def call(self, method, *args):
        self.calls.append((method, args))
        return [[0.5]] if method.startswith("embed") else {"logit_scale": 1.0, "logit_bias": 0.0}


def test_siglip_backend_routes_every_op_through_the_worker():
    from modelsvc.backends.siglip_backend import SiglipBackend

    worker = _RecordingWorker()
    backend = SiglipBackend(worker)
    assert backend.embed_image([b"jpegbytes"]) == [[0.5]]
    assert backend.embed_text(["hi"]) == [[0.5]]
    assert backend.calibration() == {"logit_scale": 1.0, "logit_bias": 0.0}
    assert [c[0] for c in worker.calls] == ["embed_image_bytes", "embed_texts", "calibration"]
    # raw bytes cross the pipe: decoding is the child's job, not the parent's
    assert worker.calls[0][1] == ([b"jpegbytes"],)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_modelsvc_backends.py::test_siglip_backend_routes_every_op_through_the_worker -v`
Expected: FAIL — `SiglipBackend.__init__() takes 3 positional arguments but 2 were given`.

- [ ] **Step 3: Add the bytes-in method to the hosted model**

In `embedding/siglip.py`, inside `SiglipEmbedder`:

```python
    def embed_image_bytes(self, images: list[bytes]) -> list[list[float]]:
        """Decode + embed. The bytes (not PIL objects) are what cross the worker
        pipe, so the decode happens here, in the child that owns torch."""
        from io import BytesIO

        return self.embed_images([Image.open(BytesIO(b)).convert("RGB") for b in images])

    def calibration(self) -> dict:
        return {"logit_scale": self.logit_scale, "logit_bias": self.logit_bias}
```

- [ ] **Step 4: Rewrite the backend to route through the worker**

Replace the body of `modelsvc/backends/siglip_backend.py`:

```python
"""SigLIP embed sub-backend (design §5.1, §8.1).

Torch-free: SigLIP lives in a `TorchWorker` child (plan 20), so this module never
imports `embedding.siglip` and the `models` parent process never builds a CUDA
context. Load/evict and GPU serialization stay with the conveyor — the registry
starts and kills the worker, `CompositeBackend` wraps each op in
`scheduler.run(["siglip"], ...)`.
"""


class SiglipBackend:
    def __init__(self, worker) -> None:
        self._worker = worker

    def embed_image(self, images: list[bytes]) -> list[list[float]]:
        return self._worker.call("embed_image_bytes", images)

    def embed_text(self, texts: list[str]) -> list[list[float]]:
        return self._worker.call("embed_texts", texts)

    def tag(self, image: bytes, dimensions: list[str]) -> dict[str, list[str]]:
        # Nothing calls /tag: taxonomy scoring stays client-side (ingest computes it
        # from embed_texts + calibration via RemoteEmbedder).
        return {}

    def calibration(self) -> dict:
        return self._worker.call("calibration")
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest tests/test_modelsvc_backends.py -v`
Expected: PASS. Fix any other test that built `SiglipBackend(model_name, device)`.

- [ ] **Step 6: Commit** — the user commits; stop and report.

---

### Task 3: nomic through the worker

**Files:**
- Modify: `embedding/text_embedder.py` (public `warm()`)
- Modify: `modelsvc/backends/text_backend.py` (take an optional worker)
- Test: `tests/test_modelsvc_backends.py`

**Interfaces:**
- Consumes: `TorchWorker` (Task 1).
- Produces: `TextBackend(client, *, text_worker=None, model_name=None)` —
  `text_embed` uses `text_worker.call("embed_texts", texts)` when a worker is
  wired, else falls back to `client.embed` (cloud's OpenAI `/embeddings`).

- [ ] **Step 1: Write the failing test**

```python
def test_text_embed_uses_the_worker_when_wired():
    from modelsvc.backends.text_backend import TextBackend

    worker = _RecordingWorker()
    backend = TextBackend(_StubClient(), text_worker=worker, model_name="gemma4-E2B")
    assert backend.text_embed("nomic", ["a caption"]) == [[0.5]]
    assert worker.calls == [("embed_texts", (["a caption"],))]


def test_text_embed_falls_back_to_the_client_without_a_worker():
    from modelsvc.backends.text_backend import TextBackend

    client = _StubClient()
    backend = TextBackend(client, model_name="qwen")
    backend.text_embed("text-embed-3", ["a caption"])
    assert client.embedded == ("text-embed-3", ["a caption"])
```

`_StubClient` needs an `embed(model, texts)` that records its arguments and
returns `[[0.1]]`; add it next to the existing stubs in the file if absent.

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_modelsvc_backends.py -k text_embed -v`
Expected: FAIL — `TextBackend.__init__() got an unexpected keyword argument 'text_worker'`.

- [ ] **Step 3: Add `warm()` to the text embedder**

In `embedding/text_embedder.py`, inside `TextEmbedder`:

```python
    def warm(self) -> None:
        """Load now, so the worker's `start()` means the model is really resident
        (the governor's budget assumes it) instead of loading on first use."""
        self._load()
```

- [ ] **Step 4: Route `text_embed` through the worker**

In `modelsvc/backends/text_backend.py`, replace the `text_embed_model`/`device`
constructor parameters with `text_worker`, and the method body with:

```python
    def text_embed(self, model: str, texts: list[str]) -> list[list[float]]:
        # nomic runs in a killable child (plan 20); cloud has a real /embeddings
        # backend and no worker, so it falls back to the injected client.
        if self._text_worker is None:
            return self._client.embed(model, texts)
        return self._text_worker.call("embed_texts", texts)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_modelsvc_backends.py -v`
Expected: PASS.

- [ ] **Step 6: Commit** — the user commits; stop and report.

---

### Task 4: Wire the workers into the registry, and update the design doc

**Files:**
- Modify: `modelsvc/backends/__init__.py` (spec wiring)
- Modify: `docs/design.md` §5.1 and §8.1
- Modify: `config.py` (honest `model_cost_mb`)
- Test: `tests/test_modelsvc_backends.py`

**Interfaces:**
- Consumes: `TorchWorker` (1), `SiglipBackend(worker)` (2), `TextBackend(..., text_worker=)` (3).
- Produces: registry specs `siglip` / `nomic` whose `load` is `worker.start`,
  `free` is `worker.stop`, and `alive` is `worker.is_alive`.

- [ ] **Step 1: Write the failing test**

```python
def test_torch_models_are_registered_as_killable_workers(tmp_path):
    backend = build_backend(Settings(data_dir=tmp_path, profile="jetson"))
    for name in ("siglip", "nomic"):
        spec = backend._registry._specs[name]
        assert spec.alive is not None
        assert spec.alive() is False  # nothing spawned at build time
    # the parent never imports torch just by building the backend
    import sys

    assert "torch" not in sys.modules
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_modelsvc_backends.py::test_torch_models_are_registered_as_killable_workers -v`
Expected: FAIL — `assert None is not None` (the siglip spec has no liveness probe).

- [ ] **Step 3: Wire the workers**

In `modelsvc/backends/__init__.py`, replace the `_load_siglip`/`_free_siglip` and
`_load_nomic`/`_free_nomic` closures with workers:

```python
    from modelsvc.torch_process import TorchWorker

    # SigLIP and nomic run in killable children: evicting them in-process does NOT
    # return the CUDA context's memory, which is the RAM gemma needs (§8.1).
    siglip_worker = TorchWorker(
        "embedding.siglip:SiglipEmbedder", (settings.embed_model_name, settings.embed_device)
    )
    registry.register(
        ModelSpec(
            "siglip",
            siglip_worker.start,
            siglip_worker.stop,
            costs["siglip"],
            alive=siglip_worker.is_alive,
        )
    )

    text_worker = None
    if text_embed_model is not None:
        text_worker = TorchWorker(
            "embedding.text_embedder:TextEmbedder",
            (text_embed_model, settings.embed_device),
            warm="warm",
        )
        registry.register(
            ModelSpec(
                "nomic",
                text_worker.start,
                text_worker.stop,
                costs["nomic"],
                alive=text_worker.is_alive,
            )
        )
```

and pass them down: `SiglipBackend(siglip_worker)` and
`TextBackend(inf, text_worker=text_worker, model_name=settings.planner_model)`.

- [ ] **Step 4: Set honest costs**

In `config.py`, `model_cost_mb` — measured resident footprints, not guesses. SigLIP
peaks at ~5.3 GB anonymous RSS while loading (fp32 checkpoint → fp16) and settles
around ~3.4 GB; the governor's guard is `free >= cost + headroom`, and an
under-estimate is what loads gemma on top of it:

```python
            "siglip": 3400,
            "nomic": 800,
```

Update the neighbouring comment to say the numbers are measured on the board and
that a worker's `free` is a process kill, so the eviction really releases them.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest -q`
Expected: all green.

- [ ] **Step 6: Update `docs/design.md` in the same turn as the code**

§8.1: replace the "Known defect — the swap does not actually give the RAM back"
paragraph with the resolved behaviour — the measurement table stays (it is *why*
the design is what it is), and the conclusion becomes: SigLIP and nomic run in
supervised children, `free()` is a process kill, so the embed↔caption swap
genuinely returns the memory; the parent never imports torch and so never holds a
CUDA context.

§5.1: state that the `models` service supervises **three** children — the
`llama-server` (gemma) and one per torch model — and that this is the same
"one model process" rule, not an exception to it: no model is loaded twice and
`app`/`worker` still reach everything over HTTP.

- [ ] **Step 7: Commit** — the user commits; stop and report.

---

### Task 5: Verify on the board

**Files:**
- Test: `/tmp/repro.py` on `lockbox@192.168.100.8` (already written: embed 16
  photos, then 3 captions through `models:9000`).

- [ ] **Step 1: Ask permission, then rebuild and restart the models service**

Ask the user before touching the board. Then:

```bash
ssh lockbox@192.168.100.8 'cd ~/ivms777 && docker compose -f compose.jetson.yaml build models && docker compose -f compose.jetson.yaml up -d models'
```

- [ ] **Step 2: Run the reproduction**

```bash
ssh lockbox@192.168.100.8 'docker exec -w /app -e PYTHONPATH=/app ivms777-models-1 /app/.venv/bin/python /tmp/repro.py'
```

Expected, and the pass/fail bar for this plan:
- `after embed` shows `resident: ["siglip"]`
- all three captions return `OK` with real text
- the caption step's `resources` shows `resident: ["gemma-vision"]`
- `free -m` after the run shows the SigLIP memory returned (available back above
  ~4.5 GB before gemma loads)

- [ ] **Step 3: Confirm the parent process holds no CUDA context**

```bash
ssh lockbox@192.168.100.8 'docker exec ivms777-models-1 sh -c "grep RssAnon /proc/1/status; ps aux | grep -c [s]pawn_main"'
```

Expected: the uvicorn parent's `RssAnon` stays in the tens of MB across an embed
batch; the torch memory belongs to a child that appears and disappears.

- [ ] **Step 4: Requeue the failed captions and watch a full run**

In the UI, hit **Reprocess** on `caption` (206 jobs are in `failed`). Watch
`docker logs -f ivms777-models-1` for OOM aborts; there must be none.

- [ ] **Step 5: Report the numbers, then stop** — the user commits.
