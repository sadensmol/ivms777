"""In-process residency manager (design §8.1, plan 15 task 5).

The IN-PROCESS successor to the cross-process `ModelCoordinator`
(`models/coordinator.py`, plan 13) — now that every model lives in ONE
process (`modelsvc`, plan 15 tasks 2-4), residency is a plain in-process
concern: no DB lease, no heartbeat, no stale-reclaim. Those existed only to
coordinate two separate processes racing for the GPU; a single process just
needs a lock.

Only two models are heavy AND in-process: SigLIP (`modelsvc.backends.siglip_backend`)
and the caption VLM (`modelsvc.backends.caption_backend`, jetson only). Ollama
text (`TextBackend`) is a separate/external backend — NEVER managed here.

**Exclusive mode** (jetson, `settings.caption_backend == "inprocess"`): SigLIP
and the caption VLM share one CUDA device and must not be co-resident —
`use()` evicts the other model before loading its own, and a HIGH-priority
`use()` (an interactive embed) preempts a LOW-priority one in flight (a
background caption) via `should_preempt()`.

**Non-exclusive mode** (mac/cloud, `settings.caption_backend == "ollama"`):
SigLIP is CPU/ample-RAM and captioning is external Ollama — no in-process
contention, so `use()` is just an idempotent ensure-loaded: no eviction, no
preemption, mac behaves exactly as before (ruling R2).

Torch-free at import: nothing here imports `embedding.siglip` (which imports
torch). `SiglipBackend` registers `load`/`release` callables that import it
LAZILY, inside themselves — `Residency` only ever holds and calls opaque
callables, never touching their internals.
"""

import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager

HIGH = 10  # interactive embeds (SiglipBackend)
LOW = 1  # background caption (CaptionBackend)


class CaptionPreempted(RuntimeError):
    """A background caption yielded to a higher-priority `use()` (an
    interactive embed) — the in-process counterpart of
    `inference.client.InferenceCancelled` (§8.1). `CaptionBackend.caption`
    catches `InferenceCancelled` from the captioner and re-raises this; the
    `/caption` route maps it to HTTP 503 (Deliverable 3)."""


class Residency:
    """Keeps at most one heavy in-process model resident at a time, in
    exclusive mode, with priority-based preemption. See module docstring for
    the two modes' semantics.
    """

    def __init__(self, *, exclusive: bool) -> None:
        self._exclusive = exclusive
        self._models: dict[str, tuple[Callable[[], None], Callable[[], None]]] = {}
        self._cond = threading.Condition()
        # Exclusive-mode state.
        self._resident: str | None = None
        self._active: str | None = None
        self._active_priority: int | None = None
        self._holders = 0
        self._preempt = False
        # Non-exclusive-mode state: models that have been ensure-loaded once.
        self._loaded: list[str] = []

    def register(self, name: str, load: Callable[[], None], release: Callable[[], None]) -> None:
        self._models[name] = (load, release)

    @contextmanager
    def use(self, name: str, *, priority: int = LOW) -> Iterator[None]:
        if not self._exclusive:
            with self._cond:
                if name not in self._loaded:
                    self._models[name][0]()
                    self._loaded.append(name)
            yield
            return

        with self._cond:
            # Wait for any active holder of a DIFFERENT model to finish. If it is
            # lower priority than us, flag that it should preempt (yield early);
            # if it is equal-or-higher priority, we simply wait our turn. `waited`
            # tracks whether THIS call actually went through that wait loop — a
            # same-model joiner (active == name) never enters it, and must NOT
            # clear `_preempt` on its way through: some other, still-waiting
            # higher-priority caller may have set it, and only the waiter that
            # set/observed it should be the one to clear it once it stops waiting.
            waited = False
            while self._active is not None and self._active != name:
                waited = True
                if self._active_priority is not None and self._active_priority < priority:
                    self._preempt = True
                self._cond.wait()
            if waited:
                self._preempt = False
            if self._resident != name:
                other = self._resident
                if other is not None:
                    self._models[other][1]()  # release the other model first
                self._models[name][0]()  # then load the one we need
                self._resident = name
            self._active = name
            self._active_priority = priority
            self._holders += 1
        try:
            yield
        finally:
            with self._cond:
                self._holders -= 1
                if self._holders == 0:
                    self._active = None
                    self._active_priority = None
                self._cond.notify_all()

    def should_preempt(self) -> bool:
        if not self._exclusive:
            return False
        with self._cond:
            return self._preempt

    def resident(self) -> list[str]:
        if not self._exclusive:
            return list(self._loaded)
        return [self._resident] if self._resident is not None else []
