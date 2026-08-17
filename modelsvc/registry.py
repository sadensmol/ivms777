"""The resident-set manager: the one place a model is loaded or freed.

Replaces the old ensure-loaded-only ``Residency``. Each model registers a real
``load``/``free`` pair and a rough resident cost; the ``MemoryGovernor`` drives
``ensure``/``unload`` against a budget. Thread-safe: the models service runs
sync handlers on the anyio threadpool, so several requests touch this at once.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    name: str
    load: Callable[[], None]
    free: Callable[[], None]
    cost_mb: int
    # Optional liveness probe for a model that can die WITHOUT us unloading it —
    # the supervised `llama-server` child aborts on its own when a CUDA malloc
    # fails mid-request (§8.1). None = the model cannot die behind our back
    # (an in-process torch model lives exactly as long as its reference).
    alive: Callable[[], bool] | None = None


class ModelRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, ModelSpec] = {}
        self._loaded: OrderedDict[str, None] = OrderedDict()  # LRU: oldest first
        self._lock = threading.RLock()

    def register(self, spec: ModelSpec) -> None:
        with self._lock:
            self._specs[spec.name] = spec

    def ensure(self, name: str) -> None:
        with self._lock:
            spec = self._specs[name]  # KeyError if unknown
            if name in self._loaded:
                if spec.alive is None or spec.alive():
                    self._loaded.move_to_end(name)
                    return
                # It died behind our back: the bookkeeping is stale, not the truth.
                # Forget it and load again — otherwise every request goes to a dead
                # llama-server port and 500s forever (§8.1).
                del self._loaded[name]
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
