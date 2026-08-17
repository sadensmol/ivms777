"""Budget-and-measurement gate over the ModelRegistry.

``acquire`` makes the requested models resident, evicting non-needed / non-pinned
residents (LRU, oldest first). Eviction is *triggered* by either pressure signal:
the configured ``budget_mb`` ceiling, or the *measured* free RAM (on the unified-
memory Jetson, measured free reflects GPU allocations, so it is the real signal).
The refusal (``InsufficientMemory``) is decided by the **budget** alone — measured
free only sheds more; it never refuses a within-budget load (a low free reading
with nothing left to evict is not a reason to fail the contract).
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass

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
    def __init__(
        self,
        registry: ModelRegistry,
        *,
        measure_free_mb: Callable[[], float],
        budget_mb: int,
        headroom_mb: int = 512,
    ) -> None:
        self._r = registry
        self._measure = measure_free_mb
        self._budget = budget_mb
        self._headroom = headroom_mb
        self._lock = threading.RLock()

    def _resident_cost(self) -> int:
        return sum(self._r.cost_mb(n) for n in self._r.resident())

    def acquire(self, needed: list[str], *, pinned: frozenset[str] = frozenset()) -> None:
        with self._lock:
            keep = set(needed) | set(pinned)
            add_cost = sum(self._r.cost_mb(n) for n in needed if not self._r.is_resident(n))

            for name in self._r.resident():  # snapshot, LRU oldest first
                projected = self._resident_cost() + add_cost
                fits_budget = projected + self._headroom <= self._budget
                fits_real = self._measure() >= add_cost + self._headroom
                if fits_budget and fits_real:
                    break
                if name in keep:
                    continue
                self._r.unload(name)

            projected = self._resident_cost() + add_cost
            if projected + self._headroom > self._budget:
                raise InsufficientMemory(
                    f"cannot fit {needed}: budget={self._budget} "
                    f"projected={projected} pinned={sorted(pinned)}"
                )

            for n in needed:
                self._r.ensure(n)  # loads missing, LRU-touches present

    def unload(self, name: str) -> None:
        with self._lock:
            self._r.unload(name)

    def state(self) -> GovernorState:
        with self._lock:
            return GovernorState(
                resident=self._r.resident(),
                used_mb=self._resident_cost(),
                free_mb=self._measure(),
                budget_mb=self._budget,
            )
