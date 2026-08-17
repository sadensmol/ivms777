"""The single entry gate for all GPU/model work in the models service.

Every op passes through ``run``: it takes one of ``concurrency`` slots (1 on
Jetson -> serialized GPU; N on mac/cloud), and when slots are contended the
highest-priority waiter goes first, so an interactive chat never waits behind a
batch of ingest captions. Once admitted, it asks the governor to make the needed
models resident, then runs the op. Threading, not asyncio: the FastAPI handlers
are sync and run on the anyio threadpool.
"""

from __future__ import annotations

import heapq
import itertools
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from enum import IntEnum
from typing import TypeVar

T = TypeVar("T")


class Priority(IntEnum):
    BATCH = 1  # ingest
    INTERACTIVE = 2  # chat / search


class Scheduler:
    def __init__(
        self,
        governor,
        *,
        concurrency: int,
        idle_ttl_s: int | None = None,
        chat_cooldown_s: int | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._gov = governor
        self.slots = concurrency
        self._free = concurrency
        self._cv = threading.Condition()
        self._waiters: list[tuple[int, int, threading.Event]] = []  # heap by -priority
        self._seq = itertools.count()
        self._idle_ttl_s = idle_ttl_s
        self._clock = clock
        self._last_use: dict[str, float] = {}
        # Sticky-chat window (§8.1): after an interactive op (chat), keep its model
        # resident and PAUSE batch work for this many seconds, so a follow-up chat is
        # instant and never triggers a reload thrash. 0 disables (mac/cloud).
        self._chat_cooldown_s = chat_cooldown_s or 0
        self._sleep = sleep
        self._hold_until = 0.0
        self._poll_s = 0.2

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
                ev.set()  # hand the slot directly to the winner
            else:
                self._free += 1

    def _await_chat_window(self, priority: Priority) -> None:
        """A BATCH op waits out the sticky-chat window (each chat extends it) WITHOUT
        holding a slot, so an interactive chat can always jump in and keeps its model
        resident. Interactive ops never wait (§8.1)."""
        if not self._chat_cooldown_s or priority != Priority.BATCH:
            return
        while (remaining := self._hold_until - self._clock()) > 0:
            self._sleep(min(remaining, self._poll_s))

    @contextmanager
    def reserve(
        self, needed: list[str], priority: Priority, *, pinned: frozenset[str] = frozenset()
    ) -> Iterator[None]:
        """Hold a slot for the whole `with` block — used for streaming, where the
        gemma slot must stay held while tokens stream, not released the instant the
        generator object is built."""
        self._await_chat_window(priority)
        self._admit(priority)
        try:
            self._gov.acquire(needed, pinned=pinned)
            yield
        finally:
            now = self._clock()
            for name in needed:
                self._last_use[name] = now
            # Open/extend the sticky-chat window: keep this chat's model resident and
            # pause batch work for the cooldown, so a follow-up chat stays instant.
            if self._chat_cooldown_s and priority == Priority.INTERACTIVE:
                self._hold_until = now + self._chat_cooldown_s
            self._release()

    def reap_idle(self, candidates: list[str]) -> None:
        """Unload any candidate model idle longer than ``idle_ttl_s`` — the
        proactive path that frees gemma (~2 GB) during long SigLIP-only batches.
        No-op when no TTL is configured (mac/cloud), or inside the sticky-chat window
        (§8.1) — a just-used chat model is kept resident in case the user asks again."""
        if self._idle_ttl_s is None:
            return
        if self._clock() < self._hold_until:
            return  # sticky-chat window: keep the chat model loaded, don't idle-reap
        now = self._clock()
        for name in candidates:
            last = self._last_use.get(name)
            if last is not None and now - last >= self._idle_ttl_s:
                self._gov.unload(name)
                self._last_use.pop(name, None)

    def run(
        self,
        needed: list[str],
        priority: Priority,
        fn: Callable[[], T],
        *,
        pinned: frozenset[str] = frozenset(),
    ) -> T:
        with self.reserve(needed, priority, pinned=pinned):
            return fn()
