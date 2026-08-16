"""The models service's current in-flight operation, for the resource bar (§13).

The `models` service is the ONE process that sees every embed/caption/text call
(§5.1), so "what is the box doing right now" is derived HERE — this replaces the
cross-process lease *workload* that plan 15 removed with the coordinator (§8.1).
`resources()` reports `current()`; each backend op wraps its body in `track()`.

Thread-safe: several requests can be in flight (an embed while a caption
streams); `track()` keeps a stack and `current()` returns the most recently
started op still running, or `None` when idle.
"""
import threading
from collections.abc import Iterator
from contextlib import contextmanager


class Activity:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stack: list[str] = []

    @contextmanager
    def track(self, label: str) -> Iterator[None]:
        with self._lock:
            self._stack.append(label)
        try:
            yield
        finally:
            with self._lock:
                if label in self._stack:
                    self._stack.remove(label)

    def current(self) -> str | None:
        with self._lock:
            return self._stack[-1] if self._stack else None
