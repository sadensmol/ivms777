"""In-process residency manager (design §8.1).

The IN-PROCESS successor to the cross-process `ModelCoordinator` — now that every
model lives in ONE process (`modelsvc`), residency is a plain in-process concern:
no DB lease, no heartbeat, no stale-reclaim.

Since plan 16 removed the in-process caption VLM (captioning is now a remote
OpenAI call to llama-server), **SigLIP is the only heavy in-process model left**.
Nothing contends with it for the GPU, so residency collapses to an idempotent
**ensure-loaded**: load a model once on first use, keep it resident, and report
what is loaded for the resource bar (§13). There is no eviction, no priority
preemption, and no swap — those existed only to arbitrate SigLIP against the old
caption VLM sharing one CUDA device.

Torch-free at import: `SiglipBackend` registers `load`/`release` callables that
import `embedding.siglip` LAZILY, inside themselves — `Residency` only ever holds
and calls opaque callables.
"""

import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager

# Priority is kept as a no-op hint (SigLIP is the only registered model, so there
# is nothing to prioritise against); callers still pass it for readability.
HIGH = 10
LOW = 1


class Residency:
    """Ensure-loaded, once, per registered model. `exclusive` is accepted for
    call-site compatibility but has no effect — there is only one heavy model."""

    def __init__(self, *, exclusive: bool = False) -> None:
        self._exclusive = exclusive
        self._models: dict[str, tuple[Callable[[], None], Callable[[], None]]] = {}
        self._loaded: list[str] = []
        self._lock = threading.Lock()

    def register(self, name: str, load: Callable[[], None], release: Callable[[], None]) -> None:
        self._models[name] = (load, release)

    @contextmanager
    def use(self, name: str, *, priority: int = LOW) -> Iterator[None]:
        with self._lock:
            if name not in self._loaded:
                self._models[name][0]()
                self._loaded.append(name)
        yield

    def resident(self) -> list[str]:
        return list(self._loaded)
