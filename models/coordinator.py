# models/coordinator.py
"""Torch-free stub of the old cross-process model coordinator (design §8.1).

Plan 15 moved every model into the `modelsvc` service (Tasks 2-5): `app` and
`worker` hold no models in-process anymore, so there is nothing left to
coordinate here. This module keeps only the call-site surface — the two
exception types and a no-op `NoopCoordinator` — so callers written against the
old `ModelCoordinator` API (`web/app.py`, `ingest/pipeline.py`) need no edits.
The real `ModelCoordinator` (DB lease, heartbeat, preempt, in-process
load/release) is deleted along with `models/lease_store.py` and
`models/workloads.py`. `modelsvc/residency.py` (plan 15 task 5) is the
in-process successor for model residency, entirely inside the service.
"""
import contextlib


class RefusedError(RuntimeError):
    """Kept for callers' `except (..., RefusedError, ...)` clauses. The
    NoopCoordinator never raises it — nothing here refuses anymore."""


class LeaseBusyError(RuntimeError):
    """Kept for callers' `except (..., LeaseBusyError, ...)` clauses. The
    NoopCoordinator never raises it — nothing here takes a lease anymore."""


class NoopCoordinator:
    """Stand-in for the deleted `ModelCoordinator`. `require(workload)` is a
    nullcontext: app/worker hold no models, so there is nothing to acquire,
    load, or release."""

    captioner = None

    @contextlib.contextmanager
    def require(self, workload):
        yield
