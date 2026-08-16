# models/coordinator.py
"""The single model-residency decision point (design §8.1). Callers declare a
workload; the coordinator guards the RAM budget, takes the cross-process lease,
loads exactly the declared model-set (evicting the rest), and releases on exit."""
import contextlib
import logging
import threading
import time
from collections.abc import Callable

from embedding.siglip import release_siglip_embedder
from models import lease_store as ls
from models import workloads as wl
from models.lease_store import INTERACTIVE, WorkloadName

logger = logging.getLogger(__name__)

# While a workload holds the lease, a daemon thread bumps its heartbeat this often
# (design §8.1) — on its OWN connection, so a holder blocked in a long caption read
# still proves it is alive. A holder silent for `_LEASE_STALE_S` is treated as dead
# and its lease is reclaimed. Stale must exceed the heartbeat interval by a healthy
# margin (missed beats under load must not look like death) yet stay well under
# `_PREEMPT_TIMEOUT_S`, so a crashed holder is reclaimed long before the safety net.
_HEARTBEAT_S = 5.0
_LEASE_STALE_S = 20.0

_PREEMPT_POLL_S = 0.25
# Must exceed one in-flight ingest unit — bounded by the caption inference timeout
# (`inference/client.py::complete`, 120s) — plus the model swap. Below that, chat
# would give up before the worker reaches its next-photo yield point and wrongly
# report "busy" while a slow caption is still running (§8.1). The forced-abort of
# the caption (see caption stage) makes the real wait far shorter; this is only the
# safety net for when the abort point is not reached in time.
_PREEMPT_TIMEOUT_S = 150.0


class RefusedError(RuntimeError):
    """A workload's model-set does not fit the RAM budget (design §8.1)."""


class LeaseBusyError(RuntimeError):
    """A background workload could not get the lease because interactive work
    holds it; the caller should defer (design §8.1)."""


class ModelCoordinator:
    def __init__(self, conn, client, *, holder, budget_mb, planner_model, caption_model,
                 load_siglip=None, release_siglip=None, captioner=None,
                 sleep=time.sleep, now=time.monotonic,
                 heartbeat_connect=None, heartbeat_interval_s=_HEARTBEAT_S, lease_stale_s=_LEASE_STALE_S):
        self._conn = conn
        self._client = client
        self._holder = holder
        self._budget_mb = budget_mb
        self._planner_model = planner_model
        self._caption_model = caption_model
        self._load_siglip = load_siglip
        self._release_siglip = release_siglip or release_siglip_embedder
        # The active captioner ADAPTER for the INGEST_CAPTION resource (design §4,
        # §8.1) — Ollama on mac/cloud (load() = warm the tag), in-process VLM on
        # jetson (load() = load weights onto the GPU). Public so a caller (the
        # ingest pipeline's caption stage) can reuse the SAME instance the
        # coordinator just loaded, rather than constructing a second one.
        self.captioner = captioner
        self._sleep = sleep
        self._now = now
        # A callable returning a NEW DB connection for the heartbeat thread (the
        # coordinator's own conn belongs to the request thread). None disables the
        # heartbeat — fine for tests and any single-holder profile that never needs
        # stale-reclaim (design §8.1).
        self._heartbeat_connect = heartbeat_connect
        self._heartbeat_interval_s = heartbeat_interval_s
        self._lease_stale_s = lease_stale_s
        self.resident: frozenset[str] = frozenset()
        self._acquired = False
        self._hb_stop: threading.Event | None = None
        self._hb_thread: threading.Thread | None = None

    def _do_load_siglip(self) -> None:
        if self._load_siglip is None:
            raise RuntimeError("ModelCoordinator.load_siglip must be injected")
        self._load_siglip()

    def _resolve(self, member: str) -> tuple[Callable[[], None], Callable[[], None], int]:
        """Resolve a want-set member to `(load, release, footprint_mb)` — the one
        place that knows HOW each resource is loaded (design §8.1), so `_reconcile`
        and `_release` stay generic. `SIGLIP` is in-process torch; `CAPTIONER` is
        the injected adapter (Ollama warm/evict on mac/cloud, in-process VLM
        load/release on jetson) — its footprint is asked of the adapter, not a
        static table, since it varies per profile; anything else is a plain LLM
        tag warmed/evicted on the shared inference client."""
        if member == wl.SIGLIP:
            return self._do_load_siglip, self._release_siglip, wl.FOOTPRINT_MB[wl.SIGLIP]
        if member == wl.CAPTIONER:
            if self.captioner is None:
                raise RuntimeError("ModelCoordinator: INGEST_CAPTION needs an injected captioner")
            return self.captioner.load, self.captioner.release, self.captioner.footprint_mb()
        return (
            lambda: self._client.warm(member),
            lambda: self._client.evict(member),
            wl.FOOTPRINT_MB.get(member, wl._FALLBACK_LLM_MB),
        )

    @contextlib.contextmanager
    def require(self, workload: WorkloadName):
        want = wl.model_set(workload, planner_model=self._planner_model, caption_model=self._caption_model)
        # RAM guard sums each member's RESOLVED footprint (not the static table),
        # so CAPTIONER's adapter-reported footprint counts (design §8.1).
        need_mb = sum(self._resolve(m)[2] for m in want)
        if need_mb > self._budget_mb:
            logger.warning(
                "refusing %s: needs %dMB > budget %dMB: %s",
                workload, need_mb, self._budget_mb, sorted(want),
            )
            raise RefusedError(
                f"{workload} needs {need_mb}MB > budget {self._budget_mb}MB: {sorted(want)}"
            )
        self._acquired = False
        self._acquire(workload)
        try:
            if self._acquired:
                self._start_heartbeat()   # prove liveness before the (slow) reconcile
                self._reconcile(want)
            yield
        finally:
            if self._acquired:
                self._stop_heartbeat()
                self._release(want)

    def _acquire(self, workload: WorkloadName) -> None:
        priority = wl.PRIORITY[workload]
        if ls.try_acquire(self._conn, self._holder, workload, priority):
            self._acquired = True
            return
        # Reclaim a DEAD holder's lease before anything else (design §8.1): a
        # holder that crashed / was killed / wedged mid-caption (incl. this same
        # holder before a restart) leaves its row with a frozen heartbeat. Reclaim
        # it FIRST — so a restarted worker never piggybacks on its own dead lease
        # and wedges chat — then acquire cleanly.
        if self._reclaim_and_acquire(workload, priority):
            return
        lease = ls.read_lease(self._conn)
        if lease is not None and lease["holder"] == self._holder:
            # Live same-process piggyback (design §8.1 (a)): CHAT and MEMORY_REBUILD
            # declare the same model-set, so if this holder already holds a FRESH
            # lease (not reclaimed above) the models we need are already resident.
            # Don't preempt, don't wait, don't take a second lease.
            self._acquired = False
            return
        if workload not in INTERACTIVE:
            # Background yields immediately (design §8.1 (b)) — no spin.
            raise LeaseBusyError(f"{workload}: lease held by interactive work; deferring")
        # Interactive, different holder: ask them to yield and wait (bounded). A
        # holder that dies mid-wait goes stale and is reclaimed inside the loop.
        deadline = self._now() + _PREEMPT_TIMEOUT_S
        ls.request_preempt(self._conn)
        while self._now() < deadline:
            self._sleep(_PREEMPT_POLL_S)
            if ls.try_acquire(self._conn, self._holder, workload, priority):
                self._acquired = True
                return
            if self._reclaim_and_acquire(workload, priority):
                return
            ls.request_preempt(self._conn)
        raise TimeoutError(f"could not acquire model lease for {workload} within {_PREEMPT_TIMEOUT_S}s")

    def _reclaim_and_acquire(self, workload: WorkloadName, priority: int) -> bool:
        """Reclaim a stale (dead-holder) lease and take it. Returns True iff we now
        hold the lease as a result. A fresh lease is never reclaimed."""
        if ls.reclaim_stale(self._conn, self._lease_stale_s) and ls.try_acquire(
            self._conn, self._holder, workload, priority
        ):
            self._acquired = True
            return True
        return False

    def _start_heartbeat(self) -> None:
        if self._heartbeat_connect is None:
            return
        self._hb_stop = threading.Event()

        def _beat() -> None:
            conn = self._heartbeat_connect()
            try:
                while not self._hb_stop.wait(self._heartbeat_interval_s):
                    try:
                        ls.heartbeat(conn, self._holder)
                    except Exception:
                        logger.exception("lease heartbeat failed for %s", self._holder)
            finally:
                with contextlib.suppress(Exception):
                    conn.close()

        self._hb_thread = threading.Thread(
            target=_beat, name=f"lease-heartbeat-{self._holder}", daemon=True
        )
        self._hb_thread.start()

    def _stop_heartbeat(self) -> None:
        if self._hb_stop is not None:
            self._hb_stop.set()
            if self._hb_thread is not None:
                self._hb_thread.join(timeout=1.0)
            self._hb_stop = None
            self._hb_thread = None

    def _reconcile(self, want: frozenset[str]) -> None:
        # Evict/release whatever we no longer want, via each member's resolved release.
        for model in self.resident - want:
            _, release, _ = self._resolve(model)
            release()
        # Load whatever we now want, via each member's resolved load.
        for model in want - self.resident:
            load, _, _ = self._resolve(model)
            load()
        self.resident = want

    def _release(self, want: frozenset[str]) -> None:
        # Release is unconditional (design §8.1 (c)): a failed unload is logged
        # but never strands the lease and wedges every future acquire.
        for model in want:
            try:
                _, release, _ = self._resolve(model)
                release()
            except Exception:
                logger.exception("unload failed for %s", model)
        self.resident = frozenset()
        ls.release(self._conn, self._holder)
