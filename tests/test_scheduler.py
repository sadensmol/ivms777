import threading
import time

from modelsvc.scheduler import Priority, Scheduler


class _StubGov:
    def __init__(self):
        self.acquired = []

    def acquire(self, needed, *, pinned=frozenset()):
        self.acquired.append(list(needed))


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
            live.append(1)
            peak[0] = max(peak[0], len(live))
        time.sleep(0.05)
        with lock:
            live.pop()

    threads = [threading.Thread(target=lambda: s.run(["m"], Priority.BATCH, body)) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert peak[0] == 1  # never more than 1 at once


def test_interactive_preempts_batch_in_queue():
    gov = _StubGov()
    s = Scheduler(gov, concurrency=1)
    order = []
    started = threading.Event()

    def hog():
        started.set()
        time.sleep(0.1)
        order.append("hog")

    hog_t = threading.Thread(target=lambda: s.run(["m"], Priority.BATCH, hog))
    hog_t.start()
    started.wait()
    # queue a BATCH then an INTERACTIVE while the hog holds the only slot
    b = threading.Thread(target=lambda: s.run(["m"], Priority.BATCH, lambda: order.append("batch")))
    i = threading.Thread(target=lambda: s.run(["m"], Priority.INTERACTIVE, lambda: order.append("inter")))
    b.start()
    time.sleep(0.01)
    i.start()
    for t in (hog_t, b, i):
        t.join()
    assert order == ["hog", "inter", "batch"]  # interactive jumps the queued batch


def test_idle_unload_frees_gemma_after_ttl():
    from modelsvc.governor import MemoryGovernor
    from modelsvc.registry import ModelRegistry, ModelSpec

    log = []
    r = ModelRegistry()
    r.register(ModelSpec("gemma", lambda: log.append("load"), lambda: log.append("free"), 2200))
    gov = MemoryGovernor(r, measure_free_mb=lambda: 9999.0, budget_mb=6000)
    now = [1000.0]
    s = Scheduler(gov, concurrency=1, idle_ttl_s=120, clock=lambda: now[0])
    s.run(["gemma"], Priority.BATCH, lambda: None)  # loads + stamps last-use
    now[0] += 200  # exceed TTL
    s.reap_idle(["gemma"])
    assert log == ["load", "free"]


def test_reap_idle_is_noop_without_ttl():
    from modelsvc.governor import MemoryGovernor
    from modelsvc.registry import ModelRegistry, ModelSpec

    log = []
    r = ModelRegistry()
    r.register(ModelSpec("gemma", lambda: log.append("load"), lambda: log.append("free"), 2200))
    gov = MemoryGovernor(r, measure_free_mb=lambda: 9999.0, budget_mb=6000)
    s = Scheduler(gov, concurrency=1)  # no idle_ttl_s
    s.run(["gemma"], Priority.BATCH, lambda: None)
    s.reap_idle(["gemma"])
    assert log == ["load"]  # never unloaded
