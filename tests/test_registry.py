import threading

import pytest

from modelsvc.registry import ModelRegistry, ModelSpec


def _spec(name, log, cost=100):
    return ModelSpec(
        name=name,
        load=lambda: log.append(("load", name)),
        free=lambda: log.append(("free", name)),
        cost_mb=cost,
    )


def test_ensure_loads_once():
    log = []
    r = ModelRegistry()
    r.register(_spec("siglip", log))
    r.ensure("siglip")
    r.ensure("siglip")
    assert log == [("load", "siglip")]
    assert r.is_resident("siglip")


def test_unload_calls_free_and_forgets():
    log = []
    r = ModelRegistry()
    r.register(_spec("siglip", log))
    r.ensure("siglip")
    r.unload("siglip")
    assert log == [("load", "siglip"), ("free", "siglip")]
    assert not r.is_resident("siglip")


def test_unload_absent_is_noop():
    log = []
    r = ModelRegistry()
    r.register(_spec("siglip", log))
    r.unload("siglip")
    assert log == []


def test_resident_is_lru_oldest_first():
    log = []
    r = ModelRegistry()
    for n in ("a", "b", "c"):
        r.register(_spec(n, log))
    r.ensure("a")
    r.ensure("b")
    r.ensure("c")
    r.touch("a")  # a becomes most-recent
    assert r.resident() == ["b", "c", "a"]


def test_ensure_reloads_a_model_that_died_out_of_band():
    # The supervised llama-server child can abort on its own (a CUDA OOM at image
    # decode). Bookkeeping alone then lies: the registry must ask the model whether
    # it is still alive and reload it, or every caption hits a dead port forever.
    log = []
    alive = {"v": True}
    r = ModelRegistry()
    r.register(
        ModelSpec(
            name="gemma",
            load=lambda: (log.append(("load", "gemma")), alive.__setitem__("v", True))[0],
            free=lambda: log.append(("free", "gemma")),
            cost_mb=100,
            alive=lambda: alive["v"],
        )
    )
    r.ensure("gemma")
    alive["v"] = False  # the child aborted; nobody told the registry
    r.ensure("gemma")
    assert log == [("load", "gemma"), ("load", "gemma")]
    assert r.is_resident("gemma")


def test_unknown_model_raises():
    r = ModelRegistry()
    with pytest.raises(KeyError):
        r.ensure("nope")


def test_cost_mb_reported():
    r = ModelRegistry()
    r.register(_spec("siglip", [], cost=1600))
    assert r.cost_mb("siglip") == 1600


def test_ensure_is_thread_safe():
    log = []
    r = ModelRegistry()
    r.register(_spec("siglip", log))
    barrier = threading.Barrier(8)

    def worker():
        barrier.wait()
        r.ensure("siglip")

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert log.count(("load", "siglip")) == 1
