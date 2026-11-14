import pytest

from modelsvc.governor import InsufficientMemory, MemoryGovernor, UnitLeased
from modelsvc.registry import ModelRegistry, ModelSpec


def _registry(log, costs):
    r = ModelRegistry()
    for name, cost in costs.items():
        r.register(
            ModelSpec(
                name,
                (lambda n=name: log.append(("load", n))),
                (lambda n=name: log.append(("free", n))),
                cost,
            )
        )
    return r


def test_acquire_loads_when_it_fits():
    log = []
    r = _registry(log, {"siglip": 1600})
    gov = MemoryGovernor(r, measure_free_mb=lambda: 4000.0, budget_mb=6000)
    gov.acquire(["siglip"])
    assert ("load", "siglip") in log
    assert r.is_resident("siglip")


def test_acquire_evicts_lru_to_fit_budget():
    log = []
    r = _registry(log, {"siglip": 1600, "gemma": 2200, "nomic": 300, "big": 3000})
    gov = MemoryGovernor(r, measure_free_mb=lambda: 10000.0, budget_mb=6000, headroom_mb=0)
    for n in ("nomic", "siglip", "gemma"):  # LRU order: nomic oldest
        gov.acquire([n])
    log.clear()
    gov.acquire(["big"])
    assert ("free", "nomic") in log
    assert ("free", "siglip") in log
    assert ("free", "gemma") not in log  # gemma newest, kept once it fits
    assert r.is_resident("big")


def test_acquire_never_evicts_needed_or_pinned():
    log = []
    r = _registry(log, {"siglip": 1600, "gemma": 2200})
    gov = MemoryGovernor(r, measure_free_mb=lambda: 0.0, budget_mb=3000, headroom_mb=0)
    gov.acquire(["gemma"])
    log.clear()
    # need siglip while gemma pinned; 1600+2200=3800 > 3000 and gemma pinned -> raise
    with pytest.raises(InsufficientMemory):
        gov.acquire(["siglip"], pinned=frozenset({"gemma"}))
    assert ("free", "gemma") not in log


def test_acquire_uses_measured_free_not_just_budget():
    # budget says 6000 but the box only reports 500 MB free -> must evict to fit reality.
    log = []
    r = _registry(log, {"gemma": 2200, "siglip": 1600})
    gov = MemoryGovernor(r, measure_free_mb=lambda: 500.0, budget_mb=6000, headroom_mb=0)
    gov.acquire(["gemma"])
    log.clear()
    gov.acquire(["siglip"])  # 500 free < 1600 -> evict gemma
    assert ("free", "gemma") in log
    assert r.is_resident("siglip")


def test_state_reports_resident_and_budget():
    r = _registry([], {"siglip": 1600})
    gov = MemoryGovernor(r, measure_free_mb=lambda: 4000.0, budget_mb=6000)
    gov.acquire(["siglip"])
    st = gov.state()
    assert st.resident == ["siglip"]
    assert st.used_mb == 1600
    assert st.budget_mb == 6000
    assert st.free_mb == 4000.0


def test_a_leased_unit_is_kept_while_others_are_evicted():
    log = []
    r = _registry(log, {"a": 1000, "b": 1000, "c": 1000})
    gov = MemoryGovernor(r, measure_free_mb=lambda: 10000.0, budget_mb=2000, headroom_mb=0)
    gov.acquire(["b"])  # b is the oldest resident
    gov.acquire(["a"])
    log.clear()
    with gov.lease("b"):
        gov.acquire(["c"])
    assert ("free", "a") in log
    assert ("free", "b") not in log
    assert r.is_resident("b") and r.is_resident("c")


def test_acquire_refuses_when_only_the_leased_unit_could_make_room():
    log = []
    r = _registry(log, {"b": 1500, "c": 1000})
    gov = MemoryGovernor(r, measure_free_mb=lambda: 10000.0, budget_mb=2000, headroom_mb=0)
    gov.acquire(["b"])
    log.clear()
    with gov.lease("b"), pytest.raises(InsufficientMemory):
        gov.acquire(["c"])
    assert ("free", "b") not in log
    assert r.is_resident("b")


def test_unloading_a_leased_unit_raises():
    r = _registry([], {"b": 1000})
    gov = MemoryGovernor(r, measure_free_mb=lambda: 10000.0, budget_mb=2000)
    gov.acquire(["b"])
    with gov.lease("b"), pytest.raises(UnitLeased):
        gov.unload("b")
    assert r.is_resident("b")


def test_a_released_lease_makes_the_unit_evictable_again():
    log = []
    r = _registry(log, {"b": 1500, "c": 1000})
    gov = MemoryGovernor(r, measure_free_mb=lambda: 10000.0, budget_mb=2000, headroom_mb=0)
    gov.acquire(["b"])
    with gov.lease("b"):
        pass
    gov.acquire(["c"])
    assert ("free", "b") in log
    assert r.is_resident("c")


def test_leased_names_the_held_leases():
    gov = MemoryGovernor(_registry([], {}), measure_free_mb=lambda: 0.0, budget_mb=0)
    assert gov.leased() == frozenset()
    with gov.lease("b"):
        assert gov.leased() == frozenset({"b"})
    assert gov.leased() == frozenset()
