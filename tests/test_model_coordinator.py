import pytest

from models import lease_store as ls
from models.coordinator import LeaseBusyError, ModelCoordinator, RefusedError


class FakeClient:
    def __init__(self): self.warmed, self.evicted = [], []
    def warm(self, model, *, timeout=120.0): self.warmed.append(model)
    def evict(self, model, *, timeout=30.0): self.evicted.append(model)


class FakeCaptioner:
    """Stand-in for the INGEST_CAPTION resource's adapter (design §4/§8.1) — an
    in-process VLM shape (jetson): `load`/`release` touch no client at all."""
    name = "cap"
    caption_model = "cap"

    def __init__(self):
        self.loaded = 0
        self.released = 0

    def load(self): self.loaded += 1
    def release(self): self.released += 1
    def footprint_mb(self): return 2700


def _coord(conn, client, holder="app", budget=99_999, loaded=None, captioner=None):
    return ModelCoordinator(
        conn, client, holder=holder, budget_mb=budget,
        planner_model="qwen2.5:3b", caption_model="qwen2.5vl:3b",
        load_siglip=lambda: loaded.append("siglip") if loaded is not None else None,
        release_siglip=lambda: loaded.append("release") if loaded is not None else None,
        captioner=captioner,
        sleep=lambda s: None,
    )


def test_require_loads_declared_set_and_evicts_the_rest(conn):
    client, loaded = FakeClient(), []
    coord = _coord(conn, client, loaded=loaded)
    with coord.require("CHAT"):
        assert "siglip" in loaded          # SigLIP loaded in-process
        assert client.warmed == ["qwen2.5:3b"]
        assert ls.read_lease(conn)["workload"] == "CHAT"
    assert ls.read_lease(conn) is None      # released on exit
    assert "qwen2.5:3b" in client.evicted   # LLM unloaded on exit
    assert "release" in loaded              # SigLIP released on exit


def test_over_budget_is_refused_not_loaded(conn):
    client = FakeClient()
    coord = _coord(conn, client, budget=100)   # nothing fits
    with pytest.raises(RefusedError), coord.require("CHAT"):
        pass
    assert client.warmed == []
    assert ls.read_lease(conn) is None


def test_interactive_preempts_a_background_holder(conn):
    client = FakeClient()
    # worker holds a background lease
    ls.try_acquire(conn, holder="worker", workload="INGEST_CAPTION", priority=1)
    coord = _coord(conn, client, holder="app")

    # app's CHAT request should flag preempt, then (once worker releases) acquire.
    calls = {"n": 0}
    def fake_sleep(_):
        calls["n"] += 1
        if calls["n"] == 1:
            assert ls.preempt_requested(conn) is True   # flag was set
            ls.release(conn, holder="worker")           # simulate worker yielding
    coord._sleep = fake_sleep

    with coord.require("CHAT"):
        assert ls.read_lease(conn)["holder"] == "app"


def test_interactive_waits_out_a_slow_caption_past_30s(conn):
    # A single mac caption (qwen2.5vl:7b) can run well past 30s; chat must keep
    # waiting for the worker to yield, not time out and falsely report "busy" (§8.1).
    client = FakeClient()
    ls.try_acquire(conn, holder="worker", workload="INGEST_CAPTION", priority=1)
    clock = {"t": 0.0}
    def now():
        return clock["t"]
    def sleep(_dt):
        clock["t"] += 1.0                      # each poll = 1 virtual second
        if clock["t"] >= 60.0:                 # worker yields only after a 60s caption
            ls.release(conn, holder="worker")
    coord = ModelCoordinator(
        conn, client, holder="app", budget_mb=99_999,
        planner_model="qwen2.5:3b", caption_model="qwen2.5vl:7b",
        load_siglip=lambda: None, release_siglip=lambda: None, sleep=sleep, now=now,
    )

    with coord.require("CHAT"):               # must NOT raise TimeoutError at t=30
        assert ls.read_lease(conn)["holder"] == "app"
    assert clock["t"] >= 60.0                  # it really waited past the old 30s cap


def test_background_yields_immediately_when_busy(conn):
    client = FakeClient()
    # app holds an interactive lease
    ls.try_acquire(conn, holder="app", workload="CHAT", priority=10)
    coord = _coord(conn, client, holder="worker")

    with pytest.raises(LeaseBusyError), coord.require("INGEST_EMBED"):
        pass

    assert ls.preempt_requested(conn) is False   # no preempt flagged
    assert client.warmed == []                    # never reconciled
    lease = ls.read_lease(conn)
    assert lease is not None and lease["holder"] == "app"   # app's lease intact


def test_same_holder_piggybacks(conn):
    ls.try_acquire(conn, holder="app", workload="CHAT", priority=10)
    client = FakeClient()
    coord = _coord(conn, client, holder="app")

    with coord.require("MEMORY_REBUILD"):
        pass

    assert client.warmed == []   # never reconciled — models already resident
    lease = ls.read_lease(conn)
    assert lease is not None and lease["holder"] == "app"   # original lease untouched


def _make_stale(conn):
    conn.execute("UPDATE model_lease SET heartbeat = datetime('now', '-300 seconds') WHERE id = 1")


def test_interactive_reclaims_a_dead_holders_stale_lease(conn):
    # A worker that crashed/was killed mid-caption leaves its lease row with a
    # frozen heartbeat. Chat must reclaim it, not wait out the whole preempt window
    # then report "busy" (design §8.1) — the wedge behind the reported hang.
    client = FakeClient()
    ls.try_acquire(conn, holder="worker", workload="INGEST_CAPTION", priority=1)
    _make_stale(conn)
    coord = _coord(conn, client, holder="app")
    with coord.require("CHAT"):
        assert ls.read_lease(conn)["holder"] == "app"
    assert ls.read_lease(conn) is None


def test_restarted_worker_reclaims_its_own_stale_lease_not_piggyback(conn):
    # After a restart the worker is holder "worker" again; its dead-self lease must
    # be RECLAIMED (real acquire + release), never piggybacked on — piggybacking on
    # a dead lease never releases and wedges chat forever (the observed wedge).
    client = FakeClient()
    ls.try_acquire(conn, holder="worker", workload="INGEST_CAPTION", priority=1)
    _make_stale(conn)
    coord = _coord(conn, client, holder="worker", captioner=FakeCaptioner())
    with coord.require("INGEST_CAPTION"):
        assert ls.read_lease(conn)["holder"] == "worker"
    assert ls.read_lease(conn) is None   # a real hold → released on exit


def test_ingest_caption_loads_and_releases_the_captioner(conn):
    # Generalized residency (design §4/§8.1): INGEST_CAPTION loads/releases the
    # INJECTED captioner adapter, not a raw LLM tag on the inference client — an
    # in-process (jetson-shaped) captioner never touches `client` at all.
    client = FakeClient()
    cap = FakeCaptioner()
    coord = ModelCoordinator(
        conn, client, holder="worker", budget_mb=99_999,
        planner_model="qwen2.5:3b", caption_model="qwen2.5vl:3b",
        load_siglip=lambda: None, release_siglip=lambda: None,
        captioner=cap, sleep=lambda s: None,
    )
    with coord.require("INGEST_CAPTION"):
        assert cap.loaded == 1
        assert client.warmed == []           # in-process path: NO ollama warm
    assert cap.released == 1
    assert client.evicted == []


def test_ingest_caption_with_ollama_captioner_warms_and_evicts_the_model(conn):
    # Mac-unchanged proof: an OllamaCaptioner adapter over a FakeClient still
    # warms/evicts the ollama tag on INGEST_CAPTION — same net effect as the
    # pre-adapter coordinator that called `client.warm`/`client.evict` directly.
    from captioning.ollama_adapter import OllamaCaptioner

    client = FakeClient()
    cap = OllamaCaptioner(client, "qwen2.5vl:3b")
    coord = ModelCoordinator(
        conn, client, holder="worker", budget_mb=99_999,
        planner_model="qwen2.5:3b", caption_model="qwen2.5vl:3b",
        load_siglip=lambda: None, release_siglip=lambda: None,
        captioner=cap, sleep=lambda s: None,
    )
    with coord.require("INGEST_CAPTION"):
        assert client.warmed == ["qwen2.5vl:3b"]
    assert client.evicted == ["qwen2.5vl:3b"]


def test_ingest_caption_without_an_injected_captioner_raises(conn):
    # A coordinator asked for INGEST_CAPTION with no captioner injected is a
    # config bug, not a silent no-op — it must fail loudly (design §8.1).
    client = FakeClient()
    coord = _coord(conn, client, holder="worker")   # captioner=None (default)
    with pytest.raises(RuntimeError), coord.require("INGEST_CAPTION"):
        pass


def test_heartbeat_keeps_the_lease_fresh_while_held_then_stops(conn, monkeypatch):
    import time as _time

    import models.coordinator as mc

    beats = {"n": 0}
    monkeypatch.setattr(mc.ls, "heartbeat", lambda _c, _h: beats.__setitem__("n", beats["n"] + 1))

    class _DummyConn:
        def close(self):
            pass

    client = FakeClient()
    coord = ModelCoordinator(
        conn, client, holder="app", budget_mb=99_999,
        planner_model="qwen2.5:3b", caption_model="qwen2.5vl:3b",
        load_siglip=lambda: None, release_siglip=lambda: None, sleep=lambda s: None,
        heartbeat_connect=lambda: _DummyConn(), heartbeat_interval_s=0.02,
    )
    with coord.require("CHAT"):
        _time.sleep(0.15)            # ~7 intervals while held
        assert beats["n"] >= 1       # the holder is bumping its heartbeat
    settled = beats["n"]
    _time.sleep(0.1)
    assert beats["n"] == settled     # thread stopped on release — no more beats
