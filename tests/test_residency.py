"""`modelsvc.residency.Residency` (design §8.1, plan 15 task 5) — the in-process
successor to the cross-process `ModelCoordinator` (`models/coordinator.py`, plan
13, untouched here). Fake load/release callables only: this module must never
load a real model.
"""

import threading
import time

from modelsvc.residency import HIGH, LOW, CaptionPreempted, Residency

_JOIN_TIMEOUT_S = 5.0


def _fake_models() -> tuple[Residency, list[str]]:
    """An exclusive `Residency` with two registered fake models ("siglip",
    "caption") whose load/release calls are recorded, in order, in `calls`."""
    residency = Residency(exclusive=True)
    calls: list[str] = []
    residency.register("siglip", lambda: calls.append("siglip.load"), lambda: calls.append("siglip.release"))
    residency.register("caption", lambda: calls.append("caption.load"), lambda: calls.append("caption.release"))
    return residency, calls


def test_use_loads_the_named_model_and_marks_it_resident():
    residency, calls = _fake_models()

    with residency.use("caption", priority=LOW):
        assert residency.resident() == ["caption"]

    assert calls == ["caption.load"]
    assert residency.resident() == ["caption"]  # stays loaded after the holder exits


def test_sequential_use_of_a_different_model_evicts_the_previous_resident():
    residency, calls = _fake_models()

    with residency.use("caption", priority=LOW):
        pass
    with residency.use("siglip", priority=HIGH):
        pass

    # The other model's release() ran before the new model's load() — never both
    # resident at once.
    assert calls == ["caption.load", "caption.release", "siglip.load"]
    assert residency.resident() == ["siglip"]


def test_should_preempt_is_false_with_no_contention():
    residency, _ = _fake_models()

    assert residency.should_preempt() is False
    with residency.use("caption", priority=LOW):
        assert residency.should_preempt() is False


def test_concurrent_use_of_the_same_model_does_not_reload_or_block():
    residency, calls = _fake_models()
    entered = threading.Barrier(2, timeout=_JOIN_TIMEOUT_S)
    release_second = threading.Event()

    def hold():
        with residency.use("siglip", priority=HIGH):
            entered.wait()
            release_second.wait(timeout=_JOIN_TIMEOUT_S)

    t = threading.Thread(target=hold)
    t.start()
    with residency.use("siglip", priority=HIGH):
        entered.wait()  # both threads are inside use("siglip") concurrently
    release_second.set()
    t.join(timeout=_JOIN_TIMEOUT_S)
    assert not t.is_alive()

    assert calls == ["siglip.load"]  # loaded once, not once per holder


def test_a_caption_in_flight_yields_to_a_higher_priority_embed():
    """The core preempt-ordering assertion: an interactive embed (HIGH) preempts
    a background caption (LOW) in flight. Deterministic via threading.Events —
    the caption thread blocks in its (fake) body until it OBSERVES
    `should_preempt()` True, polling it; bounded timeouts are a safety net only,
    never the sync mechanism itself."""
    residency, calls = _fake_models()
    caption_started = threading.Event()
    saw_preempt = threading.Event()

    def run_caption():
        with residency.use("caption", priority=LOW):
            caption_started.set()
            deadline = time.monotonic() + _JOIN_TIMEOUT_S
            while not residency.should_preempt():
                if time.monotonic() > deadline:
                    return  # safety net only; the assertion below will fail
                time.sleep(0.005)
            saw_preempt.set()
            # simulate the captioner aborting generation and the body returning

    t = threading.Thread(target=run_caption)
    t.start()
    assert caption_started.wait(timeout=_JOIN_TIMEOUT_S)

    with residency.use("siglip", priority=HIGH):
        calls.append("siglip.active")

    t.join(timeout=_JOIN_TIMEOUT_S)
    assert not t.is_alive()
    assert saw_preempt.is_set()
    # release ran before the embed's load, and the embed body ran only after
    # the caption released — a caption in flight yields to an embed.
    assert calls == ["caption.load", "caption.release", "siglip.load", "siglip.active"]
    assert residency.resident() == ["siglip"]
    assert residency.should_preempt() is False  # cleared once the embed acquired


def test_a_same_model_joiner_does_not_clear_a_pending_preempt_flag():
    """Regression (review fix round 1): `_preempt` is only ever set by a
    waiting HIGH-priority caller right before it blocks in `cond.wait()`,
    under the same lock — so a reader that observes `should_preempt()` True
    is guaranteed that caller is ALREADY blocked (the only point in its
    critical section that releases the lock). That makes polling
    `should_preempt()` itself the deterministic sync point here; no
    sleep-as-sync needed.

    A second, SAME-model ("caption") joiner entering `use()` while the
    original caption is still active must NOT clear that flag on its way
    through — only the waiter that set it may clear it, once it stops
    waiting. Before the fix, the joiner fell straight through the (skipped)
    wait loop and unconditionally reset `_preempt = False`, masking the
    preempt from the still-in-flight original caption.
    """
    residency, calls = _fake_models()
    caption_started = threading.Event()
    release_caption = threading.Event()
    observed: dict[str, bool] = {}

    def run_caption():
        with residency.use("caption", priority=LOW):
            caption_started.set()
            release_caption.wait(timeout=_JOIN_TIMEOUT_S)
            # The joiner (below) has already run by the time we're released —
            # the flag it must not have cleared should still read True here.
            observed["preempt_after_joiner"] = residency.should_preempt()

    caption_thread = threading.Thread(target=run_caption)
    caption_thread.start()
    assert caption_started.wait(timeout=_JOIN_TIMEOUT_S)

    def run_siglip():
        with residency.use("siglip", priority=HIGH):
            calls.append("siglip.active")

    siglip_thread = threading.Thread(target=run_siglip)
    siglip_thread.start()

    # Deterministic sync point: `_preempt` only becomes True right before the
    # setter calls `cond.wait()`, inside the same locked section — so once we
    # observe it True, the siglip thread is guaranteed already blocked there.
    deadline = time.monotonic() + _JOIN_TIMEOUT_S
    while not residency.should_preempt():
        assert time.monotonic() < deadline, "siglip never set the preempt flag in time"

    # A second, same-model joiner enters and exits while the original caption
    # is still in flight and siglip is still waiting on it.
    with residency.use("caption", priority=LOW):
        pass

    # The flag must survive the joiner.
    assert residency.should_preempt() is True

    release_caption.set()
    caption_thread.join(timeout=_JOIN_TIMEOUT_S)
    siglip_thread.join(timeout=_JOIN_TIMEOUT_S)
    assert not caption_thread.is_alive()
    assert not siglip_thread.is_alive()
    assert observed["preempt_after_joiner"] is True  # the ORIGINAL caption saw it too
    assert residency.should_preempt() is False  # cleared once siglip actually acquired
    assert residency.resident() == ["siglip"]


def test_non_exclusive_mode_never_evicts_or_preempts():
    residency = Residency(exclusive=False)
    calls: list[str] = []
    residency.register("siglip", lambda: calls.append("siglip.load"), lambda: calls.append("siglip.release"))
    residency.register("caption", lambda: calls.append("caption.load"), lambda: calls.append("caption.release"))

    with residency.use("caption", priority=LOW):
        assert residency.should_preempt() is False
    with residency.use("siglip", priority=HIGH):
        assert residency.should_preempt() is False

    assert calls == ["caption.load", "siglip.load"]  # no release calls at all
    assert set(residency.resident()) == {"caption", "siglip"}  # both stay loaded


def test_non_exclusive_mode_loads_a_model_only_once_across_calls():
    residency = Residency(exclusive=False)
    calls: list[str] = []
    residency.register("siglip", lambda: calls.append("siglip.load"), lambda: calls.append("siglip.release"))

    with residency.use("siglip", priority=HIGH):
        pass
    with residency.use("siglip", priority=HIGH):
        pass

    assert calls == ["siglip.load"]


def test_caption_preempted_is_a_distinct_exception_type():
    assert issubclass(CaptionPreempted, Exception)


def test_priority_constants_rank_embeds_above_captions():
    assert HIGH > LOW
