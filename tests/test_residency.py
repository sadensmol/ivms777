"""`modelsvc.residency.Residency` (design §8.1) — the in-process ensure-loaded
manager. Since plan 16 removed the in-process caption VLM, SigLIP is the only
heavy in-process model, so residency just loads a model once and reports what is
resident. Fake load/release callables only: this module must never load a real
model.
"""

import threading

from modelsvc.residency import HIGH, LOW, Residency

_JOIN_TIMEOUT_S = 5.0


def _fake_models() -> tuple[Residency, list[str]]:
    residency = Residency()
    calls: list[str] = []
    residency.register("siglip", lambda: calls.append("siglip.load"), lambda: calls.append("siglip.release"))
    residency.register("caption", lambda: calls.append("caption.load"), lambda: calls.append("caption.release"))
    return residency, calls


def test_use_loads_the_named_model_and_marks_it_resident():
    residency, calls = _fake_models()

    with residency.use("siglip", priority=HIGH):
        assert residency.resident() == ["siglip"]

    assert calls == ["siglip.load"]
    assert residency.resident() == ["siglip"]  # stays loaded after the holder exits


def test_a_model_loads_only_once_across_calls():
    residency, calls = _fake_models()

    with residency.use("siglip", priority=HIGH):
        pass
    with residency.use("siglip", priority=HIGH):
        pass

    assert calls == ["siglip.load"]  # loaded once, not once per holder


def test_models_are_never_evicted():
    residency, calls = _fake_models()

    with residency.use("caption", priority=LOW):
        pass
    with residency.use("siglip", priority=HIGH):
        pass

    # No release() ever runs — both models stay resident (only SigLIP is heavy
    # and in-process now; there is nothing to swap against).
    assert calls == ["caption.load", "siglip.load"]
    assert set(residency.resident()) == {"caption", "siglip"}


def test_concurrent_use_of_the_same_model_loads_once_and_does_not_block():
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

    assert calls == ["siglip.load"]


def test_priority_constants_rank_embeds_above_captions():
    assert HIGH > LOW
