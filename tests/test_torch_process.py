"""`TorchWorker` — the killable child that hosts a torch model (plan 20, §8.1).

Real processes, no mocks: the whole point of the class is that `free()` ENDS a
process, so the tests assert on process identity and liveness.
"""

import os

import pytest

from modelsvc.torch_process import TorchWorker


def _worker(**kw):
    return TorchWorker("tests.worker_fixtures:Doubler", (1,), **kw)


def test_calls_run_in_a_separate_process():
    w = _worker()
    w.start()
    try:
        assert w.call("double", [1, 2]) == [3, 5]
        assert w.call("pid") != os.getpid()
    finally:
        w.stop()


def test_stop_ends_the_child_and_is_alive_reports_it():
    w = _worker()
    w.start()
    assert w.is_alive()
    w.stop()
    assert not w.is_alive()


def test_start_is_idempotent_while_alive():
    w = _worker()
    w.start()
    try:
        first = w.call("pid")
        w.start()
        assert w.call("pid") == first
    finally:
        w.stop()


def test_warm_runs_at_start_when_asked():
    w = _worker(warm="warm")
    w.start()
    try:
        assert w.call("warmed") is True
    finally:
        w.stop()


def test_error_inside_the_child_surfaces_as_runtime_error():
    w = _worker()
    w.start()
    try:
        with pytest.raises(RuntimeError, match="ValueError: nope"):
            w.call("boom")
        assert w.is_alive()  # a raising method must not kill the host
    finally:
        w.stop()


def test_a_child_that_dies_mid_call_raises_and_reports_dead():
    w = _worker()
    w.start()
    try:
        with pytest.raises(RuntimeError, match="died"):
            w.call("die")
        assert not w.is_alive()
    finally:
        w.stop()
