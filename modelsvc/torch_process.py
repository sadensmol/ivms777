"""A model hosted in a child process the registry can KILL.

Freeing a torch model in-process does not return its memory: on the Jetson the CUDA
driver keeps ~2.7 GB of the process's anonymous RSS after `empty_cache()` (measured,
design §8.1), which is exactly the RAM gemma then cannot have. Ending the process is
the only operation that gives it back, so SigLIP and nomic live in a supervised child
— the same shape `llama-server` already has (`modelsvc/llm_process.py`) — and the
registry's `free` is `terminate()`.

`spawn`, never `fork`: a forked child inherits the parent's CUDA state, which would
defeat the point. The parent consequently never imports torch at all.
"""

from __future__ import annotations

import multiprocessing as mp
import threading
from importlib import import_module
from typing import Any

_CTX = mp.get_context("spawn")


def _resolve(target: str):
    module_path, _, attr = target.partition(":")
    return getattr(import_module(module_path), attr)


def _child_main(conn, target: str, args: tuple, warm: str | None) -> None:
    """Build the hosted object, then answer `(method, args)` messages forever."""
    try:
        obj = _resolve(target)(*args)
        if warm is not None:
            getattr(obj, warm)()
    except BaseException as exc:  # noqa: BLE001 - report the failure, then exit
        conn.send(("err", f"{type(exc).__name__}: {exc}"))
        return
    conn.send(("ok", None))  # ready handshake
    while True:
        try:
            message = conn.recv()
        except EOFError:
            return
        if message is None:
            return
        method, call_args = message
        try:
            conn.send(("ok", getattr(obj, method)(*call_args)))
        except BaseException as exc:  # noqa: BLE001 - a bad call must not kill the host
            conn.send(("err", f"{type(exc).__name__}: {exc}"))


class TorchWorker:
    """One model in one child. `start`/`stop` are the registry's load/free, and
    `is_alive` is its liveness probe — a child that dies on its own (an OOM kill)
    is then respawned instead of being believed resident forever (§8.1)."""

    def __init__(
        self,
        target: str,
        args: tuple = (),
        *,
        warm: str | None = None,
        ready_timeout_s: float = 300.0,
    ) -> None:
        self._target = target
        self._args = args
        self._warm = warm
        self._ready_timeout = ready_timeout_s
        self._proc = None
        self._conn = None
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            if self._proc is not None and self._proc.is_alive():
                return
            self._kill()
            parent, child = _CTX.Pipe()
            proc = _CTX.Process(
                target=_child_main,
                args=(child, self._target, self._args, self._warm),
                daemon=True,
            )
            proc.start()
            child.close()  # drop the parent's copy of the child end, or EOF never fires
            self._proc, self._conn = proc, parent
            if not parent.poll(self._ready_timeout):
                self._kill()
                raise TimeoutError(f"{self._target} not ready in {self._ready_timeout}s")
            kind, payload = parent.recv()
            if kind != "ok":
                self._kill()
                raise RuntimeError(f"{self._target} failed to load: {payload}")

    def call(self, method: str, *args: Any) -> Any:
        with self._lock:
            if self._conn is None or self._proc is None or not self._proc.is_alive():
                raise RuntimeError(f"{self._target} worker is not running")
            try:
                self._conn.send((method, args))
                kind, payload = self._conn.recv()
            except (EOFError, BrokenPipeError, ConnectionResetError):
                # The child died mid-call — an OOM abort looks exactly like this.
                self._kill()
                raise RuntimeError(f"{self._target} worker died during {method}") from None
            if kind != "ok":
                raise RuntimeError(payload)
            return payload

    def stop(self) -> None:
        with self._lock:
            self._kill()

    def is_alive(self) -> bool:
        proc = self._proc
        return proc is not None and proc.is_alive()

    def _kill(self) -> None:
        proc, conn, self._proc, self._conn = self._proc, self._conn, None, None
        if conn is not None:
            conn.close()
        if proc is None:
            return
        proc.terminate()
        proc.join(10)
        if proc.is_alive():
            proc.kill()
            proc.join(5)
