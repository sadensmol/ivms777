"""gemma's lifecycle, owned by the models service.

mac/jetson: ``SubprocessLlm`` spawns the local ``llama-server`` as a child process
(Metal binary on mac, sm_87 CUDA binary on jetson) and kills it to free ~2 GB --
this is how the governor "unloads" gemma. cloud: ``RemoteLlm`` points at a remote
vLLM and never supervises it. No ``llama_cpp`` import -- gemma stays a separate
process reached over localhost HTTP, so the one-model-process gate is untouched.
"""

from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Callable
from typing import Protocol

import httpx


def _http_probe(url: str) -> bool:
    try:
        return httpx.get(url, timeout=1.0).status_code == 200
    except httpx.HTTPError:
        return False


class LlmProcess(Protocol):
    def load(self, *, vision: bool = False) -> None: ...
    def free(self) -> None: ...
    def is_loaded(self) -> bool: ...


class SubprocessLlm:
    """One supervised ``llama-server`` child with TWO spawn modes (design §3.1).

    ``vision_args`` (the ``--mmproj`` pair) is appended only when the caller asks for
    the vision mode, so a text chat never loads the ~531 MB projector. Both modes are
    the same child on the same port, so they are mutually exclusive: asking for a mode
    that differs from the running one restarts the child.
    """

    def __init__(
        self,
        command: list[str],
        *,
        health_url: str,
        vision_args: list[str] | None = None,
        ready_timeout_s: float = 120.0,
        poll_interval_s: float = 0.5,
        env: dict[str, str] | None = None,
        probe: Callable[[str], bool] = _http_probe,
    ) -> None:
        self._cmd = command
        self._vision_args = list(vision_args or [])
        self._health = health_url
        self._timeout = ready_timeout_s
        self._poll = poll_interval_s
        self._env = env
        self._probe = probe
        self._proc: subprocess.Popen | None = None
        self._vision = False  # mode of the RUNNING child

    def command_for(self, *, vision: bool) -> list[str]:
        return [*self._cmd, *self._vision_args] if vision else list(self._cmd)

    def load(self, *, vision: bool = False) -> None:
        if self._proc is not None:
            if self._proc.poll() is None:
                if self._vision == vision:
                    return  # already running in the mode we need
                # Wrong mode (text vs vision): one child, one port — restart it.
                self.free()
            else:
                # A previous child died (e.g. an earlier OOM at model load). Reap it
                # so it does not linger as a <defunct> zombie before we respawn.
                self._proc.wait()
                self._proc = None
        # scoped env so llama-server's CUDA libs never clash with torch's in-process
        run_env = {**os.environ, **(self._env or {})}
        self._vision = vision
        self._proc = subprocess.Popen(self.command_for(vision=vision), env=run_env)
        deadline = time.monotonic() + self._timeout
        while time.monotonic() < deadline:
            if self._probe(self._health):
                return
            if self._proc.poll() is not None:
                rc = self._proc.wait()  # reap the crashed child (no zombie)
                self._proc = None
                raise RuntimeError(f"llama-server exited early: rc={rc}")
            time.sleep(self._poll)
        self.free()
        raise TimeoutError(f"llama-server not healthy within {self._timeout}s")

    def free(self) -> None:
        p, self._proc = self._proc, None
        if p is None:
            return
        p.terminate()
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait(timeout=5)

    def is_loaded(self) -> bool:
        return self._proc is not None and self._proc.poll() is None


class RemoteLlm:
    def __init__(self, *, health_url: str, probe: Callable[[str], bool] = _http_probe) -> None:
        self._health = health_url
        self._probe = probe

    def load(self, *, vision: bool = False) -> None:  # remote vLLM: nothing to supervise
        return None

    def free(self) -> None:
        return None

    def is_loaded(self) -> bool:
        return True


def build_llm_process(settings) -> LlmProcess:
    """Pick the gemma actuator for the profile: supervised child (mac/jetson) or
    remote (cloud). GGUF paths mirror ``scripts/llama-server-entrypoint.sh`` and
    the Makefile ``LLAMA_GGUF``/``LLAMA_MMPROJ`` defaults."""
    if not settings.llm_managed:
        return RemoteLlm(health_url=settings.llm_health_url)
    model_dir = settings.data_dir / "models"
    gguf = model_dir / "gemma-4-E2B-it-Q4_K_M.gguf"
    # The projector is NOT part of the base command: it is appended only for the
    # vision (captioning) mode, so a text chat never loads it (design §3.1).
    vision_args = ["--mmproj", str(model_dir / settings.mmproj_name)]
    cmd = [
        settings.llm_bin or "llama-server",
        "-m",
        str(gguf),
    ]
    # -ngl: pin a fixed GPU layer count, or (None) omit it so llama.cpp AUTO-FITS
    # layers to free GPU RAM and offloads the rest to CPU. On the 8 GB jetson gemma
    # (~5 GB) can exceed free RAM; forcing -ngl 99 makes llama.cpp abort at load
    # instead of degrading, killing the child (design §3.1/§8.1).
    if settings.llm_ngl is not None:
        cmd += ["-ngl", str(settings.llm_ngl)]
    cmd += [
        "--flash-attn",
        "on",
        "--jinja",
        "--chat-template-kwargs",
        '{"enable_thinking":false}',
        # KV-cache context. Smaller = less resident RAM (jetson: 2048 → ~0.5 GB
        # less than 4096) — chat turns are short, so this rarely truncates (§3.1).
        "-c",
        str(settings.llm_ctx),
        "--host",
        "0.0.0.0",
        "--port",
        str(settings.llm_port),
    ]
    # Child-only LD_LIBRARY_PATH so the cu130-built llama-server finds libggml-cuda
    # without shadowing the models process's own cu132 torch libs.
    ldpath = os.environ.get("IVMS777_LLM_LDPATH")
    env = {"LD_LIBRARY_PATH": ldpath} if ldpath else None
    return SubprocessLlm(
        cmd, health_url=settings.llm_health_url, vision_args=vision_args, env=env
    )
