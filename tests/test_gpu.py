# tests/test_gpu.py — the best-effort GPU-load reader (modelsvc/gpu.py, §13).
import subprocess

import modelsvc.gpu as gpu


class _FakePopen:
    def __init__(self, out: str) -> None:
        self._out = out

    def communicate(self, timeout=None):
        return self._out, ""

    def kill(self) -> None:
        pass


def _no_binary(*_args, **_kwargs):
    raise FileNotFoundError


def test_reads_jetson_load_from_tegrastats(monkeypatch):
    line = "RAM 3000/7620MB ... GR3D_FREQ 42% cpu@50C ..."
    monkeypatch.setattr(gpu.subprocess, "Popen", lambda *a, **k: _FakePopen(line))
    assert gpu.read_gpu_pct() == 42.0


def test_zero_load_is_reported_not_swallowed(monkeypatch):
    # 0% is a real reading; the reader must not treat it as "unavailable".
    monkeypatch.setattr(gpu.subprocess, "Popen", lambda *a, **k: _FakePopen("GR3D_FREQ 0%"))
    assert gpu.read_gpu_pct() == 0.0


def test_falls_back_to_nvidia_smi_when_not_a_jetson(monkeypatch):
    monkeypatch.setattr(gpu.subprocess, "Popen", _no_binary)
    done = subprocess.CompletedProcess(args=[], returncode=0, stdout="55\n", stderr="")
    monkeypatch.setattr(gpu.subprocess, "run", lambda *a, **k: done)
    assert gpu.read_gpu_pct() == 55.0


def test_returns_none_without_any_gpu_tool(monkeypatch):
    monkeypatch.setattr(gpu.subprocess, "Popen", _no_binary)
    monkeypatch.setattr(gpu.subprocess, "run", _no_binary)
    assert gpu.read_gpu_pct() is None
