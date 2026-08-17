# tests/test_gpu.py — the best-effort GPU-load reader (models/gpu.py, §13).
import subprocess

import pytest

from models import gpu


@pytest.fixture(autouse=True)
def _no_tegra_sysfs(monkeypatch):
    # sysfs is probed first; blank it so the CLI fallbacks are what each test
    # exercises (and so the suite reads the same on a real Jetson).
    monkeypatch.setattr(gpu, "TEGRA_LOAD_PATHS", ())


def test_reads_jetson_load_from_sysfs_in_per_mille(monkeypatch, tmp_path):
    # /sys/devices/platform/gpu.0/load is per mille — 995 is 99.5%, not 995%.
    node = tmp_path / "load"
    node.write_text("995\n")
    monkeypatch.setattr(gpu, "TEGRA_LOAD_PATHS", (str(node),))
    assert gpu.read_gpu_pct() == 99.5


def test_sysfs_glob_finds_the_device_node(monkeypatch, tmp_path):
    node = tmp_path / "17000000.gpu" / "load"
    node.parent.mkdir()
    node.write_text("0\n")  # idle is a reading, not "unavailable"
    monkeypatch.setattr(gpu, "TEGRA_LOAD_PATHS", (str(tmp_path / "*.gpu" / "load"),))
    assert gpu.read_gpu_pct() == 0.0


def test_falls_through_when_sysfs_node_is_absent(monkeypatch, tmp_path):
    monkeypatch.setattr(gpu, "TEGRA_LOAD_PATHS", (str(tmp_path / "missing" / "load"),))
    monkeypatch.setattr(gpu.subprocess, "Popen", lambda *a, **k: _FakePopen("GR3D_FREQ 12%"))
    assert gpu.read_gpu_pct() == 12.0


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


def test_falls_back_to_ioreg_on_mac(monkeypatch):
    # No tegrastats, no NVIDIA driver — macOS answers from the IORegistry instead.
    monkeypatch.setattr(gpu.subprocess, "Popen", _no_binary)
    ioreg = '  | {\n  |   "Device Utilization %"=37\n  |   "In use system memory"=123\n  | }\n'

    def _run(args, **_kwargs):
        if args[0] == "nvidia-smi":
            raise FileNotFoundError
        return subprocess.CompletedProcess(args=args, returncode=0, stdout=ioreg, stderr="")

    monkeypatch.setattr(gpu.subprocess, "run", _run)
    assert gpu.read_gpu_pct() == 37.0


def test_returns_none_without_any_gpu_tool(monkeypatch):
    monkeypatch.setattr(gpu.subprocess, "Popen", _no_binary)
    monkeypatch.setattr(gpu.subprocess, "run", _no_binary)
    assert gpu.read_gpu_pct() is None
