"""Best-effort GPU-utilization read for the resource bar (design §13).

The `models` service is the ONLY process with GPU access (design §5.1, the
one-model-process rule), so GPU load is read HERE and nowhere else — `app`/
`worker` stay thin and never touch the device. The source is per profile and
dependency-free (no pynvml, just the tool each box already ships):

- jetson — the Tegra CLI `tegrastats` reports the 3D engine as `GR3D_FREQ NN%`.
- cloud  — a discrete NVIDIA GPU answers `nvidia-smi`.
- mac    — the models container has no GPU and Ollama exposes no utilization
           number, so there is nothing to read; returns `None`.

`read_gpu_pct()` returns a percentage 0-100, or `None` when no GPU can be read
(the bar then simply omits the GPU field).
"""
import re
import subprocess


def read_gpu_pct() -> float | None:
    for reader in (_read_tegrastats, _read_nvidia_smi):
        pct = reader()
        if pct is not None:
            return pct
    return None


def _read_tegrastats() -> float | None:
    # tegrastats streams one sample line per `--interval` ms and never exits, so
    # bound it: grab the first line's worth of output, then kill it. Total time
    # is capped at ~1.5 s so a stuck CLI can never hang the resource poll.
    try:
        proc = subprocess.Popen(
            ["tegrastats", "--interval", "500"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
    except OSError:
        return None  # not a Jetson / tegrastats not installed
    try:
        out, _ = proc.communicate(timeout=1.5)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, _ = proc.communicate()
    match = re.search(r"GR3D_FREQ\s+(\d+)%", out or "")
    return float(match.group(1)) if match else None


def _read_nvidia_smi() -> float | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=1.5,
        )
    except (OSError, subprocess.SubprocessError):
        return None  # no discrete NVIDIA GPU / driver
    if out.returncode != 0:
        return None
    lines = out.stdout.strip().splitlines()
    if not lines:
        return None
    try:
        return float(lines[0].strip())
    except ValueError:
        return None
