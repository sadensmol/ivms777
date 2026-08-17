"""Best-effort GPU-utilization read for the resource bar (design §13).

The `models` service is the ONLY process with GPU access (design §5.1, the
one-model-process rule), so GPU load is read HERE and nowhere else — `app`/
`worker` stay thin and never touch the device. The source is per profile and
dependency-free (no pynvml, just the tool each box already ships):

- jetson — the kernel's own counter, `/sys/devices/platform/gpu.0/load` — the same
           number `tegrastats` prints as `GR3D_FREQ`, in PER MILLE (0-1000, so
           995 = 99.5%). It is read from sysfs and NOT from `tegrastats`, because
           the `models` container ships no `tegrastats` (it is a host L4T tool, and
           it wants sudo); the host `/sys` IS visible in the container, verified on
           the board. `tegrastats` stays as a fallback for a host-native run.
- cloud  — a discrete NVIDIA GPU answers `nvidia-smi` (a Jetson has none).
- mac    — `ioreg` reads the Metal driver's `Device Utilization %` out of the
           IORegistry (no sudo, unlike `powermetrics`). This works because
           `make up` runs the models service NATIVELY on macOS; inside a
           container there is no IORegistry and the read returns `None`.

`read_gpu_pct()` returns a percentage 0-100, or `None` when no GPU can be read
(the bar then simply omits the GPU field).
"""
import glob
import re
import subprocess
from pathlib import Path

# The L4T GPU-load counter. `gpu.0` is the stable alias; the glob catches the
# device node it points at (`bus@0/17000000.gpu` on JetPack 7.2) if the alias is
# ever missing. Contents: an integer in per mille.
TEGRA_LOAD_PATHS = (
    "/sys/devices/platform/gpu.0/load",
    "/sys/devices/gpu.0/load",
    "/sys/devices/platform/*/*.gpu/load",
)


def read_gpu_pct() -> float | None:
    for reader in (_read_tegra_sysfs, _read_tegrastats, _read_nvidia_smi, _read_ioreg):
        pct = reader()
        if pct is not None:
            return pct
    return None


def _read_tegra_sysfs() -> float | None:
    for pattern in TEGRA_LOAD_PATHS:
        for path in sorted(glob.glob(pattern)):
            try:
                raw = Path(path).read_text().strip()
            except OSError:
                continue  # not a Tegra / the node vanished
            try:
                per_mille = int(raw)
            except ValueError:
                continue
            return min(per_mille / 10.0, 100.0)
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
            capture_output=True, text=True, timeout=1.5, check=False,
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


def _read_ioreg() -> float | None:
    # macOS: the Apple GPU driver publishes its busy percentage in the IORegistry
    # as `"Device Utilization %"=NN` on the AGXAccelerator node. `ioreg` ships with
    # the OS and needs no sudo (`powermetrics` does). Readable only when the models
    # service runs on the host — which `make up` does; a Linux container has no
    # IORegistry, so there the call fails and the bar drops the GPU field.
    try:
        out = subprocess.run(
            ["ioreg", "-r", "-d", "1", "-w", "0", "-c", "AGXAccelerator"],
            capture_output=True, text=True, timeout=1.5, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    match = re.search(r'"Device Utilization %"\s*=\s*(\d+)', out.stdout or "")
    return float(match.group(1)) if match else None
