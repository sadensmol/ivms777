"""Best-effort CPU/GPU temperature read for the resource bar (design §13).

Read in the `models` service alongside GPU load (`models/gpu.py`) and for the
same reason: it is the one process with device access, so `app`/`worker` stay
thin and never touch the board.

The source is the kernel's own thermal zones — `/sys/devices/virtual/thermal/
thermal_zone*/{type,temp}`, millidegrees Celsius. This is what `tegrastats`
prints as `cpu@…C` / `gpu@…C`, but sysfs is used directly because the `models`
container ships no `tegrastats` (a host L4T tool that wants sudo). The host
`/sys` IS visible in the container — verified on the board with a bare
`docker run alpine cat …`, no extra mount.

The Orin Nano exposes `cpu-thermal`, `gpu-thermal`, `tj-thermal` (junction — the
hottest point, and the one throttling is keyed to) plus soc/cv zones. The cv
zones read empty on this board, so an unreadable or non-numeric zone is skipped
rather than treated as 0 °C — a bogus 0 in the bar is worse than no reading.

`read_temps()` returns only the keys it could actually read, so off-Tegra (mac,
cloud) it returns `{}` and the bar simply omits the field, exactly as it already
does for `gpu_pct`.
"""
import glob
from pathlib import Path

# Zone `type` -> the key the resource bar uses. Anything else (soc*, cv*) is
# deliberately not surfaced: two numbers the user can act on, not nine.
_ZONE_KEYS = {
    "cpu-thermal": "cpu_c",
    "gpu-thermal": "gpu_c",
    "tj-thermal": "tj_c",
}

_ZONE_GLOB = "/sys/devices/virtual/thermal/thermal_zone*"


def read_temps() -> dict[str, float]:
    """Return {cpu_c, gpu_c, tj_c} in °C for whichever zones are readable."""
    out: dict[str, float] = {}
    for zone in sorted(glob.glob(_ZONE_GLOB)):
        try:
            kind = Path(zone, "type").read_text().strip()
        except OSError:
            continue  # not a Tegra / the node vanished
        key = _ZONE_KEYS.get(kind)
        if key is None:
            continue
        try:
            millidegrees = int(Path(zone, "temp").read_text().strip())
        except (OSError, ValueError):
            continue  # empty (the cv zones) or unreadable — skip, never report 0
        out[key] = millidegrees / 1000.0
    return out
