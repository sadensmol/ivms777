"""Live resource snapshot for the top resource bar (design §13).

**The machine metrics are read HERE, in `app`, and never proxied.** RAM, CPU
load, GPU load and board temperatures are host-wide numbers that come from
`psutil` and the kernel's own sysfs counters — reading them needs no CUDA
context, no driver library and no `runtime: nvidia`, so it is *not* "touching the
GPU" in the sense of the one-model-process rule (§5.1). Verified on the board:
the thin `app` container reads `/sys/devices/platform/gpu.0/load` and the thermal
zones with no extra mount.

That means the bar **always** shows RAM/CPU/GPU/temperatures — including while
the `models` service is starting, restarting, or down. Only the two things that
service alone knows are fetched over HTTP: which models are resident, and the
op in flight. When it is unreachable those degrade to "no model info" and the
machine metrics are unaffected.

**RAM here is the WHOLE HOST, not any one process** — `psutil.virtual_memory()`
inside the container reads the host `/proc/meminfo` (memory is unified on the
Jetson, so a model's GPU pages count too). So the bar answers "how much room is
left on the board", which is the number that decides whether the next model fits —
it is NOT the resident model's footprint, and reading it as such makes an idle
SigLIP look like it costs 6.9 GB when the box total is 6.9 GB. The per-model cost
the governor budgets against is `IVMS777_MODEL_COST_MB` (§8.1), not this.
"""
import psutil

from models.gpu import read_gpu_pct
from models.thermal import read_temps


def display_names(resident, *, planner_model, caption_model, embed_model, text_embed_model):
    """Registry keys -> the FULL model names the bar shows (design §13).

    The conveyor's keys are internal handles (`gemma`, `gemma-vision`, `siglip`,
    `nomic`); the bar must name the actual model that is loaded, so "gemma" reads as
    `gemma4-E2B`. `gemma-vision` is the same weights plus the projector, so it is
    labelled with the caption model and marked as the vision mode (§3.1).
    """
    names = {
        "gemma": planner_model,
        "gemma-vision": f"{caption_model} +vision" if caption_model else None,
        "siglip": embed_model,
        "nomic": text_embed_model,
    }
    return [names.get(key) or key for key in resident]


def _machine() -> dict:
    """Host RAM/CPU/GPU/temperatures — always available, never proxied."""
    vm = psutil.virtual_memory()
    temps = read_temps()
    return {
        "ram_used_mb": (vm.total - vm.available) // (1024 * 1024),
        "ram_total_mb": vm.total // (1024 * 1024),
        "cpu_pct": psutil.cpu_percent(interval=None),
        "gpu_pct": read_gpu_pct(),
        # Always present as keys so the bar's renderer never sees `undefined`;
        # `None` only where the box genuinely has no such sensor (mac/cloud).
        "cpu_c": temps.get("cpu_c"),
        "gpu_c": temps.get("gpu_c"),
        "tj_c": temps.get("tj_c"),
    }


def snapshot(
    conn,
    *,
    planner_model: str,
    caption_model: str,
    embed_model: str | None = None,
    text_embed_model: str | None = None,
    models_client=None,
) -> dict:
    snap = _machine() | {"active": None, "models": []}
    if models_client is None:
        return snap
    try:
        r = models_client.resources(timeout=2.0)
    except Exception:  # noqa: BLE001 - model info is best-effort; the metrics still stand
        return snap
    return snap | {
        "active": r.get("active"),          # embedding / captioning / chat / planning / …
        # What is ACTUALLY loaded right now, by FULL model name.
        "models": display_names(
            r.get("resident", []),
            planner_model=planner_model,
            caption_model=caption_model,
            embed_model=embed_model,
            text_embed_model=text_embed_model,
        ),
    }
