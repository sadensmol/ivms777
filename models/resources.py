"""Live resource snapshot for the top resource bar (design §13).

The `models` service is the ONLY process that loads models and touches the GPU
(§5.1), so the meaningful numbers — the models process's RAM, GPU load, which
model is resident, and the current in-flight op — all come from ITS `/resources`.
`app`/`worker` hold nothing, so their own psutil reading is only a fallback for
when the service is unreachable (then the bar still shows the app host's RAM/CPU
and no model info).
"""
import psutil


def snapshot(conn, *, planner_model: str, caption_model: str, models_client=None) -> dict:
    if models_client is not None:
        try:
            r = models_client.resources(timeout=2.0)
            return {
                "ram_used_mb": round(r["ram_used_mb"]),
                "ram_total_mb": round(r["ram_total_mb"]),
                "cpu_pct": r["cpu_pct"],
                "gpu_pct": r.get("gpu_pct"),
                "active": r.get("active"),          # embedding / captioning / chat / planning / …
                "models": r.get("resident", []),    # what is ACTUALLY loaded right now
            }
        except Exception:  # noqa: BLE001,S110 - the bar is best-effort; fall back if the service is down
            pass
    vm = psutil.virtual_memory()
    return {
        "ram_used_mb": (vm.total - vm.available) // (1024 * 1024),
        "ram_total_mb": vm.total // (1024 * 1024),
        "cpu_pct": psutil.cpu_percent(interval=None),
        "gpu_pct": None,
        "active": None,
        "models": [],
    }
