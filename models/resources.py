"""Live resource + lease snapshot for the resource bar (design §13, §8.1)."""
import psutil

from models import lease_store as ls
from models import workloads as wl


def snapshot(conn, *, planner_model: str, caption_model: str) -> dict:
    vm = psutil.virtual_memory()
    lease = ls.read_lease(conn)
    workload = lease["workload"] if lease else None
    models: list[str] = []
    budget_used = 0
    if workload:
        want = wl.model_set(workload, planner_model=planner_model, caption_model=caption_model)
        # INGEST_CAPTION's set is the CAPTIONER sentinel (the coordinator resolves
        # it to whichever adapter is injected) — substitute the real caption model
        # tag for a meaningful bar label + footprint. The coordinator's own RAM
        # guard already uses the adapter's true footprint (§8.1); this is display-only.
        display = frozenset(caption_model if m == wl.CAPTIONER else m for m in want)
        # SigLIP first, then LLMs alphabetically — a stable, readable order (§13),
        # e.g. `siglip+qwen2.5:3b`. Plain alphabetical sort would put an LLM tag
        # ahead of "siglip" ('q' < 's'), which doesn't match the design's example.
        models = sorted(display, key=lambda m: (m != wl.SIGLIP, m))
        budget_used = wl.footprint_mb(display)
    return {
        "ram_used_mb": (vm.total - vm.available) // (1024 * 1024),
        "ram_total_mb": vm.total // (1024 * 1024),
        "cpu_pct": psutil.cpu_percent(interval=None),
        "workload": workload,
        "models": models,
        "budget_used_mb": budget_used,
    }
