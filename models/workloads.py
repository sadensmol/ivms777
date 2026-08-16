# models/workloads.py
"""Workload → model-set declaration + RAM budget guard (design §8.1). Adding a
workload is a table entry here, not new load/unload logic."""
from models.lease_store import WorkloadName

SIGLIP = "siglip"  # sentinel for the in-process SigLIP model
CAPTIONER = "captioner"  # sentinel for the active captioner adapter (design §4/§8.1)

PRIORITY: dict[WorkloadName, int] = {
    "CHAT": 10, "MEMORY_REBUILD": 10, "SEARCH": 10, "INGEST_EMBED": 1, "INGEST_CAPTION": 1,
}

# Resident-MB estimates. Initial values; tune against the resource bar (§13).
# SigLIP so400m ~1.6 GB on GPU; qwen2.5:3b ~2.2 GB (Q4); qwen2.5vl:3b ~3.3 GB.
FOOTPRINT_MB: dict[str, int] = {
    SIGLIP: 1600, "qwen2.5:3b": 2200, "qwen2.5vl:3b": 3300, "qwen2.5vl:7b": 6000,
}
_FALLBACK_LLM_MB = 3000  # unknown LLM tag → conservative estimate


def model_set(workload: WorkloadName, *, planner_model: str, caption_model: str) -> frozenset[str]:
    if workload in ("CHAT", "MEMORY_REBUILD"):
        return frozenset({SIGLIP, planner_model})
    if workload in ("INGEST_EMBED", "SEARCH"):
        return frozenset({SIGLIP})
    if workload == "INGEST_CAPTION":
        return frozenset({CAPTIONER})
    raise ValueError(f"unknown workload: {workload}")


def footprint_mb(models: frozenset[str]) -> int:
    return sum(FOOTPRINT_MB.get(m, _FALLBACK_LLM_MB) for m in models)


def fits(models: frozenset[str], budget_mb: int) -> bool:
    return footprint_mb(models) <= budget_mb
