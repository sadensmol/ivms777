# tests/test_resources_api.py — build the app directly from the `settings` fixture
from fastapi.testclient import TestClient

from web.app import create_app


def test_resources_endpoint_reports_ram_cpu_and_idle_lease(settings):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    app = create_app(settings)
    with TestClient(app) as tc:
        resp = tc.get("/api/resources")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ram_total_mb"] > 0
    assert "cpu_pct" in data
    assert data["workload"] is None          # idle: no lease held in a fresh app


def test_resources_reflects_a_held_lease(settings):
    from models import lease_store as ls
    from models import workloads as wl
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    app = create_app(settings)
    # write a lease on the app's own conn; the route reads the same DB file (WAL)
    ls.try_acquire(app.state.context.conn, holder="worker", workload="INGEST_CAPTION", priority=1)
    with TestClient(app) as tc:
        data = tc.get("/api/resources").json()
    assert data["workload"] == "INGEST_CAPTION"
    # INGEST_CAPTION's set is the CAPTIONER sentinel — the bar must substitute the
    # real caption model tag, never the raw "captioner" sentinel (design §8.1).
    assert data["models"] == [settings.caption_model]
    # A real per-model footprint from the table, not the generic LLM fallback that
    # an unresolved "captioner" string would hit.
    assert data["budget_used_mb"] == wl.FOOTPRINT_MB[settings.caption_model]


def test_resources_lists_siglip_first_then_llm(settings):
    # M3: a stable, readable model order — SigLIP first, then LLMs — so the bar
    # reads `siglip+qwen2.5:3b` matching design.md §13's example, never the
    # alphabetical `qwen2.5:3b+siglip` ('q' < 's').
    from models import lease_store as ls
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    app = create_app(settings)
    ls.try_acquire(app.state.context.conn, holder="app", workload="CHAT", priority=10)
    with TestClient(app) as tc:
        data = tc.get("/api/resources").json()
    assert data["workload"] == "CHAT"
    assert data["models"] == ["siglip", "qwen2.5:3b"]
