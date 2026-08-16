# tests/test_resources_api.py — build the app directly from the `settings` fixture
from fastapi.testclient import TestClient

from web.app import create_app


def test_resources_endpoint_reports_ram_and_cpu(settings):
    # The bar proxies the `models` service's own /resources for the real
    # memory/GPU/resident/active (§5.1, §13). With no service reachable in-test
    # the snapshot degrades to a local psutil reading with no model info.
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    app = create_app(settings)
    with TestClient(app) as tc:
        resp = tc.get("/api/resources")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ram_total_mb"] > 0
    assert "cpu_pct" in data
    # No models service reachable in-test → GPU/active degrade to None, models
    # empty; every field is always present so the bar never breaks (§13).
    assert data["gpu_pct"] is None
    assert data["active"] is None
    assert data["models"] == []
