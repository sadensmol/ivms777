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
    assert data["active"] is None
    assert data["models"] == []
    # Every machine-metric key is ALWAYS present so the bar's renderer never sees
    # `undefined` and never changes shape. Their VALUES are box-dependent (no GPU
    # or thermal sensor off-Tegra), so only presence is asserted here.
    assert {"gpu_pct", "cpu_c", "gpu_c", "tj_c"} <= data.keys()


def test_machine_metrics_survive_the_models_service_being_down(settings):
    # The bar's whole point is telling you what the box is doing — most valuable
    # exactly when the models service is wedged. RAM/CPU/GPU/temps are local reads
    # (§5.1), so a dead service must cost only the MODEL info, nothing else.
    from models.resources import snapshot

    class DeadClient:
        def resources(self, timeout=None):
            raise ConnectionError("models service is down")

    shown = snapshot(
        None, planner_model="gemma4-E2B", caption_model="gemma4-E2B",
        embed_model="siglip", text_embed_model="nomic", models_client=DeadClient(),
    )
    assert shown["ram_total_mb"] > 0          # still a real reading
    assert isinstance(shown["cpu_pct"], float)
    assert {"gpu_pct", "cpu_c", "gpu_c", "tj_c"} <= shown.keys()
    assert shown["models"] == [] and shown["active"] is None  # only this is lost


def test_model_info_comes_from_the_service_metrics_never_do(settings, monkeypatch):
    # A service that (wrongly) echoes machine metrics must not override the local
    # reads — the local ones are authoritative.
    import models.resources as resources_module
    from models.resources import snapshot

    monkeypatch.setattr(resources_module, "read_gpu_pct", lambda: 42.0)
    monkeypatch.setattr(resources_module, "read_temps", lambda: {"cpu_c": 51.4, "gpu_c": 58.1})

    class ChattyClient:
        def resources(self, timeout=None):
            return {"resident": ["gemma-vision"], "active": "captioning",
                    "gpu_pct": 999.0, "cpu_c": 999.0}

    shown = snapshot(
        None, planner_model="gemma4-E2B", caption_model="gemma4-E2B",
        embed_model="siglip", text_embed_model="nomic", models_client=ChattyClient(),
    )
    assert shown["gpu_pct"] == 42.0 and shown["cpu_c"] == 51.4   # local wins
    assert shown["models"] == ["gemma4-E2B +vision"]
    assert shown["active"] == "captioning"


def test_resident_keys_render_as_full_model_names():
    # The bar must name the MODEL that is loaded, not the conveyor's internal key:
    # "gemma" reads as the planner model, "gemma-vision" as the caption model in its
    # vision mode (design §3.1/§13).
    from models.resources import display_names

    shown = display_names(
        ["siglip", "gemma"],
        planner_model="gemma4-E2B",
        caption_model="gemma4-E2B",
        embed_model="siglip2-so400m-patch14-384",
        text_embed_model="nomic-ai/nomic-embed-text-v1.5",
    )
    assert shown == ["siglip2-so400m-patch14-384", "gemma4-E2B"]

    vision = display_names(
        ["gemma-vision"],
        planner_model="gemma4-E2B",
        caption_model="gemma4-E2B",
        embed_model="s",
        text_embed_model="n",
    )
    assert vision == ["gemma4-E2B +vision"]


def test_display_names_falls_back_to_the_key_when_unknown():
    from models.resources import display_names

    assert display_names(
        ["mystery"],
        planner_model="p", caption_model="c", embed_model="e", text_embed_model="t",
    ) == ["mystery"]
