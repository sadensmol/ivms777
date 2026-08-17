from starlette.testclient import TestClient

from modelsvc.app import create_models_app
from modelsvc.backends.fake import FakeBackend


def test_models_state_endpoint():
    c = TestClient(create_models_app(FakeBackend()))
    r = c.get("/models")
    assert r.status_code == 200
    body = r.json()
    assert "resident" in body
    assert "budget_mb" in body
    assert "free_mb" in body
    assert "used_mb" in body


def test_ensure_and_unload_endpoints():
    c = TestClient(create_models_app(FakeBackend()))
    assert c.post("/models/siglip/ensure").status_code == 200
    assert c.post("/models/siglip/unload").status_code == 200


def test_control_api_on_real_composite(tmp_path):
    # torch-free: registry with a fake spec, drive ensure/unload/state end to end.
    from modelsvc.backends.composite import CompositeBackend
    from modelsvc.governor import MemoryGovernor
    from modelsvc.registry import ModelRegistry, ModelSpec
    from modelsvc.scheduler import Scheduler

    log = []
    reg = ModelRegistry()
    reg.register(ModelSpec("siglip", lambda: log.append("load"), lambda: log.append("free"), 1600))
    gov = MemoryGovernor(reg, measure_free_mb=lambda: 9999.0, budget_mb=6000)
    sched = Scheduler(gov, concurrency=1)

    class _Embed:
        pass

    backend = CompositeBackend(embed=_Embed(), registry=reg, governor=gov, scheduler=sched)
    c = TestClient(create_models_app(backend))

    assert c.post("/models/siglip/ensure").status_code == 200
    assert c.get("/models").json()["resident"] == ["siglip"]
    assert c.post("/models/siglip/unload").status_code == 200
    assert c.get("/models").json()["resident"] == []
    assert log == ["load", "free"]
