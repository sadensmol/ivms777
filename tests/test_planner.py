import json

from inference.fakes import FakeInferenceClient
from search.planner import QuerySpec, plan, spec_to_params


def test_plan_parses_model_json():
    payload = json.dumps({"semantic": "dog on a beach",
                          "facets": {"aperture": {"lte": 2.0}}})
    spec = plan(FakeInferenceClient(responses=[payload]), "m", "moody dog", ["vibe"])
    assert spec.semantic == "dog on a beach"
    assert spec.facets["aperture"] == {"lte": 2.0}


def test_plan_falls_back_on_garbage():
    spec = plan(FakeInferenceClient(responses=["not json at all"]), "m", "sunset", ["vibe"])
    assert spec.semantic == "sunset"        # raw query preserved
    assert spec.facets == {} and spec.tags == {}


def test_plan_falls_back_when_client_raises():
    spec = plan(FakeInferenceClient(responses=[]), "m", "sunset", ["vibe"])  # empty -> asserts
    assert spec.semantic == "sunset"


def test_spec_to_params_materializes_known_predicates():
    spec = QuerySpec(
        semantic="dog on a beach", date_from="2025-06-01", date_to="2025-08-31",
        tags={"vibe": ["moody"], "bogus_dim": ["x"]},
        facets={"time_of_day": ["night"], "aperture": {"lte": 2.0}, "bogus_key": ["y"]},
    )
    params = spec_to_params(spec, query="moody dog", dimensions=["vibe", "setting"])
    assert params["q"] == "dog on a beach"
    assert params["planned"] == "1"
    assert params["t_vibe"] == "moody"
    assert "t_bogus_dim" not in params           # unknown dimension dropped
    assert params["f_time_of_day"] == "night"
    assert params["n_aperture"] == ":2.0"        # gte empty, lte 2.0
    assert params["date_from"] == "2025-06-01" and params["date_to"] == "2025-08-31"
    assert "f_bogus_key" not in params and "n_bogus_key" not in params


def test_spec_to_params_uses_raw_query_when_semantic_empty():
    params = spec_to_params(QuerySpec(), query="birthday cake", dimensions=["vibe"])
    assert params["q"] == "birthday cake" and params["planned"] == "1"
