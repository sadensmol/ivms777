from inference.client import ChatMessage
from search.planner import QuerySpec, planner_messages


def test_query_spec_defaults_are_empty():
    spec = QuerySpec()
    assert spec.semantic == "" and spec.tags == {} and spec.facets == {}
    assert spec.date_from is None and spec.date_to is None


def test_query_spec_parses_the_9_1_example():
    spec = QuerySpec.model_validate({
        "semantic": "dog on a beach",
        "date_from": "2025-06-01", "date_to": "2025-08-31",
        "tags": {"vibe": ["moody"], "setting": ["beach"]},
        "facets": {"time_of_day": ["night"], "aperture": {"lte": 2.0}},
    })
    assert spec.semantic == "dog on a beach"
    assert spec.tags["vibe"] == ["moody"]
    assert spec.facets["aperture"] == {"lte": 2.0}


def test_planner_messages_list_dimensions_and_ask_for_json():
    msgs = planner_messages("moody dog at the beach at night", ["vibe", "setting"])
    system = msgs[0]["content"]
    assert "json" in system.lower()
    assert "vibe" in system and "setting" in system   # allowed tag dimensions
    assert "aperture" in system                        # a numeric facet key
    user: ChatMessage = msgs[1]
    assert "moody dog at the beach at night" in user["content"]
