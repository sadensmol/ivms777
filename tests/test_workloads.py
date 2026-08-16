from models import workloads as w


def test_chat_needs_siglip_and_planner():
    s = w.model_set("CHAT", planner_model="qwen2.5:3b", caption_model="qwen2.5vl:3b")
    assert s == frozenset({"siglip", "qwen2.5:3b"})


def test_ingest_caption_needs_only_the_captioner_resource():
    # design §4/§8.1: INGEST_CAPTION declares the CAPTIONER sentinel, not a raw
    # LLM tag — the coordinator resolves it to whichever adapter is injected
    # (Ollama on mac/cloud, in-process VLM on jetson).
    s = w.model_set("INGEST_CAPTION", planner_model="qwen2.5:3b", caption_model="qwen2.5vl:3b")
    assert s == frozenset({w.CAPTIONER})


def test_search_needs_only_siglip():
    s = w.model_set("SEARCH", planner_model="qwen2.5:3b", caption_model="qwen2.5vl:3b")
    assert s == frozenset({"siglip"})


def test_search_is_interactive_priority():
    assert w.PRIORITY["SEARCH"] == w.PRIORITY["CHAT"]


def test_interactive_outranks_background():
    assert w.PRIORITY["CHAT"] > w.PRIORITY["INGEST_EMBED"]
    assert w.PRIORITY["SEARCH"] > w.PRIORITY["INGEST_EMBED"]


def test_fits_rejects_over_budget():
    big = frozenset({"siglip", "qwen2.5:3b"})
    assert w.fits(big, budget_mb=100) is False           # tiny budget → refuse
    assert w.fits(big, budget_mb=99_999) is True
