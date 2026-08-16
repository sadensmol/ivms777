import json

from captioning.base import CaptionResult
from captioning.ollama_adapter import OllamaCaptioner


class FakeClient:
    def __init__(self, payload): self._p = payload; self.warmed=[]; self.evicted=[]
    def complete(self, model, messages, *, json_schema=None, should_stop=None, timeout=120.0):
        return json.dumps(self._p)
    def warm(self, model, *, timeout=120.0): self.warmed.append(model)
    def evict(self, model, *, timeout=30.0): self.evicted.append(model)


def test_caption_parses_into_result():
    payload = {"caption":"a dog","title":"Dog","description":"a brown dog","tags":{"subject":["dog"]}}
    cap = OllamaCaptioner(FakeClient(payload), "qwen2.5vl:7b")
    r = cap.caption(b"imgbytes", ["subject","scene"])
    assert isinstance(r, CaptionResult)
    assert r.caption=="a dog" and r.title=="Dog" and r.tags=={"subject":["dog"]}
    assert cap.caption_model=="qwen2.5vl:7b" and cap.name=="qwen2.5vl:7b"


def test_load_release_warm_evict_the_model():
    c = FakeClient({"caption":"x","title":"x","description":"x","tags":{}})
    cap = OllamaCaptioner(c, "qwen2.5vl:7b")
    cap.load(); cap.release()
    assert c.warmed==["qwen2.5vl:7b"] and c.evicted==["qwen2.5vl:7b"]
