import json

from captioning.base import CaptionResult
from captioning.openai_captioner import OpenAICaptioner


class FakeClient:
    def __init__(self, payload):
        self._p = payload
        self.calls = []

    def complete(self, model, messages, *, json_schema=None, timeout=120.0):
        self.calls.append((model, messages, json_schema))
        return json.dumps(self._p)


def test_caption_parses_into_result():
    payload = {"caption": "a dog", "title": "Dog", "description": "a brown dog"}
    cap = OpenAICaptioner(FakeClient(payload), "gemma4-E2B")
    r = cap.caption(b"imgbytes")
    assert isinstance(r, CaptionResult)
    assert r.caption == "a dog" and r.title == "Dog" and r.description == "a brown dog"
    assert cap.caption_model == "gemma4-E2B" and cap.name == "gemma4-E2B"


def test_caption_sends_the_image_and_schema():
    c = FakeClient({"caption": "x", "title": "x", "description": "x"})
    cap = OpenAICaptioner(c, "gemma4-E2B")
    cap.caption(b"imgbytes")
    model, messages, json_schema = c.calls[0]
    assert model == "gemma4-E2B"
    assert json_schema is not None  # constrained to CAPTION_SCHEMA
    # The user message carries an image_url part (the OpenAI vision path).
    user = next(m for m in messages if m["role"] == "user")
    assert any(part.get("type") == "image_url" for part in user["content"])
