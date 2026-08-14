import httpx

from inference.client import OpenAICompatClient
from inference.fakes import FakeInferenceClient


def test_fake_stream_yields_queued_chunks():
    fake = FakeInferenceClient(streams=[["Hello ", "[photo:1]", " there"]])
    out = list(fake.stream("m", [{"role": "user", "content": "hi"}]))
    assert out == ["Hello ", "[photo:1]", " there"]
    assert fake.calls == [("m", [{"role": "user", "content": "hi"}])]


def test_openai_client_stream_parses_sse_deltas():
    body = (
        'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n'
        'data: {"choices":[{"delta":{}}]}\n\n'
        "data: [DONE]\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body)

    client = OpenAICompatClient("http://x/v1", transport=httpx.MockTransport(handler))
    out = list(client.stream("m", [{"role": "user", "content": "hi"}]))
    assert "".join(out) == "Hello"
