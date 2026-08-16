import httpx

from inference.client import OpenAICompatClient


def test_evict_sends_keep_alive_zero():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["json"] = __import__("json").loads(request.content)
        return httpx.Response(200, json={"done": True})

    client = OpenAICompatClient("http://x/v1", transport=httpx.MockTransport(handler))
    client.evict("qwen2.5vl:3b")
    assert seen["url"] == "http://x/api/generate"
    assert seen["json"]["model"] == "qwen2.5vl:3b"
    assert seen["json"]["keep_alive"] == 0


def test_warm_requests_the_model_resident():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["json"] = __import__("json").loads(request.content)
        return httpx.Response(200, json={"done": True})

    client = OpenAICompatClient("http://x/v1", transport=httpx.MockTransport(handler))
    client.warm("qwen2.5:3b")
    assert seen["url"] == "http://x/api/generate"
    assert seen["json"]["model"] == "qwen2.5:3b"
