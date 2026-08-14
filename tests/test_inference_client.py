import httpx
import pytest

from inference.client import OpenAICompatClient, encode_image
from inference.fakes import FakeInferenceClient


def test_encode_image_produces_a_data_uri():
    assert encode_image(b"abc").startswith("data:image/jpeg;base64,")


def test_fake_client_returns_queued_responses_and_records_calls():
    client = FakeInferenceClient(["first", "second"])
    messages = [{"role": "user", "content": "hi"}]

    assert client.complete("m", messages) == "first"
    assert client.complete("m", messages) == "second"
    assert len(client.calls) == 2
    assert client.calls[0][0] == "m"


def test_fake_client_raises_when_exhausted():
    client = FakeInferenceClient([])
    with pytest.raises(AssertionError):
        client.complete("m", [{"role": "user", "content": "hi"}])


def test_client_posts_to_chat_completions_and_returns_content():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["content-type"].startswith("application/json")
        return httpx.Response(200, json={"choices": [{"message": {"content": "a caption"}}]})

    transport = httpx.MockTransport(handler)
    client = OpenAICompatClient("http://inference/v1", transport=transport)

    result = client.complete("gemma4:e4b", [{"role": "user", "content": "describe"}])
    assert result == "a caption"


def test_client_passes_json_schema_as_response_format():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    client = OpenAICompatClient("http://inference/v1", transport=httpx.MockTransport(handler))
    schema = {"type": "object", "properties": {"caption": {"type": "string"}}}
    client.complete("m", [{"role": "user", "content": "x"}], json_schema=schema)

    assert captured["response_format"]["type"] == "json_schema"
    assert captured["response_format"]["json_schema"]["schema"] == schema


def test_client_raises_on_http_error():
    client = OpenAICompatClient(
        "http://inference/v1",
        transport=httpx.MockTransport(lambda request: httpx.Response(500, text="boom")),
    )
    with pytest.raises(httpx.HTTPStatusError):
        client.complete("m", [{"role": "user", "content": "x"}])
