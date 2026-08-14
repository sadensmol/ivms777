import base64
import json
from collections.abc import Iterator
from typing import Protocol, TypedDict

import httpx


class ChatMessage(TypedDict):
    role: str
    content: object


def encode_image(data: bytes, mime: str = "image/jpeg") -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


class InferenceClient(Protocol):
    def complete(
        self,
        model: str,
        messages: list[ChatMessage],
        *,
        json_schema: dict | None = None,
        timeout: float = 120.0,
    ) -> str: ...

    def stream(
        self,
        model: str,
        messages: list[ChatMessage],
        *,
        timeout: float = 120.0,
    ) -> Iterator[str]: ...


class OpenAICompatClient:
    """Talks to anything speaking the OpenAI chat-completions API.

    That covers Ollama (mac, jetson) and vLLM (cloud), so swapping the inference
    backend is a base_url change, not a code change.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str = "unused",
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            transport=transport,
        )

    def complete(
        self,
        model: str,
        messages: list[ChatMessage],
        *,
        json_schema: dict | None = None,
        timeout: float = 120.0,
    ) -> str:
        payload: dict = {"model": model, "messages": messages, "stream": False}
        if json_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": json_schema, "strict": True},
            }
        response = self._client.post("/chat/completions", json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def stream(
        self,
        model: str,
        messages: list[ChatMessage],
        *,
        timeout: float = 120.0,
    ) -> Iterator[str]:
        payload = {"model": model, "messages": messages, "stream": True}
        with self._client.stream(
            "POST", "/chat/completions", json=payload, timeout=timeout
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                delta = json.loads(data)["choices"][0].get("delta", {}).get("content")
                if delta:
                    yield delta
