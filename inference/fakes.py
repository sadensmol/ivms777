from inference.client import ChatMessage


class FakeInferenceClient:
    def __init__(
        self,
        responses: list[str] | None = None,
        streams: list[list[str]] | None = None,
    ) -> None:
        self._responses = list(responses or [])
        self._streams = list(streams or [])
        self.calls: list[tuple[str, list[ChatMessage]]] = []

    def complete(
        self,
        model: str,
        messages: list[ChatMessage],
        *,
        json_schema: dict | None = None,
        timeout: float = 120.0,
    ) -> str:
        self.calls.append((model, messages))
        assert self._responses, "FakeInferenceClient ran out of queued responses"
        return self._responses.pop(0)

    def stream(self, model, messages, *, timeout: float = 120.0):
        self.calls.append((model, messages))
        assert self._streams, "FakeInferenceClient ran out of queued streams"
        yield from self._streams.pop(0)
