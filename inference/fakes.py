from inference.client import ChatMessage


class FakeInferenceClient:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
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
