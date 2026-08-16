import hashlib
from collections.abc import Callable

from inference.client import ChatMessage, InferenceCancelled


def _fake_text_vector(text: str, dim: int = 16) -> list[float]:
    """A stable unit vector per text, so identical captions embed identically and
    similarity is reproducible offline — mirrors embedding.fakes for text."""
    raw = bytearray()
    counter = 0
    while len(raw) < dim * 2:
        raw += hashlib.sha256(text.strip().lower().encode() + counter.to_bytes(4, "big")).digest()
        counter += 1
    values = [int.from_bytes(raw[i : i + 2], "big") / 65535.0 - 0.5 for i in range(0, dim * 2, 2)]
    norm = sum(v * v for v in values) ** 0.5 or 1.0
    return [v / norm for v in values]


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
        should_stop: Callable[[], bool] | None = None,
    ) -> str:
        if should_stop is not None and should_stop():
            raise InferenceCancelled(model)   # mirrors the cancellable client path
        self.calls.append((model, messages))
        assert self._responses, "FakeInferenceClient ran out of queued responses"
        return self._responses.pop(0)

    def stream(self, model, messages, *, timeout: float = 120.0):
        self.calls.append((model, messages))
        assert self._streams, "FakeInferenceClient ran out of queued streams"
        yield from self._streams.pop(0)

    def embed(self, model: str, texts: list[str], *, timeout: float = 60.0) -> list[list[float]]:
        return [_fake_text_vector(t) for t in texts]

    def warm(self, model: str, *, timeout: float = 120.0) -> None:
        return None

    def evict(self, model: str, *, timeout: float = 30.0) -> None:
        return None
