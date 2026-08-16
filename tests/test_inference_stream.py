import threading
import time

import httpx
import pytest

from inference.client import InferenceCancelled, OpenAICompatClient
from inference.fakes import FakeInferenceClient


_SSE_BODY = (
    'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n'
    'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n'
    "data: [DONE]\n\n"
)


def _sse_client():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=_SSE_BODY)

    return OpenAICompatClient("http://x/v1", transport=httpx.MockTransport(handler))


def test_complete_with_should_stop_assembles_the_full_streamed_content():
    # should_stop never fires -> the cancellable path streams and returns the whole
    # completion, same string the non-streaming path would return.
    out = _sse_client().complete(
        "m", [{"role": "user", "content": "hi"}], should_stop=lambda: False
    )
    assert out == "Hello"


def test_complete_aborts_the_stream_when_should_stop_fires():
    # should_stop True -> the in-flight completion is cancelled (the caption
    # force-preempt, §8.1) by raising InferenceCancelled, not running to the end.
    with pytest.raises(InferenceCancelled):
        _sse_client().complete(
            "m", [{"role": "user", "content": "hi"}], should_stop=lambda: True
        )


class _BlockingStream(httpx.SyncByteStream):
    """A stream that sends NO bytes until it is closed — mimics a slow VLM caption
    whose first token (image prefill) is tens of seconds away, so the read blocks."""

    def __init__(self) -> None:
        self._released = threading.Event()

    def __iter__(self):
        self._released.wait(timeout=10)  # block as if still prefilling
        yield b'data: {"choices":[{"delta":{"content":"late"}}]}\n\n'

    def close(self) -> None:
        self._released.set()  # closing the response unblocks the read


def test_complete_aborts_during_a_blocking_prefill_before_any_line():
    # The real caption bug (§8.1): should_stop flips True while the read is BLOCKED
    # waiting for the first token. The abort must still fire promptly — a watcher
    # closes the stream — not wait out the whole prefill.
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=_BlockingStream())

    client = OpenAICompatClient("http://x/v1", transport=httpx.MockTransport(handler))
    stop = {"v": False}

    def trip() -> None:
        time.sleep(0.3)  # preempt requested shortly after the call starts
        stop["v"] = True

    threading.Thread(target=trip, daemon=True).start()
    started = time.monotonic()
    with pytest.raises(InferenceCancelled):
        client.complete("m", [{"role": "user", "content": "hi"}], should_stop=lambda: stop["v"])
    assert time.monotonic() - started < 3  # aborted promptly, not after the 10s block


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
