"""The models service's in-flight-op tracker for the resource bar (§13)."""
from modelsvc.activity import Activity
from modelsvc.backends.composite import CompositeBackend


def test_activity_is_none_when_idle_and_set_inside_track():
    a = Activity()
    assert a.current() is None
    with a.track("embedding"):
        assert a.current() == "embedding"
    assert a.current() is None


def test_nested_track_returns_the_most_recently_started_op():
    a = Activity()
    with a.track("embedding"), a.track("chat"):
        assert a.current() == "chat"
    # a partial exit leaves the outer op current
    with a.track("embedding"):
        with a.track("chat"):
            pass
        assert a.current() == "embedding"


class _RecordingEmbed:
    """An embed sub-backend that captures the composite's active label DURING
    its own call, proving the op is wrapped for its whole duration."""

    def __init__(self) -> None:
        self.during: str | None = "unset"
        self.backend: CompositeBackend | None = None

    def embed_image(self, images):
        self.during = self.backend._activity.current()
        return [[0.0]]

    def embed_text(self, texts):
        self.during = self.backend._activity.current()
        return [[0.0]]


def test_composite_marks_embedding_active_only_during_the_call():
    embed = _RecordingEmbed()
    backend = CompositeBackend(embed=embed)
    embed.backend = backend
    assert backend._activity.current() is None
    backend.embed_image([b"x"])
    assert embed.during == "embedding"
    assert backend._activity.current() is None  # cleared after


def test_composite_text_stream_stays_active_until_the_stream_is_drained():
    class _Text:
        def text_stream(self, model, messages):
            yield "a"
            yield "b"

    backend = CompositeBackend(embed=_RecordingEmbed(), text=_Text())
    gen = backend.text_stream("m", [])
    assert backend._activity.current() is None  # not entered until first iteration
    assert next(gen) == "a"
    assert backend._activity.current() == "chat"  # active while the stream is being drained
    assert list(gen) == ["b"]
    assert backend._activity.current() is None  # cleared once fully drained
