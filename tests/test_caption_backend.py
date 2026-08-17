"""`CaptionBackend` + `build_caption_backend` (design §5.1) — the `models`
service's caption sub-backend. Since plan 16 captioning is a plain OpenAI
`/v1/chat/completions` call (llama-server / vLLM), so `CaptionBackend` just adapts
a `Captioner` to the caption dict — no residency, no load/evict, no preemption.
"""

from captioning.base import CaptionResult
from captioning.openai_captioner import OpenAICaptioner
from config import Settings
from modelsvc.backends import build_caption_backend
from modelsvc.backends.caption_backend import CaptionBackend


class FakeCaptioner:
    caption_model = "fake-cap"

    def __init__(self):
        self.calls = 0

    def caption(self, image):
        self.calls += 1
        return CaptionResult(
            caption="a cat", title="Cat", description="a cat photo"
        )


def test_caption_backend_returns_the_dict_with_the_model_key():
    backend = CaptionBackend(FakeCaptioner())

    result = backend.caption(b"image-bytes")

    assert result == {
        "caption": "a cat",
        "title": "Cat",
        "description": "a cat photo",
        "model": "fake-cap",
    }


def test_caption_backend_calls_the_captioner_each_time():
    captioner = FakeCaptioner()
    backend = CaptionBackend(captioner)

    backend.caption(b"one")
    backend.caption(b"two")

    assert captioner.calls == 2


def test_build_caption_backend_selects_openai_captioner_for_mac(tmp_path):
    s = Settings(profile="mac", data_dir=tmp_path, use_fake_embedder=True)
    backend = build_caption_backend(s)
    assert isinstance(backend, CaptionBackend)
    assert isinstance(backend._captioner, OpenAICaptioner)
    assert backend._captioner.caption_model == s.caption_model


def test_build_caption_backend_selects_openai_captioner_for_jetson(tmp_path):
    s = Settings(profile="jetson", data_dir=tmp_path, use_fake_embedder=True)
    backend = build_caption_backend(s)
    assert isinstance(backend._captioner, OpenAICaptioner)
    assert backend._captioner.caption_model == s.caption_model


def test_build_caption_backend_fake_path_uses_fake_client(tmp_path):
    s = Settings(profile="jetson", data_dir=tmp_path, use_fake_embedder=True, use_fake_inference=True)
    backend = build_caption_backend(s)
    assert isinstance(backend._captioner, OpenAICaptioner)
    assert backend._captioner.caption_model == "fake"
