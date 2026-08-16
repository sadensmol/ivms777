import pytest

from captioning.base import CaptionResult
from captioning.vlm_adapter import VLMCaptioner
from tests.fixtures import jpeg_bytes


class FakeProcessor:
    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True): return "PROMPT"
    def __call__(self, text=None, images=None, return_tensors=None):
        class B(dict):
            def to(self, dev): return self
        b = B(); b["input_ids"] = [[0,1,2]]; return b
    def batch_decode(self, ids, skip_special_tokens=True):
        return ['Sure: {"caption":"a cat","title":"Cat","description":"a black cat","tags":{"subject":["cat"]}}']


class FakeModel:
    def generate(self, **kw): return [[0,1,2,3,4,5]]
    def eval(self): return self


class FakeProcessorNoCaption(FakeProcessor):
    def batch_decode(self, ids, skip_special_tokens=True):
        return ['{"title":"X","description":"Y","tags":{}}']


def _fake_loader(model_id, device):
    return FakeModel().eval(), FakeProcessor()


def _fake_loader_no_caption(model_id, device):
    return FakeModel().eval(), FakeProcessorNoCaption()


def test_vlm_caption_extracts_json_result():
    cap = VLMCaptioner("Qwen/Qwen2.5-VL-3B-Instruct", _loader=_fake_loader)
    cap.load()
    r = cap.caption(jpeg_bytes(), ["subject"])
    assert isinstance(r, CaptionResult)
    assert r.caption == "a cat" and r.tags == {"subject": ["cat"]}
    assert cap.footprint_mb() == 2700
    cap.release()   # must not raise even without CUDA


def test_caption_before_load_raises():
    cap = VLMCaptioner("m", _loader=_fake_loader)
    with pytest.raises(RuntimeError):
        cap.caption(b"x", ["subject"])


def test_caption_missing_caption_key_raises():
    cap = VLMCaptioner("m", _loader=_fake_loader_no_caption)
    cap.load()
    with pytest.raises(ValueError):
        cap.caption(jpeg_bytes(), ["subject"])
