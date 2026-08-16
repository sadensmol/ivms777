"""Profile-based captioner selection (design §4, §8.1) — mac/cloud stay on
OllamaCaptioner (today's path, unchanged); jetson selects the in-process
VLMCaptioner. `build_captioner` only *constructs* the jetson adapter — it must
never call `.load()` (that imports torch / downloads weights)."""

from captioning.ollama_adapter import OllamaCaptioner
from captioning.vlm_adapter import VLMCaptioner
from config import Settings
from inference.fakes import FakeInferenceClient


def test_mac_profile_uses_ollama_captioner(tmp_path):
    s = Settings(profile="mac", data_dir=tmp_path, use_fake_embedder=True, use_fake_inference=True)
    assert s.caption_backend == "ollama"
    assert isinstance(s.build_captioner(FakeInferenceClient([])), OllamaCaptioner)


def test_cloud_profile_uses_ollama_captioner(tmp_path):
    s = Settings(profile="cloud", data_dir=tmp_path, use_fake_embedder=True, use_fake_inference=True)
    assert s.caption_backend == "ollama"
    assert isinstance(s.build_captioner(FakeInferenceClient([])), OllamaCaptioner)


def test_jetson_profile_uses_inprocess_captioner(tmp_path):
    s = Settings(profile="jetson", data_dir=tmp_path, use_fake_embedder=True)
    assert s.caption_backend == "inprocess"
    assert s.caption_model_id == "Qwen/Qwen2.5-VL-3B-Instruct"
    cap = s.build_captioner(FakeInferenceClient([]))
    assert isinstance(cap, VLMCaptioner)   # NOT constructed/loaded — just built
    assert cap._model is None              # never loaded by build_captioner


def test_use_fake_inference_forces_ollama_captioner_on_jetson(tmp_path):
    """The offline test path must stay reachable on every profile, jetson
    included — use_fake_inference always wins over caption_backend."""
    s = Settings(profile="jetson", data_dir=tmp_path, use_fake_embedder=True, use_fake_inference=True)
    cap = s.build_captioner(FakeInferenceClient([]))
    assert isinstance(cap, OllamaCaptioner)
    assert cap.caption_model == "fake"


def test_explicit_caption_backend_beats_profile_default(tmp_path):
    s = Settings(profile="mac", data_dir=tmp_path, use_fake_embedder=True,
                 caption_backend="inprocess", caption_model_id="some/model")
    assert s.caption_backend == "inprocess"
    cap = s.build_captioner(FakeInferenceClient([]))
    assert isinstance(cap, VLMCaptioner)
