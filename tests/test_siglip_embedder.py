"""The SigLIP embedder loads the checkpoint its SLOT names (design §4.1).

Switching the `image_embed` slot rebuilds the `TorchWorker` with the new entry's
repo (`modelsvc/slots.py`), so the embedder must actually honour that argument —
a hardcoded repo makes the settings popup a silent no-op that keeps embedding
with the old model while the UI says otherwise.

torch and transformers ship in the `models` extra, which a plain `uv sync` leaves
out, so `embedding.siglip` is imported against stand-in modules and dropped from
`sys.modules` afterwards: `test_modelsvc_backends` asserts the models PARENT process
never imports torch, and a leftover import here would poison it (§8.1).
"""

import sys
import types

import pytest


class _Scalar:
    """The two calibration values `SiglipEmbedder` reads with `.exp().item()` / `.item()`."""

    def exp(self):
        return self

    def item(self):
        return 0.0


@pytest.fixture
def siglip(monkeypatch):
    torch = types.ModuleType("torch")
    torch.float16 = "float16"
    torch.float32 = "float32"
    torch.Tensor = type("Tensor", (), {})
    torch.no_grad = lambda: lambda fn: fn
    transformers = types.ModuleType("transformers")
    transformers.AutoModel = None
    transformers.AutoProcessor = None
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.delitem(sys.modules, "embedding.siglip", raising=False)

    import embedding.siglip

    yield embedding.siglip
    sys.modules.pop("embedding.siglip", None)


def _capture(monkeypatch) -> dict:
    """Stand in for `AutoModel`/`AutoProcessor` and record every load call."""
    calls: list[tuple[str, dict]] = []

    class FakeModel:
        logit_scale = _Scalar()
        logit_bias = _Scalar()

        def eval(self):  # torch's eval-mode switch, not the builtin
            return self

    class FakeAuto:
        @staticmethod
        def from_pretrained(name, **kwargs):
            calls.append((name, kwargs))
            return FakeModel()

    monkeypatch.setattr("embedding.siglip.AutoModel", FakeAuto)
    monkeypatch.setattr("embedding.siglip.AutoProcessor", FakeAuto)
    return {"calls": calls}


def test_it_loads_the_repo_it_is_given(monkeypatch, siglip):
    seen = _capture(monkeypatch)
    siglip.SiglipEmbedder("google/siglip2-so400m-patch16-512", "cpu")
    assert [name for name, _ in seen["calls"]] == ["google/siglip2-so400m-patch16-512"] * 2


def test_the_mps_device_is_passed_through_to_the_loader(monkeypatch, siglip):
    seen = _capture(monkeypatch)
    embedder = siglip.SiglipEmbedder("google/siglip2-so400m-patch14-384", "mps")
    assert embedder.device == "mps"
    assert seen["calls"][0][1]["device_map"] == "mps"
