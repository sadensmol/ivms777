"""The SigLIP embedder loads the checkpoint its SLOT names (design §4.1).

Switching the `image_embed` slot rebuilds the `TorchWorker` with the new entry's
repo (`modelsvc/slots.py`), so the embedder must actually honour that argument —
a hardcoded repo makes the settings popup a silent no-op that keeps embedding
with the old model while the UI says otherwise.

torch and `embedding.siglip` are imported INSIDE the tests, never at module
scope: `test_modelsvc_backends` asserts the models PARENT process never imports
torch, and a module-level import here would poison `sys.modules` for it (§8.1).
"""


def _capture(monkeypatch) -> dict:
    """Stand in for `AutoModel`/`AutoProcessor` and record every load call."""
    import torch

    calls: list[tuple[str, dict]] = []

    class FakeModel:
        logit_scale = torch.tensor(0.0)
        logit_bias = torch.tensor(0.0)

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


def test_it_loads_the_repo_it_is_given(monkeypatch):
    from embedding.siglip import SiglipEmbedder

    seen = _capture(monkeypatch)
    SiglipEmbedder("google/siglip2-so400m-patch16-512", "cpu")
    assert [name for name, _ in seen["calls"]] == [
        "google/siglip2-so400m-patch16-512"
    ] * 2


def test_the_mps_device_is_passed_through_to_the_loader(monkeypatch):
    from embedding.siglip import SiglipEmbedder

    seen = _capture(monkeypatch)
    embedder = SiglipEmbedder("google/siglip2-so400m-patch14-384", "mps")
    assert embedder.device == "mps"
    assert seen["calls"][0][1]["device_map"] == "mps"
