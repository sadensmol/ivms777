"""SigLIP must free its GPU allocation before the caption stage.

On the 8 GB unified-memory Jetson a resident SigLIP pins torch's CUDA allocator,
so Ollama can't get GPU buffers for the vision captioner and silently falls back
to CPU (design §8). `release_siglip_embedder` drops the cached model so the
memory returns to the driver; the drain pass calls it between the SigLIP stages
and the caption stage.

Exercises the REAL `embedding.siglip` module (needs torch — the `models`
extra, plan 15 task 6), so it is `slow` and deselected by default (pyproject
`addopts = -m 'not slow'`); run it with `pytest -m slow` on a box with torch
installed.
"""

import pytest

pytestmark = pytest.mark.slow


def test_release_drops_the_cached_model(monkeypatch):
    # Imported here, not at module top: a `slow` marker only deselects the
    # test ITEM after collection — pytest still has to import every module to
    # discover it, so an unconditional `from embedding import siglip` (torch)
    # at module scope would still ERROR collection in the torch-free default
    # suite. Deferring the import into the test body means it only runs when
    # this test actually runs (i.e. under `pytest -m slow` on a torch box).
    from embedding import siglip

    built: list[tuple[str, str]] = []

    class DummyEmbedder:
        def __init__(self, model_name: str, device: str) -> None:
            built.append((model_name, device))

    monkeypatch.setattr(siglip, "SiglipEmbedder", DummyEmbedder)
    siglip.get_siglip_embedder.cache_clear()

    first = siglip.get_siglip_embedder("m", "cpu")
    assert siglip.get_siglip_embedder("m", "cpu") is first  # cached: built once
    assert len(built) == 1

    siglip.release_siglip_embedder()  # returns the GPU allocation, clears the cache

    second = siglip.get_siglip_embedder("m", "cpu")
    assert second is not first  # released: the next use rebuilds
    assert len(built) == 2
    siglip.get_siglip_embedder.cache_clear()  # don't leak the dummy into other tests
