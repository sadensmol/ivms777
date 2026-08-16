"""SigLIP must free its GPU allocation before the caption stage.

On the 8 GB unified-memory Jetson a resident SigLIP pins torch's CUDA allocator,
so Ollama can't get GPU buffers for the vision captioner and silently falls back
to CPU (design §8). `release_siglip_embedder` drops the cached model so the
memory returns to the driver; the drain pass calls it between the SigLIP stages
and the caption stage.
"""

from embedding import siglip


def test_release_drops_the_cached_model(monkeypatch):
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
