"""SigLIP-backed `ModelBackend` (design §5.1, plan 15 task 2).

This is THE ONLY module in the repo — besides `embedding/siglip.py` itself —
that imports `embedding.siglip` (torch). Every other process (`app`, `worker`,
the CLI) reaches SigLIP only through the `models` service HTTP surface, via
`inference.remote_embedder.RemoteEmbedder`.

`embedding.siglip` is imported LAZILY, inside `_load`/`_release`/`_embedder`,
not at module top: importing `modelsvc.backends.siglip_backend` — and
therefore `modelsvc.backends` (which imports this module to build the real
backend) — never pulls torch into a process that isn't actually serving real
embeddings (e.g. the test suite, which only ever builds `FakeBackend`).

Residency (plan 15 task 5, §8.1): when a `residency` is injected, every
public method runs under `residency.use("siglip", priority=HIGH)` — the
interactive-embed priority that preempts an in-flight caption on jetson.
Without a `residency` (e.g. a bare unit test), each call is unwrapped, same
as before Task 5.
"""

from contextlib import nullcontext
from io import BytesIO

from PIL import Image

from modelsvc.residency import HIGH, Residency


class SiglipBackend:
    def __init__(self, model_name: str, device: str, residency: Residency | None = None) -> None:
        self.model_name = model_name
        self.device = device
        self._residency = residency
        if residency is not None:
            residency.register("siglip", self._load, self._release)

    def _load(self) -> None:
        self._embedder()

    def _release(self) -> None:
        from embedding.siglip import release_siglip_embedder

        release_siglip_embedder()

    def _embedder(self):
        from embedding.siglip import get_siglip_embedder

        # Cached (lru_cache inside get_siglip_embedder): the model loads once
        # and is reused across requests.
        return get_siglip_embedder(self.model_name, self.device)

    def _use(self):
        if self._residency is None:
            return nullcontext()
        return self._residency.use("siglip", priority=HIGH)

    def embed_image(self, images: list[bytes]) -> list[list[float]]:
        pil_images = [Image.open(BytesIO(image)).convert("RGB") for image in images]
        with self._use():
            return self._embedder().embed_images(pil_images)

    def embed_text(self, texts: list[str]) -> list[list[float]]:
        with self._use():
            return self._embedder().embed_texts(texts)

    def tag(self, image: bytes, dimensions: list[str]) -> dict[str, list[str]]:
        # Nothing calls /tag yet — taxonomy scoring stays client-side (ruling
        # R2 of plan 15 task 2): ingest/taxonomy.py computes it itself from
        # embed_texts + calibration, via RemoteEmbedder. Not a tagging
        # algorithm to invent here.
        with self._use():
            return {}

    def calibration(self) -> dict:
        with self._use():
            embedder = self._embedder()
            return {"logit_scale": embedder.logit_scale, "logit_bias": embedder.logit_bias}
