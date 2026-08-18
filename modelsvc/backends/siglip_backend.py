"""SigLIP embed sub-backend (design §5.1, §8.1).

Torch-free: SigLIP lives in a `TorchWorker` child (plan 20), so this module never
imports `embedding.siglip` and the `models` parent process never builds a CUDA
context — which is what makes evicting SigLIP actually return its RAM to gemma.

Load/evict and GPU serialization stay with the conveyor: the registry starts and
kills the worker, and `CompositeBackend` wraps every op in
`scheduler.run(["image_embed"], ...)`.

`worker` is a PROVIDER, not a worker: switching the `image_embed` slot builds a new
child (design §4.1), so this must ask for the live one on every call rather than
hold the one that existed at construction.
"""


class SiglipBackend:
    def __init__(self, worker) -> None:
        self._provider = worker

    def embed_image(self, images: list[bytes]) -> list[list[float]]:
        return self._provider().call("embed_image_bytes", images)

    def embed_text(self, texts: list[str]) -> list[list[float]]:
        return self._provider().call("embed_texts", texts)

    def tag(self, image: bytes, dimensions: list[str]) -> dict[str, list[str]]:
        # Nothing calls /tag yet — taxonomy scoring stays client-side: ingest
        # computes it from embed_texts + calibration via RemoteEmbedder.
        return {}

    def calibration(self) -> dict:
        return self._provider().call("calibration")
