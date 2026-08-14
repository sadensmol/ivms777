import hashlib

from PIL import Image

from embedding.base import EMBED_DIM
from embedding.vectors import l2_normalize


def _vector_from_bytes(seed: bytes) -> list[float]:
    """A stable unit vector derived from a byte seed.

    Deterministic and reproducible, so tests never touch a model, yet identical
    inputs collide exactly and different inputs almost never do.
    """
    raw = bytearray()
    counter = 0
    while len(raw) < EMBED_DIM * 2:
        raw += hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
        counter += 1
    values = [
        int.from_bytes(raw[i : i + 2], "big") / 65535.0 - 0.5
        for i in range(0, EMBED_DIM * 2, 2)
    ]
    return l2_normalize(values)


class FakeEmbedder:
    # Calibration chosen so identical vectors (cosine 1.0) score ~0.99 and
    # near-orthogonal ones ~0.007 — separable by any sane threshold in tests.
    logit_scale = 10.0
    logit_bias = -5.0

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [_vector_from_bytes(text.encode("utf-8")) for text in texts]

    def embed_images(self, images: list[Image.Image]) -> list[list[float]]:
        return [_vector_from_bytes(self._image_seed(image)) for image in images]

    @staticmethod
    def _image_seed(image: Image.Image) -> bytes:
        small = image.convert("RGB").resize((16, 16))
        return small.tobytes()
