from typing import Protocol

from PIL import Image

EMBED_DIM = 1152


class Embedder(Protocol):
    """Maps images and text into one shared vector space.

    Both sides return L2-normalized vectors of length EMBED_DIM, so a dot product
    is cosine similarity and the same query text can rank images.

    `logit_scale` (already exp'd) and `logit_bias` are SigLIP's learned zero-shot
    calibration: sigmoid(cosine·logit_scale + logit_bias) turns a raw cosine into
    a real tag probability (see embedding.vectors.siglip_probability).
    """

    logit_scale: float
    logit_bias: float

    def embed_images(self, images: list[Image.Image]) -> list[list[float]]: ...
    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...
