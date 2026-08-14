from typing import Protocol

from PIL import Image

EMBED_DIM = 1152


class Embedder(Protocol):
    """Maps images and text into one shared vector space.

    Both sides return L2-normalized vectors of length EMBED_DIM, so a dot product
    is cosine similarity and the same query text can rank images.
    """

    def embed_images(self, images: list[Image.Image]) -> list[list[float]]: ...
    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...
