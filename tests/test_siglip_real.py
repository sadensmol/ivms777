"""Calibration test for the real SigLIP model (design §14).

Marked `slow` and deselected by default (`addopts = -m 'not slow'`), so the fast
suite never downloads weights or imports torch. Run explicitly with:

    uv run pytest -m slow tests/test_siglip_real.py

It needs two real photos at tests/data/beach.jpg and tests/data/keyboard.jpg.
They are not committed (no binaries in the repo); the test skips if absent.
"""

from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

_DATA = Path(__file__).parent / "data"


@pytest.mark.skipif(
    not (_DATA / "beach.jpg").exists() or not (_DATA / "keyboard.jpg").exists(),
    reason="supply tests/data/beach.jpg and tests/data/keyboard.jpg to run",
)
def test_beach_query_ranks_a_beach_above_a_keyboard():
    from PIL import Image

    from embedding.siglip import SiglipEmbedder

    embedder = SiglipEmbedder("siglip2-so400m-patch14-384", "cpu")
    beach = Image.open(_DATA / "beach.jpg").convert("RGB")
    keyboard = Image.open(_DATA / "keyboard.jpg").convert("RGB")
    images = embedder.embed_images([beach, keyboard])
    query = embedder.embed_texts(["a photo of a beach"])[0]

    def dot(a, b):
        return sum(x * y for x, y in zip(a, b))

    assert dot(query, images[0]) > dot(query, images[1])
