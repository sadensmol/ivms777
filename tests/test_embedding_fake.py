import math

from PIL import Image

from embedding.base import EMBED_DIM
from embedding.fakes import FakeEmbedder
from embedding.vectors import from_blob, l2_normalize, siglip_probability, to_blob


def test_siglip_probability_separates_strong_from_weak_cosines():
    # SigLIP's learned scale/bias turn a small cosine into a real probability;
    # a strong match reads high, a near-orthogonal pair reads ~0.
    strong = siglip_probability(1.0, logit_scale=10.0, logit_bias=-5.0)
    weak = siglip_probability(0.0, logit_scale=10.0, logit_bias=-5.0)
    assert strong > 0.9
    assert weak < 0.1
    assert 0.0 <= weak < strong <= 1.0


def test_fake_embedder_exposes_calibration():
    fake = FakeEmbedder()
    assert fake.logit_scale > 0
    assert fake.logit_bias < 0


def test_blob_round_trips_as_float32():
    vector = l2_normalize([0.1 * i for i in range(EMBED_DIM)])
    restored = from_blob(to_blob(vector))
    assert len(restored) == EMBED_DIM
    assert all(abs(a - b) < 1e-6 for a, b in zip(vector, restored))


def test_l2_normalize_gives_a_unit_vector():
    vector = l2_normalize([3.0, 4.0] + [0.0] * (EMBED_DIM - 2))
    assert abs(math.sqrt(sum(x * x for x in vector)) - 1.0) < 1e-6


def test_fake_is_deterministic_and_unit_length():
    fake = FakeEmbedder()
    a = fake.embed_texts(["beach"])[0]
    b = fake.embed_texts(["beach"])[0]
    assert a == b
    assert abs(math.sqrt(sum(x * x for x in a)) - 1.0) < 1e-6
    assert len(a) == EMBED_DIM


def test_fake_separates_different_inputs():
    fake = FakeEmbedder()
    beach = fake.embed_texts(["beach"])[0]
    keyboard = fake.embed_texts(["keyboard"])[0]
    dot = sum(x * y for x, y in zip(beach, keyboard))
    assert dot < 0.99  # not identical


def test_fake_images_are_keyed_by_pixels():
    fake = FakeEmbedder()
    red = Image.new("RGB", (8, 8), "red")
    blue = Image.new("RGB", (8, 8), "blue")
    assert fake.embed_images([red])[0] == fake.embed_images([red])[0]
    assert fake.embed_images([red])[0] != fake.embed_images([blue])[0]
