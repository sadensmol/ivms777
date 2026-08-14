from PIL import Image

from ingest.pixels import pixel_tags


def _labels(image):
    return {(dim, label) for dim, label, _ in pixel_tags(image)}


def test_a_saturated_orange_reads_warm_and_vivid():
    labels = _labels(Image.new("RGB", (64, 64), (230, 120, 20)))
    assert ("palette", "warm") in labels
    assert ("palette", "vivid") in labels


def test_a_grey_image_reads_monochrome():
    assert ("palette", "monochrome") in _labels(Image.new("RGB", (64, 64), (128, 128, 128)))


def test_a_flat_image_reads_blurry():
    # No edges at all -> zero Laplacian variance -> blurry, never sharp.
    labels = _labels(Image.new("RGB", (64, 64), (100, 100, 100)))
    assert ("quality", "blurry") in labels
    assert ("quality", "sharp") not in labels


def test_a_blown_out_image_reads_overexposed():
    assert ("quality", "overexposed") in _labels(Image.new("RGB", (64, 64), (255, 255, 255)))


def test_a_black_image_reads_underexposed():
    assert ("quality", "underexposed") in _labels(Image.new("RGB", (64, 64), (2, 2, 2)))
