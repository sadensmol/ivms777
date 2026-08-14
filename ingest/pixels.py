import numpy as np
from PIL import Image

# Cheap palette/quality tags from pixel statistics (docs/design.md §7). SigLIP
# refines these later; the point here is a fast, model-free first pass.


def pixel_tags(image: Image.Image) -> list[tuple[str, str, float]]:
    """Return (dimension, label, score) tags for `palette` and `quality`."""
    small = image.convert("RGB").resize((128, 128))
    rgb = np.asarray(small, dtype=np.float32) / 255.0
    hsv = np.asarray(small.convert("HSV"), dtype=np.float32) / 255.0
    hue, sat, val = float(hsv[..., 0].mean()), float(hsv[..., 1].mean()), float(hsv[..., 2].mean())
    grey = rgb.mean(axis=2)
    # Laplacian-variance sharpness proxy: variance of a 4-neighbour edge response.
    lap = (
        -4 * grey
        + np.roll(grey, 1, 0) + np.roll(grey, -1, 0)
        + np.roll(grey, 1, 1) + np.roll(grey, -1, 1)
    )
    sharpness = float(lap.var())
    clipped_high = float((grey > 0.98).mean())
    clipped_low = float((grey < 0.02).mean())

    out: list[tuple[str, str, float]] = []

    def add(dimension: str, label: str, score: float) -> None:
        out.append((dimension, label, round(min(1.0, max(0.0, score)), 3)))

    # palette
    if sat < 0.12:
        add("palette", "monochrome", 1.0 - sat)
    else:
        add("palette", "warm" if (hue < 0.14 or hue > 0.92) else "cool", sat)
        if sat > 0.55:
            add("palette", "vivid", sat)
        elif sat < 0.30:
            add("palette", "pastel", 0.6)
    add("palette", "dark" if val < 0.30 else "bright", abs(val - 0.5) * 2)

    # quality
    if sharpness < 1e-4:
        add("quality", "blurry", 1.0)
    elif sharpness > 2e-3:
        add("quality", "sharp", min(1.0, sharpness * 100))
    if clipped_high > 0.25:
        add("quality", "overexposed", clipped_high)
    if clipped_low > 0.25:
        add("quality", "underexposed", clipped_low)
    return out
