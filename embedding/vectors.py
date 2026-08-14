import struct
from math import exp, sqrt


def siglip_probability(cosine: float, logit_scale: float, logit_bias: float) -> float:
    """SigLIP zero-shot probability for one image/text pair.

    SigLIP is trained with a sigmoid loss, so its raw image·text cosine is small
    (≈0.0–0.15) and not thresholdable on its own. The real signal is
    sigmoid(cosine·logit_scale + logit_bias), using the model's learned scale and
    bias — that maps a strong match near 1 and a weak one near 0.
    """
    return 1.0 / (1.0 + exp(-(cosine * logit_scale + logit_bias)))


def l2_normalize(vector: list[float]) -> list[float]:
    norm = sqrt(sum(x * x for x in vector))
    if norm == 0.0:
        return list(vector)
    return [x / norm for x in vector]


def to_blob(vector: list[float]) -> bytes:
    """Pack a vector as little-endian float32 — the sqlite-vec wire format."""
    return struct.pack(f"<{len(vector)}f", *vector)


def from_blob(blob: bytes) -> list[float]:
    return list(struct.unpack(f"<{len(blob) // 4}f", blob))
