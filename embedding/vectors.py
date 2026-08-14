import struct
from math import sqrt


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
