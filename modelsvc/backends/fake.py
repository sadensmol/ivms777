"""Deterministic `ModelBackend` (design §5.1) — no torch, no transformers, no
Ollama. Lets `modelsvc.app`, `ModelsClient`, and everything downstream be built
and tested with no real models (task 1 of the models-service refactor).

Same hashing trick as `embedding.fakes` / `inference.fakes`: a fixed seed
hashes to a stable unit vector, so identical inputs embed identically and
different inputs (almost) never collide — reproducible offline, no model.
"""

import hashlib
from collections.abc import Iterator

_DIM = 16


def _vector_from_bytes(seed: bytes, dim: int = _DIM) -> list[float]:
    raw = bytearray()
    counter = 0
    while len(raw) < dim * 2:
        raw += hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
        counter += 1
    values = [int.from_bytes(raw[i : i + 2], "big") / 65535.0 - 0.5 for i in range(0, dim * 2, 2)]
    norm = sum(v * v for v in values) ** 0.5 or 1.0
    return [v / norm for v in values]


class FakeBackend:
    """Fixed, deterministic outputs for every `ModelBackend` method."""

    def __init__(self, profile: str = "mac") -> None:
        # The slot surface (design §4.1) is real here — `app`'s settings popup is
        # tested against this backend, so switching and downloading must behave,
        # just without weights: a "download" completes instantly.
        from models import catalog

        self._catalog = catalog
        self._profile = profile
        self._slots = dict(catalog.DEFAULTS[profile])
        self._generation = 0
        self._downloaded: set[tuple[str, str]] = set()

    def embed_image(self, images: list[bytes]) -> list[list[float]]:
        return [_vector_from_bytes(b"image:" + image) for image in images]

    def embed_text(self, texts: list[str]) -> list[list[float]]:
        return [_vector_from_bytes(b"text:" + text.encode()) for text in texts]

    def tag(self, image: bytes, dimensions: list[str]) -> dict[str, list[str]]:
        return {dimension: [f"{dimension}:fake"] for dimension in dimensions}

    def calibration(self) -> dict:
        # Matches embedding.fakes.FakeEmbedder, so RemoteEmbedder built over a
        # FakeBackend behaves exactly like the in-process FakeEmbedder path.
        return {"logit_scale": 10.0, "logit_bias": -5.0}

    def caption(self, image: bytes) -> dict:
        return {
            "caption": "a fake photo",
            "title": "Fake Photo",
            "description": "a fake photo, produced by the fake models backend",
            "model": "fake",
        }

    def text_complete(
        self,
        model: str,
        messages: list[dict],
        json_schema: dict | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        last = messages[-1]["content"] if messages else ""
        return f"fake completion for: {last}"

    def text_stream(self, model: str, messages: list[dict]) -> Iterator[str]:
        yield "Hello"
        yield " from"
        yield " the fake backend."

    def text_embed(self, model: str, texts: list[str]) -> list[list[float]]:
        return [_vector_from_bytes(b"text-embed:" + text.encode()) for text in texts]

    def text_warm(self, model: str) -> None:
        return None

    def text_evict(self, model: str) -> None:
        return None

    def resources(self) -> dict:
        return {
            "ram_used_mb": 1024,
            "ram_total_mb": 8192,
            "cpu_pct": 12.5,
            "gpu_pct": None,
            "resident": ["fake"],
            "active": None,
            "slots": dict(self._slots),
            "generation": self._generation,
        }

    def models_state(self) -> dict:
        return {
            "resident": [],
            "budget_mb": 0,
            "free_mb": 0.0,
            "used_mb": 0,
            "active": None,
            "slots": dict(self._slots),
            "generation": self._generation,
        }

    def model_ensure(self, name: str) -> None:
        return None

    def model_unload(self, name: str) -> None:
        return None

    # --- slots (design §4.1) ---
    def _download_status(self, entry) -> dict:
        total = entry.size_mb * 1024 * 1024
        ready = (entry.slot, entry.key) in self._downloaded or entry.key == self._slots.get(
            entry.slot
        )
        return {
            "state": "ready" if ready else "absent",
            "bytes": total if ready else 0,
            "total": total,
            "error": None,
        }

    def catalog(self) -> dict:
        from modelsvc.slots import catalog_payload

        return catalog_payload(
            self._profile, self._slots, self._generation, self._download_status
        )

    def set_slots(self, slots: dict) -> dict:
        for slot, key in slots.items():
            if slot not in self._catalog.SLOTS or not self._catalog.has(slot, key):
                raise ValueError(f"unknown model for {slot}: {key}")
            if self._profile not in self._catalog.get(slot, key).profiles:
                raise ValueError(f"{key} is not offered on {self._profile}")
        if any(self._slots.get(slot) != key for slot, key in slots.items()):
            self._slots.update(slots)
            self._generation += 1
        return {"slots": dict(self._slots), "generation": self._generation}

    def download(self, slot: str, key: str) -> dict:
        entry = self._catalog.get(slot, key)  # KeyError on an unknown model
        self._downloaded.add((slot, key))
        return self._download_status(entry)

    def embed_spec(self) -> dict:
        entry = self._catalog.get("image_embed", self._slots["image_embed"])
        pre = entry.preprocess
        return self.calibration() | {
            "preprocess": {
                "input_px": pre.input_px,
                "resample": pre.resample,
                "mode": pre.mode,
            },
            "generation": self._generation,
        }
