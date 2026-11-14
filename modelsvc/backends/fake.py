"""Deterministic `ModelBackend` (design §5.1) — no torch, no transformers, no
Ollama. Lets `modelsvc.app`, `ModelsClient`, and everything downstream be built
and tested with no real models (task 1 of the models-service refactor).

Same hashing trick as `embedding.fakes` / `inference.fakes`: a fixed seed
hashes to a stable unit vector, so identical inputs embed identically and
different inputs (almost) never collide — reproducible offline, no model.
"""

import hashlib
import itertools
from collections.abc import Iterator

import numpy as np

from audio.speaker_embed import PIPELINE, clip_samples
from audio.whisper_asr import RATE_IN, take_samples
from modelsvc.backends.base import (
    CloningUnavailable,
    SlotOff,
    SpeakerEmbedNotOffered,
    SpeakerEmbedRefused,
    TranscribeNotOffered,
    TranscribeRefused,
)
from modelsvc.voice_bakes import BakeNotOffered, BakeRunning
from storage.wav import encode_wav

_DIM = 16
# Stable Audio Open 1.0's ceiling (design §11.1). The fake clamps to the same figure
# so a caller that asks for too much fails the same way here as on a real board.
MUSIC_MAX_SECONDS = 47.0
# The context the fake server "serves" — the smallest one we actually run (jetson's
# `llm_ctx`), so a test that builds a prompt is budgeted like the tightest real box.
FAKE_CONTEXT = 2048
# The scripted bake: each status read moves one row on, and the last row stays.
FAKE_BAKE_ADAPTER = b"fake-adapter"
FAKE_BAKE_SCRIPT = (
    {"state": "running", "stage": "preparing the takes", "fraction": 0.02, "step": 0},
    {"state": "running", "stage": "training", "fraction": 0.535, "step": 1},
    {"state": "done", "stage": "done", "fraction": 1.0, "step": 2},
)
# What the fake speech recognition hears in a take it has no script for.
FAKE_WORDS = "fake words"


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
        self._bake_ids = itertools.count(1)
        self._bakes: dict[str, dict] = {}
        # The scripted speech recognition: take sha256 -> the text it "hears".
        self.transcripts: dict[str, str] = {}
        # The scripted speaker model, by clip sha256: its vector, or the refusal's reason.
        self.speaker_vectors: dict[str, list[float]] = {}
        self.speaker_refusals: dict[str, str] = {}

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

    def speak(self, text: str, language: str = "en", voice: str | None = None,
              segments: list[str] | None = None,
              voices: list[str | None] | None = None,
              references: dict | None = None) -> dict:
        # REAL, parseable WAV bytes, not a placeholder: downstream tests measure a true
        # duration from these.
        key = self._slots.get("voice")
        if key == "off":
            raise SlotOff("voice")
        # ~14 characters a second is roughly a read-aloud pace, so a longer memory
        # yields a longer clip and `fit_to_narration` has something to fit.
        seconds = max(0.5, len(text) / 14.0)
        clip = {
            "audio": encode_wav(np.zeros(int(24000 * seconds), dtype=np.float32), 24000),
            "sample_rate": 24000,
            "model": "fake-voice",
        }
        if references is not None:
            # A saved voice (§11.1): the profile must offer voice copy for the language,
            # as on the real path. Every token with a clip comes back `built`, with a blob
            # the caller can store; a revision string, because a stored blob always has
            # one. A `use_clip=False` token speaks from its adapter alone, as the child
            # does (D76): no conditioning, and the adapter it sent comes back `loaded`.
            if self._catalog.cloning_entry_for(self._profile, language) is None:
                raise CloningUnavailable(
                    f"no voice-copy model speaks {language} on {self._profile}")
            conditioning, adapters = {}, {}
            for token, ref in references.items():
                if ref.get("use_clip", True) is False:
                    adapters[token] = {"state": "loaded", "sha256": ref["adapter"]["sha256"]}
                    continue
                conditioning[token] = {
                    "state": "built", "model": "fake-voice", "revision": "fake",
                    "reason": "no stored conditioning",
                    "blob": b"fake-conditioning:" + ref["sha256"].encode()}
            return clip | {"revision": None, "conditioning": conditioning, "adapters": adapters}
        # The language picks the model (§11.1): swap to whichever PRESET voice speaks
        # it, exactly as SlotManager does on the real path. A cloning entry speaks no
        # preset, so it is never picked here.
        current = self._catalog.get("voice", key)
        if current.clones or language not in current.languages:
            for entry in self._catalog.entries_for("voice", self._profile):
                if not entry.clones and language in entry.languages:
                    self._slots["voice"] = entry.key
                    key = entry.key
                    break
            else:
                raise SlotOff(f"no voice available for {language}")
        return clip

    def music(
        self, prompt: str = "", seconds: float = 30.0, seed: int | None = None,
        sections: list[dict] | None = None, temperature: float | None = None,
        guidance: float | None = None,
    ) -> dict:
        if self._slots.get("music") == "off":
            raise SlotOff("music")
        plan = sections or [{"brief": prompt, "seconds": seconds}]
        segments, start = [], 0.0
        for section in plan:
            length = max(0.5, min(float(section.get("seconds", seconds)),
                                  MUSIC_MAX_SECONDS))
            segments.append({"start": round(start, 3), "end": round(start + length, 3)})
            start += length
        frames = int(44100 * start)
        return {
            "audio": encode_wav(np.zeros((2, frames), dtype=np.float32), 44100),
            "sample_rate": 44100,
            "model": "fake-music",
            "segments": segments,
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

    def text_context(self) -> int | None:
        return FAKE_CONTEXT

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

    def _bake_status(self, bake: dict) -> dict:
        row = FAKE_BAKE_SCRIPT[bake["row"]] | ({"state": "cancelled"} if bake["cancelled"] else {})
        adapter = ({"sha256": hashlib.sha256(FAKE_BAKE_ADAPTER).hexdigest(),
                    "bytes": len(FAKE_BAKE_ADAPTER), "keys": 1}
                   if row["state"] == "done" else None)
        return {"id": bake["id"], **row, "steps": 2, "loss_first": None, "loss_last": None,
                "seconds_per_step": None, "stage_seconds": {}, "elapsed_s": 0.0,
                "free_mb_before": None, "min_free_mb": None, "peak_device_bytes": None,
                "device": "fake", "error": None, "adapter": adapter, "model": "fake-voice",
                "revision": "fake", "lora": "{}", "config": bake["request"].get("config") or {},
                "dataset_sha256": "fake", "clips": len(bake["request"].get("clips") or []),
                "seconds": 0.0, "tag": bake["request"].get("tag") or {}}

    def bake_start(self, request: dict) -> dict:
        if self._catalog.adapter_entry_for(self._profile) is None:
            raise BakeNotOffered("baking a voice is not offered on this device yet")
        if any(self._bake_status(b)["state"] == "running" for b in self._bakes.values()):
            raise BakeRunning("a voice is already being baked")
        bake_id = f"fake{next(self._bake_ids)}"
        self._bakes = {bake_id: {"id": bake_id, "row": 0, "cancelled": False,
                                 "request": request}}
        return self._bake_status(self._bakes[bake_id])

    def bake_status(self, bake_id: str) -> dict | None:
        bake = self._bakes.get(bake_id)
        if bake is None:
            return None
        status = self._bake_status(bake)
        bake["row"] = min(bake["row"] + 1, len(FAKE_BAKE_SCRIPT) - 1)
        return status

    def bake_current(self) -> dict | None:
        return next((self._bake_status(b) for b in self._bakes.values()), None)

    def bake_cancel(self, bake_id: str) -> dict | None:
        bake = self._bakes.get(bake_id)
        if bake is None:
            return None
        if self._bake_status(bake)["state"] == "running":
            bake["cancelled"] = True
        return self._bake_status(bake)

    def bake_adapter(self, bake_id: str) -> bytes | None:
        bake = self._bakes.get(bake_id)
        if bake is None:
            return None
        if self._bake_status(bake)["state"] != "done":
            raise BakeRunning("the bake is not done")
        return FAKE_BAKE_ADAPTER

    def bake_delete(self, bake_id: str) -> bool:
        bake = self._bakes.get(bake_id)
        if bake is None:
            return False
        if self._bake_status(bake)["state"] == "running":
            raise BakeRunning("the bake is still running; cancel it first")
        del self._bakes[bake_id]
        return True

    def transcribe(self, audio: bytes, language: str) -> dict:
        entry = self._catalog.transcriber_for(self._profile)
        if entry is None:
            raise TranscribeNotOffered("hearing the words of a take is not offered on this device yet")
        if language not in entry.languages:
            raise TranscribeRefused(f"{entry.key} does not hear {language!r}")
        text = self.transcripts.get(hashlib.sha256(audio).hexdigest(), FAKE_WORDS)
        return {"text": text, "model": "fake-asr", "revision": "fake",
                "seconds": len(take_samples(audio)) / RATE_IN, "elapsed_s": 0.0}

    def embed_voice(self, audio: bytes) -> dict:
        entry = self._catalog.speaker_embedder_for(self._profile)
        if entry is None:
            raise SpeakerEmbedNotOffered("measuring a voice is not offered on this device yet")
        sha256 = hashlib.sha256(audio).hexdigest()
        if sha256 in self.speaker_refusals:
            raise SpeakerEmbedRefused(self.speaker_refusals[sha256])
        vector = self.speaker_vectors.get(sha256) or _vector_from_bytes(b"speaker:" + audio,
                                                                        entry.dim)
        samples, rate = clip_samples(audio)
        return {"vector": vector, "model": "fake-speaker", "revision": "fake",
                "seconds": len(samples) / rate, "elapsed_s": 0.0, "pipeline": PIPELINE}

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
