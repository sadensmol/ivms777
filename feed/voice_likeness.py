"""The speaker similarity of generated speech (design §11.1): the cosine of its speaker
embedding with the centroid of the voice's own takes, on the only scale there is, the
voice's leave-one-out take range.

It measures and changes nothing else: no synthesis, no voice model, no film. No threshold
lives here, and a score is never a percent or a probability.

No models: a thin client (CLAUDE.md § "One model process"). Every vector comes from
`POST /voice/embed`. `app` caches the vectors of takes; `scripts/voice_bake_check.py` reads
the database read-only, so it keeps them in memory.
"""

import hashlib
import sqlite3
import statistics
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx
import numpy as np

from feed.build import _reason
from feed.saved_voices import SavedVoice, VoiceExample, get_voice, read_example_audio
from storage.base import Storage

METHOD = 1  # goes up when the arithmetic of a measure changes
MEASURING = "measuring the voice"
UNSCORABLE = 422  # /voice/embed's answer for a clip the speaker model cannot score
VECTOR_TYPE = "<f4"


class LikenessError(ValueError):
    """Why there is no measure, as the page shows it."""


@dataclass(frozen=True)
class Evaluator:
    model: str
    revision: str
    pipeline: int


class Meter:
    """One measurement's way to a vector. The first reply names the evaluator and every
    later one must match it, so vectors of two models or pipelines are never compared."""

    def __init__(self, client) -> None:
        self._client = client
        self.evaluator: Evaluator | None = None

    def embed(self, audio: bytes) -> list[float]:
        reply = self._client.embed_voice(audio)
        found = Evaluator(reply["model"], reply["revision"], reply["pipeline"])
        if self.evaluator is None:
            self.evaluator = found
        elif found != self.evaluator:
            raise LikenessError("the speaker model changed during the measure")
        return reply["vector"]


@dataclass(frozen=True)
class ReferenceTake:
    example_id: int
    content_hash: str
    vector: np.ndarray  # L2-normalised


@dataclass(frozen=True)
class Reference:
    """ONE snapshot of a voice's takes in one language: the centroid, the own range and
    the fingerprint of a whole measurement all come from it."""

    evaluator: Evaluator
    takes: tuple[ReferenceTake, ...]
    skipped: tuple[dict, ...]  # {example_id, reason}

    def similarity(self, vector: Sequence[float]) -> float:
        """The score of a clip: its cosine with the centroid of ALL the takes."""
        return float(_unit(vector) @ _centroid(take.vector for take in self.takes))

    def own_cosines(self) -> dict[int, float]:
        """Leave-one-out: each take against the centroid of the others. Empty under 2 takes."""
        if len(self.takes) < 2:
            return {}
        return {take.example_id: float(take.vector @ _centroid(
                    other.vector for other in self.takes if other is not take))
                for take in self.takes}

    def provenance(self) -> dict:
        """What a score was measured against, without the score: every field but `cosine`
        of `likeness_json`."""
        own = self.own_cosines()
        return {
            "evaluator": f"{self.evaluator.model}@{self.evaluator.revision}",
            "pipeline": self.evaluator.pipeline,
            "method": METHOD,
            "takes_count": len(self.takes),
            "takes": [{"example_id": take.example_id, "content_hash": take.content_hash,
                       "own_cosine": round(own[take.example_id], 6) if own else None}
                      for take in self.takes],
            "own": ({"min": round(min(own.values()), 6),
                     "median": round(statistics.median(own.values()), 6),
                     "max": round(max(own.values()), 6)} if own else None),
            "skipped": list(self.skipped),
            "reference_hash": reference_hash(take.content_hash for take in self.takes),
        }


def _unit(values: Sequence[float] | np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    return vector / np.linalg.norm(vector)


def _centroid(vectors: Iterable[np.ndarray]) -> np.ndarray:
    return _unit(np.mean(list(vectors), axis=0))


def reference_hash(content_hashes: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(content_hashes)).encode()).hexdigest()


def language_takes(voice: SavedVoice, language: str) -> list[VoiceExample]:
    """The candidates of a reference set: `transcript_error` plays no part, it concerns
    bakes. Read from the voice's takes, never from `voice_model_takes`, which outlives a
    deleted take."""
    return [take for take in voice.examples if take.language == language and not take.deleting]


def build_reference(
    meter: Meter,
    takes: list[VoiceExample],
    read_audio: Callable[[VoiceExample], bytes],
    *,
    known: Mapping[int, Sequence[float]] | None = None,
    before_embedding: Callable[[str, float], None] | None = None,
    on_embedded: Callable[[VoiceExample, list[float]], None] | None = None,
) -> Reference:
    """The reference set of `takes`. `known` holds vectors already made by `meter`'s
    evaluator; only the rest are embedded. A take the speaker model refuses, or whose file
    cannot be read, is skipped and named. Any other failure ends the measure."""
    vectors = dict(known or {})
    missing = [take for take in takes if take.id not in vectors]
    skipped = []
    for done, take in enumerate(missing):
        try:
            audio = read_audio(take)
        except OSError as exc:
            skipped.append({"example_id": take.id, "reason": f"its audio cannot be read ({exc})"})
            continue
        if before_embedding is not None:
            before_embedding(f"{MEASURING} · take {done + 1} of {len(missing)}", done / len(missing))
        try:
            vector = meter.embed(audio)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != UNSCORABLE:
                raise
            skipped.append({"example_id": take.id, "reason": _reason(exc)})
            continue
        vectors[take.id] = vector
        if on_embedded is not None:
            on_embedded(take, vector)
    usable = tuple(ReferenceTake(take.id, take.content_hash, _unit(vectors[take.id]))
                   for take in takes if take.id in vectors)
    if not usable or meter.evaluator is None:
        raise LikenessError("no take of this language can be measured")
    return Reference(meter.evaluator, usable, tuple(skipped))


def _cached_vectors(conn: sqlite3.Connection, evaluator: Evaluator, voice_id: int,
                    language: str) -> dict[int, np.ndarray]:
    rows = conn.execute(
        "SELECT example_id, vector FROM voice_take_embeddings"
        " WHERE model_key = ? AND revision = ? AND pipeline = ?"
        " AND example_id IN (SELECT id FROM voice_examples WHERE voice_id = ? AND language = ?)",
        (evaluator.model, evaluator.revision, evaluator.pipeline, voice_id, language),
    )
    return {row["example_id"]: np.frombuffer(row["vector"], dtype=VECTOR_TYPE) for row in rows}


def _cache_vector(conn: sqlite3.Connection, evaluator: Evaluator, example_id: int,
                  vector: list[float]) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO voice_take_embeddings(example_id, model_key, revision, pipeline, vector,"
        " created_at) SELECT ?, ?, ?, ?, ?, ? WHERE EXISTS (SELECT 1 FROM voice_examples WHERE id = ?)",
        (example_id, evaluator.model, evaluator.revision, evaluator.pipeline,
         np.asarray(vector, dtype=VECTOR_TYPE).tobytes(), datetime.now(UTC).isoformat(), example_id),
    )


def measure_test(
    conn: sqlite3.Connection,
    client,
    storage: Storage,
    owner_id: int,
    voice_id: int,
    *,
    language: str,
    audio: bytes,
    before_embedding: Callable[[str, float], None],
) -> dict:
    """The measure of one test, as `likeness_json` holds it. `before_embedding(stage,
    done)` runs right before every call with the fraction done, and may raise to stop.
    The test clip goes first: its reply names the evaluator the cached vectors must match,
    and a refused clip then embeds no take."""
    meter = Meter(client)
    before_embedding(MEASURING, 0.0)
    vector = meter.embed(audio)
    voice = get_voice(conn, owner_id, voice_id)
    if voice is None:
        raise LikenessError("the voice is gone")
    evaluator = meter.evaluator
    reference = build_reference(
        meter, language_takes(voice, language), lambda take: read_example_audio(storage, take),
        known=_cached_vectors(conn, evaluator, voice_id, language),
        before_embedding=before_embedding,
        on_embedded=lambda take, made: _cache_vector(conn, evaluator, take.id, made),
    )
    return {"cosine": round(reference.similarity(vector), 6), **reference.provenance()}


def measured_older_takes(likeness: dict, voice: SavedVoice, language: str) -> bool:
    """The takes of the language are no longer the ones measured against. A take that was
    skipped had no part in the score, so it changes nothing here."""
    skipped = {entry["example_id"] for entry in likeness["skipped"]}
    current = (take.content_hash for take in language_takes(voice, language)
               if take.id not in skipped)
    return reference_hash(current) != likeness["reference_hash"]
