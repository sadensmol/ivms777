import sqlite3
from dataclasses import dataclass
from pathlib import Path

import yaml

from search.signals import dimension_weights


@dataclass(frozen=True)
class Vocab:
    dimensions: dict[str, list[str]]
    # SigLIP zero-shot selection (§7): softmax over each dimension's labels, keep
    # the top one plus any runner-up within `select_ratio` of it, up to
    # `max_per_dim` labels total.
    max_per_dim: int
    select_ratio: float
    # "Similar photos" (§9): per-dimension importance weight, EXPANDED from the four
    # tier weights in the yaml (`similar_tier_weights`). Consumers keep seeing a plain
    # {dimension: weight} map, so only this file knows tiers exist.
    dimension_weights: dict[str, float]
    # What each label is SAID as to the text tower (§7), keyed by dimension then
    # label — `playful` means different things under `vibe` and `emotion`, so the
    # dimension is part of the key. A label with no entry falls back to its
    # dimension's template. Several sentences are embedded and averaged (prompt
    # ensembling), so one unlucky phrasing cannot decide a tag.
    prompts: dict[str, dict[str, list[str]]]


def load_vocab(path: Path) -> Vocab:
    data = yaml.safe_load(path.read_text())
    raw = data.get("prompts") or {}
    return Vocab(
        dimensions=data["dimensions"],
        max_per_dim=int(data.get("max_per_dim", 3)),
        select_ratio=float(data.get("select_ratio", 0.5)),
        dimension_weights=dimension_weights(data.get("similar_tier_weights")),
        prompts={
            dimension: {
                label: ([said] if isinstance(said, str) else list(said))
                for label, said in (labels or {}).items()
            }
            for dimension, labels in raw.items()
        },
    )


def seed_tags(conn: sqlite3.Connection, vocab: Vocab) -> None:
    """Insert any missing (dimension, label) rows, so tag ids are stable."""
    for dimension, labels in vocab.dimensions.items():
        conn.executemany(
            "INSERT INTO tags(dimension, label) VALUES (?, ?)"
            " ON CONFLICT(dimension, label) DO NOTHING",
            [(dimension, label) for label in labels],
        )


def tag_id_map(conn: sqlite3.Connection) -> dict[tuple[str, str], int]:
    return {
        (row["dimension"], row["label"]): row["id"]
        for row in conn.execute("SELECT id, dimension, label FROM tags")
    }
