import sqlite3
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Vocab:
    dimensions: dict[str, list[str]]
    # SigLIP zero-shot selection (§7): softmax over each dimension's labels, keep
    # the top one plus any runner-up within `select_ratio` of it, up to
    # `max_per_dim` labels total.
    max_per_dim: int
    select_ratio: float
    # "Similar photos" (§9): per-dimension importance weight; 0 ignores a dimension.
    dimension_weights: dict[str, float]


def load_vocab(path: Path) -> Vocab:
    data = yaml.safe_load(path.read_text())
    return Vocab(
        dimensions=data["dimensions"],
        max_per_dim=int(data.get("max_per_dim", 3)),
        select_ratio=float(data.get("select_ratio", 0.5)),
        dimension_weights={k: float(v) for k, v in (data.get("similar_dimension_weights") or {}).items()},
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
