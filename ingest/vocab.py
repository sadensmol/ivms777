import sqlite3
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Vocab:
    dimensions: dict[str, list[str]]
    _thresholds: dict[str, float]
    _default: float

    def threshold(self, dimension: str) -> float:
        return self._thresholds.get(dimension, self._default)


def load_vocab(path: Path) -> Vocab:
    data = yaml.safe_load(path.read_text())
    return Vocab(
        dimensions=data["dimensions"],
        _thresholds=data.get("thresholds", {}),
        _default=float(data.get("default_threshold", 0.18)),
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
