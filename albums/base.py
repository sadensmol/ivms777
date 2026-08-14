import sqlite3
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class Album:
    """A group of photos with a human title and a factual description.

    `description` is computed from data we already have (dates, cameras, counts,
    GPS). A richer AI-written "story" is added once the caption/LLM model lands —
    see docs/design.md §11.
    """

    key: str                       # stable, url-safe id within an organization
    title: str
    description: str
    photo_ids: list[int]
    cover_id: int
    meta: dict[str, str] = field(default_factory=dict)  # e.g. {"kind": "date"}

    @property
    def size(self) -> int:
        return len(self.photo_ids)


class Organizer(Protocol):
    """Turns a library into albums by one organizing principle."""

    name: str      # url value, e.g. "date"
    label: str     # dropdown label, e.g. "By date (events)"

    def organize(self, conn: sqlite3.Connection, owner_id: int) -> list[Album]: ...
