"""The `faces` stage (design §8): find the faces of a photo through the models service,
store them, and put each new face into the closest face group (§6).

Clustering is the nearest group by mean vector: a face joins the group whose mean is
closest when the cosine similarity reaches the match threshold, else it starts a new
group. A pinned face (moved by the owner) never moves.
"""

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass

from inference.models_client import ModelsClient
from ingest.worker import StageHandler
from storage.base import Storage


@dataclass(frozen=True, slots=True, kw_only=True)
class GroupMean:
    group_id: int
    mean: tuple[float, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class FaceStageParams:
    match_threshold: float
    input_edge: int


def backfill_faces(conn: sqlite3.Connection) -> int:
    """Queue a `faces` job for every photo that has none. Returns the number queued."""
    return 0


def nearest_group(
    vector: Sequence[float], groups: Sequence[GroupMean], threshold: float
) -> int | None:
    """The id of the group whose mean is closest to `vector` with cosine similarity at
    or above `threshold`; None when no group qualifies."""
    return None


def assign_to_group(conn: sqlite3.Connection, face_id: int, threshold: float) -> int | None:
    """Put one unpinned face into its nearest group of the photo's owner, or into a new
    group. Returns the group id; None when the face is pinned or missing."""
    return None


def faces_handler(
    originals: Storage, client: ModelsClient, params: FaceStageParams
) -> StageHandler:
    def handle(conn: sqlite3.Connection, photo_id: int) -> None:
        return None

    return handle
