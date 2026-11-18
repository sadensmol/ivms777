"""The `faces` stage (design §8): find the faces of a photo through the models service,
store them, and put each new face into the closest face group (§6).

Clustering is the nearest group by mean vector: a face joins the group whose mean is
closest when the cosine similarity reaches the match threshold, else it starts a new
group. A pinned face (moved by the owner) never moves.
"""

import io
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
from PIL import Image, ImageOps

from inference.models_client import DetectedFace, ModelsClient
from ingest.jobs import enqueue
from ingest.worker import StageHandler
from storage.base import Storage

# Vectors are stored as float32, so a similarity computed from them carries float32
# rounding: a face exactly at the threshold must not fall just under it.
_SIMILARITY_TOLERANCE = 1e-6
_VECTOR_DTYPE = "<f4"


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
    rows = conn.execute(
        "SELECT p.id FROM photos p WHERE NOT EXISTS"
        " (SELECT 1 FROM jobs j WHERE j.photo_id = p.id AND j.stage = 'faces')"
    ).fetchall()
    for row in rows:
        enqueue(conn, row["id"], "faces")
    return len(rows)


def nearest_group(
    vector: Sequence[float], groups: Sequence[GroupMean], threshold: float
) -> int | None:
    """The id of the group whose mean is closest to `vector` with cosine similarity at
    or above `threshold`; None when no group qualifies."""
    face = np.asarray(vector, dtype=np.float64)
    face_norm = float(np.linalg.norm(face))
    if face_norm == 0.0:
        return None
    best_id: int | None = None
    best_similarity = threshold - _SIMILARITY_TOLERANCE
    for group in groups:
        mean = np.asarray(group.mean, dtype=np.float64)
        mean_norm = float(np.linalg.norm(mean))
        if mean.shape != face.shape or mean_norm == 0.0:
            continue
        similarity = float(face @ mean) / (face_norm * mean_norm)
        if similarity >= best_similarity:
            best_id, best_similarity = group.group_id, similarity
    return best_id


def assign_to_group(conn: sqlite3.Connection, face_id: int, threshold: float) -> int | None:
    """Put one unpinned face into its nearest group of the photo's owner, or into a new
    group. Returns the group id; None when the face is pinned or missing."""
    face = conn.execute(
        "SELECT f.vector, f.pinned, p.owner_id FROM faces f JOIN photos p ON p.id = f.photo_id"
        " WHERE f.id = ?",
        (face_id,),
    ).fetchone()
    if face is None or face["pinned"]:
        return None
    owner_id = face["owner_id"]
    group_id = nearest_group(
        np.frombuffer(face["vector"], dtype=_VECTOR_DTYPE).tolist(),
        _group_means(conn, owner_id, face_id),
        threshold,
    )
    if group_id is None:
        group_id = int(
            conn.execute(
                "INSERT INTO face_groups(owner_id, created_at) VALUES (?, ?)",
                (owner_id, datetime.now(UTC).isoformat()),
            ).lastrowid
        )
    conn.execute("UPDATE faces SET group_id = ? WHERE id = ?", (group_id, face_id))
    return group_id


def _group_means(conn: sqlite3.Connection, owner_id: int, face_id: int) -> list[GroupMean]:
    members: dict[int, list[np.ndarray]] = {}
    for row in conn.execute(
        "SELECT f.group_id, f.vector FROM faces f JOIN face_groups g ON g.id = f.group_id"
        " WHERE g.owner_id = ? AND f.id != ?",
        (owner_id, face_id),
    ):
        members.setdefault(row["group_id"], []).append(
            np.frombuffer(row["vector"], dtype=_VECTOR_DTYPE)
        )
    means: list[GroupMean] = []
    for group_id, vectors in members.items():
        if len({v.shape for v in vectors}) != 1:
            continue
        means.append(GroupMean(group_id=group_id, mean=tuple(np.mean(vectors, axis=0).tolist())))
    return means


def _upright_image(original: bytes, input_edge: int) -> bytes:
    with Image.open(io.BytesIO(original)) as image:
        image.load()
        upright = ImageOps.exif_transpose(image).convert("RGB")
    upright.thumbnail((input_edge, input_edge))
    buffer = io.BytesIO()
    upright.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def _store_faces(
    conn: sqlite3.Connection, photo_id: int, faces: Sequence[DetectedFace], threshold: float
) -> None:
    now = datetime.now(UTC).isoformat()
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute("DELETE FROM faces WHERE photo_id = ?", (photo_id,))
        for face in faces:
            face_id = int(
                conn.execute(
                    "INSERT INTO faces(photo_id, box_x, box_y, box_w, box_h, score, vector, model,"
                    " created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        photo_id,
                        *face.box,
                        face.score,
                        np.asarray(face.vector, dtype=_VECTOR_DTYPE).tobytes(),
                        face.model,
                        now,
                    ),
                ).lastrowid
            )
            assign_to_group(conn, face_id, threshold)
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


def faces_handler(
    originals: Storage, client: ModelsClient, params: FaceStageParams
) -> StageHandler:
    """The `faces` stage: send the original, upright and shrunk to the face model's input
    edge, to `POST /faces/detect`, then replace the photo's faces and group each one. A
    models-service error raises, so the queue keeps the job and retries it."""

    def handle(conn: sqlite3.Connection, photo_id: int) -> None:
        row = conn.execute("SELECT storage_key FROM photos WHERE id = ?", (photo_id,)).fetchone()
        image = _upright_image(originals.read(row["storage_key"]), params.input_edge)
        faces = client.detect_faces(image)
        _store_faces(conn, photo_id, faces, params.match_threshold)

    return handle
