import sqlite3

import numpy as np

from albums.base import Album
from embedding.vectors import from_blob

# A pair of SigLIP vectors above this cosine is "visually alike". Calibrated from
# the real library's pair distribution — the 95th percentile of random pairs sits
# near 0.78, so this keeps only genuinely similar shots together.
SIMILAR_THRESHOLD = 0.78
MIN_CLUSTER = 2


class BySimilarityOrganizer:
    name = "similarity"
    label = "By similarity (visual clusters)"

    def organize(self, conn: sqlite3.Connection, owner_id: int) -> list[Album]:
        rows = conn.execute(
            "SELECT p.id AS id, v.embedding AS embedding"
            " FROM photos p JOIN photo_vec v ON v.rowid = p.id"
            " WHERE p.owner_id = ? AND p.thumb_key IS NOT NULL"
            " ORDER BY COALESCE(p.shot_at, p.created_at) DESC, p.id DESC",
            (owner_id,),
        ).fetchall()
        if not rows:
            return []

        ids = [r["id"] for r in rows]
        matrix = np.array([from_blob(r["embedding"]) for r in rows], dtype=np.float32)
        # Vectors are already L2-normalized, so the Gram matrix is pairwise cosine.
        sims = matrix @ matrix.T

        assigned = np.zeros(len(ids), dtype=bool)
        clusters: list[list[int]] = []
        # Seed-based (non-transitive) grouping: each cluster is a seed plus the
        # unassigned photos within threshold of it. Avoids single-link chaining
        # that would merge the whole library into one blob.
        for i in range(len(ids)):
            if assigned[i]:
                continue
            neighbours = np.where((sims[i] >= SIMILAR_THRESHOLD) & (~assigned))[0]
            assigned[i] = True
            if len(neighbours) >= MIN_CLUSTER:
                for j in neighbours:
                    assigned[j] = True
                clusters.append([ids[j] for j in neighbours])
            # a lone photo is simply left out of the "similar" view

        clusters.sort(key=len, reverse=True)
        return [
            Album(
                key=f"sim-{index}",
                title=f"Similar set {index + 1}",
                description=f"{len(members)} visually similar photos.",
                photo_ids=members,
                cover_id=members[0],
                meta={"kind": "similarity"},
            )
            for index, members in enumerate(clusters)
        ]
