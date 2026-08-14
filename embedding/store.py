import sqlite3

from embedding.vectors import from_blob, to_blob


def write_vector(conn: sqlite3.Connection, photo_id: int, vector: list[float]) -> None:
    # rowid == photos.id, so a delete-then-insert keeps re-embeds idempotent.
    conn.execute("DELETE FROM photo_vec WHERE rowid = ?", (photo_id,))
    conn.execute(
        "INSERT INTO photo_vec(rowid, embedding) VALUES (?, ?)", (photo_id, to_blob(vector))
    )


def read_vector(conn: sqlite3.Connection, photo_id: int) -> list[float] | None:
    row = conn.execute(
        "SELECT embedding FROM photo_vec WHERE rowid = ?", (photo_id,)
    ).fetchone()
    return from_blob(row["embedding"]) if row is not None else None


def knn(
    conn: sqlite3.Connection,
    owner_id: int,
    vector: list[float],
    k: int,
    exclude_id: int | None = None,
) -> list[tuple[int, float]]:
    """Nearest photo ids to `vector`, owner-scoped, nearest first.

    sqlite-vec's KNN is not itself owner-aware, so it over-fetches and the join to
    `photos` applies the owner filter. At this scale that is comfortably fast.
    """
    rows = conn.execute(
        "SELECT v.rowid AS photo_id, v.distance AS distance"
        " FROM photo_vec v JOIN photos p ON p.id = v.rowid"
        " WHERE v.embedding MATCH ? AND k = ? AND p.owner_id = ?",
        (to_blob(vector), k + (1 if exclude_id else 0) + 32, owner_id),
    ).fetchall()
    hits = [(row["photo_id"], row["distance"]) for row in rows if row["photo_id"] != exclude_id]
    return hits[:k]
