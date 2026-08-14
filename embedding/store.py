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


def write_caption_vector(conn: sqlite3.Connection, photo_id: int, vector: list[float]) -> None:
    """Store the caption's text embedding on the photo (§9). Kept as a BLOB on
    `photos` rather than a vec0 table so it tolerates whatever dimension the text
    model uses; similarity is a bounded in-Python scan at this scale."""
    conn.execute("UPDATE photos SET caption_vec = ? WHERE id = ?", (to_blob(vector), photo_id))


def read_caption_vector(conn: sqlite3.Connection, photo_id: int) -> list[float] | None:
    row = conn.execute("SELECT caption_vec FROM photos WHERE id = ?", (photo_id,)).fetchone()
    return from_blob(row["caption_vec"]) if row and row["caption_vec"] is not None else None


def all_caption_vectors(conn: sqlite3.Connection, owner_id: int) -> dict[int, list[float]]:
    """Every owner photo's caption vector, for the similar-photos caption scan."""
    return {
        row["id"]: from_blob(row["caption_vec"])
        for row in conn.execute(
            "SELECT id, caption_vec FROM photos WHERE owner_id = ? AND caption_vec IS NOT NULL",
            (owner_id,),
        )
    }


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
