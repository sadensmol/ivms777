"""The `photo_vec` table's declared width, and how it changes (design §4.1, §6).

`photo_vec` is a `sqlite-vec` virtual table with a FIXED width, and that width is
the selected `image_embed` model's `dim`. Vectors from two different encoders are
not comparable — not even loosely — so a switch to a model of a different width
**drops and recreates** the table rather than migrating it, and the embed stage is
requeued to refill it. That is the only honest option: keeping the old rows would
leave every KNN query silently ranking against a space the query no longer lives in.
"""

import re
import sqlite3

_DECLARED = re.compile(r"float\[(\d+)\]")


def vec_dim(conn: sqlite3.Connection) -> int | None:
    """The width `photo_vec` is declared with, or None when it does not exist."""
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'photo_vec'"
    ).fetchone()
    if row is None or not row["sql"]:
        return None
    match = _DECLARED.search(row["sql"])
    return int(match.group(1)) if match else None


def ensure_vec_dim(conn: sqlite3.Connection, dim: int) -> bool:
    """Make `photo_vec` `dim` wide. Returns True when it had to be rebuilt (and
    every stored vector was therefore dropped)."""
    if vec_dim(conn) == dim:
        return False
    conn.execute("DROP TABLE IF EXISTS photo_vec")
    conn.execute(f"CREATE VIRTUAL TABLE photo_vec USING vec0(embedding float[{dim}])")
    return True
