import sqlite3

from embedding.base import Embedder
from embedding.store import knn, read_vector
from embedding.vectors import l2_normalize


def search_photos(
    conn: sqlite3.Connection, embedder: Embedder, owner_id: int, query: str, k: int
) -> list[int]:
    """Photo ids best matching a natural-language query, best first."""
    if not query.strip():
        return []
    vector = l2_normalize(embedder.embed_texts([query])[0])
    return [photo_id for photo_id, _ in knn(conn, owner_id, vector, k)]


def similar_photos(
    conn: sqlite3.Connection, owner_id: int, photo_id: int, k: int
) -> list[int]:
    """Photo ids nearest to a given photo, excluding itself. Empty if unembedded."""
    vector = read_vector(conn, photo_id)
    if vector is None:
        return []
    return [pid for pid, _ in knn(conn, owner_id, vector, k, exclude_id=photo_id)]
