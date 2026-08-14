import sqlite3


def keyword_search(conn: sqlite3.Connection, owner_id: int, query: str, k: int) -> list[int]:
    """Photo ids best-matching `query` by FTS5 BM25, owner-scoped, best first.

    FTS5 is not owner-aware, so the join to `photos` applies the owner filter. The
    query is wrapped as a phrase so punctuation never breaks the MATCH parse.
    """
    text = query.strip()
    if not text:
        return []
    match = '"' + text.replace('"', '""') + '"'
    rows = conn.execute(
        "SELECT f.rowid AS photo_id FROM photo_fts f"
        " JOIN photos p ON p.id = f.rowid"
        " WHERE photo_fts MATCH ? AND p.owner_id = ?"
        " ORDER BY bm25(photo_fts) LIMIT ?",
        (match, owner_id, k),
    ).fetchall()
    return [row["photo_id"] for row in rows]
