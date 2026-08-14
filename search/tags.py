import sqlite3


def parse_tag_filters(params: dict[str, str]) -> dict[str, list[str]]:
    """`t_<dimension>=a,b` -> `{dimension: [a, b]}`."""
    out: dict[str, list[str]] = {}
    for name, raw in params.items():
        if name.startswith("t_"):
            labels = [part for part in raw.split(",") if part]
            if labels:
                out[name[2:]] = labels
    return out


def tag_where(filters: dict[str, list[str]], score_min: float) -> tuple[str, list]:
    """A WHERE fragment for splicing after `... FROM photos p WHERE ...`.

    OR within a dimension (any of its labels), AND across dimensions.
    """
    fragments: list[str] = []
    params: list = []
    for dimension, labels in filters.items():
        placeholders = ", ".join("?" for _ in labels)
        fragments.append(
            " AND EXISTS (SELECT 1 FROM photo_tags pt JOIN tags t ON t.id = pt.tag_id"
            f" WHERE pt.photo_id = p.id AND t.dimension = ? AND t.label IN ({placeholders})"
            " AND pt.score >= ?)"
        )
        params += [dimension, *labels, score_min]
    return "".join(fragments), params


def tag_sidebar(
    conn: sqlite3.Connection, owner_id: int, dimensions: list[str], score_min: float, limit: int
) -> list[dict]:
    groups = []
    for dimension in dimensions:
        rows = conn.execute(
            "SELECT t.label AS label, count(*) AS n FROM photo_tags pt"
            " JOIN tags t ON t.id = pt.tag_id JOIN photos p ON p.id = pt.photo_id"
            " WHERE t.dimension = ? AND p.owner_id = ? AND pt.score >= ?"
            " GROUP BY t.label ORDER BY n DESC, t.label LIMIT ?",
            (dimension, owner_id, score_min, limit),
        ).fetchall()
        if rows:
            groups.append({
                "dimension": dimension,
                "values": [{"label": r["label"], "count": r["n"]} for r in rows],
            })
    return groups
