def date_where(params: dict[str, str]) -> tuple[str, list]:
    """Bound photos.shot_at by date_from/date_to (inclusive).

    ISO date strings sort lexicographically, so plain string comparison is
    correct. `date_to` is extended to end-of-day so the whole day is included.
    """
    fragment = ""
    bound: list = []
    date_from = (params.get("date_from") or "").strip()
    date_to = (params.get("date_to") or "").strip()
    if date_from:
        fragment += " AND p.shot_at >= ?"
        bound.append(date_from)
    if date_to:
        fragment += " AND p.shot_at <= ?"
        bound.append(date_to + "T23:59:59")
    return fragment, bound
