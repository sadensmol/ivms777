import math
import sqlite3

from embedding.base import Embedder
from embedding.store import all_caption_vectors, knn, read_caption_vector, read_vector
from embedding.vectors import l2_normalize

# "similar" spans the photo's WHOLE character — subject, vibe, emotion, setting,
# light, palette… weighted per dimension (below), plus the caption's full meaning
# via its text embedding, plus a mild image-look-alike signal.


def search_photos(
    conn: sqlite3.Connection, embedder: Embedder, owner_id: int, query: str, k: int
) -> list[int]:
    """Photo ids best matching a natural-language query, best first."""
    if not query.strip():
        return []
    vector = l2_normalize(embedder.embed_texts([query])[0])
    return [photo_id for photo_id, _ in knn(conn, owner_id, vector, k)]


# Per-dimension importance for "similar" (§9); overridden from vocab.yaml. subject
# dominates, quality is ignored. Kept here so search/ has a sane default in tests.
DEFAULT_SIMILAR_DIMENSION_WEIGHTS = {
    "subject": 3.0, "setting": 1.5, "occasion": 1.5, "vibe": 1.0, "emotion": 1.0,
    "light": 0.8, "season_weather": 0.8, "composition": 0.8, "palette": 0.5, "quality": 0.0,
}
_CAPTION_WEIGHT = 3.0   # a shared rare caption word ranks like a strong subject match
_VECTOR_WEIGHT = 1.0    # holistic look, a mild tiebreak
_DECAY = 0.6            # each extra reason counts less, so one strong match beats many weak
# "Content" = what the photo IS. A candidate must share at least one content signal
# (a subject tag, a caption that means the same, or a genuine visual near-dup) to
# be "similar" at all. Style/scene dimensions (composition, vibe, palette, light,
# season, occasion, setting, emotion, quality) only RERANK content matches — two
# photos both shot top-down in cool overcast light are not "the same thing".
_CONTENT_TAG_DIMENSIONS = ("subject",)


def _tag_similarity(
    conn: sqlite3.Connection, owner_id: int, photo_id: int, weights: dict[str, float]
) -> dict[int, list[dict]]:
    """Per-shared-tag similarity contributions to every other photo (§9).

    Each shared tag contributes ``dimension_weight × agreement × idf`` where
    agreement is the WEAKER of the two photos' confidences (0.71 close-up matching
    1.00 close-up agree at 0.71, not 1.00) and idf (normalised 0–1) damps common
    tags. The per-dimension weight is what makes `subject` outrank `palette` and
    silences `quality` (weight 0) — the caller then decays extra contributions so
    one strong match beats a pile of weak ones. Returns
    {pid: [{"text", "pct", "contrib", "tag": (dim, label)}]}.
    """
    # MAX(score) GROUP BY tag_id folds a tag carried by two sources (siglip + vlm)
    # into one, so it is never counted — or shown as a reason — twice.
    src = conn.execute(
        "SELECT tag_id, MAX(score) AS score FROM photo_tags"
        " WHERE photo_id = ? AND score >= 0.1 GROUP BY tag_id",
        (photo_id,),
    ).fetchall()
    if not src:
        return {}
    total = conn.execute(
        "SELECT COUNT(*) AS n FROM photos WHERE owner_id = ?", (owner_id,)
    ).fetchone()["n"] or 1
    log_total = math.log(total) if total > 1 else 0.0
    hits: dict[int, list[dict]] = {}
    for tag_id, src_score in src:
        info = conn.execute(
            "SELECT t.dimension, t.label,"
            " (SELECT COUNT(DISTINCT photo_id) FROM photo_tags WHERE tag_id = t.id) AS df"
            " FROM tags t WHERE t.id = ?",
            (tag_id,),
        ).fetchone()
        dim_weight = weights.get(info["dimension"], 1.0)
        if dim_weight <= 0.0:
            continue  # this dimension does not matter for similarity (e.g. quality)
        idf = (math.log(total / (info["df"] or 1)) / log_total) if log_total > 0 else 1.0
        if idf <= 0.0:
            continue  # a tag on every photo carries no similarity signal
        for pid, cand_score in conn.execute(
            "SELECT pt.photo_id, MAX(pt.score) AS score FROM photo_tags pt"
            " JOIN photos p ON p.id = pt.photo_id"
            " WHERE pt.tag_id = ? AND pt.photo_id != ? AND p.owner_id = ?"
            " GROUP BY pt.photo_id",
            (tag_id, photo_id, owner_id),
        ):
            agreement = min(src_score, cand_score)
            hits.setdefault(pid, []).append({
                "text": f"{info['dimension']}: {info['label']}",
                "pct": round(agreement * 100),
                "contrib": dim_weight * agreement * idf,
                "tag": (info["dimension"], info["label"]),
            })
    return hits


def _caption_similarity(
    conn: sqlite3.Connection, owner_id: int, photo_id: int, min_cosine: float
) -> dict[int, float]:
    """Photos whose CAPTION means something similar (§9).

    Cosine between caption **text embeddings**, so the whole sentence's meaning is
    compared — "a dog stands on a rooftop" ≈ "a dog lounges on a dashboard", while
    "a small teddy bear" ≠ "a small domino tile". This replaced a crude match on a
    single shared word, which let generic words like "small" fake a match. Returns
    {pid: cosine} for photos at or above `min_cosine`.
    """
    src = read_caption_vector(conn, photo_id)
    if src is None:
        return {}
    src = l2_normalize(src)
    hits: dict[int, float] = {}
    for pid, vec in all_caption_vectors(conn, owner_id).items():
        if pid == photo_id:
            continue
        cosine = sum(a * b for a, b in zip(src, l2_normalize(vec)))
        if cosine >= min_cosine:
            hits[pid] = cosine
    return hits


def similar_photos(
    conn: sqlite3.Connection,
    owner_id: int,
    photo_id: int,
    k: int,
    min_cosine: float = 0.5,
    caption_min: float = 0.6,
    dimension_weights: dict[str, float] | None = None,
) -> list[dict]:
    """Photos similar to a given one, with the REASON each is similar (§9).

    Progressive, by what the pipeline has produced (graceful degradation):
      * no embedding yet -> nothing (the photo can't be compared);
      * embedding only    -> visual neighbours (image-vector KNN, cosine floor);
      * + taxonomy        -> add shared tags, per-dimension weighted;
      * + captions        -> add caption-meaning similarity (text-embedding cosine).

    Each facet is a scored **contribution** — a tag (`dimension_weight × agreement
    × idf`), the caption's semantic closeness (`_CAPTION_WEIGHT × caption cosine`),
    or the image look-alike (`_VECTOR_WEIGHT × image cosine`). A candidate's score
    is its contributions sorted high-to-low and summed with a `_DECAY` falloff, so
    ONE strong match (same subject, or a caption that means the same thing) beats a
    PILE of weak ones (five faint scene tags). Each result's `reasons` are its top-3
    contributions, shown sorted by match %, highest first.

    Returns [{"id", "score", "cosine", "tags": [(dim, label)],
    "reasons": [{"text", "pct"}]}], best first.
    """
    weights = dimension_weights or DEFAULT_SIMILAR_DIMENSION_WEIGHTS
    vector = read_vector(conn, photo_id)
    if vector is None:
        return []  # no embedding: nothing to compare against yet
    vector = l2_normalize(vector)
    cosine = {
        pid: 1.0 - dist * dist / 2.0  # unit vectors: L2^2 = 2(1-cos)
        for pid, dist in knn(conn, owner_id, vector, max(k, 30), exclude_id=photo_id)
    }
    tag_hits = _tag_similarity(conn, owner_id, photo_id, weights)
    caption_hits = _caption_similarity(conn, owner_id, photo_id, caption_min)

    results: list[dict] = []
    candidates = set(tag_hits) | set(caption_hits) | {p for p, c in cosine.items() if c >= min_cosine}
    for pid in candidates:
        tag_contribs = tag_hits.get(pid, [])
        cap_cos = caption_hits.get(pid)
        cos = cosine.get(pid, 0.0)
        near = cos >= min_cosine
        # GATE: a candidate is "similar" only if it shares a CONTENT signal — the
        # same subject, a caption that means the same, or a genuine near-dup look.
        # Style/scene tags alone (top-down + overcast + cool…) never qualify it.
        has_subject = any(c["tag"][0] in _CONTENT_TAG_DIMENSIONS for c in tag_contribs)
        if not (has_subject or cap_cos is not None or near):
            continue
        contribs: list[dict] = list(tag_contribs)
        if cap_cos is not None:
            contribs.append({"text": "caption (meaning)", "pct": round(cap_cos * 100),
                             "contrib": _CAPTION_WEIGHT * cap_cos})
        if near:
            contribs.append({"text": "looks alike", "pct": round(cos * 100),
                             "contrib": _VECTOR_WEIGHT * cos})
        if not contribs:
            continue
        contribs.sort(key=lambda c: -c["contrib"])
        # Decayed sum: the strongest facet counts fully, each next one less — so a
        # single subject match outweighs a heap of faint scene tags.
        score = sum(c["contrib"] * (_DECAY ** i) for i, c in enumerate(contribs))
        # Reasons in CONTRIBUTION order (relevance = importance × rarity × match),
        # most relevant first — so a generic style tag like composition:top-down
        # never leads over a subject/caption match just for a higher raw %.
        reasons = [{"text": c["text"], "pct": c["pct"]} for c in contribs[:3]]
        results.append({
            "id": pid, "score": score, "cosine": cos,
            "tags": [c["tag"] for c in tag_hits.get(pid, [])],
            "reasons": reasons,
        })
    results.sort(key=lambda r: -r["score"])
    return results[:k]


def similarity_breakdown(
    conn: sqlite3.Connection,
    owner_id: int,
    origin_id: int,
    current_id: int,
    dimension_weights: dict[str, float] | None = None,
) -> list[dict]:
    """Full facet-by-facet explanation of why `current` is similar to `origin`
    (§9, §13). One row per shared facet — tag, caption word, or visual — showing
    both photos' values and their match, sorted by how much each **drove** the
    similarity (its contribution), strongest first. This is what the tech panel
    renders so a weak match is self-evidently weak.

    Returns [{"param", "origin", "current", "match", "contrib"}].
    """
    weights = dimension_weights or DEFAULT_SIMILAR_DIMENSION_WEIGHTS
    total = conn.execute(
        "SELECT COUNT(*) AS n FROM photos WHERE owner_id = ?", (owner_id,)
    ).fetchone()["n"] or 1
    log_total = math.log(total) if total > 1 else 0.0

    def tag_scores(pid: int) -> dict[tuple[str, str], tuple[float, int]]:
        return {
            (r["dimension"], r["label"]): (r["score"], r["df"])
            for r in conn.execute(
                "SELECT t.dimension, t.label, MAX(pt.score) AS score,"
                " (SELECT COUNT(DISTINCT photo_id) FROM photo_tags WHERE tag_id = t.id) AS df"
                " FROM photo_tags pt JOIN tags t ON t.id = pt.tag_id"
                " WHERE pt.photo_id = ? GROUP BY t.id",
                (pid,),
            )
        }

    origin_tags, current_tags = tag_scores(origin_id), tag_scores(current_id)
    rows: list[dict] = []
    for (dim, label) in set(origin_tags) & set(current_tags):
        weight = weights.get(dim, 1.0)
        if weight <= 0.0:
            continue
        os_score, df = origin_tags[(dim, label)]
        cur_score, _ = current_tags[(dim, label)]
        idf = (math.log(total / (df or 1)) / log_total) if log_total > 0 else 1.0
        if idf <= 0.0:
            continue
        agreement = min(os_score, cur_score)
        rows.append({
            "param": f"{dim}: {label}", "origin": f"{round(os_score * 100)}%",
            "current": f"{round(cur_score * 100)}%", "match": round(agreement * 100),
            "contrib": weight * agreement * idf,
        })

    o_capvec, c_capvec = read_caption_vector(conn, origin_id), read_caption_vector(conn, current_id)
    if o_capvec is not None and c_capvec is not None:
        cap_cos = max(0.0, sum(a * b for a, b in zip(l2_normalize(o_capvec), l2_normalize(c_capvec))))
        rows.append({"param": "caption (meaning)", "origin": "—", "current": "—",
                     "match": round(cap_cos * 100), "contrib": _CAPTION_WEIGHT * cap_cos})

    o_vec, c_vec = read_vector(conn, origin_id), read_vector(conn, current_id)
    if o_vec is not None and c_vec is not None:
        cos = max(0.0, sum(a * b for a, b in zip(l2_normalize(o_vec), l2_normalize(c_vec))))
        rows.append({"param": "visual (image)", "origin": "—", "current": "—",
                     "match": round(cos * 100), "contrib": _VECTOR_WEIGHT * cos})

    # Sorted by contribution (relevance = importance × rarity × match), most
    # relevant first — so a generic style facet (composition/palette/light) sinks
    # below the subject/caption match even when its raw match % is higher.
    rows.sort(key=lambda r: -r["contrib"])
    return rows
