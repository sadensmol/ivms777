import math
import sqlite3
from dataclasses import replace

from embedding.base import Embedder
from embedding.store import all_caption_vectors, knn, read_caption_vector, read_vector, read_vectors
from embedding.vectors import l2_normalize
from search.moment import gap, read_moment
from search.scoring import (
    caption_contribution,
    image_contribution,
    moment_contribution,
    tag_contribution,
)
from search.signals import LOOSE, SIGNAL_OFF, STRICT, Gates, moment_strength
from search.signals import dimension_weights as tier_dimension_weights

# "similar" spans the photo's WHOLE character — what it is, where and when it was
# taken, and how it looks. Seven signals, each with one gate and one weight, composed
# by noisy-OR. `search/signals.py` owns the numbers and the reasoning behind them.


def search_photos(
    conn: sqlite3.Connection, embedder: Embedder, owner_id: int, query: str, k: int
) -> list[int]:
    """Photo ids best matching a natural-language query, best first.

    The query is **lower-cased** before it reaches SigLIP. Its text tower is trained
    on lower-case web captions and is case-sensitive in a way that actively misleads
    here: an upper-case word reads as text PRINTED IN the image, so "DOG" retrieved
    documents and ID cards while "dog" returned the actual dog photo at rank 1.
    Measured on the board — "show me all photos with DOG on it!" found nothing, the
    same sentence lower-cased found it first. Users type in whatever case they like,
    so normalising here fixes every caller at once (chat search, library search).
    """
    return [pid for pid, _ in search_photos_scored(conn, embedder, owner_id, query, k)]


def search_photos_scored(
    conn: sqlite3.Connection, embedder: Embedder, owner_id: int, query: str, k: int
) -> list[tuple[int, float]]:
    """`search_photos` plus each hit's RELATIVE strength, 0–1, best first.

    Strength is the hit's cosine divided by the best hit's, so 1.0 is "the closest
    match in this result set" and 0.5 is "half as close". It is deliberately NOT an
    absolute score. Measured on the real library, SigLIP's calibrated probability
    for a CORRECT top hit ranges 0.76 %–9.8 %, and raw top-1 cosines for subjects
    that ARE present (0.0889–0.1313) overlap those for subjects that are NOT
    (0.0137–0.0921) — "a birthday cake" (present, 0.0889) scores below "sushi on a
    plate" (absent, 0.0921). No global threshold separates them, because the
    magnitude tracks the PHRASING as much as the library. Within one query the
    ordering is still meaningful, so that is the only thing exposed.
    """
    if not query.strip():
        return []
    vector = l2_normalize(embedder.embed_texts([query.lower()])[0])
    hits = knn(conn, owner_id, vector, k)
    if not hits:
        return []
    # vec0 is L2 over unit vectors, so cos = 1 - d²/2.
    scored = [(pid, 1.0 - (distance * distance) / 2.0) for pid, distance in hits]
    best = max(cos for _, cos in scored)
    if best <= 0:
        return [(pid, 0.0) for pid, _ in scored]
    return [(pid, max(0.0, cos) / best) for pid, cos in scored]


# Per-dimension importance for "similar" (§9), expanded from the four tier weights in
# `search/signals.py` and overridable from vocab.yaml. Kept here so search/ has a sane
# default in tests and so callers keep passing a plain {dimension: weight} map.
DEFAULT_SIMILAR_DIMENSION_WEIGHTS = tier_dimension_weights()
# A cosine no real pair can reach — the ablation switch (§9.3). Passed as a gate it
# makes every comparison fall below the bar, which is exactly "no signal" for both the
# recall union and the scoring contribution.
CAPTION_SIGNAL_OFF = SIGNAL_OFF


def _tag_similarity(
    conn: sqlite3.Connection, owner_id: int, photo_id: int,
    weights: dict[str, float], gates: Gates = STRICT,
) -> dict[int, list[dict]]:
    """Per-shared-tag similarity contributions to every other photo (§9).

    Each shared tag is scored by `scoring.tag_contribution` — the tier's weight times
    the agreement (the WEAKER of the two photos' confidences: a 0.71 close-up matching
    a 1.00 close-up agree at 0.71, not 1.00) times an idf damp for common labels — and
    is dropped entirely below its tier's gate. Returns
    {pid: [{"text", "pct", "evidence", "content", "tag": (dim, label)}]}.
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
            contrib = tag_contribution(
                info["dimension"], info["label"], min(src_score, cand_score), idf,
                dim_weight, gates,
            )
            if contrib is not None:
                hits.setdefault(pid, []).append(contrib)
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
    min_cosine: float | None = None,
    caption_min: float | None = None,
    score_min: float | None = None,
    dimension_weights: dict[str, float] | None = None,
    use_captions: bool = True,
    loose: bool = False,
) -> list[dict]:
    """Photos similar to a given one, with the REASON each is similar (§9).

    A thin wrapper over the retriever core (§9.2, plan 12 task 3):
    ``refine(candidates(Query(seed_photo_id=photo_id)))``. Progressive, by what
    the pipeline has produced (graceful degradation):
      * no embedding yet -> nothing visual, but EXIF alone can still find a `moment`;
      * embedding only    -> visual neighbours (image-vector cosine above its gate);
      * + taxonomy        -> add shared tags, per-tier weighted;
      * + captions        -> add caption-meaning similarity (text-embedding cosine).

    `use_captions=False` (settings `similar_use_captions`, §9.3) removes the caption
    signal from BOTH halves — the candidate union and the scoring contribution — so
    the result is exactly "tags + image look-alike + moment". It is the ablation
    switch for measuring what the caption embedding is worth; not a tuning knob.

    Each facet is a scored **contribution** carrying `evidence` in the shared unit of
    `search/signals.py`; a candidate's score is `combine()` over them — a noisy-OR, so
    ONE strong match beats a PILE of weak ones and the result is a true 0–1. A
    candidate with no CONTENT signal (image, subject, caption, moment) is dropped
    however many style facets it shares. Each result's `reasons` are its top-3
    contributions by evidence, shown sorted by match %, highest first.

    The seed path is image-vector KNN dispatch ONLY — no planner, no SigLIP text
    encoder, no `client.embed` — so `embedder`/`client` are never constructed here;
    `candidates()`/`refine()` structurally cannot reach them for a seed query.

    Returns [{"id", "score", "cosine", "tags": [(dim, label)],
    "reasons": [{"text", "pct"}]}], best first.
    """
    # Local import: search.retriever imports _tag_similarity/_caption_similarity/
    # DEFAULT_SIMILAR_DIMENSION_WEIGHTS/search_photos FROM this module, so a
    # top-level import here would be circular. By call time both modules are
    # already fully loaded, so this is safe.
    from search.retriever import Query, candidates, refine

    weights = dimension_weights or DEFAULT_SIMILAR_DIMENSION_WEIGHTS
    gates = LOOSE if loose else STRICT
    if min_cosine is not None:
        gates = replace(gates, image=min_cosine)
    if caption_min is not None:
        gates = replace(gates, caption=caption_min)
    if not use_captions:
        # A cosine can never exceed 1.0, so an impossible gate silences the caption
        # signal in `_caption_similarity` (recall) AND `caption_contribution`
        # (scoring) without either of them growing a flag of its own (§9.3).
        gates = replace(gates, caption=CAPTION_SIGNAL_OFF)
    # The score floor stops a photo with nothing genuinely like it from filling the
    # strip with 12 fillers: below it, a candidate is not "similar", it is just the
    # best of a bad set (settings `similar_score_min`, §9).
    floor = gates.score_min if score_min is None else score_min
    query = Query(seed_photo_id=photo_id, k=k, weights=weights, floor=floor, gates=gates)
    ids = candidates(conn, None, owner_id, query)
    results = refine(conn, None, None, owner_id, query, ids)

    # The core's generic contract has no "cosine" field (search/chat/memory never
    # need it); similar's own contract always has — batch-read the FINAL results'
    # vectors (bounded to <= k, never one query per candidate) to add it back.
    seed_vector = read_vector(conn, photo_id)
    seed_vector = l2_normalize(seed_vector) if seed_vector is not None else None
    result_vectors = read_vectors(conn, [r["id"] for r in results])

    def _cosine_to_seed(pid: int) -> float:
        cand_vector = result_vectors.get(pid)
        if seed_vector is None or cand_vector is None:
            return 0.0
        return sum(a * b for a, b in zip(seed_vector, l2_normalize(cand_vector)))

    return [
        {"id": r["id"], "score": r["score"], "cosine": _cosine_to_seed(r["id"]),
         "tags": r["tags"], "reasons": r["reasons"]}
        for r in results
    ]


def similarity_breakdown(
    conn: sqlite3.Connection,
    owner_id: int,
    origin_id: int,
    current_id: int,
    dimension_weights: dict[str, float] | None = None,
    min_cosine: float | None = None,
    caption_min: float | None = None,
    use_captions: bool = True,
    loose: bool = False,
) -> list[dict]:
    """Full facet-by-facet explanation of why `current` is similar to `origin`
    (§9, §13). One row per facet that ACTUALLY SCORED — a shared tag, the caption
    meaning, the visual look-alike, the shared moment — showing both photos' values
    and their match, sorted strongest first. This is what the tech panel renders, so
    a weak match is self-evidently weak.

    A facet below its gate contributed nothing to the score, so it gets NO row: the
    panel and the ranking must never disagree about what drove a match.

    Returns [{"param", "origin", "current", "match", "contrib"}].
    """
    weights = dimension_weights or DEFAULT_SIMILAR_DIMENSION_WEIGHTS
    gates = LOOSE if loose else STRICT
    if min_cosine is not None:
        gates = replace(gates, image=min_cosine)
    if caption_min is not None:
        gates = replace(gates, caption=caption_min)
    if not use_captions:
        gates = replace(gates, caption=CAPTION_SIGNAL_OFF)
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
        contrib = tag_contribution(dim, label, agreement, idf, weight, gates)
        if contrib is None:
            continue  # below this dimension's gate — it drove nothing, so show nothing
        rows.append({
            "param": f"{dim}: {label}", "origin": f"{round(os_score * 100)}%",
            "current": f"{round(cur_score * 100)}%", "match": round(agreement * 100),
            "contrib": contrib["evidence"],
        })

    # Mirrors `similar_photos`: with the signal off the panel must not claim a row
    # that no longer drives the score (§9.3).
    o_capvec = read_caption_vector(conn, origin_id)
    c_capvec = read_caption_vector(conn, current_id)
    if o_capvec is not None and c_capvec is not None:
        cap_cos = max(0.0, sum(a * b for a, b in zip(l2_normalize(o_capvec), l2_normalize(c_capvec))))
        # Below the bar the caption contributes nothing to the score, so the panel
        # must not show a row for it. Above it, the row reports the SAME rescaled
        # strength the score used — never the raw cosine, whose 0.6-ish noise floor
        # would read as a strong match between two unrelated photos.
        contrib = caption_contribution(cap_cos, gates.caption)
        if contrib is not None:
            rows.append({"param": "caption (meaning)", "origin": "—", "current": "—",
                         "match": contrib["pct"], "contrib": contrib["evidence"]})

    o_vec, c_vec = read_vector(conn, origin_id), read_vector(conn, current_id)
    if o_vec is not None and c_vec is not None:
        cos = max(0.0, sum(a * b for a, b in zip(l2_normalize(o_vec), l2_normalize(c_vec))))
        # Same rule as the caption row: below its gate the look-alike signal did not
        # score, so the panel must not show it. SigLIP image cosines sit ~0.5–0.65 for
        # ANY two photos, so an ungated row read as "57% visual match" between two
        # unrelated photos — a number that drove nothing.
        contrib = image_contribution(cos, gates.image)
        if contrib is not None:
            rows.append({"param": "visual (image)", "origin": "—", "current": "—",
                         "match": contrib["pct"], "contrib": contrib["evidence"]})

    # The one signal that ignores the picture entirely (§9): same stretch of time,
    # same place. Shown with the actual gap so "same time & place" is checkable.
    origin_moment, current_moment = read_moment(conn, origin_id), read_moment(conn, current_id)
    if origin_moment is not None and current_moment is not None:
        hours, metres = gap(origin_moment, current_moment)
        contrib = moment_contribution(moment_strength(hours, metres), gates.moment)
        if contrib is not None:
            rows.append({
                "param": "same time & place", "match": contrib["pct"],
                "origin": _format_gap_time(hours),
                "current": _format_gap_place(metres),
                "contrib": contrib["evidence"],
            })

    # Sorted by what DROVE the match, strongest first (docs/ui.md). Not by match %:
    # `quality: sharp` agrees at 100% on almost every pair and drives ~0.01, so a
    # %-sorted panel led with it — the same complaint that started this rework, where
    # `light: low light 69%` headlined two photos that had nothing to do with
    # each other. Match % breaks ties.
    rows.sort(key=lambda r: (-r["contrib"], -r["match"]))
    return rows


def _format_gap_time(hours: float) -> str:
    """The time gap, said the way a person would read it."""
    minutes = hours * 60
    if minutes < 90:
        return f"{round(minutes)} min apart"
    if hours < 36:
        return f"{round(hours)} h apart"
    return f"{round(hours / 24)} days apart"


def _format_gap_place(metres: float | None) -> str:
    """The distance gap, or an honest "no GPS" when either photo lacks coordinates."""
    if metres is None:
        return "no GPS"
    if metres < 1000:
        return f"{round(metres)} m apart"
    return f"{metres / 1000:.1f} km apart"
