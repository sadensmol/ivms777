"""Additive contribution scoring for the retriever core (§9.2, plan 12 task 2).

Extracted-in-spirit from `similar_photos` (search/semantic.py): a candidate's score
is its scored facets (a shared tag, caption-meaning, an image look-alike, or — for a
text query with no seed image vector — its fused search rank) sorted high-to-low and
summed with a `_DECAY` falloff, so ONE strong match beats a PILE of weak ones. A
candidate must carry at least one CONTENT contribution (a subject tag, caption
meaning, or a genuine visual/search match) to qualify at all — style/scene signals
(vibe, light, palette…) only rerank a content match, they never create one.

This module imports the constants straight from `search.semantic` so a seed query
(similar) and a text query (search/retriever.py) score with the exact SAME numbers —
one scoring rule everywhere, per the plan's single-pipeline invariant.
"""

from search.semantic import _CAPTION_WEIGHT, _CONTENT_TAG_DIMENSIONS, _DECAY, _VECTOR_WEIGHT


def tag_contribution(
    dimension: str, label: str, agreement: float, idf: float, dimension_weight: float
) -> dict | None:
    """One tag-match contribution: `dimension_weight × agreement × idf` (§9).

    `agreement` is how strongly the tag applies (e.g. the weaker of two photos'
    confidences for a shared tag, or a soft planner hint's own confidence on the
    candidate). Returns None when the dimension is silenced (weight <= 0) or the
    tag is so common it carries no signal (idf <= 0) — mirrors `_tag_similarity`'s
    skip logic so a seed and a soft-tag hint are judged the same way.
    """
    if dimension_weight <= 0.0 or idf <= 0.0:
        return None
    return {
        "text": f"{dimension}: {label}",
        "pct": round(agreement * 100),
        "contrib": dimension_weight * agreement * idf,
        "tag": (dimension, label),
    }


def caption_contribution(cosine: float, min_cosine: float) -> dict | None:
    """The candidate's caption MEANS something close to the query/seed caption
    (§9) — a text-embedding cosine, not a shared word. None below `min_cosine`
    (the signal is simply absent below the bar, not a zero contribution)."""
    if cosine < min_cosine:
        return None
    return {
        "text": "caption (meaning)",
        "pct": round(cosine * 100),
        "contrib": _CAPTION_WEIGHT * cosine,
        "content": True,
    }


def image_contribution(cosine: float, min_cosine: float) -> dict | None:
    """A genuine visual near-dup — image-vector cosine (§9), a mild tiebreak, not
    every vaguely-similar image. None below `min_cosine`."""
    if cosine < min_cosine:
        return None
    return {
        "text": "looks alike",
        "pct": round(cosine * 100),
        "contrib": _VECTOR_WEIGHT * cosine,
        "content": True,
    }


def fusion_rank_contribution(rank: int, total: int) -> dict:
    """A text query has no seed image vector to compare against, so its rank in
    the semantic+keyword fusion (search/fusion.py RRF) is folded in AS the visual
    contribution instead — "search quality must not regress" (plan 12): the fused
    order is a contribution, never discarded. Rank 0 (best) scores like a strong
    cosine; the tail decays toward 0. Always CONTENT — a fused hit already matched
    the query by keyword or semantic meaning, so it earns the gate on its own.
    """
    frac = 1.0 - (rank / total) if total > 0 else 1.0
    return {
        "text": "matches your search",
        "pct": round(frac * 100),
        "contrib": _VECTOR_WEIGHT * frac,
        "content": True,
    }


def _is_content(contribution: dict) -> bool:
    if contribution.get("content"):
        return True
    tag = contribution.get("tag")
    return bool(tag) and tag[0] in _CONTENT_TAG_DIMENSIONS


def score_candidates(contributions: dict[int, list[dict]]) -> list[dict]:
    """Turn per-candidate contributions into ranked, gated, explained results (§9).

    For each candidate: sort its contributions high-to-low, decayed-sum them
    (`sum(c["contrib"] * _DECAY**i)` — the strongest facet counts fully, each next
    one less, so one strong match beats a heap of faint ones), and keep the top 3
    as `reasons`. A candidate without at least one CONTENT contribution (a subject
    tag, caption meaning, a visual match, or a fused search rank) is dropped —
    style/scene contributions alone never make a match.

    Returns `[{"id", "score", "reasons": [{"text","pct"}], "tags": [(dim,label)]}]`
    sorted best first.
    """
    results: list[dict] = []
    for photo_id, contribs in contributions.items():
        if not contribs or not any(_is_content(c) for c in contribs):
            continue
        ordered = sorted(contribs, key=lambda c: -c["contrib"])
        score = sum(c["contrib"] * (_DECAY**i) for i, c in enumerate(ordered))
        reasons = [{"text": c["text"], "pct": c["pct"]} for c in ordered[:3]]
        tags = [c["tag"] for c in ordered if c.get("tag")]
        results.append({"id": photo_id, "score": score, "reasons": reasons, "tags": tags})
    results.sort(key=lambda r: -r["score"])
    return results
