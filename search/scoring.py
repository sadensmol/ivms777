"""Turning signals into ranked, gated, explained results (§9.2).

Every retrieval path — a seed photo (`similar`) and a text query (`search/retriever.py`)
— builds the same kind of **contribution** here and composes it with the same rule, so
the two score identically by construction rather than by convention.

A contribution is one fact about a candidate:

```
{"text": "subject: dog", "pct": 88, "evidence": 0.51, "content": True, "tag": (...)}
```

* `evidence` — what the fact is worth, in the shared unit of `search/signals.py`.
  This is what ranks.
* `pct` — what the fact means to a human ("these two photos agree on `dog` at 88 %").
  This is what the UI shows, and it is deliberately NOT the evidence: agreement and
  rarity are different questions, and the panel should answer the first one.
* `content` — may this fact make two photos similar ALL BY ITSELF? A shared
  `subject`, a caption that means the same, a visual near-dup, or the same moment
  can. Style and scene facets cannot: two photos both shot top-down in cool overcast
  light are not "the same thing", however many such facets they share.
"""

from search.signals import (
    CAPTION_GATE,
    CAPTION_WEIGHT,
    IMAGE_GATE,
    IMAGE_WEIGHT,
    MOMENT_GATE,
    MOMENT_WEIGHT,
    RANK_WEIGHT,
    TAG_TIERS,
    combine,
    content_dimensions,
    cosine_strength,
    dimension_gates,
    tag_strength,
)

_TAG_GATES = dimension_gates()
_CONTENT_DIMENSIONS = content_dimensions()
_DEFAULT_TAG_GATE = TAG_TIERS["look"].gate


def tag_contribution(
    dimension: str, label: str, agreement: float, idf: float, dimension_weight: float
) -> dict | None:
    """One shared tag (§9). None when the dimension is silenced, the tag is on every
    photo, or the two photos do not agree strongly enough for this dimension's gate.

    `agreement` is how strongly the tag applies to BOTH photos — the weaker of their
    two confidences for a shared tag, or a soft planner hint's own confidence on the
    candidate — so a seed and a hint are judged by the same bar.
    """
    if dimension_weight <= 0.0 or idf <= 0.0:
        return None
    gate = _TAG_GATES.get(dimension, _DEFAULT_TAG_GATE)
    strength = tag_strength(agreement, idf, gate)
    if strength is None:
        return None
    return {
        "text": f"{dimension}: {label}",
        "pct": round(agreement * 100),
        "evidence": dimension_weight * strength,
        "content": dimension in _CONTENT_DIMENSIONS,
        "tag": (dimension, label),
    }


def caption_contribution(cosine: float, gate: float = CAPTION_GATE) -> dict | None:
    """The candidate's caption MEANS something close to the query/seed caption (§9) —
    a text-embedding cosine, not a shared word."""
    strength = cosine_strength(cosine, gate)
    if strength is None:
        return None
    return {
        "text": "caption (meaning)",
        "pct": round(strength * 100),
        "evidence": CAPTION_WEIGHT * strength,
        "content": True,
    }


def image_contribution(cosine: float, gate: float = IMAGE_GATE) -> dict | None:
    """A genuine visual near-dup — the image-vector cosine (§9), and the heaviest
    signal there is: above the gate it is the top 4 % of all pairs."""
    strength = cosine_strength(cosine, gate)
    if strength is None:
        return None
    return {
        "text": "looks alike",
        "pct": round(strength * 100),
        "evidence": IMAGE_WEIGHT * strength,
        "content": True,
    }


def moment_contribution(strength: float, gate: float = MOMENT_GATE) -> dict | None:
    """Same stretch of time, same place (§9) — the one signal that does not look at
    the picture. `strength` comes from `search/moment.py`."""
    ramped = cosine_strength(strength, gate)
    if ramped is None:
        return None
    return {
        "text": "same time & place",
        "pct": round(ramped * 100),
        "evidence": MOMENT_WEIGHT * ramped,
        "content": True,
    }


def fusion_rank_contribution(rank: int, total: int) -> dict:
    """A text query has no seed image vector to compare against, so its rank in the
    semantic+keyword fusion (`search/fusion.py`) stands in for the image signal and
    inherits its weight. Always content — a fused hit already matched the query by
    keyword or by meaning, so it earns the gate on its own.
    """
    frac = 1.0 - (rank / total) if total > 0 else 1.0
    return {
        "text": "matches your search",
        "pct": round(frac * 100),
        "evidence": RANK_WEIGHT * frac,
        "content": True,
    }


def score_candidates(contributions: dict[int, list[dict]]) -> list[dict]:
    """Rank, gate, and explain (§9).

    A candidate without a single CONTENT contribution is dropped — style and scene
    facets only ever rerank. The rest are scored by `signals.combine` (noisy-OR), so
    the score is a true 0–1 and a pile of weak facets can never add up to a match.

    Returns `[{"id", "score", "reasons": [{"text","pct"}], "tags": [(dim,label)]}]`,
    best first.
    """
    results: list[dict] = []
    for photo_id, contribs in contributions.items():
        if not contribs or not any(c.get("content") for c in contribs):
            continue
        ordered = sorted(contribs, key=lambda c: -c["evidence"])
        score = combine([c["evidence"] for c in ordered])
        # Reasons are the top 3 BY EVIDENCE, shown in that order — what actually drove
        # the match leads (§13). Ordering them by match % instead put `quality: sharp
        # 100%` at the top of 403 result cards: `sharp` is on 201/206 photos, so it
        # agrees perfectly and means nothing. A big percentage is not a big reason.
        results.append({
            "id": photo_id,
            "score": score,
            "reasons": [{"text": c["text"], "pct": c["pct"]} for c in ordered[:3]],
            "tags": [c["tag"] for c in ordered if c.get("tag")],
        })
    results.sort(key=lambda r: -r["score"])
    return results
