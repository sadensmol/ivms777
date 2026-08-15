"""The one retrieval pipeline (§9.2, plan 12) — used by search, similar, chat, and
memory alike, so nothing else scores/fuses/narrows/floors photos on its own.

Two stages: `candidates()` (fast, no scoring, no LLM) then `refine()` (the graceful
additive scoring extracted into `search/scoring.py`). `retrieve() = refine(candidates())`
is the synchronous convenience; interactive callers (similar's two-phase UI) call the
stages separately so first paint never waits on the richer stage.

The ADR this module encodes: HARD filters (EXIF facets + explicit date, §6.2) gate —
a non-matching photo is OUT before scoring, exact. Every model-derived signal (tags,
caption meaning, image look-alike, fused search rank) is SOFT — an additive
contribution that is simply *unavailable*, never zero, when its source data is
missing, so one missing signal drops only that contribution, never the candidate.
"""

import math
import sqlite3
from dataclasses import dataclass, field

from embedding.base import Embedder
from embedding.store import knn, read_caption_vector, read_vector
from embedding.vectors import l2_normalize
from inference.client import InferenceClient
from search.dates import date_where
from search.facets import build_where, parse_filters
from search.fusion import reciprocal_rank_fusion
from search.keyword import keyword_search
from search.scoring import (
    caption_contribution,
    fusion_rank_contribution,
    image_contribution,
    score_candidates,
    tag_contribution,
)
from search.semantic import (
    DEFAULT_SIMILAR_DIMENSION_WEIGHTS,
    _caption_similarity,
    _tag_similarity,
    search_photos,
)


@dataclass
class Query:
    """A retrieval request (§9.2). Exactly one of `text` / `seed_photo_id` is set —
    `text` for search/chat/memory theme discovery, `seed_photo_id` for similar/memory
    "more like this".

    `hard_filters` are sidebar-style params (`f_*`/`n_*`/`date_from`/`date_to`) — EXIF
    facets plus explicit date, applied as an EXACT cut before scoring (§6.2, the ADR).
    `soft_tags` are planner tag hints (`{dimension: [label, ...]}`) that only SCORE,
    never gate — the fix for the §10 bug where a hinted-but-untaxonomised tag used to
    empty the result. `floor` is a caller-set relevance cut; None ranks everyone
    (search/similar), a chat caller sets it for an honest-empty answer.
    """

    text: str | None = None
    seed_photo_id: int | None = None
    hard_filters: dict[str, str] = field(default_factory=dict)
    soft_tags: dict[str, list[str]] = field(default_factory=dict)
    k: int = 30
    weights: dict[str, float] | None = None
    floor: float | None = None


def candidates(
    conn: sqlite3.Connection,
    embedder: Embedder,
    owner_id: int,
    query: Query,
    *,
    caption_min: float = 0.6,
) -> list[int]:
    """Phase 1 — fast candidate generation (§9.2). No scoring, no LLM.

    A SEED query is the same UNION `similar_photos` has always used — image-vector
    KNN alone is NARROWER: a photo sharing a strong `subject` tag, or whose caption
    means the same thing, can sit outside the seed's top-K visual neighbours yet
    still be "similar" (plan 12 task 3 parity fix). All three sources are cheap
    SQL / local scans (`knn`, `_tag_similarity`, `_caption_similarity`) — it must
    never touch the planner, the SigLIP text encoder, or keyword search (the
    "similar stays incredible-fast" guarantee, plan 12). A TEXT query fuses
    semantic (SigLIP text->image KNN) and keyword (FTS) via reciprocal-rank fusion.
    """
    if query.seed_photo_id is not None:
        seed_id = query.seed_photo_id
        vector = read_vector(conn, seed_id)
        if vector is None:
            return []  # no embedding yet: nothing to compare against
        vector = l2_normalize(vector)
        knn_ids = [pid for pid, _ in knn(conn, owner_id, vector, max(query.k, 30), exclude_id=seed_id)]
        weights = query.weights or DEFAULT_SIMILAR_DIMENSION_WEIGHTS
        tag_ids = set(_tag_similarity(conn, owner_id, seed_id, weights))
        caption_ids = set(_caption_similarity(conn, owner_id, seed_id, caption_min))
        seen = set(knn_ids)
        extra = [pid for pid in (tag_ids | caption_ids) if pid not in seen]
        return knn_ids + extra
    if query.text and query.text.strip():
        k = max(query.k, 30)
        semantic = search_photos(conn, embedder, owner_id, query.text, k=k)
        keyword = keyword_search(conn, owner_id, query.text, k=k)
        return reciprocal_rank_fusion([semantic, keyword])
    return []


def refine(
    conn: sqlite3.Connection,
    embedder: Embedder,
    client: InferenceClient,
    owner_id: int,
    query: Query,
    ids: list[int],
    *,
    caption_model: str = "",
    min_cosine: float = 0.5,
    caption_min: float = 0.6,
) -> list[dict]:
    """Phase 2 — hard pre-filter, then graceful additive scoring (§9.2).

    (1) `hard_filters` cut `ids` EXACTLY — a non-matching photo is removed before
    any scoring happens; tags are never a hard filter (see `_soft_tag_contributions`).
    (2) Build each surviving id's contributions — shared tags / caption meaning /
    image look-alike vs the seed, or caption meaning / fused rank / soft-tag hints
    for a text query — via `search/scoring.py` so both paths score identically.
    A missing per-photo signal (no caption_vec, no tag, no vector) just drops that
    ONE contribution, never the candidate (mirrors `search/rerank.py`).
    (3) Score via `scoring.score_candidates` (decayed sum, content gate, top-3
    reasons). (4) `query.floor`, if set, cuts by the final score; None ranks
    everyone — a floor never removes a candidate on a signal it never had a chance
    to score on, because that candidate's score already reflects only what it has.
    """
    ids = _hard_filter(conn, owner_id, query.hard_filters, ids)
    if not ids:
        return []
    weights = query.weights or DEFAULT_SIMILAR_DIMENSION_WEIGHTS
    if query.seed_photo_id is not None:
        contributions = _seed_contributions(
            conn, owner_id, query.seed_photo_id, ids, weights, min_cosine, caption_min
        )
    else:
        contributions = _text_contributions(
            conn, client, owner_id, query, ids, weights, caption_model, caption_min
        )
    results = score_candidates(contributions)
    if query.floor is not None:
        results = [r for r in results if r["score"] >= query.floor]
    return results[: query.k]


def retrieve(
    conn: sqlite3.Connection,
    embedder: Embedder,
    client: InferenceClient,
    owner_id: int,
    query: Query,
    *,
    caption_model: str = "",
    min_cosine: float = 0.5,
    caption_min: float = 0.6,
) -> list[dict]:
    """`retrieve() = refine(candidates())` — the synchronous convenience for search
    and memory. Chat/similar's two-phase UX calls the two stages separately."""
    ids = candidates(conn, embedder, owner_id, query, caption_min=caption_min)
    return refine(
        conn,
        embedder,
        client,
        owner_id,
        query,
        ids,
        caption_model=caption_model,
        min_cosine=min_cosine,
        caption_min=caption_min,
    )


def _hard_filter(
    conn: sqlite3.Connection, owner_id: int, hard_filters: dict[str, str], ids: list[int]
) -> list[int]:
    """EXIF facets + explicit date — the only HARD gate (§6.2, the ADR). An exact
    cut before scoring; a photo that fails it is OUT, no matter how well it would
    have scored on soft signals."""
    if not ids or not hard_filters:
        return ids
    facet_where, facet_params = build_where(parse_filters(hard_filters))
    date_frag, date_params = date_where(hard_filters)
    where = facet_where + date_frag
    if not where:
        return ids
    placeholders = ", ".join("?" for _ in ids)
    allowed = {
        row["id"]
        for row in conn.execute(
            "SELECT p.id FROM photos p WHERE p.owner_id = ?"
            + where
            + f" AND p.id IN ({placeholders})",
            (owner_id, *facet_params, *date_params, *ids),
        )
    }
    return [pid for pid in ids if pid in allowed]


def _seed_contributions(
    conn: sqlite3.Connection,
    owner_id: int,
    seed_id: int,
    ids: list[int],
    weights: dict[str, float],
    min_cosine: float,
    caption_min: float,
) -> dict[int, list[dict]]:
    """A seed's contributions: shared tags, caption meaning, and image look-alike —
    the exact `similar_photos` math (§9), reused so the two paths agree by
    construction rather than by convention."""
    ids_set = set(ids)
    contributions: dict[int, list[dict]] = {pid: [] for pid in ids}

    for pid, contribs in _tag_similarity(conn, owner_id, seed_id, weights).items():
        if pid in ids_set:
            contributions[pid].extend(contribs)

    for pid, cosine in _caption_similarity(conn, owner_id, seed_id, caption_min).items():
        if pid in ids_set:
            contrib = caption_contribution(cosine, caption_min)
            if contrib is not None:
                contributions[pid].append(contrib)

    seed_vec = read_vector(conn, seed_id)
    if seed_vec is not None:
        seed_vec = l2_normalize(seed_vec)
        for pid in ids:
            cand_vec = read_vector(conn, pid)
            if cand_vec is None:
                continue  # no embedding yet — skip this ONE contribution, keep the candidate
            cosine = sum(a * b for a, b in zip(seed_vec, l2_normalize(cand_vec)))
            contrib = image_contribution(cosine, min_cosine)
            if contrib is not None:
                contributions[pid].append(contrib)

    return contributions


def _text_contributions(
    conn: sqlite3.Connection,
    client: InferenceClient,
    owner_id: int,
    query: Query,
    ids: list[int],
    weights: dict[str, float],
    caption_model: str,
    caption_min: float,
) -> dict[int, list[dict]]:
    """A text query's contributions: caption meaning vs the query (in the
    inference-client's caption embed space, §9 two-embedding-spaces rule), the
    fused semantic+keyword rank folded in as the visual contribution (no seed image
    vector to compare against), and soft planner tag hints — score-only, never a
    gate (the §10 fix this plan makes structural)."""
    contributions: dict[int, list[dict]] = {pid: [] for pid in ids}

    query_vec = l2_normalize(client.embed(caption_model, [query.text])[0])
    for pid in ids:
        cap_vec = read_caption_vector(conn, pid)
        if cap_vec is None:
            continue  # caption vector not computed yet — unavailable, not zero
        cosine = sum(a * b for a, b in zip(query_vec, l2_normalize(cap_vec)))
        contrib = caption_contribution(cosine, caption_min)
        if contrib is not None:
            contributions[pid].append(contrib)

    total = len(ids)
    for rank, pid in enumerate(ids):
        contributions[pid].append(fusion_rank_contribution(rank, total))

    if query.soft_tags:
        for pid in ids:
            contributions[pid].extend(
                _soft_tag_contributions(conn, owner_id, pid, query.soft_tags, weights)
            )

    return contributions


def _soft_tag_contributions(
    conn: sqlite3.Connection,
    owner_id: int,
    photo_id: int,
    soft_tags: dict[str, list[str]],
    weights: dict[str, float],
) -> list[dict]:
    """Planner tag hints matched against a candidate's OWN tags — SOFT, they only
    score (§10, the ADR). A hint the taxonomy never scored simply contributes
    nothing for that candidate; it can never empty the result the way a hard
    filter would (that was the §10 bug this plan supersedes)."""
    total = (
        conn.execute("SELECT COUNT(*) AS n FROM photos WHERE owner_id = ?", (owner_id,)).fetchone()[
            "n"
        ]
        or 1
    )
    log_total = math.log(total) if total > 1 else 0.0
    contribs: list[dict] = []
    for dimension, labels in soft_tags.items():
        dim_weight = weights.get(dimension, 1.0)
        if dim_weight <= 0.0 or not labels:
            continue
        placeholders = ", ".join("?" for _ in labels)
        rows = conn.execute(
            "SELECT t.label AS label, MAX(pt.score) AS score,"
            " (SELECT COUNT(DISTINCT photo_id) FROM photo_tags WHERE tag_id = t.id) AS df"
            " FROM photo_tags pt JOIN tags t ON t.id = pt.tag_id"
            f" WHERE pt.photo_id = ? AND t.dimension = ? AND t.label IN ({placeholders})"
            " GROUP BY t.label",
            (photo_id, dimension, *labels),
        ).fetchall()
        for row in rows:
            idf = (math.log(total / (row["df"] or 1)) / log_total) if log_total > 0 else 1.0
            contrib = tag_contribution(dimension, row["label"], row["score"], idf, dim_weight)
            if contrib is not None:
                contribs.append(contrib)
    return contribs
