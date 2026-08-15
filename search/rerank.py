import json
import sqlite3

from embedding.store import read_caption_vector
from embedding.vectors import l2_normalize
from inference.client import InferenceClient

# Conservative default; tuned against a hand-labelled dev set in the bake-off
# (scripts/rerank_bakeoff.py, §17). Tests pass an explicit floor, never this.
RERANK_FLOOR = 0.4


def rerank(
    conn: sqlite3.Connection,
    query_vec: list[float],
    candidate_ids: list[int],
    *,
    floor: float,
) -> list[tuple[int, float]]:
    """Rerank candidates by caption-meaning cosine (§10). `query_vec` must be a
    vector in the caption embed space (the inference client's `caption_embed_model`);
    it is L2-normalized here.

    A candidate WITHOUT a caption vector has an UNAVAILABLE signal, not a zero one:
    it is kept — sunk below the scored matches, in input order — so a photo whose
    caption vector is not computed yet is never floored out of chat (the agent
    revalidates it). Only candidates that HAVE a vector are subject to `floor`.
    Returns (photo_id, cosine) best first; kept-but-unscored candidates report 0.0.
    """
    q = l2_normalize(query_vec)
    scored: list[tuple[int, float]] = []
    unscored: list[tuple[int, float]] = []
    for pid in candidate_ids:
        vec = read_caption_vector(conn, pid)
        if vec is None:
            unscored.append((pid, 0.0))  # signal unavailable — keep, don't floor
            continue
        cosine = sum(a * b for a, b in zip(q, l2_normalize(vec)))
        if cosine >= floor:
            scored.append((pid, cosine))
    scored.sort(key=lambda pair: -pair[1])
    return scored + unscored


def rerank_llm(
    client: InferenceClient,
    model: str,
    conn: sqlite3.Connection,
    query: str,
    candidate_ids: list[int],
    *,
    keep_min: int = 2,
) -> list[int]:
    """One batched relevance call: score each candidate's caption 0-3 against the
    query, keep those >= `keep_min`, in input order (§10, layer-1 LLM variant
    behind a flag). Any failure keeps the input order unchanged."""
    if not candidate_ids:
        return []
    lines: list[str] = []
    for pid in candidate_ids:
        row = conn.execute("SELECT caption FROM photos WHERE id = ?", (pid,)).fetchone()
        lines.append(f"{pid}: {(row['caption'] if row else None) or '(no caption)'}")
    system = (
        "Score how well each photo caption matches the query, 0 (no) to 3 (exact). "
        "Reply with ONLY a JSON object mapping the id (as a string) to its score."
    )
    user = f"Query: {query}\nPhotos:\n" + "\n".join(lines)
    try:
        raw = client.complete(
            model,
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            timeout=30.0,
        )
        start, end = raw.find("{"), raw.rfind("}")
        scores = json.loads(raw[start : end + 1]) if start != -1 and end > start else {}
        return [pid for pid in candidate_ids if int(scores.get(str(pid), 0)) >= keep_min]
    except (json.JSONDecodeError, ValueError, KeyError, TypeError):
        return list(candidate_ids)
