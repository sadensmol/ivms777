import sqlite3

from embedding.base import Embedder
from inference.client import InferenceClient
from inference.prompts import intent_messages
from search.retriever import Query, candidates


def is_photo_question(client: InferenceClient, model: str, question: str) -> bool:
    """Yes/no gate: is this question about the photo library? (§10)

    Off-topic questions (advice, trivia) skip retrieval so the library is never
    dumped as false evidence. Strongly biased to answer: only a clear "no"
    refuses, and any error treats the question as on-topic — a small classifier
    that occasionally lets an off-topic question through is far better than one
    that blocks real photo searches.
    """
    try:
        answer = client.complete(model, intent_messages(question), timeout=10.0, temperature=0)
        return not answer.strip().lower().startswith("n")
    except Exception:  # noqa: BLE001 - the gate is best-effort; default to answering
        return True


def retrieve(
    conn: sqlite3.Connection,
    embedder: Embedder,
    owner_id: int,
    question: str,
    k: int = 30,
) -> list[int]:
    """Top photo ids for a chat question — the documented outer fallback (§10) for
    when the precise planner -> core -> rerank path throws.

    Routes through the retriever core's candidate stage
    (`search/retriever.py::candidates`, §9.2) so fusion lives in exactly ONE place
    (plan 12 single-pipeline invariant): the same semantic + keyword RRF, minus the
    sidebar filters and the caption-cosine floor. Empty question or no matches -> [].
    """
    if not question.strip():
        return []
    return candidates(conn, embedder, owner_id, Query(text=question, k=200))[:k]
