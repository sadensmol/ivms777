import sqlite3

from embedding.base import Embedder
from inference.client import InferenceClient
from inference.prompts import intent_messages
from search.fusion import reciprocal_rank_fusion
from search.keyword import keyword_search
from search.semantic import search_photos


def is_photo_question(client: InferenceClient, model: str, question: str) -> bool:
    """Yes/no gate: is this question about the photo library? (§10)

    Off-topic questions (advice, trivia) skip retrieval so the library is never
    dumped as false evidence. Strongly biased to answer: only a clear "no"
    refuses, and any error treats the question as on-topic — a small classifier
    that occasionally lets an off-topic question through is far better than one
    that blocks real photo searches.
    """
    try:
        answer = client.complete(model, intent_messages(question), timeout=10.0)
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
    """Top photo ids for a chat question, via semantic + keyword fusion (§9).

    The interactive-search path minus the sidebar filters: a raw question in,
    the most relevant photos out. Empty question or no matches -> [].
    """
    if not question.strip():
        return []
    semantic = search_photos(conn, embedder, owner_id, question, k=200)
    keyword = keyword_search(conn, owner_id, question, k=200)
    return reciprocal_rank_fusion([semantic, keyword])[:k]
