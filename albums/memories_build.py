import logging
import sqlite3
from collections.abc import Callable

from albums.compose import compose_memory
from albums.memory_store import (
    Memory,
    current_signature,
    photo_ids_for_group,
    replace_memories,
    rewrite_memory,
    stored_signature,
)
from albums.seeds import Candidate, seed_candidates
from inference.client import InferenceClient

logger = logging.getLogger(__name__)


def build_memories(
    conn: sqlite3.Connection, client: InferenceClient, model: str, owner_id: int,
    *, force: bool = False, progress: Callable[[int, int], None] | None = None,
    use_captions: bool = True,
) -> int:
    """Seed candidates, curate each with the agent, and store the kept memories.

    Signature-guarded: when the library has not changed since the last build and
    `force` is false, it does no model work and returns the stored count (§11).
    `progress(done, total)` is called after each candidate so the UI can show a
    percent — the composing loop is the whole cost, one model call per candidate.
    """
    signature = current_signature(conn, owner_id)
    if not force and stored_signature(conn, owner_id) == signature:
        return _stored_count(conn, owner_id)
    candidates = seed_candidates(conn, owner_id)
    total = len(candidates)
    if progress is not None:
        progress(0, total)
    memories: list[Memory] = []
    for index, candidate in enumerate(candidates):
        # Per-candidate isolation: composing one memory involves a model call and
        # several read-only tool queries, and ANY of them failing must cost only
        # that candidate. Without this an exception escaped to the caller — which
        # is a bare `threading.Thread` (§11) — killing the rebuild mid-run, so the
        # remaining candidates were never tried and nothing was stored.
        try:
            memory = compose_memory(conn, client, model, owner_id, candidate,
                                    signature=signature, use_captions=use_captions)
        except Exception:  # one candidate must never abort the build
            logger.exception("composing memory %d/%d failed; skipping it", index + 1, total)
            memory = None
        if memory is not None:
            memories.append(memory)
        if progress is not None:
            progress(index + 1, total)
    replace_memories(conn, owner_id, memories)
    return len(memories)


def rebuild_memory(
    conn: sqlite3.Connection, client: InferenceClient, model: str, owner_id: int,
    group_id: int, *, use_captions: bool = True,
) -> Memory | None:
    """Retell ONE memory: a new title, a new description, a new photo order.

    Rebuilding every memory is a model call per candidate over the whole library, and
    it re-seeds the clusters, so a memory you liked can come back split or merged. This
    asks the same agent one question about one memory and keeps everything else
    untouched — including the memory's own identity, so the viewer's notes survive it
    (§11.3) and steer the retelling.

    The candidate is the memory's CURRENT photographs, so the agent is re-ordering and
    re-describing what is already there rather than re-clustering the library.

    Returns the new memory, or None when the agent could not produce one — in which
    case nothing is written and the old memory stands.
    """
    photo_ids = photo_ids_for_group(conn, group_id)
    if len(photo_ids) < 2:
        return None
    memory = compose_memory(
        conn, client, model, owner_id,
        Candidate(photo_ids=photo_ids),
        signature=current_signature(conn, owner_id),
        use_captions=use_captions,
    )
    if memory is None:
        return None
    rewrite_memory(conn, group_id, memory)
    return memory


def _stored_count(conn: sqlite3.Connection, owner_id: int) -> int:
    return conn.execute(
        "SELECT count(*) AS n FROM groups WHERE owner_id = ? AND kind = 'memory'",
        (owner_id,),
    ).fetchone()["n"]
