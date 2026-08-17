import logging
import sqlite3
from collections.abc import Callable

from albums.compose import compose_memory
from albums.memory_store import (
    Memory,
    current_signature,
    replace_memories,
    stored_signature,
)
from albums.seeds import seed_candidates
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


def _stored_count(conn: sqlite3.Connection, owner_id: int) -> int:
    return conn.execute(
        "SELECT count(*) AS n FROM groups WHERE owner_id = ? AND kind = 'memory'",
        (owner_id,),
    ).fetchone()["n"]
