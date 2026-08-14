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


def build_memories(
    conn: sqlite3.Connection, client: InferenceClient, model: str, owner_id: int,
    *, force: bool = False, progress: Callable[[int, int], None] | None = None,
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
        memory = compose_memory(conn, client, model, owner_id, candidate, signature=signature)
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
