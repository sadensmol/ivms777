import json
import logging
import sqlite3

from albums.memory_store import Memory
from albums.seeds import Candidate
from inference.client import ChatMessage, InferenceClient
from search.semantic import similar_photos

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You compose ONE memory from a set of photos taken close together in time and "
    "place. Read their summaries (date, place, caption, tags). Decide if they form "
    "one coherent memory — a place, a day, an occasion. "
    "Reply with ONLY a JSON object, no prose. To ask for more context first, use "
    '{\"action\":\"expand\",\"tool\":\"similar|facets|nearby\",\"photo_id\":<id>}. '
    "When ready, answer with "
    '{\"action\":\"answer\",\"keep\":<bool>,\"title\":\"...\",\"description\":\"...\",'
    '\"drop_photo_ids\":[...]}. '
    "The title and description must use ONLY facts in the summaries — real dates, "
    "the place, what the captions show. Never invent people, names, or events. If "
    "the photos are unrelated (they merely happened close in time), set keep=false. "
    "Drop any photo that does not belong via drop_photo_ids."
)

# The prompt must fit the SMALLEST context we run: jetson's `llm_ctx: 2048` (§3.1,
# deliberately small to shrink gemma's resident footprint on the 8 GB board). At
# ~40 tokens per photo line, 24 lines plus the system prompt and the agent's own
# tool rounds leaves comfortable room; unbounded, a large cluster produced a 3326-
# token request that llama-server rejected outright. Captions are the long part of
# a line, so they are clipped too — a caption is one sentence in practice, and the
# cap only bites on pathological ones.
MAX_SUMMARY_PHOTOS = 24
MAX_CAPTION_CHARS = 160


def compose_memory(
    conn: sqlite3.Connection,
    client: InferenceClient,
    model: str,
    owner_id: int,
    candidate: Candidate,
    *,
    signature: str,
    max_rounds: int = 3,
    use_captions: bool = True,
) -> Memory | None:
    """Curate + narrate one candidate via a bounded agent loop (§11, §9.1).

    Returns a Memory when the agent keeps the candidate, or None when it skips
    it, drops it below two photos, or emits output we cannot use.
    """
    messages: list[ChatMessage] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _summarise(conn, owner_id, candidate.photo_ids)},
    ]
    for round_no in range(max_rounds + 1):
        force_answer = round_no == max_rounds
        turn = _complete(client, model, messages, force_answer)
        if turn is None:
            return None
        if turn.get("action") == "expand" and not force_answer:
            messages.append({"role": "assistant", "content": json.dumps(turn)})
            messages.append(
                {"role": "user", "content": _run_tool(conn, owner_id, turn, use_captions)}
            )
            continue
        if turn.get("action") != "answer" or not turn.get("keep"):
            return None
        dropped = set(turn.get("drop_photo_ids") or [])
        kept = [pid for pid in candidate.photo_ids if pid not in dropped]
        title = (turn.get("title") or "").strip()
        description = (turn.get("description") or "").strip()
        if len(kept) < 2 or not title or not description:
            return None
        return Memory(title, description, kept, signature)
    return None


def _summarise(
    conn: sqlite3.Connection, owner_id: int, photo_ids: list[int], *, limit: int = MAX_SUMMARY_PHOTOS
) -> str:
    """One compact line per photo: id, date, place, caption, top tags (~40 tokens).

    **Capped at `limit` lines.** A seed candidate is every photo in a time/place
    cluster and has no upper bound — a busy day can hold a hundred. At ~40 tokens
    each that overruns the jetson's 2048-token context and `llama-server` rejects
    the whole request ("request (3326 tokens) exceeds the available context size"),
    which used to abort the entire rebuild. The agent only needs enough of the
    cluster to judge whether it is one coherent memory, so the summary is truncated
    and says so — the omitted photos are still kept in the memory, they just do not
    each get a line in the prompt.
    """
    lines: list[str] = []
    omitted = max(0, len(photo_ids) - limit)
    for photo_id in photo_ids[:limit]:
        photo = conn.execute(
            "SELECT shot_at, gps_lat, gps_lon, caption FROM photos WHERE id = ? AND owner_id = ?",
            (photo_id, owner_id),
        ).fetchone()
        if photo is None:
            continue
        place = (
            f"{photo['gps_lat']:.2f},{photo['gps_lon']:.2f}"
            if photo["gps_lat"] is not None and photo["gps_lon"] is not None
            else "no location"
        )
        tags = ", ".join(
            r["label"] for r in conn.execute(
                "SELECT t.label FROM photo_tags pt JOIN tags t ON t.id = pt.tag_id"
                " WHERE pt.photo_id = ? ORDER BY pt.score DESC LIMIT 4",
                (photo_id,),
            )
        )
        caption = (photo["caption"] or "")[:MAX_CAPTION_CHARS]
        lines.append(
            f"[{photo_id}] {photo['shot_at'] or 'no date'} · {place} · "
            f"{caption}" + (f" · tags: {tags}" if tags else "")
        )
    summary = "Photos:\n" + "\n".join(lines)
    if omitted:
        # Say it outright: the agent must not read the truncation as "the cluster
        # is only this big" and judge coherence on a partial view.
        summary += f"\n(+{omitted} more photos in this cluster, not listed)"
    return summary


def _run_tool(
    conn: sqlite3.Connection, owner_id: int, turn: dict, use_captions: bool = True
) -> str:
    """Run one read-only expansion tool; always returns a short text block."""
    tool = turn.get("tool")
    photo_id = turn.get("photo_id")
    if tool == "similar" and isinstance(photo_id, int):
        ids = [
            r["id"]
            for r in similar_photos(conn, owner_id, photo_id, k=5, use_captions=use_captions)
        ]
        return "Similar photos: " + (_summarise(conn, owner_id, ids) if ids else "(none)")
    if tool == "facets" and isinstance(photo_id, int):
        facts = [
            f"{r['key']}={r['value_text'] if r['value_text'] is not None else r['value_num']}"
            for r in conn.execute(
                "SELECT key, value_text, value_num FROM photo_facets WHERE photo_id = ?",
                (photo_id,),
            )
        ]
        return f"Facets for [{photo_id}]: " + (", ".join(facts) or "(none)")
    if tool == "nearby" and isinstance(photo_id, int):
        near = [
            r["id"] for r in conn.execute(
                "SELECT p2.id FROM photos p1 JOIN photos p2 ON p2.owner_id = p1.owner_id"
                " WHERE p1.id = ? AND p2.id != p1.id AND p2.shot_at IS NOT NULL"
                " AND abs(julianday(p2.shot_at) - julianday(p1.shot_at)) < 0.25"
                " ORDER BY p2.shot_at LIMIT 5",
                (photo_id,),
            )
        ]
        return "Nearby in time: " + (_summarise(conn, owner_id, near) if near else "(none)")
    return "(no result)"


def _complete(
    client: InferenceClient, model: str, messages: list[ChatMessage], force_answer: bool
) -> dict | None:
    """One model turn -> parsed JSON dict, or None if the output is unusable.

    No strict json_schema (Ollama's strict mode is unreliable with the mixed
    expand/answer shape); the reply is expected to be a bare JSON object and
    parsed leniently. Anything unparseable skips the candidate (§11).

    A FAILED CALL SKIPS THIS CANDIDATE — it never aborts the build. The catch used
    to list only the parse errors, so an `httpx.HTTPStatusError` (a 500 from
    `/text/complete`, e.g. a prompt over the context limit) propagated out through
    `build_memories` and killed the background thread: one bad cluster and the
    whole rebuild died with nothing stored. The failure is logged rather than
    swallowed silently, because a rebuild that quietly keeps zero memories is
    indistinguishable from a library with no memories in it.
    """
    turn_messages = messages
    if force_answer:
        turn_messages = [
            *messages,
            {"role": "user", "content": "Answer now with a keep/skip decision (action=answer)."},
        ]
    try:
        raw = client.complete(model, turn_messages, timeout=60.0)
    except Exception:  # one candidate must never abort the build
        logger.exception("memory composition call failed; skipping this candidate")
        return None
    try:
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end <= start:
            return None
        parsed = json.loads(raw[start : end + 1])
        return parsed if isinstance(parsed, dict) else None
    except (json.JSONDecodeError, ValueError, KeyError):
        return None
