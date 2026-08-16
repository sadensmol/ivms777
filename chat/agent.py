import json
import re
import sqlite3

from chat.retrieve import retrieve as fusion_retrieve
from embedding.base import Embedder
from embedding.vectors import l2_normalize
from inference.client import InferenceClient
from search.keyword import keyword_search
from search.planner import plan, spec_to_params
from search.rerank import RERANK_FLOOR, rerank
from search.retriever import Query, _hard_filter, candidates
from search.semantic import search_photos, similar_photos

# The agent only handles the SEMANTIC tail — open-ended photo search. Counts,
# memories, and periods are answered deterministically by `direct_answer` BEFORE
# the lease (§10), so the loop needs no fact/memory tools; its whole job is to pull
# and verify candidate photos. Tool-calls are schema-constrained (see
# `_TOOL_CALL_SCHEMA`), so the model can only emit a valid action naming a real
# tool — the "malformed JSON / made-up tool" failure mode is gone.
_AGENT_SYSTEM = (
    "You find the photos in a personal library that answer the user's question. "
    "You are given candidate photos (id, date, caption, tags). Return ONLY the ids "
    "whose caption/tags actually match the question — verify each one; never invent "
    "a match. Reply with ONLY a JSON object.\n"
    "To pull more candidates, expand with a tool:\n"
    '- {"action":"expand","tool":"search","query":"..."} — more photos matching a phrase.\n'
    '- {"action":"expand","tool":"similar","photo_id":<id>} — photos that look like that one.\n'
    '- {"action":"expand","tool":"nearby","photo_id":<id>} — photos taken around the same time.\n'
    'When ready, answer with {"action":"answer","photo_ids":[...]} — the ids you verified, '
    "or an empty list if none match.\n"
    "Never state a count or total you did not actually count; speak only about the "
    "photos shown to you.\n"
    "Examples:\n"
    'Q: dogs on a beach → {"action":"expand","tool":"search","query":"dog on a beach"}\n'
    'then → {"action":"answer","photo_ids":[12,40]}\n'
    'Q: nothing fits → {"action":"answer","photo_ids":[]}'
)

# Strict tool-call shape (OpenAI structured-output compatible: every key required,
# optionals made nullable). The keystone of the hybrid — the weak planner cannot
# emit invalid JSON or a tool that does not exist.
_TOOL_CALL_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "action": {"type": "string", "enum": ["expand", "answer"]},
        "tool": {"type": ["string", "null"], "enum": ["search", "similar", "nearby", None]},
        "query": {"type": ["string", "null"]},
        "photo_id": {"type": ["integer", "null"]},
        "photo_ids": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["action", "tool", "query", "photo_id", "photo_ids"],
}


def retrieve(
    conn: sqlite3.Connection,
    embedder: Embedder,
    client: InferenceClient,
    *,
    owner_id: int,
    question: str,
    dimensions: list[str],
    caption_model: str,
    tag_score_min: float,
    planner_model: str,
    k: int = 8,
    floor: float = RERANK_FLOOR,
) -> list[int]:
    """Precise chat retrieval (§10): plan -> core candidates -> hard-filter ->
    rerank -> floor.

    The planner splits the question into HARD facet/date filters and SOFT tag
    hints, then candidate generation runs through the retriever core
    (`search/retriever.py`) so chat no longer rolls its own fusion. Final ranking
    stays `search/rerank.py::rerank` — caption-meaning cosine + floor, the exact
    tuned §10 mechanism — rather than the core's `refine()`. The core's text-query
    scoring folds fusion/KNN proximity in as an UNCONDITIONAL content contribution
    (§9.2, right for `/library` search: a semantic neighbour there IS a result),
    which would be fatal here: a KNN always returns k neighbours, so if fusion
    proximity alone could clear the content gate, chat would always answer
    something and confabulate again — the exact bug §10 killed. Reusing `rerank`'s
    caption-cosine floor keeps honest-empty structurally intact: only a genuine
    caption-meaning match clears it. `tag_score_min` is accepted for signature
    parity; planner tags are soft (never gate) and, as before this refactor, are
    not an alternate way to clear the floor.

    Returns the verified matches (<= k), or [] when nothing clears the floor. Any
    failure degrades to today's fusion retrieval so chat never breaks."""
    if not question.strip():
        return []
    try:
        spec = plan(client, planner_model, question, dimensions)
        hard_filters = spec_to_params(spec, query=question, dimensions=dimensions)
        query = Query(
            text=spec.semantic or question, hard_filters=hard_filters,
            soft_tags=spec.tags, k=200, floor=floor,
        )
        ids = candidates(conn, embedder, owner_id, query)
        ids = _hard_filter(conn, owner_id, query.hard_filters, ids)
        if not ids:
            return []
        query_vec = l2_normalize(client.embed(caption_model, [question])[0])
        return [pid for pid, _ in rerank(conn, query_vec, ids, floor=floor)][:k]
    except Exception:  # noqa: BLE001 — degrade to fusion, never crash the chat route
        return fusion_retrieve(conn, embedder, owner_id, question, k=k)


def agent_retrieve(
    conn: sqlite3.Connection,
    embedder: Embedder,
    client: InferenceClient,
    *,
    owner_id: int,
    question: str,
    dimensions: list[str],
    caption_model: str,
    tag_score_min: float,
    planner_model: str,
    k: int = 8,
    floor: float = RERANK_FLOOR,
    max_rounds: int = 3,
) -> list[int]:
    """Bounded verify/refine loop over the retriever's candidates for the SEMANTIC
    tail of chat (§10, §9.1 exception). Counts, memories, and periods are answered
    by `direct_answer` before the lease, so this loop only ever sees open-ended
    photo questions: it pulls candidates (`search`/`similar`/`nearby`), verifies
    them, and returns the ids the model stood behind (<= k), or [] when none match.
    Any loop failure degrades to the seed candidates."""
    seed = retrieve(
        conn, embedder, client, owner_id=owner_id, question=question,
        dimensions=dimensions, caption_model=caption_model, tag_score_min=tag_score_min,
        planner_model=planner_model, k=max(k, 30), floor=floor,
    )
    if not seed:
        return []
    known = set(seed)
    messages = [
        {"role": "system", "content": _AGENT_SYSTEM},
        {"role": "user", "content": f"Question: {question}\n{_summarise(conn, owner_id, seed)}"},
    ]
    try:
        for round_no in range(max_rounds + 1):
            force = round_no == max_rounds
            turn = _turn(client, planner_model, messages, force)
            if turn is None:
                return seed[:k]
            if turn.get("action") == "expand" and not force:
                extra = _tool(conn, embedder, owner_id, turn)
                known.update(extra)
                messages.append({"role": "assistant", "content": json.dumps(turn)})
                messages.append({"role": "user", "content": _summarise(conn, owner_id, extra)})
                continue
            verified = [pid for pid in (turn.get("photo_ids") or []) if pid in known]
            return verified[:k]
    except Exception:  # noqa: BLE001 — degrade to the seed candidates
        return seed[:k]
    return seed[:k]


def _summarise(conn: sqlite3.Connection, owner_id: int, photo_ids: list[int]) -> str:
    lines: list[str] = []
    for pid in photo_ids:
        row = conn.execute(
            "SELECT shot_at, caption FROM photos WHERE id = ? AND owner_id = ?",
            (pid, owner_id),
        ).fetchone()
        if row is None:
            continue
        tags = ", ".join(
            r["label"] for r in conn.execute(
                "SELECT t.label FROM photo_tags pt JOIN tags t ON t.id = pt.tag_id"
                " WHERE pt.photo_id = ? ORDER BY pt.score DESC LIMIT 4", (pid,))
        )
        lines.append(
            f"[{pid}] {row['shot_at'] or 'no date'} · {row['caption'] or ''}"
            + (f" · tags: {tags}" if tags else "")
        )
    return "Candidates:\n" + "\n".join(lines) if lines else "Candidates: (none)"


def _tool(
    conn: sqlite3.Connection, embedder: Embedder, owner_id: int, turn: dict
) -> list[int]:
    """Run one read-only candidate tool; returns more candidate photo ids — never
    raises. `search`/`similar`/`nearby` are the only tools: the agent's job is to
    widen the candidate pool, not to compute facts (those are `direct_answer`'s)."""
    tool, query, photo_id = turn.get("tool"), turn.get("query"), turn.get("photo_id")
    if tool == "search" and isinstance(query, str):
        return search_photos(conn, embedder, owner_id, query, k=10)
    if tool == "similar" and isinstance(photo_id, int):
        return [r["id"] for r in similar_photos(conn, owner_id, photo_id, k=5)]
    if tool == "nearby" and isinstance(photo_id, int):
        return [
            r["id"] for r in conn.execute(
                "SELECT p2.id FROM photos p1 JOIN photos p2 ON p2.owner_id = p1.owner_id"
                " WHERE p1.id = ? AND p2.id != p1.id AND p2.shot_at IS NOT NULL"
                " AND abs(julianday(p2.shot_at) - julianday(p1.shot_at)) < 0.25"
                " ORDER BY p2.shot_at LIMIT 5", (photo_id,))
        ]
    return []


# --- Aggregate tools: real numbers, never inferred from the shown candidates ---

_COUNT_INTENT = re.compile(r"\b(how many|how much|number of|count of|total number|total of)\b", re.IGNORECASE)
# The thing being counted, when the question narrows it: "...with dogs", "...of the dog".
_COUNT_SUBJECT = re.compile(
    r"\b(?:with|of|containing|showing|that (?:have|contain|show)|tagged)\s+(.+?)[?.!]*\s*$", re.IGNORECASE
)
# Words in a bare "how many photos …" total question that are NOT a narrowing subject.
_PHOTO_WORD = re.compile(r"\b(?:photos?|images?|pictures?|pics?|photographs?|shots?)\b", re.IGNORECASE)
_TOTAL_FILLER = re.compile(
    r"\b(?:do|does|did|i|we|you|my|our|us|the|a|an|there|are|is|be|been|have|has|had|"
    r"got|hold|holding|currently|now|right|stored|saved|in|of|total|altogether|all|together)\b",
    re.IGNORECASE,
)


def _plain_total(question: str) -> bool:
    """True when a "how many …" question is the plain whole-library total —
    answerable straight from the DB. False when it narrows to a referent the DB
    cannot count deterministically ("how many similar to this dog" needs embedding
    similarity, not a keyword count): those decline here and reach the model path
    (§10), so chat never confabulates the library total for a relational count.

    Removes the count phrase, photo words, and generic filler; if only library
    words (incl. the common "libray" typo class) or nothing remain, it is the whole
    library — any other leftover word means it narrows to something else."""
    rest = _TOTAL_FILLER.sub(" ", _PHOTO_WORD.sub(" ", _COUNT_INTENT.sub(" ", question)))
    leftovers = (token.strip("?.!,'\"“”") for token in rest.split())
    return all(not token or token.lower().startswith("libr") for token in leftovers)


def is_aggregate_question(question: str) -> bool:
    """True when the question is a count/aggregate ("how many …", "number of …").
    Chat answers these from the deterministic facts alone, never by counting the
    handful of photos it happened to retrieve (§10, the "8 photos" bug)."""
    return bool(_COUNT_INTENT.search(question))


# The counted noun right after "how many" / "number of" — the reliable signal for
# WHICH thing is being counted (photos vs memories vs months/years). Reading it
# fixes "how many photos are in my Borjomi memory" (counted noun = photos, not a
# memory count) and "how many photos this year" (photos, not a periods span).
_COUNTED = re.compile(r"\b(?:how many|number of|count of)\s+([a-z]+)", re.IGNORECASE)


def _count_subject(question: str) -> str:
    """The clean subject of a subject-count ("...with dogs" -> "dogs", "number of
    beach photos" -> "beach"), or "" when there is none. Photo words and leading
    determiners are stripped so the residue is a real FTS term, never the word
    "photos" itself."""
    match = _COUNT_SUBJECT.search(question)
    if not match:
        return ""
    subject = _PHOTO_WORD.sub(" ", match.group(1))
    subject = re.sub(r"^\s*(?:my|the|a|an|any|some)\s+", "", subject, flags=re.IGNORECASE)
    return " ".join(subject.split()).strip(" ?.!,'\"“”")


def _memory_show_answer(conn: sqlite3.Connection, owner_id: int, question: str) -> str:
    """The no-model text for a "show/find a memory" question — the memory itself is
    rendered as a card by the web layer (re-derived from the same question). A
    plural/all request names the count; a single hit names the memory; a miss is
    an honest "couldn't find it", never a trip to the planner."""
    memories = memories_for_show(conn, owner_id, question)
    if not memories:
        terms = _memory_terms(question)
        return (f'I could not find a memory matching "{terms}".' if terms
                else "You do not have any memories yet.")
    if len(memories) == 1:
        m = memories[0]
        return f'Here is your "{m["name"]}" memory — {m["size"]} photo(s).'
    return f"Here are your {len(memories)} memories."


def direct_answer(conn: sqlite3.Connection, owner_id: int, question: str) -> str | None:
    """Answer any DB-answerable question straight from SQLite — NO model, NO CHAT
    lease (§8.1, §10) — so the weak planner never sees it. Returns None to DECLINE
    (semantic / relational / ambiguous), so the question falls through to the agent
    loop. Every matcher is conservative: it fires only when confident and NEVER
    returns a confidently-wrong answer — a decline is always safe, a wrong answer is
    the bug that "all" was. See the routing matrix in tests/test_chat_routing.py."""
    # Memory show/list — deterministic from the question, not a count.
    if is_memory_show(question):
        return _memory_show_answer(conn, owner_id, question)
    q = question.lower()
    if not _COUNT_INTENT.search(q):
        return None
    counted = _COUNTED.search(q)
    noun = counted.group(1) if counted else ""
    if noun.startswith("memor"):
        n = len(list_memories(conn, owner_id))
        return f"You have {n} " + ("memory" if n == 1 else "memories") + " in your library."
    if noun in ("month", "months", "year", "years"):
        grain = "year" if noun.startswith("year") else "month"
        n, _ = count_periods(conn, owner_id, grain)
        return f"Your photos span {n} {grain if n == 1 else grain + 's'}."
    subject = _count_subject(question)
    if subject and not re.search(r"\b(photo|image|picture|librar|total|memor)\b", subject, re.IGNORECASE):
        n = count_photos(conn, owner_id, subject)
        return f"You have {n} photo(s) matching “{subject}”."
    if not _plain_total(question):
        return None  # a narrowing / relational count we cannot compute deterministically → agent
    n = count_photos(conn, owner_id, "")
    return f"You have {n} photo(s) in your library."


def count_photos(conn: sqlite3.Connection, owner_id: int, query: str) -> int:
    """How many photos match `query` — a real keyword (FTS) count over captions
    and tags, never a top-k KNN dump. An empty query or "all" answers the total
    photo count (§10) — this is what fixes chat confabulating a total from the
    handful of retrieved candidates."""
    q = query.strip().lower()
    if not q or q == "all":
        return conn.execute(
            "SELECT COUNT(*) AS n FROM photos WHERE owner_id = ?", (owner_id,)
        ).fetchone()["n"]
    return len(keyword_search(conn, owner_id, query, k=1_000_000))


def list_memories(conn: sqlite3.Connection, owner_id: int) -> list[dict]:
    """This owner's memories: name, date (its earliest photo), and size (§10, §11)."""
    rows = conn.execute(
        "SELECT g.name AS name,"
        " (SELECT MIN(p.shot_at) FROM group_photos gp JOIN photos p ON p.id = gp.photo_id"
        "  WHERE gp.group_id = g.id) AS date,"
        " (SELECT COUNT(*) FROM group_photos gp WHERE gp.group_id = g.id) AS size"
        " FROM groups g WHERE g.owner_id = ? AND g.kind = 'memory' ORDER BY g.id",
        (owner_id,),
    ).fetchall()
    return [{"name": r["name"], "date": r["date"], "size": r["size"]} for r in rows]


_MEMORY_INTENT = re.compile(r"\bmemor(?:y|ies)\b", re.IGNORECASE)
# Words to drop when turning "find memory in borjomi" into the FTS term "borjomi".
# `all/every/each/list` are quantifiers, never a memory's name — dropping them is
# what keeps "show me ALL my memories" from FTS-searching for a memory called
# "all" (which matches nothing) and confabulating "no memory matches 'all'".
_MEMORY_STOP = re.compile(
    r"\b(find|show|open|display|give|get|see|list|me|us|my|any|some|one|all|every|"
    r"each|of|the|a|an|please|in|on|at|from|about|with|memory|memories|which|what|"
    r"where|is|are|do|does|i|have|there)\b",
    re.IGNORECASE,
)
# A plural / "all" memory request — show EVERY memory, not one. Plural "memories"
# alone counts (bare "show me my memories"), as do the quantifiers "all/every/each"
# and "list". A specific narrowing term (non-empty `_memory_terms`) overrides this.
_ALL_MEMORIES = re.compile(r"\b(all|every|each|list)\b|\bmemories\b", re.IGNORECASE)


def is_memory_show(question: str) -> bool:
    """True for "find/show me a memory" questions — a memory to display, not a
    count. "How many memories" is a count and stays with the aggregate facts."""
    return bool(_MEMORY_INTENT.search(question)) and not is_aggregate_question(question)


def _memory_terms(question: str) -> str:
    """The searchable remainder of a memory question — "find memory in borjomi" ->
    "borjomi". Empty means a generic "show me a memory" (no place/name given)."""
    return " ".join(_MEMORY_STOP.sub(" ", question).split())


def find_memories(
    conn: sqlite3.Connection, owner_id: int, query: str, k: int = 3
) -> list[dict]:
    """This owner's memories best-matching `query` by FTS over name/description,
    best first, each with its cover photo ids to show (§10, §11).

    A non-empty `query` that matches nothing returns [] — an honest "no such
    memory", never an unrelated one. An empty `query` ("show me a memory") returns
    the largest memories so the request always yields something."""
    text = query.strip()
    if text:
        match = '"' + text.replace('"', '""') + '"'  # phrase-wrap: punctuation never breaks MATCH
        try:
            rows = conn.execute(
                _MEMORY_SELECT
                + " JOIN memory_fts f ON f.rowid = g.id"
                " WHERE memory_fts MATCH ? AND g.owner_id = ? AND g.kind = 'memory'"
                " ORDER BY bm25(memory_fts) LIMIT ?",
                (match, owner_id, k),
            ).fetchall()
        except sqlite3.OperationalError:
            rows = []
    else:
        rows = conn.execute(
            _MEMORY_SELECT
            + " WHERE g.owner_id = ? AND g.kind = 'memory' ORDER BY size DESC, g.id LIMIT ?",
            (owner_id, k),
        ).fetchall()
    memories: list[dict] = []
    for r in rows:
        photo_ids = [
            row["photo_id"] for row in conn.execute(
                "SELECT photo_id FROM group_photos WHERE group_id = ? ORDER BY rank LIMIT 12",
                (r["id"],),
            )
        ]
        memories.append({
            "id": r["id"], "name": r["name"], "description": r["description"] or "",
            "date": r["date"], "size": r["size"], "photo_ids": photo_ids,
        })
    return memories


_MEMORY_SELECT = (
    "SELECT g.id AS id, g.name AS name, g.description AS description,"
    " (SELECT MIN(p.shot_at) FROM group_photos gp JOIN photos p ON p.id = gp.photo_id"
    "  WHERE gp.group_id = g.id) AS date,"
    " (SELECT COUNT(*) FROM group_photos gp WHERE gp.group_id = g.id) AS size"
    " FROM groups g"
)


def memories_for_show(
    conn: sqlite3.Connection, owner_id: int, question: str
) -> list[dict]:
    """The memories a "find/show me … memory/memories" question refers to (§10).

    Returns the full memory dicts — id, name, description, photo_ids — so the chat
    UI can render each as its own Organize memory card and link into its grid:

    - a narrowing term ("… borjomi") → the one best-matching memory (or [] on a miss);
    - a plural / "all" request ("show me all my memories", "list my memories") → EVERY
      memory, largest first;
    - a bare singular "show me a memory" → the largest one.

    Empty list when the question is not about showing a memory, or matches none."""
    if not is_memory_show(question):
        return []
    terms = _memory_terms(question)
    if terms:
        return find_memories(conn, owner_id, terms, k=1)
    if _ALL_MEMORIES.search(question):
        return find_memories(conn, owner_id, "", k=1_000)
    return find_memories(conn, owner_id, "", k=1)


def count_periods(conn: sqlite3.Connection, owner_id: int, grain: str) -> tuple[int, list[str]]:
    """Distinct month/year buckets with photos — count and the list, oldest first
    (§10). `grain` is "month" or "year"."""
    length = 4 if grain == "year" else 7
    rows = conn.execute(
        f"SELECT DISTINCT substr(shot_at, 1, {length}) AS bucket FROM photos"
        " WHERE owner_id = ? AND shot_at IS NOT NULL ORDER BY bucket",
        (owner_id,),
    ).fetchall()
    buckets = [row["bucket"] for row in rows]
    return len(buckets), buckets


def _turn(
    client: InferenceClient, model: str, messages: list[dict], force: bool
) -> dict | None:
    """One model turn -> parsed JSON dict, or None if the output is unusable.

    The tool-call is schema-constrained (`_TOOL_CALL_SCHEMA`) so the model cannot
    emit malformed JSON or a tool that does not exist. A backend that rejects the
    schema falls back to a plain call — the `{...}` extraction below still guards
    output either way, so the loop degrades rather than breaks."""
    turn_messages = messages
    if force:
        turn_messages = [*messages, {"role": "user", "content": "Answer now (action=answer)."}]
    try:
        raw = client.complete(model, turn_messages, timeout=60.0, json_schema=_TOOL_CALL_SCHEMA)
    except Exception:  # noqa: BLE001 — backend without structured output → plain call
        raw = client.complete(model, turn_messages, timeout=60.0)
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        return None
    parsed = json.loads(raw[start : end + 1])
    return parsed if isinstance(parsed, dict) else None
