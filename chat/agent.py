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

_AGENT_SYSTEM = (
    "You find the photos in a personal library that answer the user's question. "
    "You are given candidate photos (id, date, caption, tags). Return ONLY the ids "
    "whose caption/tags actually match the question — verify each one; never invent "
    "a match. Reply with ONLY a JSON object. To pull more candidates first, use "
    '{"action":"expand","tool":"search|similar|nearby","query":"...","photo_id":<id>}. '
    "For a count or total question, use "
    '{"action":"expand","tool":"count","query":"..."} — the number of photos '
    'matching `query` (an empty query or "all" gives the total photo count). For '
    "the library's memories, use "
    '{"action":"expand","tool":"memories"} — each memory\'s name, date, and size. '
    "To find or show a SPECIFIC memory — by place, occasion, or name — use "
    '{"action":"expand","tool":"find_memory","query":"..."}; it returns that '
    "memory and its photos to show. "
    "For how many months or years have photos, use "
    '{"action":"expand","tool":"periods","grain":"month"} (or "year") — the '
    "distinct bucket count. These tools return a FACT, not more candidate photos — "
    "use it in your final answer; never guess a count from the few candidates "
    'shown. When ready, answer with {"action":"answer","photo_ids":[...]}. '
    "If none match, answer with an empty photo_ids list."
)


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
) -> tuple[list[int], list[str]]:
    """Bounded verify/refine loop over the retriever's candidates (§10, §9.1
    exception). Read-only tools, capped rounds, verify-before-answer. Returns
    (verified ids (<= k), gathered facts). `facts` collects every `count` /
    `memories` / `periods` tool result the agent pulled, in call order — the
    caller threads these into the final answer's grounding context so a count or
    total question is answered from a real number, never by counting the handful
    of candidate photos shown (the "8 photos" bug). Any loop failure degrades to
    the plain retriever, with no facts."""
    # Deterministic aggregate routing: a small planner model is unreliable at
    # CHOOSING the count/memories/periods tool, so when the question is clearly a
    # count/aggregate we compute the fact ourselves and seed it — the same tool
    # functions, fired by intent instead of the model's tool-call. This is what
    # makes "how many photos in total" answer the REAL total on a weak model.
    gathered: list[str] = _auto_facts(conn, owner_id, question)
    # A "find/show me a memory" question is answered by the memory index, not photo
    # retrieval: the seed photo search finds nothing for a memory's place/name (it
    # is not in any caption), so route it deterministically to `find_memory` and
    # return the memory's own photos to show (§10, §11).
    mem = _auto_memory(conn, owner_id, question)
    if mem is not None:
        fact, mem_ids = mem
        return mem_ids[:k], [fact]
    seed = retrieve(
        conn, embedder, client, owner_id=owner_id, question=question,
        dimensions=dimensions, caption_model=caption_model, tag_score_min=tag_score_min,
        planner_model=planner_model, k=max(k, 30), floor=floor,
    )
    if not seed:
        return [], gathered  # a count/aggregate answer needs no photos, only the fact
    known = set(seed)
    messages = [
        {"role": "system", "content": _AGENT_SYSTEM},
        {"role": "user", "content": f"Question: {question}\n{_summarise(conn, owner_id, seed)}"},
    ]
    if gathered:  # hand the deterministic facts to the agent too, so it can use them
        messages.append({"role": "user", "content": "Known facts:\n" + "\n".join(gathered)})
    try:
        for round_no in range(max_rounds + 1):
            force = round_no == max_rounds
            turn = _turn(client, planner_model, messages, force)
            if turn is None:
                return seed[:k], gathered
            if turn.get("action") == "expand" and not force:
                extra, fact = _tool(conn, embedder, owner_id, turn)
                known.update(extra)
                messages.append({"role": "assistant", "content": json.dumps(turn)})
                if fact is not None:
                    if fact not in gathered:
                        gathered.append(fact)
                    messages.append({"role": "user", "content": fact})
                else:
                    messages.append({"role": "user", "content": _summarise(conn, owner_id, extra)})
                continue
            verified = [pid for pid in (turn.get("photo_ids") or []) if pid in known]
            return verified[:k], gathered
    except Exception:  # noqa: BLE001 — degrade to the seed candidates, keep the facts
        return seed[:k], gathered
    return seed[:k], gathered


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
) -> tuple[list[int], str | None]:
    """Run one read-only tool; returns (candidate ids, fact text) — never raises.

    `search`/`similar`/`nearby` pull more CANDIDATE photos (a fact of None).
    `count`/`memories`/`periods` compute a FACT — a real number the model cannot
    get by counting the few candidates it has been shown (§10) — and return no
    ids.
    """
    tool, query, photo_id = turn.get("tool"), turn.get("query"), turn.get("photo_id")
    if tool == "search" and isinstance(query, str):
        return search_photos(conn, embedder, owner_id, query, k=10), None
    if tool == "similar" and isinstance(photo_id, int):
        return [r["id"] for r in similar_photos(conn, owner_id, photo_id, k=5)], None
    if tool == "nearby" and isinstance(photo_id, int):
        return [
            r["id"] for r in conn.execute(
                "SELECT p2.id FROM photos p1 JOIN photos p2 ON p2.owner_id = p1.owner_id"
                " WHERE p1.id = ? AND p2.id != p1.id AND p2.shot_at IS NOT NULL"
                " AND abs(julianday(p2.shot_at) - julianday(p1.shot_at)) < 0.25"
                " ORDER BY p2.shot_at LIMIT 5", (photo_id,))
        ], None
    if tool == "count":
        return [], _count_fact(conn, owner_id, query if isinstance(query, str) else "")
    if tool == "memories":
        return [], _memories_fact(conn, owner_id)
    if tool == "find_memory" and isinstance(query, str):
        mems = find_memories(conn, owner_id, query, k=1)
        if not mems:
            return [], f'no memory matches "{query}".'
        return mems[0]["photo_ids"], _memory_fact(mems[0])
    if tool == "periods":
        grain = turn.get("grain")
        return [], _periods_fact(conn, owner_id, grain if grain in ("month", "year") else "month")
    return [], None


# --- Aggregate tools: real numbers, never inferred from the shown candidates ---

_COUNT_INTENT = re.compile(r"\b(how many|how much|number of|count of|total number|total of)\b", re.IGNORECASE)
# The thing being counted, when the question narrows it: "...with dogs", "...of the dog".
_COUNT_SUBJECT = re.compile(
    r"\b(?:with|of|containing|showing|that (?:have|contain|show)|tagged)\s+(.+?)[?.!]*\s*$", re.IGNORECASE
)


def is_aggregate_question(question: str) -> bool:
    """True when the question is a count/aggregate ("how many …", "number of …").
    Chat answers these from the deterministic facts alone, never by counting the
    handful of photos it happened to retrieve (§10, the "8 photos" bug)."""
    return bool(_COUNT_INTENT.search(question))


def _auto_facts(conn: sqlite3.Connection, owner_id: int, question: str) -> list[str]:
    """Deterministically answer count/aggregate questions (§10), so a weak planner
    model that never emits a tool-call still gets the real number. Same tool
    functions as `_tool`, chosen here by intent: memories, months/years, a
    subject count, or the plain total."""
    q = question.lower()
    if not _COUNT_INTENT.search(q):
        return []
    if "memor" in q:
        # A "how many memories" answer is the COUNT alone — the per-memory sizes in
        # the full list confuse a weak model into echoing a member size (e.g. 24).
        return [f"count: {len(list_memories(conn, owner_id))} memory(ies) in total."]
    if "month" in q or "year" in q:
        return [_periods_fact(conn, owner_id, "year" if "year" in q else "month")]
    match = _COUNT_SUBJECT.search(question)
    subject = match.group(1).strip() if match else ""
    # A subject that is just "photos"/"the library"/"total" means the whole library.
    if subject and not re.search(r"\b(photo|image|picture|librar|total)\b", subject, re.IGNORECASE):
        return [_count_fact(conn, owner_id, subject)]
    return [_count_fact(conn, owner_id, "")]


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
_MEMORY_STOP = re.compile(
    r"\b(find|show|open|display|give|get|see|me|us|my|any|some|one|of|the|a|an|"
    r"please|in|on|at|from|about|with|memory|memories|which|what|where|is|are|"
    r"do|does|i|have|there)\b",
    re.IGNORECASE,
)


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


def _memory_fact(memory: dict) -> str:
    return (
        f'memory "{memory["name"]}" — {memory["size"]} photo(s), earliest '
        f'{memory["date"] or "no date"}. Open: /organize?by=memories'
    )


def _auto_memory(
    conn: sqlite3.Connection, owner_id: int, question: str
) -> tuple[str, list[int]] | None:
    """Deterministic memory routing (§10): for a "find/show a memory" question
    return (fact, photo_ids) — the matched memory's fact line and its cover photos
    to show. None when the question is not about showing a memory."""
    if not is_memory_show(question):
        return None
    terms = _memory_terms(question)
    memories = find_memories(conn, owner_id, terms, k=1)
    if not memories:
        return (f'no memory matches "{terms}".' if terms else "memories: 0 total.", [])
    return _memory_fact(memories[0]), memories[0]["photo_ids"]


def memory_for_show(
    conn: sqlite3.Connection, owner_id: int, question: str
) -> dict | None:
    """The memory a "find/show me a memory" question refers to, or None.

    Same deterministic route `_auto_memory` uses (`is_memory_show` + `find_memories`
    over the memory index), but returns the full memory dict — id, name,
    description, photo_ids — so the chat UI can render it as the Organize memory
    card and link into that memory's grid (§10). A miss ("antarctica") returns
    None; a bare "show me a memory" returns the largest."""
    if not is_memory_show(question):
        return None
    memories = find_memories(conn, owner_id, _memory_terms(question), k=1)
    return memories[0] if memories else None


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


def _count_fact(conn: sqlite3.Connection, owner_id: int, query: str) -> str:
    n = count_photos(conn, owner_id, query)
    label = "in total" if not query.strip() or query.strip().lower() == "all" else f'matching "{query}"'
    return f"count: {n} photo(s) {label}."


def _memories_fact(conn: sqlite3.Connection, owner_id: int) -> str:
    memories = list_memories(conn, owner_id)
    if not memories:
        return "memories: 0 total."
    lines = [
        f"- {m['name']}: {m['size']} photo(s), earliest {m['date'] or 'no date'}"
        for m in memories
    ]
    return f"memories: {len(memories)} total:\n" + "\n".join(lines)


def _periods_fact(conn: sqlite3.Connection, owner_id: int, grain: str) -> str:
    n, buckets = count_periods(conn, owner_id, grain)
    if not buckets:
        return f"{grain}s with photos: 0."
    return f"{grain}s with photos: {n} ({', '.join(buckets)})."


def _turn(
    client: InferenceClient, model: str, messages: list[dict], force: bool
) -> dict | None:
    """One model turn -> parsed JSON dict, or None if the output is unusable."""
    turn_messages = messages
    if force:
        turn_messages = [*messages, {"role": "user", "content": "Answer now (action=answer)."}]
    raw = client.complete(model, turn_messages, timeout=60.0)
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        return None
    parsed = json.loads(raw[start : end + 1])
    return parsed if isinstance(parsed, dict) else None
