# Plan 10: Agentic RAG + reranking for chat retrieval

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `/chat` answers precise. "find me a photo with a dog" returns the
photos that actually contain a dog (or says there are none), not 30
nearest-neighbour thumbnails narrated into a hallucination.

**Architecture:** Three layers, cheapest first, one entry point.
(1) Route the chat question through the existing query planner (`search/planner.py`)
so tags + caption-FTS + facets do the precise work — the same path `/library`
search uses. (2) Rerank the fused candidates by caption-meaning cosine and drop
everything below a relevance floor, so "nothing matches" is a real outcome.
(3) Seed a bounded, read-only agent loop (modelled on `albums/compose.py`) with
those candidates; it verifies each match and may pull more context via tools
before returning the verified id set. The natural-language answer still streams
over SSE, grounded only on verified matches.

**Tech Stack:** Python, `uv`, SQLite (+ `sqlite-vec`, FTS5), FastAPI/SSE,
pytest with `FakeInferenceClient`/`FakeEmbedder`.

**Spec:** `docs/design.md` §10 (Ask-your-library chat) and §9.1 (query planner /
interactive-exception note). Task 1 rewrites §10 before any code lands.

## Global Constraints

- **Doc is source of truth.** `docs/design.md` describes current behaviour. Task 1
  updates §10 + §9.1 **before** implementation, per project `CLAUDE.md`.
- **Streaming stays.** §10 requires the answer to stream token-by-token over SSE.
  The agent loop drives *retrieval*, not the prose answer — the answer is still
  produced by `chat_messages(...)` + `client.stream(...)` over the verified ids.
- **Grounded only.** The model may cite only photos it was given; sources shown /
  stored are the cited ids (`chat/history.py::cited_ids`), a subset of verified
  matches. Never dump the candidate set.
- **Degrade, never crash.** Any planner / rerank / agent / embed failure falls
  back to today's fusion retrieval (`chat/retrieve.py::retrieve`). The gate
  (`is_photo_question`) is unchanged and runs first.
- **Two embedding spaces — do not mix.** SigLIP image/text vectors (1152-d,
  `photo_vec`, via `Embedder.embed_texts`/`read_vector`) are a different space
  from caption vectors (`caption_vec`, via `InferenceClient.embed(caption_embed_model)`,
  `read_caption_vector`). Rerank compares the query to *caption* vectors, so the
  query must be embedded with `settings.caption_embed_model`.
- **Deterministic tests.** Under `FakeInferenceClient`, `embed`/`complete`/`stream`
  are queued and text-hashed: an identical query and caption embed to the same
  vector (cosine ≈ 1.0). Tests pass explicit floors and queued turns — never the
  tuned default constant.

---

## File Structure

- `docs/design.md` — §10 rewrite, §9.1 exception note, §17/§18 status (Task 1, Task 6).
- `search/rerank.py` *(new)* — pure caption-cosine rerank + floor; optional LLM rerank. (Task 2)
- `chat/agent.py` *(new)* — planner→fusion→filter→rerank→floor retriever (Task 3) +
  bounded verify/refine agent loop (Task 4).
- `web/app.py` — `chat_stream` wired to the agent retriever (Task 5).
- `scripts/rerank_bakeoff.py` *(new)* — floor calibration over a hand-labelled dev set (Task 6).
- Tests: `tests/test_rerank.py`, `tests/test_chat_agent.py`, and edits to
  `tests/test_web_chat.py`.

Reused as-is (no changes): `search/planner.py` (`plan`, `spec_to_params`),
`search/semantic.py` (`search_photos`), `search/keyword.py` (`keyword_search`),
`search/fusion.py` (`reciprocal_rank_fusion`), `search/facets.py`
(`parse_filters`, `build_where`), `search/tags.py` (`parse_tag_filters`,
`tag_where`), `search/dates.py` (`date_where`), `embedding/store.py`
(`read_caption_vector`, `read_vector`), `embedding/vectors.py` (`l2_normalize`).

---

### Task 1: Rewrite the design (doc-first)

Project `CLAUDE.md` requires `docs/design.md` to match new behaviour **before**
code. This task changes the doc only; no code, no tests.

**Files:**
- Modify: `docs/design.md` — §10 (Ask-your-library chat), §9.1 (planner note),
  §17/§18 (status).

- [ ] **Step 1: Rewrite §10 retrieval.** Replace the "retrieval returns the top 30
  photos … force-feed the model" description with the agentic path:

  1. Off-topic gate first (unchanged, `is_photo_question`).
  2. **Plan** the question into a `QuerySpec` (§9.1) → tags + caption-FTS + facet
     predicates, the same precise path `/library` search uses.
  3. **Fuse** semantic + keyword rankings and **narrow** them by the planner's
     tag/facet/date predicates.
  4. **Rerank** survivors by caption-meaning cosine (query embedded with the
     caption model vs each photo's `caption_vec`) and **drop below a relevance
     floor** — when nothing clears the floor the answer is an honest "I couldn't
     find photos of X", with **no sources**.
  5. **Bounded verify/refine agent loop** (the documented interactive exception to
     §9.1) seeded with those candidates: read-only tools (`search`,
     `filter_tag`, `filter_facet`, `nearby`), a few rounds, verify-before-answer;
     returns the **verified** id set.
  6. The answer still **streams** over SSE, grounded only on verified matches;
     shown/stored sources are the cited ids (a subset). State that captions are
     imperfect so thumbnails remain the evidence.

- [ ] **Step 2: Add the §9.1 exception note.** After "This applies to *interactive*
  retrieval only", record that **chat is the one interactive exception**: the user
  accepted the latency of an agent loop for chat specifically, so chat may run a
  bounded multi-round loop where `/library` search stays a single call.

- [ ] **Step 3: Update status.** §17 risks: note the rerank floor is tuned on a
  ~100-photo hand-labelled dev set (like the SigLIP calibration). §18 future work:
  change the "Agentic RAG + reranking for chat retrieval (plan 10)" bullet from
  future to **implemented (plan 10)**; leave multi-turn memory and a learned
  reranker as future.

- [ ] **Step 4: Commit.**

```bash
git add docs/design.md
git commit -m "docs: rewrite §10 for agentic RAG chat retrieval (plan 10)"
```

---

### Task 2: Rerank module (`search/rerank.py`)

Pure caption-cosine rerank + relevance floor. The query arrives **already
embedded** in caption space, so the function needs no client and is trivially
deterministic. Optional LLM rerank lives here behind its own function.

**Files:**
- Create: `search/rerank.py`
- Test: `tests/test_rerank.py`

**Interfaces:**
- Consumes: `embedding.store.read_caption_vector`, `embedding.vectors.l2_normalize`.
- Produces:
  - `RERANK_FLOOR: float` — conservative default (tuned in Task 6).
  - `rerank(conn: sqlite3.Connection, query_vec: list[float], candidate_ids: list[int], *, floor: float) -> list[tuple[int, float]]`
    — `(photo_id, cosine)` for candidates whose caption-cosine ≥ `floor`, best
    first. A candidate with no `caption_vec` scores 0.0 and is dropped by any
    positive floor.
  - `rerank_llm(client, model: str, conn, query: str, candidate_ids: list[int], *, keep_min: int = 2) -> list[int]`
    — one batched 0–3 relevance call; keep ids scoring ≥ `keep_min`, input order.

- [ ] **Step 1: Write the failing tests.**

```python
# tests/test_rerank.py
import sqlite3

from embedding.store import write_caption_vector
from embedding.vectors import l2_normalize
from inference.fakes import FakeInferenceClient
from search.rerank import rerank, rerank_llm
from tests.factories import add_photo


def _cap(conn, pid, text):
    add_photo(conn, photo_id=pid, content_hash=f"h{pid}", thumb_key=f"{pid}.jpg", caption=text)
    # caption vectors live in the inference-client embed space (settings.caption_embed_model)
    write_caption_vector(conn, pid, l2_normalize(FakeInferenceClient().embed("fake", [text])[0]))


def _qvec(text):
    return l2_normalize(FakeInferenceClient().embed("fake", [text])[0])


def test_matches_rank_above_non_matches(conn):
    _cap(conn, 1, "a dog on a beach")
    _cap(conn, 2, "a plate of pasta")
    ranked = rerank(conn, _qvec("a dog on a beach"), [1, 2], floor=0.0)
    assert [pid for pid, _ in ranked][0] == 1


def test_floor_drops_everything_when_nothing_matches(conn):
    _cap(conn, 1, "a plate of pasta")
    assert rerank(conn, _qvec("a dog on a beach"), [1], floor=0.9) == []


def test_photo_without_caption_vector_is_dropped(conn):
    add_photo(conn, photo_id=3, content_hash="h3", thumb_key="3.jpg", caption=None)
    assert rerank(conn, _qvec("anything"), [3], floor=0.01) == []


def test_llm_rerank_keeps_only_high_scores(conn):
    for pid in (1, 2):
        add_photo(conn, photo_id=pid, content_hash=f"h{pid}", thumb_key=f"{pid}.jpg", caption="x")
    client = FakeInferenceClient(['{"1": 3, "2": 0}'])
    assert rerank_llm(client, "m", conn, "a dog", [1, 2], keep_min=2) == [1]
```

- [ ] **Step 2: Run to verify they fail.**

Run: `uv run pytest tests/test_rerank.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'search.rerank'`.

- [ ] **Step 3: Implement `search/rerank.py`.**

```python
import json
import sqlite3

from embedding.store import read_caption_vector
from embedding.vectors import l2_normalize
from inference.client import InferenceClient

# Conservative default; tuned against a hand-labelled dev set in Task 6 (§17).
RERANK_FLOOR = 0.4


def rerank(
    conn: sqlite3.Connection,
    query_vec: list[float],
    candidate_ids: list[int],
    *,
    floor: float,
) -> list[tuple[int, float]]:
    """Rerank candidates by caption-meaning cosine, dropping everything below
    `floor` (§10). `query_vec` must already be L2-normalized in the caption embed
    space. A candidate without a caption vector scores 0.0."""
    q = l2_normalize(query_vec)
    scored: list[tuple[int, float]] = []
    for pid in candidate_ids:
        vec = read_caption_vector(conn, pid)
        cosine = sum(a * b for a, b in zip(q, l2_normalize(vec))) if vec is not None else 0.0
        if cosine >= floor:
            scored.append((pid, cosine))
    scored.sort(key=lambda pair: -pair[1])
    return scored


def rerank_llm(
    client: InferenceClient,
    model: str,
    conn: sqlite3.Connection,
    query: str,
    candidate_ids: list[int],
    *,
    keep_min: int = 2,
) -> list[int]:
    """One batched relevance call: score each candidate's caption 0–3 against the
    query, keep those ≥ `keep_min` (§10, layer-1 LLM variant behind a flag). Any
    failure keeps the input order unchanged."""
    if not candidate_ids:
        return []
    lines = []
    for pid in candidate_ids:
        row = conn.execute("SELECT caption FROM photos WHERE id = ?", (pid,)).fetchone()
        lines.append(f"{pid}: {(row['caption'] if row else None) or '(no caption)'}")
    system = (
        "Score how well each photo caption matches the query, 0 (no) to 3 (exact). "
        'Reply with ONLY a JSON object mapping the id (as a string) to its score.'
    )
    user = f"Query: {query}\nPhotos:\n" + "\n".join(lines)
    try:
        raw = client.complete(model, [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ], timeout=30.0)
        start, end = raw.find("{"), raw.rfind("}")
        scores = json.loads(raw[start : end + 1]) if start != -1 and end > start else {}
        return [pid for pid in candidate_ids if int(scores.get(str(pid), 0)) >= keep_min]
    except (json.JSONDecodeError, ValueError, KeyError, TypeError):
        return list(candidate_ids)
```

- [ ] **Step 4: Run to verify they pass.**

Run: `uv run pytest tests/test_rerank.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit.**

```bash
git add search/rerank.py tests/test_rerank.py
git commit -m "feat: caption-cosine rerank + relevance floor for chat (plan 10)"
```

---

### Task 3: Chat retriever v2 (`chat/agent.py::retrieve`)

Planner → semantic+keyword fusion → narrow by the planner's tag/facet/date
predicates → rerank + floor → top-k verified ids. Falls back to today's fusion
retrieval on any error.

**Files:**
- Create: `chat/agent.py`
- Test: `tests/test_chat_agent.py`

**Interfaces:**
- Consumes: `search.planner.plan`/`spec_to_params`, `search.semantic.search_photos`,
  `search.keyword.keyword_search`, `search.fusion.reciprocal_rank_fusion`,
  `search.facets.parse_filters`/`build_where`, `search.tags.parse_tag_filters`/`tag_where`,
  `search.dates.date_where`, `search.rerank.rerank`/`RERANK_FLOOR`,
  `embedding.vectors.l2_normalize`, `chat.retrieve.retrieve` (fusion fallback).
- Produces:
  `retrieve(conn, embedder, client, *, owner_id: int, question: str, dimensions: list[str], caption_model: str, tag_score_min: float, planner_model: str, k: int = 8, floor: float = RERANK_FLOOR) -> list[int]`
  — verified photo ids, best first, ≤ `k`.

- [ ] **Step 1: Write the failing tests.**

```python
# tests/test_chat_agent.py
from embedding.fakes import FakeEmbedder
from embedding.store import write_caption_vector
from embedding.vectors import l2_normalize
from inference.fakes import FakeInferenceClient
from chat.agent import retrieve
from tests.factories import add_photo

DIMS = ["subject", "setting", "vibe"]


def _photo(conn, pid, caption):
    add_photo(conn, photo_id=pid, content_hash=f"h{pid}", thumb_key=f"{pid}.jpg", caption=caption)
    write_caption_vector(conn, pid, l2_normalize(FakeInferenceClient().embed("fake", [caption])[0]))


def _client(spec_json):
    # First embed() = the query in caption space; complete() = the planner's QuerySpec.
    return FakeInferenceClient(responses=[spec_json])


def test_returns_only_photos_above_the_floor(conn):
    _photo(conn, 1, "a dog on a beach")
    _photo(conn, 2, "a plate of pasta")
    client = _client('{"semantic": "a dog on a beach"}')
    ids = retrieve(conn, FakeEmbedder(), client, owner_id=1, question="a dog on a beach",
                   dimensions=DIMS, caption_model="fake", tag_score_min=0.2,
                   planner_model="fake", k=8, floor=0.5)
    assert ids == [1]


def test_empty_when_nothing_clears_the_floor(conn):
    _photo(conn, 1, "a plate of pasta")
    client = _client('{"semantic": "a dog on a beach"}')
    ids = retrieve(conn, FakeEmbedder(), client, owner_id=1, question="a dog on a beach",
                   dimensions=DIMS, caption_model="fake", tag_score_min=0.2,
                   planner_model="fake", k=8, floor=0.9)
    assert ids == []


def test_planner_failure_falls_back_to_fusion(conn, monkeypatch):
    _photo(conn, 1, "a dog on a beach")
    client = _client('{"semantic": "a dog on a beach"}')
    monkeypatch.setattr("chat.agent.rerank", lambda *a, **k: (_ for _ in ()).throw(RuntimeError()))
    ids = retrieve(conn, FakeEmbedder(), client, owner_id=1, question="a dog on a beach",
                   dimensions=DIMS, caption_model="fake", tag_score_min=0.2,
                   planner_model="fake", k=8, floor=0.5)
    assert 1 in ids  # degraded to chat.retrieve.retrieve, never crashes
```

- [ ] **Step 2: Run to verify they fail.**

Run: `uv run pytest tests/test_chat_agent.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'chat.agent'`.

- [ ] **Step 3: Implement `chat/agent.py` (retriever only for this task).**

```python
import sqlite3

from embedding.base import Embedder
from embedding.vectors import l2_normalize
from inference.client import InferenceClient
from chat.retrieve import retrieve as fusion_retrieve
from search.dates import date_where
from search.facets import build_where, parse_filters
from search.fusion import reciprocal_rank_fusion
from search.keyword import keyword_search
from search.planner import plan, spec_to_params
from search.rerank import RERANK_FLOOR, rerank
from search.semantic import search_photos
from search.tags import parse_tag_filters, tag_where


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
    """Precise chat retrieval (§10): plan → fuse → narrow → rerank → floor.

    Returns the verified matches (≤ k), or [] when nothing clears the floor. Any
    failure degrades to today's fusion retrieval so chat never breaks."""
    if not question.strip():
        return []
    try:
        spec = plan(client, planner_model, question, dimensions)
        params = spec_to_params(spec, query=question, dimensions=dimensions)
        semantic = search_photos(conn, embedder, owner_id, spec.semantic or question, k=200)
        keyword = keyword_search(conn, owner_id, question, k=200)
        fused = reciprocal_rank_fusion([semantic, keyword])
        fused = _narrow(conn, owner_id, params, tag_score_min, fused)
        if not fused:
            return []
        query_vec = l2_normalize(client.embed(caption_model, [question])[0])
        return [pid for pid, _ in rerank(conn, query_vec, fused, floor=floor)][:k]
    except Exception:  # noqa: BLE001 — degrade to fusion, never crash the chat route
        return fusion_retrieve(conn, embedder, owner_id, question, k=k)


def _narrow(
    conn: sqlite3.Connection,
    owner_id: int,
    params: dict[str, str],
    tag_score_min: float,
    ids: list[int],
) -> list[int]:
    """Keep only ids satisfying the planner's facet/tag/date predicates — the same
    filters `/library` search applies (web/app.py::_search_page)."""
    if not ids:
        return []
    facet_where, facet_params = build_where(parse_filters(params))
    tags_where, tags_params = tag_where(parse_tag_filters(params), tag_score_min)
    date_frag, date_params = date_where(params)
    where = facet_where + tags_where + date_frag
    if not where:
        return ids
    placeholders = ", ".join("?" for _ in ids)
    allowed = {
        row["id"] for row in conn.execute(
            "SELECT p.id FROM photos p WHERE p.owner_id = ?" + where
            + f" AND p.id IN ({placeholders})",
            (owner_id, *facet_params, *tags_params, *date_params, *ids),
        )
    }
    return [pid for pid in ids if pid in allowed]
```

- [ ] **Step 4: Run to verify they pass.**

Run: `uv run pytest tests/test_chat_agent.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit.**

```bash
git add chat/agent.py tests/test_chat_agent.py
git commit -m "feat: planner-backed reranked chat retriever v2 (plan 10)"
```

---

### Task 4: Bounded verify/refine agent loop (`chat/agent.py::agent_retrieve`)

Seed the loop with Task 3's candidates; the model verifies each match and may
pull more context via read-only tools before returning the verified id set.
Mirrors `albums/compose.py` (expand/answer turns, round cap, force-answer,
lenient JSON). The prose answer is NOT produced here — the route streams it over
the returned ids.

**Files:**
- Modify: `chat/agent.py`
- Test: `tests/test_chat_agent.py` (add cases)

**Interfaces:**
- Consumes: Task 3 `retrieve`; `search.semantic.search_photos`/`similar_photos`;
  `search.keyword.keyword_search`; `inference.client.ChatMessage`.
- Produces:
  `agent_retrieve(conn, embedder, client, *, owner_id, question, dimensions, caption_model, tag_score_min, planner_model, k=8, floor=RERANK_FLOOR, max_rounds=3) -> list[int]`
  — verified photo ids, best first, ≤ `k`. Falls back to Task 3 `retrieve` on any
  loop failure.

- [ ] **Step 1: Write the failing tests.**

```python
# add to tests/test_chat_agent.py
import json
from chat.agent import agent_retrieve


def test_agent_verifies_a_subset_of_the_seed(conn):
    _photo(conn, 1, "a dog on a beach")
    _photo(conn, 2, "a dog in a park")
    # planner spec, then the query embed happens inside retrieve(); then one agent turn.
    client = FakeInferenceClient(responses=[
        '{"semantic": "a dog"}',
        json.dumps({"action": "answer", "photo_ids": [1]}),
    ])
    ids = agent_retrieve(conn, FakeEmbedder(), client, owner_id=1, question="a dog",
                         dimensions=DIMS, caption_model="fake", tag_score_min=0.2,
                         planner_model="fake", k=8, floor=0.0, max_rounds=3)
    assert ids == [1]


def test_agent_can_expand_then_answer(conn):
    _photo(conn, 1, "a dog on a beach")
    client = FakeInferenceClient(responses=[
        '{"semantic": "a dog"}',
        json.dumps({"action": "expand", "tool": "search", "query": "dog"}),
        json.dumps({"action": "answer", "photo_ids": [1]}),
    ])
    ids = agent_retrieve(conn, FakeEmbedder(), client, owner_id=1, question="a dog",
                         dimensions=DIMS, caption_model="fake", tag_score_min=0.2,
                         planner_model="fake", k=8, floor=0.0, max_rounds=3)
    assert ids == [1]


def test_agent_answering_none_returns_empty(conn):
    _photo(conn, 1, "a plate of pasta")
    client = FakeInferenceClient(responses=[
        '{"semantic": "a dog"}',
        json.dumps({"action": "answer", "photo_ids": []}),
    ])
    ids = agent_retrieve(conn, FakeEmbedder(), client, owner_id=1, question="a dog",
                         dimensions=DIMS, caption_model="fake", tag_score_min=0.2,
                         planner_model="fake", k=8, floor=0.0, max_rounds=3)
    assert ids == []
```

- [ ] **Step 2: Run to verify they fail.**

Run: `uv run pytest tests/test_chat_agent.py -k agent -v`
Expected: FAIL — `ImportError: cannot import name 'agent_retrieve'`.

- [ ] **Step 3: Implement `agent_retrieve` + helpers in `chat/agent.py`.**

```python
import json  # add to existing imports
from search.semantic import similar_photos  # add

_AGENT_SYSTEM = (
    "You find the photos in a personal library that answer the user's question. "
    "You are given candidate photos (id, date, caption, tags). Return ONLY the ids "
    "whose caption/tags actually match the question — verify each one; never invent "
    "a match. Reply with ONLY a JSON object. To pull more candidates first, use "
    '{\"action\":\"expand\",\"tool\":\"search|similar|nearby\",\"query\":\"...\",\"photo_id\":<id>}. '
    'When ready, answer with {\"action\":\"answer\",\"photo_ids\":[...]}. '
    "If none match, answer with an empty photo_ids list."
)


def agent_retrieve(
    conn, embedder, client, *, owner_id, question, dimensions, caption_model,
    tag_score_min, planner_model, k=8, floor=RERANK_FLOOR, max_rounds=3,
):
    """Bounded verify/refine loop over Task-3 candidates (§10, §9.1 exception).

    Read-only tools, capped rounds, verify-before-answer. Returns the verified ids
    (≤ k). Any failure degrades to the plain retriever."""
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
    except Exception:  # noqa: BLE001
        return seed[:k]
    return seed[:k]


def _summarise(conn, owner_id, photo_ids):
    lines = []
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
        lines.append(f"[{pid}] {row['shot_at'] or 'no date'} · {row['caption'] or ''}"
                     + (f" · tags: {tags}" if tags else ""))
    return "Candidates:\n" + "\n".join(lines) if lines else "Candidates: (none)"


def _tool(conn, embedder, owner_id, turn):
    tool, query, photo_id = turn.get("tool"), turn.get("query"), turn.get("photo_id")
    if tool == "search" and isinstance(query, str):
        return search_photos(conn, embedder, owner_id, query, k=10)
    if tool == "similar" and isinstance(photo_id, int):
        return [r["id"] for r in similar_photos(conn, owner_id, photo_id, k=5)]
    if tool == "nearby" and isinstance(photo_id, int):
        return [r["id"] for r in conn.execute(
            "SELECT p2.id FROM photos p1 JOIN photos p2 ON p2.owner_id = p1.owner_id"
            " WHERE p1.id = ? AND p2.id != p1.id AND p2.shot_at IS NOT NULL"
            " AND abs(julianday(p2.shot_at) - julianday(p1.shot_at)) < 0.25"
            " ORDER BY p2.shot_at LIMIT 5", (photo_id,))]
    return []


def _turn(client, model, messages, force):
    turn_messages = messages
    if force:
        turn_messages = [*messages,
                         {"role": "user", "content": "Answer now (action=answer)."}]
    raw = client.complete(model, turn_messages, timeout=60.0)
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        return None
    parsed = json.loads(raw[start : end + 1])
    return parsed if isinstance(parsed, dict) else None
```

- [ ] **Step 4: Run to verify all pass.**

Run: `uv run pytest tests/test_chat_agent.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit.**

```bash
git add chat/agent.py tests/test_chat_agent.py
git commit -m "feat: bounded verify/refine agent loop for chat retrieval (plan 10)"
```

---

### Task 5: Wire the agent retriever into `/chat/stream`

Replace the fusion `retrieve(...)` call with `agent_retrieve(...)`; ground the
streamed answer only on verified matches; honest empty state when none.

**Files:**
- Modify: `web/app.py` (`chat_stream`, ~lines 693–723; import line ~29)
- Test: `tests/test_web_chat.py`

**Interfaces:**
- Consumes: `chat.agent.agent_retrieve`; existing `build_chat_context`,
  `chat_messages`, `cited_ids`, `add_message`, `current_session`,
  `is_photo_question`.

- [ ] **Step 1: Write the failing test.** Assert a streamed answer is grounded on
  the verified match and that an unmatched question yields no sources.

```python
# add to tests/test_web_chat.py — follow the file's existing client/fixture setup.
# Queue order per question: is_photo_question (complete) → planner spec (complete)
#   → query embed (embed, automatic) → agent turn (complete) → answer (stream).
def test_chat_streams_answer_grounded_on_verified_match(chat_client, conn_with_dog_photo):
    # photo 1 caption "a dog on a beach"; fake inference queued accordingly.
    body = chat_client.get("/chat/stream", params={"q": "a dog on a beach"}).text
    assert "[photo:1]" in body            # cited the verified match
    assert "photo:2" not in body          # the pasta photo was never grounded


def test_chat_reports_nothing_found_when_floor_rejects_all(chat_client, conn_only_pasta):
    body = chat_client.get("/chat/stream", params={"q": "a dog on a beach"}).text
    # no source ids streamed; the grounded prompt said "no photos matched"
    assert "photo:" not in body
```

- [ ] **Step 2: Run to verify they fail.**

Run: `uv run pytest tests/test_web_chat.py -k "verified or nothing_found" -v`
Expected: FAIL (the route still calls the fusion `retrieve`, dumps 30 candidates).

- [ ] **Step 3: Edit `web/app.py`.** Change the import and the retrieval call:

```python
# near line 29 — replace `from chat.retrieve import is_photo_question, retrieve`
from chat.retrieve import is_photo_question
from chat.agent import agent_retrieve
```

```python
# inside chat_stream.events(), replace `ids = retrieve(ctx.conn, embedder, owner_id, q, k=30)`
ids = agent_retrieve(
    ctx.conn, embedder, client,
    owner_id=owner_id, question=q, dimensions=list(vocab.dimensions),
    caption_model=ctx.settings.caption_embed_model,
    tag_score_min=ctx.settings.tag_score_min,
    planner_model=model,
)
messages = chat_messages(q, build_chat_context(ctx.conn, ids))
```

`build_chat_context([])` already returns "No photos matched.", so the existing
grounded prompt handles the empty case — no extra branch. `vocab` is already in
scope in `create_app` (used by `_tag_sidebar`).

- [ ] **Step 4: Run the web chat tests.**

Run: `uv run pytest tests/test_web_chat.py -v`
Expected: PASS. Update any existing test that queued a single stream for the old
one-shot path to also queue the planner-spec + agent-answer `complete` responses.

- [ ] **Step 5: Full suite + commit.**

```bash
uv run pytest -q
git add web/app.py tests/test_web_chat.py
git commit -m "feat: chat streams grounded on agent-verified matches (plan 10)"
```

---

### Task 6: Floor bake-off + record the tuned value

Calibrate `RERANK_FLOOR` on a small hand-labelled dev set and record the result,
mirroring the SigLIP calibration (§17).

**Files:**
- Create: `scripts/rerank_bakeoff.py`
- Modify: `search/rerank.py` (`RERANK_FLOOR` value), `docs/design.md` (§17 note)

- [ ] **Step 1: Write the bake-off script.** Load a hand-labelled dev set
  (`{query: [relevant_photo_id, ...]}` JSON), embed each query in caption space,
  run `rerank(...)` at floors `0.2..0.6` step `0.05`, print precision/recall/F1
  per floor, and print the F1-maximising floor. Real inference client + DB; no
  test framework.

```python
# scripts/rerank_bakeoff.py — sketch; fill query set at run time.
import json
import sys

from config import get_settings
from db.connection import connect  # match the project's existing DB entrypoint
from embedding.vectors import l2_normalize
from search.rerank import rerank


def main(dev_path: str) -> None:
    settings = get_settings()
    conn = connect(settings)
    client, _ = settings.build_inference_client()
    dev = json.loads(open(dev_path).read())  # {query: [relevant ids]}
    for floor in [round(0.2 + 0.05 * i, 2) for i in range(9)]:
        tp = fp = fn = 0
        for query, gold in dev.items():
            qv = l2_normalize(client.embed(settings.caption_embed_model, [query])[0])
            got = {pid for pid, _ in rerank(conn, qv, _candidates(conn, query), floor=floor)}
            gold = set(gold)
            tp += len(got & gold); fp += len(got - gold); fn += len(gold - got)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        print(f"floor={floor}  P={p:.2f} R={r:.2f} F1={f1:.2f}")


if __name__ == "__main__":
    main(sys.argv[1])
```

  (`_candidates` = the same fuse+narrow the retriever uses; import from `chat.agent`
  if factored out, else inline the fusion call. Confirm the project's DB
  connection entrypoint before running.)

- [ ] **Step 2: Run it, pick the F1-max floor, set `RERANK_FLOOR`.** Update the
  constant in `search/rerank.py` to the chosen value.

- [ ] **Step 3: Record in `docs/design.md` §17.** State the tuned floor and that it
  was chosen by F1 on the dev set. Re-run `uv run pytest -q` to confirm green
  (tests pass explicit floors, so the constant change cannot break them).

- [ ] **Step 4: Commit.**

```bash
git add scripts/rerank_bakeoff.py search/rerank.py docs/design.md
git commit -m "feat: calibrate chat rerank floor on dev set (plan 10)"
```

---

## Non-goals

- Changing `/library` search (already planner-backed and precise).
- Multi-turn conversational memory (chat stays per-question; §10).
- A learned reranker model — caption-cosine + optional LLM rerank is enough here.
- Streaming the answer from inside the agent loop — the loop drives retrieval; the
  answer still streams via `chat_messages` + `client.stream`.

## Self-Review

- **Spec coverage:** §10 rewrite (T1) ✓; relevance floor + rerank (T2) ✓; reuse
  query planner (T3) ✓; bounded agent loop w/ verify (T4) ✓; wire-in, sources =
  verified only, honest empty (T5) ✓; §9.1 exception note (T1) ✓; floor bake-off
  (T6) ✓.
- **Types:** `retrieve` / `agent_retrieve` share the same keyword signature; both
  return `list[int]`. `rerank` returns `list[tuple[int, float]]`, unpacked in T3.
  `caption_model`/`planner_model`/`tag_score_min` sourced from `settings` in T5.
- **Determinism:** query and caption share the fake embed space (`FakeInferenceClient.embed`),
  so cosine is exact; agent turns are queued `complete` responses, like
  `test_memory_compose.py`.
