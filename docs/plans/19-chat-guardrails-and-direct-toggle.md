# Chat Guardrails + Direct-Answer Toggle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two global per-owner chat toggles — *Guardrails* (off by default) and *Direct answers* (on by default) — that reshape the §10 RAG pipeline.

**Architecture:** Two booleans persist in a new `chat_prefs` table (global per owner, all sessions). `/chat/stream` reads them each turn. *Guardrails ON* reuses `route`'s existing `none` verdict as an on-topic gate and streams a fixed refusal for off-topic questions. *Direct answers OFF* skips the whole deterministic `direct_answer` step and runs a bounded, fully-agentic tool loop where the model calls real `count_photos` / `list_memories` / `count_periods` / `search` tools, then a final grounded answer streams. Both toggles are independent.

**Tech Stack:** Python, FastAPI, SQLite (WAL), SSE streaming, pytest, Jinja2, vanilla JS.

**Spec:** `docs/design.md` §10 (this plan updates §10 prose + the §10 mermaid in the same turn as the code).

## Global Constraints

- One model process only: `app` stays a thin client; all model calls go through the injected `InferenceClient`. Never import torch/transformers here. (CLAUDE.md)
- `docs/design.md` §10 is the source of truth; update it (prose + mermaid) in the SAME turn as the code.
- Defaults are load-bearing: `guardrails=0` (off), `direct_answers=1` (on). With defaults, behaviour is **byte-for-byte today's** — every existing test in `tests/test_web_chat.py` must stay green.
- Every matcher/branch degrades, never crashes (route failure → `none`; tool/loop failure → graceful answer + `done`).
- Tests live in `tests/`; every new module needs tests.

---

### Task 1: `chat_prefs` table + schema bump

**Files:**
- Modify: `db/schema.sql` (append table)
- Modify: `db/connection.py:7` (`SCHEMA_VERSION = 8` → `9`)
- Test: `tests/test_chat_prefs.py`

**Interfaces:**
- Produces: `chat_prefs(owner_id INTEGER PRIMARY KEY, guardrails INTEGER NOT NULL DEFAULT 0, direct_answers INTEGER NOT NULL DEFAULT 1)`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_chat_prefs.py
from db.connection import connect, migrate

def test_migrate_creates_chat_prefs_table(tmp_path):
    conn = connect(tmp_path / "t.db")
    migrate(conn)
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(chat_prefs)")}
    assert cols == {"owner_id", "guardrails", "direct_answers"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_chat_prefs.py -v`
Expected: FAIL (no `chat_prefs` table).

- [ ] **Step 3: Implement**

Append to `db/schema.sql`:

```sql

-- Global per-owner chat toggles (§10): guardrails (off) restricts chat to the
-- library; direct_answers (on) keeps the deterministic direct-DB step. Applied
-- across every session, not per-session.
CREATE TABLE IF NOT EXISTS chat_prefs (
  owner_id       INTEGER PRIMARY KEY,
  guardrails     INTEGER NOT NULL DEFAULT 0,
  direct_answers INTEGER NOT NULL DEFAULT 1
);
```

Set `db/connection.py`: `SCHEMA_VERSION = 9`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_chat_prefs.py -v`
Expected: PASS.

- [ ] **Step 5: Commit** (user commits — skip actual `git commit`).

---

### Task 2: `chat/prefs.py` — read/write helpers

**Files:**
- Create: `chat/prefs.py`
- Test: `tests/test_chat_prefs.py` (extend)

**Interfaces:**
- Produces:
  - `ChatPrefs` (frozen dataclass): `guardrails: bool = False`, `direct_answers: bool = True`
  - `get_prefs(conn, owner_id) -> ChatPrefs`
  - `set_prefs(conn, owner_id, *, guardrails: bool, direct_answers: bool) -> None`

- [ ] **Step 1: Write the failing tests**

```python
from chat.prefs import ChatPrefs, get_prefs, set_prefs

def test_defaults_when_no_row(tmp_path):
    conn = connect(tmp_path / "t.db"); migrate(conn)
    assert get_prefs(conn, 1) == ChatPrefs(guardrails=False, direct_answers=True)

def test_set_then_get_round_trips(tmp_path):
    conn = connect(tmp_path / "t.db"); migrate(conn)
    set_prefs(conn, 1, guardrails=True, direct_answers=False)
    assert get_prefs(conn, 1) == ChatPrefs(guardrails=True, direct_answers=False)

def test_set_upserts_second_write(tmp_path):
    conn = connect(tmp_path / "t.db"); migrate(conn)
    set_prefs(conn, 1, guardrails=True, direct_answers=True)
    set_prefs(conn, 1, guardrails=False, direct_answers=False)
    assert get_prefs(conn, 1) == ChatPrefs(guardrails=False, direct_answers=False)
```

- [ ] **Step 2: Run — FAIL** (`chat.prefs` missing).

- [ ] **Step 3: Implement `chat/prefs.py`**

```python
import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True)
class ChatPrefs:
    """Global per-owner chat toggles (§10). Defaults reproduce today's pipeline."""
    guardrails: bool = False
    direct_answers: bool = True


def get_prefs(conn: sqlite3.Connection, owner_id: int) -> ChatPrefs:
    row = conn.execute(
        "SELECT guardrails, direct_answers FROM chat_prefs WHERE owner_id = ?",
        (owner_id,),
    ).fetchone()
    if row is None:
        return ChatPrefs()
    return ChatPrefs(bool(row["guardrails"]), bool(row["direct_answers"]))


def set_prefs(
    conn: sqlite3.Connection, owner_id: int, *, guardrails: bool, direct_answers: bool
) -> None:
    conn.execute(
        "INSERT INTO chat_prefs(owner_id, guardrails, direct_answers) VALUES (?, ?, ?)"
        " ON CONFLICT(owner_id) DO UPDATE SET"
        " guardrails = excluded.guardrails, direct_answers = excluded.direct_answers",
        (owner_id, int(guardrails), int(direct_answers)),
    )
```

- [ ] **Step 4: Run — PASS.**
- [ ] **Step 5: Commit.**

---

### Task 3: Prompts — refusal constant + agentic-answer prompt

**Files:**
- Modify: `inference/prompts.py` (append)
- Test: `tests/test_prompts.py` (extend)

**Interfaces:**
- Produces:
  - `GUARDRAIL_REFUSAL: str` — fixed on-topic redirect, contains "only answer questions about your photos".
  - `agentic_answer_messages(question: str, gathered_block: str) -> list[ChatMessage]`

- [ ] **Step 1: Write the failing tests**

```python
from inference.prompts import GUARDRAIL_REFUSAL, agentic_answer_messages

def test_guardrail_refusal_is_on_topic_redirect():
    assert "only answer questions about your photos" in GUARDRAIL_REFUSAL.lower()

def test_agentic_answer_prompt_carries_facts_and_question():
    msgs = agentic_answer_messages("how many dogs?", "count: 4 photo(s) matching \"dogs\"")
    assert msgs[0]["role"] == "system"
    assert "count: 4" in msgs[1]["content"]
    assert "how many dogs?" in msgs[1]["content"]
```

- [ ] **Step 2: Run — FAIL.**

- [ ] **Step 3: Implement (append to `inference/prompts.py`)**

```python
# Guardrails ON (§10): a fixed, model-free refusal for a question routed as NOT
# about the user's photos/memories. Streamed verbatim — no lease, no generation.
GUARDRAIL_REFUSAL = (
    "I can only answer questions about your photos and this app — finding photos, "
    "memories, counts, or how the library works. Ask me about those."
)


# Direct answers OFF (§10): the fully-agentic loop already gathered REAL facts
# (count/memories/periods lines) and candidate photos; this final call turns them
# into the streamed answer. Unlike `_CHAT_SYSTEM` it must state a count even when
# no photo is cited (a "how many" answer has a number, not a thumbnail).
_AGENTIC_ANSWER_SYSTEM = (
    "You answer a question about a personal photo library using ONLY the gathered "
    "facts and photos below. Fact lines starting with 'count:', 'memories:', or "
    "'month(s)/year(s) with photos:' are REAL numbers computed for you — state them "
    "in a natural sentence; never invent or infer a count. For each photo that "
    "matches, cite it inline as [photo:ID] using the exact id. Never invent a "
    "subject, person, place, or date not written in a caption/tag. If nothing below "
    "answers the question, say so plainly."
)


def agentic_answer_messages(question: str, gathered_block: str) -> list[ChatMessage]:
    """Final grounded answer for the fully-agentic direct-OFF path (§10)."""
    return [
        {"role": "system", "content": _AGENTIC_ANSWER_SYSTEM},
        {"role": "user", "content": f"Gathered:\n{gathered_block}\n\nQuestion: {question}"},
    ]
```

- [ ] **Step 4: Run — PASS.**
- [ ] **Step 5: Commit.**

---

### Task 4: `chat/agent.py` — fully-agentic gather loop with count tools

**Files:**
- Modify: `chat/agent.py` (add schema, system, loop, tool runner; add `build_context` import)
- Test: `tests/test_chat_agent.py` (extend)

**Interfaces:**
- Consumes: `search_photos`, `count_photos`, `list_memories`, `count_periods`, `_summarise`, `build_context`.
- Produces:
  - `agentic_gather(conn, embedder, client, model, owner_id, question, *, max_rounds=4) -> tuple[str, bool]`
    returns `(gathered_block, grounded)`; `grounded=False` ⇒ answer from general knowledge.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_chat_agent.py (extend)
import json
from chat.agent import agentic_gather
from embedding.fakes import FakeEmbedder
from inference.fakes import FakeInferenceClient
from tests.factories import add_photo

def test_agentic_gather_counts_via_the_count_tool(chat_conn):  # chat_conn: migrated conn
    for pid in range(1, 6):
        add_photo(chat_conn, photo_id=pid, content_hash=f"h{pid}", thumb_key=f"{pid}.jpg")
    fake = FakeInferenceClient(responses=[
        json.dumps({"action": "count_photos", "query": "", "grain": None}),
        json.dumps({"action": "answer", "query": None, "grain": None}),
    ])
    block, grounded = agentic_gather(chat_conn, FakeEmbedder(), fake, "fake", 1, "how many photos?")
    assert grounded is True
    assert "count: 5" in block

def test_agentic_gather_general_question_is_not_grounded(chat_conn):
    fake = FakeInferenceClient(responses=[
        json.dumps({"action": "answer", "query": None, "grain": None}),
    ])
    block, grounded = agentic_gather(chat_conn, FakeEmbedder(), fake, "fake", 1, "hi there")
    assert grounded is False
    assert block == ""
```

Add a `chat_conn` fixture (migrated in-memory/temp DB) if not already shared — mirror existing agent tests' setup.

- [ ] **Step 2: Run — FAIL.**

- [ ] **Step 3: Implement (in `chat/agent.py`)**

Add import near the top: `from chat.context import build_context`.

```python
_AGENTIC_SYSTEM = (
    "You are an agent for a personal photo app. Use tools to get REAL data before "
    "answering — never guess a count or invent a photo. One JSON action per step:\n"
    '- {"action":"search","query":"..."} — candidate photos matching a phrase.\n'
    '- {"action":"count_photos","query":"..."} — the REAL count matching a phrase '
    '(empty query = whole-library total).\n'
    '- {"action":"list_memories"} — the user\'s saved memories with sizes.\n'
    '- {"action":"count_periods","grain":"month"|"year"} — distinct months/years with photos.\n'
    '- {"action":"answer"} — you have enough; stop gathering.\n'
    "Call as many tools as you need, then answer. Reply with ONLY the JSON object."
)

_AGENTIC_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "action": {
            "type": "string",
            "enum": ["search", "count_photos", "list_memories", "count_periods", "answer"],
        },
        "query": {"type": ["string", "null"]},
        "grain": {"type": ["string", "null"], "enum": ["month", "year", None]},
    },
    "required": ["action", "query", "grain"],
}


def agentic_gather(
    conn, embedder, client, model, owner_id, question, *, max_rounds: int = 4
) -> tuple[str, bool]:
    """Fully-agentic direct-OFF path (§10): the model calls REAL count/search tools,
    then we build a grounded block for the final answer. Returns (block, grounded);
    grounded False ⇒ nothing was gathered, so answer from general knowledge. Any
    failure degrades to whatever was gathered so far (never raises)."""
    messages = [
        {"role": "system", "content": _AGENTIC_SYSTEM},
        {"role": "user", "content": question},
    ]
    facts: list[str] = []
    photo_ids: list[int] = []
    seen: set[int] = set()
    try:
        for round_no in range(max_rounds):
            turn = _agentic_turn(client, model, messages, force=round_no == max_rounds - 1)
            if turn is None or turn.get("action") == "answer":
                break
            result = _run_agentic_tool(conn, embedder, owner_id, turn, photo_ids, seen, facts)
            messages.append({"role": "assistant", "content": json.dumps(turn)})
            messages.append({"role": "user", "content": result})
    except Exception:  # noqa: BLE001 — degrade to what we gathered; never crash chat
        pass
    parts = list(facts)
    if photo_ids:
        parts.append(build_context(conn, photo_ids))
    grounded = bool(facts or photo_ids)
    return ("\n".join(parts) if grounded else ""), grounded


def _run_agentic_tool(conn, embedder, owner_id, turn, photo_ids, seen, facts) -> str:
    """Run one agentic tool; append its result to facts/photo_ids and return a short
    text line for the model's next turn. Search widens the candidate pool; the three
    count tools produce REAL numbers as fact lines the final answer states."""
    action = turn.get("action")
    if action == "search":
        query = turn.get("query")
        ids = search_photos(conn, embedder, owner_id, query, k=10) if isinstance(query, str) and query else []
        for pid in ids:
            if pid not in seen:
                seen.add(pid)
                photo_ids.append(pid)
        return _summarise(conn, owner_id, ids)
    if action == "count_photos":
        query = (turn.get("query") or "").strip()
        n = count_photos(conn, owner_id, query)
        line = f"count: {n} photo(s)" + (f' matching "{query}"' if query else " in the library")
        facts.append(line)
        return line
    if action == "list_memories":
        mems = list_memories(conn, owner_id)
        line = "memories: " + ("; ".join(f"{m['name']} ({m['size']} photos)" for m in mems) or "none")
        facts.append(line)
        return line
    if action == "count_periods":
        grain = "year" if (turn.get("grain") or "").startswith("year") else "month"
        n, _ = count_periods(conn, owner_id, grain)
        line = f"{grain}(s) with photos: {n}"
        facts.append(line)
        return line
    return "no result"


def _agentic_turn(client, model, messages, force: bool):
    """One agentic tool-selection turn -> parsed dict, or None. Schema-constrained;
    a backend without structured output falls back to a plain call. `force` nudges
    the last round to stop gathering."""
    turn_messages = messages
    if force:
        turn_messages = [*messages, {"role": "user", "content": "Answer now (action=answer)."}]
    try:
        raw = client.complete(model, turn_messages, timeout=60.0, json_schema=_AGENTIC_SCHEMA)
    except Exception:  # noqa: BLE001 — backend without structured output → plain call
        raw = client.complete(model, turn_messages, timeout=60.0)
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        return None
    parsed = json.loads(raw[start : end + 1])
    return parsed if isinstance(parsed, dict) else None
```

- [ ] **Step 4: Run — PASS** (`uv run pytest tests/test_chat_agent.py -v`).
- [ ] **Step 5: Commit.**

---

### Task 5: `POST /chat/prefs` + seed checkbox state into `/chat`

**Files:**
- Modify: `web/app.py` (import prefs; add endpoint; pass prefs to `chat.html`)
- Modify: `web/templates/chat.html` (two checkboxes)
- Modify: `web/static/chat.js` (post on toggle)
- Modify: `web/static/app.css` (checkbox row styling — minimal)
- Test: `tests/test_web_chat.py` (extend)

**Interfaces:**
- Consumes: `get_prefs`, `set_prefs` (Task 2).
- Produces: `POST /chat/prefs` (form fields `guardrails`, `direct_answers`; unchecked box omits the field) → 303 redirect to `/chat`. `/chat` context gains `prefs`.

- [ ] **Step 1: Write the failing tests**

```python
def test_chat_page_shows_toggles_with_defaults(chat_client):
    body = chat_client.get("/chat").text
    assert 'name="guardrails"' in body and 'name="direct_answers"' in body
    # direct answers ON by default (checked), guardrails OFF (unchecked)
    import re
    direct = re.search(r'<input[^>]*name="direct_answers"[^>]*>', body).group(0)
    guard = re.search(r'<input[^>]*name="guardrails"[^>]*>', body).group(0)
    assert "checked" in direct
    assert "checked" not in guard

def test_prefs_post_persists_and_reflects(chat_client):
    chat_client.post("/chat/prefs", data={"guardrails": "on"})  # direct omitted → off
    body = chat_client.get("/chat").text
    import re
    guard = re.search(r'<input[^>]*name="guardrails"[^>]*>', body).group(0)
    direct = re.search(r'<input[^>]*name="direct_answers"[^>]*>', body).group(0)
    assert "checked" in guard
    assert "checked" not in direct
```

- [ ] **Step 2: Run — FAIL.**

- [ ] **Step 3: Implement**

`web/app.py` — import: `from chat.prefs import get_prefs, set_prefs`.

In `chat_page`, add `prefs = get_prefs(ctx.conn, ctx.settings.owner_id)` and pass `"prefs": prefs` into the template context.

Add endpoint:

```python
    @app.post("/chat/prefs")
    def chat_prefs(guardrails: str | None = Form(None), direct_answers: str | None = Form(None)) -> RedirectResponse:
        # Global per-owner chat toggles (§10). An unchecked box omits its field, so
        # presence == on. Persisted across every session.
        ctx = context()
        set_prefs(
            ctx.conn, ctx.settings.owner_id,
            guardrails=guardrails is not None, direct_answers=direct_answers is not None,
        )
        return RedirectResponse("/chat", status_code=303)
```

`web/templates/chat.html` — inside `.chat-head`, add a toggles row (a plain form auto-submitting on change via chat.js):

```html
      <form id="chat-prefs" method="post" action="/chat/prefs" class="chat-toggles">
        <label><input type="checkbox" name="direct_answers" {% if prefs.direct_answers %}checked{% endif %}> Direct answers</label>
        <label><input type="checkbox" name="guardrails" {% if prefs.guardrails %}checked{% endif %}> Guardrails</label>
      </form>
```

`web/static/chat.js` — in `initChat()` (after wiring the textarea), auto-submit the prefs form on any checkbox change:

```javascript
  const prefs = document.getElementById("chat-prefs");
  if (prefs) prefs.addEventListener("change", () => prefs.submit());
```

`web/static/app.css` — minimal row styling:

```css
.chat-toggles { display: flex; gap: 1rem; font-size: 0.85rem; align-items: center; }
.chat-toggles label { display: inline-flex; gap: 0.3rem; align-items: center; cursor: pointer; }
```

- [ ] **Step 4: Run — PASS.**
- [ ] **Step 5: Commit.**

---

### Task 6: Wire toggles into `/chat/stream`

**Files:**
- Modify: `web/app.py` (`chat_stream.events()` — branch on prefs)
- Test: `tests/test_web_chat.py` (extend)

**Interfaces:**
- Consumes: `get_prefs` (Task 2), `agentic_gather` (Task 4), `GUARDRAIL_REFUSAL` + `agentic_answer_messages` (Task 3), existing `direct_answer` / `route` / `search_library` / `search_memories`.

**Behaviour matrix (defaults = guardrails off, direct on → today's flow unchanged):**
- direct ON: `direct_answer` runs first (as today).
- After it declines (or direct OFF): take the CHAT lease.
- Inside the lease compute `decision = route(...)` **only if** `guardrails` OR `direct_answers` (fully-agentic + guardrails-off needs no route).
- guardrails ON and `decision["tool"] == "none"` → stream `GUARDRAIL_REFUSAL`, persist, `done`, return.
- direct ON → existing one-shot path using `decision` (search_library / search_memories / none → chat_messages / general_chat_messages).
- direct OFF → `agentic_gather` → `agentic_answer_messages` (grounded) or `general_chat_messages` (not grounded) → stream.

- [ ] **Step 1: Write the failing tests**

```python
def test_guardrails_on_refuses_off_topic(settings, monkeypatch):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    fake = FakeInferenceClient(responses=['{"tool": "none", "query": null}'])  # no stream needed
    monkeypatch.setattr(Settings, "build_inference_client", lambda self: (fake, "fake"))
    app = create_app(settings)
    from chat.prefs import set_prefs
    set_prefs(app.state.context.conn, settings.owner_id, guardrails=True, direct_answers=True)
    with TestClient(app) as tc:
        body = tc.get("/chat/stream?q=should+I+walk+or+drive").text
    assert "only answer questions about your photos" in body.lower()
    assert "event: done" in body

def test_guardrails_off_answers_off_topic_generally(settings, monkeypatch):
    # unchanged default behaviour — mirrors test_off_topic_question_is_answered_generally
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    fake = FakeInferenceClient(responses=['{"tool": "none", "query": null}'], streams=[["Walking ", "is ", "fine."]])
    monkeypatch.setattr(Settings, "build_inference_client", lambda self: (fake, "fake"))
    app = create_app(settings)
    with TestClient(app) as tc:
        body = tc.get("/chat/stream?q=should+I+walk+or+drive").text
    assert "Walking" in body
    assert "only answer questions about your photos" not in body.lower()

def test_direct_off_counts_go_through_the_agentic_count_tool(settings, monkeypatch):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    fake = FakeInferenceClient(
        responses=[  # the agentic gather loop
            '{"action": "count_photos", "query": "", "grain": null}',
            '{"action": "answer", "query": null, "grain": null}',
        ],
        streams=[["You have ", "7 ", "photos."]],  # the final grounded answer
    )
    monkeypatch.setattr(Settings, "build_inference_client", lambda self: (fake, "fake"))
    app = create_app(settings)
    conn = app.state.context.conn
    for pid in range(1, 8):
        add_photo(conn, photo_id=pid, content_hash=f"h{pid}", thumb_key=f"{pid}.jpg")
    from chat.prefs import set_prefs
    set_prefs(conn, settings.owner_id, guardrails=False, direct_answers=False)
    with TestClient(app) as tc:
        body = tc.get("/chat/stream?q=how+many+photos+do+I+have").text
    assert "event: done" in body
    # the REAL count reached the final prompt as a fact line
    final_msgs = fake.calls[-1][1]
    assert "count: 7" in final_msgs[1]["content"]

def test_direct_off_memory_show_has_no_card(settings, monkeypatch):
    # whole direct-DB step skipped → memory-show is prose, no rendered card event
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    fake = FakeInferenceClient(
        responses=['{"action": "answer", "query": null, "grain": null}'],
        streams=[["You have some memories."]],
    )
    monkeypatch.setattr(Settings, "build_inference_client", lambda self: (fake, "fake"))
    app = create_app(settings)
    from chat.prefs import set_prefs
    set_prefs(app.state.context.conn, settings.owner_id, guardrails=False, direct_answers=False)
    with TestClient(app) as tc:
        body = tc.get("/chat/stream?q=show+me+my+memories").text
    assert "event: memory" not in body
```

- [ ] **Step 2: Run — FAIL.**

- [ ] **Step 3: Implement** — rewrite `events()` per the behaviour matrix above. Read `prefs = get_prefs(ctx.conn, owner_id)` at the top of `chat_stream`. Gate `direct_answer` behind `prefs.direct_answers`. Inside the lease, compute `decision` once (only when needed), apply the guardrail refusal, then branch direct-ON (existing one-shot) vs direct-OFF (`agentic_gather` + final stream). Reuse the existing `_done()` / persistence / streaming-loop code; the refusal path persists with `add_message(..., [])` and emits `_done()`.

- [ ] **Step 4: Run the whole web-chat suite — PASS** (`uv run pytest tests/test_web_chat.py -v`), including every pre-existing test (defaults unchanged).
- [ ] **Step 5: Commit.**

---

### Task 7: Update `docs/design.md` §10 (prose + mermaid)

**Files:**
- Modify: `docs/design.md` §10

- [ ] **Step 1:** Add a short subsection documenting the two global toggles: *Guardrails* (default off; ON = reuse `route`'s `none` verdict as an on-topic gate, stream a fixed refusal) and *Direct answers* (default on; OFF = skip the whole direct-DB step, run the fully-agentic loop with real `count_photos`/`list_memories`/`count_periods`/`search` tools, memory-show becomes prose with no card). State that defaults reproduce today's pipeline.
- [ ] **Step 2:** Update the §10 mermaid: a `chat_prefs` decision up front — `direct_answers?` gates the "Direct-DB answerable?" node; a `guardrails?` branch off `route==none` that leads to a "Refuse (fixed, no model)" node; and a "Fully-agentic loop (count/search tools) → grounded stream" node for direct-OFF.
- [ ] **Step 3:** Re-read §10 end-to-end; confirm no sentence still claims "Direct-DB first (no model)" or "No off-topic gate" as unconditional — both are now pref-gated.
- [ ] **Step 4: Commit.**

---

### Task 8: Full suite + self-review

- [ ] **Step 1:** `uv run pytest -q` — everything green.
- [ ] **Step 2:** Manual smoke via `/run` or the app: toggle each box, confirm persistence across reload, guardrails refusal, direct-off count answer.
- [ ] **Step 3:** Confirm `docs/design.md` §10 matches the running behaviour (CLAUDE.md invariant).
- [ ] **Step 4: Commit.**

## Self-Review

- **Spec coverage:** Guardrails toggle (Tasks 1,2,3,5,6,7) ✓; Direct-answer toggle + fully-agentic count tools (Tasks 1,2,4,6,7) ✓; global persistence (Tasks 1,2,5) ✓; UI checkboxes (Task 5) ✓; §10 doc (Task 7) ✓.
- **Placeholder scan:** none — every step has concrete code.
- **Type consistency:** `ChatPrefs(guardrails, direct_answers)`, `get_prefs`/`set_prefs`, `agentic_gather -> (str, bool)`, `agentic_answer_messages`, `GUARDRAIL_REFUSAL` used consistently across tasks.
