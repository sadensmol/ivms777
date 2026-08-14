# Ask-your-library Chat Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the `/chat` page (§10): a question retrieves the most relevant photos, an LLM answers grounded only in their captions/tags/EXIF, tokens stream over SSE, and `[photo:ID]` citations render inline as clickable thumbnails.

**Architecture:** A new `chat/` package does retrieval (reusing the existing semantic + keyword + RRF fusion) and packs the top photos into a compact context block. `inference/client.py` gains a streaming `stream()` method. A FastAPI SSE endpoint feeds the block to the planner model and relays token deltas plus the retrieved photo ids to a small vanilla-JS `EventSource` client that renders citations.

**Tech Stack:** FastAPI `StreamingResponse` (`text/event-stream`), Jinja2, HTMX (already loaded) for the page shell, plain-JS `EventSource` for the stream, httpx for the OpenAI-compatible backend.

**Spec:** `docs/design.md` §10 (chat), §9 (retrieval), §9.1 (planner is optional with a fusion fallback), §13 (`/chat` page), §14 (code layout).

## Global Constraints

- **No query planner in v1.** §10 step 1 routes questions through the planner (§9.1), which is not built. §9.1 makes it optional with a fusion fallback, so v1 chat retrieves via semantic + keyword fusion on the raw question. Task 1 updates §10 to describe this before any code lands. The planner stays a documented future enhancement.
- **The owner does all git commits.** Never run `git commit`/`git add`. Each task ends at a green test suite; the owner commits.
- **Flat layout** (§14): imports are `from chat.retrieve import ...`, `from inference.client import ...`. No wrapping package dir.
- **Owner-scoped** (§3.2): every query filters on `owner_id`. Retrieval already does; the chat route passes `ctx.settings.owner_id`.
- **Tests use fakes** (§15): `use_fake_embedder=True, use_fake_inference=True`. No torch, no network, milliseconds.
- **Grounding is mandatory** (§10): the model answers only from retrieved context and is told to say so when nothing relevant is retrieved. Sources always render as thumbnails.
- Run tests with `uv run pytest`.

---

### Task 1: Update the design doc (design-first gate)

Per project rule, the doc changes before the code. No code, no test — an edit that makes §10/§13/§14 match what this plan builds.

**Files:**
- Modify: `docs/design.md` (§10, §13 nav sentence, §14 `chat/` block)

- [ ] **Step 1: Rewrite §10 step 1** so it reads (keep the rest of §10 intact):

```
1. The question is retrieved directly by semantic + keyword fusion (§9), the
   same path interactive search uses. The query planner (§9.1) is a future
   enhancement; when it lands, the question will pass through it first. Until
   then the raw question drives retrieval.
```

- [ ] **Step 2: In §10**, change "Interactive requests use the planner model" note to state the chat route calls the planner model directly (`planner_model`), which stays loaded and is small.

- [ ] **Step 3: Update the §13 nav-order sentence** to `Upload → Library → Organize → Chat` and confirm the `/chat` bullet already lists "question box, streamed answer, inline thumbnail citations" (it does).

- [ ] **Step 4: Fill the `chat/` block in §14** so it reads:

```
chat/
  retrieve.py          # question -> top photo ids via fusion (reuses search/)
  context.py           # photo ids -> compact grounding block + evidence list
```

- [ ] **Step 5: Re-read §10 end to end** and verify no sentence now describes behaviour the plan will not build (no planner, no numeric relevance floor beyond "retrieval returned nothing").

---

### Task 2: Streaming on the inference client

**Files:**
- Modify: `inference/client.py`
- Modify: `inference/fakes.py`
- Test: `tests/test_inference_stream.py`

**Interfaces:**
- Produces:
  - `InferenceClient.stream(model: str, messages: list[ChatMessage], *, timeout: float = 120.0) -> Iterator[str]` — yields answer text deltas in order.
  - `FakeInferenceClient(responses=None, streams=None)` where `streams: list[list[str]]`; each `stream()` call pops and yields the next chunk list.

- [ ] **Step 1: Write the failing test** `tests/test_inference_stream.py`:

```python
import httpx

from inference.client import OpenAICompatClient
from inference.fakes import FakeInferenceClient


def test_fake_stream_yields_queued_chunks():
    fake = FakeInferenceClient(streams=[["Hello ", "[photo:1]", " there"]])
    out = list(fake.stream("m", [{"role": "user", "content": "hi"}]))
    assert out == ["Hello ", "[photo:1]", " there"]
    assert fake.calls == [("m", [{"role": "user", "content": "hi"}])]


def test_openai_client_stream_parses_sse_deltas():
    body = (
        'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n'
        'data: {"choices":[{"delta":{}}]}\n\n'
        "data: [DONE]\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body)

    client = OpenAICompatClient("http://x/v1", transport=httpx.MockTransport(handler))
    out = list(client.stream("m", [{"role": "user", "content": "hi"}]))
    assert "".join(out) == "Hello"
```

- [ ] **Step 2: Run it, verify it fails**

Run: `uv run pytest tests/test_inference_stream.py -v`
Expected: FAIL — `stream` not defined.

- [ ] **Step 3: Add `stream` to the protocol and `OpenAICompatClient`** in `inference/client.py`:

```python
from collections.abc import Iterator
import json
```

Add to the `InferenceClient` Protocol:

```python
    def stream(
        self,
        model: str,
        messages: list[ChatMessage],
        *,
        timeout: float = 120.0,
    ) -> Iterator[str]: ...
```

Add to `OpenAICompatClient`:

```python
    def stream(
        self,
        model: str,
        messages: list[ChatMessage],
        *,
        timeout: float = 120.0,
    ) -> Iterator[str]:
        payload = {"model": model, "messages": messages, "stream": True}
        with self._client.stream(
            "POST", "/chat/completions", json=payload, timeout=timeout
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                delta = json.loads(data)["choices"][0].get("delta", {}).get("content")
                if delta:
                    yield delta
```

- [ ] **Step 4: Add `stream` to `FakeInferenceClient`** in `inference/fakes.py`:

```python
    def __init__(self, responses=None, streams=None) -> None:
        self._responses = list(responses or [])
        self._streams = list(streams or [])
        self.calls: list[tuple[str, list[ChatMessage]]] = []

    def stream(self, model, messages, *, timeout: float = 120.0):
        self.calls.append((model, messages))
        assert self._streams, "FakeInferenceClient ran out of queued streams"
        yield from self._streams.pop(0)
```

Keep the existing `complete` working: it still reads `self._responses`. Update its call recording to remain unchanged.

- [ ] **Step 5: Run tests, verify pass**

Run: `uv run pytest tests/test_inference_stream.py tests/test_caption_stage.py -v`
Expected: PASS (caption test proves `complete`/`FakeInferenceClient` still work).

---

### Task 3: Chat retrieval

**Files:**
- Create: `chat/__init__.py` (empty)
- Create: `chat/retrieve.py`
- Test: `tests/test_chat_retrieve.py`

**Interfaces:**
- Consumes: `search.semantic.search_photos`, `search.keyword.keyword_search`, `search.fusion.reciprocal_rank_fusion`, `embedding.base.Embedder`.
- Produces: `retrieve(conn, embedder, owner_id: int, question: str, k: int = 30) -> list[int]` — fused photo ids, best first, capped at `k`. Empty question or no matches → `[]`.

- [ ] **Step 1: Write the failing test** `tests/test_chat_retrieve.py`:

```python
from chat.retrieve import retrieve
from embedding.fakes import FakeEmbedder
from embedding.store import write_vector
from tests.factories import add_photo


def test_retrieve_fuses_semantic_and_keyword(conn):
    fake = FakeEmbedder()
    for pid, word in ((1, "beach"), (2, "keyboard")):
        add_photo(conn, photo_id=pid, content_hash=word, thumb_key=f"{word}.jpg")
        write_vector(conn, pid, fake.embed_texts([word])[0])
    ids = retrieve(conn, fake, owner_id=1, question="beach", k=30)
    assert ids and ids[0] == 1


def test_retrieve_empty_question_returns_nothing(conn):
    assert retrieve(conn, FakeEmbedder(), owner_id=1, question="   ", k=30) == []


def test_retrieve_caps_at_k(conn):
    fake = FakeEmbedder()
    for pid in range(1, 6):
        add_photo(conn, photo_id=pid, content_hash=f"h{pid}" * 8, thumb_key=f"{pid}.jpg")
        write_vector(conn, pid, fake.embed_texts([f"h{pid}"])[0])
    assert len(retrieve(conn, fake, owner_id=1, question="anything", k=2)) <= 2
```

- [ ] **Step 2: Run it, verify it fails**

Run: `uv run pytest tests/test_chat_retrieve.py -v`
Expected: FAIL — `chat.retrieve` missing.

- [ ] **Step 3: Implement** `chat/retrieve.py`:

```python
import sqlite3

from embedding.base import Embedder
from search.fusion import reciprocal_rank_fusion
from search.keyword import keyword_search
from search.semantic import search_photos


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
```

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest tests/test_chat_retrieve.py -v`
Expected: PASS.

---

### Task 4: Context assembly

**Files:**
- Create: `chat/context.py`
- Test: `tests/test_chat_context.py`

**Interfaces:**
- Produces:
  - `build_context(conn, photo_ids: list[int]) -> str` — one compact block per photo: `[photo:ID]` header, date, caption, top tags, and EXIF facts (camera, lens, iso, aperture, shutter_speed, focal_length, gps). ~60 tokens each. Empty list -> `"No photos matched."`.
- The `[photo:ID]` tokens in the block are what the model is told to cite, so the ids the client renders come straight from here.

- [ ] **Step 1: Write the failing test** `tests/test_chat_context.py`:

```python
from chat.context import build_context
from tests.factories import add_photo


def test_block_carries_id_caption_and_facts(conn):
    pid = add_photo(
        conn, content_hash="a" * 64, thumb_key="a.jpg",
        caption="A dog on a beach", shot_at="2025-06-14T18:30:00", camera="Canon R6",
    )
    conn.execute(
        "INSERT INTO photo_facets(photo_id, key, value_text) VALUES (?, 'lens', '50mm')",
        (pid,),
    )
    block = build_context(conn, [pid])
    assert f"[photo:{pid}]" in block
    assert "A dog on a beach" in block
    assert "Canon R6" in block
    assert "50mm" in block


def test_empty_list_is_a_no_match_sentinel(conn):
    assert build_context(conn, []) == "No photos matched."
```

- [ ] **Step 2: Run it, verify it fails**

Run: `uv run pytest tests/test_chat_context.py -v`
Expected: FAIL — `chat.context` missing.

- [ ] **Step 3: Implement** `chat/context.py`:

```python
import sqlite3

# EXIF facts worth carrying into an answer (§10 step 3); order is the display order.
_FACT_KEYS = ("camera_model", "lens", "iso", "aperture", "shutter_speed",
              "focal_length", "place_city", "place_country")


def build_context(conn: sqlite3.Connection, photo_ids: list[int]) -> str:
    """A compact, grounded block per photo for the chat prompt (§10 step 3)."""
    if not photo_ids:
        return "No photos matched."
    blocks: list[str] = []
    for photo_id in photo_ids:
        photo = conn.execute(
            "SELECT id, shot_at, camera, caption FROM photos WHERE id = ?", (photo_id,)
        ).fetchone()
        if photo is None:
            continue
        facts = {
            row["key"]: (row["value_text"] if row["value_text"] is not None else row["value_num"])
            for row in conn.execute(
                "SELECT key, value_text, value_num FROM photo_facets WHERE photo_id = ?",
                (photo_id,),
            )
        }
        tags = [
            row["label"] for row in conn.execute(
                "SELECT t.label FROM photo_tags pt JOIN tags t ON t.id = pt.tag_id"
                " WHERE pt.photo_id = ? ORDER BY pt.score DESC LIMIT 6",
                (photo_id,),
            )
        ]
        parts = [f"[photo:{photo['id']}]"]
        if photo["shot_at"]:
            parts.append(f"date: {photo['shot_at']}")
        if photo["caption"]:
            parts.append(f"caption: {photo['caption']}")
        if tags:
            parts.append("tags: " + ", ".join(tags))
        if photo["camera"]:
            parts.append(f"camera: {photo['camera']}")
        facts_line = ", ".join(f"{k}={facts[k]}" for k in _FACT_KEYS if k in facts)
        if facts_line:
            parts.append(facts_line)
        blocks.append("\n".join(parts))
    return "\n\n".join(blocks)
```

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest tests/test_chat_context.py -v`
Expected: PASS.

---

### Task 5: Chat prompt

**Files:**
- Modify: `inference/prompts.py`
- Test: `tests/test_prompts.py` (add cases; file exists)

**Interfaces:**
- Produces: `chat_messages(question: str, context_block: str) -> list[ChatMessage]` — a system message enforcing grounding + `[photo:ID]` citation + say-so-when-empty, and a user message carrying the block then the question.

- [ ] **Step 1: Write the failing test** — add to `tests/test_prompts.py`:

```python
from inference.prompts import chat_messages


def test_chat_messages_ground_and_require_citation():
    msgs = chat_messages("what lens did I use?", "[photo:7]\ncaption: a cat")
    system = msgs[0]["content"]
    assert "[photo:" in system  # instructs the citation format
    assert "only" in system.lower()  # grounded only in the provided photos
    user = msgs[1]["content"]
    assert "[photo:7]" in user
    assert "what lens did I use?" in user


def test_chat_messages_handle_no_matches():
    msgs = chat_messages("anything", "No photos matched.")
    assert "No photos matched." in msgs[1]["content"]
```

- [ ] **Step 2: Run it, verify it fails**

Run: `uv run pytest tests/test_prompts.py -k chat -v`
Expected: FAIL — `chat_messages` missing.

- [ ] **Step 3: Implement** — append to `inference/prompts.py`:

```python
_CHAT_SYSTEM = (
    "You answer questions about a personal photo library. Use ONLY the photos "
    "provided below — their captions, tags, and EXIF facts. Never invent photos, "
    "people, places, or dates. Cite every photo you rely on inline as [photo:ID], "
    "using the exact ID from the context. If no photos are provided, or none are "
    "relevant, say you have no photos matching that and stop."
)


def chat_messages(question: str, context_block: str) -> list[ChatMessage]:
    user = f"Photos:\n{context_block}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": _CHAT_SYSTEM},
        {"role": "user", "content": user},
    ]
```

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest tests/test_prompts.py -v`
Expected: PASS.

---

### Task 6: `/chat` page, SSE endpoint, and citation rendering

**Files:**
- Modify: `web/app.py`
- Create: `web/templates/chat.html`
- Create: `web/static/chat.js`
- Modify: `web/templates/base.html` (nav link)
- Test: `tests/test_web_chat.py`

**Interfaces:**
- Consumes: `chat.retrieve.retrieve`, `chat.context.build_context`, `inference.prompts.chat_messages`, `ctx.settings.build_embedder`, `ctx.settings.build_inference_client`, `ctx.settings.planner_model`.
- SSE wire format on `GET /chat/stream?q=...` (`text/event-stream`):
  - `event: sources\ndata: {"ids": [1, 2]}\n\n` — retrieved photo ids, first.
  - `data: {"delta": "..."}\n\n` — one per answer token, in order.
  - `event: done\ndata: {}\n\n` — terminator.

- [ ] **Step 1: Write the failing test** `tests/test_web_chat.py`:

```python
import pytest
from fastapi.testclient import TestClient

from config import Settings
from embedding.fakes import FakeEmbedder
from embedding.store import write_vector
from inference.fakes import FakeInferenceClient
from tests.factories import add_photo
from web.app import create_app


@pytest.fixture
def chat_client(settings, monkeypatch):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    fake_inf = FakeInferenceClient(streams=[["A beach ", "[photo:1]", "."]])
    monkeypatch.setattr(
        Settings, "build_inference_client", lambda self: (fake_inf, "fake")
    )
    app = create_app(settings)
    conn = app.state.context.conn
    fe = FakeEmbedder()
    for pid, word in ((1, "beach"), (2, "keyboard")):
        add_photo(conn, photo_id=pid, content_hash=word, thumb_key=f"{word}.jpg")
        write_vector(conn, pid, fe.embed_texts([word])[0])
    with TestClient(app) as tc:
        yield tc


def test_chat_page_has_question_box_and_nav(chat_client):
    body = chat_client.get("/chat").text
    assert 'id="chat-form"' in body
    assert 'href="/chat"' in body  # nav link present


def test_stream_emits_sources_then_tokens_then_done(chat_client):
    body = chat_client.get("/chat/stream?q=beach").text
    assert "event: sources" in body
    assert '"ids"' in body and '1' in body
    assert "A beach " in body
    assert "[photo:1]" in body  # citation passes through untouched
    assert "event: done" in body


def test_stream_says_so_when_nothing_retrieved(chat_client, monkeypatch):
    # No query -> no retrieval; the model is handed the no-match sentinel.
    fake_inf = FakeInferenceClient(streams=[["I have no photos matching that."]])
    monkeypatch.setattr(Settings, "build_inference_client", lambda self: (fake_inf, "fake"))
    body = chat_client.get("/chat/stream?q=%20").text
    assert "event: sources" in body
    assert "no photos matching" in body.lower()
```

- [ ] **Step 2: Run it, verify it fails**

Run: `uv run pytest tests/test_web_chat.py -v`
Expected: FAIL — `/chat` routes and templates missing.

- [ ] **Step 3: Add the nav link** — in `web/templates/base.html`, add after the Organize link:

```html
      <a href="/chat">Chat</a>
```

- [ ] **Step 4: Create** `web/templates/chat.html`:

```html
{% extends "base.html" %}
{% block title %}Chat — ivms777{% endblock %}
{% block content %}
<h1>Ask your library</h1>
<form id="chat-form" onsubmit="return askLibrary(event)">
  <input id="chat-q" name="q" type="text" autocomplete="off"
         placeholder="e.g. what lens did I use most in Italy?" size="60">
  <button type="submit">Ask</button>
</form>
<div id="chat-sources" class="chat-sources"></div>
<div id="chat-answer" class="chat-answer"></div>
<script src="/static/chat.js"></script>
{% endblock %}
```

- [ ] **Step 5: Create** `web/static/chat.js` — consumes the SSE stream, renders `[photo:ID]` as thumbnails:

```javascript
let source = null;

function renderAnswer(text) {
  const html = text.replace(/\[photo:(\d+)\]/g, (_, id) =>
    `<a href="/photo/${id}"><img class="cite" src="/thumb/${id}" alt="photo ${id}"></a>`
  );
  document.getElementById("chat-answer").innerHTML = html;
}

function askLibrary(event) {
  event.preventDefault();
  const q = document.getElementById("chat-q").value;
  const answer = document.getElementById("chat-answer");
  const sources = document.getElementById("chat-sources");
  answer.textContent = "";
  sources.innerHTML = "";
  if (source) source.close();
  let buffer = "";

  source = new EventSource("/chat/stream?q=" + encodeURIComponent(q));
  source.addEventListener("sources", (e) => {
    const ids = JSON.parse(e.data).ids;
    sources.innerHTML = ids.map((id) =>
      `<a href="/photo/${id}"><img class="cite" src="/thumb/${id}" alt="photo ${id}"></a>`
    ).join("");
  });
  source.onmessage = (e) => {
    buffer += JSON.parse(e.data).delta;
    renderAnswer(buffer);
  };
  source.addEventListener("done", () => source.close());
  source.onerror = () => source.close();
  return false;
}
```

- [ ] **Step 6: Wire the routes** in `web/app.py`. Add imports near the other `chat`/`search` imports:

```python
import json
from fastapi.responses import StreamingResponse
from chat.context import build_context
from chat.retrieve import retrieve
from inference.prompts import chat_messages
```

Add both routes inside `create_app`, next to the other `@app.get` handlers:

```python
    @app.get("/chat", response_class=HTMLResponse)
    def chat_page(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(request, "chat.html", {})

    @app.get("/chat/stream")
    def chat_stream(q: str = "") -> StreamingResponse:
        ctx = context()
        embedder, _ = ctx.settings.build_embedder()
        client, _ = ctx.settings.build_inference_client()
        model = ctx.settings.planner_model or "fake"
        ids = retrieve(ctx.conn, embedder, ctx.settings.owner_id, q, k=30)
        block = build_context(ctx.conn, ids)
        messages = chat_messages(q, block)

        def events():
            yield f"event: sources\ndata: {json.dumps({'ids': ids})}\n\n"
            for delta in client.stream(model, messages):
                yield f"data: {json.dumps({'delta': delta})}\n\n"
            yield "event: done\ndata: {}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")
```

- [ ] **Step 7: Run tests, verify pass**

Run: `uv run pytest tests/test_web_chat.py -v`
Expected: PASS.

- [ ] **Step 8: Add CSS** for the citation thumbnails — append to `web/static/app.css`:

```css
.chat-sources { display: flex; flex-wrap: wrap; gap: 4px; margin: 12px 0; }
.chat-answer { line-height: 1.6; }
img.cite { height: 40px; width: 40px; object-fit: cover; border-radius: 4px;
  vertical-align: middle; margin: 0 2px; }
```

- [ ] **Step 9: Run the full suite**

Run: `uv run pytest`
Expected: PASS — no regressions.

---

## Self-Review

**Spec coverage (§10 steps 1–5):**
1. Question → retrieval — Task 3 (`retrieve`, fusion; planner deferred per Task 1 doc edit). ✓
2. Top 30 photos — Task 3 `k=30`. ✓
3. Compact per-photo context with EXIF facts — Task 4 `build_context`. ✓
4. Model answers, cites `[photo:ID]` — Task 5 prompt + Task 6 route. ✓
5. SSE streaming, inline thumbnail citations — Task 2 `stream()` + Task 6 endpoint/JS. ✓
- Grounding / say-so when empty (§10): prompt (Task 5) + `test_stream_says_so_when_nothing_retrieved` (Task 6). ✓
- Sources always shown as thumbnails (§10): `event: sources` + `chat-sources` render (Task 6). ✓
- `/chat` nav + page (§13): Task 6 steps 3–4. ✓
- Code layout (§14): Task 1 step 4. ✓

**Placeholder scan:** every code step carries real code; no TBD/TODO. ✓

**Type consistency:** `retrieve(conn, embedder, owner_id, question, k)` and `build_context(conn, photo_ids)` used identically in Tasks 3/4 and Task 6. `stream(model, messages, *, timeout)` matches across protocol, real client, fake, and route. SSE event names (`sources`/default/`done`) match between Task 6 route, JS, and tests. ✓
