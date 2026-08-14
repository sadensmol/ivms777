# Photo Library Organizer — Plan 07: Memories (agentic albums)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the Organize tab's "By similarity" slot into **Memories** — named, described albums that read like memories ("Family night in Ontario", *"A family having fun, 22 Nov 1999"*) instead of blobs of look-alike photos. A background job seeds candidate clusters cheaply, an agent curates and narrates each one from the data already on hand, and the results are stored so the tab renders instantly.

**Architecture:** Three steps behind one entry point `build_memories(...)`:
1. **Seed** — pure, no model: capture-time runs (> 6 h gap), split/merged by GPS cell and SigLIP similarity.
2. **Curate** — an application-level agent loop over the existing `InferenceClient` (structured output). For each candidate the planner model reads photo summaries and may request a few rounds of extra context (similar photos, facet lookups, photos near in time) before returning a keep/skip decision, a title, a description, and any outlier photos to drop.
3. **Persist** — memories are written to `groups(kind='memory')` + `group_photos`; the `memories` organizer reads them back. A library signature stored on each memory guards against needless rebuilds.

The agent loop is deliberate and matches design §9.1: batch, offline, per-step latency paid once at build time, never on a page load.

**Tech Stack:** Python 3.12, SQLite, NumPy, FastAPI, Jinja2, HTMX. Reuses `InferenceClient` (`inference/client.py`), `FakeInferenceClient` (`inference/fakes.py`), `similar_photos` (`search/semantic.py`), and the SigLIP vectors already in `photo_vec`.

**Spec:** `docs/design.md` — §11 (Organize / Memories), §6 (`groups.description`), §9.1 (batch agent loop), §13 (`/organize` UI), §14 (`albums/memories*.py`), §16 (phase 5).

**Builds on:** plan 03 (embeddings + `similar_photos`), plan 05 (captions on `photos.caption`), plan 06 (the planner model + `settings.build_inference_client()` / planner model name, and `search/facets.py`). Memories consumes captions, tags, EXIF facets, and embeddings — all present after phase 4.

**Supersedes:** the "phase 5 = event/cluster/duplicate groups, `/groups`" line in plan 03's *Following plans* table. Plans are snapshots; that footer is not rewritten — this plan is the current phase-5 unit of work.

**Covers:** the whole Memories organizer. **Deferred (YAGNI for v1):** merging photos *across* candidate clusters (the agent may drop outliers from a candidate — a split — but candidates are not merged); a live progress bar finer than "building… / N memories".

## Global Constraints

- Python 3.12. Dependencies via `uv` with a committed `uv.lock`.
- **Never run `git commit`/`git add`.** The user commits. Every task ends at a checkpoint.
- Every user-scoped query filters on `owner_id` (constant `settings.owner_id`).
- Tests must not hit the network or load a real model. The agent is exercised with `FakeInferenceClient`; embeddings use `FakeEmbedder`.
- The full fast suite passes at the end of every task: `uv run pytest -q`, and `uv run ruff check .` is clean.
- Single-owner assumption (§3.2) holds: one build runs at a time per process; no cross-owner concurrency.

---

### Task 1: `groups.description` and the memory repository

Give the `groups`/`group_photos` tables — reserved and unused until now — a read/write layer for memories, and add the `description` column §6 now specifies.

**Files:**
- Modify: `db/schema.sql`
- Modify: `db/connection.py`
- Create: `albums/memory_store.py`
- Create: `tests/test_memory_store.py`

**Interfaces:**
- Produces:
  - `albums.memory_store.Memory` — dataclass `(title, description, photo_ids, signature)`; `photo_ids` in cover-first rank order.
  - `albums.memory_store.replace_memories(conn, owner_id, memories: list[Memory]) -> None` — atomically deletes the owner's `kind='memory'` groups and writes the new set (rank = position, cover = rank 0).
  - `albums.memory_store.read_memories(conn, owner_id) -> list[Memory]` — stored memories, largest first.
  - `albums.memory_store.current_signature(conn, owner_id) -> str` — `"{photo_count}:{max_updated_at}"` over the owner's captioned photos.
  - `albums.memory_store.stored_signature(conn, owner_id) -> str | None` — the signature the stored memories were built from, or `None` if there are none.

- [ ] **Step 1: Write the failing test**

Create `tests/test_memory_store.py`:

```python
from albums.memory_store import (
    Memory, current_signature, read_memories, replace_memories, stored_signature,
)
from tests.factories import add_photo


def _photo(conn, pid, **cols):
    return add_photo(conn, photo_id=pid, content_hash=f"h{pid}", thumb_key=f"{pid}.jpg", **cols)


def test_replace_then_read_round_trips_in_rank_order(conn):
    _photo(conn, 1); _photo(conn, 2); _photo(conn, 3)
    replace_memories(conn, owner_id=1, memories=[
        Memory(title="Family night in Ontario",
               description="A family having fun, 22 Nov 1999.",
               photo_ids=[2, 3, 1], signature="3:2026-01-01T00:00:00"),
    ])
    stored = read_memories(conn, owner_id=1)
    assert len(stored) == 1
    assert stored[0].title == "Family night in Ontario"
    assert stored[0].photo_ids == [2, 3, 1]  # cover (rank 0) first


def test_replace_clears_the_previous_memories(conn):
    _photo(conn, 1)
    sig = "1:2026-01-01T00:00:00"
    replace_memories(conn, 1, [Memory("Old", "old.", [1], sig)])
    replace_memories(conn, 1, [Memory("New", "new.", [1], sig)])
    titles = [m.title for m in read_memories(conn, 1)]
    assert titles == ["New"]


def test_memories_are_owner_scoped(conn):
    add_photo(conn, photo_id=1, owner_id=1, content_hash="a", thumb_key="a.jpg")
    add_photo(conn, photo_id=2, owner_id=2, content_hash="b", thumb_key="b.jpg")
    replace_memories(conn, 1, [Memory("Mine", "d.", [1], "s")])
    replace_memories(conn, 2, [Memory("Theirs", "d.", [2], "s")])
    assert [m.title for m in read_memories(conn, 1)] == ["Mine"]


def test_signature_changes_when_a_photo_is_added(conn):
    _photo(conn, 1, caption="a", updated_at="2026-01-01T00:00:00")
    before = current_signature(conn, 1)
    _photo(conn, 2, caption="b", updated_at="2026-02-02T00:00:00")
    assert current_signature(conn, 1) != before


def test_stored_signature_reflects_the_built_set(conn):
    _photo(conn, 1, caption="a")
    assert stored_signature(conn, 1) is None
    replace_memories(conn, 1, [Memory("M", "d.", [1], "sig-42")])
    assert stored_signature(conn, 1) == "sig-42"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_memory_store.py -q`
Expected: FAIL — `ModuleNotFoundError: albums.memory_store` (and, once imported, a missing `description` column).

- [ ] **Step 3: Add the `description` column and bump the schema**

In `db/schema.sql`, add the column to `groups`:

```sql
CREATE TABLE IF NOT EXISTS groups (
  id          INTEGER PRIMARY KEY,
  owner_id    INTEGER NOT NULL,
  kind        TEXT NOT NULL,
  name        TEXT NOT NULL,
  description TEXT,
  params      TEXT,
  status      TEXT NOT NULL,
  created_at  TEXT NOT NULL
);
```

In `db/connection.py` bump `SCHEMA_VERSION = 3`. `migrate()` re-runs `schema.sql` (idempotent `CREATE IF NOT EXISTS`), which covers fresh databases. For an existing v2 database the table already exists, so add a guarded column add at the top of `migrate()` before the `executescript`, applied only on the v2→v3 hop:

```python
    if version == 2:
        # groups predates the memory columns; add them in place.
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(groups)")}
        if "description" not in cols:
            conn.execute("ALTER TABLE groups ADD COLUMN description TEXT")
```

Keep the existing `SchemaTooOldError` path for version 0 untouched.

- [ ] **Step 4: Write the repository**

Create `albums/memory_store.py`. `replace_memories` runs inside one transaction so a build never leaves the tab half-populated. Store the signature and seed metadata in `params` (JSON); `status='accepted'` (memories are not suggestions).

```python
import json
import sqlite3
from dataclasses import dataclass

NOW_SQL = "strftime('%Y-%m-%dT%H:%M:%SZ','now')"


@dataclass(frozen=True)
class Memory:
    title: str
    description: str
    photo_ids: list[int]   # cover first
    signature: str


def current_signature(conn: sqlite3.Connection, owner_id: int) -> str:
    row = conn.execute(
        "SELECT count(*) AS n, COALESCE(max(updated_at), '') AS m"
        " FROM photos WHERE owner_id = ? AND caption IS NOT NULL",
        (owner_id,),
    ).fetchone()
    return f"{row['n']}:{row['m']}"


def stored_signature(conn: sqlite3.Connection, owner_id: int) -> str | None:
    row = conn.execute(
        "SELECT params FROM groups WHERE owner_id = ? AND kind = 'memory'"
        " ORDER BY id LIMIT 1",
        (owner_id,),
    ).fetchone()
    if row is None:
        return None
    return json.loads(row["params"] or "{}").get("signature")


def replace_memories(conn: sqlite3.Connection, owner_id: int, memories: list[Memory]) -> None:
    conn.execute("BEGIN")
    try:
        conn.execute(
            "DELETE FROM groups WHERE owner_id = ? AND kind = 'memory'", (owner_id,)
        )
        for memory in memories:
            cursor = conn.execute(
                "INSERT INTO groups(owner_id, kind, name, description, params, status, created_at)"
                f" VALUES (?, 'memory', ?, ?, ?, 'accepted', {NOW_SQL})",
                (owner_id, memory.title, memory.description,
                 json.dumps({"signature": memory.signature})),
            )
            group_id = int(cursor.lastrowid)
            conn.executemany(
                "INSERT INTO group_photos(group_id, photo_id, rank) VALUES (?, ?, ?)",
                [(group_id, pid, rank) for rank, pid in enumerate(memory.photo_ids)],
            )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def read_memories(conn: sqlite3.Connection, owner_id: int) -> list[Memory]:
    rows = conn.execute(
        "SELECT g.id, g.name, g.description, g.params,"
        " (SELECT count(*) FROM group_photos gp WHERE gp.group_id = g.id) AS n"
        " FROM groups g WHERE g.owner_id = ? AND g.kind = 'memory'"
        " ORDER BY n DESC, g.id",
        (owner_id,),
    ).fetchall()
    memories: list[Memory] = []
    for row in rows:
        photo_ids = [
            r["photo_id"] for r in conn.execute(
                "SELECT photo_id FROM group_photos WHERE group_id = ? ORDER BY rank",
                (row["id"],),
            )
        ]
        signature = json.loads(row["params"] or "{}").get("signature", "")
        memories.append(Memory(row["name"], row["description"], photo_ids, signature))
    return memories
```

- [ ] **Step 5: Run the repository tests**

Run: `uv run pytest tests/test_memory_store.py -q`
Expected: PASS.

- [ ] **Step 6: Run the whole suite**

Run: `uv run pytest -q`
Expected: PASS (the schema bump does not disturb existing tables).

- [ ] **Step 7: Checkpoint** — report the schema bump, the repository contract, and the signature format. Stop. Do not commit.

---

### Task 2: Candidate seeding

Form candidate clusters with no model, so the expensive agent only ever sees coherent starting points. Capture-time runs (a gap > 6 h starts a new run — the same 6 h clustering the old "events" view used, now internal to Memories only) are split when one run spans two GPS cells, so a "morning downtown, afternoon at the lake" day becomes two candidates.

**Files:**
- Create: `albums/seeds.py`
- Create: `tests/test_memory_seeds.py`

**Interfaces:**
- Consumes: `photos` rows (`id, shot_at, gps_lat, gps_lon`); the `~1 km` GPS cell rule already in `albums/by_place.py`. The 6 h gap constant lives here in `seeds.py` (the "By date" organizer no longer carries an events grain).
- Produces:
  - `albums.seeds.Candidate` — dataclass `(photo_ids: list[int])`.
  - `albums.seeds.seed_candidates(conn, owner_id, *, min_size=3) -> list[Candidate]` — time-and-place-contiguous runs of at least `min_size` captioned photos, newest first.

- [ ] **Step 1: Write the failing test**

Create `tests/test_memory_seeds.py`:

```python
from albums.seeds import seed_candidates
from tests.factories import add_photo


def _p(conn, pid, shot_at, **cols):
    return add_photo(conn, photo_id=pid, content_hash=f"h{pid}", thumb_key=f"{pid}.jpg",
                     caption="x", shot_at=shot_at, **cols)


def test_a_contiguous_run_becomes_one_candidate(conn):
    for pid in range(1, 5):
        _p(conn, pid, f"2025-07-12T1{pid}:00:00")
    cands = seed_candidates(conn, owner_id=1, min_size=3)
    assert len(cands) == 1
    assert sorted(cands[0].photo_ids) == [1, 2, 3, 4]


def test_a_six_hour_gap_splits_candidates(conn):
    for pid in (1, 2, 3):
        _p(conn, pid, f"2025-07-12T09:0{pid}:00")
    for pid in (4, 5, 6):
        _p(conn, pid, f"2025-07-13T20:0{pid}:00")
    assert len(seed_candidates(conn, owner_id=1, min_size=3)) == 2


def test_one_event_across_two_places_splits_by_gps(conn):
    # same afternoon, two ~1 km-apart locations -> two candidates
    for pid in (1, 2, 3):
        _p(conn, pid, f"2025-07-12T12:0{pid}:00", gps_lat=51.50, gps_lon=-0.12)
    for pid in (4, 5, 6):
        _p(conn, pid, f"2025-07-12T12:1{pid}:00", gps_lat=48.85, gps_lon=2.35)
    assert len(seed_candidates(conn, owner_id=1, min_size=3)) == 2


def test_runs_below_min_size_are_dropped(conn):
    _p(conn, 1, "2025-07-12T09:00:00")
    _p(conn, 2, "2025-07-12T09:05:00")  # only two -> not a memory
    assert seed_candidates(conn, owner_id=1, min_size=3) == []


def test_uncaptioned_photos_are_ignored(conn):
    for pid in (1, 2, 3):
        _p(conn, pid, f"2025-07-12T09:0{pid}:00")
    add_photo(conn, photo_id=4, content_hash="h4", thumb_key="4.jpg",
              shot_at="2025-07-12T09:04:00")  # no caption
    cands = seed_candidates(conn, owner_id=1, min_size=3)
    assert cands[0].photo_ids == [1, 2, 3]
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_memory_seeds.py -q`
Expected: FAIL — `ModuleNotFoundError: albums.seeds`.

- [ ] **Step 3: Write the seeder**

Create `albums/seeds.py`. Walk the owner's captioned photos in time order; start a new candidate on a > 6 h gap *or* a GPS-cell change (photos with no GPS never force a split — they inherit the current run). Reuse the cell rule from `by_place.py`.

```python
import sqlite3
from dataclasses import dataclass
from datetime import datetime

from albums.by_place import _cell

# A gap longer than this starts a new run. Owned here, not by the date organizer:
# time-gap clustering is a Memories seeding step, not a user-facing date view.
EVENT_GAP_HOURS = 6.0


@dataclass(frozen=True)
class Candidate:
    photo_ids: list[int]


def _parse(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None


def seed_candidates(conn: sqlite3.Connection, owner_id: int, *, min_size: int = 3) -> list[Candidate]:
    rows = conn.execute(
        "SELECT id, shot_at, gps_lat, gps_lon FROM photos"
        " WHERE owner_id = ? AND thumb_key IS NOT NULL"
        " AND shot_at IS NOT NULL AND caption IS NOT NULL"
        " ORDER BY shot_at",
        (owner_id,),
    ).fetchall()

    runs: list[list[int]] = []
    last_time: datetime | None = None
    last_cell = None
    for row in rows:
        when = _parse(row["shot_at"])
        if when is None:
            continue
        cell = _cell(row["gps_lat"], row["gps_lon"]) if row["gps_lat"] is not None else None
        gap = last_time is None or (when - last_time).total_seconds() > EVENT_GAP_HOURS * 3600
        moved = cell is not None and last_cell is not None and cell != last_cell
        if gap or moved:
            runs.append([])
        runs[-1].append(row["id"])
        last_time = when
        if cell is not None:
            last_cell = cell

    candidates = [Candidate(ids) for ids in runs if len(ids) >= min_size]
    candidates.sort(key=lambda c: c.photo_ids[0], reverse=True)  # newest first
    return candidates
```

- [ ] **Step 4: Run the seed tests**

Run: `uv run pytest tests/test_memory_seeds.py -q`
Expected: PASS.

- [ ] **Step 5: Run the whole suite**

Run: `uv run pytest -q`
Expected: PASS.

- [ ] **Step 6: Checkpoint** — report the split rules (time gap, GPS move) and the `min_size` floor. Stop. Do not commit.

---

### Task 3: The agentic composer

For one candidate, run a bounded agent loop: summarise its photos, let the planner model ask for a few rounds of extra context, then take its keep/skip decision, title, description, and dropped outliers. Every model turn is one `InferenceClient.complete(...)` with a strict JSON schema, so `FakeInferenceClient` makes the whole loop deterministic in tests.

**Files:**
- Create: `albums/compose.py`
- Create: `tests/test_memory_compose.py`

**Interfaces:**
- Consumes: `InferenceClient`; `similar_photos` (`search/semantic.py`); a facet lookup over `photo_facets`; the photos' `caption`, tags, and facets.
- Produces:
  - `albums.compose.compose_memory(conn, client, model, owner_id, candidate, *, max_rounds=3) -> Memory | None` — a `Memory` (task 1) when the agent keeps the candidate, `None` when it skips it or the model output is unusable.

**Agent protocol (JSON schema, `strict`):** each turn the model returns one of:

```json
{"action": "expand", "tool": "similar|facets|nearby", "photo_id": 123}
{"action": "answer", "keep": true, "title": "Family night in Ontario",
 "description": "A family having fun, 22 Nov 1999.", "drop_photo_ids": [456]}
```

- `expand` → the app runs the named read-only tool, appends a short result block to the transcript, and calls the model again. Capped at `max_rounds` expansions; after the cap the model is asked to answer.
- `answer` with `keep=false` → the candidate is skipped (`compose_memory` returns `None`).
- `answer` with `keep=true` → build a `Memory`; drop any `drop_photo_ids`; if fewer than 2 photos remain, skip. Signature is stamped by the caller (task 4), so `compose_memory` sets `signature=""` and task 4 fills it — *or* accept `signature` as a parameter. Pass it in: `compose_memory(..., signature)`.

Grounding rule baked into the prompt: the title and description must use only facts present in the summaries (dates, place, what the captions say). No invented names or events. If the candidate is incoherent (unrelated photos that merely happened close in time), `keep=false`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_memory_compose.py`. Queue the fake's JSON turns to drive each path:

```python
import json

from albums.compose import compose_memory
from albums.seeds import Candidate
from albums.memory_store import Memory
from inference.fakes import FakeInferenceClient
from tests.factories import add_photo


def _photo(conn, pid, **cols):
    return add_photo(conn, photo_id=pid, content_hash=f"h{pid}", thumb_key=f"{pid}.jpg",
                     caption=cols.pop("caption", "a photo"), **cols)


def test_a_kept_candidate_becomes_a_named_described_memory(conn):
    for pid in (1, 2, 3):
        _photo(conn, pid, shot_at="1999-11-22T20:00:00")
    client = FakeInferenceClient([json.dumps({
        "action": "answer", "keep": True,
        "title": "Family night in Ontario",
        "description": "A family having fun, 22 Nov 1999.",
        "drop_photo_ids": [],
    })])
    memory = compose_memory(conn, client, "planner", owner_id=1,
                            candidate=Candidate([1, 2, 3]), signature="sig")
    assert isinstance(memory, Memory)
    assert memory.title == "Family night in Ontario"
    assert memory.photo_ids == [1, 2, 3]
    assert memory.signature == "sig"


def test_a_skipped_candidate_returns_none(conn):
    for pid in (1, 2, 3):
        _photo(conn, pid)
    client = FakeInferenceClient([json.dumps(
        {"action": "answer", "keep": False, "title": "", "description": "", "drop_photo_ids": []}
    )])
    assert compose_memory(conn, client, "planner", 1, Candidate([1, 2, 3]), signature="s") is None


def test_dropped_outliers_are_removed_from_the_memory(conn):
    for pid in (1, 2, 3, 4):
        _photo(conn, pid)
    client = FakeInferenceClient([json.dumps({
        "action": "answer", "keep": True, "title": "Beach day",
        "description": "An afternoon by the water.", "drop_photo_ids": [4],
    })])
    memory = compose_memory(conn, client, "planner", 1, Candidate([1, 2, 3, 4]), signature="s")
    assert 4 not in memory.photo_ids
    assert memory.photo_ids == [1, 2, 3]


def test_the_agent_can_request_context_then_answer(conn):
    for pid in (1, 2, 3):
        _photo(conn, pid)
    client = FakeInferenceClient([
        json.dumps({"action": "expand", "tool": "similar", "photo_id": 1}),
        json.dumps({"action": "answer", "keep": True, "title": "T",
                    "description": "d.", "drop_photo_ids": []}),
    ])
    memory = compose_memory(conn, client, "planner", 1, Candidate([1, 2, 3]),
                            signature="s", max_rounds=3)
    assert memory is not None
    assert len(client.calls) == 2  # one expand round, then the answer


def test_dropping_below_two_photos_skips_the_memory(conn):
    for pid in (1, 2):
        _photo(conn, pid)
    client = FakeInferenceClient([json.dumps({
        "action": "answer", "keep": True, "title": "T", "description": "d.",
        "drop_photo_ids": [2],
    })])
    assert compose_memory(conn, client, "planner", 1, Candidate([1, 2]), signature="s") is None


def test_unusable_model_output_is_skipped_not_raised(conn):
    for pid in (1, 2, 3):
        _photo(conn, pid)
    client = FakeInferenceClient(["not json at all"])
    assert compose_memory(conn, client, "planner", 1, Candidate([1, 2, 3]), signature="s") is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_memory_compose.py -q`
Expected: FAIL — `ModuleNotFoundError: albums.compose`.

- [ ] **Step 3: Write the composer**

Create `albums/compose.py`. Keep the tool set small and read-only; every tool returns a short text block appended to the running transcript. Parse defensively — any bad/again-invalid JSON, or a schema the model never resolves, yields `None` (the candidate is simply left out; §11 says a lone/incoherent set is dropped, not forced).

Sketch:

```python
import json
import sqlite3

from albums.memory_store import Memory
from albums.seeds import Candidate
from inference.client import ChatMessage, InferenceClient
from search.semantic import similar_photos

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"enum": ["expand", "answer"]},
        "tool": {"enum": ["similar", "facets", "nearby"]},
        "photo_id": {"type": "integer"},
        "keep": {"type": "boolean"},
        "title": {"type": "string"},
        "description": {"type": "string"},
        "drop_photo_ids": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["action"],
}


def compose_memory(conn, client: InferenceClient, model: str, owner_id: int,
                   candidate: Candidate, *, signature: str, max_rounds: int = 3) -> Memory | None:
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
            messages.append({"role": "user", "content": _run_tool(conn, owner_id, turn)})
            continue
        if turn.get("action") != "answer" or not turn.get("keep"):
            return None
        kept = [pid for pid in candidate.photo_ids if pid not in set(turn.get("drop_photo_ids", []))]
        if len(kept) < 2 or not turn.get("title") or not turn.get("description"):
            return None
        return Memory(turn["title"].strip(), turn["description"].strip(), kept, signature)
    return None
```

Helpers to implement in the same module:
- `_summarise(conn, owner_id, photo_ids)` — one compact line per photo: id, date, place (GPS cell or "no location"), caption, top tags. This is the RAG context; keep it to ~40 tokens per photo.
- `_run_tool(conn, owner_id, turn)` — dispatch `similar` → `similar_photos(...)` summarised; `facets` → a `photo_facets` lookup for the photo; `nearby` → photos within a few hours of the given one. Always returns a short text block; unknown tool → `"(no result)"`.
- `_complete(client, model, messages, force_answer)` — append a nudge to answer when `force_answer`, call `client.complete(model, messages, json_schema=ANSWER_SCHEMA)`, `json.loads` it, and return the dict; return `None` on any `JSONDecodeError`/`KeyError`.
- `_SYSTEM_PROMPT` — the grounding rules above: compose one memory from these photos, use only stated facts, no invented people or events, skip if incoherent.

- [ ] **Step 4: Run the composer tests**

Run: `uv run pytest tests/test_memory_compose.py -q`
Expected: PASS.

- [ ] **Step 5: Run the whole suite**

Run: `uv run pytest -q`
Expected: PASS.

- [ ] **Step 6: Checkpoint** — report the agent JSON protocol, the tool set, and the "skip on unusable output" rule. Stop. Do not commit.

---

### Task 4: `build_memories` and the `memories` organizer

Tie seeding and composing together behind one signature-guarded entry point, and expose the stored memories through the `Organizer` protocol so the tab reads them like any other organizer.

**Files:**
- Create: `albums/memories_build.py`
- Create: `albums/memories.py`
- Modify: `albums/registry.py`
- Create: `tests/test_memories_build.py`
- Modify: `tests/test_albums.py` (registry now also exposes `memories`)

**Interfaces:**
- Produces:
  - `albums.memories_build.build_memories(conn, client, model, owner_id, *, force=False) -> int` — returns the number of memories written. When the current signature already matches the stored one and `force` is false, it returns the existing count and does no model work.
  - `albums.memories.MemoriesOrganizer` — `name="memories"`, `label="Memories"`; `organize(...)` reads stored memories and maps each to an `Album` (`key=f"memory-{i}"`, `cover_id=photo_ids[0]`, `meta={"kind": "memory"}`). `grain` is ignored.

- [ ] **Step 1: Write the failing test**

Create `tests/test_memories_build.py`:

```python
import json

from albums.memories import MemoriesOrganizer
from albums.memories_build import build_memories
from albums.memory_store import read_memories, current_signature
from inference.fakes import FakeInferenceClient
from tests.factories import add_photo


def _run(conn, pids, shot):
    for pid in pids:
        add_photo(conn, photo_id=pid, content_hash=f"h{pid}", thumb_key=f"{pid}.jpg",
                  caption="x", shot_at=shot.format(pid))


def _keep(title):
    return json.dumps({"action": "answer", "keep": True, "title": title,
                       "description": f"{title}.", "drop_photo_ids": []})


def test_build_writes_one_memory_per_kept_candidate(conn):
    _run(conn, (1, 2, 3), "2025-07-12T1{}:00:00")
    n = build_memories(conn, FakeInferenceClient([_keep("Beach day")]), "planner", owner_id=1)
    assert n == 1
    assert read_memories(conn, 1)[0].title == "Beach day"


def test_build_is_skipped_when_the_signature_matches(conn):
    _run(conn, (1, 2, 3), "2025-07-12T1{}:00:00")
    build_memories(conn, FakeInferenceClient([_keep("A")]), "planner", 1)
    # No queued responses: a second build must NOT call the model.
    empty = FakeInferenceClient([])
    n = build_memories(conn, empty, "planner", 1)
    assert n == 1
    assert empty.calls == []


def test_force_rebuilds_even_when_the_signature_matches(conn):
    _run(conn, (1, 2, 3), "2025-07-12T1{}:00:00")
    build_memories(conn, FakeInferenceClient([_keep("A")]), "planner", 1)
    build_memories(conn, FakeInferenceClient([_keep("B")]), "planner", 1, force=True)
    assert read_memories(conn, 1)[0].title == "B"


def test_stored_memory_shows_up_as_an_album(conn):
    _run(conn, (1, 2, 3), "2025-07-12T1{}:00:00")
    build_memories(conn, FakeInferenceClient([_keep("Beach day")]), "planner", 1)
    albums = MemoriesOrganizer().organize(conn, owner_id=1)
    assert albums[0].title == "Beach day"
    assert albums[0].cover_id in (1, 2, 3)
    assert albums[0].meta["kind"] == "memory"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_memories_build.py -q`
Expected: FAIL — `ModuleNotFoundError: albums.memories_build`.

- [ ] **Step 3: Write `build_memories`**

Create `albums/memories_build.py`:

```python
import sqlite3

from albums.compose import compose_memory
from albums.memory_store import (
    Memory, current_signature, replace_memories, stored_signature,
)
from albums.seeds import seed_candidates
from inference.client import InferenceClient


def build_memories(
    conn: sqlite3.Connection, client: InferenceClient, model: str, owner_id: int,
    *, force: bool = False,
) -> int:
    signature = current_signature(conn, owner_id)
    if not force and stored_signature(conn, owner_id) == signature:
        return len(seed_candidates(conn, owner_id)) and _stored_count(conn, owner_id) or \
            _stored_count(conn, owner_id)
    memories: list[Memory] = []
    for candidate in seed_candidates(conn, owner_id):
        memory = compose_memory(conn, client, model, owner_id, candidate, signature=signature)
        if memory is not None:
            memories.append(memory)
    replace_memories(conn, owner_id, memories)
    return len(memories)


def _stored_count(conn: sqlite3.Connection, owner_id: int) -> int:
    return conn.execute(
        "SELECT count(*) AS n FROM groups WHERE owner_id = ? AND kind = 'memory'",
        (owner_id,),
    ).fetchone()["n"]
```

Simplify the skip-return to just `return _stored_count(conn, owner_id)` — the seed call in the sketch above is redundant; drop it.

- [ ] **Step 4: Write the organizer**

Create `albums/memories.py`:

```python
import sqlite3

from albums.base import Album
from albums.memory_store import read_memories


class MemoriesOrganizer:
    name = "memories"
    label = "Memories"

    def organize(self, conn: sqlite3.Connection, owner_id: int, grain: str | None = None) -> list[Album]:
        albums: list[Album] = []
        for index, memory in enumerate(read_memories(conn, owner_id)):
            albums.append(
                Album(
                    key=f"memory-{index}",
                    title=memory.title,
                    description=memory.description,
                    photo_ids=memory.photo_ids,
                    cover_id=memory.photo_ids[0],
                    meta={"kind": "memory"},
                )
            )
        return albums
```

- [ ] **Step 5: Register it (second in the dropdown, after date)**

In `albums/registry.py`, add `MemoriesOrganizer()` right after `ByDateOrganizer()` so the dropdown order is date → memories → camera → place. Update the registry assertion in `tests/test_albums.py`:

```python
    assert set(ORGANIZERS) == {"date", "memories", "camera", "place"}
```

- [ ] **Step 6: Run the tests**

Run: `uv run pytest tests/test_memories_build.py tests/test_albums.py -q`
Expected: PASS.

- [ ] **Step 7: Run the whole suite**

Run: `uv run pytest -q`
Expected: PASS.

- [ ] **Step 8: Checkpoint** — report the signature guard, the build count contract, and the dropdown position. Stop. Do not commit.

---

### Task 5: `/organize?by=memories` and the rebuild control

Serve stored memories on the Organize tab and add a "Rebuild memories" button. Because a build makes many model calls, it runs in a background thread; the page shows whether memories are stale (current signature ≠ stored) and whether a build is in flight.

**Files:**
- Modify: `web/app.py`
- Modify: `web/templates/organize.html`
- Modify: `web/static/app.css`
- Create: `tests/test_web_memories.py`

**Interfaces:**
- Produces:
  - `GET /organize?by=memories` — renders stored memories (empty state prompts a first build).
  - `POST /organize/memories/rebuild` — starts a background build (guarded by an in-process flag), then redirects back to `GET /organize?by=memories`.
  - Template context: `memories_stale` (bool), `memories_building` (bool), shown only when the memories organizer is active.

- [ ] **Step 1: Write the failing test**

Create `tests/test_web_memories.py`. Drive the build with a fake client by injecting it into the app context (add a test seam: `settings.use_fake_inference` or an overridable `context.inference_client`, mirroring `build_embedder`'s fake-by-default switch from plan 03). Run the build synchronously in tests by setting the app's thread launcher to run inline (expose `app.state.run_build` that tests can call directly, or assert on the flag + a joined thread).

```python
import json
import pytest
from fastapi.testclient import TestClient

from tests.factories import add_photo
from web.app import create_app


@pytest.fixture
def client(settings):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    app = create_app(settings)
    conn = app.state.context.conn
    for pid in (1, 2, 3):
        add_photo(conn, photo_id=pid, content_hash=f"h{pid}", thumb_key=f"{pid}.jpg",
                  caption="x", shot_at=f"2025-07-12T1{pid}:00:00")
    with TestClient(app) as test_client:
        yield test_client


def test_memories_is_in_the_dropdown(client):
    assert "Memories" in client.get("/organize?by=memories").text


def test_empty_state_prompts_a_first_build(client):
    body = client.get("/organize?by=memories").text
    assert "Rebuild memories" in body  # the build control is present


def test_rebuild_builds_and_shows_the_memory(client):
    # settings/context provide a FakeInferenceClient queued to keep one memory.
    client.app.state.context.queue_inference([json.dumps(
        {"action": "answer", "keep": True, "title": "Beach day",
         "description": "An afternoon by the water.", "drop_photo_ids": []}
    )])
    client.post("/organize/memories/rebuild", follow_redirects=False)
    client.app.state.await_build()  # test seam: join the build thread
    assert "Beach day" in client.get("/organize?by=memories").text
```

Adapt the exact seam names to what plan 06 exposes for the inference client; the key requirements are (a) tests never hit a network, and (b) the build is joinable in tests.

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_web_memories.py -q`
Expected: FAIL — no `memories` route wiring / no rebuild endpoint.

- [ ] **Step 3: Wire the rebuild endpoint and status**

In `web/app.py`:
- Register `MemoriesOrganizer` is already done (task 4); the existing `/organize` route now renders memories via the registry with no change beyond status context.
- When the active organizer is `memories`, compute `memories_stale = current_signature(...) != stored_signature(...)` and pass `memories_building` from an in-process flag.
- Add `POST /organize/memories/rebuild`: if a build is not already running, set the flag and launch a daemon `threading.Thread` that opens/uses the owner connection, calls `build_memories(conn, client, planner_model, owner_id, force=True)`, and clears the flag in a `finally`. Redirect to `/organize?by=memories`. Expose an `await_build()` seam on `app.state` for tests (join the thread if present).

Keep the single-owner assumption explicit in a comment: one build at a time per process (§3.2).

- [ ] **Step 4: Update the template and styles**

In `web/templates/organize.html`, when the memories organizer is active, render a small toolbar above the album grid:

```html
{% if active == "memories" %}
  <form method="post" action="/organize/memories/rebuild" class="memories-bar">
    <button type="submit" {% if memories_building %}disabled{% endif %}>Rebuild memories</button>
    {% if memories_building %}<span class="muted">building…</span>
    {% elif memories_stale %}<span class="muted">library changed since the last build</span>{% endif %}
  </form>
{% endif %}
```

Add minimal styles for `.memories-bar` in `app.css`. The empty state (no memories yet) already falls through to the existing "No albums… try another type" block; the rebuild button above it is enough of a prompt.

- [ ] **Step 5: Run the memories web tests**

Run: `uv run pytest tests/test_web_memories.py -q`
Expected: PASS.

- [ ] **Step 6: Run the whole suite and lint**

Run: `uv run pytest -q && uv run ruff check .`
Expected: PASS, clean.

- [ ] **Step 7: Checkpoint** — report the rebuild flow, the stale/building indicators, and the single-build guard. Stop. Do not commit.

---

### Task 6: Verify against the real model and update the docs

The fast suite never calls a real LLM. Confirm the memories read like memories once, by hand.

- [ ] **Step 1: Build memories against the real planner**

With a real inference backend configured (per plan 06), on a library that has captions:

```bash
IVMS777_DATA_DIR=~/.ivms777 uv run uvicorn web.app:app_factory --factory --port 8100
```

Open `/organize?by=memories`, click **Rebuild memories**, and wait for the build to finish.

- [ ] **Step 2: Judge the output**

Confirm, and record in the checkpoint:
1. Titles name a real thing (a place, an occasion, a day) — not "Similar set 3" and not an invented person or event.
2. Descriptions state only facts visible in the data (dates, place, what the photos show).
3. Incoherent time-and-place runs were skipped, not forced into a memory.
4. Re-opening the tab is instant (no model call), and a second **Rebuild** with no library change is a no-op (signature guard).

Note the wall-clock for the build and the rough per-candidate model-call count.

- [ ] **Step 3: Update the docs**

- `README.md`: note that Memories is built on demand from the Organize tab and needs captions (phase 3) and the planner (phase 4) present first.
- `docs/design.md` §16: mark phase 5's Memories organizer as delivered by plan 07. Leave §11 as the source of truth for behaviour — confirm the implementation matches it (agentic build, persisted, signature-guarded rebuild); fix either the code or §11 if they disagree.

- [ ] **Step 4: Checkpoint** — report the manual judgement, the build timing, and any §11 discrepancies found and fixed. Stop. Do not commit.

---

## What plan 07 delivers

The Organize tab gains **Memories**: open it, hit Rebuild, and the library comes back as named, described albums — "Family night in Ontario", *"A family having fun, 22 Nov 1999"* — instead of the old color-alike clusters. Under the hood a cheap seeder cuts the library into time-and-place-contiguous candidates, an agent grounded in captions, tags, and EXIF curates and narrates each one over a few bounded rounds, and the results are stored so the tab is instant on every later visit. A library signature keeps rebuilds honest: nothing re-runs the model unless the photos actually changed or the user forces it.

**Not in this plan:** merging photos across candidate clusters, a fine-grained progress bar, and any auto-rebuild on upload (the user rebuilds when they want fresh memories). Those are future work if they earn their keep.
