# Agentic-Hybrid Chat — Plan & Decision Record

> Snapshot (this plan supersedes the intent-routing parts of `10-chat-agentic-rerank.md`).
> Spec home is `docs/design.md` §10; that section is rewritten **when this is
> implemented**, in the same turn as the code. Until then design.md still
> describes the *current* deterministic-branch behaviour — do not pre-edit it.

**Goal:** Kill the "guess the intent in Python" layer in chat. The agent loop's
tool-calls decide *what to do* for every question; deterministic code only
supplies **tools that return truth** and one narrow **pre-lease cache**. This
removes the class of bug where a new phrasing needs a new regex branch (the
`"show me all my memories"` → `no memory matches "all"` bug that started this).

## The decision: hybrid, not either pole

We compared two poles and chose the middle. The whole tension is **one weak 3B
planner** (`qwen2.5:3b`, `config.py`) being asked to route reliably.

**A — current hybrid (intent regexes route; agent loop only for photo search)**
- ✅ counts exact & instant (no model, no lease, works mid-ingest, §8.1)
- ✅ reliable on a weak model; deterministic memory cards need no stored state
- ❌ intent detection is brittle regex — every new phrasing is a new branch
- ❌ duplicated logic across `auto_answer` / `_auto_facts` / `_auto_memory`
- ❌ "agentic" is a lie — the interesting questions are hardcoded

**B — fully agentic (tool-calls drive everything, no intent branches)**
- ✅ one uniform path — the "all"-class bug disappears; new capability = new tool
- ✅ genuinely agentic; handles compound questions the branches can't; less code
- ❌ correctness rides on the 3B model picking the right tool + valid JSON
- ❌ counts now take the model + CHAT lease → slower, can say "busy" mid-ingest
- ❌ agent's card choice must be persisted per message (new state)

**Key finding that tips it toward B's brain:** the inference client already
supports **strict structured output** (`json_schema` + `"strict": true`,
`inference/client.py`), but the agent loop's `_turn()` and `plan()` don't use it —
they prompt for JSON and hand-parse `{…}`. Constraining tool-calls with that
existing lever **eliminates the malformed-output failure mode entirely** and
collapses B's residual risk to "wrong tool / lazy answer", both promptable. That
is *most* of why the branches felt necessary.

**Chosen shape — direct-DB for the deterministic, agent for the semantic:**
- **Direct-DB layer (no planner, pre-lease):** every question the DB can answer
  *unambiguously* is answered straight from SQLite — total & subject counts,
  memory count, memory show/list, periods. Fast, exact, works mid-ingest (§8.1),
  and — critically — **never routed through the weak planner**, so a phrasing like
  `"all"` cannot break it there.
- **Agent brain (from B):** only genuinely *semantic* questions (photo search,
  "what lens in Italy", relational counts) reach the loop. There the intent
  regexes are gone; the agent owns intent via **schema-constrained** tool-calls so
  it physically cannot emit garbage or a non-existent tool.
- **Hands (keep):** tools stay pure SQL/FTS truth. The model decides *whether* to
  count; it never *does* the counting — so a count is exact whenever a tool runs.

### The anti-"all" rule (why this never regresses again)

The `"all"` bug was NOT "determinism is wrong" — it was a matcher that **fired and
answered wrong** (treated the quantifier `all` as a search term) *and had no test
for that phrasing*. Two guarantees fix the class, not the instance:

1. **Conservative matchers — fire-confident or decline.** A direct-DB matcher
   answers only when it is sure; anything ambiguous or unrecognised returns `None`
   and falls through to the agent. A confident-wrong answer is a bug; a decline is
   always safe (the agent picks it up). Quantifiers (`all/every/each/any/my/the`)
   are **never** search terms — they mean "everything / the largest", not a literal
   FTS query.
2. **An exhaustive routing test matrix** (below) — every class × many phrasings,
   including the adversarial quantifiers, typos, and the negatives that must NOT be
   caught. This is the regression net that `"all"` slipped through because it
   didn't exist. New phrasing that misbehaves = add a row, not a new branch.

Net: A's exactness + instant §8.1 counts for everything deterministic; B's honest
single agent path for the semantic tail; and the brittleness that caused `"all"`
is contained by "decline when unsure" + a real test matrix.

## What changes

### Delete — DONE (only the agent-seeding shims; the matchers were KEPT)
- [x] `auto_answer` (superseded by `direct_answer`), `_auto_facts`, `_auto_memory`.
- [x] the agent's fact/show tools in `_tool` + `_AGENT_SYSTEM`
      (count/memories/periods/find_memory) and the four fact-builder helpers
      (`_count_fact`/`_memories_fact`/`_periods_fact`/`_memory_fact`); the semantic
      tail keeps only search/similar/nearby, so `agent_retrieve` → `list[int]`.
- [x] `chat_stream`: the `is_aggregate_question` context branch and the post-loop
      memory-card block. **Kept** the memory-card re-derivation from the question.
- KEPT (load-bearing, NOT deleted as an earlier draft of this plan wrongly said):
  the whole direct-DB matcher set — `is_memory_show`, `is_aggregate_question` (used
  by `is_memory_show`), `memories_for_show`, `_memory_terms`, `_plain_total`, and the
  regexes `_COUNT_*`/`_PHOTO_WORD`/`_TOTAL_FILLER`/`_MEMORY_*`/`_ALL_MEMORIES`. They
  live in `direct_answer` now, hardened and pinned by the routing matrix.

### Keep & harden (the direct-DB layer) — DONE
- [x] `auto_answer` → `direct_answer`: keeps ALL DB-answerable classes (total,
      subject count, memory count, memory show/list, periods), NOT narrowed, made
      **conservative** (fire-confident or return `None`), covered by the matrix.
- [x] Hardened matchers: quantifiers never leak as terms (`_count_subject` strips
      photo-words; `_ALL_MEMORIES` → show-all); a counted-noun read (`_COUNTED`)
      separates "how many photos … memory" (photos) from "how many memories"; any
      class it cannot answer confidently returns `None` → agent.

### Add (net-new — the real work)
- [ ] **Schema-constrained tool-calls (keystone).** `_turn()` calls
      `complete(..., json_schema=TOOL_CALL_SCHEMA)`. Schema (strict):
      `action ∈ {expand, answer}`; when `expand`: `tool ∈
      {search, similar, nearby}` + optional `query` / `photo_id`; when `answer`:
      `photo_ids: [int]`. The model can no longer emit malformed JSON or a fake
      tool. (The agent's tail is *semantic* only, so its toolset shrinks to the
      candidate-pullers; count/periods/memories are owned by the direct-DB layer.)
- [ ] **Prompt guards** in `_AGENT_SYSTEM`: 2–3 few-shot tool-call examples, and a
      firm rule — *never state a number you did not get from a fact* — so the
      "8 photos" confabulation stays dead for any relational count the direct-DB
      layer declines into the agent.

### Simplified out (were needed only by the fully-agentic pole)
- **No persistence column.** Because memory-show is a *direct-DB* class,
  deterministic from the question, cards re-derive on reload exactly as today
  (`_memory_cards_html(question)`). No `chat_messages.memories` column, no
  `SCHEMA_VERSION` bump, no `add_message` change.
- **No `agent_retrieve` 3-tuple / find_memory tool.** The agent never chooses
  cards (memory-show never reaches it), so `agent_retrieve` keeps its
  `(photo_ids, facts)` shape and needs no `find_memory` tool. Memory "all" mode
  lives in the direct-DB layer's `find_memories`, fired by the matcher — the fix
  already written in `chat/agent.py`, minus the deleted intent regexes.

### Also (docs hygiene, requested)
- [ ] `docs/design.md` §10 + its mermaid: rewrite to this flow (remove the
      `auto_answer`/deterministic-routing narrative; keep the narrow total cache).
- [ ] Kill stale **present-tense "Gemma 4"** claims across §4/§7/§8/§10 and
      `_CHAT_SYSTEM` — the running planner/caption models are **Qwen2.5** (`config.py`);
      keep §4's honest "Gemma 4 is the intended future target" note.

## Routing test matrix (the regression net)

One parametrised test asserts, per phrasing, **which layer answers** and **the
result shape**. `direct` = answered by `direct_answer` with the model untouched
(`fake.calls == []`); `agent` = declined by `direct_answer` (returns `None`) and
handled by the loop. Adversarial rows (quantifiers, typos, negatives) are the
point — this is what `"all"` needed and lacked.

### Direct-DB — total count
| Phrasing | Route | Expect |
|---|---|---|
| "how many photos do I have?" | direct | total |
| "how many images in my libray?" (typo) | direct | total |
| "total number of pics" | direct | total |
| "how many photos do I have in total?" | direct | total |
| "how many photos altogether" | direct | total |

### Direct-DB — subject count
| "how many photos with dogs" | direct | count FTS "dogs" |
| "number of beach photos" | direct | count FTS "beach" |
| "how many photos of my car" | direct | count FTS "car" |

### Direct-DB — memory count
| "how many memories do I have?" | direct | len(memories) |
| "number of memories" | direct | len(memories) |

### Direct-DB — memory show / list  *(the "all" battleground)*
| "show me all my memories" | direct | show ALL |
| "show me my memories" | direct | show ALL |
| "list my memories" | direct | show ALL |
| "list all memories" | direct | show ALL |
| "show me every memory" | direct | show ALL |
| "show me a memory" | direct | largest |
| "show me my borjomi memory" | direct | specific: borjomi |
| "open my antarctica memory" (no match) | direct | honest none (NOT planner) |

### Direct-DB — periods
| "how many months have photos" | direct | months |
| "how many years" | direct | years |
| "what years do my photos span" | direct | years |

### Must DECLINE → agent (never caught by direct-DB)
| "photos of dogs on a beach" | agent | semantic search |
| "show me sunset pictures" | agent | semantic — "show me X", X≠memories |
| "what lens did I use most in Italy?" | agent | EXIF over candidates |
| "how many photos similar to this dog?" | agent | relational — decline the count |
| "how many photos are in my Borjomi memory?" | agent | per-memory count (no direct tool yet) |
| "do I have any photos of cats?" | agent | existence / semantic |

**Invariants the matrix enforces**
- A memory-show matcher requires a `memor` token — `"show me sunset pictures"`
  must NOT be read as memory-show.
- Quantifier-only remainder ⇒ ALL/largest, never an FTS term (`"all"` guard).
- A relational count (`similar to`, `like this`) ⇒ decline, never the bogus total.
- A specific memory miss ⇒ honest "no such memory" from the DB, not a planner trip.

## Risks & mitigations
- **Weak model picks the wrong tool.** Mitigate: strict schema (enum), few-shot,
  the number-only-from-a-fact rule, and a bounded loop that already force-answers
  on the last round. Residual risk accepted; the worst/most-common case (the
  whole-library total) is still hard-cached.
- **Latency/power on Jetson.** Non-total counts now pay the CHAT lease + a model
  load + a few decodes they used to skip, and can preempt ingest (§8.1). Accepted;
  it is the price of one honest path.
- **Non-total counts can reply "busy" mid-ingest** (previously instant). Accepted.

## Out of scope (future)
- Folding the off-topic gate and the planner into the agent loop as one call
  (fewer round-trips) — a separate simplification, not needed for the hybrid.
- Bumping the planner to a larger model to further de-risk tool choice.
