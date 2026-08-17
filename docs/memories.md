# Memories — the composition pipeline

The exact agentic composition pipeline, its persistence, and rebuild mechanics.
Design §11 carries the governing principle (the LLM decides every membership), the
two kinds of memory, and the canonical flow diagram; this file is the steps. Section
references point into `docs/design.md`.

Memories composes the library into named, described *memories* — "A day at Borjomi",
"Family night in Ontario" — not sets of look-alike photos. **The governing
principle:** the LLM decides every membership; heuristics only make the problem
tractable (they hand the agent a small working set, never decide contents). The two
kinds — **event** (time/place-bounded) and **theme** (a thread across time) — overlap
by design: one photo may sit in one event and several themes, which `group_photos`
(many-to-many) supports with no schema change.

## The composition pipeline (all decisions are the agent's)

1. **Pool (cheap, no decisions).** Group the owner's **processed** photos (caption +
   embedding present) into coarse *sessions* by time and ~50 km region purely to
   bound context size — this is not the memory boundary, just a tractable batch the
   agent can read at once. Only processed photos participate, so composition is run
   **after** captioning/embedding.
2. **Compose events (agent, per session).** For each session the agent reads compact
   per-photo summaries (date, place, caption, tags) and **decides the carve** — one
   memory, or several chapters, or skip — pulling extra context on demand via bounded
   tools (similar photos, facet lookups, photos nearby in time, same-subject
   retrieval) so it can reach *across* sessions when an event spans a pool boundary.
   It returns each memory as `{title, description, photo_ids[]}`, grounded only in the
   data.
3. **Discover themes (agent + RAG).** Separately, an agent proposes recurring threads
   — a subject that appears often (the dog), a place, an occasion, a season — and for
   each **retrieves** candidate photos (semantic + tag + facet) then curates the set.
   This is retrieval-augmented: the theme is the query, the agent judges membership.
   Themes deliberately pull photos already in events → overlap.
4. **Reconcile (agent).** A final pass dedupes near-identical memories, merges
   fragments the pooling split, and writes final titles/covers. It merges *memories*,
   never collapses the overlap between an event and a theme.
5. **Persist.** Each memory → `groups(kind='memory')` + `group_photos`; a photo may
   land in many. `params` records how it was built (kind, seed, model). The swap is
   atomic (`albums/memory_store.py::replace_memories`) and, in the same transaction,
   re-indexes each memory's name/description into `memory_fts` — the index chat's
   memory-show searches (`find_memories`, §10). Rebuilding memories rebuilds that
   index in lockstep, so a memory is findable in chat the moment it exists and a
   dropped memory disappears from search with it.

The event-composition agent's `similar` expand tool (`albums/compose.py`) goes
through the one retriever core — `similar_photos` is a core wrapper (§9.2) — so how
candidates reach the agent changed, but the agent still decides every membership
itself, unchanged.

## Cost and rebuild

**Cost is bounded (§9.1's batch, offline exception to the one-call rule):**
per-session and per-theme agent loops are capped at a few rounds and tool calls; the
whole build is signature-guarded and run only on demand.

**Rebuilding.** Manual only ("Rebuild memories"), on a **background thread**, one
build at a time per process, signature-guarded (owner photo count + newest
`updated_at`, stored in each memory's `params`) so opening the tab never silently
re-runs the agent; the tab flags **stale** when the signature moves. Because only
processed photos participate, **rebuild after captioning/embedding completes**.

> The earlier heuristic seed→curate (one time/place run → one memory, minus
> outliers) is superseded by the above: it let a distance rule, not the model, decide
> contents, could not produce overlapping or thematic memories, and fragmented one
> outing across nearby spots. The coarse time/region seeder is kept **only** as the
> step-1 pooling that bounds context — never as the decider.

The `groups`/`group_photos` tables — reserved and unused in the earlier design — now
back Memories. They remain available for a future "save this album" action on the
live organizers.
