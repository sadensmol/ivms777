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
   It returns each memory as `{title, description, photo_ids[]}`.

   **The place in a summary is a NAME, never coordinates** — `Borjomi, Georgia`, read
   from the `place_city`/`place_country` facets (§6.2); a photo whose place cannot be
   named reads "place unknown". Raw lat/long used to be passed straight through, and a
   small model can neither recognise nor hide it, so it produced titles like
   "Activities in a location on December 1, 2023" and descriptions ending "at
   coordinates 42.17, 42.93" (design §11: a place is a name a person recognises;
   coordinates live on `/photo` alone).

   **The summary also carries the day's spine, from EXIF** — a single `Facts —
   When: 2023-11-25, Saturday, morning · Where: Tbilisi, Georgia · Camera: …` line per
   cluster, from the facets stage (§6.2). The captions say what is in a frame; this
   says what the day *was*, and it is what turns "photos of a street" into "a Saturday
   morning in Tbilisi". **Technical tag dimensions are withheld** (`composition`,
   `palette`, `quality`): fed "sharp, top-down, pastel" the model wrote them into the
   prose ("a moment filled with sharp, joyful holiday cheer") and they crowded out the
   tags that carry meaning.

   **Voice — this is written for the person whose life it is.** The title is short and
   warm and names the place ("A winter day in Borjomi"); the description is 2-4
   sentences about the day as one whole — never a per-photo list ("another view
   shows…", "There were scenes of…").

   **Warmth never buys invention.** A 2B model told to be warm fills in the humans it
   expects: two photos of a birthday cake became "Friends gathered around the sweet
   treats". So the rule is hard — **only people a caption mentions may appear, described
   as the caption describes them; if no caption mentions a person, the memory has no
   people in it** (no friends, family, guests, "everyone", "we", "you"). Warmth comes
   from the real place, light, season, and occasion. Tags are hints, not words to
   quote: a `summer` tag on a November day loses to the date. The one thing the model
   may add from its own knowledge is a **short, well-known touch about a place the
   summaries already name** (its mountains, its old town) — the thing a bigger model
   did for free and a 2B one must be told to do. Examples in the prompt are shapes to
   follow, never words to copy: the example title "Christmas at home in Tbilisi" was
   pasted verbatim onto outdoor photos, so no example carries a setting any more.
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
