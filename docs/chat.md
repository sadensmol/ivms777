# Ask-your-library chat — the pipeline in detail

The exact chat retrieval path: the direct-DB matchers, the router, the tools, the
grounding block, and the two preference toggles. Design §10 carries the agentic-RAG
narrative and the canonical flow diagram; this file is the mechanics. Section
references point into `docs/design.md`.

Retrieval is **agentic RAG**, not a one-shot dump. The old path fused the top-30
neighbours and force-fed them to the model, which then invented matches ("a dog on
the dashboard" with no dog) because a semantic KNN always returns *k* neighbours —
there was no "nothing matches". The path is now precise, cheapest layer first, and
the deterministic questions never touch the model at all.

## The path

0. **Direct-DB layer — answer everything structural before touching a model.** Every
   question the DB can answer *unambiguously* — the whole-library total, a subject
   FTS count, the memory count, the month/year span, and **showing a memory or all
   memories** — is answered straight from SQLite by `direct_answer`, with **no model
   call at all**: it replies instantly even while ingest is captioning inside the
   `models` service (§8.1), and the phrasing **never reaches the weak planner**. Each
   matcher is **conservative** — it fires only when confident and otherwise returns
   `None` to fall through to the agent, so an unrecognised phrasing degrades to the
   agent, never to a confidently-wrong answer. This kills the "all" class of bug (a
   matcher that fired *and* answered wrong): quantifiers (`all`/`every`/`my`) are
   never search terms, and the whole boundary is pinned by a routing matrix
   (`tests/test_chat_routing.py`) of every class × many adversarial phrasings (typos,
   negatives like "show me sunset pictures" that must **not** read as a memory-show,
   relational counts that must decline). A relational count ("how many *similar to
   this dog*?") the DB cannot compute is declined here and answered by the agent —
   never the meaningless library total. A memory-show turn also renders the matched
   memory (or, for a plural/all request, **every** memory) as the same Organize
   memory card — re-derived from the question (deterministic, so history reload needs
   no stored state), each cover linked with `ctx=chat-memory:<key>` (§13.1) so
   opening one pages **within that memory** and "close" returns to the conversation.
1. **Route — one agentic decision (plan 17).** A single schema-constrained call to
   the planner model (`chat/agent.py::route`, temperature 0) classifies the message
   into ONE tool, using the user's own photos/memories ONLY when the message is about
   them: `search_library` (they want photos OF something — a person, animal, object,
   place, scene), `search_memories` (a saved memory/trip/album by name or theme), or
   `none` (general knowledge, chit-chat — **anything else**). Any routing failure
   falls back to `none`.
2. **Run the tool (only for a library/memory question).**
   - `search_library` → **SigLIP image↔text** (`search_photos`, §9.2): the query text
     embedded by SigLIP's text tower, matched against each photo's **image** vector.
     This is what SigLIP is built for. Captions / `caption_vec` are **not** used here
     (plan 17): a lossy text↔text hop that ranked scene-neighbours (an SUV photo whose
     caption shared "car/sign") next to the real match. Returns the top-k photos.
   - `search_memories` → memories whose name/theme matches (`memories_for_show`).
   - `none` → no retrieval.
3. **Answer — grounded or general, streamed at temperature 0.** For a tool result the
   model is given ONLY those photos/memories (per photo: id, date, caption, top tags,
   EXIF facts — camera, lens, ISO, aperture, shutter, focal length, coords — ~60
   tokens) and answers **strictly from them**, citing `[photo:ID]`; the grounding
   prompt forbids inventing a subject not written in a caption and says to report none
   when the results do not match (honest-empty). For `none`, the model answers from
   general knowledge with no library mention. Counts / memory-show / periods never
   reach here — `direct_answer` (step 0) served them straight from SQLite with no
   model. The answer **streams** over SSE, each `[photo:ID]` rendered as a clickable
   thumbnail.

## The two preference toggles (`chat_prefs`, per owner, every session)

Chat exposes two checkboxes whose state persists in a `chat_prefs` row and is read on
every turn. **Their defaults reproduce the exact pipeline above** — nothing changes
unless the user flips a box.

- **Direct answers** (default **on**). ON is the direct-DB layer (step 0). OFF
  **skips the whole direct-DB step** — counts, memory show/list, AND periods — and
  every turn instead runs the **fully-agentic loop**: a bounded, schema-constrained
  tool loop where the model calls REAL tools — `count_photos`, `list_memories`,
  `count_periods`, and `search` — to gather facts and candidate photos, then a final
  grounded answer streams from those facts (via `agentic_answer_messages`). A count is
  a real number the model asked for, never one inferred from the retrieved handful.
  The loop's **first instruction is on-topic-or-not**: a message about the world, not
  about the user's own photos/memories, must answer with **no tool call at all**. Of
  the four tools only `search` runs SigLIP, and on the Jetson SigLIP cannot be
  co-resident with gemma (design §8.1) — one stray search evicts gemma and pays a
  llama-server respawn mid-turn — so `search` is reserved for "show me my photos of X".
  Memory-show in OFF mode is answered as prose (the model reads `list_memories`) — it
  does **not** render the Organize memory card (the card is a direct-DB affordance).
- **Guardrails** (default **off**). OFF is the general-assistant behaviour (a `none`
  message is answered from the model's own knowledge). ON reuses the router's `none`
  verdict as an **on-topic gate**: a message routed `none` is refused with a fixed
  redirect (`GUARDRAIL_REFUSAL`) — **no model generation, no library search** — while
  a `search_library` / `search_memories` message answers normally. This is the
  plan-17 off-topic gate, but opt-in. **App-specific questions are NEVER off-topic**:
  a deterministic `is_app_topic` check (counts, memories, albums, uploads, tags,
  cameras, the library itself) overrides the weak router, so a mislabelled `none` on
  an app question is answered, never refused.

The two are independent: direct-OFF + guardrails-ON = the fully-agentic loop confined
to library topics.

## Grounding, degradation, and history

The answer to a library/memory question is grounded ONLY on that tool's results'
captions/tags/EXIF, and the grounding prompt forbids inventing a subject not written
in a caption — so when the search returns nothing that matches, the model says it has
none rather than fabricating one (honest-empty). Captions are model-generated and
imperfect, so the chat view always shows its sources — the thumbnails are the
evidence. Shown and stored sources are the ids the answer actually **cites**, never
the raw candidate set.

**Degrade, never crash.** Any route, search, embed, or generation failure falls back
to a plain answer (a failed route defaults to `none`; a failed `search_library`
degrades to the core's `candidates()` fusion, §9.2), so chat always answers and no
ranking is re-implemented outside the core (plan 12 single-pipeline invariant).

**History is persisted.** Each answered turn (question, full answer, cited photo ids)
is written to `chat_messages` under the owner's current `chat_sessions` row (§6). On
load, `/chat` renders the current session's turns server-side as static history, so
switching away and back — or restarting the app — keeps the conversation. A **New
session** button opens a fresh, empty session; older sessions stay in the database.
Persistence is the visible transcript only: each question is answered independently
against freshly retrieved photos, not against prior turns — there is no multi-turn
model memory.

Chat and indexing share one inference service. The chat route calls the planner model
directly, which is small and stays loaded, so a question during indexing does not
evict the captioner.
