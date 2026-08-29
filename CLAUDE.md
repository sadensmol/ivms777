# ivms777 — project instructions

## docs/design.md is the source of truth

`docs/design.md` describes the **current** state of the app, not a historical
proposal. It is the first thing to read and the first thing to change.

Before any change — a feature, a refactor, a schema edit, a new dependency:

1. **Read** the relevant section of `docs/design.md`.
2. **Decide** whether the change contradicts what is written there.
   - It contradicts the design → update `docs/design.md` first, in the same
     turn, before writing code.
   - It is already described there → implement it as described.
3. **Implement**, then verify the doc still matches reality.

Never leave `docs/design.md` describing behaviour the code does not have, and
never ship behaviour the doc does not mention. If the two disagree, the doc is
wrong until it is fixed.

Sections are numbered. Cite them when discussing changes (`§6.1`, `§3.2`) so it
is obvious what is being changed.

**The mermaid diagrams are part of that source of truth.** They are canonical —
one diagram per concern, no ASCII copy to drift:

- `§5` — system/container architecture (the boxes and the flows between them).
- `§9` — search & fusion flow, and the similar-photo scoring flow.
- `§10` — the agentic-RAG chat flow.
- `§11` — the memory-composition flow.

Any change that adds, removes, or re-wires a step, signal, tool, filter, or a flow
between them — a new retrieval stage, a changed rerank/floor, a new agent tool, a
moved responsibility, a new service or external writer — updates the owning mermaid
block in the **same turn** as the code, exactly like any other design edit. If a
diagram and the running system disagree, the diagram is wrong until it is fixed.

## Plans

`docs/plans/NN-*.md` are execution plans derived from the design. They are
snapshots — once written they describe a unit of work, not the current state.
When the design changes, the plan does not get rewritten retroactively; a new
plan supersedes it.

**Every plan lives in `docs/plans/`, no exceptions.** This holds even when a plan
is produced by an external skill or tool (`planning:make`, `system-design`,
`superpowers:writing-plans`, any other). Point the skill at `docs/plans/` and use
the `NN-*.md` naming. Never write a plan anywhere else — not a scratch dir, not
the repo root, not the skill's own default location.

## UI: never lose the user's place

Any state the user set is sacred. A search query, active filters, the chosen
sort, the scroll position, an expanded panel — all of it must **survive
navigation**. Drilling into a photo (or anything) and coming back returns the
user to *exactly* what they had: same filters, same query, same scroll.

- Prefer letting the browser restore state (`history.back()` / bfcache) over
  rebuilding a fresh page — it is the only thing that also restores scroll.
- Never hardcode a navigation that drops query state (e.g. a close button that
  links to a bare `/library` instead of returning to the filtered view).
- When a drill-down has its own navigation (prev/next), don't let it bury the
  origin in history — replace, don't stack, so "close" still returns to the list.

This is a hard rule for every feature, current and future — resetting the user's
context is a bug, not a detail.

## Navigation is layered — every view knows its ancestry

The whole UI is a hierarchy of **layers**, and this shape is a hard rule for
*every* feature, not just photos. A leaf is always reached *through* its
enclosing layers, and it must stay bound to them. Examples of the hierarchy:

```
Library (filters/search/sort) ─▶ photo
Organize ─▶ album | memory | group ─▶ photo
Chat ─▶ (a cited photo) ─▶ back to the conversation
```

The full, authoritative rules live in **`docs/design.md` §13.1** — read them
before touching any navigation. The non-negotiable core:

1. **A leaf is exactly ONE level below a grid.** A grid is a browsable list
   (library with filters/search/sort, an Organize album, a memory); a leaf is a
   detail view. There is **no leaf-inside-a-leaf nesting** — a "similar" photo is
   still a leaf one level under a grid (its grid is the library), not nested under
   the photo it is similar to.

2. **Carry the grid down in a `ctx` URL parameter**, never guessed —
   `ctx=library`(+filters), `ctx=album:…`, `ctx=similar:<id>` (grid = library; the
   origin photo is shown as clickable *context*, not the parent).

3. **Show the enclosing grid's identity first** — its title/description/`N / M`
   position — then the leaf's own data. Never show a leaf in a vacuum.

4. **HISTORY IS ALWAYS `[grid, leaf]` — depth two.** Grid→leaf is the ONE `push`.
   **Every** leaf-level move — prev/next, opening a similar photo, clicking the
   origin thumbnail — uses `location.replace`, NEVER a push. If you ever `push` on
   a leaf→leaf move you break this and close starts replaying visited photos.

5. **Close/Esc goes UP exactly ONE level** — for a normal leaf that is the grid:
   `history.back()` (restores the grid's scroll + state via bfcache), with the
   computed grid URL as the `href` deep-link fallback. It must never walk back
   through visited photos (rule 4 guarantees none are in history).
   A `ctx=similar:<id>` leaf has ONE extra level above it — the origin photo — so
   close goes there first (`location.replace`, no push), and the next close does the
   normal `history.back()` to the grid: similar → origin → grid.

6. **Prev/next move only *within* the current layer's order**, carrying `ctx`
   forward — never leaking into a sibling memory/album or the wider library.

7. **A leaf may hold levels of its OWN, and Esc unwinds those first.** A recording in
   progress, a film playing — each is a state the viewer is inside, and closing the page
   out from under it loses their place instead of going up one level. Esc: undo what is
   running, then close.

This layering *is* the "never lose the user's place" rule applied to structure:
one action up always returns to the grid with its state and scroll intact.

## One model process — NEVER load a model, LLM, or AI library twice (HARD RULE)

**Exactly ONE process loads and runs models.** Every model, LLM, embedder, and
heavy AI library (torch, transformers, bitsandbytes, SigLIP, the caption VLM, …)
is imported and held resident in a **single inference service** and nowhere else.
Every other process — `app`, `worker`, the CLI — is a **thin client** that reaches
it over an API (HTTP/IPC). They MUST NOT `import torch`/`transformers`, build an
embedder, or load any model in-process.

- **Never load the same model or AI library in two processes.** On the 8 GB Jetson
  a duplicated SigLIP + two torch CUDA contexts is what blows the memory budget.
  One process, one copy, coordinated in-process.
- A model that has no ready server (SigLIP image embeddings, the in-process caption
  VLM) is wrapped by **our** inference service and exposed as an endpoint — it is
  never loaded inline in `app`/`worker`.
- **Only the inference gateway talks to a model backend — including Ollama.**
  `app`/`worker` never call Ollama/vLLM directly; text generation and text
  embeddings are reached *through* the `models` service, which is the single client
  of every backend. One gateway in front of all inference.

This is non-negotiable for every feature, current and future. If a change would
import an AI library or load a model outside the inference service, it is wrong.

## Source folders are sacred — the app NEVER touches them

The folders on the user's disk are **sources**. The cloud app only ever holds an
uploaded *copy* (content-addressed) plus derived metadata. Nothing in the app —
upload, reprocess, and above all **"delete folder"** — may read, move, rename, or
delete anything on the user's filesystem. The only component that ever writes to
disk is `ivms777-sync` (stage 2), and only on an explicitly confirmed plan.

**"Delete a folder" means delete from the LIBRARY, never from disk.** It removes:
the folder's entry from the upload list, every photo uploaded from that folder
(matched by its internal folder name / `root_label`), and *all* of each deleted
photo's metadata — embeddings, caption vector, tags, facets, FTS, jobs, group
memberships, thumbnails, and the stored original. A photo whose identical bytes
still belong to another folder is kept (only that folder's source path is
dropped). The source folder on disk is untouched, always.

**Deletion runs through an outbox + worker**, never inline in the request: the
endpoint records the deletion intent and returns; a worker drains it and does the
cascade. So a delete is reliable across restarts and never blocks the UI.

## The running app belongs to the user — verify with TESTS, never by driving it

The user runs and tests the UI. Claude verifies with **`pytest` and offline checks
only**.

Never, against the running stack:

- `POST`/`GET` to `localhost:8000` or `localhost:9000` to trigger work
- start a Prepare / Regenerate / rebuild of any kind
- drive the browser at it
- poll it in a loop

Two reasons this is a hard rule, both already paid for: a build started here **cancels
or queues behind** whatever the user is testing, and a `make up` restart mid-run kills
the daemon thread so the work silently produces nothing — which then gets
mis-diagnosed as a code bug.

Read-only inspection of files and the SQLite database is fine. If something can only be
proven through the live UI, **say so and hand it to the user** rather than doing it.

## NEVER load a model of your own — the user's stack is always running

`make up` is up. Every model is already loaded in it, and that is the ONLY copy that may
exist. To try a prompt, measure a context, check a voice or verify a model runs, **point
at the running services**:

- `http://localhost:8080` — `llama-server` (gemma): `/v1/chat/completions`, `/props`,
  `/tokenize`. This is where a prompt is tried.
- `http://localhost:9000` — the models service: `/speak`, `/music`, `/text/*`,
  `/models`. Small, single calls only.

**Never start a model process yourself.** No `from_pretrained` in a `uv run python`
script, no second `llama-server`, no "throwaway" benchmark that loads weights. It cost
this project a 7.5 GB download and two 10 GB processes running beside the user's own
stack, for numbers the running services could have answered.

Still forbidden, for the reasons above: triggering WORK on the stack — a Prepare, a
Regenerate, a rebuild, a browser session, a polling loop. One short generation to check
a prompt is fine; a film build is not.

**If nothing is listening on those ports, ASK THE USER TO RUN `make up`.** Never run it
yourself, and never work around it by loading the model locally.

## Everything hot-reloads — NEVER ask the user to restart

`make up` runs every process with reload: `app` (`--reload`), `worker`
(`watchfiles`), and `models` (`--reload`, scoped to `modelsvc audio models embedding
captioning inference`). A code change is live within seconds.

So **never end a task with "restart `make up`"**. If a change genuinely does not take
effect without a restart, that is a **bug in the Makefile**, not an instruction for
the user — fix the reload configuration instead. This rule exists because a stale
`models` service was diagnosed three separate times as a broken feature: Russian
narration "did not work" for an hour while the code was already correct.

Corollary: **before diagnosing anything against the running stack, check the service
is not stale.** Compare its start time with the file mtimes, or look for a string you
just changed in its output. A stale process produces symptoms that look exactly like
real bugs.

## Verify a model RUNS before claiming it works

Installing a dependency, resolving a lockfile, and constructing a pipeline are not
evidence that a model works. **Generate something with it and inspect the output** —
duration, sample rate, byte count — before telling the user a feature is ready.

Stable Audio Open passed "the import works" and "the pipeline constructs" and still
could not produce a single second of audio: its SDE sampler drives `torchsde` into
non-terminating recursion. That cost a whole redesign and was discoverable in one
five-second call.

## A model's own metadata is DECLARED, never inferred at the call site

Anything the app needs to know about a model in order to USE it well — which voices a
TTS model offers and what each one sounds like, which languages it covers, what its
limits are — lives in `models/catalog.py` beside the model, as data. Never hardcoded in
whatever happens to consume it, and never guessed from a name at runtime.

**Installing or switching to a new voice model is not done until its speakers are
declared.** For each one: the exact name `POST /speak` takes, its gender, its register,
and a sentence saying what it sounds like. Produce that by RUNNING the model — generate
a line with every speaker, listen to or measure the output, and write down what is
actually there. A speaker name that has not been verified to load is a name that will
fail a film, because narration is mandatory (§11.1) and a bad name raises.

That table is what everything else casts from: the story pass is shown it and chooses
the narrator and any per-passage voice; `feed/voice.py` falls back to it when no model
runs; validation rejects anything not in it. Adding a voice model without it means the
new voices exist and nothing can reach them.

## No "off" / "none" options for models

Every model slot names a real model. Do not add a "no narration" / "no music" /
"disabled" entry to the catalog and do not offer one in the settings popup. A film
with no soundtrack is a broken film, not a configuration.

Where a choice genuinely follows from context, **make it automatically and hide it**:
`voice` is picked from the film's LANGUAGE and the memory's mood (§11.1), so it is in
`catalog.AUTOMATIC_SLOTS` and never appears in settings. Nobody should have to know
that Kokoro cannot speak Russian in order to get Russian narration.

## Long jobs need a real progress bar and a cancel

Any operation that can exceed a few seconds — preparing a film, rebuilding memories,
generating audio — must report a **named stage and a percentage**, and offer
**Cancel**. A static "Working…" is indistinguishable from a hung process, and a
button that appears to do nothing is indistinguishable from a broken one.

When an action is refused because something else holds the lock, do not print a dead
end ("wait, or cancel it elsewhere"): cancel the other and queue this one. And when a
regeneration starts, the page must **immediately** show progress and stop showing the
previous result, even though the old row still exists in the database.

## Code

- Python, `uv` for dependency management. Run things with `uv run`.
- Tests live in `tests/`. Every new module needs tests.
- The layered boundaries in `§5` are real: `app` serves reads, `worker` owns
  writes, storage and inference are reached only through their interfaces.
