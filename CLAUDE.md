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

5. **Close/Esc goes UP to the grid, once** — `history.back()` (restores the grid's
   scroll + state via bfcache), with the computed grid URL as the `href` deep-link
   fallback. It must never walk back through visited photos (rule 4 guarantees
   none are in history).

6. **Prev/next move only *within* the current layer's order**, carrying `ctx`
   forward — never leaking into a sibling memory/album or the wider library.

This layering *is* the "never lose the user's place" rule applied to structure:
one action up always returns to the grid with its state and scroll intact.

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

## Code

- Python, `uv` for dependency management. Run things with `uv run`.
- Tests live in `tests/`. Every new module needs tests.
- The layered boundaries in `§5` are real: `app` serves reads, `worker` owns
  writes, storage and inference are reached only through their interfaces.
