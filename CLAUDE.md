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

Whenever you build a drill-down, these all apply:

1. **Carry the context down.** The thing you open knows which parent it came from.
   That context travels explicitly (a `ctx`/query parameter in the URL), never
   guessed — the same content can be reached from several parents, so the parent
   has to be recorded, not inferred.

2. **Show every layer, top to bottom.** The detail view leads with the identity of
   *each* enclosing layer — album/memory title + description + `N / M` position —
   and only *then* the item's own data (caption, AI, EXIF). Opening the 3rd photo
   of a memory shows that memory just as fully as the 1st. Never show a leaf in a
   vacuum; a breadcrumb of its ancestry comes first.

3. **Move *within* a layer, never across.** Prev/next (and arrow keys) page within
   the current layer, in that layer's order — never leaking into a sibling
   memory/album or the wider library.

4. **Close/back goes *up one layer*, to the exact parent view — never sideways to
   a prior sibling.** Every in-layer navigation uses `replace`, so history stays
   `[parent, current]` and `history.back()` lands on the parent with its state and
   scroll intact; the close control's `href` is the computed parent URL as the
   deep-link fallback.

This layering *is* the "never lose the user's place" rule applied to structure:
the user always knows where they are (ancestry shown) and one action up always
returns them there.

## Code

- Python, `uv` for dependency management. Run things with `uv run`.
- Tests live in `tests/`. Every new module needs tests.
- The layered boundaries in `§5` are real: `app` serves reads, `worker` owns
  writes, storage and inference are reached only through their interfaces.
