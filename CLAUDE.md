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

## Code

- Python, `uv` for dependency management. Run things with `uv run`.
- Tests live in `tests/`. Every new module needs tests.
- The layered boundaries in `§5` are real: `app` serves reads, `worker` owns
  writes, storage and inference are reached only through their interfaces.
