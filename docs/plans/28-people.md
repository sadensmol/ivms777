# PLAN 28 — TECH-29: people own voices and faces

- **Issue:** TECH-29 — https://linear.app/sadensmol/issue/TECH-29
- **Repo:** `ivms777` only. Same PR as plan 27.
- **Supersedes:** plan 27 decisions U1 (a person IS a `saved_voices` row) and D4 (keep
  faces on the Voices page). Plan 27 stays a snapshot.
- **Design:** `docs/design.md` §5, §6, §11.1, §13, §13.1; `docs/data-model.md`
  (`people`, `saved_voices.person_id`, `face_groups.person_id`, schema 16);
  `docs/ui.md` `/people`, `/people/{id}`, `/faces/{id}`, `/voices`, `/voices/{id}`;
  `docs/memories.md` "Playing a memory".

This plan is a snapshot of one unit of work. The design is the current state.

## 1. Goal

A person is a record of its own. Saved voices and face groups belong to a person, never
the other way round. A **People** tab lists the owner's people and holds "Faces to
name"; a person page shows the person's voices, face groups and photos. The photo page
and the film brief read names from people.

## 2. Decisions

| Id | Decision | Why |
|---|---|---|
| U2 | A person is its own `people` row (one list per owner, a unique name). A saved voice belongs to one person; a face group belongs to one person. | The owner, on the plan 27 PR: "voices belong to people not vice versa" and "voices should be linked to some person!" |
| D5 | The migration gives every saved voice with no person a person with the voice's name and owner (or the owner's person of that name when one exists). A face group attached to a voice moves to that voice's person. | Every existing name, face assignment and narrator choice stays the same; nothing is guessed and nothing is deleted. Voice names are unique per owner, so no two new people collide. |
| D6 | `/people` is a grid; `/people/{id}` is a leaf one level under it. The person page shows voices, face groups and photos in place and links to no other leaf. `/faces/{id}` is a leaf of People. | §13.1: a leaf is one level under a grid, and a leaf never opens another leaf. |
| D7 | A person can be deleted only when they own no voice (one being deleted included). Their face groups then return to "Faces to name". The delete runs inline. | A person holds no file, so there is nothing for an outbox to drain. A voice never loses its person. |
| D8 | "New voice" names the voice and picks its person: an existing person, or by default a new person with the voice's name. A taken person name is refused with its reason. A voice can move to another person of the owner. | Keeps the one-step New voice flow, and never links a voice to a person by a silent name match. |

Rejected: one shared "Unknown" person for old voices (the owner must re-sort every
voice); voices with no person (the owner rejects that); merging voices by similar names
(a guess that can join two people); deleting a person's voices with the person (a hidden
voice delete); the person page as a photo grid with a new `ctx=person:<id>` (a larger
navigation change, kept as future work in `docs/roadmap.md`).

## 3. Data model (schema 16)

| Table | Change |
|---|---|
| `people` (new) | `id` AUTOINCREMENT (never reused), `owner_id`, `name` (1-80 characters, unique per owner, compared case-blind in code), `created_at`, `updated_at`; `UNIQUE(owner_id, name)` |
| `saved_voices` | new `person_id` → `people(id)` `ON DELETE RESTRICT`, added as a nullable column (SQLite); every write sets it, every read tolerates NULL |
| `face_groups` | `person_id` → `people(id)` `ON DELETE SET NULL`, no longer → `saved_voices` |

The migration is additive and idempotent, in the migrate transaction: people for voices
first (D5), then, when `face_groups` still links to `saved_voices` (`PRAGMA
foreign_key_list`), `faces` and `face_groups` are rebuilt with every row and id kept and
each group's `person_id` mapped from its voice to that voice's person. SQLite cannot
change a foreign key in place, and dropping `face_groups` with foreign keys on would set
every face's group to NULL. `PRAGMA foreign_key_check` runs at the end. A database with
no face tables migrates too.

## 4. Steps

Waves: (1) docs + data and migration, (2) People web + Voices web, (3) QA verify.

1. **Design first** (S11). `docs/design.md` §5 (app box: people pages), §6 (`people`,
   voice → person, group → person, the migration), §11.1 (the brief names come from
   people), §13 (People tab, `/people`, `/people/{id}`, "Faces to name" on People, the
   voice page names its person and moves it, no Faces tab on a voice), §13.1 (People is a
   grid with two leaves, both close to `/people`). Mirror in `docs/data-model.md`,
   `docs/ui.md`, `docs/ingest-pipeline.md`, `docs/memories.md`, `docs/code-layout.md`,
   `docs/roadmap.md`.
2. **Data and migration** (S12). `db/schema.sql`, `db/connection.py`
   (`SCHEMA_VERSION` 16, `_migrate_locked`). `ingest/people.py`: list, get, create,
   rename and delete a person (typed errors for an empty, too long or taken name; delete
   refused while the person owns a voice), the voices and photos of a person; attaching
   a group checks `people`; naming a group with a new name creates a person only;
   `people_in_photos` and the group queries join `people`. `feed/saved_voices.py`:
   `create_voice` takes a person or creates one with the voice's name; moving a voice to
   another person of the owner; `SavedVoice` carries `person_id` and `person_name`.
3. **People pages** (S13). The nav gains **People** (Upload → Library → Chat →
   Organize → People → Voices). `web/people_api.py`: `GET /people`, `POST /people`,
   `GET /people/{id}`, `POST /people/{id}/rename`, `POST /people/{id}/delete`.
   "Faces to name" moves from `/voices` to `/people`. `/faces/{id}` reads
   "People › …", its select lists people, and close returns to `/people`.
4. **Voices follow the person** (S14). `/voices` loses "Faces to name"; each card
   names its person. New voice picks a person (default: a new one with the voice's
   name). The voice page names its person as plain text and moves the voice
   (`POST /voices/{id}/person`); its Faces tab goes away.
5. **QA verify.** Move the tests that used a saved voice as a person to real people;
   run `uv run pytest -q`, `uv run ruff check .` and the one-model-process gate.

## 5. Acceptance criteria

AC16–AC24 of the run plan: schema 16 with `people` (AC16); the v15 migration (AC17);
the People grid with New person and "Faces to name" (AC18); the person page (AC19);
the face-group leaf under People (AC20); a new voice needs a person, voices move, no
Faces tab (AC21); person delete rules (AC22); photo and brief names from people
(AC23); the docs describe people as built (AC24).

## 6. Risks

- The `faces`/`face_groups` rebuild must keep every face's group: rebuild both tables,
  never drop `face_groups` alone, and check foreign keys at the end.
- `saved_voices.person_id` is nullable in the database; read paths must tolerate a
  voice with no person (rows written by raw SQL).
- Tests that used `create_voice(...).id` as a person id can pass by id coincidence on a
  fresh database; QA verify moves them to real people.
