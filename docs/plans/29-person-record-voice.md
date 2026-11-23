# PLAN 29 — TECH-32: record a voice from a person

- **Issue:** TECH-32 (Bug 1) — https://linear.app/sadensmol/issue/TECH-32
- **Repo:** `ivms777` only.
- **Design:** `docs/design.md` §13 (`/people/{id}`, `/voices`), §13.1 (People);
  `docs/ui.md` `/people/{id}`, `/voices`.
- **Builds on:** plan 28 D6 (the person page links to no other leaf) and D8 (New voice
  picks a person).

This plan is a snapshot of one unit of work. The design is the current state.

## 1. Goal

The person page said "No voices yet." and gave no way to record a voice: recording lives
on the Voices tab, and nothing on the person page led there. The person page now has a
**Record a voice** link that opens the Voices tab with New voice set to that person.

## 2. Decisions

| Id | Decision | Why |
|---|---|---|
| D1 | The person page links to the Voices grid, `/voices?person=<id>`; no recorder on the person page and no voice created there. | The voice page already holds the recorder. A grid is not a leaf, so §13.1 rule 1 holds; a recorder or a create-then-open on the person page would put a leaf inside a leaf or make a leaf-to-leaf move. |
| D2 | On a profile with no cloning entry, the person page shows "Voice copy is not available on this device." instead of the link. | `/voices` offers no New voice on such a profile; a link to a missing form is a dead end. |
| D3 | `/voices?person=<id>` selects the owner's person in New voice and fills the name with the person's name when the person owns no voice. A missing, foreign or non-numeric value gives the default form, never an error. | The query only preselects; a stale link must still open the grid. A person with a voice likely wants a new name for the next voice. |

## 3. Steps

### S1 — Docs first

`docs/design.md` §13 `/people/{id}` and `/voices`, §13.1 People paragraph; `docs/ui.md`
`/people/{id}` and `/voices`; this snapshot. No mermaid block changes: no new flow,
service or tool.

### S2 — Person page link

`web/people_api.py:_leaf` passes whether voice copy is available
(`models.catalog.cloning_languages(profile)`). `web/templates/person.html`: the Voices
section ends with the plain link **Record a voice** to `/voices?person=<id>`, or the
reason text when voice copy is not available. No other link to a leaf.

### S3 — Voices grid preselect

`web/voices_api.py:voices_page` reads an optional text query `person`. For the owner's
person, `web/templates/voices.html` marks that option `selected` and prefills the name
input with the person's name when the person owns no voice. Any other value gives
today's page.

## 4. Acceptance criteria

- AC1 — On a profile with voice copy, `/people/{id}` shows **Record a voice** with
  `href="/voices?person=<id>"`, with or without voices.
- AC2 — On a profile with no voice copy, the person page shows "Voice copy is not
  available on this device." and no link to `/voices?person=`.
- AC3 — `GET /voices?person=<id>` for the owner's person answers 200 and selects that
  person; "New person with this name" is not selected.
- AC4 — The name field holds the person's name when the person owns no voice, and is
  empty when they own one.
- AC5 — A missing id, another owner's person or a non-numeric value answers 200 with
  "New person with this name" selected and an empty name.
- AC6 — `POST /voices` from that form gives the voice to that person, and the person
  page lists it.
- AC7 — The person page holds no link to a voice page, a face group page or a photo.
- AC8 — The design and `docs/ui.md` describe the link and the `person` query.

## 5. Tests

`tests/test_web_person_record_voice.py` (AC1–AC7). Regression: `tests/test_web_people.py`,
`tests/test_web_person_leaf_rules.py`, `tests/test_web_voices.py`,
`tests/test_voices_person_select.py`.
