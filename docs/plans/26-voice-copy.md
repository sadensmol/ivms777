# PLAN — IVMS777-21: voice copy with local models (implementation)

- **Issue:** IVMS777-21 — https://linear.app/sadensmol/issue/IVMS777-21/randd-how-to-use-voice-copy-with-local-models. Status: In Progress. It stays In Progress until the PR merges.
- **Team / project:** `ivms777` / `voice copy` (Linear workspace `sadensmol`, server `linear-sadensmol`).
- **Branch:** `ivms777-21-rd-how-to-use-voice-copy-with-local-models`, off `main` at `261ee2f`.
- **Base:** `<base>` (written `<base>` below). Repo: `<base>/ivms777`.
- **Repos touched:** `ivms777` only. One PR to `main`.
- **Scope:** voice copy with VoxCPM2 on the `mac` and `cloud` profiles:
  1. Saved voices, reached from Organize: record or upload examples, and name the voice.
  2. A cloning entry in the `voice` slot and its models-service child, pinned to one Hub revision.
  3. A stored-voice input through the whole `POST /speak` chain, with conditioning save/load bound to the model revision.
  4. Film casting: a narrator picker on `/feed` at Prepare and ↻ Regenerate. A `[voice]` bracket can cast a saved voice.
  5. A validation script that the user runs in their own stack.
  6. Phase 3 (added 2026-11-02, user request in `## Decisions`, pair t17): the Organize memory card shows each film by language, its state and its narrator;
     the card's ↻ Rebuild becomes "↻ Retell memory" with a visible explanation; `/feed` always shows a narrator picker with Automatic, the preset voices
     and the saved voices; every film says who chose its voice: the user, the story or the memory's mood (§2.15; pair t18).
  7. Phase 4 (added 2026-11-02, user request in `## Decisions`, pair t24): a voice workbench on `/organize/voices`. Per example: ▶ Original and
     ▶ Generated (a test made through the models service `POST /speak`, with its text, model, revision, time and built/restored state); test texts
     (the reading text, fixed test sentences, the user's own text); per language "Use for films" and a "used for films" badge. A test stores the
     conditioning that films then restore. Stage, percent and Cancel on every test (§2.16).
  8. Phase 5 (added 2026-11-02, user requests in `## Decisions`, pair t28 + t30): a voice model per person. A guided recording kit (slow, normal
     and fast reading, moods, questions, emphasis, articulation, numbers and names); the user confirms the words of every take; a LoRA adapter of
     VoxCPM2 baked from ALL takes in a models-service child, with stage, percent and Cancel; films and tests of a baked voice use it. All voice
     work moves to a top-level Voices menu (`/voices` grid, `/voices/{id}` leaf). Slice A (the Mac feasibility bake) beside Slice V (the Voices
     page), then Slice B (§2.17).
     **Changed 2026-11-02 (rework, t39; the user's three requests in `## Decisions`):** films and tests of a saved voice speak ONLY through
     its voice model, baked from ALL its takes (the adapter alone, `use_clip=false`). No take is chosen for films, and there is no one-take
     fallback: a voice with no usable model says "bake required". The model stays when takes are added, edited or deleted; Rebake trains a
     new version from all current takes. One Test per person: a language, then a fixed sentence, the words of a take or your own text.
     Slices A and V are implemented (§4 steps 23-28b). Slice B starts after this rework is approved; it does not wait for §6.5 or §6.6.
     **Changed 2026-11-02 (ASR, t42; the user's two decisions of 2026-11-02 in `## Decisions`):** a take is free speech, and the kit text
     is only a suggestion. The user types no words and confirms nothing. Before a bake trains, local speech recognition in the models
     service (`openai/whisper-large-v3-turbo`, D90) hears the words of every take, also of uploads and of the takes stored before; the
     bake trains on those words. A take it cannot hear stays in the voice and shows why (§2.17 K2).
  9. Phase 6 (added 2026-11-03, the user's two requests of 2026-11-03 in `## Decisions`, pair t50): Automatic may pick a saved voice.
     **Changed 2026-11-03 (Automatic, t50; the user's "Yes, presets + saved" in `## Decisions`):** with Automatic, the story pass may pick
     as the narrator a preset voice or one of the user's saved voices whose voice model can narrate the film's language here. The mood
     fallback still picks a preset. One voice engine per film stays (D17): a saved narrator's passages use saved voices only, a preset
     narrator's passages presets only. With no such saved voice, Automatic picks presets only, as before (§2.18). A narrator that the user
     chose reads the whole film; that part is pair t50 and is built outside these lanes.
  10. Phase 7 (added 2026-11-03, the user's request in `## Decisions`, pair t62): editable narration. When a film is made, its narration
     shows in "Tell the story better" on `/feed` as timed rows tagged "narration", in the same list as the notes, on load, with no import
     step. The user changes the words; ↻ Regenerate speaks them as written. The story pass is skipped; the voice, the timing, the score
     and the conductor run again. **Changed 2026-11-03 (t63):** the edited words stay, per memory and language, for every later
     ↻ Regenerate until the user chooses "Write a new story". A film made before this change shows why it has no rows (§2.19).
     Also (the user's "Fix in this task" in `## Decisions`, pair t64): the existing shot/audio order defect is fixed in this phase
     (§2.19 I). The told sentence order is the film's order for the words, the voice and the pictures; the memory's photo order is
     asked for, checked and corrected once.
- **Out of scope:** a Jetson engine, Jetson validation and Jetson UI wording; removing Jetson from code or docs; replacing Silero or MusicGen;
  face detection and person clustering (`docs/design.md:32`); ASR (**changed 2026-11-02, ASR, t42:** local ASR that hears the words of a
  take for a bake is in scope, §2.17 K2; ASR for anything else stays out); training or fine-tuning (**changed 2026-11-02, Phase 5, user + pair t28:** a per-voice LoRA adapter of VoxCPM2,
  trained in a models-service child, is in scope, §2.17; full fine-tuning (SFT), training on the CPU and training outside the models service stay
  out); XTTS-v2; CPU inference; remote inference.
- **History:** the r&d plan of this task is delivered: Linear document https://linear.app/sadensmol/document/voice-copy-with-local-models-randd-findings-ivms777-21-d484ff61427f (local copy `<base>/.scratch/research/findings.md`, with `family-a-cloning-tts.md`, `family-b-voice-conversion.md`, `deps/`).
- **Gate:** no code before the user approves this plan. This plan has no mermaid block.
  Phase 3: no Phase 3 code before the user approves this amendment (t17: "PLAN.md amended via planner, then plan gate, then code").
  Phase 4: no Phase 4 code before the user approves this amendment, including the "Use for films" question (`## Open items outside scope`; t24).
  **Approved 2026-11-02** (`## Decisions`: "Yes (Recommended)" to the "Use for films" question, and "approved go ahead" to the t25 fix).
  §6.3 stays pending; Phase 4 does not wait for it (t24).
  Phase 5: no Phase 5 code before the user approves this amendment (t28, `<base>/.pair/decisions.md:320`: planner amends PLAN.md, docs first, plan gate, then code).
  Slice A and Slice V start after that approval; Slice B starts after §6.5 PASS, §6.6 PASS and the answer to the deleted-take question
  (`## Open items outside scope`). The ASCII wireframes of §2.17 V and K6 are part of this gate (t30).
  **Approved 2026-11-02** ("The user approved.", `## Decisions`). Slices A and V are implemented and green (§4 steps 23-28b). The user
  checks §6.5 and §6.6 are pending.
  **Rework of Slice B (2026-11-02, pair t39 and the user's three requests in `## Decisions`):** no Slice B code before the user approves
  this rework. Slice B no longer waits for §6.5 PASS, §6.6 PASS or a gate answer: the deleted-take question is answered (keep the model,
  Rebake). The ASCII wireframes of §2.17 K6 are part of this gate. Until the user bakes a voice in the app, that saved voice cannot narrate
  a film: the film says "bake required". If §6.5 fails, saved voices cannot narrate on this Mac until that is solved. The main session
  reports that to the user as a blocker and never puts back a one-take fallback.
  **ASR amendment (2026-11-02, t42 and the user's "Yes, add it (Recommended)" in `## Decisions`):** no code of this amendment before the
  user approves it at the plan gate: lanes 30b, 31c and 31d, and the parts of 32a, 32b and 32c that it changes (§2.17 K2, K4-K7). Lanes 30
  and 31 are done in the working tree; lane 31's K2 parts stay frozen as typed until 31c reworks them. Lane 31b and 32b's model-only
  casting work (on the settled K3 names) go on under the earlier approval. Lane 31d edits the models service and waits like 31b (D84).
  The two ASCII wireframes of §2.17 K6 "Changed (ASR, t42)" are part of this gate.
  **Automatic amendment (2026-11-03, pair t50 and the user's "Yes, presets + saved" in `## Decisions`):** no code of this amendment
  before the user approves it at the plan gate: lanes 35, 36a and 36b (§4 Phase 6). It does not stop the work in flight: the t50 builder
  (a chosen narrator reads the whole film), the t49 recorder Pause and the Slice B lanes go on under their own agreements. The lanes of
  this amendment start after them (§4 Phase 6, D111). This amendment has no wireframe.
  **Narration amendment (2026-11-03, pair t62 and the user's request in `## Decisions`):** no Phase 7 code before the user approves this
  amendment at the plan gate: lanes 39, 40a, 40b and 40c (§4 Phase 7). It does not stop the work in flight: steps 33, 34, 37 and 38 and
  the t60 and t61 fixers go on under their own agreements. The ASCII wireframes of §2.19 E are part of this gate.
  **Changed 2026-11-03 (plan check, t63):** §2.19 A-H are revised (the draft is rebased and kept, `draft_token`, `narration-data`, the
  save barrier, the texts, the card remap, the app limit). **Changed 2026-11-03 (pair t64):** §2.19 I (the shot/audio order fix, the
  user's "Fix in this task") is part of this amendment: lane 39b. No code of it before the user approves this amendment.

## 1. Facts

### 1.1 Ticket, user and pair input

- Title: "r&d | how to use voice copy with local models?". Description: "research what local models we could use for copy voice and what to use to reproduce later?".
  Project summary: "we need to add ability to copy user voice to be used later in voice over and stuff". No attachments. One comment (the findings link).
- Earlier user direction (2026-10-31), verbatim: "i guess we should have some buttons to write/load voice examples and create a virtual person from it? then during the story creation and voiceover we could choose the voice."
- The user's 2026-11-01 decisions are in `## Decisions` (verbatim): implement in this task; commercial-use models only; skip Jetson.
- Pair decision t4 (`<base>/.pair/decisions.md:87-95`):
  - Saved voices are reached from Organize, separate from the "Organize photos" selector (`web/templates/organize.html:5-22`). No "Organize files…" on it.
  - "Register" = attach examples to a named saved voice. A saved voice is user data, never a `catalog.Speaker`. A typed person name is fine. No face work (`docs/design.md:32`).
  - The narrator picker sits on `/feed` at Prepare and ↻ Regenerate (`web/templates/feed.html:42-46`, `:139-142`), not on the card's ↻ Rebuild (`docs/design.md:1713-1715`).
  - The speech model stays automatic (`models/catalog.py:485-494`). A recording in progress is a leaf level that Esc cancels first (`ivms777/CLAUDE.md:112-115`).
  - A typed transcript only when the chosen model needs one. Full slice: stored voice through `/speak`, cloning engine + validation, Saved voices UI, film casting.
- Pair decision t8 (`<base>/.pair/decisions.md:97-117`, the later entry wins):
  - Engine: VoxCPM2 on mac + cloud, `voxcpm==2.0.3`, weights `openbmb/VoxCPM2@32279effe8c19989596f05d353d1447f51d9e915`, `load_denoiser=False`, no `timestamps` extra.
  - First steps: resolve deps with builds enabled; check py3.12; check every transitive licence for commercial use; validate GPU runs in the models service (the user runs it).
  - Build revision-bound conditioning save/load; add a revision field (`HfSource` has none, `models/catalog.py:62-66`); catalog `profiles=("mac", "cloud")` (pattern `models/catalog.py:420`).
  - Generic capability rule: no cloning entry on the active profile → Saved voices says voice copy is not available and refuses register/assign;
    an explicit stored-voice request fails the film with a clear reason (`feed/build.py:296-306`), never the preset fallback (`feed/build.py:266-276`).
  - The jetson profile still exists in code (`config.py:10`, `models/catalog.py:136-137`). Removing it is not this task.
  - Silero and MusicGen licences: an open item outside scope. No ticket. Never call Silero a commercial fallback.
- Pair decision t9 (`<base>/.pair/decisions.md:118-125`):
  - WAV check inline in `feed/saved_voices.py:add_example` with stdlib `wave`: uncompressed, 16-bit, mono, exactly 48000 Hz, every frame present. `storage/wav.py` does not change.
  - Deletion runs through an outbox + drain, never inline (`ivms777/CLAUDE.md:159-161`; D15). The request records the intent and returns a "deleting…" fragment.
    Pending targets reject mutations and new casting. An explicit saved narrator never becomes Automatic or a preset; it fails clearly.
  - The drain sits in `drain_pass` group 1 (`ingest/pipeline.py:53`), run by the worker (`ingest/cli.py:28`). With `inline_drain` on, a background schedule runs it
    after a delete and at start. Never a synchronous `drain_now`.
- Pair decision t10 (`<base>/.pair/decisions.md:126-137`; detail `<base>/.pair/files/merge-plan-check-t10.md`):
  - Keep `memory_narrators.voice_id` `ON DELETE SET NULL` and the selected "unavailable" option. An absent `narrator` field keeps the stored choice.
    A stored choice always shows in the picker, also on a profile with no cloning entry.
  - `AUTOINCREMENT` ids on `saved_voices` and `voice_examples`. Check-then-write in one `BEGIN IMMEDIATE`. A write-ahead cleanup row before every voice file write.
  - The background drain also wakes after each film build and each add-example request, with a `running` flag under its lock.
  - Films finished before the delete keep their sound until regenerated; a film still being made fails. This is a gate question with that recommendation.
- Pair decision t11 (`<base>/.pair/decisions.md:138-147`): film audio is written only at "saving the film"; a failed voice check writes none; no failure path prunes audio (D34).
- User request of 2026-11-02 (verbatim in `## Decisions`): the Organize window does not say which language a film is in, which films exist,
  what ↻ Rebuild does, which voice narrates, or where to choose a voice.
- Pair decision t17 (`<base>/.pair/decisions.md:185-192`; writer view `<base>/.pair/files/view-organize-ux-writer.md`):
  1. Card: one row per `LANGUAGES` entry, a plain link `/feed/{key}?lang=xx` labelled "▶ Play English film" / "Prepare English film" / "Open English film"
     (ready = status ready AND `narration_key`; building/queued first), its state and the recorded narrator. New `feed/store.py:film_states`. The chat card stays.
  2. Card ↻ Rebuild → "↻ Retell memory" with visible text (rewrites title, description, photo order; makes no film; deletes the films in <languages with a row>,
     or "No film to delete"). No narrator picker on it (t4). The global "Rebuild memories" keeps its label.
  3. `/feed`: "Film language" buttons with ✓ when ready; "This film: <language> · narrated by X" (missing → "Voice not recorded"); the picker
     "Narrator for the next film" always shown: Automatic · Preset voices (`feed.voice.offered`) · Saved voices; the hint "Add a voice under Organize → Saved voices"
     (no link) or a disabled "Voice copy not available on this device"; one line about [bracket] notes.
  4. `memory_narrators.preset_speaker TEXT`, a CHECK in the CREATE, an `_ADDED_COLUMNS` entry, the exclusion enforced by the store; `"preset:<name>"` validated with
     `feed.voice.resolve`; the build: a chosen preset wins over `story.narrator`, brackets cast presets, `narrator_chosen_by="user"`, cast labels.
  5. Docs first: `docs/design.md` §6, §11.1 (+ film mermaid), §13; the comment `web/app.py:506-507`; `docs/ui.md`; `docs/data-model.md`. D17 still holds.
- User addition of 2026-11-02 (verbatim in `## Decisions`): "don't forget about auto mode - when you choose voices on your own".
- Pair decision t18 (`<base>/.pair/decisions.md:194-202`; writer view `<base>/.pair/files/view-auto-voice-writer.md`), an addition to t17. No new storage; casting does not change (D17, the passage gate):
  1. Picker option "Automatic — the app picks a preset voice for this memory"; the hint "The story picks the voice; otherwise the memory's mood guides the choice.
     Saved voices are never picked automatically." Automatic stays selected after a build (D18). No predicted preset before preparation.
  2. `decisions["cast"].narrator_chosen_by` = `"user"` | `"story"` (`resolve(language, story.narrator)` succeeded) | `"mood"` (`pick_speaker`).
     Old rows: `"story"` + `narrator_asked_was_real` false → mood, true → story; a cast with no `narrator_chosen_by` is Automatic, story/mood by the same flag;
     flag absent → "picked automatically" with no reason; no cast, or `narrator_used` null → "Voice not recorded".
  3. "This film": "narrated by Heart (female, warm) — picked automatically by the story" / "— picked automatically using the memory's mood" / "— your choice".
     Card: "English · ready · Heart (auto)" or "(your choice)". Never from the current `memory_narrators` row.
  4. `/feed` "Other voices: passage N → <label>" only for `passages[].used` set and ≠ `narrator_used`, N from 1; never `passages[].asked` or `model_cast`.
     The recorded note "a note asked for a different voice and none was cast" shows only when the build recorded it.
  5. Bracket line: Automatic "A [voice] note can request another preset voice for one passage"; a saved narrator: another saved voice.
  6. Docs first with t17 (§11.1, §13, `docs/ui.md`). Tests: preset vs Automatic with the same speaker; story vs mood; a rejected passage cast not listed; an old row; card and `/feed` text per case.
- Pair plan check t19 (`<base>/.pair/decisions.md:204-216`): the card's film block refreshes in place (poll while building, `pageshow` on a bfcache return);
  the 409 text "«<name>» cannot narrate in <Language>."; the "·" separator on card rows; a follow-up for friendly preset names. The `chat.js` reload was disputed;
  the user settled it on 2026-11-02: "Fix now (reviewer)" (`## Decisions`), so lane 15b guards it.
- User report of 2026-11-02 (verbatim in `## Decisions`): on the `/feed` ready screen the Narrator select overlaps ↓ Download; a film narrated by the saved voice
  "me" does not sound like the user. The user answered "Not now" to a high-likeness (ref_continuation) test in Phase 3.
- Pair decision t20 (`<base>/.pair/decisions.md:217-222`; detail `<base>/.pair/files/merge-overlap-similarity-t20.md`):
  - Layout: one viewport-bounded header, no fixed offsets between controls: a wrapping action row (language, ↓ Download, ×) and a settings region with the
    `.feed-regen` form (This film, Other voices, the full picker, the visible help lines); bounded height, internal scroll; the select `min-width:0; max-width:100%`.
    Ids, classes, POST ownership and the history JS stay; each opacity rule stays on its own control. A markup test; the user checks 360/800/1280 px.
  - Voice: more recordings do not help (only the newest same-language example is used, D19). Production stays in reference mode; `CONTROLS` and the limits stay (D16).
    No transcript UI. `voices.html` may add clean-speech and newest-example guidance, with no likeness or length promise.
- User question of 2026-11-02 (verbatim in `## Decisions`): why the recorder always gives the same text. t20 resumed (`<base>/.pair/decisions.md:224-233`):
  four reading texts per language, a `prompt_id` on `add_example`, a preparation panel with "Different text" and "Start recording" (§2.15 I).
- User request of 2026-11-02 (verbatim in `## Decisions`): are all saved examples used? A "bake" button, a test button, "listen original, listen
  generated", more tools to make the voice reliable. The user chose "Phase 4 in this task (Recommended)".
- Pair decision t24 (`<base>/.pair/decisions.md:257-264`; writer view `<base>/.pair/files/view-voice-workbench-writer.md`):
  - Answer: no. A film uses only the newest non-deleting example of its language (D19); examples never combine; reference mode, one clip, no
    transcript (D16). A bake = the encode of ONE example; it already happens on first use and is stored. Baking earlier only removes that encode
    from the first film: no memory saving, no likeness gain, no speed number. A model made from all examples = fine-tuning: out of scope.
    **Changed 2026-11-02 (Phase 5, t28):** in scope now as a per-voice LoRA adapter (§2.17).
  - Phase 4 on `/organize/voices`: (1) a per-language "used for films" badge; (2) per example ▶ Original + ▶ Generated through `POST /speak`, with
    the reference id, language, text, model/revision, elapsed time, built/restored; (3) test texts: recorded = the stored reading text, labelled
    "not a verified transcript"; uploads = fixed or own text, no ASR; fixed unseen same-language sentences and a free text; (4) "Use for films" per
    language: it supersedes D19 and needs the user's decision at the gate; one resolver for films and tests; the default stays the newest; a deleted
    chosen example shows as unavailable and asks for a new choice (the planner specifies), never a silent substitute; a test never changes casting.
  - Generate-test IS the bake (`save_conditioning`; no second cache; no Bake button). Stage + % + a real Cancel; the state survives a list refresh;
    a test is bound to example, text, language and revision; stale or cancelled results are dropped; test files belong to the voice outbox.
  - Rejected: clip concatenation or averaging, fine-tuning, automatic likeness scores, a UI button that runs voice_check, ref_continuation now.
    D16 and `CONTROLS` stay. **Changed 2026-11-02 (Phase 5, t28):** fine-tuning is in scope as a per-voice LoRA adapter (§2.17), and the
    working tree already uses ref_continuation for a take with words (§1.8).
- User request of 2026-11-02 while testing Phase 4 (verbatim in `## Decisions`): "… you fucking idiot should use ALL THE SAMPLES! …
  THEN YOU SHOULD USE ALL THE SAMPLES TO BAKE IT INTO THE PERSON'S MODEL …"; and the answer "This Mac first (Recommended)".
- Pair decision t28 (`<base>/.pair/decisions.md:315-321`; detail `<base>/.pair/files/merge-bake-all-samples-t28.md`; writer view
  `<base>/.pair/files/view-bake-all-samples-writer.md`):
  - Answer: "used for films" = the ONE example new films of that voice use in that language, the user's pick, else the newest
    (`feed/saved_voices.py:640-670`). No model is trained from the collection. Clips with stored text already use ref_continuation
    (`audio/voxcpm_voice.py:83-94`, `:269-275`); the cache encodes one clip and changes no weights.
  - Phase 5: a guided recording kit; keep all recordings; a per-person LoRA adapter from the full dataset; films and tests use it. Pinned APIs
    `voxcpm2.py:1111-1182`, `:1195-1256`, `training/data.py:19-61`; the wheel has no trainer script; "5-10 min" is the package authors' guidance.
  - Rules: the user confirms or corrects the words of each take; uploads need a transcript; unresolved clips are shown and block the bake, never
    dropped. The planner defines the snapshot, the revision and the training config, a versioned adapter, stale and rebuild rules, progress and
    cancel, and which adapter version films and tests use. Training and adapter inference stay in the models-service child under the scheduler;
    a switch resets, then loads, one generation at a time; `load_lora` needs a LoRAConfig at build (`core.py:317-335`), and our child has none.
  - Process: amend PLAN.md and replace "training or fine-tuning" (PLAN.md:22, 109, 117, 1592) and D16, D19, D46-D48; docs first; the plan gate;
    code. The first slice is a feasibility bake. No time, memory, likeness or mood promise before it is validated.
- User request of 2026-11-02 (verbatim in `## Decisions`): "add Voices menu item, move all voices stuff there. improve UX/UI what we have now is
  total shit!"
- Pair decision t30 (`<base>/.pair/decisions.md:323-331`; detail `<base>/.pair/files/merge2-voices-menu-t30.md`; writer view
  `<base>/.pair/files/view-voices-menu-writer.md`): D20 is reversed.
  - Nav: Upload, Library, Chat, Organize, Voices; the Organize link goes. `/voices` is a top-level grid with no close; cards are plain links with
    the person, language minutes and real state (no playback); "New voice" at the top; a card opens `/voices/{id}` (the one push); no ctx.
  - Leaf: Voices › name, model state with bake stage/%/Cancel; rename and delete in a secondary menu; tabs Recordings / Voice model / Test, `?tab`
    via replaceState; kit texts and their takes share rows; the recorder opens at its prompt; Test keeps target, text, Generate, progress and
    playback together, and the comparison names the selected real take.
  - Esc: live recording → unsaved take → prepared recorder → focused field → close; tabs are no level. Close = `history.back()` only when the
    previous entry is known to be `/voices`, else `/voices`, never on `history.length > 1`; `photo.html:181` / `feed.html:289` = a follow-up.
  - `voices.js` lifecycle is an acceptance condition: bind once to current nodes, a once-guard, a question before a boosted nav that would
    discard a recording, stop the mic, timers and stale callbacks on leave, running jobs reconnect.
  - Routes move to `/voices/*` and `/voices/{id}/*`; `/voices/preview` and its cancel stay global; literal paths before `/voices/{voice_id}`;
    `GET /organize/voices` → 308. Slices: A (the Mac bake) beside V (nav, grid, leaf with today's features and one per-language film-example
    selector with its tombstone; `web/`, docs, tests), then B. Docs first per slice; ASCII wireframes in PLAN.md.
- Pair fixes in the Slice A and Slice V code (`<base>/.pair/decisions.md:347-402`), all in the working tree; this plan only states them:
  - t32 (`:347-354`): `model_ensure("voice")` enters the counted `VoiceGate` like a speak; a bake's drain also waits for manual voice ensures.
  - t33 (`:356-360`): a Cancel during the last training step saves nothing.
  - t34 (`:370-373`): a bake publishes its end, opens the gate and clears `active` in one lock block, before a next start can begin.
  - t35 (`:362-368`): `POST /models/voice_train/ensure` ALWAYS answers 409; only a bake loads `voice_train`.
  - t36 (`:394-402`): `voices.js` owns every async reply through body decoding: a dead page session never swaps, retries or reads.
  - t37 (`:375-381`): a bake take needs a positive frame count and every frame present.
  - t38 (`:383-392`): the check script's console table is blind; `key.json` and `report.json` reveal the mapping; listen first.
- User requests of 2026-11-02 after Slice V landed (verbatim in `## Decisions`): films must use ALL recordings, never one; one Test per
  person with a chosen or typed text; do the rework now and do not wait for the §6.5 bake check. The user's answer to the deleted-take gate
  question (also in `## Decisions`): keep the model, add a button to rebake.
- Pair decision t39 (`<base>/.pair/decisions.md:404-416`; writer view `<base>/.pair/files/view-films-all-recordings-writer.md`):
  - Films and tests use ONLY the adapter baked from ALL takes of the person, with `use_clip=False`. No prompt take, no film-take select, no
    pin, no one-take fallback (`feed/saved_voices.py:654-670`, `:850-859`). D73's prompt take, D69's select and D46-D48 are superseded.
  - No usable model → "bake required"; never a take, never Automatic. The model stays after additions, word edits and deletions; Rebake
    trains a new version. The page shows the model version, the training snapshot and the changes since it.
  - §6.5: variant C (the adapter alone) must pass; A and B only help to diagnose. If C fails, report the blocker; never restore a take
    fallback. Slice V gets no patch now: its film area changes with Slice B, docs first; §6.6 steps 2, 3 and 9 reopen then.
- The writer's one-test proposal (`<base>/.pair/files/view-one-test-per-person-writer.md`), not yet settled with the reviewer: one Test per
  person (language, a fixed sentence, the words of any take, or your own text; one Generate; through the model only; ▶ Original beside
  ▶ Generated for a take's words); no model → "Bake the voice model first"; the per-take Test buttons go; a test row stores the voice and the
  model version. This plan refines it on the code (§2.17 K5, D80, D85).

### 1.2 VoxCPM2 facts at `voxcpm==2.0.3` (tag `2.0.3` = commit `19b6bf7`), read 2026-11-01

Source tree: `https://github.com/OpenBMB/VoxCPM/tree/19b6bf7590025418821a86dcb817504e0ad7e5df/src/voxcpm`. Label `code finding` unless marked.

- Licence: Apache-2.0 for the code (`pyproject.toml` `license`) and the weights (findings [A21], `published`).
- Python: `requires-python = ">=3.10"`; classifiers list 3.10 and 3.11 only (`pyproject.toml`).
- Direct dependencies: `torch>=2.5.0`, `torchaudio>=2.5.0`, `torchcodec`, `transformers>=4.36.2`, `einops`, `gradio>=6,<7`, `inflect`, `addict`, `wetext`,
  `modelscope>=1.22.0`, `datasets>=3,<4`, `huggingface-hub`, `pydantic`, `tqdm`, `simplejson`, `sortedcontainers`, `soundfile`, `librosa`, `matplotlib`,
  `funasr`, `spaces`, `argbind`, `safetensors` (`pyproject.toml`).
- `VoxCPM.__init__(voxcpm_model_path, zipenhancer_model_path, enable_denoiser, optimize, device, …)` — `core.py:15-23`.
  `VoxCPM.from_pretrained(hf_model_id, load_denoiser=True, …, optimize=True, device=None)` — `core.py:104-113`.
  A local directory skips the download (`core.py:153-154`). A Hub id downloads with NO revision (`core.py:157-161`).
- The denoiser (ModelScope) is imported only when it is enabled (`core.py:89-93`). `load_denoiser=False` passes `zipenhancer_model_path=None` (`core.py:164-166`).
  The text normaliser (wetext) loads only with `normalize=True` (`core.py:276-281`).
- `optimize=True` runs a warm-up generate (`core.py:96-101`). `VoxCPM2Model.optimize` compiles only on CUDA with triton; else it prints a warning (`model/voxcpm2.py:274-295`).
- `import voxcpm` imports `torchaudio` (`core.py:9` → `model/voxcpm.py:27`). The VoxCPM2 path reads audio with librosa (`model/voxcpm2.py:412`) and never calls `torchaudio.load`.
- Device: an explicit `"mps"` or `"cuda"` raises when it is not available; there is no fallback (`model/utils.py:198-224`). On MPS the dtype is forced to float32 (`model/utils.py:146-171`).
- `build_prompt_cache(prompt_text, prompt_wav_path, reference_wav_path, trim_silence_vad)` — `model/voxcpm2.py:681-740`.
  A reference only gives `{"ref_audio_feat": <CPU tensor>, "mode": "reference"}`. The latents come from `audio_vae.encode(...).cpu()` (`model/voxcpm2.py:421`).
- `generate_with_prompt_cache(target_text, prompt_cache, min_len=2, max_len=2000, inference_timesteps=10, cfg_value=2.0, retry_badcase=False, …)` — `model/voxcpm2.py:774-797`.
  `prompt_cache=None` is zero-shot (`model/voxcpm2.py:819-822`).
  The public `generate` rebuilds the cache on every call (`core.py:268-273`) and uses `retry_badcase=True`, `max_len=4096` (`core.py:180-195`).
- No `seed` argument (`core.py:180-195`). No timestamps module in the tag. No save/load API (findings §4.2; upstream issue #417).
- Output rate: `tts_model.sample_rate` (`model/voxcpm2.py:244`); 48 kHz in the checkpoint config (findings [A21], `published`).
- Weights `openbmb/VoxCPM2@32279ef…`: 2,290,004,544 params; `model.safetensors` 4,580,080,592 B + `audiovae.pth` 376,951,122 B (findings [A21], `published`).

### 1.3 ivms777 code facts (branch head `261ee2f` plus the user's uncommitted changes)

- Speak chain: `SpeakRequest` = `text, language, voice, segments, voices` (`modelsvc/app.py:51-64`) → `/speak` (`modelsvc/app.py:229-242`) →
  `CompositeBackend.speak` → `ensure_voice_for(language)` → `_run(["voice"], Priority.BATCH, "narrating", …)` (`modelsvc/backends/composite.py:107-121`) →
  `VoiceBackend.speak` → child `call("speak", text, voice, segments, voices)` (`modelsvc/backends/voice_backend.py:12-19`).
- Reply: `AudioResponse` = `audio, sample_rate, model, segments` (`modelsvc/app.py:89-101`); `_audio_response` passes `result["model"]` through (`modelsvc/app.py:194-200`).
  The client decodes it (`inference/models_client.py:13-21`). Client `speak` timeout = 300 s (`inference/models_client.py:53-67`).
- Routing: `VOICE_TARGETS` has `kokoro-82m` and `silero-v4-ru` (`modelsvc/slots.py:45-48`). `voice_entry_for` = selected entry, else the first offered entry for the language (`modelsvc/slots.py:187-200`).
  `ensure_voice_for` swaps the unit (`modelsvc/slots.py:202-218`). `_register_voice` falls back to Kokoro for an unknown key (`modelsvc/slots.py:224`).
  Child args = `(repo or local path, embed_device)` (`modelsvc/slots.py:128-163`). Children run in `spawn` processes (`modelsvc/torch_process.py:24`).
- Catalog: `HfSource(repo)` (`models/catalog.py:62-66`); `ModelEntry` (`models/catalog.py:114-133`); voice entries (`models/catalog.py:341-394`); `voice` is automatic (`models/catalog.py:485-494`).
- Children: Kokoro returns its Hub repo as `model` (`audio/kokoro_voice.py:73`); Silero returns `silero-v4-ru` (`audio/silero_voice.py:118`).
  Silero splits text at sentences, ≤ 500 symbols (`audio/silero_voice.py:32-67`). Same-voice passages are one call (`audio/segments.py:40-86`).
- Casting: `offered()` = speakers of the FIRST catalog voice entry for the language (`feed/voice.py:35-40`); `resolve()` accepts only those names (`feed/voice.py:43-56`).
  The story pass sees `catalogue_for_prompt` (`feed/build.py:228-230`). Narrator = `resolve(...) or pick_speaker(...)` (`feed/build.py:270`).
  Per-passage voices: `_passage_voices` (`feed/build.py:407-437`). The speak call and its failure path (`feed/build.py:296-306`), `_fail` (`feed/build.py:566-591`).
  Progress ticking exists for music only (`feed/build.py:82-136`, `:318-356`).
- Storage: `prune_orphan_audio` deletes EVERY blob in `data_dir/audio` that no presentation references (`feed/store.py:87-105`).
  `Settings` dirs (`config.py:283-298`); `AppContext` holds `derived`, `originals`, `audio` (`web/deps.py:21-34`, `:48-63`).
  WAV helpers are numpy + `wave` only, PCM-16 (`storage/wav.py:1-57`). Upload disk guard: `FREE_SPACE_FLOOR` (`web/upload_api.py:21`, `:59-67`).
  `decode_wav` reads every payload as int16 and downmixes any channel count; it checks no sample width (`storage/wav.py:49-57`).
  It is shared with `audio/` in the models image (`Dockerfile.models:42-47`), `feed/beats.py:83` and `feed/build.py:527`.
  Stdlib `wave` (venv Python 3.13.5, read 2026-11-01): a float WAV raises `wave.Error("unknown format: 3")`; a truncated data chunk gives a short `readframes` and no error.
- Deletion and drains:
  - Folder deletion is the outbox pattern: an idempotent enqueue (`ingest/folders.py:41-47`), a drain (`:50-66`), a "deleting…" flag (`:20-38`, `web/templates/upload.html:19-20`).
  - `drain_pass` group 1 runs thumbnails, place facets and `process_folder_deletions` with no guard (`ingest/pipeline.py:50-62`). The worker builds the same
    `AppContext` (`ingest/cli.py:15`) and loops `drain_pass` every `POLL_SECONDS = 10` (`ingest/cli.py:10`, `:27-29`).
  - `drain_now` runs `drain_pass` only when `inline_drain` is on (`web/app.py:239-255`). Its only caller is upload finish (`web/upload_api.py:157-163`).
    `inline_drain` defaults to True (`config.py:191`); compose sets it False for `app` (`compose.yaml:16`). `make up` sets no `IVMS777_INLINE_DRAIN` (`Makefile:88-107`)
    and also runs the worker (`Makefile:96`), so on mac both `app` and the worker can drain.
  - Connections autocommit (`db/connection.py:24`) with a 5 s busy timeout (`:29`). `BEGIN IMMEDIATE` precedent: `models/slots.py:147-157`.
    `AppContext.conn` is thread-local (`web/deps.py:37-45`). Background threads are daemons with an `await_*` test seam (`web/app.py:232-237`, `:1042-1044`).
  - `LocalStorage.delete` is `unlink(missing_ok=True)` (`storage/local.py:37-40`). `LocalStorage.write` truncates the file in place (`storage/local.py:32-35`).
    The Prepare submit ignores the reply status (`web/templates/feed.html:242-252`). `feed_prepare` reads `lang: str = Form(default="")` (`web/app.py:978`).
  - SQLite 3.47.1 (venv, read 2026-11-01): an `INTEGER PRIMARY KEY` reuses the highest id after its row is deleted; `AUTOINCREMENT` does not.
  - A finished film keeps its audio while its row references it (`feed/store.py:94-103`), and `feed_audio` serves it (`web/app.py:966-975`).
    The ready save is one upsert (`feed/store.py:36-60`) at "saving the film" (`feed/build.py:381-404`). `_fail` stores `narration_key=None` (`feed/build.py:566-591`).
- Schema: `SCHEMA_VERSION = 13` (`db/connection.py:8`); `migrate` runs `schema.sql` every time (`db/connection.py:72-102`); full DDL doc in `docs/data-model.md`.
  Past table additions bumped `SCHEMA_VERSION` (commits `f5e4e44`, `5f1bcab`).
- Web: the film is built in `app` on a daemon thread (`web/app.py:995-1043`); Prepare/Regenerate POST to `/feed/{key}/prepare` (`web/app.py:977-993`).
  The feed page submits those forms with `fetch` + `location.replace` (`web/templates/feed.html:240-251`) and handles Esc (`web/templates/feed.html:209-225`).
  Settings never shows `voice` (`web/app.py:492-495`). The only `MediaRecorder` use is the film export (`web/static/feed-gl.js:1430-1570`).
- Packaging and gates: the wheel ships listed top-level packages only (`pyproject.toml:72`). The one-model gate scans `chat search ingest web albums storage db embedding inference feed`
  (`tests/test_one_model_process_gate.py:39-42`). The models image runs `uv sync --extra models` (`Dockerfile.models:24`) and copies `audio/` and `storage/wav.py` (`Dockerfile.models:42-47`).
- Mac stack: `make models` = `uv sync --extra models` + `scripts.fetch_models` (`Makefile:75-83`); the models service listens on `9100` (`Makefile:64`).
  `fetch_models` pre-fetches EVERY entry of an automatic slot (`scripts/fetch_models.py:25-46`). The user's `make up` runs from the source checkout, not the worktree
  (`work_sadensmol/skills/task/references/finish.md`, Step 6b).

### 1.4 Binding rules (`ivms777/CLAUDE.md`)

- `:3-36` `docs/design.md` and its mermaid diagrams change FIRST, in the same change. `:120-141` one model process: VoxCPM2 loads only in the models service.
- `:51-66` never lose the user's place. `:68-118` layered navigation, `[grid, leaf]` history; a leaf's own levels (a recording) unwind first on Esc.
- `:163-204` verify with tests only; never call `localhost:8000/9000/9100/8080`; never load a model. `:223-232` a model is proven only by a run (the user's).
- `:236-251` model metadata is declared in the catalog. `:253-262` no "off" options. `:264-274` long jobs show a moving stage + percent and a Cancel.
- `:276-281` `uv`; every new module has tests; `app` reads, inference only through the models service.

### 1.5 Repo state — the user's uncommitted changes (read 2026-11-01)

`git -C <base>/ivms777 status --short`: 21 modified files and 2 untracked files. None of them is about voice copy:
`Makefile`, `albums/{compose,memories_build,seeds}.py`, `audio/kokoro_voice.py`, `db/connection.py`, `docs/{design,memories,models}.md`, `modelsvc/torch_process.py`,
`pyproject.toml`, `scripts/fetch_models.py`, `tests/{test_fetch_models,test_memories_build,test_memory_compose,test_memory_seeds,test_migrate,test_torch_process}.py`,
`tests/worker_fixtures.py`, `uv.lock`, `web/app.py`; untracked `audio/espeak_path.py`, `tests/test_espeak_path.py`.

Rules for every lane:

1. Keep the user's changes. Before a lane edits a file, it reads `git -C <base>/ivms777 diff -- <file>`.
2. Commit only our own hunks (`git add -p`). Ask the user only on a real conflict.
3. Overlaps found now:
   - `pyproject.toml` (`:46-51`, `:65-67`: `en-core-web-sm`) and `uv.lock` (`revision = 3` → `5`, `en-core-web-sm` entries). The dependency step edits both files.
     `uv lock` rewrites `uv.lock` as one file. Our hunks then sit next to the user's hunks. **This is a real conflict at commit time. The gate asks the user one question (see `## Open items outside scope` → "Question at the gate").**
     §3 step 7 is applied (2026-11-01). Our `voxcpm` and `[tool.uv]` hunks now sit in `pyproject.toml`, and `uv.lock` equals `<base>/.scratch/deps-voxcpm/proj/uv.lock`.
   - `db/connection.py`: the user's hunks at `:1-5`, `:29-46` and `:72-115`. We change only `:8` (`SCHEMA_VERSION`). Split the hunk with `git add -p` → `s`.
   - `scripts/fetch_models.py`: the user's hunks are the docstring (`:1-13`), `:79`, `:94-98`. We change only `wanted` (`:42-43`) and `_fetch` (`:49-55`).
   - `tests/test_fetch_models.py`: the user's hunks at `:63-95`. We append new tests at the end of the file.
   - `docs/models.md`: the user's paragraph at `:193-208`. We never edit `:185-210`. Our VoxCPM2 text goes into new lines after `:91`.
   - `docs/design.md`: the user's hunk at `:1395` (§11 memory-composition mermaid). We do not edit §11 (`:1319-1418`).
   - `web/app.py`: the user's hunk at `:1755-1785`. We edit only imports, the router registration near `:257`, and the feed section `:769-1100`.
     The t9 background drain lives in `web/voices_api.py`, so `drain_now` (`:239-255`) does not change.
   - Added by t9: `ingest/pipeline.py`, `tests/test_pipeline_pass.py`, `docs/ingest-pipeline.md`. `git diff -- <file>` is empty for each (read 2026-11-01). No overlap.
     `db/schema.sql`, `config.py`, `web/deps.py`, `web/templates/feed.html`, `docs/data-model.md` and `docs/ui.md` also have no user hunks.
4. Untouched by this plan: `Makefile`, `albums/*`, `audio/kokoro_voice.py`, `audio/espeak_path.py`, `docs/memories.md`, `modelsvc/torch_process.py`,
   `tests/test_migrate.py`, `tests/test_memor*.py`, `tests/test_torch_process.py`, `tests/worker_fixtures.py`, `tests/test_espeak_path.py`.
5. The user's stack runs from the source checkout. Our edits do not hot-reload it.

### 1.6 Phase 3 code facts (working tree with the implemented phases 1-2, read 2026-11-02)

- Card: `web/templates/_album_card.html:22-36` shows an unlabelled "▶ Play" (`:25`, link `play_url` = `/feed/{key}`, `web/templates/organize.html:51`) and
  "↻ Rebuild" (`:29-35`, the explanation only in a `title` tooltip). The note links use `play_url` too (`:50`). The chat renders the same partial with only
  `album` and `ctx` (`web/app.py:1590-1592`), so it gets no `play_url` and no controls.
- `organize()` is `web/app.py:659-717`; `notes_by_key` per memory at `:686-693`. A link with no `lang` opens the owner's stored `feed_language` (`web/app.py:767-775`).
- Films are per `(group_id, language)` (`db/schema.sql:129-146`). Statuses: `ready`, `failed` (`feed/build.py:750-775`), `running` (`feed/build.py:677-698`).
  The page plays a row only when it is `ready`, has a `narration_key`, and is neither building nor queued (`web/app.py:852-854`).
  Building and queued are `app.state.feed_building` / `feed_queued` = `(key, language)` (`web/app.py:814-815`).
- `decisions["cast"]` holds `narrator_used` (a preset name or a `saved:<id>` token), `narrator_chosen_by` (`"user"` | `"story"`), `narrator_asked_was_real`
  and `passages` (`feed/build.py:325-338`). No label is stored. A rejected passage cast is still listed with `used` = the narrator (`:325-327`, `:615-621`).
  HEAD's cast has no `narrator_chosen_by` and has `narrator_asked_was_real` (`git show HEAD:feed/build.py`, `:270-290`).
  The note "a note asked for a different voice and none was cast" goes into the ready row's `error` (`feed/build.py:340-343`, `:488`); `/feed` shows it as `missing`
  (`web/app.py:871-872`, `web/templates/feed.html:87-89`).
- `feed_rebuild` (`web/app.py:1004-1039`) → `rebuild_memory` (`albums/memories_build.py:63-93`) → `rewrite_memory` deletes every presentation of the group
  (`albums/memory_store.py:103`). `memory_narrators` rows stay. A retelling that fails writes nothing (`albums/memories_build.py:90-91`).
  The memory prompt fixes no language (`grep -i language albums/compose.py` finds nothing).
- Picker: `web/templates/feed.html:20-34`, shown only when `narrators` or `narrator_unavailable` is set (`:21`); the hint line `:163-165`.
  Language buttons `:42-47` use `feed.language.DISPLAY` ("English", "Русский", `feed/language.py:14`).
  `_narrator_picker` / `_store_narrator`: `web/app.py:882-940`. `feed_prepare` reads the raw form, so `""` and an absent field differ (`web/app.py:1060-1089`).
- Presets: `offered(language)` = the speakers of the first non-cloning voice entry for the language (`feed/voice.py:40-48`); Kokoro 10 en, Silero 5 ru
  (`models/catalog.py:360-394`). `Speaker` = `name, gender, register, note` (`models/catalog.py:113-116`). `resolve` is case-insensitive and returns the catalog
  name (`feed/voice.py:77-90`). `catalogue_for_prompt` formats `"<name> — <gender>, <register>: <note>"` (`feed/voice.py:93-97`).
- Build casting: `chosen` (`feed/build.py:236`) means "saved film" at `:237-238`, `:265`, `:307`, `:336`, `:350`, `:395`, `:401`, `:494`.
  The Automatic cast is `resolve(story.narrator) or pick_speaker(...)` (`:316-318`). The comment `:300-301` says the user chooses neither model nor speaker.
- English language names: `web/voices_api.py:52-59` (`language_name`) and `feed/story.py:177` (`language_names` in `feed/build.py:56`).
- Store: `NarratorChoice(voice_id)` (`feed/saved_voices.py:86-87`); `read_narrator` / `set_narrator` (`:612-650`).
  `migrate` runs `schema.sql` (only `CREATE … IF NOT EXISTS`), then `_ensure_columns` adds each `_ADDED_COLUMNS` entry by type only (`db/connection.py:62-69`, `:97-101`, `:139-145`).
  The user's DB (`~/.ivms777/ivms777.db`, read-only `sqlite3`, 2026-11-02): `memory_narrators` exists with no `preset_speaker`; `PRAGMA user_version` = 14.
- Settings comment `web/app.py:506-507`: "`voice` is deliberately absent: which voice narrates a film follows from its language and the memory's mood (§11.1),
  so there is nothing to choose." `feed/voice.py:3-4` says the user never chooses the model. `docs/design.md:1761` says the same; it stays true for the MODEL.
- Tests: the card tests live in `tests/test_web_feed.py:206-213` and `:724-765` (fixture with one memory and `FakeModels`, `:78-97`).
  The picker tests compare whole option lists (`tests/test_web_feed.py:863-873`, `:961-1014`), and two assert the picker is hidden (`:875-887`).
  `tests/test_feed_store.py` and `tests/test_web_organize.py` have no uncommitted hunks.
- `/feed` controls (t20; read 2026-11-02): `.feed-lang` is `position: fixed; top/left 1rem` (`web/static/feed.css:111-119`); `.feed-regen` is fixed at
  `top: 1rem; right: 4rem` and grows to the left (`:135-141`); `.feed-download` is fixed at `right: 11.5rem` (`:322-326`). The select inside `.feed-regen`
  (`web/templates/feed.html:60-65`) has no CSS. Opacity rules sit on each control: `:86-87` (×), `:155-156`, `:339-347` (exporting Download), `:549-557` (stale Regenerate),
  `:572-576` (building). The × close is fixed at `top/right 1rem` (`:71-75`) and rendered outside the ready branch (`feed.html:37-38`); the refusal line is inserted after the form (`feed.html:281-288`).
- `web/static/chat.js:80-83`: a window `pageshow` listener, behind `window.__chatPageshowWired`, calls `location.reload()` on every `e.persisted`.
  `#chat-q` exists only on the chat page (`web/templates/chat.html:43`). JS is tested by reading the source (`tests/test_web_feed.py:695`).
- `web/static/feed.css`, `web/static/chat.js` and `tests/test_web_chat.py` have no uncommitted hunks; `web/templates/voices.html` is ours (untracked).
- User hunks near Phase 3 edits (`git diff -U0`, 2026-11-02): `db/connection.py:73-85`, `:96-98`, `:104-115` (ours: `:8`; Phase 3 adds a line at `:68-69`);
  `web/app.py:1861-1884` (`log_albums_to`, `app_factory`); `docs/design.md:1497` (§11 mermaid; §11 is now `:1421-1520`). Phase 3 edits none of them.

### 1.7 Phase 4 code facts (working tree with phases 1-3, read 2026-11-02)

- Which example a film uses: `example_for` = the newest non-pending example of the voice in the language (`feed/saved_voices.py:613-624`,
  `ORDER BY e.id DESC LIMIT 1`); `castable` pairs each non-pending voice with it (`:627-636`). Callers: the build (`feed/build.py:559`),
  `_narrator_picker` (`web/app.py:942`) and `_store_narrator` (`web/app.py:1010`). An older example is never used while a newer one exists (D19).
- The film's reference: `_reference` (`feed/build.py:568-577`) = audio, sha256, language, transcript and every stored conditioning of ONE example.
  The speak call (`:380-401`; saved films inside `_Ticking`, `:391-396`), then `_keep_conditioning` stores each `built` blob with
  `save_conditioning` (`:403-405`, `:580-593`). `_saved_cast` holds the three failure texts (`:542-565`). The ready save re-checks every used
  voice and example under one write lock (`:515-535`, `_deleted_voice` `:596-605`).
- The child: reference mode only (`audio/voxcpm_voice.py:35`, `:246`) and `CONTROLS` (`:37-45`) (**changed 2026-11-02:** the working tree
  now also has `ref_continuation` for a take with words, §1.8). `_conditioning` restores a stored blob when model,
  revision, voxcpm version and sha256 match, else it encodes the clip (`:226-253`). The reply carries `model`, `revision`, `conditioning` (`:217-224`).
- `/speak` takes at most 8 references of ≤ 4 MiB each (`modelsvc/app.py:52-53`, route `:315-339`). A reference request runs
  `ensure_voice_for(language, cloning=True)`, then `_run(["voice"], Priority.BATCH, "narrating", …)` (`modelsvc/backends/composite.py:126-145`).
  `ensure_voice_for` swaps the unit outside `_run` (`modelsvc/slots.py:211-230`), and a swap first unloads the unit, which stops its child
  (`modelsvc/slots.py:129-133`; `modelsvc/registry.py:54-59`). `grep -i cancel modelsvc/app.py modelsvc/backends/composite.py` finds nothing.
  **Corrected 2026-11-02 (t25, round 2):** `stop` takes the worker's `_lock`, which `call` holds across `recv` (`modelsvc/torch_process.py:169-186`),
  so a swap waits for a running call and does not cut it. The race is elsewhere: `VoiceBackend` asks for the live worker on every call
  (`modelsvc/backends/voice_backend.py:7-8`, `:21-24`), so a swap that lands between one request's `ensure_voice_for` and its call
  (`modelsvc/backends/composite.py:135-138`) sends that call to the other model's child.
- `ModelsClient.speak(…, references=None, timeout=None)`: 1800 s with references (`inference/models_client.py:14-15`, `:84-108`).
- Films are the only `/speak` caller in `app`: one film at a time per process (`web/app.py:221-243`). Cancel bumps `feed_run`, frees the user at
  once, and the thread drops its result; the model call finishes in the background (`web/app.py:1268-1280`). A Prepare with no film building starts
  at once (`:1153-1158`), so a cancelled film's speak call can still run beside the next film's.
- The long-job pattern: run id + daemon thread + `{"stage", "fraction"}` + a polled fragment with a bar, "<stage> · N%" and a Cancel form
  (`web/app.py:1170-1226`, `:1228-1266`). `_Ticking` moves a bar on an estimate during one long call (`feed/build.py:106-160`);
  `CLONE_RTF_ESTIMATE = 1.0` (`:103`); `Cancelled` (`:163-165`); `_reason` reads the service's `detail` (`:799-813`).
- Page: `_voices_list.html:33-53` lists each example as language · seconds · source · date · ▶ `<audio>` · Delete; the example Delete form has no
  confirm (`:46-49`); the tip says "Only the newest example in each language is used." (`:68`). The recorder panel sits outside `#voices`
  (`voices.html:31-54`).
- `web/static/voices.js`: every action replaces `#voices` (`replaceList`, `:42-50`); while a row is deleting, the list is fetched again every 2 s
  (`pollWhileDeleting`, `:58-69`), but not while a text input in `#voices` has focus (`:53-56`); a form's `data-confirm` is asked before its POST
  (`:94-101`). Recorder state: `prepared`, `live`, `take` (`:105-108`). Esc: live → take → prepared → focused text field → close (`:370-381`).
- Routes: `web/voices_api.py:157-270`; `_refuse` maps `VoiceNameTaken`/`VoiceDeleting` to 409, else 400 (`:153-155`); add-example has the disk guard
  and `drain.wake()` (`:223-227`, `:243-246`). `register` runs inside `create_app` (`web/app.py:264`).
- Drain: the cascade collects file keys from `voice_examples` and `voice_conditionings` only (`feed/saved_voices.py:358-367`); the ref check reads
  those two tables only (`:404-408`). A new kind of voice file needs both.
- Schema: `SCHEMA_VERSION = 14` (`db/connection.py:8`); `_ADDED_COLUMNS` (`:62-70`); `migrate` runs `schema.sql` on every start (`:98-99`), so a new
  `CREATE TABLE IF NOT EXISTS` reaches an existing DB with no version bump (D39). `PRAGMA foreign_keys=ON` (`:31`).
- Test sentences: `scripts/voice_check.py:47-71` holds EN-1..5 and RU-1..5 (findings §10); no reading text uses them (§2.15 I). The wheel ships no
  `scripts` package (`pyproject.toml:75`), so app code cannot import it.
- User hunks near Phase 4 edits (`git diff -U0`, 2026-11-02): `web/app.py:1940-1957`, `:1961-1963` (`log_albums_to`, `app_factory`);
  `docs/design.md` §11 mermaid (`:1497-1520`). Phase 4 edits none of them.

### 1.8 Phase 5 facts (pinned source and working tree with phases 1-4, read 2026-11-02)

Pinned library: `voxcpm` 2.0.3 unpacked at `<base>/.scratch/deps-voxcpm/src/voxcpm-2.0.3/voxcpm/` (written `V/` below). Label `code finding` unless marked.

- LoRA in the package:
  - `LoRAConfig` defaults: `enable_lm=False, enable_dit=False, enable_proj=False, r=8, alpha=16, dropout=0.0`; targets `q_proj, v_proj, k_proj, o_proj`
    for the LM and the DiT (`V/model/voxcpm2.py:134-147`).
  - LoRA layers exist only when the model is BUILT with a config (`voxcpm2.py:246-247`; `_apply_lora` `:249-272`: base LM + residual LM, the
    DiT estimator, optional projections). No LoRA layer sits in the AudioVAE.
  - `VoxCPM(…, lora_config=None, lora_weights_path=None)` (`V/core.py:15-24`). A weights path without a config builds
    `LoRAConfig(enable_lm=True, enable_dit=True, enable_proj=False)` (`core.py:48-55`). `from_pretrained` passes both through (`core.py:104-115`, `:163-172`).
  - `VoxCPM.load_lora` raises unless the model was built with a `lora_config` (`core.py:317-335`). Our child builds with none (`audio/voxcpm_voice.py:172`).
  - `load_lora_weights(path)` reads `lora_weights.safetensors` / `.ckpt` in a folder, or one `.safetensors` / `.ckpt` / `.pth` file (`voxcpm2.py:1195-1227`).
    It copies only the keys the model has and returns `(loaded, skipped)`; a skipped key raises nothing (`:1233-1242`).
  - `reset_lora_weights` sets A to a kaiming draw and B to zeros (`voxcpm2.py:1249-1252`, `V/modules/layers/lora.py:67-71`). A zero B adds no
    adapter contribution (`lora.py:58-65`), but the two LoRA matmuls still run, also after `set_enabled(False)` (`lora.py:61-65`, `:73-76`); their
    cost is not measured. `reset_lora_weights` touches only `lora_A` / `lora_B`, never a base weight.
  - `load_lora_weights` copies into ANY parameter whose name matches, base weights included (`voxcpm2.py:1229-1241`: `dict(self.named_parameters())`,
    then `.data.copy_`). `copy_` broadcasts a shape that broadcasts and raises halfway through the loop on one that does not (pair t31,
    `<base>/.pair/files/merge2-plan-check-phase5-t31.md` R2).
  - `get_lora_state_dict` = every parameter whose name holds `lora_` (`voxcpm2.py:1254-1256`). This is the export.
- Training mode: `VoxCPM2Model.from_local(path, optimize, training, device, lora_config)` (`voxcpm2.py:1111-1182`). `training=True` skips the dtype cast
  (`:1143-1145`), freezes the AudioVAE and, with a LoRA config, every non-LoRA weight (`:1146-1153`), keeps the VAE in float32 (`:1154`), loads the
  checkpoint non-strict (`:1179`) and returns BEFORE `.to(device).eval().optimize()` (`:1180-1182`). So training is another model state, and the
  trainer moves the model to the device itself.
- `forward(text_tokens, text_mask, audio_feats, audio_mask, loss_mask, position_ids, labels, *, progress, sample_generate)` returns `loss/diff` and
  `loss/stop` (`voxcpm2.py:297-391`). Text ids come from `self.text_tokenizer(text)` (`:182`, `:487`).
- `build_prompt_cache` encodes the clip with the AudioVAE only (`voxcpm2.py:681-740`; `_encode_wav` `:396-422`). So a stored conditioning blob does
  not depend on an adapter.
- `voxcpm.training` exports `Accelerator`, `TrainingTracker`, `load_audio_text_datasets`, `HFVoxCPMDataset`, `build_dataloader`, `BatchProcessor`,
  `TrainingState`, `validate_manifest` (`V/training/__init__.py:9-29`):
  - `load_audio_text_datasets(train_manifest, val_manifest="", text_column="text", audio_column="audio", …, sample_rate=16_000, num_proc=1)` reads a
    JSON-lines manifest with HF `datasets` and casts `audio` to `Audio(sampling_rate)` (`training/data.py:19-61`).
  - `HFVoxCPMDataset` needs a `text_ids` column (`data.py:138-151`); nothing in the package makes it. `BatchProcessor(config, audio_vae, dataset_cnt,
    device)` packs a batch and VAE-encodes the audio under `no_grad` (`data.py:192-236`, `training/packers.py:51-68`); its keys are the `forward`
    arguments (`packers.py:221-232`). `build_dataloader(…, accelerator, batch_size, num_workers)` (`data.py:239-256`).
  - `Accelerator(amp=False, seed=42)` seeds torch, numpy and random (`training/accelerator.py:23-34`, `:53-59`). Its `device` is cuda, else mps,
    else the CPU, with no warning (`:106-112`). `prepare_model` moves the model there (`:84-92`).
  - `TrainingTracker(writer=None, …)` needs no TensorBoard (`training/tracker.py:19-33`). `validate_manifest` warns above 30 s ("may cause OOM")
    and below 0.3 s (`training/validate.py:189-195`).
  - Imports: `argbind`, `torch`, `datasets` (`data.py:4-6`), `argbind` and `yaml` (`training/config.py:3-4`), `soundfile` lazily (`validate.py:41`, `:62`).
- Not in the wheel: the trainer script and its configs. `voxcpm-2.0.3.dist-info/RECORD` lists 0 `scripts/` and 0 `conf/` paths; METADATA:574-580 names
  `scripts/train_voxcpm_finetune.py` and `conf/voxcpm_v2/voxcpm_finetune_lora.yaml`. So the package pins no learning rate, step count or batch size.
- `published` (METADATA): "With as little as **5–10 minutes** of audio, you can adapt to a specific speaker, language, or domain" (`:571`); LoRA is
  "recommended" (`:574`); SFT / LoRA ✅ for VoxCPM2 (`:359`); "~8 GB" VRAM for inference (`:362`). This is guidance from the package authors, not a promise.
- The pinned checkpoint config (`~/.cache/huggingface/hub/models--openbmb--VoxCPM2/snapshots/32279effe8c19989596f05d353d1447f51d9e915/config.json`,
  read-only, 2026-11-02): AudioVAE `sample_rate` 16000 and `out_sample_rate` 48000; LM hidden 2048, 28 layers, 16 heads, 2 KV heads, `kv_channels`
  128; residual LM 8 layers; DiT hidden 1024, 12 layers; `max_length` 8192; `dtype` bfloat16.
  Planner arithmetic, an estimate: r = 8 on q/k/v/o of both LMs and the DiT ≈ 4.5 M LoRA parameters ≈ 18 MB in float32. Slice A measures the file.
- Dependencies (`<base>/ivms777/uv.lock`): `datasets` 3.6.0 (`:1106`), `argbind` 0.3.9 (`:281`), `pyyaml` 6.0.3 (`:3731`), `soundfile` 0.14.0
  (`:4369`), `safetensors` 0.8.0 (`:4077`), `librosa` 1.0.0 (`:2055`) are locked already, as voxcpm dependencies (METADATA:22-44). `tensorboard` is not
  locked and is not needed. Phase 5 adds no package (§3 step 8).

ivms777:

- The child clones a clip WITH a transcript in `ref_continuation` mode and a clip without one in `reference` mode (`audio/voxcpm_voice.py:83-94`,
  `:269-275`); the blob records the mode and the transcript (`:97-115`, `:278-282`). `docs/models.md:128-139` documents it with a measured pace
  (6.58 against 4.90 syllables per voiced second). D16 still reads "reference mode for every example": the code and the docs differ from it.
- The voice child is built with no LoRA config (`audio/voxcpm_voice.py:172`). Its args are `(repo, device, revision)` (`modelsvc/slots.py:134-138`).
- `TorchWorker.call` holds `_lock` from `send` to `recv` (`modelsvc/torch_process.py:169-182`), and `stop` takes the same lock (`:184-186`). The child
  answers one message at a time (`:51-62`). So one long call blocks every other call to that child, and a `stop` waits for it.
- The governor evicts any resident outside the CURRENT call's `needed ∪ pinned` while it holds its lock (`modelsvc/governor.py:51-64`); it knows no
  other running op. `registry.unload` → `free` → `worker.stop` (`modelsvc/registry.py:54-59`, `modelsvc/slots.py:147-150`).
- mac: `gpu_concurrency` 3, `ram_budget_mb` 24000 (`config.py:42-44`). `Scheduler.reserve` holds a slot for a whole `with` block (`modelsvc/scheduler.py:85-105`).
  A speak waits in `_admit` with no time limit before its `acquire` (`modelsvc/backends/composite.py:118-138`, `modelsvc/scheduler.py:92-96`), so a
  speak that passed a check before a bake can load `voice` after the bake unloaded it (pair t31 R1). `POST /models/{name}/ensure` is open for every
  unit (`modelsvc/app.py:418-420` → `composite.py:277-281`). `_register_torch_unit` starts with `registry.unload(unit)` (`modelsvc/slots.py:129-132`).
- `datasets` 3.6.0 fingerprints every `map` transform that has no `new_fingerprint` with a recursive dill pickler, which pickles a closure over a
  model and turns each tensor into a CPU numpy copy (pair t31 W1: `datasets/fingerprint.py:430-435`, `datasets/utils/_dill.py:103`, `:161-176`).
  A risk read from the code, not seen in a run.
- `CompositeBackend.model_unload` calls `registry.unload` without the governor (`modelsvc/backends/composite.py:283-285`); `voice_check` calls it
  (`scripts/voice_check.py:266`, `:289`).
- The models service shares `data_dir` with `app`: compose mounts `ivms777-data:/data` on all three services (`compose.yaml:18`, `:30`, `:41`); on mac
  `make` exports the same `IVMS777_DATA_DIR` (`Makefile:82`, `:89`). The models image copies `config.py`, `modelsvc`, `audio`, `models/catalog.py`
  and `storage/wav.py`, and no other `storage` module (`Dockerfile.models:36-47`).
- `make up` reloads the models service on any change in `modelsvc audio models embedding captioning inference` (`ivms777/CLAUDE.md`, "Everything
  hot-reloads"). A reload ends every child, a training child included.
- Publication barriers today: `run_preview` checks `cancelled()` and calls `save_preview` while the job lock is held (`feed/voice_preview.py:185-200`);
  a saved film runs its check, its WAV writes and its ready row in one `BEGIN IMMEDIATE` with `_deleted_voice` (`feed/build.py:560-577`, `:637-646`).
  A native bfcache restore runs no script again; `organize.html:64-68` mounts on `pageshow` with `e.persisted`.
- Store: `add_example` stores the selected reading text as `transcript` for a recording and `None` for an upload (`feed/saved_voices.py:579-587`); no
  route edits a transcript. `film_example` is the one example resolver (`:654-670`); `reference_for` builds the `/speak` reference (`:850-859`). The
  drain's cascade and ref check read `voice_examples`, `voice_conditionings` and `voice_previews` (`:381-393`, `:430-435`).
- Page: one leaf at `/organize/voices` with 14 routes (`web/voices_api.py:323-507`), a × back to `/organize` (`web/templates/voices.html:4-12`), linked
  from Organize (`web/templates/organize.html:23-25`). The text "Organize → Saved voices" is in `web/app.py:1035-1039`, `web/templates/feed.html:52`
  and `feed/build.py:607-616`.
- Shell: four boosted nav links that swap only `<main>` (`web/templates/base.html:11-20`); Settings is a button (`:27-30`). The vendored htmx is
  `version:"2.0.3"` and has the events `htmx:confirm`, `htmx:beforeHistorySave` and `htmx:historyRestore` (`grep -o` over `web/static/htmx.min.js`).
  Its history restore runs the page scripts again (pair t30, `<base>/.pair/files/merge2-voices-menu-t30.md`). `voices.js` checks `#voices-page` once,
  captures nodes at load (`web/static/voices.js:13-23`) and adds document listeners (`:112`, `:354`, `:366`, `:738`, `:771`, `:774`). The once-guard
  pattern is `web/templates/organize.html:64-66`.
- Tests that name the old page or its texts: `tests/test_web_voices.py` (96 lines hold `/organize/voices`), `tests/test_web_feed.py`,
  `tests/test_feed_build.py`, `tests/test_saved_voices.py`.

### 1.9 Slice B rework facts (working tree with Slices A and V implemented and green, read 2026-11-02)

Label `code finding` unless marked. Line numbers are those of the uncommitted working tree in `<base>/ivms777`.

The one-take path that the rework removes:

- `film_example` is the one take resolver: the user's pin in `voice_film_examples`, else the newest take (`feed/saved_voices.py:654-670`;
  `example_for` `:640-651`). `set_film_example` writes the pin (`:673-708`). `castable` pairs each voice with that take (`:711-721`).
  `reference_for` builds the one-take `/speak` reference: audio, sha256, language, transcript and every stored conditioning (`:850-859`).
- The build reads the narrator and the saved cast (`feed/build.py:268-284`), pairs each used token with its take (`:403-407`), reads the
  references and speaks inside one `voice_turn` (`:411-447`), stores each built conditioning (`_keep_conditioning`, `:620-633`) and records
  `decisions["voice"]` (`:455-465`). The ready save checks every used voice AND its take in one `BEGIN IMMEDIATE` (`:557-581`;
  `_deleted_voice` `:636-645`). `_saved_cast` holds the failure texts of a deleted or pending pinned take and of "no <Language> example"
  (`:584-617`).
- The `/feed` picker lists only castable voices (`web/app.py:944-995`); a stored choice that cannot narrate is a selected "unavailable"
  option with a take label (`:982-992`). `_store_narrator` answers 409 with the same take texts (`:1029-1038`). The template renders the
  saved group and the line "Add a voice under Voices." (`web/templates/feed.html:22-56`). "This film" reads `film_narrator`
  (`feed/store.py:108`; template `web/templates/feed.html:98-102`).
- The test is per take: `PreviewJob(voice_id, example_id, language, text, revision)` (`feed/voice_preview.py:76-82`); `preview_text` kinds
  `reading`, `sentence`, `custom` (`:89-109`; `PREVIEW_SENTENCES` `:43-64`; `CUSTOM_TEXT_MAX = 300` `:65`); `run_preview` sends the take's
  reference (`:155`), stores a built conditioning (`:177-183`) and stores the test under the job lock (`:185-200`). `voice_previews` is keyed by
  `example_id` (`db/schema.sql:254-268`; store `feed/saved_voices.py:874-942`).
- The leaf: the header lines "<Language>: films use take #N (newest | your choice)" and the tombstone line (`web/templates/_voice_body.html:35-42`);
  a Test button and the pinned-take Delete confirm on every take row (`:78-86`); the tip "Films use one example per language: the newest,
  unless you choose one on the Voice model tab." (`:107`); the Voice model tab's film-take select and tombstone (`:118-149`); the Test tab's
  take select and its per-take `<template class="take-compare">` (`:152-188`). The card's "Films: take #…" lines
  (`web/templates/_voice_cards.html:18-31`). The panel's text select with the "reading" option, outside `#voices`
  (`web/templates/voice.html:44-64`). The job fragment names an example (`web/templates/_voice_preview.html:6-21`).
- `web/voices_api.py`: `_film` (`:287-298`) feeds the cards and the leaf; `_leaf_context` passes `previews` and `revisions` (`:318-342`); the
  routes `POST /voices/{id}/examples/{eid}/preview` (`:424-457`), `POST /voices/{id}/film-example` (`:459-474`) and
  `GET /voices/{id}/examples/{eid}/preview/audio` (`:476-490`); the test job `VoicePreviews` with `epoch` and `completed` (`:157-271`).
- `web/static/voices.js`: `carryInputs` keeps `.voice-language` and `.test-take` across a swap (`:430-431`); `testTarget` reads `.test-take`
  (`:790-799`); `syncTest` binds the text options to the take (`:811-838`); `selectTake` (`:840-845`); `generatePreview` posts the per-take route
  (`:952-963`); a mounted leaf selects the running test's take (`:995-998`); the `.example-preview` click (`:1070-1071`) and the
  `.film-take-select` change (`:1079`).
- The check script takes the prompt take of A and B from `film_example` and `reference_for` (`scripts/voice_bake_check.py:32-40`, `:239-244`);
  C is that reference with `use_clip=False` (`:257-259`). `THRESHOLDS` already states the C-only pass rule and the listen-first rule (`:59-65`).
  It tags its bake `{"check": <name>}` (`:357`), follows it to any end and deletes it in `finally` (`:192-227`, delete `:219-224`). A bake that
  another caller cancels ends its run with exit 1 (`:369-370`).

The `/speak` contract:

- `VoiceReference.audio` and `sha256` are required (`modelsvc/app.py:80-92`). `_decode_references` decodes and checks the audio of every
  reference (`:309-313`), also of one with `use_clip=False` (`:324-329`). The client always sends `audio` and `sha256`
  (`inference/models_client.py:41-57`).
- The voice child skips the clip of a `use_clip=False` token: no cache, no conditioning entry (`audio/voxcpm_voice.py:283-285`), and it records
  the adapter state per token (`:294-296`). So that audio is decoded and checked, then not used.
- `make up` reloads the models service on a change under `modelsvc audio models embedding captioning inference` (`ivms777/CLAUDE.md:206-216`);
  a reload ends a running bake (§1.8). §6.5 needs a models service with the uncommitted Slice A code, which exists only in `<base>/ivms777`.

Storage:

- No Slice B table exists yet: `db/schema.sql` has no `voice_models`, `voice_model_takes`, `voice_bake_runs` or `voice_tests`, and
  `voice_examples` has no `transcript_confirmed_at` or `kit_id` (`:218-233`). `_ADDED_COLUMNS` ends with `memory_narrators.preset_speaker`
  (`db/connection.py:62-70`); `SCHEMA_VERSION = 14` (`:8`).
- Phase 4 and Slice V write `voice_film_examples` (`db/schema.sql:243-253`), `voice_previews` (`:254-268`) and `voice_conditionings` (`:235-242`).
  The drain collects the files of `voice_examples`, `voice_conditionings` and `voice_previews` (`feed/saved_voices.py:381-393`), and its ref
  check reads the same three tables (`:430-435`).

The Slice A service facts the rework uses:

- `VoiceGate` counts speaks and manual `voice` ensures in (`modelsvc/voice_bakes.py:77-126`; `CompositeBackend.model_ensure`
  `modelsvc/backends/composite.py:290-298`, t32). A `voice_train` ensure always raises `BakeOnly` (`:291-292`, t35).
- `_finish` publishes the end, opens the gate and clears `active` under one lock (`modelsvc/voice_bakes.py:429-447`, t34). `start` refuses a second
  bake with `BakeRunning` (`:216-218`) and removes the previous ended bake's folder (`_prepare`, `:234-238`).
- `_wav_seconds` refuses a take with no frames or a cut payload (`:493-509`, t37). The trainer saves nothing after a Cancel in the last step
  (`audio/voxcpm_trainer.py:139-140`, `:319-320`, t33).

### 1.10 ASR amendment facts (added 2026-11-02, ASR, t42; working tree with lanes 30 and 31 written, read 2026-11-02)

Label `code finding` unless marked. No weight file was downloaded: the Hub facts come from the read-only API
(`https://huggingface.co/api/models/<repo>?blobs=true`) and the raw `config.json`, `preprocessor_config.json` and `README.md` at the revision.

The ASR model (`published`, read 2026-11-02):

- `openai/whisper-large-v3-turbo`: `sha` `41f01f3fe87f28c78e2fbf8b568835947dd65ed9` (lastModified 2024-10-04), `gated: false`, `private: false`;
  `cardData.license` `mit`, tag `license:mit`, README `license: mit` (`:102`); `pipeline_tag` `automatic-speech-recognition`, `library_name`
  `transformers`; `cardData.language` lists 99 codes, `en` and `ru` among them.
- Size: `safetensors.total` 808,878,080 parameters (`F16`); one weight file, `model.safetensors`, 1,617,824,864 bytes; all 13 files
  1,622,466,054 bytes.
- `config.json`: `encoder_layers` 32, `decoder_layers` 4, `d_model` 1280, `num_mel_bins` 128, `max_target_positions` 448, `torch_dtype`
  `float16`. `preprocessor_config.json`: `sampling_rate` 16000, `chunk_length` 30, `n_samples` 480000.
- README `:124-125`: "a finetuned version of a pruned Whisper large-v3 … the number of decoding layers have reduced from 32 to 4 … way
  faster, at the expense of a minor quality degradation"; `:306`: "Whisper has a receptive field of 30-seconds".
- The other candidates, same API: `openai/whisper-large-v3` `06f233fe06e710322aca913c1bc4249a0d71fce1`, `apache-2.0`, 1,543,490,560
  parameters; `openai/whisper-medium` `abdf7c39ab9d0397620ccaea8974cc764cd0953e`, `apache-2.0`, 763,857,920; `openai/whisper-small`
  `973afd24965f72e36ca33b3055d56a652f456b4d`, `apache-2.0`, 241,734,912. None is gated.
- Planner arithmetic, an estimate: float32 weights = 808,878,080 × 4 B = 3,235,512,320 B ≈ 3086 MiB; float16 ≈ 1543 MiB. Not measured.
- Not on this Mac yet: `ls ~/.cache/huggingface/hub/` (2026-11-02) lists no `models--openai--whisper-*` folder.

The library, at the lock (no package is added):

- `transformers` 5.15.0 (`ivms777/uv.lock:4830-4831`; the `models` extra, `pyproject.toml:31-32`) holds `models/whisper/`:
  `WhisperForConditionalGeneration` (`modeling_whisper.py:963`); `WhisperFeatureExtractor.__call__(raw_speech, …, sampling_rate=None, …)`
  (`feature_extraction_whisper.py:193-206`) refuses another rate (`:266-270`); `generate(…, task=None, language=None, …)`
  (`generation_whisper.py:383-393`) takes a two-letter code (`:1466-1477`; `tokenization_whisper.py:41`, `:45`: `en`, `ru`). `from_pretrained`
  takes `dtype` (`torch_dtype` is deprecated, `modeling_utils.py:1492`); the repo loads with `dtype=…, device_map=device`
  (`embedding/siglip.py:27-29`, the host-copy note `:22-26`).
- `scipy` 1.18.0 (`uv.lock:4152-4153`) is locked as a dependency of `librosa` 1.0.0 (`uv.lock:2055-2070`), which `voxcpm` needs; it is
  installed in `<base>/ivms777/.venv`. 48000 / 16000 = 3, so `scipy.signal.resample_poly(x, 1, 3)` resamples a take exactly.

Where the words of a take are read today:

- The service bake: one manifest line `{"audio", "text"}` per take (`modelsvc/voice_bakes.py:247`); `dataset_sha256` hashes the sorted
  `"<sha256>\t<language>\t<text>"` lines (`:252-256`); a text outside 1-1000 characters after strip → 422 (`:474-477`); the trainer
  tokenizes each text (`audio/voxcpm_trainer.py:285`). So a take with no words cannot be sent.
- The app store (lane 31, uncommitted): `add_example` stores the kit text of `prompt_id` (else the first text) as `transcript` and `kit_id`,
  and confirms a posted `transcript` (`feed/saved_voices.py:748-832`); `set_transcript` (`:835-872`); `VoiceExample.transcript_confirmed_at`,
  `.kit_id`, `.unresolved` (`:252-271`); `TRANSCRIPT_MAX = 1000` (`:229`); `_fingerprint` and `snapshot_hash` hash the transcript
  (`:1261-1268`); `BakeSnapshot.unresolved` (`:1246-1251`, `:1282-1295`); `ModelChanges(added, deleted, edited, unresolved)` (`:1210-1214`)
  and `voice_model_for` (`:1389-1420`); `BakeRun(voice_id, takes, dataset_hash, cancel_requested_at)` (`:1254-1258`, `:1325-1335`).
  `grep -rn 'set_transcript\|TRANSCRIPT_MAX\|\.unresolved\|transcript_confirmed_at\|transcript='` over the repo (no `.venv`) finds these
  names only in `feed/saved_voices.py`, `db/schema.sql`, `db/connection.py:70` and `tests/test_saved_voices.py`; its `transcript=` hits in
  `tests/test_voxcpm_voice.py` and `tests/test_modelsvc_api.py` are the `/speak` reference field.
- The per-take test of Phase 4 (still in the code): `preview_text` reads a recorded take's transcript (`feed/voice_preview.py:92-95`);
  `reference_for` sends it (`feed/saved_voices.py:1089-1096`).
- The check script labels every stored transcript "reading text, not checked" (`scripts/voice_bake_check.py:79`, `:132-135`).
- The add route posts `prompt_id or None` and no transcript (`web/voices_api.py:520`, `:544`).

The user's database (read-only, `sqlite3 "file:$HOME/.ivms777/ivms777.db?mode=ro"`, 2026-11-02):

- `voice_examples` already has `transcript_confirmed_at` and `kit_id` (lane 31's `_ADDED_COLUMNS` ran under `--reload`), and the tables
  `voice_bake_runs`, `voice_models`, `voice_model_takes` and `voice_tests` exist. It holds 5 takes, all recorded, all with a stored
  transcript (the shown text, t42), from 10.56 to 27.78 s.

Take length and the models service:

- The store refuses a take outside 5-30 s (`feed/saved_voices.py:225`, `:735-745`); the recorder stops at 30 s (`web/static/voices.js:18`,
  `:658`, `:727-730`). So every take fits Whisper's 30 s window in one pass.
- `SlotManager.apply` registers `voice_train` once, after the slots (`modelsvc/slots.py:93-126`, `:255-268`); `_child_args` gives
  `(repo, device, revision)` for a pinned entry with no adapter (`:137-147`). A bake unloads `voice` after the drain (`modelsvc/voice_bakes.py:325-341`).
  `model_ensure` counts only `voice` in the gate (`modelsvc/backends/composite.py:290-298`). `ModelRegistry.unload` of a unit that is not
  loaded returns at once (`modelsvc/registry.py:54-59`). `governor.state().free_mb` is measured (`modelsvc/governor.py:39-53`, `:105-112`).
- `tests/test_catalog.py:33-35` requires every `catalog.CATALOG` entry's slot to be in `SLOTS`; `catalog_payload` and `models/slots.py:65`
  iterate `SLOTS`; `scripts/fetch_models.py:25-48` fetches the entries of the selected slots (every entry of an automatic slot) only.
- `inference/models_client.py` raises on any non-2xx (`raise_for_status`, `:253-261` for `bake_start`).

## 2. Design

### 2.1 The flow of one film narrated by a saved voice

1. The user records or uploads an example on `/organize/voices`. The app stores a mono PCM-16 WAV in `data_dir/voices` and one `voice_examples` row.
2. On `/feed/{key}`, the user picks the saved voice as narrator and presses Prepare or ↻ Regenerate. The app stores `memory_narrators(group_id, language, voice_id)` and starts the build.
3. `feed/build.py` reads the choice. It shows the story pass the saved voices of that language, not the presets. Every passage is a saved voice.
4. `feed/build.py` calls `POST /speak` with `references`: per voice token, the example WAV, its sha256, and every stored conditioning blob of that example.
5. The models service routes to the cloning entry for the language (VoxCPM2). Preset routing never picks a cloning entry, and a stored-voice request never reaches a preset child.
6. The child restores a blob when its model key, revision, `voxcpm` version and clip sha256 all match. Else it builds the conditioning and returns the new blob.
7. The gateway checks the reply identity (catalog key + revision). The app stores each new blob in `voice_conditionings` and records the voice decisions with the film.

### 2.2 Catalog and routing — `models/catalog.py`, `modelsvc/slots.py`

`models/catalog.py`:

- `HfSource.revision: str | None = None` — a pinned Hub commit. `None` = the repo's `main`, as today.
- `ModelEntry.clones: bool = False` — the entry speaks SAVED voices from a reference clip. It declares no `speakers`.
- New entry, after `silero-v4-ru`:

  ```python
  ModelEntry(
      key="voxcpm2", slot="voice", display="VoxCPM2 · voice copy",
      source=HfSource("openbmb/VoxCPM2", revision="32279effe8c19989596f05d353d1447f51d9e915"),
      size_mb=4728, cost_mb=10500, cost_measured=False,
      profiles=("mac", "cloud"), languages=("en", "ru"), clones=True,
      note="Voice copy (§4.1, §11.1): speaks a SAVED voice from its example clip; no preset speakers. "
           "2.29B params, Apache-2.0 code and weights, 48 kHz mono. Pinned to one Hub commit; loaded with "
           "load_denoiser=False and no timestamps extra. MPS runs it in float32 (the package forces it), so "
           "the cost is an ESTIMATE: 2,290,004,544 x 4 B + the 377 MB AudioVAE + runtime. Measure it and pin it "
           "with IVMS777_MODEL_COST_MB.",
  )
  ```

- `cloning_entry_for(profile: str, language: str) -> ModelEntry | None` and `cloning_languages(profile: str) -> tuple[str, ...]` (in `feed.language.LANGUAGES` order).
- `DEFAULTS` and `AUTOMATIC_SLOTS` do not change. Kokoro stays the voice default on every profile.

`modelsvc/slots.py`:

- `VOICE_TARGETS["voxcpm2"] = "audio.voxcpm_voice:VoxCPMVoice"`.
- `voice_entry_for(language, *, cloning=False)`: it considers only entries with `entry.clones == cloning`. Selected entry first, then `entries_for` order.
- `ensure_voice_for(language, *, cloning=False) -> bool`: same swap as today, with the same filter.
- `_register_voice`: `VOICE_TARGETS[key]`. An unknown key raises `KeyError`. No Kokoro fallback.
- `_register_torch_unit`: child args = `(child_arg, device)`, plus `(revision,)` when the entry is an `HfSource` with a revision. Unpinned entries keep today's args.

`modelsvc/downloads.py` and `scripts/fetch_models.py` (revision aware):

- Downloads: `_hf_snapshot`, `_hf_present`, `_hf_total_bytes` and `_hf_fetch` take `revision`. `_source_id` = `hf:<repo>@<revision>` for a pinned entry.
  Unpinned entries keep today's calls, so the existing tests and fakes stay valid.
- `fetch_models.wanted` lists a pinned entry as `"<repo>@<revision>"`. `_fetch` splits on `@` and calls `snapshot_download(repo, revision=revision)`. A Hub repo id has no `@`.
- Effect: on mac, `make models` also pre-fetches VoxCPM2 at the pinned commit (≈ 4.7 GB). Jetson lists no VoxCPM2.

### 2.3 `POST /speak` contract — `modelsvc/app.py`, `inference/models_client.py`

```python
class VoiceConditioning(BaseModel):
    model: str            # catalog key that made the blob, e.g. "voxcpm2"
    revision: str         # Hub commit that made it
    blob: str             # base64; opaque to everyone except the child that made it

class VoiceReference(BaseModel):
    audio: str            # base64 PCM-16 WAV of ONE saved example
    sha256: str           # hex sha256 of the decoded WAV bytes
    language: str         # the example's language
    transcript: str | None = None              # the fixed prompt for a recorded example; unused by VoxCPM2 reference mode
    conditioning: list[VoiceConditioning] = []

class SpeakRequest(BaseModel):   # existing fields unchanged
    ...
    references: dict[str, VoiceReference] | None = None   # voice token ("saved:<voice_id>") -> example

class AudioResponse(BaseModel):  # existing fields unchanged
    ...
    revision: str | None = None                # the loaded Hub commit; None for Kokoro, Silero, music
    conditioning: dict[str, dict] = {}         # token -> {state: "built"|"restored", model, revision, reason, blob: base64|None}
```

- With `references`: `voice` is required and must be a key of `references`. Every non-empty name in `voices` must be a key too.
  A request never mixes saved voices and preset speakers. At most 8 references. Each decoded WAV ≤ 4 MiB. The decoded sha256 must match `sha256`.
  Any breach → **422** with the reason.
- The parent decodes base64 to bytes before it calls the backend. It passes `references=` to `backend.speak` only when the request carries them,
  so every existing backend fake keeps its signature.
- `CloningUnavailable(SlotOff)` (new, `modelsvc/backends/base.py`) → **503** `voice copy is not available: <reason>`. Other `SlotOff` → 503 as today. Other errors → 500 with the model's message, as today.
- `_audio_response` copies `revision` and `conditioning` and base64-encodes each blob. A `restored` entry has `blob: None`.
- `ModelsClient.speak(text, language, voice, segments, voices, *, references=None, timeout=None)`:
  `references` = `{token: {"audio": bytes, "sha256", "language", "transcript", "conditioning": [{"model", "revision", "blob": bytes}]}}`.
  The client encodes base64. `timeout=None` → 1800 s with references, else 300 s. The reply dict gains `revision` and `conditioning` (blobs as bytes).

### 2.4 Gateway — `modelsvc/backends/{base,composite,voice_backend,fake}.py`

- `ModelBackend.speak(…, references: dict | None = None)` in the protocol.
- `CompositeBackend.speak(text, language, voice, segments, voices, references=None)`:
  - Without references: today's path. It calls `ensure_voice_for(language)` (presets only).
  - With references: `ensure_voice_for(language, cloning=True)`. False → `CloningUnavailable(f"no voice-copy model speaks {language} on {profile}")`.
    It runs on the `voice` unit at `Priority.BATCH`, label `"narrating"`, and passes `references` to the voice backend.
  - Identity check after a reference call: the reply `model` must equal the routed entry key, and `revision` must equal `entry.source.revision`. Else `RuntimeError("voice identity mismatch: …")`.
  - With no slot manager (bare unit tests), references pass straight to the voice backend.
- `VoiceBackend.speak(text, voice=None, segments=None, voices=None, references=None)`: it sends `references` as a fifth argument only when it is not `None`.
  Kokoro and Silero do not change.
- `FakeBackend.speak`: preset language swap skips cloning entries. With references it returns a real WAV, `model="fake-voice"`, `revision=None`, and a `built` conditioning per token.

### 2.5 The VoxCPM2 child — `audio/voxcpm_voice.py` (new)

Models-service-only. Module scope imports no torch and no voxcpm (pattern `audio/silero_voice.py:72-83`), so its pure helpers are testable.

- `VoxCPMVoice(repo: str, device: str, revision: str)` — `TorchWorker` builds it with the args from §2.2.
  1. Refuse a device that is not `"mps"` and does not start with `"cuda"` (GPU only, `docs/design.md:118-121`).
  2. `path = huggingface_hub.snapshot_download(repo, revision=revision, local_files_only=True)`.
     Weights missing → `RuntimeError("VoxCPM2 weights for <revision> are not on disk: run `make models`, or POST /models/download {slot: voice, key: voxcpm2}")`.
  3. Verify `Path(path).name == revision` (the snapshot folder is the commit). Else raise. Keep it as `self.revision`.
  4. `VoxCPM.from_pretrained(path, load_denoiser=False, optimize=False, device=device)`; `self._tts = model.tts_model`; `self.sample_rate = int(self._tts.sample_rate)`.
  5. Never set `VOXCPM_MPS_DTYPE` or `PYTORCH_ENABLE_MPS_FALLBACK`.
- `warm()`: one zero-shot `generate_with_prompt_cache(target_text="Ready.", prompt_cache=None, max_len=64, retry_badcase=False)`. Resident must mean resident (§8.1).
- `speak(text, voice, segments, voices, references) -> dict`:
  - `check_references(voice, voices, references)` (pure): `voice` and every non-empty `voices[i]` are keys of `references`. Else `ValueError`.
  - Per used token: `cache, state, blob, reason = self._conditioning(token, references[token])`. <!-- pragma: allowlist secret -->
  - `speak_one(passage, token)`: normalise whitespace (as `core.py:262-263`); split with `audio.silero_voice.split_text` (≤ 500 characters, at sentences);
    per piece `wav, _, _ = self._tts.generate_with_prompt_cache(target_text=piece, prompt_cache=cache, **CONTROLS)`; concatenate `wav.squeeze(0).cpu().numpy()`.
  - `CONTROLS = {"min_len": 2, "max_len": 4096, "inference_timesteps": 10, "cfg_value": 2.0, "retry_badcase": True, "retry_badcase_max_times": 3, "retry_badcase_ratio_threshold": 6.0}`
    — the package's own `generate` defaults (`core.py:180-195`). No seed (2.0.3 has none).
  - `speak_passages(chunks, voices, voice, speak_one, self.sample_rate)` (`audio/segments.py:40-86`).
  - Returns `{"audio": encode_wav(...), "sample_rate", "model": "voxcpm2", "revision": self.revision, "segments", "conditioning": {token: {"state", "model", "revision", "reason", "blob"}}}`.
- `_conditioning(token, ref)`:
  - `pick_conditioning(ref, "voxcpm2", self.revision)` (pure) chooses the stored blob for this key + revision.
  - `load(blob)` = `torch.load(BytesIO(blob), map_location="cpu", weights_only=True)`. The payload must carry
    `{"format": 1, "model": "voxcpm2", "revision", "voxcpm": importlib.metadata.version("voxcpm"), "sha256": ref sha256, "mode": "reference", "cache": {...}}`.
    All fields match → `("restored", blob None)`.
  - Else build: write `ref["audio"]` to a temporary `.wav`, call `self._tts.build_prompt_cache(reference_wav_path=tmp)`, delete the file,
    `blob = save(payload)` (`torch.save` into `BytesIO`) → `("built", blob, reason)`. Reasons: `no stored conditioning`, `stored for another revision`, `voxcpm version changed`, `clip changed`.
  - `save` and `load` are injectable keyword arguments of a test constructor (`VoxCPMVoice.loaded(tts, revision, *, save, load)`), so tests use pickle fakes and no torch.

### 2.6 Data model — `db/schema.sql`, `db/connection.py`, `config.py`, `web/deps.py`

New tables (`CREATE TABLE IF NOT EXISTS`; `migrate` creates them on every start, `db/connection.py:72-102`). `SCHEMA_VERSION` 13 → 14.

```sql
-- A SAVED VOICE (design §6, §11.1): the name the user gave a person's voice. User data — never a catalog Speaker.
CREATE TABLE IF NOT EXISTS saved_voices (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,   -- never reused: a delete intent, a stale tab or a narrator id never hits a new voice
  owner_id   INTEGER NOT NULL,
  name       TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(owner_id, name)
);
-- One example clip of a saved voice, in ONE language. The model-independent source of truth.
CREATE TABLE IF NOT EXISTS voice_examples (
  id           INTEGER PRIMARY KEY AUTOINCREMENT, -- never reused, as saved_voices.id
  voice_id     INTEGER NOT NULL REFERENCES saved_voices(id) ON DELETE CASCADE,
  language     TEXT NOT NULL,          -- feed.language.LANGUAGES
  source       TEXT NOT NULL,          -- 'recorded' | 'uploaded'
  transcript   TEXT,                   -- the fixed prompt when recorded; NULL when uploaded
  storage_key  TEXT NOT NULL,          -- PCM-16 mono 48 kHz WAV in data_dir/voices (content-addressed)
  content_hash TEXT NOT NULL,          -- sha256 of that WAV
  seconds      REAL NOT NULL,
  sample_rate  INTEGER NOT NULL,
  consent_at   TEXT NOT NULL,          -- when the user confirmed the consent statement
  created_at   TEXT NOT NULL,
  UNIQUE(voice_id, content_hash)
);
CREATE INDEX IF NOT EXISTS voice_examples_by_voice ON voice_examples(voice_id, language);
-- What a cloning model built from one example: an opaque blob, valid only for that model + revision. A cache — rebuilt when it does not match.
CREATE TABLE IF NOT EXISTS voice_conditionings (
  example_id  INTEGER NOT NULL REFERENCES voice_examples(id) ON DELETE CASCADE,
  model_key   TEXT NOT NULL,
  revision    TEXT NOT NULL,
  storage_key TEXT NOT NULL,           -- blob in data_dir/voices
  created_at  TEXT NOT NULL,
  PRIMARY KEY (example_id, model_key, revision)
);
-- Who narrates one memory's film in one language, when the user chose a saved voice. No row = Automatic.
-- voice_id NULL = the chosen saved voice was deleted: the film fails until the user picks again (D33).
CREATE TABLE IF NOT EXISTS memory_narrators (
  group_id INTEGER NOT NULL REFERENCES groups(id) ON DELETE CASCADE,
  language TEXT NOT NULL,
  voice_id INTEGER REFERENCES saved_voices(id) ON DELETE SET NULL,
  PRIMARY KEY (group_id, language)
);
-- Intent to delete a saved voice or ONE of its examples (design §6; the §3.2c outbox pattern). The request only inserts here.
-- No foreign key: the intent outlives its target and ends only after the files are gone (D33).
CREATE TABLE IF NOT EXISTS voice_deletions (
  id           INTEGER PRIMARY KEY,
  owner_id     INTEGER NOT NULL,
  kind         TEXT NOT NULL CHECK (kind IN ('voice', 'example')),
  target_id    INTEGER NOT NULL,       -- saved_voices.id or voice_examples.id
  requested_at TEXT NOT NULL,
  cascaded_at  TEXT,                   -- set in the transaction that deletes the rows; NULL = the rows are still there
  UNIQUE(owner_id, kind, target_id)
);
-- A file in data_dir/voices that may have lost its last row, or may never get one: an example WAV or a conditioning blob.
-- The drain removes it only when no row references it, then deletes every row of that key (the "ack").
CREATE TABLE IF NOT EXISTS voice_file_deletions (
  id           INTEGER PRIMARY KEY,
  storage_key  TEXT NOT NULL,
  deletion_id  INTEGER REFERENCES voice_deletions(id),  -- NULL = the write-ahead row of a file write (§2.7), or a replaced conditioning blob
  requested_at TEXT NOT NULL,
  UNIQUE(storage_key, deletion_id)
);
CREATE INDEX IF NOT EXISTS voice_file_deletions_by_deletion ON voice_file_deletions(deletion_id);
```

- Files: `Settings.voices_dir = data_dir / "voices"` (new property next to `audio_dir`, `config.py:295-298`). `AppContext.voices = LocalStorage(settings.voices_dir)` (`web/deps.py`).
  The worker gets it too, because it builds the same context (`ingest/cli.py:15`). The film's `prune_orphan_audio` never sees this store (`feed/store.py:87-105` walks `data_dir/audio` only).
- Keys: example WAV = `content_key(sha256, ".wav")`; blob = `content_key(sha256(blob), ".cond")` (`storage/keys.py`).
  Two voices may hold identical bytes, and a deleted clip may be added again. So only the drain removes a voice file, and only when no row references its key (§2.7, D33).
- The foreign key from `voice_file_deletions` keeps an intent alive until its last key is acknowledged (`PRAGMA foreign_keys=ON`, `db/connection.py:31`).
- Ids: a plain `INTEGER PRIMARY KEY` reuses the highest id after a delete (§1.3). A reused voice id would meet its old, cascaded but unfinished intent:
  `ON CONFLICT DO NOTHING` then swallows the new delete. So `saved_voices` and `voice_examples` use `AUTOINCREMENT`.
- A memory rebuild recycles group ids (`db/schema.sql:125-128`), so it drops the memory's narrator choice together with its films.

### 2.7 Store API — `feed/saved_voices.py` (new, thin client: SQL + storage, no model)

**[Superseded 2026-11-02 by §2.15 I]** the one-text `RECORDING_PROMPTS` below and its `transcript` rule: Phase 3 holds four reading texts per language.
These two texts stay, as the first text of each language.

```python
RECORDING_PROMPTS: dict[str, str] = {
    "en": "The quick morning light crept across the kitchen table. I poured a cup of coffee, opened the window, "
          "and listened to the city slowly waking up below.",
    "ru": "Утренний свет медленно скользил по кухонному столу. За окном понемногу просыпался город, "
          "а в комнате пахло свежим кофе и тёплым хлебом.",
}
MIN_SECONDS, MAX_SECONDS, MAX_BYTES, NAME_MAX = 5.0, 30.0, 4 * 1024 * 1024, 80
SILENCE_RMS = 0.00316          # -50 dBFS
SAMPLE_RATE = 48000            # D21: the only rate the browser and afconvert make
CONSENT = "This is my voice, or I have this person's permission to copy it."

class VoiceError(ValueError): ...            # message shown to the user
class VoiceDeleting(VoiceError): ...         # the target is being deleted; the route answers 409
@dataclass(frozen=True) class VoiceExample: id, voice_id, language, source, transcript, storage_key, content_hash, seconds, sample_rate, created_at, deleting: bool
@dataclass(frozen=True) class SavedVoice: id, name, examples: tuple[VoiceExample, ...], deleting: bool
@dataclass(frozen=True) class NarratorChoice: voice_id: int | None   # None = the chosen saved voice was deleted

def token(voice_id: int) -> str                      # "saved:<id>"
def list_voices(conn, owner_id) -> list[SavedVoice]                  # pending voices and examples too, with deleting=True
def get_voice(conn, owner_id, voice_id) -> SavedVoice | None
def create_voice(conn, owner_id, name) -> SavedVoice                 # strip; 1..NAME_MAX; unique per owner by casefold(); a pending voice's name → VoiceDeleting
def rename_voice(conn, owner_id, voice_id, name) -> SavedVoice       # VoiceDeleting when pending
def request_voice_deletion(conn, owner_id, voice_id) -> bool         # enqueue only; False = unknown or foreign
def request_example_deletion(conn, owner_id, voice_id, example_id) -> bool
def process_voice_deletions(conn, storage, owner_id) -> int          # the drain; returns rows still pending (0 = done)
def pending_voice_deletions(conn) -> int                             # rows in voice_deletions + voice_file_deletions
def add_example(conn, storage, owner_id, voice_id, *, language, source, data: bytes, consent: bool) -> VoiceExample
def read_example_audio(storage, example) -> bytes
def example_for(conn, owner_id, voice_id, language) -> VoiceExample | None     # the NEWEST example in that language that is not pending
def castable(conn, owner_id, language) -> list[tuple[SavedVoice, VoiceExample]] # voices not pending, with an example in that language
def read_narrator(conn, group_id, language) -> NarratorChoice | None           # None = no row = Automatic
def set_narrator(conn, group_id, language, voice_id: int | None) -> None       # None deletes the row
def conditionings(conn, storage, example_id) -> list[dict]                     # [{"model", "revision", "blob": bytes}]
def save_conditioning(conn, storage, example_id, model, revision, blob) -> bool  # upsert; False = the example is gone or pending
```

`add_example` checks, in order, each with a `VoiceError`: `consent` is true; `language` in `LANGUAGES`; `source` in (`recorded`, `uploaded`); `len(data) <= MAX_BYTES`;
then the WAV format, inline with stdlib `wave.open(BytesIO(data))`:

- `getcomptype() == "NONE"`, `getsampwidth() == 2`, `getnchannels() == 1`, `getframerate() == SAMPLE_RATE`, `getnframes() > 0`.
- `len(readframes(nframes)) == nframes * 2` (a truncated file reads short with no error, §1.3).
- `wave.Error` and `EOFError` → `VoiceError("This file is not a WAV file.")`. Each other breach → a `VoiceError` that names the format found,
  e.g. "The example must be mono 16-bit PCM at 48000 Hz; this file is 24-bit stereo at 44100 Hz."
- A float WAV fails as `wave.Error` on the venv's Python 3.13 (§1.3). A `wave` that reads float (system Python 3.15.2, read 2026-11-01) reports sample width 4.
  Both end in a `VoiceError`.

Then `storage.wav.decode_wav` gives the samples: `MIN_SECONDS <= seconds <= MAX_SECONDS`; RMS ≥ `SILENCE_RMS`. `storage/wav.py` does not change.
`content_hash = sha256(data)` and the file holds `data` exactly as posted and checked. `transcript = RECORDING_PROMPTS[language]` when recorded, else `None` (**superseded 2026-11-02**: the selected reading text, §2.15 I).

Writes that make a voice file (`add_example`, `save_conditioning`), with a write-ahead cleanup row (precedent for `BEGIN IMMEDIATE`: `models/slots.py:147-157`):

1. Commit a write-ahead row: `INSERT INTO voice_file_deletions(storage_key, deletion_id, requested_at)` with `deletion_id` NULL. Keep its `id`.
2. `BEGIN IMMEDIATE` → re-read that exact row by `id`. Gone (a drain acknowledged the key in the gap) → `ROLLBACK`, commit a new write-ahead row, and repeat step 2.
3. Re-read the parent. `add_example`: the voice exists for the owner and is not pending, else `ROLLBACK` and `VoiceDeleting`.
   `save_conditioning`: the example exists and is not pending, else `ROLLBACK` and `False` (nothing is stored; the ready check of §2.8 then fails the film).
   The write-ahead row stays in both cases; the drain acknowledges it.
4. The file exists and the sha256 of its bytes matches the key → skip the write (`LocalStorage.write` truncates in place, `storage/local.py:32-35`,
   so a rewrite could break a file that another row holds). Else `storage.write(key, data)`.
5. Insert or upsert the row. `save_conditioning` puts the replaced row's old key, when it differs, into `voice_file_deletions` with `deletion_id` NULL.
6. Delete the write-ahead row by `id` → `COMMIT`. On any exception: `ROLLBACK` and re-raise. The write-ahead row is already durable, so the drain removes the file if no row holds it.

SQLite serialises writers across every connection to the file, in `app` and in the worker (`db/connection.py:21-23`).
So no drain can check and unlink a key between steps 2 and 6. A process that dies in steps 2-6 leaves its write-ahead row, and the drain removes the file.
No extra clip check in `save_conditioning`: ids are never reused (§2.6), and the child rebuilds any blob whose clip sha256 differs (§2.5).

`create_voice`, `rename_voice`, `request_voice_deletion`, `request_example_deletion` and `set_narrator` each run their check and their write in one `BEGIN IMMEDIATE`.

Deletion (D33). The request side never deletes a row or a file:

- `request_voice_deletion` / `request_example_deletion`: the target exists for the owner, else `False` (→ 404). Then
  `INSERT INTO voice_deletions(owner_id, kind, target_id, requested_at) … ON CONFLICT(owner_id, kind, target_id) DO NOTHING` (pattern `ingest/folders.py:41-47`).
  A second request is a no-op. An example of a pending voice is already covered: the call returns `True` and inserts nothing.
- Pending = a `voice_deletions` row with `cascaded_at IS NULL` names the voice, or names the example, or names the example's voice.
  A pending target rejects every mutation (`rename_voice`, `add_example` → `VoiceDeleting`; `save_conditioning` → `False`) and every new cast (`castable`, `example_for`).
- `process_voice_deletions(conn, storage, owner_id) -> int`, three steps. Each transaction re-reads its row first, so two drainers (mac `make up`, §1.3) are safe:
  1. Cascade. Per intent of `owner_id` with `cascaded_at IS NULL`, in `id` order: `BEGIN IMMEDIATE` → collect the target's example WAV keys and the keys of their
     conditionings → insert each into `voice_file_deletions(storage_key, deletion_id, requested_at)` with `ON CONFLICT(storage_key, deletion_id) DO NOTHING` →
     delete the target row, owner-scoped (a voice: `voice_examples` and `voice_conditionings` cascade, `memory_narrators.voice_id` becomes NULL; an example:
     its `voice_conditionings` cascade) → set `cascaded_at` → `COMMIT`. A target that is already gone gets only `cascaded_at`.
  2. Files. Per distinct `storage_key` in `voice_file_deletions`: `BEGIN IMMEDIATE` → ref check: a `voice_examples` or `voice_conditionings` row holds the key →
     keep the file; else `storage.delete(key)` (`missing_ok`, `storage/local.py:37-40`) → ack: delete every `voice_file_deletions` row of that key → `COMMIT`.
  3. Finish. Delete each cascaded intent of `owner_id` that has no `voice_file_deletions` row left. The intent ends only after its files are gone.
  - A failed step → `ROLLBACK`, log it, go on with the next row. The row stays for the next run. It never raises for one row:
    `drain_pass` group 1 has no guard (`ingest/pipeline.py:50-62`), and a raise would stop the worker loop.
  - Returns `pending_voice_deletions(conn)`.
- A process crash (the process dies; SQLite rolls back its open transaction) at any point re-runs safely: before the cascade `COMMIT` (nothing changed),
  after it (the keys are durable), after an unlink and before its `COMMIT` (the key row stays; the next unlink is a no-op), after the acks (step 3 runs again).
  This plan makes no claim for power loss: `LocalStorage.write` and `delete` call no `fsync` (`storage/local.py:32-40`).

### 2.8 Casting and the film build — `feed/voice.py`, `feed/build.py`

`feed/voice.py` (stays pure):

- `offered(language)` skips entries with `clones=True` explicitly.
- `@dataclass(frozen=True) class SavedCast: token: str; name: str`.
- `resolve_saved(name, cast) -> str | None`: the token for a case-folded name match, or for the token itself. Else `None`.
- `saved_for_prompt(cast) -> str`: one line per voice, `"  <name> — a saved voice: this person's own voice"`.

`feed/build.py`:

- `build_presentation(…, voices: Storage | None = None)` (new keyword).
- After "planning the shots" and before the story pass: `chosen = read_narrator(conn, group_id, language)`.
  - `chosen` set → a saved film. It never becomes Automatic or a preset. Each case below is `_fail(…, reason)`, with no speak call:
    - `chosen.voice_id is None`, or `get_voice(…)` is `None` (deleted since the read) → "The saved voice chosen for this film was deleted. Choose another narrator."
    - the voice is pending (D33) → "The saved voice «<name>» is being deleted. Choose another narrator."
    - the voice is not in `cast = castable(conn, owner_id, language)` (no example in this language, or no `voices` store) →
      "The saved voice «<name>» has no <Language> example. Add one under Organize → Saved voices, or choose another narrator."
  - `chosen` set → the story pass gets `speakers=saved_for_prompt(cast)`. Else `catalogue_for_prompt(language)`, as today.
  - `cast` holds no pending voice or example, so a `[voice]` bracket cannot cast one.
- At "recording the voice":
  - Saved film: `speaker = token(chosen.voice_id)` (the user's choice wins over `story.narrator`).
    `passage_voices = _passage_voices(story, language, speaker, bracketed, resolver=lambda n: resolve_saved(n, cast_list))`.
  - Automatic film: today's code, unchanged (`feed/build.py:270-272`).
  - `_passage_voices(…, *, resolver=None)`; the default resolver is `resolve(language, name)`.
- Saved film speak call:
  - `references` = every used token → `{"audio": read_example_audio(...), "sha256": example.content_hash, "language", "transcript", "conditioning": conditionings(...)}`.
  - `client.speak(..., references=references)` — the keyword is passed only for a saved film, so every existing test fake keeps working.
  - Wrapped in `_Ticking(progress, "recording the voice", (0.30, 0.92), expect)`, `expect = len(story.narration) / 14 * CLONE_RTF_ESTIMATE`.
    `CLONE_RTF_ESTIMATE = 1.0` sits beside `MUSIC_RATE` and is labelled an estimate for display only.
  - Any exception goes to the existing `_fail` path (`feed/build.py:302-306`), with the service's own reason (`_reason`). Never a second call with a preset.
  - After success: `save_conditioning` for every `built` entry with a blob. `False` (the example went pending during the build) stores nothing.
  - A drain that removes a clip while the film builds makes `read_example_audio` fail → the `_fail` path (§2.12 "example WAV missing on disk").
- Film audio is written only at "saving the film", for EVERY film (D34). Both `_store_audio` calls move there: narration (today `feed/build.py:307`)
  and music (today `:352`). The steps before it use the bytes only: timing (`:312-316`), pulse and arrangement (`:359-379`).
  The keys enter the `Presentation` only at `:391-392`. Automatic film: write the WAVs, then `save_presentation` as today (`:400`).
- At "saving the film" (`feed/build.py:381-404`), saved film only (the recommended answer to the gate question; D15, D33):
  - One `BEGIN IMMEDIATE` holds, in this order: the check (every used token's voice and its example exist and are not pending; this covers the narrator and
    every bracket voice) → narration `_store_audio` (+ music if any) → the ready `save_presentation` (`feed/build.py:400`) → `COMMIT`. No model call inside it.
  - A miss → `ROLLBACK` → the unchanged `_fail(…, "The saved voice «<name>» was deleted while this film was made.")`. No film audio was written.
  - Any other exception inside → `ROLLBACK` and re-raise.
  - `_fail` gets no new keyword, and no failure path prunes audio: a storage key comes from the bytes alone (`feed/build.py:641-646`; the music seed is the film's own
    identity, `:547-551`), so two builds can make the same key, and a key says nothing about which build owns it.
  - The guarantee is narrow: a failed check stores no narration and no music; a cancel seen by `step("saving the film")` (`feed/build.py:185-189`) also comes before
    the writes. Not claimed: a cancel after that check, an I/O or save error after a WAV write (`ROLLBACK` does not undo a file), the success-path prune (`:403`),
    conditioning blobs (the voice outbox, §2.7), and any bound on how long the writes take.
  - So a film finished before the delete keeps its sound until the user regenerates it (`feed/store.py:94-103`, `web/app.py:966-975`). A film still being made fails.
- Decisions: `decisions["cast"]` gains `"narrator_chosen_by": "user" | "story"`; `offered` lists saved names for a saved film.
  `decisions["voice"] = {"saved": bool, "model", "revision", "conditioning": {token: state}, "wall_seconds", "narration_seconds", "seed": "unsupported (voxcpm 2.0.3)"}`. <!-- pragma: allowlist secret -->

### 2.9 Saved voices page — `web/voices_api.py` (new) and templates

Routes (pattern `web/upload_api.py:46`; `register(app, context, templates)` called next to `register_upload_api`, `web/app.py:257`):

| Method | Path | Does | Reply |
|---|---|---|---|
| GET | `/organize/voices` | the page | `voices.html` |
| GET | `/organize/voices/list` | the list only (polled while a row is deleting) | `_voices_list.html` fragment |
| POST | `/organize/voices` | form `name` → create | fragment |
| POST | `/organize/voices/{voice_id}/rename` | form `name` | fragment |
| POST | `/organize/voices/{voice_id}/delete` | `request_voice_deletion`, then `voice_drain.schedule()` | fragment, the voice "deleting…" |
| POST | `/organize/voices/{voice_id}/examples` | multipart `language`, `source`, `consent`, `file` (WAV) | fragment |
| POST | `/organize/voices/{voice_id}/examples/{example_id}/delete` | `request_example_deletion`, then `voice_drain.schedule()` | fragment, the example "deleting…" |
| GET | `/organize/voices/{voice_id}/examples/{example_id}/audio` | play it | `audio/wav` |

- Status codes: 400 `VoiceError` text; 404 unknown or foreign id, and the audio of a pending example; 409 duplicate name, voice copy not available on this profile
  (create and add example), or `VoiceDeleting` (rename, add example, create with a pending voice's name); 413 file over `MAX_BYTES`;
  507 when `ctx.voices.free_bytes() < FREE_SPACE_FLOOR` (`web/upload_api.py:21`). Errors are JSON `{"detail": …}`.
- Delete routes never delete a row or a file, and never call `drain_now` or `process_voice_deletions`. A second delete of a pending target returns the same fragment.
  A delete after the drain finished answers 404.
- Rename and delete stay allowed on every profile.
- Placement: `web/templates/organize.html` gets `<a class="organize-voices" href="/organize/voices">Saved voices</a>` in `.organize-head`, OUTSIDE the
  `organize-picker` form (`:5-22`). A plain link — the nav boost does not touch `<main>` links (`web/templates/base.html:11-16`).
- `voices.html` extends `base.html`. Content: title "Saved voices"; a × close (`data-origin="/organize"`); the not-available notice when it applies;
  "New voice" (name + Create); per voice: name with inline rename, Delete voice (confirm: "Delete «<name>» and its examples? Films already narrated with this voice keep
  their sound until you regenerate them."), examples (language · seconds · recorded/uploaded · date · ▶ `<audio preload="none">` · Delete),
  "Add an example": a language select (only `cloning_languages(profile)`), the consent checkbox with `CONSENT`, **Record** and **Upload a file**.
  No "Organize files…".
- Deleting state (pattern `web/templates/upload.html:19-20`): a pending voice shows its name and "deleting…", with no rename, Delete, examples controls or "Add an example".
  A pending example shows its line and "deleting…", with no ▶ and no Delete. Each such row carries `data-deleting`. When the drain is done, the row is gone.
- `web/static/voices.js` (new):
  - Every action uses `fetch`. A 2xx reply replaces `#voices`; a 4xx/5xx reply shows its `detail` in `#voices-error`. No action pushes history (§13.1 rule 3).
  - While `#voices` holds a `[data-deleting]` row, it fetches `GET /organize/voices/list` every 2 s and replaces `#voices`. It stops when no row is deleting.
  - Record: `getUserMedia({audio: {channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false}})` → `MediaRecorder`.
    The panel shows the fixed prompt for the chosen language, a timer and Stop. It stops by itself at 30 s.
    No `navigator.mediaDevices` or no secure context → Record is hidden with "Recording needs HTTPS or localhost. Upload a file instead."
  - After Stop, and for an upload (`accept="audio/*"`): `new AudioContext({sampleRate: 48000}).decodeAudioData` → mix down to mono → encode PCM-16 WAV in JS → preview → Save / Discard.
    Save posts `language`, `source`, `consent=yes`, `file=example.wav`. The JS checks 5-30 s first; the server checks again.
  - Esc, in this order: a live recording → stop and discard it; else an unsaved take → discard it; else a focused text field → blur it;
    else close: `history.back()`, or `location.href = "/organize"` when there is no history (pattern `web/templates/feed.html:213-225`).
- `web/static/app.css`: styles for `.organize-voices`, the page, the recorder panel, the deleting rows.
- Who runs the drain (D33):
  - With a worker: `drain_pass` group 1 calls `process_voice_deletions(conn, context.voices, settings.owner_id)` right after `process_folder_deletions`
    (`ingest/pipeline.py:53-56`). The worker loop runs it every pass (`ingest/cli.py:27-29`). This also covers `drain_now` after an upload.
  - Without a worker (`inline_drain` True, `config.py:191`): `VoiceDeletionDrain` in `web/voices_api.py` runs it on a background thread. The request never waits for it.
- `class VoiceDeletionDrain(context: Callable[[], AppContext])`, `RETRY_SECONDS = 10` (the worker's `POLL_SECONDS`, `ingest/cli.py:10`):
  - `schedule() -> None`: returns at once when `context().settings.inline_drain` is False (the worker drains it). Else, under a `threading.Lock`:
    `running` True → set `rerun = True`; else set `running = True` and start one daemon thread (pattern `web/app.py:1042-1044`). At most one drain thread per process.
    It never asks `Thread.is_alive()`.
  - `wake() -> None`: `schedule()` when `pending_voice_deletions(conn) > 0`, else nothing.
  - The thread loops `left = process_voice_deletions(ctx.conn, ctx.voices, ctx.settings.owner_id)` on its own thread-local connection (`web/deps.py:37-45`).
    `left > 0` (a step failed) or an exception (logged) → sleep `RETRY_SECONDS`, run again. `left == 0` → under the lock: `rerun` set → clear it and run again;
    else set `running = False` in that same locked block and end. So a `schedule()` in the exit window always gets a run.
  - `register()` builds it as `app.state.voice_drain`, adds the test seam `app.state.await_voice_drain` (joins the thread; pattern `web/app.py:232-237`),
    and calls `wake()`. `register()` runs inside `create_app`, so this is the start-up run.
  - The delete routes enqueue, render the fragment, then call `schedule()` and reply. So the reply always shows "deleting…", even when the drain is fast.
  - The add-example route calls `wake()` after every request, and the `_start_film` thread calls `app.state.voice_drain.wake()` in its `finally` after every build
    (`web/app.py:1029-1040`). Replaced blobs and write-ahead rows (§2.7) then never wait for a delete or a restart.
  - Mac `make up` runs both this thread and the worker (§1.3). §2.7 makes two drainers safe.

### 2.10 Narrator picker — `web/app.py` feed section, `web/templates/feed.html`

- `feed()` adds `narrators` = `[(voice.id, voice.name) for voice, _ in castable(conn, owner, language)]` when `cloning_entry_for(profile, language)` exists, else `[]`;
  and `narrator` = `read_narrator(conn, group_id, language)`.
  - A stored saved choice that is not castable → `unavailable = (value, label)`, checked in this order, on every profile:
    deleted → `"deleted"` and "A deleted saved voice"; pending → the voice id and "«<name>» — being deleted";
    no cloning entry for the profile and language → the voice id and "«<name>» — voice copy is not available on this device";
    no example in the language → the voice id and "«<name>» — no <Language> example". Else `unavailable = None`.
- `feed.html`: inside BOTH forms (Regenerate `:42-46`, Prepare `:139-142`) a `<select name="narrator">` with "Automatic" (`""`) and the saved voices; the stored choice is `selected`.
  An `unavailable` narrator is one more option, `selected`, so the page never shows Automatic in place of the user's choice (D33).
  **[Superseded 2026-11-02 by §2.15 C]** Shown only when `narrators` is not empty or `unavailable` is set. With voice copy available and no castable voice, the Prepare card shows one line of text (no link):
  "Saved voices under Organize can narrate this film." **[end superseded]** The existing `fetch` + `location.replace` submit (`:242-252`) posts the field unchanged.
  One change to that submit: a 4xx reply shows its `detail` under the form and the page stays. Other replies → `location.replace`, as today.
- `feed_prepare(key, lang: str = Form(default=""), narrator: str | None = Form(None))` (today `lang` only, `web/app.py:978`). Before any build starts or queues:
  (Implemented with the raw form instead, `web/app.py:1060-1089`, §1.6. Phase 3 adds the `preset:<name>` value, §2.15 C.)
  `None` (no field: the picker is not on the page) → keep the stored row, no `set_narrator` call. `""` → `set_narrator(None)` (Automatic).
  `"deleted"` → 409 "The saved voice chosen for this film was deleted. Choose another narrator."; unknown or foreign voice → 404;
  no cloning entry for the profile and language → 409; the voice is pending → 409 "«<name>» is being deleted. Choose another narrator.";
  no example in the language → 409. Valid → `set_narrator(voice_id)`.
- `_start_film` passes `voices=ctx.voices` to `build_presentation`. `feed_building` and `feed_queued` stay `(key, language)`; the build reads the choice from the table.
- **[Superseded 2026-11-02 by §2.15 B]** The card's ↻ Rebuild does not change (`docs/design.md:1713-1715`).
  Phase 3 relabels it "↻ Retell memory" with visible text. Its route and behaviour do not change, and it still carries no narrator picker (t4).

### 2.11 The generic capability rule

- `app` decides from the catalog and its own profile: `cloning_entry_for(settings.profile, language)` and `cloning_languages(settings.profile)`. No call to the models service.
- No cloning language → the Saved voices page shows "Voice copy is not available on this device." It hides New voice, Record and Upload; the API answers 409.
  **[Superseded 2026-11-02 by §2.15 C]** The picker is hidden, unless a saved choice is stored for that film. **[end superseded]** Then it shows "Automatic" and that choice, `selected`,
  labelled "«<name>» — voice copy is not available on this device" (§2.10). A Prepare with a saved narrator answers 409. The wording names no profile.
  Phase 3: the picker is always shown (Automatic and the preset voices); its "Saved voices" group shows the disabled line "Voice copy not available on this device".
- The models service checks again (§2.4). A request that still arrives fails with 503 and the film fails with that reason.

### 2.12 Error paths

| Case | Where | Result |
|---|---|---|
| no voice-copy model on the profile | page / API / picker | notice; 409; picker hidden, or shown with the stored choice labelled "not available" (**superseded 2026-11-02**: the picker is always shown, §2.15 C) |
| same, at the service | `/speak` | 503 → film `failed` with the reason (`feed/build.py:302-306`) |
| narrator has no example in the film language | build | `failed` with the reason; no speak call |
| example WAV missing on disk | build | `failed` with the reason |
| mixed or unknown voice tokens; bad base64; sha mismatch; too large; > 8 refs | `/speak` | 422 |
| weights not on disk; revision mismatch; CPU device | child load | 500 with the message → film `failed` |
| reply model or revision ≠ routed entry | gateway | 500 → film `failed` |
| stored blob stale | child | rebuilt; state `built` with the reason |
| upload: no consent, unknown language, < 5 s, > 30 s, silent | API | 400 |
| upload: not a WAV, malformed header, truncated data, compressed or float, 8/24/32-bit, stereo or more channels, rate ≠ 48000 Hz | API | 400 with the format found |
| upload > 4 MiB / low disk | API | 413 / 507 |
| blank, too long, duplicate name | API | 400 / 400 / 409 |
| rename, add example, or create with the name of a voice being deleted | API | 409 `VoiceDeleting` |
| delete again while pending / after the drain finished | API | same "deleting…" fragment / 404 |
| audio of an example being deleted | API | 404 |
| chosen narrator being deleted, or deleted | Prepare / build | 409 with the reason / `failed` with the reason; no speak call; never Automatic |
| file unlink fails during the drain | drain | `ROLLBACK`; the key stays; retried next pass (`RETRY_SECONDS` without a worker) |
| a used voice or example deleted while the film is made (narrator or bracket) | build, at the ready save | `failed` "was deleted while this film was made"; no film audio written |
| a film finished before the delete | `/feed` | keeps its sound until the user regenerates it |
| the process dies during a voice file write | next drain | the write-ahead row stays; the drain removes the file when no row holds it |

### 2.13 Validation script — `scripts/voice_check.py` (new; the user runs it)

A thin client (no torch). `uv run --no-sync python -m scripts.voice_check --wav PATH --language en|ru [--cross] [--out DIR]`.

1. It reads the WAV and checks the clip contract of `add_example` (§2.7): ≤ 4 MiB; stdlib `wave` reads it; uncompressed, 16-bit, mono, exactly 48000 Hz;
   frames > 0 and `readframes` returns `nframes × 2` bytes; 5-30 s; RMS ≥ -50 dBFS. A breach stops the script with the format found.
   It keeps its own small check, because lane 4b lands in phase 1, before `feed/saved_voices.py` exists. Then it builds one reference (token `check`) from the clip.
2. It unloads the voice unit (`ModelsClient.model_unload("voice")`) and reads `models_state()["free_mb"]`.
3. Pass 1 (fresh): the fixed sentences of the language (EN-1..5 or RU-1..5, findings §10), then EN-5/RU-5 as 5 `segments` in one call. No stored conditioning.
   It reads `free_mb` again after the first call (resident cost ≈ the drop).
4. It unloads the voice unit again. Pass 2 (restore): the same calls with the pass-1 blob.
5. `--cross`: also the other language's sentences with this clip. Reported only.
6. Output in `--out` (default `<data_dir>/voice-check/<UTC stamp>/`): every WAV as `<id>-A.wav` / `<id>-B.wav` in random fresh/restored order, `key.json` with the order,
   and `report.json`. Per call: duration, sample rate, bytes, RMS, NaN, `model`, `revision`, conditioning state, wall seconds, RTF, chars per second, segments monotonic.
7. A reply whose `model` ≠ `voxcpm2` or `revision` ≠ the catalog revision is marked VOID (findings §10, "Declared identity").
8. It prints a summary table, the resident-cost drop and the pass thresholds (§6.1).

### 2.14 Docs to change (docs first, per phase)

Phase 1 (engine):

- `docs/design.md` §4.1: the `voice` row (Kokoro en · Silero ru · VoxCPM2 = saved voices, mac/cloud); a paragraph on `clones` entries
  (no preset speakers; picked only for a stored-voice request; preset routing never picks one; the speakers rule of CLAUDE.md:236-251 does not apply to them);
  the `source` bullet gains the pinned `revision`.
- §5.1: rewrite the `POST /speak` bullet to the full current contract: `segments`, `voices`, `references`; reply `segments`, `revision`, `conditioning`; 503/422/500.
- §8.1: the `voice` unit can hold VoxCPM2 (≈ 9.1 GB fp32 weights on MPS; cost 10500 MB is an estimate; measure it).
- `docs/models.md`: `:17` and `:44` rows; a new "Voice copy — VoxCPM2" section after `:91`: pin, loading (`load_denoiser=False`, `optimize=False`, explicit device, fp32 on MPS),
  controls (§2.5), reference mode, conditioning blob and invalidation, `make models` pre-fetch size, the dependency licence results (§3), "no seed in 2.0.3".
  The licence results include the five approved exceptions with their obligations (§3 step 5, `$S/exceptions.md`) and the funasr removal (D35).

Phase 2 (product):

- §5 mermaid (`docs/design.md:476-515`): browser label + `/organize/voices`; `app` label + saved-voice receipt; store node + "saved-voice examples + conditioning (data_dir/voices)";
  the `app → models` edge label + "/speak with saved-voice references"; the worker node's "deletions" (`:486`) → "deletions (folders · saved voices)";
  the `worker → store` edge (`:505`) + "remove deleted saved-voice files".
- §6: bullets for `saved_voices`, `voice_examples`, `voice_conditionings`, `memory_narrators` (a NULL voice = the chosen voice was deleted), `voice_deletions`,
  `voice_file_deletions`, and `data_dir/voices`. `docs/data-model.md`: the DDL of §2.6.
  One §6 paragraph in the §3.2c style (`docs/design.md:309-313`): "Deleting a saved voice or an example runs through an outbox + worker, never inline."
  The request records the intent and returns; the row shows "deleting…" and takes no edit and no new casting. The drain (`process_voice_deletions`, `drain_pass` group 1)
  deletes the rows and records their files in one transaction, then removes each file that no row still holds. With no worker (`inline_drain`), `app` runs the same
  drain on one background thread after a delete, a film build, an added example, and at start. A film whose chosen narrator is being deleted, or was deleted,
  fails with that reason. A film finished before the delete keeps its sound until it is regenerated; a film still being made fails at its save.
  Every voice file write first commits a cleanup row, so a process that dies mid-write leaves no file that nothing removes. Ids are never reused.
- §11.1: casting text (`docs/design.md:1680-1693`): the narrator picker, one engine per film, the failure rule (+ "a narrator being deleted or deleted fails the film;
  a voice deleted while the film is made fails it at the save, before any film audio is written; a finished film keeps its sound, §6");
  the "Language" paragraph (`:1652-1660`): a saved voice needs an example in the film language.
  Film-pipeline mermaid (`:1664-1678`): trigger node + "narrator: Automatic or a saved voice"; node 4a + "voxcpm2 — a SAVED voice (mac/cloud): clip + stored conditioning";
  the failure edge + "saved voice unavailable"; persist node (`:1675`) + "new voice conditioning → voice_conditionings" and "saved film: voice check →
  WAVs → row in one transaction"; a new edge `persist -- "a used saved voice was deleted (no WAV written)" --> failed`.
- §11.3 `voice` row (`:2017`): with a saved-voice narrator a bracket casts another saved voice of that language; with Automatic, presets only.
- §13: the `/organize` bullet gains the Saved voices link and page; the `/feed` bullet gains the narrator picker.
  §13.1: rule 1 — `/organize/voices` is a leaf under Organize; rule 4 — its own levels (live recording, unsaved take) unwind first.
- `docs/ui.md`: the `/organize` route text (`:165-175`), a new `/organize/voices` route entry (with its "deleting…" rows and the Delete-voice confirm text),
  and the `/feed` Prepare/Regenerate text (`:176-200`; with the unavailable-narrator option, also on a profile with no voice copy).
- `docs/ingest-pipeline.md:111-115`: the group 1 list "thumbnail, EXIF place facets, folder deletions" + "saved-voice deletions".
- Validate every edited mermaid block with the script in `<task-skill>/references/mermaid.md` (`mmdc` is at `/opt/homebrew/bin/mmdc`). Every block must print `OK`.

Phase 3 (organize and film clarity, §2.15; line numbers read 2026-11-02). It extends the Phase 2 items for §6, §11.1, §11.3, §13 and `docs/ui.md` above.
Where the two differ, Phase 3 wins. Phase 2's "The card's ↻ Rebuild does not change" (§2.10) no longer holds.

- `docs/design.md` §6, the `memory_narrators` bullet (`:772-776`): a saved voice OR a preset voice per `(group_id, language)`; `preset_speaker` is a catalog name;
  the CHECK; both NULL = the chosen saved voice was deleted (unchanged).
- §11.1 casting text (`:1801-1815`): "The user may choose the narrator: Automatic, a preset voice or a saved voice." A preset narrator: the story pass sees the presets,
  the user's preset narrates whatever the story pass returns, and a `[voice]` bracket casts another preset. The picker is "Narrator for the next film", always shown.
  `:1817-1826` gains: a chosen preset that the catalog no longer offers for the language fails the film with the reason, never Automatic.
  The `:1761` sentence ("the user never selects it") stays: it is about the MODEL.
  `:1786-1799` gains (t18): Automatic picks a preset only, never a saved voice; the film records who chose its narrator (`user`, `story`, `mood`) and its labels,
  and `/feed` and the card show them; nothing predicts the Automatic voice before the film is made.
- §11.1 film mermaid (`:1769-1784`): the trigger node (`:1771`) "narrator: Automatic or a saved voice" → "narrator: Automatic · a preset voice · a saved voice";
  the failure edge (`:1776`) "cannot speak · saved voice unavailable" → "cannot speak · chosen narrator unavailable".
- §11.1 retell paragraph (`:1846-1861`): "↻ Rebuild" → "↻ Retell memory"; the card says what it does: it rewrites the title, description and photo order,
  makes no film, and deletes the films of the languages that have one; the narrator choices stay.
- §11.3 `voice` row (`:2150`): "with Automatic, presets only" → "with Automatic or a preset narrator, presets only" (D41).
- §13 `/organize` (`:2233-2242`): a memory card lists one film per language (link, state, narrator with "(auto)" or "(your choice)") and "↻ Retell memory" with its text.
  The film summaries refresh in place (t19): on a cached return from `/feed`, and while a film of that memory runs. No reload, so the grid keeps its scroll,
  its controls and its expanded notes. Lane 15b owns this code.
  `/feed/{key}` (`:2243-2254`): "Film language" with ✓, "This film: … · narrated by … — <who chose it>", "Other voices", the always-shown picker
  with its Automatic text, the hints and the bracket line; "Reached from a ▶ Play control" → "Reached from a film link on a memory card".
- `docs/ui.md` `/organize` (`:165-177`): the card's film rows and their states, "↻ Retell memory" and its text, and the in-place refresh of the rows (t19). `/feed/{key}` (`:206-230`): the "Film language" label
  and ✓; the "This film" and "Other voices" lines; the **Narrator** bullet (`:218-230`) rewritten to §2.15 C (groups, labels, Automatic text and hint,
  saved-voice hint, disabled line, bracket line); the old hint line goes. The "language selector … sits top-left" sentence (`:207`) → the t20 header:
  an action row (language, ↓ Download, ×) that wraps, and below it, on a ready film, a settings region with the Regenerate form that scrolls inside.
  `/organize/voices` "Add an example" (`:186`): the recording tip of §2.15 H.
  `:189` "opens a panel with a fixed prompt to read" → a preparation panel with one of four reading texts, "Different text", "Start recording", then the
  timer and Stop; the text stays fixed through the recording and the unsaved take (§2.15 I). The Esc bullet (`:203-205`) adds "a prepared panel is closed"
  after the unsaved take.
- `docs/design.md:763` "with its fixed prompt as `transcript`" → "with the selected reading prompt as `transcript`" (it never says the speech was checked
  against it). §13.1 (`:2344-2346`): the Saved voices Esc order adds "closes a prepared panel" after "discards an unsaved take".
- `docs/data-model.md:174`: the `transcript` comment → "the selected reading prompt when recorded; NULL when uploaded".
- `docs/design.md` §11.1, after "**Autoplay is a click.**" (`:1751-1752`): one "**Controls.**" paragraph on the t20 header (one header, no fixed offsets
  between controls; each control fades on its own; an exporting Download stays fully visible).
- `docs/data-model.md` `memory_narrators` (`:196-204`): the DDL of §2.15 D, plus one line: an existing table gains the column through `_ADDED_COLUMNS`, without the CHECK.
- Comments, in the same change (`ivms777/CLAUDE.md:3-36`): `web/app.py:506-507` → "`voice` is deliberately absent: the film's language picks the voice MODEL (§11.1).
  The narrator is chosen per film on `/feed`, not here." `feed/voice.py:3-4` and `feed/build.py:300-301` say the user may choose the speaker (D41).
- Validate the edited §11.1 mermaid block with the same script. It must print `OK`.

Phase 4 (voice workbench, §2.16; line numbers read 2026-11-02). Docs first, in lane 18. Where an item differs from Phase 2 or 3, Phase 4 wins.

- `docs/design.md` §5 mermaid (`:501-537`):
  - the `app` node (`:510`): "saved-voice receipt" → "saved-voice receipt · voice tests";
  - the `store` node (`:515`): "saved-voice examples + conditioning (data_dir/voices)" → "saved-voice examples + conditioning + voice tests (data_dir/voices)";
  - the `app → store` edge (`:529`): "read originals / thumbnails" → "read originals / thumbnails · write saved-voice examples, conditioning and voice tests";
  - the `app → models` edge (`:532`): "/speak with saved-voice references (HTTP)" → "/speak with saved-voice references: films and voice tests (HTTP)".
  - The planner rendered these four labels with `mmdc` on a copy of `docs/design.md` (2026-11-02): all 7 blocks `OK`.
  - §5 text (`:546`): "saving a voice example" → "saving a voice example or a voice test".
- §6 (`:761-799`): two new bullets after `voice_conditionings` (`:767-771`):
  `voice_film_examples` (the example films use per voice and language when the user chose one; no row = the newest; `example_id` NULL = the chosen
  example was deleted, and films with that voice in that language fail until the user chooses again) and `voice_previews` (the last test of one
  example: its text, model, revision, built/restored, time and a WAV; never used by a film). The `data_dir/voices` bullet (`:783-785`) adds "test WAVs".
  The outbox paragraph (`:787-799`) adds: "Test WAVs follow the same rules: written behind a cleanup row, removed only by the drain."
- §11.1 "Language" (`:1773-1776`): "and the newest such example wins" → "the example the user chose for films in that language (Use for films), else
  the newest such example; a chosen example that was deleted or is being deleted fails the film with the reason, never another example".
- §11.1 film mermaid (`:1786`): node 4a "voxcpm2 — a SAVED voice (mac/cloud): clip + stored conditioning" → "voxcpm2 — a SAVED voice (mac/cloud):
  its film example (chosen, else the newest) + stored conditioning". Rendered `OK` by the planner on a copy (above).
- §11.1 casting (`:1834-1836`): "per voice, its example WAV" → "per voice, the WAV of its film example". The failure list (`:1841-1843`) adds
  "or when the example chosen for films in the film's language was deleted or is being deleted". A narration call waits while another voice call of
  `app` runs, and the bar then reads "waiting for the voice model" (D52). **Changed 2026-11-02 (t25):** one turn covers reading the references,
  the speak call and storing the built conditioning, so a film and a test of the same example restore each other's conditioning.
- §11.1, a new paragraph after `:1853`, "**Testing a saved voice.**": `/organize/voices` makes a test of ONE example through the same `POST /speak`
  and the same reference a film sends; the text is the reading text (labelled as not checked against the recording), a fixed test sentence
  (the §6.1 voice-check sentences) or the user's own text (≤ 300 characters); the language is the example's. The conditioning the test builds is stored
  as a film's is, so the next film restores it: a test is the bake, and there is no Bake button. The last test per example is kept with its text,
  model, revision, time and built/restored. One test at a time per process, with stage, percent and Cancel; Cancel frees the page at once and drops the
  result, while the running model call ends in the background. **Changed 2026-11-02 (t25):** a test that was cancelled or replaced is never stored
  after the cancel or the new Generate returns (the store step and the cancel share one lock). **Changed 2026-11-02 (t25, round 2):** "Cancel is
  logical, not a GPU stop: it may wait for a test that is being stored, never for synthesis. A cancel before the model call makes no call; a call
  already sent runs to its end in the models service, and the next test or film waits for it ('waiting for the voice model'). The 1800-second
  limit is the HTTP timeout of the call, not a GPU deadline. After an app restart the panel says that the test stopped." A test never changes
  which example or narrator films use. No likeness score.
- §13 `/organize` (`:2267-2271`): the Saved voices text adds: "tests each example (▶ Original, ▶ Generated, Test this example) and chooses per
  language which example films use (Use for films; the newest by default)".
- §13.1 rule 1 (`:2337-2339`): the action list adds "test an example, use an example for films". Rule 4, Saved voices (`:2388-2391`): after
  "blurs a focused text field" add "→ closes the test panel when no test runs"; and one sentence: "A test that runs is not one of these levels: it
  runs on in `app`, and the page shows it again when it is opened."
- `docs/ui.md` `/organize/voices` (`:203-237`): the example line (`:209-210`) → "language · seconds · recorded/uploaded · date · #id · used for films
  (newest | your choice) or not used for films · Use for films · ▶ Original · ▶ Generated with its line · Test this example · Delete"; new bullets
  for "Use for films" (the badge texts; the button on every usable example except the chosen one, so the newest can be pinned — changed 2026-11-02,
  t25; the unavailable line; the Delete confirm of a chosen example) and for the test panel (texts, Generate, the job line with stage, percent and Cancel;
  changed 2026-11-02, t25 round 2: "Cancel stops the test for this page. A voice generation that has already started finishes in the background,
  and the next test waits for it."; the restart line "The app restarted, so this test stopped. Generate it again.";
  the "older model revision" note, the panel outside the list); the tip (`:214-215`) → the §2.16 F text; the Esc bullet (`:234-236`) → §2.16 G.
- `docs/data-model.md`: the DDL of §2.16 A after `voice_conditionings` (`:185-194`); the `voice_file_deletions` comment (`:223-225`, `:229`) adds
  "or a test WAV".
- Validate every edited mermaid block (`docs/design.md` §5 and §11.1) with the same script. Each must print `OK`.

Phase 5 (baked voice models and the Voices page, §2.17; line numbers read 2026-11-02). Docs first in each slice. Where an item differs from
Phases 2-4, Phase 5 wins. One lane (step 23) writes the Slice A and Slice V items before any Phase 5 code; step 30 writes the Slice B items.
The Slice A and Slice V items are done (step 23, 2026-11-02). **Changed 2026-11-02 (rework, t39):** the Slice B items below are rewritten;
where a Slice V text says which take films use, the Slice B item replaces it.

Slice A (engine):

- `docs/design.md` §4.1, after the `clones` paragraph (`:384-400`): "**A cloning entry may declare an `AdapterSpec`**" — the LoRA shape, the
  training cost (an estimate until measured), where a bake is offered (mac), and the package authors' 5-10 minutes as guidance, not a promise.
  An adapter is a per-voice LoRA file trained on the voice's takes; it is used wherever the entry runs and only at its pinned revision.
- §5 mermaid, the `models` node (`:515`): "· trains a saved voice's LoRA adapter in a killable `voice_train` child (§8.1)" and the HTTP list
  "· /voice/bakes". Validate the block.
- §5.1, the `/speak` bullet (`:620-657`): a reference may carry `adapter` (`blob`, `sha256`) and `use_clip`; the reply gains `adapters`; caps and
  422s; a voice request during a bake → 409. A new bullet after it (before `:658`): `POST /voice/bakes`, `GET /voice/bakes/current`,
  `GET /voice/bakes/{id}`, `POST …/cancel`, `GET …/adapter`, `DELETE …` with the request, the status, the stages, what Cancel stops, the work
  folder and its sweep (§2.17 E). Also (t31): the voice admission gate; `POST /models/voice/ensure` → 409 during a bake and
  `POST /models/voice_train/ensure` → 409 outside one; 503 "baking a voice is not offered on this device yet", apart from `voice copy is not
  available`; callers delete every ended bake.
- §8.1, after the `voice` unit paragraph (`:984-994`): "**The `voice_train` unit**" — one bake at a time; the bake unloads `voice` first, so one
  VoxCPM2 copy is resident; a governor lease keeps the training child resident until the bake ends; its cost is an estimate; the child trains on
  a thread of its own and answers status and cancel between steps; a hot reload of the models service ends a bake. Also (t31): the bake first
  closes the voice gate and waits for the speaks it already let in, and opens it only after the trainer is torn down; `voice_train` is
  registered once, so a slot switch never unloads it.
- `docs/models.md`: a new "### Voice baking — LoRA adapters" section after the VoxCPM2 section (before `## Switching a slot`, `:238`): the
  pinned training API (§1.8), what is not in the wheel, the recipe and its planner defaults (D58), the adapter file and its metadata, the child's
  switch and checks (D60, D61), the voice child built with the LoRA config, "no new dependency" with the lock evidence, and "unmeasured until the
  user's Slice A run". Also (t31): the names-and-shapes check before any load, and reset plus a cleared sha256 after any failure; text ids by
  `add_column`; "the LoRA layers, about 18 MB in float32, sit inside cost_mb 10500; with no adapter they add no adapter contribution, and their
  extra matmuls are not measured". The voxcpm2 row of the catalog table (`docs/models.md:17`, `:44`) gets the same sentence.

Slice V (the Voices page):

- `docs/design.md` §5 mermaid, the `browser` node (`:508`): "/organize · /organize/voices · /feed" → "/organize · /voices · /feed". Validate the block.
- §11.1 "**Testing a saved voice.**" (`:1886`): "`/organize/voices` makes a test" → "The Test tab of a voice (`/voices/{id}`) makes a test".
- §13 (`:2321-2324`): "the four top-level links" → "the five top-level links". `/organize` (`:2335-2346`): the Saved voices sentences go. A new
  bullet after `/organize`: **`/voices`** — the grid of voices (cards with per-language minutes, takes and the film take; New voice) and
  **`/voices/{id}`** — a voice's leaf: the header with the state and the ⋯ menu, the tabs Recordings / Voice model / Test (`?tab`, never a history
  entry), the takes, the recorder that opens in place, the per-language film-take select with its tombstone, and the test with its comparison.
- §13.1: `:2400` "Only the four top-level nav links" → "Only the five". Rule 1 (`:2411-2413`) → "**Voices (`/voices`) is a top-level grid**, and
  `/voices/{id}` is its one leaf: a card is a plain link (the one push); tabs and every action on the leaf push no history." Rule 2, after the feed
  bullet (`:2422-2425`): "The **voice leaf** needs no `ctx` either: it has one grid, and that grid has no filter, sort or search to carry. Close goes
  back only when the leaf was opened from the grid, else to `/voices`." Rule 4 (`:2462-2467`): the Esc order of §2.17 V4 (the test panel is no
  longer a level; tabs are none).
- `docs/ui.md`: `/organize` (`:176`): the Saved voices link goes. `:203-297` (`/organize/voices`) → two entries `/voices` and `/voices/{id}`, with
  §2.17 V's content: cards, tabs, the moved paths, the 308, close and Esc, and "Leaving with a recording or an unsaved take asks first." The picker
  hints (`:345`): "under Organize → Saved voices" → "under Voices". `## Nav order` (`:394-400`): "Upload → Library → Chat → Organize → Voices", with
  one sentence: voices are the people who narrate the films, a resource of their own, so they sit last.
- The comments that name the old page: `web/voices_api.py`, `web/templates/*voice*`, `web/static/voices.js` headers (in lane 27, with the code).

Slice B (kit, words, bake, the voice model as the only way a saved voice speaks). **Changed 2026-11-02 (rework, t39): rewritten.** Line
numbers read 2026-11-02 in the working tree:

- `docs/design.md` §5 mermaid: the `app` node (`:538`) "· voice tests" → "· voice tests · voice bakes"; the `store` node (`:543`)
  "saved-voice examples + conditioning + voice tests" → "saved-voice examples + voice models + voice tests"; the `app → store` edge (`:557`)
  "write saved-voice examples, conditioning and voice tests" → "write saved-voice examples, voice models and voice tests"; the `app → models`
  edge (`:560`) "/speak with saved-voice references: films and voice tests" → "/speak with saved voices' models: films and voice tests ·
  /voice/bakes". Validate the block.
- §5.1, the adapter bullet (`:673-698`): "A reference with `use_clip: false` needs no `audio` and no `sha256`: it speaks from the adapter
  alone. With `use_clip: true` (the default) a reference without `audio` is a **422** ("the reference of '<token>' needs its audio"). Audio
  sent with `use_clip: false` is still checked and is not used." The client bullet (`:708-712`): "the client sends `audio` and `sha256` only
  when the reference has them." `/voice/bakes` (`:713-717`): "`app` holds no bake yet; the user runs `scripts/voice_bake_check.py` …" →
  "`app` bakes a saved voice from the Voices page (§11.1), and the user's `scripts/voice_bake_check.py` bakes one read-only for the §6.5
  check. A bake that `app` starts carries `source: "app"` in its tag; `app` never cleans up a bake it did not start."
- §6 (`:897-929`): the `voice_examples` bullet adds the confirmed words (`transcript_confirmed_at`) and the kit text (`kit_id`). New bullets:
  `voice_models` — the voice model of one saved voice, the LoRA adapter baked from ALL its takes; films and tests of the voice speak only
  through it; one row per voice, versioned, bound to the model key, revision and LoRA shape, stored at most once per bake, and kept when
  takes are added, edited or deleted. `voice_model_takes` — what a voice model learned: one row per trained take with its language, length
  and fingerprint, kept after the take is deleted. `voice_bake_runs` — the immutable snapshot of a bake `app` started, with
  `cancel_requested_at`, which Cancel, replace and a voice delete set before they answer, so a cancelled bake is never stored, also after an
  app restart (Changed 2026-11-02, rework, t40). `voice_tests` — the
  last test of one voice, made through its voice model. The bullets of `voice_conditionings`, `voice_film_examples` and `voice_previews`
  (`:904-915`) each add: "Films and tests no longer use it (Phase 5, t39); its rows stay, and the drain removes their files with their
  take." The `data_dir/voices` bullet (`:927-929`) adds "voice model files". The outbox paragraph adds: "A voice model and a voice test leave
  with their voice. A deleted take never removes the voice model: Rebake makes a new one."
- §11.1 "Language" (`:1960-1974`): the saved-voice sentences → "A **saved voice** narrates a film only through its voice model: the LoRA
  adapter baked from ALL its takes (§4.1). No take is sent and none is chosen for films. The voice can narrate a language only when its
  model learned takes in that language. A voice with no voice model, a model made for another model revision, or a model with no takes in
  the film's language cannot narrate, and the narrator picker shows it disabled with the reason."
- §11.1 film mermaid: node 4a (`:1984`) "voxcpm2 — a SAVED voice (mac/cloud): its film example (chosen, else the newest) + stored
  conditioning" → "voxcpm2 — a SAVED voice (mac/cloud): its voice model alone (no take)"; in the `persist` node (`:1989`) "new voice
  conditioning → voice_conditionings" goes, and "saved film: voice check → WAVs → row in one transaction" stays. Validate the block.
- §11.1 casting (`:2026-2041`): "per voice, the WAV of its film example and every stored conditioning" → "per voice, its voice model with
  `use_clip: false`"; the sentences about built conditionings and about a film and a test restoring each other's conditioning go; the reply
  must report every voice's model as loaded or kept, else the film fails.
- §11.1 "A saved narrator never degrades to a preset" (`:2043-2056`): the take reasons → "has no voice model, has a model made for another
  model revision, or has a model with no takes in the film's language"; the ready-save check covers the used VOICES only: a take deleted
  while the film is made does not fail it.
- §11.1 "Testing a saved voice" (`:2058-2075`) → one test per person: a language (those its voice model learned), a fixed sentence, the
  words of one of its takes or the user's own text, one Generate, through the voice model only; the last test per voice is kept
  (`voice_tests`) with its model version; a test that the model changed under is dropped; no model → "Bake the voice model first". A new
  paragraph after it: "**Baking a saved voice.**" — the kit, the confirmed words, unresolved takes block a bake, the bake job (stage,
  percent, Cancel that stops at the next step), one bake at a time, films and tests wait during it, the stored version, the snapshot it
  learned from and the changes since it, Rebake; a deleted take never removes the model; a cancelled, replaced or deleted bake stores
  nothing (t31); a bake that `app` did not start is stopped only after a question. Changed 2026-11-02 (rework, t40): the cancel request is stored with the run,
  so an app restart never stores a cancelled or replaced bake; after a restart, `app` finishes a running bake from its stored snapshot only,
  also when its takes were deleted or a new take still needs its words.
- §13 (`:2521-2545`): the `/voices` card "the film take (…)" → the voice model state (K6); `/voices/{id}`: the header state line, the
  Recordings rows (no Test button, no film-take confirm), the kit rows, the words of each take, Voice model (version, snapshot, changes since,
  Bake / Rebake, the job, the meter), Test (K6). The film-take select, its tombstone and the per-take test go.
- §13.1 (`:2655-2665`): Esc and close do not change; check that each sentence about "the test" still holds for one test per person.
- `docs/ui.md` `/voices` (`:201-234`): the card lines and the route list (the bake, transcript and test routes of K4 and K5 in; the
  `film-example` route and the two per-take preview routes out). `/voices/{id}` (`:235-345`): every text of §2.17 K6 and K7. The picker
  (`:376-410`): a saved voice that cannot narrate is a disabled option with its reason, and a selected one adds the link line of K5.
- `docs/data-model.md` (`:180-216`): the DDL of §2.17 K3 after `voice_previews`; the K2 columns in `voice_examples`; the "no longer used"
  comment on `voice_conditionings`, `voice_film_examples` and `voice_previews`.
- `docs/models.md` "Voice baking" (`:239-370`): the app side — where the model is stored, that films and tests send the adapter alone
  (variant C of the check), stale versus Rebake, and that the check's A and B take the newest take as their prompt (D82).
- Validate every edited mermaid block with the script in `<task-skill>/references/mermaid.md`. Each must print
  `OK`.

**Changed 2026-11-02 (ASR, t42):** lane 30 wrote the Slice B items above (done in the working tree, with the t41 fixes). Where they
describe confirmed words, unresolved takes, the words box, the transcript route or "checked" words, the items below replace them.

Slice B, ASR rework (lane 30b; **added 2026-11-02, ASR, t42**). Line numbers read 2026-11-02 in the working tree. The rule for every
item: a take is free speech, the kit text is a suggestion, ASR in the models service hears the words before a bake, the user types and
confirms nothing, a take ASR cannot use stays and shows why (§2.17 K2). No plan or turn markers in the docs (t41 item 4).

- `docs/design.md` §4.1, after the `AdapterSpec` bullets (`:403-420`): "**The bake's speech recognition**" — `catalog.TRANSCRIBERS` holds
  `whisper-large-v3-turbo` (`openai/whisper-large-v3-turbo` at `41f01f3…`, MIT weights, en and ru, mac only, `cost_mb` 4000 an estimate);
  it is no slot and no user choice; `transcriber_for(profile)` names it; only a bake's transcription stage uses it (§2.17 K2c, D90).
- §5 mermaid, the `models` node (`:540`): "· hears the words of a saved voice's takes in a killable asr child (§8.1)" and the HTTP list
  "· /voice/transcribe"; the `app → models` edge (`:560`): "· /voice/transcribe". Validate the block.
- §5.1: a new bullet after `/voice/bakes` (after `:797`): **`POST /voice/transcribe`** — the request, the reply, the codes of §2.17 K2c,
  the voice gate, BATCH, no database, "the app stores the words". In `/voice/bakes`: the stage row "waiting for the GPU" (`:766`) →
  "`voice` and `asr` are unloaded"; the voice gate bullet (`:784-792`) adds the transcribe calls and `POST /models/asr/ensure`; the
  `ModelsClient` line (`:796-797`) adds `transcribe`.
- §6, the `voice_examples` bullet (`:905-913`): "`recorded`, with the selected reading prompt as `transcript`" → the take holds the words
  ASR heard (`transcript`, with `transcript_model` = the ASR model and revision, or `transcript_error` = why it found none) and the kit
  text it was shown (`kit_id`, a suggestion; NULL for an upload or a recording made before the kit); `transcript_confirmed_at` is unused
  and kept. The `voice_bake_runs` bullet: the snapshot holds the heard words.
- §8.1, after the `voice_train` paragraph (`:1161-1195`): "**The `asr` unit**" — registered once; one Whisper child; counted in the voice
  gate; BATCH; unloaded by a bake before the training lease, so it never shares memory with the training child; a hot reload ends it like
  any child; its cost is an estimate (**Changed 2026-11-02 (ASR plan check, t44):** the log line of a load gives the net free-memory
  change of that request, evictions included, and is never called the model's cost; D103).
- §11.1 "Testing a saved voice" (`:2096-2101`): "the words of one of its takes in that language (labelled checked or not checked)" →
  "what the user said in one of its takes (the words ASR heard at a bake)".
- §11.1 "Baking a saved voice" (`:2124-2154`): the sentences "The user confirms or corrects …", "A take with no words, or with words not
  confirmed, is unresolved … a bake never leaves a take out", "words edited", "or a new take still needs its words" and "the meter shows
  the checked minutes" → a take is free speech; the kit texts are suggestions; the bake first hears the words of every take that has none
  from ASR ("transcribing the takes · take N of M", Cancel between takes), stores them on the take, then writes the snapshot; a take with
  no usable words stays, shows "not used by the voice model: <reason>", is left out and never blocks a bake; a bake needs at least one
  take with heard words; the meter shows the minutes of the takes. **Changed 2026-11-02 (ASR plan check, t44):** the two reasons are
  "the speech recognition found no words in it" (blank or symbols only) and "it holds more than 1000 characters of words"; the docs
  never say "no speech"; a take that arrives while the stage runs is heard before the snapshot (the cutoff is the snapshot's read), and a
  take deleted during the stage is skipped.
- §13 (`:2612-2632`): the take row's words states and "What you said" go; the "not used" text; "Other takes"; the recorder's hint; the
  Voice model tab's "Not used" line and its disabled state; the Test text "Your words in take #N".
- §13.1 (`:2700`): "add a take, confirm its words, delete, bake, test" → "add a take, delete, bake, test".
- `docs/ui.md`: `:229` (the add route's form fields: `transcript` goes); `:232-233` (the transcript route goes); `:252-254` (the count
  "words of 1 take edited" goes); `:261-262` ("m:ss of words checked" → "m:ss recorded"); `:274` ("Your uploads" → "Other takes": uploads
  and recordings with no kit text); `:276-281` (the words states → the not-used text of §2.17 K6); `:284-302` (the recorder's hint; "Save
  sends its id, so the take stores which kit text it read" → "which kit text it showed"); `:304-310` ("What you said" goes: Save posts the
  take as it is); `:316-334` (Voice model: the "Since" counts, the disabled states, the "Not used" line, the stage list with
  "transcribing the takes · take N of M", the meter "Your recordings:"); `:345-355` (the refusals: no list of unresolved takes; the
  "No take can be learned from: <reasons>" text (changed, ASR plan check, t44); the words route sentence goes); `:358-363` (Test: "Your words in take #<id>" for takes with heard words).
- `docs/data-model.md`: `:174` (the `transcript` comment), `:181` (`transcript_confirmed_at`: unused, kept), two new lines
  `transcript_model` and `transcript_error`, `:182` (`kit_id`: the kit text shown, a suggestion), `:185-186` (the two comments: the
  `_ADDED_COLUMNS` list, and "a take with no heard words is heard at the next bake; one with `transcript_error` is left out"), `:238`
  (the snapshot's `transcript` is the heard words).
- `docs/models.md`: the catalog table (`:44`) and the slot table note (`:9-24`) name the bake's ASR model, which is no slot; "Voice baking"
  (`:239-395`): the words come from ASR; the check script's labels (`:350-354`, D104); a new section "### Hearing the words of a take —
  Whisper" before `## Switching a slot` (`:396`): `openai/whisper-large-v3-turbo` @ `41f01f3fe87f28c78e2fbf8b568835947dd65ed9`, MIT
  (Hub `cardData.license`), 808,878,080 parameters, float32 on MPS, `cost_mb` 4000 an estimate (3086 MiB of weights + an allowance), a 30 s
  window, 16 kHz input resampled from 48 kHz, en and ru forced by the take's language, "no new dependency" (`transformers` 5.15.0; `scipy`
  1.18.0 through `librosa`), one child in the models service, unloaded before training, "an estimate; not measured" (§1.10; changed,
  ASR plan check, t44: the load's log line reports the net free-memory change of that request, not a cost);
  "Model download" (`:499-503`): `make models` also fetches the bake's Whisper on mac.
- Validate the §5 mermaid block with the script in `<task-skill>/references/mermaid.md`; it must print `OK`.
- After the lane: `grep -n -i "words checked\|What you said\|check the words\|needs its words\|need their words\|unresolved\|Save words\|(checked)\|not checked\|no speech" docs/`
  finds nothing about takes (D101).

### 2.15 Organize and film clarity (Phase 3; user 2026-11-02, pair t17 + t18)

Goal: per memory, the user sees which films exist, in which language, who narrates them, and who chose that voice (the user, the story or the memory's mood).
The user knows what ↻ Retell does. The user can choose any narrator, a preset or a saved voice, on `/feed`, or leave it to Automatic.
D17 holds: one film speaks presets only or saved voices only. Automatic never picks a saved voice. Nothing predicts the Automatic voice before the film is made (t18).
Every "who narrates" text comes from the film's stored `decisions["cast"]`, never from the current `memory_narrators` row (t18).

**A. Store — `feed/store.py`, `feed/voice.py` (pure, no model)**

```python
# feed/voice.py
@dataclass(frozen=True)
class VoiceLabel:
    name: str               # short: "af_heart", "Grandma", "a saved voice"
    label: str              # full: "af_heart (female, warm)", "Grandma (saved voice)", "a saved voice"

def voice_label(name: str | None, language: str, cast: Sequence[SavedCast] = ()) -> VoiceLabel | None

# feed/store.py
@dataclass(frozen=True)
class FilmNarrator:
    name: str
    label: str
    chosen_by: str          # "user" | "story" | "mood" | "auto" (Automatic, reason not recorded)
    others: tuple[tuple[int, str], ...]   # (passage number from 1, label) per passage voice ≠ the narrator

@dataclass(frozen=True)
class FilmState:
    language: str
    status: str             # the stored row: "ready" | "failed" | "running"
    ready: bool             # status == "ready" and narration_key is not None (the page's own rule, web/app.py:852-854)
    narrator: FilmNarrator | None   # film_narrator(decisions); None = "Voice not recorded"
    error: str | None

def film_narrator(decisions: dict | None) -> FilmNarrator | None
def film_states(conn, group_id) -> dict[str, FilmState]   # one SELECT; only languages that have a row, in LANGUAGES order
```

- `voice_label` (D37): empty → `None`; a token in `cast` → `("<name>", "<name> (saved voice)")`; a `saved:` token not in `cast` → `("a saved voice", "a saved voice")`;
  a speaker of `offered(language)` → `("<name>", "<name> (<gender>, <register>)")`, e.g. `"af_heart (female, warm)"`; else `(name, name)`.
- `film_narrator(decisions)` (t18 read rules):
  - no `cast`, or `cast["narrator_used"]` empty → `None`.
  - `name`, `label`: `cast["narrator_name"]` and `cast["narrator_label"]` when both are set (a Phase 3 film); else `voice_label(narrator_used, decisions["language"])`.
  - `chosen_by`: `"user"` → `"user"`; `"mood"` → `"mood"`; `"story"` or absent (HEAD rows are all Automatic, §1.6) → `"story"` when `narrator_asked_was_real` is true,
    `"mood"` when it is false, `"auto"` when the flag is absent.
  - `others`: per `cast["passages"]` entry whose `used` is set and differs from `narrator_used`: `(passage + 1, used_label or voice_label(used, language).label)`.
    `asked` and `model_cast` are never read: they hold names the build may have rejected (`feed/build.py:325-327`, `:615-621`).
- `feed/voice.py:3-4` docstring: add "The user may choose the speaker on `/feed`: a preset or a saved voice (§11.1)."

**B. Organize memory card — `web/app.py:organize()`, `web/templates/_album_card.html`, `web/templates/organize.html`**

- `organize()` (`web/app.py:659-717`), memories only, in the loop of `:686-693`: `films_by_key[album.key] = {"rows": [...], "retell": "<text>"}` from
  `film_states(conn, group_id)` and `app.state.feed_building` / `feed_queued`. Passed as `"films_by_key"`. `organize.html:51-53` adds
  `{% set films = (films_by_key or {}).get(album.key) %}` before the include.
- One row per `LANGUAGES` entry, in that order: `{"href": "/feed/{key}?lang={code}", "label", "state"}`. `<Language>` = `language_name(code)` ("English", "Russian").
  The first matching line wins (D36):

  | Condition | Link label | State text |
  |---|---|---|
  | `feed_building == (key, code)` | `Open <Language> film` | `being made…` |
  | `feed_queued == (key, code)` | `Open <Language> film` | `waiting to be made…` |
  | `FilmState.ready` | `▶ Play <Language> film` | `ready · <narrator.name> (your choice)` when `chosen_by == "user"`, else `ready · <narrator.name> (auto)`; `ready · voice not recorded` when `narrator` is `None` |
  | no row | `Prepare <Language> film` | `not made yet` |
  | status `failed` | `Prepare <Language> film` | `the last try failed` |
  | status `running` | `Prepare <Language> film` | `interrupted` |
  | status `ready`, no `narration_key` | `Prepare <Language> film` | `no voice` |

- `_album_card.html:22-36`, inside `{% if play_url %}`: a `<ul class="album-films">` with one `<li>` per row:
  `<a class="album-film" href="{{ row.href }}">{{ row.label }}</a> · <span class="album-film-state">{{ row.state }}</span>`, so a row reads like t18's
  "English · ready · af_heart (auto)" (t19). Plain links: a leaf under the grid,
  never boosted (§13.1; `web/templates/base.html:11-16`). They replace the unlabelled "▶ Play" (`:25`). `play_url` stays for the note links (`:50`).
- Retell, same form and route (`POST /feed/{key}/rebuild`, `:29-35`): the button reads `↻ Retell memory` (`↻ Retelling…` while one runs, disabled as today).
  The `title` tooltip goes. Under the button, `<p class="album-retell-note">`:
  - a row exists: `Rewrites the title, description and photo order. Makes no film. Deletes the films in: English, Russian.` (the languages with a row, `LANGUAGES` order);
  - no row: `Rewrites the title, description and photo order. Makes no film. No film to delete.`
- The global "Rebuild memories" (`organize.html:34-36`) keeps its label. The chat card (`web/app.py:1590-1592`) passes no `play_url` and no `films`, so it does not change.
- `feed_rebuild` (`web/app.py:1004-1039`) does not change. Its docstring already says "Retell THIS memory" (`:1006`).
- `web/static/app.css`: `.album-films`, `.album-film`, `.album-film-state`, `.album-retell-note`. `.album-play` (`:377`) goes when nothing uses it.
- In-place refresh of the film rows (pair t19, `<base>/.pair/decisions.md:204-216`). Close on `/feed` is `history.back()` (`web/templates/feed.html:236-249`),
  so the grid returns from bfcache (`_album_card.html:23-24`), with the card as it was before the film was made. No page reload, GET `/organize` only, no new route, no build trigger:
  - Each memory card's film rows, retell form and retell note sit in one keyed block: `<div class="album-film-block" id="films-{{ album.key }}">`.
    The notes (`_album_card.html:37-55`) stay outside it.
  - While a row of that card is building or queued, the block polls Organize, as the memories bar does (`organize.html:31-32`):
    `hx-get="/organize?by=memories" hx-select="#films-{{ album.key }}" hx-trigger="every 1500ms" hx-swap="outerHTML"`. `organize()` sets `films["polling"]`.
    The block that comes back has no `hx-trigger` once the work ends, so the polling stops; it also stops when the element leaves the DOM.
  - One `pageshow` listener on `window`, registered once behind a flag (`window.__filmBlocksWired`, the pattern of `web/static/chat.js:80-83`), in an inline
    `<script>` at the end of `organize.html`. It acts only on `e.persisted` AND when the document has `.album-film-block`: one `fetch(location.href, {cache: "no-store"})`,
    then each `#films-<key>` block is replaced (`outerHTML`) and passed to `htmx.process`. A failed fetch keeps the old content. Nothing else on the page changes.
  - `web/static/chat.js:80-83` also reloads every bfcache-restored page, because its window listener survives a boosted move from Chat to Organize
    (`web/templates/base.html:11-16`). The user chose to fix it now (2026-11-02, "Fix now (reviewer)"): line `:82` reloads only when `e.persisted` AND
    `document.getElementById("chat-q")` exists. The flag `window.__chatPageshowWired` stays. So Chat → Organize → film → Close keeps Organize and its open notes.

**C. `/feed` — `web/app.py:feed()`, `_narrator_picker`, `_store_narrator`; `web/templates/feed.html`**

- `feed()` (`web/app.py:785-880`) reads `states = film_states(conn, group_id)`.
  `"languages"` = `[(code, LANGUAGE_DISPLAY[code], done)]`, `done` = `states[code].ready` and `(key, code)` is neither building nor queued.
  `"this_film"` = `film_narrator(presentation.decisions)` for the ready film. `"picker"` replaces `narrators`, `narrator`, `narrator_unavailable`, `voice_copy` (`:865-868`).
- Language form (`feed.html:42-47`): `<span class="feed-lang-label">Film language</span>` first; each button reads `{{ label }}` plus ` ✓` when `done`.
- Ready film, at the top of the `.feed-regen` form (`:60`), `<Language>` = `language_name(language)`:
  - `<p class="feed-this-film">This film: <Language> · narrated by <label> — <why></p>`, where `<why>` by `chosen_by` is:
    `user` → `your choice`; `story` → `picked automatically by the story`; `mood` → `picked automatically using the memory's mood`; `auto` → `picked automatically`.
    `this_film` is `None` → `This film: <Language> · Voice not recorded`.
  - When `this_film.others` is not empty: `<p class="feed-other-voices">Other voices: passage 3 → am_adam (male, neutral); passage 5 → bm_george (male, deep)</p>`
    (entries joined with "; ", D42).
  - The recorded note "a note asked for a different voice and none was cast" keeps its place in `missing` (`web/app.py:871-872`, `feed.html:87-89`). It shows only when the build recorded it. No change.
- The picker shows the stored choice. After an Automatic build no row exists, so "Automatic" stays selected while "This film" names the voice it picked (D18, t18).
- `_narrator_picker(ctx, group_id, language) -> dict` (`web/app.py:882-908`):
  `{"presets": [(value, label)], "saved": [(value, label)], "selected": str, "unavailable": (group, value, label) | None, "voice_copy": bool}`.
  - `presets` = `("preset:" + s.name, f"{s.name} — {s.gender}, {s.register}: {s.note}")` per `offered(language)`, on every profile.
  - `saved` = castable voices when `cloning_entry_for(profile, language)` exists, else `[]`, as today.
  - Stored choice: none → `""`. A preset that `resolve(language, name)` accepts → `"preset:<catalog name>"`.
    A preset it rejects → `("preset", "preset:<name>", "«<name>» — not offered for <Language> films")` (D38). A saved choice → today's rules (`:893-908`), group `"saved"`.
- The macro (`feed.html:17-34`) renders in BOTH forms, always:

  ```html
  <label class="feed-narrator">Narrator for the next film
    <select name="narrator">
      <option value="">Automatic — the app picks a preset voice for this memory</option>   <!-- selected when picker.selected == "" -->
      <optgroup label="Preset voices"> one option per preset; a "preset" unavailable option here, selected </optgroup>
      <optgroup label="Saved voices"> the castable voices; a "saved" unavailable option here, selected;
        with no voice copy: <option disabled>Voice copy not available on this device</option> </optgroup>
    </select>
  </label>
  ```

  - The "Saved voices" group is rendered only when it has an option, or when voice copy is not available (D40).
  - Under the select, in this order, each a `<p class="feed-note">`:
    1. always (t18): `The story picks the voice; otherwise the memory's mood guides the choice. Saved voices are never picked automatically.`
    2. when `picker.voice_copy`: `Add a voice under Organize → Saved voices.` (text, no link). It replaces `feed.html:163-165`.
    3. the bracket line (t18), by the stored choice (D40): a saved narrator (the selected value is a saved id or `deleted`) →
       `A [voice] note can request another saved voice for one passage.`; else (Automatic or a preset) → `A [voice] note can request another preset voice for one passage.`
  - The macro holds these lines, and only one form shows at a time (`feed.html:56`, `:108`), so each page shows them once.
  - The option markup keeps today's shape (`<option value="…" selected>label</option>`, `:24-30`), so the test helper still parses it.
- `_store_narrator(group_id, language, narrator)` (`web/app.py:911-940`), checked BEFORE any build starts or queues (§2.10 style):
  - `preset:<name>`: `found = resolve(language, name.strip())`. Found → `set_narrator(conn, group_id, language, preset_speaker=found)`.
    A speaker of another language (`resolve(code, name)` for another `code` in `LANGUAGES`) → **409** `«<name>» cannot narrate in <Language>. Choose another narrator.` (t19)
    Else (unknown, or `preset:` with no name) → **404** `Unknown preset voice.`
  - `""`, `"deleted"` and a saved id: unchanged (§2.10). A saved id clears `preset_speaker` (D).
  - `_prepare` (`web/app.py:1072-1089`) does not change. The picker is now always on the page, so both forms always post the field.
  - The 4xx reply shows its `detail` under the form, and the page stays (the existing submit, `feed.html:265-294`).
- Ready-screen layout (pair t20). Cause: the Phase 2 select widened the fixed `.feed-regen` form (`right: 4rem`, grows left) into ↓ Download's own fixed
  offset (`right: 11.5rem`), and no CSS came with the select (§1.6). Fix in `feed.html` and `feed.css`, no fixed offsets between controls:
  - One header at the top of `<body>`: `<header class="feed-head">`. It is the only positioned element of the controls (`position: fixed; top: 1rem;
    left: 1rem; right: 1rem; z-index: 10`, so it never exceeds the viewport). Its children are in normal flow (D43). The `z-index` is needed:
    `.feed-start` (`feed.css:49-62`, `feed.html:76`) is a later fixed full-screen element and would otherwise paint over the header (t21).
  - `<div class="feed-actions">` (flex, `flex-wrap: wrap`): the `.feed-lang` form, ↓ Download (`#download`, ready only), the × `.feed-close`, in that order.
    The × moves into it from `feed.html:37-38`; it is rendered in every state, as today.
  - `<div class="feed-settings">` below the action row, ready only: the `.feed-regen` form with "This film", "Other voices", the full picker and the help lines.
    The help lines stay visible (D40), never inside a `<details>`. `max-height: 45vh; overflow-y: auto` (D43). The 4xx `.feed-refused` line still goes after the
    form (`feed.html:281-288`), so it lands inside this region. The region is left-aligned: the notes sheet and the bubbles own the right column
    (`feed.css:354-358`, `:391-394`) (t21).
  - `@media (max-width: 66rem) { .feed-body:has(.feed-settings) .feed-notes { max-height: calc(55vh - 7rem); } }`: on a narrow window the notes sheet
    (`feed.css:391-404`, `max-height: 70vh`) would otherwise cover the settings region (t21).
  - `.feed-narrator select { min-width: 0; max-width: 100%; }`; option and label text wraps; the region is at most `min(36rem, 100%)` wide (D43).
  - `.feed-lang`, `.feed-regen`, `.feed-download` and `.feed-close` lose `position: fixed` and their `top/left/right` offsets (`feed.css:71-75`, `:111-119`,
    `:135-141`, `:322-326`). Ids, classes, the POST forms, the Esc/close script and the in-film `location.replace` moves stay (`feed.html:232-296`).
  - Each opacity rule stays on its own control, never on `.feed-head`, `.feed-actions` or `.feed-settings` (`feed.css:86-87`, `:155-156`, `:339-347`,
    `:549-557`, `:572-576`), so an exporting Download stays fully visible and a stale Regenerate stays lit.
  - The empty parts of the header pass clicks through to the film and to ▶ Play: `.feed-head { pointer-events: none }`; the action controls AND
    `.feed-settings` itself get `pointer-events: auto`, so the region scrolls by touch, wheel and scrollbar (D43, t21).

**D. Storage — `db/schema.sql`, `db/connection.py`, `feed/saved_voices.py`**

```sql
-- Who narrates one memory's film in one language, when the user chose a narrator. No row = Automatic.
-- voice_id = a saved voice. preset_speaker = a catalog speaker name (the route checks it with feed.voice.resolve).
-- Both NULL = the chosen saved voice was deleted: the film fails until the user picks again (D33).
CREATE TABLE IF NOT EXISTS memory_narrators (
  group_id       INTEGER NOT NULL REFERENCES groups(id) ON DELETE CASCADE,
  language       TEXT NOT NULL,
  voice_id       INTEGER REFERENCES saved_voices(id) ON DELETE SET NULL,
  preset_speaker TEXT,
  PRIMARY KEY (group_id, language),
  CHECK (voice_id IS NULL OR preset_speaker IS NULL)
);
```

- `db/connection.py:62-69`: append `("memory_narrators", "preset_speaker", "TEXT")` to `_ADDED_COLUMNS`. The user's table (§1.6) gains the column on the next start,
  without the CHECK (`_ensure_columns` adds the type only). No index on the new column: `schema.sql` runs before `_ensure_columns` (`db/connection.py:97-99`),
  so an index would fail on an old table. `SCHEMA_VERSION` stays 14 (D39). Commit split: our line at `:68-69` sits next to the user's hunk at `:73-85`; use `git add -p` → `s`, else `e`.
- `feed/saved_voices.py`:
  - `NarratorChoice(voice_id: int | None, preset_speaker: str | None = None)`. `NarratorChoice(5)` still equals today's value, so existing tests keep passing.
  - `read_narrator` selects both columns.
  - `set_narrator(conn, group_id, language, voice_id: int | None = None, *, preset_speaker: str | None = None)`, one `BEGIN IMMEDIATE` as today:
    both given → `VoiceError("A narrator is a saved voice or a preset voice, not both.")` and no write; a blank `preset_speaker` → `VoiceError`;
    `preset_speaker` → upsert `voice_id = NULL, preset_speaker = ?`; `voice_id` → today's check (`:621-650`), then upsert `voice_id = ?, preset_speaker = NULL`;
    neither → delete the row (Automatic), as today. The store does not check the name against the catalog; the route and the build do.
  - The voice drain's `ON DELETE SET NULL` touches only rows with a `voice_id`. A preset row stays. The t10 tombstone (both NULL) keeps its meaning.

**E. Build — `feed/build.py`**

- `saved_film = chosen is not None and chosen.preset_speaker is None`. A tombstone row is a saved film, so it fails as today.
  Every "saved film" test of §1.6 (`:237-238`, `:265`, `:307`, `:336`, `:350`, `:395`, `:401`, `:494`) uses `saved_film`.
- Preset film, right after the read (`:236`), before the story pass: `preset = resolve(language, chosen.preset_speaker)`.
  `None` → `_fail(…, "The preset voice «<name>» is not offered for <Language> films. Choose another narrator.")`, with no speak call (D38).
- The story pass gets `catalogue_for_prompt(language)` for a preset film, as for Automatic.
- "recording the voice" (`:307-318`), preset film: `speaker = preset` (it wins over `story.narrator`);
  `passage_voices = _passage_voices(story, language, speaker, bracketed)` (the default preset resolver, so a bracket casts another preset);
  `narrator_asked_was_real = bool(resolve(language, story.narrator))`. No references; the speak call, the WAV writes and the ready save follow the Automatic path (`:364-366`, `:494-495`).
- `decisions["cast"]` (`:328-338`):
  - `narrator_chosen_by` (t18): `"user"` for any stored choice (preset or saved); Automatic → `"story"` when `resolve(language, story.narrator)` returned the speaker,
    `"mood"` when `pick_speaker` chose it (`:316`). Casting itself does not change.
  - new `"narrator_name"` and `"narrator_label"` = `voice_label(speaker, language, saved_cast)`; each `passages` entry gains `"used_label"` = `voice_label(used, …).label`.
    The film keeps these strings, so a later delete or rename does not change what it says.
  - `offered` lists the presets for a preset film.
- Comment `:300-301` → "The language picks the MODEL (in the models service). The SPEAKER is the user's choice when they made one on /feed, else the story
  pass's cast, else the memory's mood (§11.1)."

**F. Error paths (Phase 3)**

| Case | Where | Result |
|---|---|---|
| `preset:<name>` names a speaker of another language | Prepare | 409 `«<name>» cannot narrate in <Language>. Choose another narrator.`; no build |
| `preset:<name>` unknown, or `preset:` with no name | Prepare | 404 `Unknown preset voice.`; no build |
| a stored preset the catalog no longer offers | `/feed` | one more option, `selected`: `«<name>» — not offered for <Language> films` |
| the same, at the build | build | `failed` "The preset voice «<name>» is not offered for <Language> films. Choose another narrator."; no speak call; never Automatic |
| `voice_id` and `preset_speaker` both given | store | `VoiceError`; no write |
| a row with both columns set | DB (fresh schema) | the CHECK raises `sqlite3.IntegrityError` |
| a film made before Phase 3 | card / `/feed` | the label from `narrator_used` ("a saved voice" for a token); who chose it by the t18 read rules (§2.15 A) |
| a row with no `cast`, or `narrator_used` null | card / `/feed` | "ready · voice not recorded" / "This film: <Language> · Voice not recorded" |

**G. Binding rules**

- `ivms777/CLAUDE.md:253-262`: the voice MODEL stays automatic. `AUTOMATIC_SLOTS` and Settings do not change. The narrator is a per-film choice on `/feed`.
- `ivms777/CLAUDE.md:236-251`: the preset list comes from the catalog (`offered`), never from a list in a template.
- `ivms777/CLAUDE.md:51-66`, `:68-118`: the choice is stored per memory and language (D18). Card links are plain links to the leaf; in-film moves still use `location.replace` (`feed.html:251-294`).

**H. Saved voices — recording tip and likeness (pair t20)**

- `web/templates/_voices_list.html:57-72`: one `<p class="voices-tip muted">` inside each available `.voice-add` ("Add an example"), before Record and Upload (D44, t21;
  `voices.html` has no "Add an example" area):
  `Record clean, normal speech in a quiet room. Only the newest example in each language is used.` No promise of likeness or of a length.
  **Changed 2026-11-02 (Phase 4, §2.16 F):** the second sentence becomes "Films use one example per language: the newest, unless you choose one with
  Use for films."
- Production does not change: VoxCPM2 reference mode (D16), `CONTROLS` and the clip limits (§2.5, §2.7), the newest same-language example (D19). No transcript upload or edit UI.
  (**Changed 2026-11-02, Phase 4:** the newest stays the default; "Use for films" can choose another example, §2.16 B.)
- A high-likeness A/B (ref_continuation) is not in this task: the user answered "Not now" (`## Decisions`, `## Open items outside scope`).
  (**Changed 2026-11-02, Phase 5:** the working tree uses ref_continuation for a take with words, §1.8; Phase 5 bakes a voice model, §2.17.)

**I. Reading texts for a recording (pair t20, resumed; `<base>/.pair/decisions.md:224-233`)**

The user asked why every recording reads the same text (`## Decisions`). Voice copy does not need those exact words: VoxCPM2 runs in reference mode
and never reads the transcript (D16). So the recorder offers several texts. Nothing here claims that another text gives a better likeness, and each film
still uses only the newest example of its language (D19). (**Changed 2026-11-02, Phase 5:** the child now reads the words of a recorded take
(`ref_continuation`, §1.8), and Slice B asks the user to confirm them, D71; the four texts become the "normal" group of the kit, D70.)

- `feed/saved_voices.py:35-40`: `RECORDING_PROMPTS: dict[str, dict[str, str]]` = language → ordered `{prompt_id: text}`. The first id of each language
  holds today's text. Each text reads in about 12-15 s (the clip length of findings §10, `<base>/.scratch/research/findings.md:315`; the recorder stops at 30 s,
  `web/static/voices.js:167`, `:226`). The Russian texts avoid first-person past-tense verbs (D23). No text uses or paraphrases the voice_check sentences
  EN-1..5 / RU-1..5 (`scripts/voice_check.py:50-71`: the lake, fog, the dog, Julia, a July date, the cold, the road through the hills, the porch and stars).
  The ids name the language, so an id of the other language is refused (D45):

  ```python
  RECORDING_PROMPTS: dict[str, dict[str, str]] = {
      "en": {
          "en-kitchen": "The quick morning light crept across the kitchen table. I poured a cup of coffee, opened the window, "
                        "and listened to the city slowly waking up below.",
          "en-market":  "On Sunday mornings the market fills the square. Someone sells warm bread from a cart, a child counts "
                        "coins for apples, and the clock above the bakery runs ten minutes slow.",
          "en-rain":    "Rain tapped on the roof all afternoon, so we stayed inside. My sister found an old board game in the "
                        "cupboard, and we argued about the rules over cups of hot tea.",
          "en-train":   "The train was quiet except for the steady rattle of the wheels. A man across the aisle read a thick "
                        "paperback, and small towns slid past the window one after another.",
      },
      "ru": {
          "ru-kitchen": "Утренний свет медленно скользил по кухонному столу. За окном понемногу просыпался город, "
                        "а в комнате пахло свежим кофе и тёплым хлебом.",
          "ru-market":  "По воскресеньям рынок на площади открывается рано. Продавец предлагает тёплый хлеб, ребёнок "
                        "пересчитывает монеты на яблоки, а часы над пекарней отстают на десять минут.",
          "ru-rain":    "Весь день по крыше стучал дождь, и мы остались дома. Сестра нашла в шкафу старую настольную игру, "
                        "и мы долго спорили о правилах за чашкой горячего чая.",
          "ru-train":   "В поезде было тихо, только мерно стучали колёса. Пассажир напротив читал толстую книгу, а за окном "
                        "один за другим проплывали маленькие города.",
      },
  }
  ```

  Lengths (planner check, 2026-11-02): today's texts are 149 / 135 characters; the new ones are 162-172 (en) and 142-168 (ru). None holds a word of the list above.
- `add_example(…, prompt_id: str | None = None)`. It replaces the fixed assignment at `feed/saved_voices.py:537`. After the `source` check, before the size and WAV checks:
  - `recorded` + a `prompt_id` of `RECORDING_PROMPTS[language]` → `transcript` = that text;
  - `recorded` + `None` (an old client) → the first text of the language;
  - `recorded` + an unknown id, or an id of the other language → `VoiceError("Unknown reading text for <Language>: '<id>'.")` → 400 (`_refuse`, `web/voices_api.py:153-155`);
  - `uploaded` → `transcript = None`; a posted `prompt_id` is ignored (D45).
  Old rows, the schema, inference, D16, D19, `CONTROLS` and the limits do not change. `scripts/voice_check.py` and its own prompt copy (`:74-79`, `:386-387`) do not change.
- `web/voices_api.py`: `voices_add_example` (`:208-244`) gains `prompt_id: str = Form(default="")` and passes `prompt_id=prompt_id or None`. The page context
  (`:164`) keeps `"prompts": RECORDING_PROMPTS`, now nested.
- `web/templates/voices.html` recorder panel (`:31-46`), still outside `#voices`:
  - one `<p class="recorder-prompt" data-lang="<code>" data-prompt-id="<id>" hidden>` per text;
  - a preparation box `<div class="recorder-prepare" hidden>` with `<button type="button" class="recorder-next">Different text</button>` and
    `<button type="button" class="recorder-start">Start recording</button>`. Consent, Stop, the timer, the preview and Save / Discard stay.
- `web/static/voices.js` (today Record starts capture at once, `:137-170`):
  - Record (after the consent check, `:124-134`) opens the panel in a PREPARED state: it binds the voice and the language of that click and shows one text,
    the one after the text of this language that was last saved in this page session (the first text when none was saved). No microphone yet.
  - "Different text" shows the next text of that language, in order, and wraps to the first.
  - "Start recording" runs today's capture (`getUserMedia` → `MediaRecorder`). From then on the voice, the language and the prompt id are frozen:
    the preparation box hides, and the same text stays on screen through the live recording and the unsaved take.
  - Save posts `prompt_id` with `language`, `source`, `consent` and `file` (`:247-253`). Only a successful Save (`:254`) marks its text as the last one used.
  - State rules (t21): Start consumes the PREPARED state. Discarding a live recording or an unsaved take closes the panel and clears its target
    (`voices.js:155-162`, `:240-245`). Esc from PREPARED closes it. An Upload from PREPARED ends it: the box hides and the target clears (`:211-212`, `:266`).
    Record on another voice, while PREPARED, binds that voice and its language.
  - Esc (`:290-300`), in this order: a live recording → stop and discard it; an unsaved take → discard it; a prepared panel → close it;
    a focused text field → blur it; else close the page (`ivms777/CLAUDE.md:112-115`).
- The §2.15 H tip stays as it is. No claim about length or likeness anywhere. (**Changed 2026-11-02, Phase 4:** its second sentence changes, §2.16 F.)

### 2.16 Voice workbench (Phase 4; user 2026-11-02, pair t24)

Goal: on `/organize/voices` the user hears each example and a test made from it, side by side, and chooses per language which example films use.
Examples never combine. A test IS the bake. Nothing promises likeness or speed. This section extends §2.9 and §2.15 H-I; where they differ, Phase 4
wins. D16, D17, `CONTROLS`, the clip limits (§2.7), the outbox (D33) and D34 stay. The models service, the child, the catalog and
`scripts/voice_check.py` do not change.

**A. Storage — `db/schema.sql`, two new tables after `voice_conditionings` (D46, D50)**

```sql
-- Which example films use for one saved voice in one language, when the user chose one ("Use for films", design §6, §11.1).
-- No row = the newest example of that language. example_id NULL = the chosen example was deleted: films with this voice in
-- this language fail until the user chooses another. The app never picks another example in its place.
CREATE TABLE IF NOT EXISTS voice_film_examples (
  voice_id   INTEGER NOT NULL REFERENCES saved_voices(id) ON DELETE CASCADE,
  language   TEXT NOT NULL,                                          -- feed.language.LANGUAGES
  example_id INTEGER REFERENCES voice_examples(id) ON DELETE SET NULL,
  chosen_at  TEXT NOT NULL,
  PRIMARY KEY (voice_id, language)
);
CREATE INDEX IF NOT EXISTS voice_film_examples_by_example ON voice_film_examples(example_id);
-- The last test of one example ("▶ Generated", design §6, §11.1): a WAV in data_dir/voices to compare with the example.
-- Never used by a film. Its file leaves only through the voice outbox drain.
CREATE TABLE IF NOT EXISTS voice_previews (
  example_id      INTEGER PRIMARY KEY REFERENCES voice_examples(id) ON DELETE CASCADE,
  text_kind       TEXT NOT NULL CHECK (text_kind IN ('reading', 'sentence', 'custom')),
  text_id         TEXT,               -- the test-sentence id ('EN-2') when text_kind = 'sentence'; else NULL
  text            TEXT NOT NULL,      -- the words the model was asked to say
  model_key       TEXT NOT NULL,      -- the reply's model
  revision        TEXT,               -- the reply's revision
  conditioning    TEXT NOT NULL CHECK (conditioning IN ('built', 'restored')),
  elapsed_seconds REAL NOT NULL,      -- wall time of the /speak call
  seconds         REAL NOT NULL,      -- length of the test audio
  storage_key     TEXT NOT NULL,      -- PCM-16 WAV in data_dir/voices (content-addressed)
  created_at      TEXT NOT NULL
);
```

- Both tables are new, so `migrate` creates them on the next start of an existing DB (§1.7). No `_ADDED_COLUMNS` entry (no existing table gains a
  column). `SCHEMA_VERSION` stays 14 (D39). The CHECKs sit in new tables only, so D39's "no CHECK on an existing table" holds.
- The `voice_file_deletions` comment adds "or a test WAV". A test WAV's key is `content_key(sha256(wav), ".wav")` in `data_dir/voices`.
- A voice delete cascades its `voice_film_examples` rows; an example delete sets `example_id` NULL (the tombstone). Both happen inside the drain's
  cascade (`feed/saved_voices.py:375-383`) through the foreign keys (`PRAGMA foreign_keys=ON`, §1.7). No code is needed for them.

**B. Store — `feed/saved_voices.py` (D47, D48, D50)**

```python
@dataclass(frozen=True)
class FilmExample:
    example: VoiceExample | None   # the example films of this voice use in the language; None = none can
    chosen: bool                   # True = the user's "Use for films"; False = the newest (the default)
    problem: str | None            # None | "none" (no usable example) | "deleted" | "deleting" (the chosen one)

@dataclass(frozen=True)
class VoicePreview:
    example_id: int
    text_kind: str                 # "reading" | "sentence" | "custom"
    text_id: str | None
    text: str
    model: str
    revision: str | None
    conditioning: str              # "built" | "restored"
    elapsed_seconds: float
    seconds: float
    storage_key: str
    created_at: str

def film_example(conn, owner_id, voice_id, language) -> FilmExample
def set_film_example(conn, owner_id, voice_id, language, example_id) -> None
def reference_for(conn, storage, example: VoiceExample) -> dict          # moved here from feed/build.py:_reference; the same dict
def save_preview(conn, storage, example_id, *, text_kind, text_id, text, model, revision, conditioning,
                 elapsed_seconds, audio: bytes) -> VoicePreview | None   # None = the example is gone or pending; nothing stored
def previews(conn, owner_id) -> dict[int, VoicePreview]                # example_id -> its last test
def read_preview_audio(storage, preview: VoicePreview) -> bytes
```

- `film_example` is the ONE resolver (D47); owner-scoped. With a `voice_film_examples` row for (voice, language):
  `example_id` NULL → `FilmExample(None, True, "deleted")`; the example or its voice is pending → `FilmExample(None, True, "deleting")`;
  else → `FilmExample(<that example>, True, None)`, also when newer examples exist.
  With no row → `example_for(...)` (the newest non-pending, unchanged) → `FilmExample(<it>, False, None)`, or `FilmExample(None, False, "none")`.
  A chosen example that cannot be used is never replaced by another one.
- `castable` pairs each non-pending voice with `film_example(...).example` and skips `None`. So the build, the `/feed` picker and the Prepare check
  (§1.7 callers) follow the choice without a change of their own. `example_for` keeps its meaning (the newest) and its tests.
- `set_film_example`: `language` not in `LANGUAGES` → `VoiceError`. Then one `BEGIN IMMEDIATE`: the voice exists for the owner and is not pending,
  else `VoiceDeleting("This voice is being deleted.")`; the example belongs to that voice, else `VoiceError("Unknown example.")`; its language is
  `language`, else `VoiceError("This example is in <Language>, not <Language>.")`; it is not pending, else
  `VoiceDeleting("This example is being deleted.")`; then upsert `(voice_id, language, example_id, chosen_at)`. This also ends a tombstone.
- `reference_for` = the body of today's `_reference` (`feed/build.py:568-577`). The film and the test send the same dict for the same example.
- `save_preview`: `_write_voice_file` (`:438-481`) with the `parent_ok` of `save_conditioning` (the example exists and is not pending) and a
  `write_row` that upserts by `example_id`. An old `storage_key` that differs goes to `voice_file_deletions` with `deletion_id` NULL, as in
  `save_conditioning` (`:754-759`). `seconds = duration_seconds(audio)` (`storage/wav.py:60`).
- `previews`: one SELECT of `voice_previews` joined to `voice_examples` and `saved_voices` on `owner_id`.
- Drain: `_cascade_one` (`:358-367`) adds `UNION SELECT p.storage_key FROM voice_previews p JOIN voice_examples e ON e.id = p.example_id
  JOIN saved_voices v ON v.id = e.voice_id WHERE {scope}` (the args repeat three times). `_remove_one` (`:404-408`) adds
  `UNION ALL SELECT 1 FROM voice_previews WHERE storage_key = ?`. The module docstring (`:1-15`) says test WAVs follow the same two rules.

**C. Casting — `feed/build.py` (D47, D52)**

- `_reference` goes; the build calls `reference_for` (`:373-374`).
- `_saved_cast` (`:542-565`), when the chosen voice is not castable, reads `found = film_example(conn, owner_id, voice.id, language)`:
  `"deleted"` → "The <Language> example of «<name>» chosen for films was deleted. Choose another under Organize → Saved voices, or choose another
  narrator."; `"deleting"` → "The <Language> example of «<name>» chosen for films is being deleted. Choose another under Organize → Saved voices,
  or choose another narrator."; else today's "has no <Language> example" text. `_fail` as today; no speak call.
- New `voice_turn(lock: threading.Lock | None, progress, cancelled, fraction: float)`, a context manager. `None` → it does nothing. Else, in this
  order (**changed 2026-11-02, t25**): `cancelled()` → `Cancelled("waiting for the voice model")` before the first acquire; then
  `lock.acquire(timeout=1.0)` in a loop, where the first miss reports `progress("waiting for the voice model", fraction)` and each miss checks
  `cancelled()` and raises the same `Cancelled`; after the acquire succeeds, `cancelled()` once more → release and raise. The body runs in a
  `try`, and the lock is released in `finally`, whatever the body raises.
- `build_presentation(…, voice_lock: threading.Lock | None = None)`. One `with voice_turn(voice_lock, progress, cancelled, stages["recording the
  voice"]):` covers, for preset and saved films (**changed 2026-11-02, t25**): the references read (`:370-379`), the speak block (`:380-401`) and
  `_keep_conditioning` (`:402-405`). So a test that waits for this film reads the blob the film stored, and a film that waits for a test reads the
  test's blob (restored either way). The cast and its examples are fixed before the wait (`_saved_cast`, `:248-250`) and do not change during it.
  The `try … except Exception → _fail` blocks stay inside the `with`, so a cancel during the wait is a cancel, not a failed film, and a `_fail`
  inside releases the lock. `started` is read after the lock is held, so `wall_seconds` does not count the wait.

**D. The test — `feed/voice_preview.py` (new; thin client, no model; D49, D51)**

```python
PREVIEW_SENTENCES: dict[str, dict[str, str]]   # {"en": {"EN-1": …, …, "EN-5": …}, "ru": {"RU-1": …, …}} = scripts/voice_check.py:47-71
CUSTOM_TEXT_MAX = 300
TEXT_KINDS = ("reading", "sentence", "custom")

@dataclass(frozen=True)
class PreviewText:
    kind: str                # "reading" | "sentence" | "custom"
    text_id: str | None      # the sentence id; None for "reading" and "custom"
    text: str

@dataclass(frozen=True)
class PreviewJob:
    voice_id: int
    example_id: int
    language: str            # the example's language
    text: PreviewText
    revision: str            # cloning_entry_for(profile, language).source.revision when the test starts

class PreviewFailed(VoiceError): ...   # the reason the test panel shows

def preview_text(example: VoiceExample, kind: str, text_id: str, custom: str) -> PreviewText
def run_preview(conn, client, voices: Storage, owner_id: int, job: PreviewJob, *,
                voice_lock=None, job_lock=None, on_saved=None, progress=None, cancelled=None) -> VoicePreview
# job_lock: the VoicePreviews lock (the publication barrier, changed 2026-11-02, t25); on_saved(preview) runs while it is held
```

- `preview_text`, each refusal a `VoiceError` (→ 400):
  - `"reading"`: a recorded example with a `transcript` → `PreviewText("reading", None, transcript)`; else "This example was uploaded, so it has
    no reading text. Choose a test sentence or type your own.";
  - `"sentence"`: `text_id` in `PREVIEW_SENTENCES[example.language]` → that sentence; else "Unknown test sentence for <Language>: '<id>'.";
  - `"custom"`: `custom.strip()`; empty → "Type a text to test."; longer than `CUSTOM_TEXT_MAX` → "A test text can have at most 300 characters.";
  - any other kind → "Unknown kind of test text.".
- `run_preview`, in this order. `progress(stage, fraction)` works as the film's. It imports `CLONE_RTF_ESTIMATE`, `Cancelled`, `_Ticking`, `_reason`
  and `voice_turn` from `feed.build` (D55):
  1. `with voice_turn(voice_lock, progress, cancelled, 0.02):` holds steps 2-4.
  2. "reading the example" (0.05): re-read the example (`get_voice`; it is in `voice.examples` and not `deleting`), else
     `PreviewFailed("The example was deleted, so the test was dropped.")`. `ref = reference_for(…)`, under the lock, so a test right after a film
     restores that film's blob. `OSError` → `PreviewFailed("The example could not be read (<reason>).")`.
  3. "generating the voice" (0.10 → 0.97): **changed 2026-11-02 (t25, round 2):** first `cancelled()` → `Cancelled`, just before the call and
     OUTSIDE the `try` below (`Cancelled` is a `RuntimeError`, `feed/build.py:163`, so the broad handler would turn it into a failure). Step 2
     reads the WAV and the blobs after the post-acquire check; without this check a cancel there would still start a full synthesis that holds
     `voice_calls`. Then, inside `_Ticking(progress, "generating the voice", (0.10, 0.97), len(text) / 14 * CLONE_RTF_ESTIMATE)`, a `try` around
     `reply = client.speak(job.text.text, job.language, token(job.voice_id), None, None, references={token(job.voice_id): ref})` only:
     any exception → `PreviewFailed(f"The models service could not make the test ({_reason(exc)}).")`. `elapsed` = the call's wall time.
  4. A `built` entry with a blob → `save_conditioning(conn, voices, example.id, model, revision, blob)`, also when the test was cancelled (D51);
     an error here is logged, never raised. This is the bake.
  5. "saving the test" (0.97), after the voice lock is released. **Changed 2026-11-02 (t25):** the whole step runs while `job_lock` is held
     (the publication barrier; `None` → no lock, for unit tests): `cancelled()` → `Cancelled`; `reply["revision"] != job.revision` →
     `PreviewFailed("The voice model changed while the test ran. Generate it again.")`; `save_preview(…)` returns `None` →
     `PreviewFailed("The example was deleted, so the test was dropped.")`; then `on_saved(preview)`. No model call runs while `job_lock` is held:
     only the run check, one SQLite write and one file write.
  6. It returns the `VoicePreview`.
- It never writes `memory_narrators` or `voice_film_examples`, and never starts a film. Because `start` and `cancel` take the same `job_lock`
  (§2.16 E), a cancel or a newer Generate either comes first (the run check fails, nothing is stored) or comes after the test is stored and
  marked `done`. A test is never stored after the `cancel()` or `start()` that superseded it has returned.
- **Cancel is logical, not a GPU stop** (changed 2026-11-02, t25 round 2; D51). The cancel route never waits for synthesis; it may wait for a
  publication already in progress (step 5 holds the job lock for one `BEGIN IMMEDIATE` write — busy timeout 5 s, `db/connection.py:29` — and one
  file write). A cancel seen before `client.speak` is sent → no model call. Once the call is sent, nothing stops it: `TorchWorker.call` holds its
  `_lock` across `recv`, and `stop` takes the same lock (`modelsvc/torch_process.py:169-186`); the service has no cancel route (§1.7); `app`
  never calls `model_unload` or `stop`. `ModelsClient` supplies the 1800-second HTTP timeout (`inference/models_client.py:15`, `:101-107`); it
  does not impose a GPU deadline. The stale thread keeps `voice_calls` until that client call returns or raises, plus the conditioning store when
  a reply exists; an HTTP failure releases the app lock without proving that the backend work stopped. The stale thread stores a `built`
  conditioning (D51) and never stores the test. Meanwhile a new test or a film reads "waiting for the voice model"; a waiting job checks
  `cancelled()` at each 1-second acquire timeout and after the acquire, so its own Cancel takes effect at the next check (no hard time promise).
  When the publication wins, the test stays stored and is shown; a Cancel after it only dismisses the job display.

**E. The job and the routes — `web/voices_api.py`, `web/app.py` (D51, D52, D54)**

- `web/app.py`: `app.state.voice_calls = threading.Lock()` beside the film state (`:221-243`), before `register_voices_api` (`:264`). `_start_film`
  passes `voice_lock=app.state.voice_calls` (`:1191-1195`). `_narrator_picker` (`:961-976`): a stored saved voice that is not castable reads
  `film_example(…).problem` first: `"deleted"` → "«<name>» — the <Language> example chosen for films was deleted"; `"deleting"` →
  "«<name>» — the <Language> example chosen for films is being deleted"; else today's labels. `_store_narrator` (`:1009-1012`): the same two cases
  → 409 with the §2.16 C texts. No other `web/app.py` line changes.
- `class VoicePreviews(context, voice_lock, drain)` in `web/voices_api.py`: the film's pattern (`web/app.py:1170-1280`) without a queue.
  - Fields: `lock` (its own `threading.Lock`, the job lock), `run: int`, `epoch: str`, `completed: int`, `state: dict`, `thread`.
    **Changed 2026-11-02 (t25, round 2):** `epoch = secrets.token_hex(16)` (32 hex characters), made once per `VoicePreviews` instance; it is an
    identity, never an order. `completed` = the run of the last test THIS instance stored (0 = none); `on_saved` sets it, under the job lock.
  - `start(job)`: under `lock`: `run += 1`; `state = {"state": "running", "run", "job", "stage": "starting", "fraction": 0.0}`; start one daemon
    thread for this run. A thread of an older run is now stale: it stops at its next check (the voice-lock wait, or the step-5 barrier), and its
    result is dropped. The voice lock already puts the model calls in a row, so no queue is needed (D51).
  - `cancel()`: under `lock`: `run += 1`; `state = {"state": "idle", "run"}`. It also clears a `done` or `failed` state.
  - The thread: `run_preview(ctx.conn, ctx.settings.build_models_client(), ctx.voices, owner, job, voice_lock=…, job_lock=self.lock,
    on_saved=…, progress=…, cancelled=lambda: self.run != run)`. `on_saved` sets `completed = run` and `state = {"state": "done", "run", "job"}`
    WITHOUT taking `lock`
    (run_preview already holds it; the lock is not re-entrant). `PreviewFailed` → `{"state": "failed", "run", "job", "error": str(e)}`; any other
    exception → logged, `"error": f"The test failed ({_reason(exc)})."`; `Cancelled` → no change. Progress and a failure are written under `lock`,
    and only while `self.run == run`. `finally: drain.wake()` (a replaced test WAV, a blob, or a write-ahead row). (**Changed 2026-11-02, t25:**
    the publication barrier of §2.16 D step 5.)
  - `snapshot() -> dict` for the templates: `state`, `run`, `epoch`, `completed`, `voice_id`, `example_id`, `language`, `text` (the first 80
    characters, + "…" when cut), `stage`, `percent` (`int(100 * fraction)`, clamped to 0-100), `error`. Every fragment carries `completed`.
  - `join()` is the test seam: `app.state.voice_previews`, `app.state.await_voice_preview`.
- Routes, registered BEFORE the `/organize/voices/{voice_id}/…` routes:

  | Method | Path | Does | Reply |
  |---|---|---|---|
  | GET | `/organize/voices/preview` | the test state (polled every 1 s while a test runs; also the reconciliation GET, §2.16 G) | `_voice_preview.html`, `Cache-Control: no-store` |
  | POST | `/organize/voices/{voice_id}/examples/{example_id}/preview` | form `text_kind`, `text_id`, `text` → `preview_text` → `start` (replaces a running test) | `_voice_preview.html` |
  | POST | `/organize/voices/preview/cancel` | `cancel`: ends a running test logically (§2.16 D, "Cancel is logical"; it may wait for a publication in progress, never for synthesis), or clears a finished one | `_voice_preview.html` (idle) |
  | GET | `/organize/voices/{voice_id}/examples/{example_id}/preview/audio` | the last test WAV | `audio/wav` |
  | POST | `/organize/voices/{voice_id}/film-example` | form `language`, `example_id` → `set_film_example` | `_voices_list.html` |

- Status codes. Test POST: 404 an unknown or foreign voice, or an example of another voice; 409 the voice or the example is pending, or no cloning
  entry speaks the example's language on this profile (`NOT_AVAILABLE`); 400 a `preview_text` refusal; 507 `ctx.voices.free_bytes() <
  FREE_SPACE_FLOOR`. Test audio: 404 when there is no test, the example is pending, or the file is missing. Film example: 404 an unknown voice or
  an example of another voice; 400 a `VoiceError`; 409 `VoiceDeleting`, or no cloning entry for `language` here (`NOT_AVAILABLE`). Errors are JSON
  `{"detail": …}` through `_refuse`, as today.
- `_list_context` adds `"film"`: `{voice_id: {language: FilmExample}}` for every non-pending voice and every `LANGUAGES` code where the voice has an
  example or a `voice_film_examples` row; `"previews"`: `previews(conn, owner)`; `"revisions"`: `{language: cloning_entry_for(profile,
  language).source.revision}` for `cloning_languages(profile)`. The page context adds `"preview"` (the snapshot), `"sentences"`
  (`PREVIEW_SENTENCES`) and `"custom_max"` (`CUSTOM_TEXT_MAX`).

**F. Page — `_voices_list.html`, `voices.html`, `_voice_preview.html` (new), `app.css` (D53)**

- Each example row that is not pending (`_voices_list.html:35-51`) gains, after the date and in this order:
  - `#<id>`;
  - the film state (**changed 2026-11-02, t25**): on the film example of its language `<span class="example-films">used for films (newest)</span>`,
    or `used for films (your choice)` when `chosen`; on every other example `<span class="example-films muted">not used for films</span>`;
  - the button `Use for films` (a `js-voices-form` POST to `/organize/voices/<v>/film-example` with hidden `language` and `example_id`) on every
    example whose language voice copy speaks here, EXCEPT the chosen one. So the default newest has it too, and the user can pin it;
  - `<span class="example-label">Original</span>` before the existing `<audio>`;
  - with a stored test: `<span class="example-label">Generated</span>`, `<audio controls preload="none" src="…/preview/audio?v=<first 12 of
    storage_key>">` and `<span class="example-test muted">“<text>” · <model> @ <first 7 of revision> · made in <elapsed, 1 decimal> s ·
    conditioning built|restored · <date></span>`; when its revision ≠ `revisions[language]`, add "· made with an older model revision — generate
    it again";
  - `<button type="button" class="example-preview" data-voice-id data-example-id data-language data-source data-reading-text>Test this
    example</button>` (`data-reading-text` only on a recorded example with a transcript), only when voice copy speaks the language here.
  - The Delete form of the CHOSEN example gets `data-confirm="This example is used for films in <Language>. Films narrated by «<name>» in
    <Language> will fail until you choose another example. Delete it?"`. Other example deletes keep no confirm.
- Per voice, under its examples, one `<p class="voice-film-missing">` per language whose `FilmExample.problem` is `"deleted"` or `"deleting"`:
  "The <Language> example chosen for films was deleted." (or "… is being deleted.") + " Films narrated by «<name>» in <Language> fail until you
  choose another with Use for films."
- The tip (`_voices_list.html:68`): "Record clean, normal speech in a quiet room. Films use one example per language: the newest, unless you choose
  one with Use for films."
- `voices.html`: after the recorder panel, OUTSIDE `#voices`: `<div id="voice-preview" class="voice-preview" hidden>` with `<p class="preview-target">`;
  `<select class="preview-text">` with `<option value="reading">The text you were asked to read (not checked against the recording)</option>`, one
  `<option value="sentence:<id>" data-lang="<code>"><id> — <sentence></option>` per `PREVIEW_SENTENCES` entry, and `<option value="custom">Your own
  text</option>`; `<textarea class="preview-custom" maxlength="300" hidden>`; `.preview-generate` "Generate"; `.preview-close` "Close"; then
  `{% include "_voice_preview.html" %}`.
- `_voice_preview.html`: `<div id="voice-preview-job" data-state="<state>" data-run="<run>" data-epoch="<epoch>" data-completed="<completed>"
  data-voice-id data-example-id data-language>`, by state:
  - `idle`: empty;
  - `running`: `<p class="preview-what">Testing example #<id> (<Language>): “<text>”</p>`, `<div class="preview-bar"><span style="width:
    <percent>%"></span></div>`, `<p class="preview-step"><stage> · <percent>%</p>`, `<button type="button" class="preview-cancel">Cancel</button>`;
  - `failed`: `<p class="preview-error"><error></p>`, `<button type="button" class="preview-dismiss">Close</button>`;
  - `done`: `<p class="preview-done">Done: ▶ Generated is under example #<id>.</p>`.
  Jinja autoescape covers every field (no hand-built HTML, unlike `web/app.py:1249-1266`).
- `web/static/app.css`: `.example-films`, `.example-label`, `.example-test`, `.voice-film-missing`, `.voice-preview`, `.preview-bar`,
  `.preview-step`, `.preview-error`.

**G. Script — `web/static/voices.js` (D53)**

- New state, outside `#voices` (**changed 2026-11-02, t25 round 2; D53**): `previewTarget` (`{voiceId, exampleId, language, source,
  readingText}` or `null`); `previewSeen = {epoch, run, state}`, seeded at load from the page's own `#voice-preview-job`; `retiredEpochs` (a
  `Set` of epochs this page has left); `generation` (one integer per page); `refreshTarget` (`{epoch, completed}` or `null`); `previewRefreshed`
  (the `{epoch, completed}` of the last refresh that replaced the list; it replaces `previewRun`); `previewTimer`. `previewBusy()` =
  `#voice-preview-job` has `data-state="running"`. `closePreview()` hides the panel and sets `previewTarget = null`.
- `openPreview(button)`: binds `previewTarget` from the button's data; shows only the options of its language (`data-lang`), plus `reading` when
  `readingText` is set (that option then also shows the text); selects the first shown option; sets `.preview-target` to "Test example #<id> ·
  <Language> · recorded|uploaded"; opens the panel. It works in every recorder state and never touches `prepared`, `live` or `take`.
- Pure functions, with no DOM access, so §5 runs them under node:
  - `previewVerdict(seen, retired, fragment) -> "apply" | "ignore" | "reconcile"`: the fragment's epoch is in `retired` → `ignore`; its epoch ≠
    `seen.epoch` → `reconcile`; same epoch and a lower run → `ignore`; same run and `running` after `done` or `failed` → `ignore`; else `apply`.
    Runs are never compared across epochs.
  - `completionKey(fragment) -> {epoch, completed} | null`: `null` when `completed` is 0. Keys are compared by equality of both fields, never by order.
  - `refreshLoop(key, deps)`, with injected `fetchList`, `replaceList`, `typing`, `schedule`, `isCurrent(key, generation)` and
    `markRefreshed(key)`: it captures `{key, generation}`. After EVERY await (the fetch, `response.text()`) and just before `replaceList`, it checks
    that `key` is still `refreshTarget` and the generation is current; else it stops with no replace and no marker. Typing, a network error or a
    non-ok reply → `schedule(1000)` and try again, only while current. After `replaceList`, `markRefreshed(key)` sets `previewRefreshed = key`.
- `showPreviewState(html, generation)`: a reply whose captured generation is no longer current is dropped. Else, by `previewVerdict`:
  - `ignore` → nothing changes;
  - `apply` → replace `#voice-preview-job` and set `previewSeen`; `running` → `pollPreview()`; then the completion rule;
  - `reconcile` (an unknown epoch: the app restarted) → never applied. Bump `generation`, stop the old timers, and GET `/organize/voices/preview`
    with `{cache: "no-store"}`. Only that reply, still current and with an epoch not in `retiredEpochs`, adopts its epoch: the old epoch goes into
    `retiredEpochs`, `previewSeen` = the new snapshot, the fragment is applied, and `refreshTarget = {newEpoch, 0}` with one guarded list refresh
    (a test published just before the restart lives in SQLite, while the new process reports `completed` 0). When the panel showed `running` and
    the adopted snapshot is `idle`, `#voice-preview-job` gets `<p class="preview-what">The app restarted, so this test stopped. Generate it
    again.</p>`. A failed reconciliation GET tries again in 1 s. The textarea and the recorder state stay; a job does not resume; stored Generated
    lines stay.
  - The completion rule: after any applied or adopted fragment with `completed > 0` and a key ≠ `previewRefreshed`, that key becomes
    `refreshTarget` and one `refreshLoop` runs (a loop for another key stops at its next check). So a Cancel or Generate reply re-arms a completion
    that a generation bump dropped.
- Generate (`generatePreview()`: `text_kind`, `text_id` from `sentence:<id>`, `text` from the textarea), Cancel, Dismiss and the clearing Close each
  bump `generation`, stop the old timers, POST, and pass a 2xx reply to `showPreviewState`. A 4xx/5xx or network failure of a current-generation
  mutation keeps its error shown (`showError(detail)`) and GETs `/organize/voices/preview` with `no-store`; that reply goes through the same rules.
  A mutation is never replayed.
- `pollPreview()` GETs `/organize/voices/preview` with `no-store` every 1 s while the state is `running`, the page is open and its generation is
  current.
- `.preview-cancel` and `.preview-dismiss` POST `/organize/voices/preview/cancel` → `showPreviewState`. `.preview-close` is hidden while `running`;
  it hides the panel, and with a `done` or `failed` state it also POSTs the cancel route to clear it. The textarea keeps its text for the page session.
- `decorate()` (it runs after every `replaceList`): when `previewTarget` names an example that is no longer in `#voices`, or that carries
  `data-deleting`, `.preview-target` reads "This example is being deleted." and Generate is disabled.
- `startRecording` pauses every `<audio>` of the page before `getUserMedia`, so a playing Original or Generated is not recorded.
- On load: a `running` or `failed` state from the server opens the panel bound to the job's example; `running` starts `pollPreview()`.
- Esc (`:370-381`), in this order: a live recording → an unsaved take → a prepared panel → a focused text field (the textarea too) → the test panel
  when no test runs (`previewTarget && !previewBusy()` → `closePreview()`) → close the page. A running test is not an Esc level: it runs on in `app`,
  and the page shows it again when it is opened (D53).
- **Changed 2026-11-02 (IMPL, pair t27, reviewer STOP):** every `#voices` body carries a completion certificate, and one gate decides every
  replacement (`send`, `pollWhileDeleting`, `refreshLoop`). `_list_context` takes `snapshot()` BEFORE its first DB read and stamps `#voices` with
  `data-epoch` / `data-completed` (the page reuses that snapshot as `preview`). `previewSync` seeds `previewSeen.completed` and `previewRefreshed`
  from the page's `#voices`; `sync.acceptList(cert) -> accept | reject | reconcile` accepts only the adopted epoch with the observed `completed`
  (keys equal, 0 included; epochs never ordered), rejects a missing, mismatched or retired certificate, and sends an unknown epoch to the
  reconcile. Only an accepted body swaps, decorates, polls, sets `previewRefreshed` and clears `listWanted`. A rejected body sets `listWanted`;
  one repair loop with an ownership ticket GETs status (no-store) then the list, re-checking ticket and generation after every await; typing or a
  failed fetch retries in 1 s; `bump` retires the owner and restarts one repair when `listWanted` is set. GETs only, no mutation replay; `send()`
  still returns true on every 2xx. Not covered: the order of two list bodies with no test involved (Phase 2 paths).

**H. Error paths (Phase 4)**

| Case | Where | Result |
|---|---|---|
| the example chosen for films was deleted | page / `/feed` / Prepare / build | the line "The <Language> example chosen for films was deleted…"; the option "«<name>» — the <Language> example chosen for films was deleted", `selected`; 409; `failed` with the reason, no speak call; never another example |
| the example chosen for films is being deleted | the same | "… is being deleted" in each place |
| Use for films: an unknown example, or one of another voice | API | 404 |
| Use for films: an example of another language | API | 400 |
| Use for films or a test: the voice or the example is being deleted | API | 409 |
| Use for films or a test: no voice copy for that language here | API | 409 "Voice copy is not available on this device." |
| test text: the reading text of an upload; an unknown or other-language sentence id; blank or over 300 characters; an unknown kind | API | 400 with the reason |
| a test on low disk | API | 507 |
| the example deleted while its test runs | job | `failed` "The example was deleted, so the test was dropped."; the drain removes a written file |
| the reply's revision ≠ the revision at the start | job | `failed` "The voice model changed while the test ran. Generate it again." |
| the models service refuses or fails | job | `failed` with the service's reason |
| the example WAV is missing on disk | job | `failed` "The example could not be read (…)." |
| Cancel, or a new Generate, while a test runs | job | changed 2026-11-02 (t25): Cancel is logical, not a GPU stop. The route may wait for a publication in progress, never for synthesis; a cancel before `client.speak` makes no model call; a call already sent runs to its reply and holds `voice_calls` until then; the old test is not stored; a built conditioning is kept. When the publication won, the test stays stored and shown |
| the app restarts (hot reload) while a test runs | page | the panel reads "The app restarted, so this test stopped. Generate it again."; the typed text and the stored Generated lines stay; the list refreshes once for the new process |
| a film narration and a test at the same time | `app` | one waits ("waiting for the voice model"); never two voice calls at once |
| a stored test of another revision | page | shown with "made with an older model revision — generate it again" |

**I. Binding rules**

- `ivms777/CLAUDE.md:120-141`: every test is `POST /speak` through `ModelsClient`. `feed/voice_preview.py` imports no model, and the one-model gate
  scans `feed` (`tests/test_one_model_process_gate.py:39-42`).
- `:264-274`: stage + percent + Cancel; a new Generate during a test replaces it, never a dead end.
- `:51-66`, `:68-118`: the panel lives outside `#voices`, so a list refresh never takes it; the job lives in `app`, so a return to the page shows
  it; no action pushes history; the Esc order of §2.16 G.
- `:159-161`: a test WAV leaves only through the outbox drain.
- `:3-36`: docs first (lane 18). `:163-204`, `:276-281`: tests in `tests/`, fakes only, no model load.

**J. Rejected (t24)**

Clip concatenation or averaging; fine-tuning or training; automatic likeness scores; a page button that runs `scripts/voice_check.py` (it unloads
the voice unit twice, `scripts/voice_check.py:266`, `:289`); ref_continuation now (the user: "Not now"); a Bake button, a second cache, and a
cache-only prepare with dummy text; any change to D16 or `CONTROLS`; any promise of likeness or speed. Also not done here: a cancel route or a voice
lock inside the models service (D52).
**Changed 2026-11-02 (Phase 5, t28):** "fine-tuning or training" and "a Bake button" are reversed for a per-voice LoRA adapter (§2.17), and
"any change to D16" is overtaken by the working tree (§1.8, D16).

### 2.17 Baked voice models and the Voices page (Phase 5; user 2026-11-02, pair t28 + t30)

Goal: a saved voice learns from ALL its takes. The user records a guided kit (slow, normal and fast reading, moods, questions, emphasis,
articulation, numbers and names), confirms the words of every take, and bakes a per-person LoRA adapter of VoxCPM2 from the full set. Films
and tests of a baked voice use that adapter. All voice work moves to a top-level **Voices** menu (t30). Nothing here promises a time, a memory
figure, a likeness or a mood before the user's Slice A run measures it (`ivms777/CLAUDE.md:223-232`).

**The answer the user asked for (t28).** "Used for films" is the ONE take that new films of «me» in Russian send to VoxCPM2: the user's pick, else
the newest (`feed/saved_voices.py:654-670`). The app trains no model from the collection today. A take with stored words already uses
`ref_continuation` (§1.8). The conditioning cache encodes one clip and changes no weights. From Phase 5 on, "bake" names the voice model only;
a test still stores the conditioning (§2.16), and the page no longer calls that a bake. **The prompt take's role after a bake:** a baked voice
still starts each passage from one real take (the film take, D47), because the adapter shapes the model and the take gives the passage its
starting sound. Slice A measures whether the take helps (variant A against C, §2.17 I); a result against it goes back to the planner.

**Changed 2026-11-02 (rework, t39; the user: "films should not use any specific recording. BUT ALL RECORDINGS!").** The prompt-take
paragraph above is superseded. Films and tests of a saved voice speak ONLY through its voice model, with `use_clip=False` (variant C of
§2.17 I): no take is sent, none is chosen, and nothing falls back to one take. A voice with no usable model cannot narrate and says
"bake required" (§2.17 K5). The model stays when takes are added, edited or deleted; Rebake trains a new version from all current takes.
One Test per person. Slices A and V are implemented as written below (their film-take parts in V are replaced by Slice B, K6). Slice B is
re-planned in §2.17 K; its rules carry the marker "Changed 2026-11-02 (rework, t39)".

**Changed 2026-11-02 (ASR, t42; the user: "i can say ANYTHING, OR UPLOAD ANYTHING! NOT RELATED TO THE TEXT AND YOU SHOULD USE IT", then
"Yes, add it (Recommended)").** The user no longer confirms or types words. A take is free speech; the kit text is a suggestion. Local
speech recognition in the models service hears the words of every take before a bake, and the bake trains on them (K2, D90-D95). A take
it cannot hear stays and shows why. Films and tests do not change (D76). These rules carry the marker "Changed 2026-11-02 (ASR, t42)".

**A. Slices (D66)**

| Slice | What | Files | Starts | Ends with |
|---|---|---|---|---|
| A | feasibility bake: trainer, bake job in the models service, adapters in `/speak`, `scripts/voice_bake_check.py` | `models/`, `modelsvc/`, `audio/`, `inference/`, `scripts/`, their tests, docs | the plan gate | §6.5, the user's run on this Mac; a FAIL stops B for the user's decision |
| V | Voices menu, grid and leaf with today's features (t30) | `web/`, three text lines of `feed/build.py` (D67), their tests, docs | the plan gate, beside A | §6.6 |
| B | kit, transcripts, the bake in `app`, baked-model state, films and tests on the adapter | `db/`, `feed/`, `web/`, their tests, docs | §6.5 PASS and §6.6 PASS, and the gate answer on deleted takes (`## Open items outside scope`) | §6.7 |

**Changed 2026-11-02 (rework, t39):** A and V are implemented (§4 steps 23-28b); §6.5 and §6.6 are with the user. B starts at the gate of this
rework and does not wait for §6.5 or §6.6. Its files: `db/`, `feed/`, `web/`, `scripts/voice_bake_check.py` (D82), and one small
models-service lane for the `/speak` contract of D76 (`modelsvc/app.py`, `inference/models_client.py`), which waits until no §6.5 bake runs
(D84). It ends with §6.7. **Changed 2026-11-02 (rework, t40):** that lane also owns `modelsvc/backends/fake.py`; the store lane only adds
names, and a last lane (32d) removes the one-take names once nothing imports them (D89).
**Changed 2026-11-02 (ASR, t42):** Slice B also adds the bake's speech recognition: in the models service (`models/catalog.py`,
`modelsvc/slots.py`, `audio/whisper_asr.py` new, `/voice/transcribe`, the bake's unload, the client, `scripts/fetch_models.py`; lane 31d,
which waits like 31b, D84) and in `app` (the take's words in the store, lane 31c; the transcription stage of the bake job, lane 32a; the
page, lane 32c). A docs lane (30b) goes first.

**B. Catalog — `models/catalog.py` (D62)**

```python
@dataclass(frozen=True)
class AdapterSpec:
    """A cloning entry that can learn ONE saved voice as a LoRA adapter (design §4.1, §8.1)."""
    lora: dict                      # LoRAConfig fields, the same for the trainer and the voice child
    train_cost_mb: int              # resident cost of the training child — an ESTIMATE until Slice A measures it
    train_cost_measured: bool
    train_profiles: tuple[str, ...] # where a bake is offered; adapters are USED wherever the entry runs
    guidance_minutes: tuple[float, float]  # the package authors' figure (METADATA:571): guidance, never a promise

# ModelEntry gains:  adapter: AdapterSpec | None = None
# voxcpm2 gets:
adapter=AdapterSpec(
    lora={"enable_lm": True, "enable_dit": True, "enable_proj": False,
          "r": 8, "alpha": 16, "dropout": 0.0,
          "target_modules_lm": ["q_proj", "v_proj", "k_proj", "o_proj"],
          "target_modules_dit": ["q_proj", "v_proj", "k_proj", "o_proj"]},
    train_cost_mb=14000, train_cost_measured=False, train_profiles=("mac",), guidance_minutes=(5.0, 10.0),
),

def adapter_entry_for(profile: str) -> ModelEntry | None   # the cloning entry with an AdapterSpec that trains on `profile`
```

- `lora` = the package defaults (`voxcpm2.py:139-145`) with LM and DiT on and projections off, the config the package builds for itself when
  it loads weights (`core.py:48-55`). The trainer and the child read it from here, so their shapes always agree.
- `train_cost_mb` 14000 = the 9095 MB float32 weights of D25 + an unmeasured allowance for activations and gradients. `IVMS777_MODEL_COST_MB`
  overrides it per unit (`voice_train`), as for every unit (`modelsvc/slots.py:124-127`). Slice A reports the measured drop.
- The voxcpm2 `note` adds (t31): "The voice child is built with LoRA layers for baked voices: about 18 MB in float32 (an estimate), inside
  cost_mb 10500. With no adapter loaded they add no adapter contribution; their extra matmuls are not measured." `cost_mb` stays 10500.
- **Changed 2026-11-02 (ASR, t42; D90) — lane 31d:** the bake's ASR model is declared beside the slots, in a tuple of its own. It is no
  slot and no user choice, so `SLOTS`, `DEFAULTS`, `CATALOG`, the settings popup and `tests/test_catalog.py:33-35` do not change (§1.10):

  ```python
  # --- asr (the words of a take for a bake, design §4.1, §8.1) -----------------------------------
  # No slot and no user choice: only a bake's transcription stage asks for it (§11.1).
  TRANSCRIBERS: tuple[ModelEntry, ...] = (
      ModelEntry(
          key="whisper-large-v3-turbo", slot="asr", display="Whisper large-v3 turbo · words of a take",
          source=HfSource("openai/whisper-large-v3-turbo", revision="41f01f3fe87f28c78e2fbf8b568835947dd65ed9"),
          size_mb=1547, cost_mb=4000, cost_measured=False,
          profiles=("mac",), languages=("en", "ru"),
          note="Hears the words of a saved voice's take before a bake (§4.1, §8.1, §11.1). 808,878,080 params, MIT weights, "
               "16 kHz input, a 30 s window (a take is 5-30 s). Loaded in float32 on MPS, so the cost is an ESTIMATE: "
               "808,878,080 x 4 B + runtime. Measure it and pin it with IVMS777_MODEL_COST_MB {\"asr\": ...}.",
      ),
  )

  def transcriber_for(profile: str) -> ModelEntry | None   # the TRANSCRIBERS entry offered on `profile`, else None
  ```

  `size_mb` 1547 = the 1,622,466,054 bytes of the 13 files (§1.10). `cost_mb` 4000 = 3086 MiB of float32 weights + an unmeasured
  allowance for the encoder activations of one 30 s window and the runtime. `profiles` follow `adapter.train_profiles` (mac, D62): a bake
  is the only user.

**C. Units, lease, one copy — `modelsvc/slots.py`, `modelsvc/governor.py` (D57, D61)**

- `TRAIN_UNIT = "voice_train"`. `SlotManager._register_voice_train()` runs in `apply()` and registers ONLY when `voice_train` is not registered
  yet (**changed 2026-11-02, t31**): `_register_torch_unit` first unloads its unit (`slots.py:129-132`), and the lease does not guard that path, so
  a later `apply()` (a music or embedder switch) never touches a bake. When `catalog.adapter_entry_for(profile)` exists, it registers a torch
  unit with target `"audio.voxcpm_trainer:VoxCPMTrainer"`, args `(repo, embed_device, revision, json.dumps(lora, sort_keys=True),
  str(data_dir / "voice-bakes"))`, cost `settings.model_cost_mb.get("voice_train", adapter.train_cost_mb)`, no `warm`.
  It is no slot and no catalog choice: nothing outside a bake ever asks for it.
- The `voice` child of an entry with an `AdapterSpec` gets one more arg: the same LoRA JSON (`_register_torch_unit`, `:134-138`). Kokoro and Silero
  keep their args.
- `MemoryGovernor.lease(name)` — a context manager: while it is held, `acquire` treats `name` as kept (`keep = needed ∪ pinned ∪ leased`,
  `governor.py:53`), and `unload(name)` raises `UnitLeased`. `leased() -> frozenset[str]`. `CompositeBackend.model_unload` goes through the governor
  when one exists, so `POST /models/voice_train/unload` during a bake answers **409**.
- **One admission barrier for the voice path** (**changed 2026-11-02, t31**; D57). `VoiceGate` (in `modelsvc/voice_bakes.py`): `enter()` raises
  `BakeRunning` when the gate is closed, else counts one call in; `leave()` counts it out; `close()`, `drain()` (waits until the count is 0),
  `open()`, `closed()`. `CompositeBackend.speak` (preset and saved) calls `enter()` BEFORE `ensure_voice_for` and `leave()` in `finally`, so a speak
  that waits in `_admit` is still counted (§1.8). A bake closes the gate, drains it, and only then unloads `voice`; it opens the gate again only
  after the trainer is torn down. So no accepted speak can load `voice` beside the training child, whatever the budget sums are.
  `CompositeBackend.model_ensure("voice")` raises `BakeRunning` while the gate is closed, and `model_ensure("voice_train")` raises
  `BakeOnly("voice_train is loaded only by a bake")` outside a bake; the route answers **409** for both.
- **Changed 2026-11-02 (pair t32, in the code):** `model_ensure("voice")` takes the counted gate entry like a speak
  (`with self._gate.admitted()`, `modelsvc/backends/composite.py:294`): `enter()` BEFORE `scheduler.run` or `registry.ensure`, in both
  branches, and `leave()` in `finally` only after `enter()` succeeded; there is no separate `closed()` check. So a bake's drain also waits
  for a manual voice ensure. `VoiceGate` does the closed check and the count-in under one condition and never holds it across a scheduler
  wait or a load (`modelsvc/voice_bakes.py:77-126`).
- **Changed 2026-11-02 (pair t35, in the code):** `model_ensure("voice_train")` ALWAYS raises `BakeOnly` — before the gate, the scheduler
  and the registry, also during a bake (`modelsvc/backends/composite.py:291-292`). Only a bake loads `voice_train`, through its own lease and
  reserve. `POST /models/voice_train/ensure` always answers **409**.
- **Changed 2026-11-02 (ASR, t42; D93) — lane 31d:** `ASR_UNIT = "asr"`. `SlotManager._register_asr()` runs in `apply()` beside
  `_register_voice_train()` and registers ONLY when `asr` is not registered yet (the reason of t31: a registration first unloads its
  unit). When `catalog.transcriber_for(profile)` exists, it registers a torch unit with target `"audio.whisper_asr:WhisperASR"`, args
  `self._child_args(entry)` = `(repo, embed_device, revision)`, cost `settings.model_cost_mb.get("asr", entry.cost_mb)`, no `warm`.
  `CompositeBackend.model_ensure` counts `asr` in the voice gate like `voice` (`name in ("voice", ASR_UNIT)`), so `POST /models/asr/ensure`
  answers **409** during a bake and a bake's drain waits for one in flight.

**D. The training child — `audio/voxcpm_trainer.py` (new; models-service-only; D57, D58)**

Module scope imports no torch, no voxcpm, no datasets (pattern `audio/voxcpm_voice.py:8-11`), so the pure parts are unit-tested.

```python
@dataclass(frozen=True)
class TrainConfig:
    steps: int
    learning_rate: float = 1e-4
    batch_size: int = 1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    seed: int = 42                         # the package Accelerator's own default (accelerator.py:23)

def default_steps(clips: int) -> int                       # min(1000, max(200, 20 * clips))
def train_config(request: dict, clips: int) -> TrainConfig # validates: steps 1..5000 (None → default_steps), 0 < lr <= 1e-2, batch 1..8,
                                                           # 0 <= wd <= 1, 0 < grad norm <= 100; else ValueError with the field
def check_trainable(names: list[str]) -> None              # ValueError unless every trainable name holds "lora_" and there is at least one
def adapter_metadata(*, revision, version, lora: str, dataset_sha256, config: TrainConfig, steps_done) -> dict[str, str]
def run_training(config, *, batches, train_step, cancelled, report, memory, clock) -> dict
    # batches: an endless iterator; train_step(batch, progress) -> float (forward, backward, clip, optimizer step)
    # returns {"state": "done" | "cancelled", "steps_done", "loss_first", "loss_last", "seconds_per_step", "peak_device_bytes"}

class VoxCPMTrainer:
    def __init__(self, repo: str, device: str, revision: str, lora: str, bake_root: str) -> None
    @classmethod
    def loaded(cls, model, revision: str, *, lora: str, data, optimizer, clip, save, memory, version="test") -> "VoxCPMTrainer"  # test seam
    def start(self, job: dict) -> None    # job = {"manifest", "work_dir", "clips", "config", "dataset_sha256"}; a second start → RuntimeError
    def status(self) -> dict              # never blocks on training
    def cancel(self) -> None              # sets an Event; the loop stops at the next step boundary
```

- `__init__`, in this order: refuse a device that is not `mps` and does not start with `cuda` (`audio/voxcpm_voice.py:147-148`); set
  `HF_DATASETS_CACHE=<bake_root>/hf-cache` and `HF_DATASETS_OFFLINE=1` before any `datasets` import; resolve the snapshot with
  `snapshot_download(repo, revision=revision, local_files_only=True)` and check the folder name (as `audio/voxcpm_voice.py:155-166`); build
  `Accelerator(amp=False, seed=42)` FIRST, so the LoRA A draw at model build is seeded; refuse when `accelerator.device.type` is not the device's
  type (the package falls back to the CPU in silence, `accelerator.py:106-112`); `model = VoxCPM2Model.from_local(path, optimize=False,
  training=True, device=device, lora_config=LoRAConfig(**json.loads(lora)))`; `model = accelerator.prepare_model(model)`; `model.train()`;
  `check_trainable([n for n, p in model.named_parameters() if p.requires_grad])`.
- `start(job)` starts ONE daemon thread inside the child and returns. The child keeps answering `status` and `cancel` while it trains, because
  `TorchWorker.call` is held only for each short call (§1.8). The thread, with `stage` set at each step:
  1. "reading the takes": `ds, _ = load_audio_text_datasets(train_manifest=job["manifest"], sample_rate=model.audio_vae.sample_rate)`
     (16000 for this checkpoint, §1.8); `tok = model.text_tokenizer`; `ds = ds.add_column("text_ids", [list(tok(t)) for t in ds["text"]])`
     (**changed 2026-11-02, t31**: no `ds.map` closure over the model, because the `datasets` fingerprint pickler would copy every tensor, §1.8);
     `loader = build_dataloader(ds, accelerator=accelerator, batch_size=config.batch_size, num_workers=0)`; `batches` = a generator
     `while True: yield from loader`, so each pass shuffles again (not `itertools.cycle`, which keeps every batch);
     `processor = BatchProcessor(config=model.config, audio_vae=model.audio_vae, dataset_cnt=1, device=accelerator.device)`.
  2. "training": `optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=…, weight_decay=…)`;
     `run_training(...)` with `train_step` = `packed = processor(batch)` → `out = model(**{k: packed[k] for k in FORWARD_KEYS}, progress=progress)`
     → `loss = out["loss/diff"] + out["loss/stop"]` → `loss.backward()` → `clip_grad_norm_(trainable, max_grad_norm)` → `optimizer.step()` →
     `optimizer.zero_grad(set_to_none=True)` → `float(loss)`. `FORWARD_KEYS` = the seven `forward` arguments (`voxcpm2.py:297-306`). A loss that
     is not finite → `RuntimeError(f"the training loss is not a number at step {n}")`. `memory()` = `torch.mps.driver_allocated_memory()` on MPS,
     `torch.cuda.max_memory_allocated()` on CUDA; `peak_device_bytes` is the largest value seen, as torch reports it.
  3. "saving the voice model": `safetensors.torch.save_file({k: v.detach().cpu().contiguous() for k, v in model.get_lora_state_dict().items()},
     <work_dir>/adapter.safetensors, metadata=adapter_metadata(...))`; then its sha256, size and key count go into the status.
  A cancel before step 3 saves nothing. Any exception → `state = "failed"` with `f"{type(exc).__name__}: {exc}"`.
  **Changed 2026-11-02 (pair t33, in the code):** a cancel during the LAST training step saves nothing either. `run_training` turns a
  `done` into `cancelled` when `cancelled()` is true after the loop (`audio/voxcpm_trainer.py:139-140`), and the thread returns `cancelled`
  before "saving the voice model" when the result is `cancelled` or the cancel event is set (`:319-320`).
- `status()` = `{"state": "idle" | "running" | "done" | "failed" | "cancelled", "stage", "step", "steps", "loss_first", "loss_last",
  "seconds_per_step", "peak_device_bytes", "device", "error", "adapter": {"path", "sha256", "bytes", "keys"} | None}`, read under a small lock.
- Adapter metadata (every value a string): `format` = `"ivms777-voxcpm2-lora-1"`, `model` = `"voxcpm2"`, `revision`, `voxcpm` (installed version),
  `lora` (the sorted JSON), `dataset_sha256`, `config` (JSON of `TrainConfig`), `steps_done`.

**E. The bake job in the service — `modelsvc/voice_bakes.py` (new; parent side, no torch; D59, D63)**

```python
class BakeRunning(RuntimeError): ...     # a second bake, or a voice op during a bake → 409
class BakeRefused(ValueError): ...       # a malformed request → 422

MAX_BAKE_CLIPS = 300
MAX_CLIP_BYTES = 4 * 1024 * 1024         # the add_example limit (§2.7)
MAX_BAKE_AUDIO_BYTES = 192 * 1024 * 1024 # ≈ 33 minutes of 48 kHz mono PCM-16
TEXT_MAX = 1000
TAG_MAX_BYTES = 64 * 1024
POLL_SECONDS = 1.0
LOAD_ESTIMATE_S = 180.0                  # display only: moves the bar while the child loads

class VoiceBakes:
    def __init__(self, *, slots, registry, governor, scheduler, activity, bake_root: Path, clock=time.monotonic, sleep=time.sleep) -> None
    def start(self, request: dict) -> dict        # BakeRunning, BakeRefused, BakeNotOffered (no AdapterSpec trains on this profile)
    def status(self, bake_id: str) -> dict | None
    def current(self) -> dict | None              # the running bake, else the last finished one not yet deleted
    def cancel(self, bake_id: str) -> dict | None
    def adapter(self, bake_id: str) -> bytes      # only in state "done"; else BakeRunning("not done")
    def delete(self, bake_id: str) -> bool        # removes the work dir; a running bake → BakeRunning
    def active(self) -> bool
    def sweep(self) -> None                       # at service start: remove everything under bake_root
```

- The request: `{"model": "voxcpm2", "clips": [{"audio": <base64 WAV>, "sha256", "text", "language"}], "config": {…}, "tag": {…}}`.
  Checks, each a `BakeRefused` (→ 422) with the reason: 1..`MAX_BAKE_CLIPS` clips; each decoded clip ≤ `MAX_CLIP_BYTES`, its sha256 equal, stdlib
  `wave` reads it as mono PCM-16 at 48000 Hz (the §2.7 contract, `storage.wav` is in the models image); the total ≤ `MAX_BAKE_AUDIO_BYTES`; `text`
  1..`TEXT_MAX` characters after strip; `language` in the entry's `languages`; `train_config` accepts `config`; `tag` ≤ `TAG_MAX_BYTES` as JSON.
  `model` must be `adapter_entry_for(profile).key`.
  **Changed 2026-11-02 (pair t37, in the code):** a take must also hold audio and every frame of it: `_wav_seconds` reads the payload inside
  the `wave` context; `frames <= 0` → "take N: holds no audio"; a payload shorter than `frames * 2` bytes → "take N: is cut short: part of its
  audio is missing" (`modelsvc/voice_bakes.py:493-509`, the mirror of `feed/saved_voices.py:_check_wav`, with no import of `feed/`). The
  check runs before the work folder and the thread, so a refused take closes no gate and unloads nothing.
- `start`: `bake_id = secrets.token_hex(8)`. It removes the previous finished bake's folder, writes each clip to `<bake_root>/<bake_id>/clips/
  <n>.wav` and one JSON line per clip `{"audio": <absolute path>, "text": <text>}` to `train.jsonl`, computes `dataset_sha256` = sha256 of the sorted
  `"<sha256>\t<language>\t<text>"` lines, sets `active`, and starts one daemon thread. It returns the status at once (202).
- The thread, inside `activity.track("baking a voice")`; the status carries `stage` and `fraction`:

  | Stage | Fraction | What |
  |---|---|---|
  | preparing the takes | 0.00-0.02 | the files above (done in `start`) |
  | waiting for the voice calls to end | 0.02 | `gate.close()`, then `gate.drain()`: every speak the gate let in ends first, also one that waits for a scheduler slot (t31), and every manual `voice` ensure (changed 2026-11-02, t32) |
  | waiting for the GPU | 0.02 | `governor.unload("voice")` (one VoxCPM2 copy), `governor.lease("voice_train")`, `scheduler.reserve(["voice_train"], Priority.BATCH)` |
  | loading the voice model for training | 0.02-0.10 | inside the reserve: the governor loads the child; the fraction moves on `elapsed / LOAD_ESTIMATE_S`, capped at 0.099 |
  | reading the takes | 0.10-0.12 | `worker.call("start", job)`, then `status` every `POLL_SECONDS` |
  | training · step N of M | 0.12-0.95 | `0.12 + 0.83 * step / steps` |
  | saving the voice model | 0.95-0.99 | |
  | done | 1.00 | |

  Each poll also reads `governor.state().free_mb` and keeps the smallest value (`min_free_mb`), and keeps `peak_device_bytes`. `finally`: when
  the child is alive, `call("status")` once more; then release the lease, `governor.unload("voice_train")` (the child ends, its memory returns,
  §8.1), leave the reserve, remove `<bake_root>/hf-cache`, clear `active`, and only then `gate.open()` (t31). The next `/speak` loads the voice
  child again.
  **Changed 2026-11-02 (pair t34, in the code):** after the teardown and the cache removal, `_finish` sets the end state, the adapter and the
  error, calls `gate.open()` and clears `active` inside ONE `with self._lock:` block (`modelsvc/voice_bakes.py:429-447`). So a next `start`
  sees all three or none: it cannot remove this bake's folder before its end is published, and this bake cannot open the gate after the next
  bake closed it. Nesting `_lock` → the gate's condition is safe: `VoiceGate.enter` reads the percent after it leaves its condition.
  **Changed 2026-11-02 (ASR, t42; D93) — lane 31d:** in "waiting for the GPU", right after `governor.unload("voice")`, the bake calls
  `governor.unload(ASR_UNIT)` (a no-op when it is not loaded or not registered, `modelsvc/registry.py:54-59`). So the Whisper child never
  shares memory with the training child; the next transcribe loads it again. The stage row reads "`voice` and `asr` are unloaded".
- Status: `{"id", "state": "running" | "done" | "failed" | "cancelled", "stage", "fraction", "step", "steps", "loss_first", "loss_last",
  "seconds_per_step", "stage_seconds": {stage: seconds}, "elapsed_s", "free_mb_before", "min_free_mb", "peak_device_bytes", "device", "error",
  "adapter": {"sha256", "bytes", "keys"} | None, "model", "revision", "lora", "config", "dataset_sha256", "clips", "seconds", "tag"}`.
- **What Cancel stops** (D63). Before "reading the takes": the thread checks the flag after the drain, before the reserve and after the load, and
  ends without a training call. While the child loads, the load runs to its end first (`TorchWorker.start` holds its lock, `torch_process.py:124-167`). In "reading
  the takes" and "training": one `call("cancel")`; the loop stops at the next step boundary and saves nothing. The status says
  `cancelled` only after the child reports it. Then the `finally` above ends the child. No "within N seconds" promise.
- While the gate is closed, `VoiceGate.enter()` raises `BakeRunning(f"a voice is being baked ({percent}%); try again when it ends, or cancel it")`
  → **409** for every voice request, preset or saved, and for `POST /models/voice/ensure` (t31). Music, embeddings and chat run as before; the lease keeps the training child resident, and an op
  that cannot fit next to it fails with the governor's own `InsufficientMemory` reason, as today.
- A models-service restart (a hot reload of `modelsvc` or `audio`, §1.8) ends the bake with its child. The next process knows no bake: every id is
  404. `sweep()` at start removes the old work folders, which hold copies of the user's takes.
- Routes (`modelsvc/app.py`), the literal paths before `{bake_id}`:

  | Method | Path | Reply |
  |---|---|---|
  | POST | `/voice/bakes` | 202 status; 409 `BakeRunning`; 422 `BakeRefused`; 503 `BakeNotOffered` "baking a voice is not offered on this device yet" (t31; apart from `voice copy is not available`) |
  | GET | `/voice/bakes/current` | status, or 404 |
  | GET | `/voice/bakes/{bake_id}` | status, or 404 |
  | POST | `/voice/bakes/{bake_id}/cancel` | status, or 404 |
  | GET | `/voice/bakes/{bake_id}/adapter` | `application/octet-stream` (safetensors); 409 when not done; 404 |
  | DELETE | `/voice/bakes/{bake_id}` | 204; 409 while it runs; 404 |

- Callers end every bake with `DELETE` (t31): on `done`, `failed` and `cancelled`, because the folder holds copies of the takes (D59). A caller that
  cancelled waits for ANY end state (`cancelled`, `done` or `failed`), and keeps no adapter.
- `modelsvc/backends/base.py` declares the six bake methods; `composite.py` delegates them to `VoiceBakes`; `backends/__init__.py` builds it
  beside the scheduler and calls `sweep()`; `fake.py` gets a scripted bake (a fixed status sequence, a few bytes as the adapter) for Slice B's tests.

**F. Adapters in `/speak` and in the voice child — `modelsvc/app.py`, `audio/voxcpm_voice.py` (D60, D61)**

```python
MAX_ADAPTERS = 4                          # references with an adapter, per request
MAX_ADAPTER_BYTES = 64 * 1024 * 1024      # decoded; the estimate is ≈ 18 MB (§1.8)

class VoiceAdapter(BaseModel):
    blob: str        # base64 of the safetensors file a bake wrote
    sha256: str      # of the decoded bytes

class VoiceReference(BaseModel):           # existing fields unchanged
    ...
    adapter: VoiceAdapter | None = None   # the voice's baked model
    use_clip: bool = True                 # False = speak from the adapter alone (no prompt take); only with an adapter

class AudioResponse(BaseModel):           # existing fields unchanged
    ...
    adapters: dict[str, dict] = {}        # token -> {"state": "loaded" | "kept" | "none", "sha256": str | None}
```

- `_decode_references` adds: more than `MAX_ADAPTERS` adapters, a decoded adapter over `MAX_ADAPTER_BYTES`, an sha256 mismatch, bad base64, or
  `use_clip=False` without an adapter → **422** with the reason.
- **Changed 2026-11-02 (rework, t39; D76) — Slice B, lane 31b.** A reference that speaks from the adapter alone carries no take:
  `VoiceReference.audio: str | None = None` and `sha256: str | None = None` (today required, `modelsvc/app.py:80-92`). `_decode_references`
  (`:289-330`): with `use_clip=True` (the default) a missing `audio` or `sha256` → **422** "the reference of '<token>' needs its audio"; with
  `use_clip=False` and an adapter, `audio` may be absent — then the decoded reference has no `audio` key and no `conditioning`; audio that is
  sent with `use_clip=False` is still decoded and checked as today and is not used (the check script's variant C sends it,
  `scripts/voice_bake_check.py:257-259`). `language` stays required. The child does not change: it never reads the audio of a `use_clip=False`
  token (`audio/voxcpm_voice.py:283-285`). `MAX_REFERENCES` (8) and `MAX_ADAPTERS` (4) stay, so one film can speak at most 4 saved voices.
- `VoxCPMVoice(repo, device, revision, lora: str | None = None)`: with `lora`, `VoxCPM.from_pretrained(path, load_denoiser=False, optimize=False,
  device=device, lora_config=LoRAConfig(**json.loads(lora)))`, and it keeps `self._lora_shapes = {k: tuple(v.shape) for k, v in
  tts.get_lora_state_dict().items()}`. With no adapter loaded every B is zero, so the LoRA layers add no adapter contribution (§1.8; their extra
  matmuls run, and §6.5 reports B's RTF). `None` keeps today's build (the test seam and old args).
- Pure helpers: `adapter_mismatch(metadata: dict, revision: str, lora: str) -> str | None` ("not a voice model of this app", "made for another
  model revision", "made with another LoRA shape"); `tensor_mismatch(shapes: dict[str, tuple], expected: dict[str, tuple]) -> str | None`
  (**changed 2026-11-02, t31**): the names must equal the expected LoRA names EXACTLY and every shape must be equal (`"the voice model holds N
  keys that are not LoRA weights of this model"`, `"the voice model misses N keys"`, `"N tensors of the voice model have another shape"`);
  `load_mismatch(loaded: list, skipped: list, expected: set) -> str | None` stays as a second check after the load.
- `_use_adapter(ref) -> state`, called by `speak_one` before each run of a token (`audio/segments.py` speaks same-voice runs in order):
  - no adapter: when one is loaded, `reset_lora_weights()` → `"none"`;
  - the loaded sha256 → `"kept"` (no load);
  - another one: write the blob to a temporary `.safetensors` file (`load_lora_weights` picks the reader by suffix, `voxcpm2.py:1216-1218`);
    with `safetensors.safe_open` read its metadata and each tensor's name and shape (`get_slice(name).get_shape()`), BEFORE any load (t31):
    `adapter_mismatch` or `tensor_mismatch` → `ValueError`, and nothing is copied. Then `reset_lora_weights()`, `load_lora_weights(tmp)`,
    `load_mismatch`; delete the file; → `"loaded"`. The pre-check keeps a base weight from ever being overwritten (`load_lora_weights` copies into
    any parameter of that name, §1.8).
  - on ANY failure in this step (a check, an I/O error, an exception inside the load): `reset_lora_weights()` and `self._adapter_sha = None`, then
    `ValueError`. So no later request reads `"kept"` for a half-loaded model, and the next request with an adapter loads it from zero.
  A request with an adapter to a child built without `lora` → `ValueError("this voice child was built without LoRA")`. Every `ValueError` is a
  500 with the model's message (§2.3), and the film or test fails with it: a voice model that does not fit is never spoken in part.
- `use_clip=False`: the token's runs use `prompt_cache=None`; no conditioning is built or restored for it, and the reply's `conditioning` has
  no entry for it.
- One child, one generation at a time (the `TorchWorker` lock), so a switch never races a generation. The reply's `adapters` holds the state of
  the token's LAST run.

**G. Client — `inference/models_client.py`**

- `speak(…, references=…)`: a reference may carry `"adapter": {"blob": bytes, "sha256": str}` and `"use_clip": bool`; the client base64-encodes the
  blob. The reply's `adapters` passes through.
- **Changed 2026-11-02 (rework, t39; D76) — lane 31b:** `_reference_payload` (`inference/models_client.py:41-57`) sends `audio` and `sha256`
  only when the reference has `audio`; the other fields are sent as today. A model-only reference from `app` is
  `{"language", "adapter": {"blob", "sha256"}, "use_clip": False}`.
- **Changed 2026-11-02 (rework, t40) — lane 31b:** `FakeBackend.speak` (`modelsvc/backends/fake.py:86-116`) builds today a `built`
  conditioning from `ref["sha256"]` for every token (`:110-115`), so a reference with no audio would fail there. For a token with
  `use_clip=False` it builds no conditioning entry and returns `adapters[token] = {"state": "loaded", "sha256": <the sent adapter sha256>}`;
  other tokens keep today's reply. One HTTP test sends a model-only reference with no audio through `ModelsClient` against
  `create_models_app(FakeBackend())` (the pattern of `tests/test_models_client.py:18`) and reads that reply.
- `bake_start(clips: list[dict], *, tag: dict, config: dict | None = None, timeout: float = 600.0) -> dict` (clip audio as bytes),
  `bake_status(bake_id, *, timeout=10.0) -> dict`, `bake_current(*, timeout=10.0) -> dict | None` (404 → None), `bake_cancel(bake_id, *,
  timeout=30.0) -> dict`, `bake_adapter(bake_id, *, timeout=120.0) -> bytes`, `bake_delete(bake_id, *, timeout=30.0) -> None`.
- **Changed 2026-11-02 (ASR, t42; D92) — lane 31d:** `transcribe(audio: bytes, language: str, *, timeout: float = 600.0) -> dict` posts
  `{"audio": <base64>, "sha256", "language"}` to `/voice/transcribe` and returns `{"text", "model", "revision", "seconds", "elapsed_s"}`;
  a non-2xx raises `httpx.HTTPStatusError`, as `bake_start` does (§1.10). 600 s covers the first call's Whisper load and a wait for a
  scheduler slot.

**H. Who waits, in Slice A.** The user runs the check script; `app` holds no bake. A film or a test that reaches `/speak` during that bake fails
with the 409 text. §6.5 tells the user not to start one. Slice B's `app` job holds the voice lock for its whole bake, so `app`'s own films and tests
wait instead (D74).
**Changed 2026-11-02 (rework, t39):** Slice B now runs while the user may run §6.5. A bake that `app` did not start (the check script's,
tag `{"check": NAME}`) is never adopted, cancelled or deleted by `app` on its own; a Bake request during it asks first (D83, K4). A film or a
test that reaches `/speak` during that foreign bake still fails with the 409 text, as above.

**I. The Slice A check — `scripts/voice_bake_check.py` (new; the user runs it; D64, D65)**

A thin client (no torch), D30 style: `uv run --no-sync python -m scripts.voice_bake_check --voice NAME (--list | --confirm-texts) [--text ID=WORDS
…] [--steps N] [--learning-rate X] [--out DIR]`.

1. It opens the user's database READ-ONLY (`sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)`) and reads the voice by name with
   `feed.saved_voices.list_voices`, and each take with `read_example_audio` from `data_dir/voices`. It writes no row and no voice file.
2. `--list` prints every take that is not being deleted: id, language, seconds, recorded/uploaded, and its words, labelled "reading text, not
   checked" or "NO WORDS". It ends there.
3. `--text ID=WORDS` gives or corrects the words of one take for this run only. A run needs `--confirm-texts` (the user read the list) and words
   for every take. A take with no words stops the run and is named; no take is left out in silence.
4. `models_state()` → `free_mb` before. `bake_start(clips, tag={"check": NAME}, config={"steps": …, "learning_rate": …})`. Every 2 s
   `bake_status` prints "stage · N% · step n of m · loss x" on one line, and `models_state()["free_mb"]` adds to the minimum. Ctrl-C →
   `bake_cancel`; the script then waits for ANY end state (`cancelled`, `done` or `failed`), keeps no adapter, writes the report and exits 130
   (**changed 2026-11-02, t31**).
5. `done` with no Ctrl-C → `bake_adapter` → `<out>/adapter.safetensors` (it stays only there); `failed` → the report with the error, exit 1.
   `bake_delete` runs in a `finally` on every end (t31).
6. Per language of the takes, the prompt take = `film_example(...)` (read-only), and the sentences = `scripts.voice_check.SENTENCES[language]`.
   Per sentence three calls through `POST /speak` with the token `saved:check`:
   - **A** — the adapter + the prompt take;
   - **B** — the prompt take alone (today's films);
   - **C** — the adapter alone (`use_clip=False`).
   The files are `<id>-X.wav`, `<id>-Y.wav`, `<id>-Z.wav` in a random order per sentence; `key.json` holds the order.
7. `report.json`: the bake status (stage seconds, steps, seconds per step, first and last loss, `peak_device_bytes`, `free_mb_before`,
   `min_free_mb`, the adapter's sha256, bytes and keys, the device); the takes (count, seconds per language, total); per call the model, revision,
   `adapters` state, conditioning state, wall seconds, RTF, duration, RMS, NaN; the mean RTF of each variant, B's labelled "today's films on the
   LoRA-built child" (t31); and the automatic checks of D65.
8. It prints a summary, the total recorded minutes beside the package authors' 5-10 minutes ("their guidance, not a promise"), and the §6.5
   thresholds. Default `--out` = `<data_dir>/voice-bake-check/<UTC stamp>/`.

**Changed 2026-11-02 (pair t38, in the code):** the console is blind. `format_summary` prints one row per file: the sentence, the file
(X/Y/Z), the duration, the dBFS and the identity (ok/VOID); no variant, adapter state, conditioning, wall seconds or RTF per row. An error row
prints the sentence, the file and "ERROR" with no reason; `speak_variants` sets each row's file name when it creates the row. The summary keeps
the bake, the stages, `seconds_per_step`, the memory figures, the adapter size, the mean RTF of A, B and C (B labelled), the checks and the
thresholds. `report.json` and `key.json` keep their content; both reveal the mapping. `THRESHOLDS` says: "Listen to and score every X/Y/Z
file of a sentence BEFORE you open key.json or report.json." (`scripts/voice_bake_check.py:59-65`.)
**Changed 2026-11-02 (rework, t39):** C (the adapter alone, `use_clip=False`) is the film path and the pass rule (§6.5 step 8, D65); A and B
only help to diagnose (`THRESHOLDS` says so already). Step 6's prompt take of A and B becomes the NEWEST take of the language (`example_for`),
because `film_example` leaves the code (D82; lane 31). `report.json`'s `prompt_takes` then records `{"example_id", "rule": "newest"}`.
**Changed 2026-11-02 (ASR, t42; D104) — lane 31c:** `--list` names where each take's words come from: "heard by <model>@<first 7 of
revision>" (`transcript_model` set, words stored), "shown text, not heard" (a take stored before t42: its `transcript` is the text it was
shown), "no usable words: <transcript_error>", or "NO WORDS". The run rules do not change: a run needs `--confirm-texts` and words for
every take, and `--text ID=WORDS` gives them for this run. The script never calls `/voice/transcribe` and writes no row.

**J. Error paths (Slice A)**

| Case | Where | Result |
|---|---|---|
| a bake already runs | `POST /voice/bakes` | 409 |
| no adapter entry trains on this profile (jetson, cloud for now) | `POST /voice/bakes` | 503 "baking a voice is not offered on this device yet" — not the `voice copy is not available` text (t31) |
| too many clips, a clip too large, sha mismatch, not 48 kHz mono PCM-16, blank or long text, unknown language, bad config, big tag | `POST /voice/bakes` | 422 with the reason |
| a voice request during a bake | `/speak` | 409 "a voice is being baked (N%)…" |
| a speak accepted just before the bake (also one waiting for a slot) | gate | the bake waits for it ("waiting for the voice calls to end"), then unloads `voice` |
| `POST /models/voice/ensure` during a bake; `POST /models/voice_train/ensure` outside one | models API | 409 |
| `POST /models/voice_train/unload` during a bake | models API | 409 |
| CPU device, or the package would train on another device | training child load | bake `failed` with the message |
| weights not on disk, revision mismatch | training child load | bake `failed` with the load message (`audio/voxcpm_voice.py:155-166` texts) |
| the training child does not fit the budget | reserve | bake `failed` with the `InsufficientMemory` reason |
| a non-LoRA weight is trainable | training child load | bake `failed` "…" (`check_trainable`) |
| the loss is not a number | training | bake `failed` "the training loss is not a number at step N" |
| the child dies (an out-of-memory kill) | status poll | bake `failed` "audio.voxcpm_trainer:VoxCPMTrainer worker died during status" (`torch_process.py:176-179`) |
| Cancel | any stage | `cancelled` at the next check or step; no adapter |
| the models service restarts | — | the bake is gone (404); the work folder is swept at start |
| an adapter of another revision or LoRA shape, missing or extra keys, a tensor of another shape | `/speak` | 500 with the reason before any copy; LoRA reset, the kept sha cleared; nothing is spoken with a part of a model |
| more than 4 adapters, one over 64 MiB, sha mismatch, `use_clip=False` without an adapter | `/speak` | 422 |
| an adapter to a child built without LoRA | `/speak` | 500 "this voice child was built without LoRA" |
| the script: an unknown voice, a take with no words, no `--confirm-texts` | script | exit 2 with the reason; no bake starts |
| a take with no frames, or with missing frames (t37) | `POST /voice/bakes` | 422 "take N: holds no audio" / "take N: is cut short: part of its audio is missing"; no folder, the gate stays open |
| `POST /models/voice_train/ensure` at any time (t35) | models API | 409 |
| changed 2026-11-02 (rework, t39; lane 31b): a reference with `use_clip=true` and no `audio` | `/speak` | 422 "the reference of '<token>' needs its audio" |
| changed 2026-11-02 (rework, t39; lane 31b): a reference with `use_clip=false`, an adapter and no `audio` | `/speak` | spoken from the adapter alone |
| changed 2026-11-02 (ASR, t42; lane 31d): a transcribe during a bake | `/voice/transcribe` | 409 "a voice is being baked (N%)…" (the voice gate) |
| changed 2026-11-02 (ASR, t42): bad base64, an sha mismatch, a clip over 4 MiB, not a mono PCM-16 WAV at 48 kHz, no frames, cut short, over 30 s, a language the ASR entry does not speak | `/voice/transcribe` | 422 with the reason; no load |
| changed 2026-11-02 (ASR, t42): no ASR entry on this profile (jetson, cloud) | `/voice/transcribe` | 503 "hearing the words of a take is not offered on this device yet" |
| changed 2026-11-02 (ASR, t42): Whisper weights not on disk, another snapshot folder, a CPU device | ASR child load | 500 with the load message ("… not on disk: run `make models`") |
| changed 2026-11-02 (ASR, t42): the child reports another revision than the catalog's | `/voice/transcribe` | 500 "asr identity mismatch: …" (the D26 rule) |
| changed 2026-11-02 (ASR, t42): `POST /models/asr/ensure` during a bake | models API | 409 |

**V. The Voices page — Slice V (pair t30; D67, D68, D69). Today's features, a new place and a new layout. D20 is reversed.**

V1. Nav and routes.

- `web/templates/base.html` nav: Upload, Library, Chat, Organize, **Voices** (`/voices`), five boosted links; Settings stays a button (`:27-30`).
  The nav comment (`:11-15`) says "five". `web/templates/organize.html:23-25` (the "Saved voices" link and its comment) goes.
- Every `/organize/voices/*` route moves (`web/voices_api.py`; owner checks `_voice_or_404` stay). Literal paths register BEFORE
  `/voices/{voice_id}` (`voice_id` is an `int`, so a later literal path would answer 422):

  | Method | Path | Does | Reply |
  |---|---|---|---|
  | GET | `/voices` | the grid | `voices.html` |
  | GET | `/voices/list` | the cards (polled every 2 s while a card is deleting) | `_voice_cards.html` |
  | POST | `/voices` | form `name` → `create_voice` | `_voice_cards.html`; the new card carries `data-created` and the script opens it |
  | GET | `/voices/preview` | the test state (global) | `_voice_preview.html`, `Cache-Control: no-store` |
  | POST | `/voices/preview/cancel` | cancel or clear the test | `_voice_preview.html` |
  | GET | `/voices/{voice_id}` | the leaf, `?tab=recordings` (default) \| `model` \| `test` | `voice.html` |
  | GET | `/voices/{voice_id}/list` | the leaf body (polled while a take is deleting) | `_voice_body.html` |
  | POST | `/voices/{voice_id}/rename` | form `name` | `_voice_body.html` |
  | POST | `/voices/{voice_id}/delete` | enqueue + `voice_drain.schedule()` | `_voice_body.html` (deleting); the script then closes to the grid |
  | POST | `/voices/{voice_id}/examples` | add a take (multipart, as today) | `_voice_body.html` |
  | POST | `/voices/{voice_id}/examples/{example_id}/delete` | enqueue + schedule | `_voice_body.html` |
  | GET | `/voices/{voice_id}/examples/{example_id}/audio` | the take | `audio/wav` |
  | POST | `/voices/{voice_id}/examples/{example_id}/preview` | a test (form as today) | `_voice_preview.html` |
  | GET | `/voices/{voice_id}/examples/{example_id}/preview/audio` | the last test | `audio/wav` |
  | POST | `/voices/{voice_id}/film-example` | form `language`, `example_id` | `_voice_body.html` |
  | GET | `/organize/voices` | old bookmarks | **308** `Location: /voices` |

  Status codes, the disk guard, `drain.wake()` and the t27 completion certificate stay as today; they move with their routes.
  **Changed 2026-11-02 (rework, t39):** Slice B removes the three per-take rows above (`…/examples/{example_id}/preview`,
  `…/examples/{example_id}/preview/audio`, `/voices/{voice_id}/film-example`) and adds the test, bake and transcript routes of K4 and K5.
- Templates: `voices.html` becomes the grid; `voice.html` (new) is the leaf; `_voice_cards.html` (new) is `#voice-cards`; `_voices_list.html`
  becomes `_voice_body.html` (`#voices`, one voice); `_voice_preview.html` stays (its paths change).
- Text elsewhere (D67): `feed.html:52` → `Add a voice under <a href="/voices">Voices</a>.`; `web/app.py:1035-1039` and `feed/build.py:607-616`
  "under Organize → Saved voices" → "under Voices". Nothing else in `feed/` changes in this slice.

V2. The grid `/voices` — a top-level grid (§13.1), no ×, no `ctx`. The figures and takes in these wireframes are examples, not your stored
voice (t31; t28: no "3 clips / 37 s" claim unless it is read from the stored rows):

```
┌───────────────────────────────────────────────────────────────────────────────┐
│ Upload   Library   Chat   Organize   [Voices]                 RAM 18/32 GB  ⚙ │
├───────────────────────────────────────────────────────────────────────────────┤
│ Voices                                                                        │
│ A voice is a person's own voice. It can narrate a film in each language it    │
│ has takes for.                                                                │
│ New voice  [ Grandma____________ ]  [Create]                                  │
│                                                                               │
│ ┌───────────────────────────────┐  ┌───────────────────────────────┐          │
│ │ me                            │  │ Grandma                       │          │
│ │ Russian · 0:37 · 3 takes      │  │ English · 1:12 · 6 takes      │          │
│ │ Films: take #5 (newest)       │  │ Films: take #9 (your choice)  │          │
│ └───────────────────────────────┘  └───────────────────────────────┘          │
│ ┌───────────────────────────────┐  ┌───────────────────────────────┐          │
│ │ Dad                           │  │ Mom · deleting…               │          │
│ │ No takes yet                  │  │                               │          │
│ └───────────────────────────────┘  └───────────────────────────────┘          │
└───────────────────────────────────────────────────────────────────────────────┘
```

- A card is a plain `<a class="voice-card" href="/voices/{id}">` (a full navigation, never boosted, §13.1) with the name and, per language with
  takes, "<Language> · m:ss · N takes" and the film take ("Films: take #5 (newest)" / "(your choice)" / "Films: the chosen take was deleted").
  No playback on a card. A pending voice is a `<div>` with "deleting…" and no link; the 2 s poll of `/voices/list` runs while one is shown.
  **Changed 2026-11-02 (rework, t39):** Slice B replaces the "Films: take …" line (and the wireframe's) with the voice model state of K6.
- With no voice copy on the device: "Voice copy is not available on this device." and no New voice; existing cards still open (rename, delete).
- Create: the reply is the cards; the script opens the card that carries `data-created` with `location.assign` — the grid→leaf push.

V3. The leaf `/voices/{id}` — one level under the grid. Tabs are state of the leaf, never history. The figures and takes below are examples:

```
┌───────────────────────────────────────────────────────────────────────────────┐
│ Upload   Library   Chat   Organize   [Voices]                 RAM 18/32 GB  ⚙ │
├───────────────────────────────────────────────────────────────────────────────┤
│ Voices › me                                                       [⋯]   [×]  │
│ Russian: films use take #5 (newest)              ⋯ = Rename… · Delete voice…  │
│                                                                               │
│ [ Recordings ]   Voice model    Test                                          │
│ ───────────────────────────────────────────────────────────────────────────── │
│ Russian · 3 takes · 0:37                                                      │
│   #5  [▶ 0:12]  recorded · 2026-11-02  “Весь день по крыше стучал…”  Test  Delete │
│   #4  [▶ 0:12]  recorded · 2026-11-02  “По воскресеньям рынок…”      Test  Delete │
│   #3  [▶ 0:13]  recorded · 2026-11-01  “Утренний свет медленно…”     Test  Delete │
│ English · no takes                                                            │
│                                                                               │
│ Add a take   Language [Russian ▾]                                             │
│   [x] This is my voice, or I have this person's permission to copy it.        │
│   Record clean, normal speech in a quiet room. …                              │
│   [Record]  [Upload a file]                                                   │
│   ┌ recorder — opens here ───────────────────────────────────────────────┐    │
│   │ Read aloud:  “В поезде было тихо, только мерно стучали колёса…”      │    │
│   │ [Different text]   [Start recording]                                 │    │
│   └──────────────────────────────────────────────────────────────────────┘    │
└───────────────────────────────────────────────────────────────────────────────┘

  Voice model tab                                 Test tab
┌──────────────────────────────────────────────┐ ┌──────────────────────────────────────────────┐
│ Films narrated by «me» copy the voice from    │ │ Take   [#5 · Russian · 0:12 · recorded ▾]    │
│ one take per language.                        │ │ Text   [The text you read (not checked) ▾]   │
│ Russian  [#5 · 0:12 · “Весь день…” (newest) ▾]│ │        “Весь день по крыше стучал дождь…”    │
│ English  no takes                             │ │ [Generate]                                   │
│                                               │ │ generating the voice · 42%  [████░░░░] Cancel│
│ (tombstone)                                   │ │ ──────────────────────────────────────────── │
│ Russian [— the chosen take was deleted — ▾]   │ │ Original  take #5  “Весь день по крыше…” [▶] │
│ The Russian take chosen for films was         │ │ Generated “RU-1 …” · voxcpm2 @ 32279ef ·     │
│ deleted. Films narrated by «me» in Russian    │ │   made in 18.2 s · conditioning restored [▶] │
│ fail until you choose another.                │ │                                              │
└──────────────────────────────────────────────┘ └──────────────────────────────────────────────┘
```

- Header: "Voices" is a link with the close behaviour (V4), then the name and one state line per language with takes (the film take, or the
  tombstone). `<details class="voice-menu">` holds the Rename form and Delete voice (its confirm as today, `_voices_list.html:31`). ×.
- Tabs: `<div role="tablist">` with three `<button role="tab" data-tab="recordings|model|test">`. The server renders `?tab` (unknown → `recordings`);
  a switch shows the `[data-tab-part]` blocks of that tab and calls `history.replaceState(history.state, "", <url with ?tab>)`. A switch keeps the
  recorder, the typed test text and the selections.
- `#voices` (`_voice_body.html`) holds the server-rendered parts of the leaf: the header name and state lines, the takes per language, the "Add a
  take" row and the Voice model selectors (D69). The recorder (`#voice-recorder`) and the test panel (`#voice-preview`) stay outside `#voices` (D53).
- Recordings: one row per take, "#id · ▶ · seconds · recorded/uploaded · date · the reading text (60 characters) · Test · Delete". Test switches to
  the Test tab with that take selected. The recorder opens directly under "Add a take" (D68 keeps it there across a refresh).
- Test: a take select (every usable take of this voice, "#id · Language · m:ss · recorded|uploaded"), the text select and textarea of §2.16 F,
  Generate, the job line (stage, percent, Cancel), and the comparison: "Original — take #<id>: “<its reading text>”" ▶ and the take's stored test
  line ▶ (§2.16 F). The comparison always names the selected take. A test of another voice shows "Testing «<name>» #<id> · N% · Cancel".
- **Changed 2026-11-02 (rework, t39):** this is Slice V as built. Slice B replaces its film parts: the header state lines and the wireframe's
  "Russian: films use take #5 (newest)"; the Test button and the film-take Delete confirm on every take row; the tip "Films use one example
  per language…"; the Voice model tab's film-take select and tombstone; the Test tab's take select and per-take comparison. K6 holds the
  new header, Recordings, Voice model and Test. The tabs, the recorder, Esc and close do not change.

V4. Esc and close.

- Esc: a live recording → stop and discard; an unsaved take → discard; a prepared recorder → close it; a focused text field → blur; else close.
  Tabs are not an Esc level, and the test panel is no longer one (it is a tab).
- Close (×, the "Voices" link, Esc): when `#voices-page[data-from-grid]` is set → `history.back()`; else `location.replace("/voices")`. The script
  sets `data-from-grid` at mount when `document.referrer` has the same origin and the path `/voices` (card links are full navigations, so the
  referrer IS the grid exactly then). The attribute lives in the DOM, so a bfcache or htmx history restore keeps it, and a deep link never has it.
  `history.length` is never read. (`photo.html:181` and `feed.html:289` keep `history.length > 1`; aligning them is a follow-up, t30.)

V5. `voices.js` lifecycle — an acceptance condition (t30; D68).

- One global wiring behind `if (!window.__voicesWired) { window.__voicesWired = true; … }` (pattern `organize.html:64-66`). Every listener sits on
  `document` and resolves its nodes at event time (`document.getElementById("voices-page")`, then `closest`). No node is captured at module scope
  (today `voices.js:16-23`).
- `mount()` runs when `#voices-page` is present and has no `data-mounted`: at script run, on `htmx:historyRestore`, on `htmx:afterSettle`, and on
  `pageshow` with `e.persisted` (**changed 2026-11-02, t31**: a native bfcache restore — the leaf's `history.back()` to the grid, for one — runs no
  script again; pattern `organize.html:64-68`). Each path ends with exactly one poller on the current nodes. It
  sets `data-mounted`, `data-from-grid` (V4), reads `?tab`, and reconnects every job from the server: the deleting poll, the test job (GET
  `/voices/preview`, no-store), and in Slice B the bake job.
- `unmount()` runs on `htmx:beforeHistorySave` and on `pagehide`: it stops every microphone track and the `MediaRecorder`, clears every timer,
  bumps `generation` (so a late fetch reply or callback does nothing), resets the recorder to hidden, and removes `data-mounted` before htmx
  takes its snapshot.
- On `htmx:confirm` from a link inside `nav.nav` (not `#settings-open`) while a recording runs or a take is unsaved: `preventDefault()` and
  `confirm("Discard the recording you have not saved?")`; yes → `unmount()` and `issueRequest()`; no → nothing changes. Close asks the same.
- Pure parts for node tests: `cameFromGrid(referrer, origin)`, `tabFrom(search)`, `leaveNeedsConfirm(state)`, and `lifecycle(deps)` with
  injected timers, tracks and fetch.
- **Changed 2026-11-02 (pair t36, in the code): ownership through body decoding.** The lifecycle's `session.fetch` resolves to a guarded
  response: `text()` and `json()` fulfil, and reject, to a promise that never settles once the session is dead; `ok`, `status` and
  `headers` pass through, and `detailOf` inherits the guard. Every async site captures its owner `own` (and `const s = sync`) before its
  first await and checks `own.alive()` right before each DOM effect, retry or recovery read: `pollWhileDeleting`, `send` (detail and
  success), `pollPreview`, `mutate` (with its recovery reads), and `showPreviewState` with the captured sync. No code reads the global `sync`
  or `mounted` after an await. `fetchStatus(own)` and `stopPreviewTimers(own)` act through the owner, so an old page's reconcile or repair
  cannot touch the new page's timers. Slice B's bake poller follows the same rules (K6).

**K. Slice B — kit, transcripts, the bake in `app`, the voice model as the only way a saved voice speaks (D70-D74, D76-D89).**
**Changed 2026-11-02 (rework, t39): Slice B starts after the user approves this rework. It does not wait for §6.5 PASS, §6.6 PASS or a gate
answer.** (Before: "Starts only after §6.5 PASS, §6.6 PASS and the gate answer.")
**Changed 2026-11-02 (ASR, t42): K2 is rewritten (the words of a take come from local speech recognition; D90-D104); K0, K1, K3-K7 carry
the changes it brings.**

K0. What the rework changes (2026-11-02, t39 and the user's requests in `## Decisions`).

- **Films and tests speak ONLY through the voice model.** The voice model is the LoRA adapter baked from ALL the person's takes. A film or a
  test sends it with `use_clip=False` (the adapter alone, variant C of §2.17 I; `audio/voxcpm_voice.py:283-285`), and the reference carries
  no take (D76). There is no prompt take, no film-take select, no pin and no one-take fallback.
- **A voice can narrate a film in language L only when its model is usable and learned at least one take in L** (D77). Otherwise the film,
  the Prepare check and the picker refuse with a "bake required" text and a way to the voice page (K5, D78). An explicit saved narrator that
  cannot speak never becomes Automatic or a preset (D33, D38).
- **The model stays** when takes are added, words are edited or takes are deleted (the user's gate answer). Rebake trains a new version from
  all current takes. The page shows the version, the snapshot it learned from (takes, minutes, languages) and the changes since it (D79).
- **One Test per person** (the user's request; D80): a language of the model, then a fixed sentence, the words of a take or your own text;
  one Generate; through the model only.
- **What goes from the code** (§1.9): `FilmExample`, `film_example`, `set_film_example`, the take pairs of `castable`, `save_conditioning`,
  `_keep_conditioning`, `VoicePreview`, `save_preview`, `previews`, `read_preview_audio`; the routes `POST /voices/{id}/film-example`,
  `POST /voices/{id}/examples/{eid}/preview` and `GET /voices/{id}/examples/{eid}/preview/audio`; every page text that names a take for films.
  `example_for`, `conditionings` and `reference_for` stay, for the check script's diagnostic variants A and B only (D82).
  **Changed 2026-11-02 (rework, t40):** these names leave the store only in lane 32d, after lanes 32b and 32c no longer import them. `app`
  runs with `--reload` and must keep loading at every step, so lane 31 only ADDS: it keeps every name that `feed/build.py`,
  `feed/voice_preview.py`, `web/app.py` and `web/voices_api.py` import today (§1.9), and `castable` keeps its take pairs until 32d. The model
  form is a new name, `castable_models` (D89).
- **What stays from the earlier K:** K1 (the kit); K2 (the words; unresolved takes block a bake and are shown, never left out); the immutable
  bake snapshot; the publication barrier; the reattach by `bake_key`; versions that only rise; t31 items 4 and 5. **t31 item 6 is superseded**
  (a pending or deleted training take no longer makes the model unusable, and no publication re-checks the trained take ids): the user chose
  to keep the model.
  **Changed 2026-11-02 (ASR, t42):** K2 no longer stays: the user confirms no words, and no take blocks a bake for its words. ASR hears
  the words before the snapshot (K2). A bake is refused only with no take, or when no take holds heard words after the transcription
  stage.
- **Existing data** (D81): films already made from one take stay as they are until they are regenerated (D33). The rows of
  `voice_film_examples`, `voice_previews` and `voice_conditionings` stay. Films, tests and the page no longer read or write them; the drain
  still removes their files with their take or voice. No table is dropped, no row is deleted or migrated, and no file is removed by the
  rework itself.
- **Until a voice is baked, it cannot narrate a film.** No voice has a model today (§1.9: no `voice_models` table exists yet), so after
  Slice B lands, a film narrated by a saved voice fails with "«<name>» has no voice model yet…" until the user bakes that voice in the app.
  **If §6.5 fails** (C does not pass on this Mac), saved voices cannot narrate on this Mac until that is solved. The main session reports it
  to the user as a blocker. It never puts back a one-take path.

K1. The recording kit — `feed/saved_voices.py` (D70).

```python
@dataclass(frozen=True)
class KitText:
    id: str      # "<lang>-<style>-<n>", or one of the four reading-text ids (D45)
    style: str   # a KIT_STYLES key
    text: str    # the exact words to read; numbers written as words

KIT_STYLES: dict[str, tuple[str, str]] = {   # style -> (label, instruction shown over the text)
    "normal":    ("Normal", "Read in your normal voice."),
    "slow":      ("Slow", "Read slowly and clearly, with short pauses."),
    "fast":      ("Fast", "Read quickly, as if you are in a hurry, but keep every word clear."),
    "calm":      ("Calm", "Read calmly and softly."),
    "happy":     ("Happy", "Read with joy, as good news."),
    "sad":       ("Sad", "Read sadly and quietly."),
    "serious":   ("Serious", "Read seriously, as an important announcement."),
    "excited":   ("Excited", "Read with excitement."),
    "question":  ("Questions", "Read every question as a real question."),
    "emphasis":  ("Emphasis", "Read with energy and stress the words that matter."),
    "artic":     ("Articulation", "Read clearly and take your time with each sound."),
    "numbers":   ("Numbers and names", "Read the numbers and names as they are written."),
}
RECORDING_KIT: dict[str, tuple[KitText, ...]]                    # language -> 26 texts: the four of §2.15 I first, then two per other style
RECORDING_PROMPTS = {lang: {k.id: k.text for k in kit} for lang, kit in RECORDING_KIT.items()}   # derived; add_example's check is unchanged
```

| id | text |
|---|---|
| en-slow-1 | Every morning the baker opens the shop at six. First the lights come on, then the ovens, and soon the whole street smells of fresh bread. |
| en-slow-2 | Take one deep breath. Hold it for a moment. Now let it go, slowly, and listen to the room around you grow quiet. |
| en-fast-1 | Quick, grab your coat, the bus leaves in two minutes, and if we miss it we will be late for the film again! |
| en-fast-2 | She packed the bags, locked the door, checked the tickets twice, and ran down the stairs before anyone could stop her. |
| en-calm-1 | The garden is quiet in the evening. A light wind moves the leaves, and the last birds sing somewhere far away. |
| en-calm-2 | There is no need to hurry. The candle is lit, the book is open, and the whole afternoon belongs to us. |
| en-happy-1 | We did it! The letter came this morning, and they said yes. I cannot stop smiling, and I want to tell everyone. |
| en-happy-2 | What a wonderful surprise! The whole family is here, the music is playing, and the cake is bigger than the table. |
| en-sad-1 | The platform was empty when the last bus left. I waved until I could not see it any more, and then I walked home alone. |
| en-sad-2 | Her old chair still stands by the window. Nobody sits in it now, but every evening someone turns on the lamp beside it. |
| en-serious-1 | Attention, please. The museum will close in fifteen minutes. Please take your belongings and walk to the main exit. |
| en-serious-2 | This decision is not easy, and it will change many things. We have to think about it carefully before we agree. |
| en-excited-1 | Look up, look up, it is snowing! The first snow of the year, and it is already covering the cars and the roofs! |
| en-excited-2 | The doors are opening, the crowd is cheering, and here they come, right now, the whole band on one stage! |
| en-question-1 | Did you remember the keys? And the tickets? Are you sure we have everything, or should we check one more time? |
| en-question-2 | Where did you find this photo? Who is the man in the hat, and why is everyone in the room looking at him? |
| en-emphasis-1 | No, not the blue box, the red one! The red box goes on the top shelf, and the blue one stays right here! |
| en-emphasis-2 | I said I would be there at noon, and I was there at noon. Not one minute late. Not one! |
| en-artic-1 | Six slim swans swam swiftly south, while three thin thrushes sang through the thick green thistles. |
| en-artic-2 | A proper copper coffee pot keeps the coffee properly hot; a plain cheap pot does not keep it hot at all. |
| en-numbers-1 | Anna and Peter moved to Lisbon on the third of March, two thousand nineteen, into flat number forty-seven. |
| en-numbers-2 | The recipe needs two hundred grams of flour, three eggs, and one and a half cups of milk, says Aunt Maria. |
| ru-slow-1 | Каждое утро пекарь открывает лавку в шесть часов. Сначала загорается свет, потом печи, и скоро вся улица пахнет свежим хлебом. |
| ru-slow-2 | Сделайте глубокий вдох. Задержите дыхание. А теперь медленно выдохните и послушайте, как в комнате становится тихо. |
| ru-fast-1 | Быстрее, бери куртку, автобус уходит через две минуты, и если мы опоздаем, то опять пропустим начало фильма! |
| ru-fast-2 | Она собрала сумки, заперла дверь, дважды проверила билеты и сбежала по лестнице, пока никто не успел её остановить. |
| ru-calm-1 | Вечером в саду тихо. Лёгкий ветер шевелит листья, и где-то далеко поют последние птицы. |
| ru-calm-2 | Спешить некуда. Свеча горит, книга открыта на любимой странице, и весь этот длинный день принадлежит только нам. |
| ru-happy-1 | У нас получилось! Утром пришло письмо, и нам ответили «да». Невозможно перестать улыбаться, хочется рассказать всем! |
| ru-happy-2 | Какой чудесный сюрприз! Вся семья в сборе, играет музыка, а торт больше, чем сам стол. |
| ru-sad-1 | Остановка опустела, когда ушёл последний автобус. Мы долго махали вслед, а потом молча побрели домой. |
| ru-sad-2 | Её старое кресло всё ещё стоит у окна. Теперь в нём никто не сидит, но каждый вечер кто-то включает рядом лампу. |
| ru-serious-1 | Внимание. Музей закрывается через пятнадцать минут. Пожалуйста, возьмите свои вещи и пройдите к главному выходу. |
| ru-serious-2 | Это решение непростое, и оно многое изменит в нашей жизни. Нужно спокойно всё обдумать, прежде чем соглашаться. |
| ru-excited-1 | Смотри, смотри, пошёл снег! Первый снег в этом году, крупный и мягкий, и он уже засыпает машины, крыши и скамейки! |
| ru-excited-2 | Двери открываются, зал аплодирует, и вот они выходят, прямо сейчас, вся группа на одной сцене! |
| ru-question-1 | Ключи взяли? А билеты? Паспорта тоже? Точно ничего не забыли, или лучше проверить ещё раз, пока не поздно? |
| ru-question-2 | Где нашлась эта фотография? Кто этот человек в шляпе, и почему все в комнате на него смотрят? |
| ru-emphasis-1 | Нет, не синюю коробку, а красную! Красная идёт на верхнюю полку, а синяя остаётся прямо здесь! |
| ru-emphasis-2 | Встреча была назначена ровно на полдень, и все пришли ровно в полдень. Ни минутой позже. Ни одной! |
| ru-artic-1 | Шла Саша по шоссе и сосала сушку. Карл у Клары украл кораллы, а Клара у Карла украла кларнет. |
| ru-artic-2 | От топота копыт пыль по полю летит. Сшит колпак не по-колпаковски, надо колпак переколпаковать. |
| ru-numbers-1 | Анна и Пётр переехали в Лиссабон третьего марта две тысячи девятнадцатого года, в квартиру номер сорок семь. |
| ru-numbers-2 | Для пирога нужно двести граммов муки, три яйца и полтора стакана молока, так говорит тётя Мария. |

- The four texts of §2.15 I are the "normal" style and keep their ids and their order, so the first text of each language does not change.
- No text holds a word of the voice_check sentences (§2.15 I list: lake, fog, dog, Julia, July, cold, road, hills, valley, porch, stars). The Russian
  texts use no first-person singular past-tense verb (D23). Each text has 86-137 characters, about 6-10 s at 14 characters a second (the D28
  rate; planner check, 2026-11-02); slow reading takes longer. A take outside 5-30 s is refused as today (§2.7).
- A kit text can have several takes; "Record again" adds one.
- **Changed 2026-11-02 (ASR, t42; D99, D101):** every kit text is a suggestion. The recorder shows it with one more line: "Read it, or say
  anything else: the bake hears what you say." `kit_id` records the text the take was shown, so the take sits on that kit row; a
  recording posted with no `prompt_id` stores `kit_id` NULL (D99). Takes with no kit text (uploads, and the recordings stored before the
  kit) are listed under "Other takes".

K2. The words of a take — local speech recognition (D90-D95). **Changed 2026-11-02 (ASR, t42): rewritten.** The earlier K2 is superseded
and kept short at the end of this item.

K2a. The rule.

- A take is free speech. The kit text is a suggestion only: `add_example` stores `kit_id` (the text shown) and never copies the kit text
  into `transcript`. The user types nothing, confirms nothing and sees no words editor.
- Before a bake trains, ASR in the models service hears the words of every take that has no ASR result yet: new recordings, uploads, and
  the takes stored before this amendment, whose `transcript` holds the text they were shown (§1.10). The bake trains on (audio, heard words).
- The heard words are stored on the take BEFORE the snapshot is written, so the snapshot (`voice_bake_runs`), `snapshot_hash`, the
  service's `dataset_sha256` and `voice_model_takes.fingerprint` hash the same words (`modelsvc/voice_bakes.py:252-256`,
  `feed/saved_voices.py:1261-1268`).
- A take that ASR heard but that gives no usable words (no letter and no digit in them, or more than `TRANSCRIPT_MAX` = 1000 characters,
  the service's `TEXT_MAX`) stays in the voice with its audio, and shows why on its row and on the Voice model tab. A bake leaves it out
  and says so; it never blocks a bake, and it is never dropped in silence (D94). A transcribe call that fails (a service error, a timeout)
  ends the job and stores nothing for that take.
- A take is heard once. Its words never change after that, so a model's fingerprints stay exact and no take is "edited" any more.
- The D76 speak path does not change: films and tests send the adapter alone, with no take and no words.
- **Changed 2026-11-02 (ASR plan check, t44):**
  - Words with a letter OR a digit are usable ("1 2 3 4 5" is kept). Blank or symbol-only words get the reason "the speech recognition
    found no words in it" (`NO_WORDS`); the page never says "no speech", because the model reports no such thing. Over 1000 characters
    keeps its own reason (`TOO_LONG`) in every text that names a left-out take. The state is named `unusable` (it was `unheard`).
  - `add_example` takes no lock (`web/voices_api.py:514-548`), so a take can arrive while the stage runs. The stage repeats until the
    snapshot read under `job_lock` finds no `waiting` take; that read is the cutoff (K4). A take deleted during the stage is skipped (K4).
  - The take's language label stays the decoding language: there is no language detection (D77, kept per t43).

K2b. Storage — `db/schema.sql`, `db/connection.py`, `feed/saved_voices.py` (lane 31c; D91, D96-D100).

```sql
-- voice_examples, in its CREATE and as two _ADDED_COLUMNS entries (D39: no CHECK, no version bump):
  transcript       TEXT,  -- the words ASR heard; for a take stored before ASR, the text it was shown (replaced at its next bake); NULL = none yet
  transcript_model TEXT,  -- '<asr key>@<revision>' that heard this take; NULL = not heard yet (it is heard at the next bake)
  transcript_error TEXT,  -- why ASR gave no usable words ('the speech recognition found no words in it', …); NULL otherwise
  transcript_confirmed_at TEXT,  -- unused; kept: databases already hold it, and no column is dropped (D39)
  kit_id           TEXT,  -- the recording-kit text the take was shown, a suggestion; NULL for an upload or a recording with no kit text
```

| state (`VoiceExample.words_state`) | condition | row text | in a bake |
|---|---|---|---|
| `waiting` | `transcript_model` IS NULL | nothing | heard first; then trained when its words are usable |
| `heard` | `transcript_model` set, `transcript` set | nothing | trained |
| `unusable` | `transcript_model` set, `transcript` NULL, `transcript_error` set | "not used by the voice model: <transcript_error>" | left out, named |

```python
TRANSCRIPT_MAX = 1000                                    # kept; equals the service's TEXT_MAX
NO_WORDS = "the speech recognition found no words in it"
TOO_LONG = f"it holds more than {TRANSCRIPT_MAX} characters of words"
@dataclass(frozen=True)
class VoiceExample: id, voice_id, language, source, transcript, storage_key, content_hash, seconds, sample_rate, created_at, deleting,
                    transcript_model, transcript_error, kit_id          # transcript_confirmed_at and `unresolved` go
    @property
    def words_state(self) -> str                         # "waiting" | "heard" | "unusable"
def heard_words(text: str) -> tuple[str | None, str | None]
    # (words, error): whitespace runs → one space, strip; no character that is a letter or a digit (str.isalpha() or str.isdigit())
    # → (None, NO_WORDS); over TRANSCRIPT_MAX → (None, TOO_LONG). Changed (ASR plan check, t44): digits count.
def not_used_text(takes: Iterable[VoiceExample]) -> str
    # changed (ASR plan check, t44): "take #9 — the speech recognition found no words in it; take #4 — it holds more than 1000 characters
    # of words": one entry per `unusable` take with its own `transcript_error`, by id; the blocked, done, "Not used" and Test texts use it
def takes_to_hear(conn, owner_id, voice_id) -> list[VoiceExample]     # non-pending takes with transcript_model NULL, by id
def take_to_hear(conn, owner_id, voice_id, example_id) -> VoiceExample | None
    # changed (ASR plan check, t44): that take when it exists, it and its voice are not pending, and transcript_model is NULL; else None
def set_heard_words(conn, owner_id, voice_id, example_id, *, text: str, model: str) -> VoiceExample | None
    # one BEGIN IMMEDIATE: the voice and the take exist, are not pending, and transcript_model is still NULL; then transcript, transcript_model,
    # transcript_error from heard_words(text). None = nothing written (the take went away, or another job heard it first).
@dataclass(frozen=True)
class BakeSnapshot:
    takes: tuple[VoiceExample, ...]      # the `heard` takes that are not pending, by id: what a bake sends now
    waiting: tuple[VoiceExample, ...]    # not heard yet
    unusable: tuple[VoiceExample, ...]    # heard, no usable words: left out and named
    seconds: float                       # of `takes`
    by_language: dict[str, float]        # of `takes`
    dataset_hash: str                    # snapshot_hash(takes)
@dataclass(frozen=True)
class ModelChanges:
    added: int     # non-pending takes that are not in voice_model_takes and not `unusable` (a Rebake hears and trains them)
    deleted: int   # voice_model_takes rows whose take is gone or pending
    unusable: int   # non-pending `unusable` takes (no Rebake trains them)
```

- `add_example(conn, storage, owner_id, voice_id, *, language, source, data, consent, prompt_id=None)`: the `transcript` parameter goes. A
  recording with a `prompt_id` checks it against `RECORDING_PROMPTS[language]` as today and stores it as `kit_id`; with none, `kit_id`
  is NULL (D99). `transcript`, `transcript_model` and `transcript_error` start NULL for every take.
- `set_transcript` goes, and so does `VoiceExample.unresolved`. `ModelChanges.edited` goes (a heard take's words never change).
- `voice_model_for` with no row → `ModelChanges(added=<waiting + heard>, deleted=0, unusable=<unusable>)`; with a row, as the table says.
  `film_model`, `castable_models`, `save_voice_model`, the snapshot JSON and `read_bake_run` do not change (`BakeRun`, D96).
- No take is migrated. A take stored before t42 keeps its shown text in `transcript` until its first bake, which replaces it with the
  heard words (D91). No code reads that shown text as words: `takes_to_hear` lists it, `words_state` says `waiting`, and the Test tab does
  not offer it (K5).

K2c. ASR in the models service — `audio/whisper_asr.py` (new), `modelsvc/app.py`, `modelsvc/backends/*`, `inference/models_client.py`
(lane 31d; D90, D93, D103). The catalog entry is in §2.17 B, the unit in C, the bake's unload in E, the client in G.

```python
# audio/whisper_asr.py — models-service-only. Module scope imports no torch, transformers, huggingface_hub or scipy
# (the pattern of audio/voxcpm_voice.py:8-11), so take_samples is unit-tested.
RATE_IN = 48000          # the take format (D21)
RATE_ASR = 16000         # preprocessor_config.json sampling_rate (§1.10)
MAX_SECONDS = 30.0       # Whisper's window; a take is 5-30 s (§1.10)
LANGUAGES = ("en", "ru")
def take_samples(data: bytes) -> np.ndarray   # mono PCM-16 at 48000 Hz with every frame, 0 < seconds <= 30 → float32 in [-1, 1]; else ValueError
class WhisperASR:
    def __init__(self, repo: str, device: str, revision: str) -> None
    @classmethod
    def loaded(cls, model, processor, revision: str, *, resample, device: str = "mps") -> "WhisperASR"   # test seam
    def transcribe(self, audio: bytes, language: str) -> dict   # {"text", "revision", "device", "seconds", "elapsed_s"}
```

- `__init__`, in this order: refuse a device that is not `mps` and does not start with `cuda` (`audio/voxcpm_voice.py:195-196`);
  `snapshot_download(repo, revision=revision, local_files_only=True)`, `LocalEntryNotFoundError` → `RuntimeError("Whisper weights for
  <revision> are not on disk: run `make models`")`; the snapshot folder name must equal the revision (`audio/voxcpm_voice.py:203-214`);
  `WhisperProcessor.from_pretrained(path)`; `WhisperForConditionalGeneration.from_pretrained(path, dtype=torch.float32,
  device_map=device).eval()` (the repo's load pattern, §1.10; float32 because nothing measured float16 on MPS here).
- `transcribe`: `language` not in `LANGUAGES` → `ValueError`; `samples = take_samples(audio)`; `x = resample(samples)` (the real one:
  `scipy.signal.resample_poly(samples, 1, 3)`, 48000 → 16000); `features = processor(x, sampling_rate=16000,
  return_tensors="pt").input_features.to(device, torch.float32)`; under `torch.inference_mode()`: `ids = model.generate(features,
  language=language, task="transcribe")`; `text = processor.batch_decode(ids, skip_special_tokens=True)[0]`. The child returns the text
  as decoded: `heard_words` in `app` is the one rule that judges it.
- `CompositeBackend.transcribe(audio: bytes, language: str) -> dict`: no `asr` unit (`slots is None or not slots.has_unit(ASR_UNIT)`) →
  `TranscribeNotOffered` (503); `language` not in the entry's `languages` → `TranscribeRefused` (422); `with self._gate.admitted():`
  (closed → `BakeRunning`, 409) read `loaded_before = registry.is_resident(ASR_UNIT)` and `free_before = governor.state().free_mb`, then
  `self._run([ASR_UNIT], Priority.BATCH, "hearing the words of a take", lambda: slots.worker(ASR_UNIT).call("transcribe", audio,
  language))`. The reply's `revision` must equal the entry's, else `RuntimeError("asr identity mismatch: …")` (D26). When the call loaded
  the child, one log line (D103). It returns `{"text", "model": entry.key, "revision", "seconds", "elapsed_s"}`.
  **Changed 2026-11-02 (ASR plan check, t44):** `governor.acquire` may evict other units to make room (`modelsvc/governor.py:58-81`), so
  the `free_mb` change across the call is the net change of the request, never the model's cost. The line reads `asr <key>@<first 7 of
  revision> loaded by this request: free_mb <before> → <after>, net change <N> MB; evicted: <units, or none>`, where the evicted units
  are `governor.state().resident` before the call minus after it. It never uses the words "cost", "drop" or "measured"; `cost_measured`
  stays False.
- `modelsvc/app.py`: `TranscribeRequest(audio: str, sha256: str, language: str)`, `TranscribeResponse(text, model, revision, seconds,
  elapsed_s)`; `POST /voice/transcribe` → 200; 409 `BakeRunning`; 422 bad base64, an sha mismatch, over `MAX_CLIP_BYTES`, the WAV checks
  of `voice_bakes._wav_seconds` (`modelsvc/voice_bakes.py:493-509`), over 30 s, `TranscribeRefused`; 503 `TranscribeNotOffered`
  "hearing the words of a take is not offered on this device yet"; 500 a load or model error with its message (§2.3). The parent decodes
  and checks before any load, so a refused clip loads nothing.
- `modelsvc/backends/base.py` declares `transcribe` and the two errors, beside `SlotOff` and `CloningUnavailable`:
  `class TranscribeNotOffered(RuntimeError)` and `class TranscribeRefused(ValueError)`; `fake.py`: `FakeBackend.transcribe` returns a scripted text per sha256 (default
  `"fake words"`), `model` `"fake-asr"`, `revision` `"fake"`. `inference/models_client.py`: `transcribe` (§2.17 G).
- `scripts/fetch_models.py:wanted` appends `("asr", "<repo>@<revision>")` of `transcriber_for(profile)` when `adapter_entry_for(profile)`
  exists, so `make models` (and `make up`) fetches Whisper on mac once, about 1.6 GB (§1.10). The agent never runs it.
- Memory: the Whisper child (estimate 4000 MB) can sit beside the voice child (10500) under the mac budget of 24000 (§1.8); the governor
  evicts it like any unit, and a bake unloads it before its lease (§2.17 E). It is never resident beside the training child.

K2d. The transcription stage is part of the bake job in `app` (K4, D92, D95).

Superseded (kept for the record; the earlier K2, D71): `voice_examples.transcript_confirmed_at` marked confirmed words; `add_example(…,
transcript=)` and `set_transcript` stored and confirmed typed words; a recording kept the kit text, not confirmed; a take with no words or
unconfirmed words was "unresolved", shown on its row ("needs its words", "check the words: Confirm · Edit"), listed beside the Bake button
and blocked a bake; the take preview showed "What you said"; the Test tab labelled words "(checked)" / "(not checked)".

K3. Storage — `db/schema.sql`, `feed/saved_voices.py` (D72, D79, D80, D81). **Changed 2026-11-02 (rework, t39): rewritten.**

```sql
-- One bake that `app` started (design §6): the snapshot of the takes it sent, written BEFORE the bake starts and never changed (t31).
-- A reattach reads it back; the row goes when the bake has ended and its result is stored or dropped.
CREATE TABLE IF NOT EXISTS voice_bake_runs (
  bake_key      TEXT PRIMARY KEY,      -- secrets.token_hex(16); the bake's tag carries it
  owner_id      INTEGER NOT NULL,
  voice_id      INTEGER NOT NULL REFERENCES saved_voices(id) ON DELETE CASCADE,
  snapshot_json TEXT NOT NULL,         -- [{example_id, content_hash, language, seconds, transcript, kit_id, fingerprint}] in send order
  dataset_hash  TEXT NOT NULL,         -- snapshot_hash of those takes; equals the service's dataset_sha256
  cancel_requested_at TEXT,            -- changed (rework, t40): set by Cancel, replace or a voice delete, under the job lock, before they
                                       -- answer; a run with it set is never stored, also after an app restart (D88)
  created_at    TEXT NOT NULL
);
-- The voice model of one saved voice (design §6, §11.1): the LoRA adapter baked from ALL its takes. Films and tests of the voice speak
-- ONLY through it. One row per voice; a Rebake updates it in place (version + 1). It stays when takes are added, edited or deleted.
-- Its file leaves only through the voice outbox drain.
CREATE TABLE IF NOT EXISTS voice_models (
  voice_id      INTEGER PRIMARY KEY REFERENCES saved_voices(id) ON DELETE CASCADE,
  version       INTEGER NOT NULL,      -- 1, then the stored version + 1 at each Rebake
  bake_key      TEXT NOT NULL UNIQUE,  -- the bake it came from: one bake is stored at most once (t31)
  model_key     TEXT NOT NULL,         -- the catalog key it was trained for ('voxcpm2')
  revision      TEXT NOT NULL,         -- the pinned Hub commit it was trained on
  lora_json     TEXT NOT NULL,         -- AdapterSpec.lora, sorted JSON
  config_json   TEXT NOT NULL,         -- the TrainConfig it ran with
  dataset_hash  TEXT NOT NULL,         -- snapshot_hash of the takes it learned from
  takes         INTEGER NOT NULL,
  seconds       REAL NOT NULL,         -- the length of those takes
  storage_key   TEXT NOT NULL,         -- safetensors in data_dir/voices (content-addressed)
  sha256        TEXT NOT NULL,
  bytes         INTEGER NOT NULL,
  metrics_json  TEXT NOT NULL,         -- the service's bake status: stage seconds, steps, loss, peak memory, as measured
  created_at    TEXT NOT NULL
);
-- What a voice model learned: one row per trained take. No FK to voice_examples: the row outlives a deleted take, so the page can show
-- the snapshot and count "deleted since vN".
CREATE TABLE IF NOT EXISTS voice_model_takes (
  voice_id    INTEGER NOT NULL REFERENCES voice_models(voice_id) ON DELETE CASCADE,
  example_id  INTEGER NOT NULL,
  language    TEXT NOT NULL,          -- the take's language when the bake started: the model's languages come from here
  seconds     REAL NOT NULL,
  fingerprint TEXT NOT NULL,          -- sha256 of "<content_hash>\t<language>\t<transcript>" when the bake started (t31)
  PRIMARY KEY (voice_id, example_id)
);
-- The last test of one saved voice (design §6, §11.1): speech made through its voice model, to judge by ear. One row per voice; a new
-- test replaces it. Never used by a film. Its file leaves only through the voice outbox drain.
CREATE TABLE IF NOT EXISTS voice_tests (
  voice_id        INTEGER PRIMARY KEY REFERENCES saved_voices(id) ON DELETE CASCADE,
  language        TEXT NOT NULL,
  text_kind       TEXT NOT NULL CHECK (text_kind IN ('sentence', 'take', 'custom')),
  text_id         TEXT,               -- the sentence id ('RU-1') when text_kind = 'sentence'; else NULL
  example_id      INTEGER REFERENCES voice_examples(id) ON DELETE SET NULL,  -- the take whose words were spoken; NULL once it is deleted
  text            TEXT NOT NULL,      -- the words the model was asked to say
  model_version   INTEGER NOT NULL,   -- voice_models.version that spoke
  model_sha256    TEXT NOT NULL,
  model_key       TEXT NOT NULL,      -- the reply's model
  revision        TEXT NOT NULL,      -- the reply's revision
  elapsed_seconds REAL NOT NULL,      -- wall time of the /speak call
  seconds         REAL NOT NULL,      -- length of the test audio
  storage_key     TEXT NOT NULL,      -- PCM-16 WAV in data_dir/voices (content-addressed)
  created_at      TEXT NOT NULL
);
```

- All four tables are new, so `migrate` creates them on every database with no version bump (D39); a new table may have a CHECK.
- Dropped from the earlier K3: `voice_previews.model_version` / `model_sha256` (tests move to `voice_tests`); `saved_voices.last_model_version`
  (no model is retired now, so the stored row's version + 1 always rises); the index `voice_model_takes_by_example` (the drain no longer
  looks up models by take).
- Kept as they are (D81): `voice_film_examples`, `voice_previews`, `voice_conditionings`. Their `CREATE` statements stay; each comment adds
  "Films and tests no longer use it (Phase 5, t39); its rows stay until the drain removes them with their take or voice." No `DROP`, no
  row delete, no data migration.
- Store API (`feed/saved_voices.py`):

  ```python
  @dataclass(frozen=True)
  class VoiceModel: voice_id, version, model_key, revision, lora_json, config_json, dataset_hash, takes, seconds, storage_key, sha256, bytes, metrics, created_at
  @dataclass(frozen=True)
  class ModelChanges:
      added: int        # takes that are not pending and not in voice_model_takes
      deleted: int      # voice_model_takes rows whose take is gone or pending
      edited: int       # takes whose current fingerprint differs from the stored one
      unresolved: int   # takes with no words or unconfirmed words (they block a Rebake)
  @dataclass(frozen=True)
  class VoiceModelState:          # the page, the cards, the test
      model: VoiceModel | None
      usable: bool                # True = films and tests can speak through it
      problem: str | None         # None | "none" (never baked) | "revision" (another model revision or LoRA shape)
      languages: dict[str, tuple[int, float]]   # what it learned: language -> (takes, seconds), from voice_model_takes
      changes: ModelChanges
  @dataclass(frozen=True)
  class FilmModel:                # THE answer to "can this voice narrate in this language, and through what" (D77)
      model: VoiceModel | None    # set only when problem is None
      problem: str | None         # None | "none" | "revision" | "language" | "deleting"
  @dataclass(frozen=True)
  class SnapshotTake: example_id: int; content_hash: str; language: str; seconds: float; transcript: str; kit_id: str | None; fingerprint: str
  @dataclass(frozen=True)
  class BakeSnapshot: takes: tuple[VoiceExample, ...]; unresolved: tuple[VoiceExample, ...]; seconds: float; by_language: dict[str, float]; dataset_hash: str
  @dataclass(frozen=True)
  class VoiceTest: voice_id, language, text_kind, text_id, example_id, text, model_version, model_sha256, model_key, revision, elapsed_seconds, seconds, storage_key, created_at

  def set_transcript(conn, owner_id, voice_id, example_id, text) -> VoiceExample
  def snapshot_hash(takes) -> str                    # sha256 of the sorted "<content_hash>\t<language>\t<transcript>" lines, joined by "\n"
  def bake_snapshot(conn, owner_id, voice_id) -> BakeSnapshot     # every non-pending take
  def start_bake_run(conn, owner_id, voice_id, snapshot: BakeSnapshot) -> str
  def read_bake_run(conn, owner_id, bake_key) -> tuple[int, tuple[SnapshotTake, ...], str] | None
  def end_bake_run(conn, bake_key) -> None
  def request_bake_cancel(conn, bake_key) -> bool   # changed (rework, t40): sets cancel_requested_at on an open run; False = no run row
                                                    # (the run already ended: its result was stored or dropped)
  class BakeCancelled(VoiceError): ...              # changed (rework, t40): save_voice_model refuses a run with a cancel request
  def voice_model_for(conn, owner_id, voice_id) -> VoiceModelState
  def film_model(conn, owner_id, voice_id, language) -> FilmModel
  def castable_models(conn, owner_id, language) -> list[tuple[SavedVoice, VoiceModel]]   # new (t40): voices whose film_model has no problem
  # castable(conn, owner_id, language) keeps its take pairs until lane 32d removes it (t40)
  def save_voice_model(conn, storage, owner_id, voice_id, *, bake_key: str, data: bytes, status: dict,
                       takes: tuple[SnapshotTake, ...], dataset_hash: str) -> VoiceModel | None
  def read_voice_model(storage, model: VoiceModel) -> bytes
  def model_reference(model: VoiceModel, data: bytes, language: str) -> dict
      # {"language": language, "adapter": {"blob": data, "sha256": model.sha256}, "use_clip": False} — no audio, no take (D76)
  def save_test(conn, storage, owner_id, voice_id, *, language, text_kind, text_id, example_id, text, model_version,
                model_sha256, model, revision, elapsed_seconds, audio) -> VoiceTest | None
  def voice_test(conn, owner_id, voice_id) -> VoiceTest | None
  def read_test_audio(storage, test: VoiceTest) -> bytes
  # Kept for scripts/voice_bake_check.py only (variants A and B, D82): example_for, conditionings, reference_for.
  # Removed by lane 32d, after 32b and 32c are GREEN (changed, rework, t40): FilmExample, film_example, set_film_example, castable,
  # save_conditioning, VoicePreview, save_preview, previews, read_preview_audio. Lane 31 removes nothing.
  ```

- `voice_model_for`: no row → `(None, False, "none", {}, ModelChanges(added=<takes>, 0, 0, unresolved))`. A row whose `model_key` is no longer a
  catalog voice entry, or whose `revision` or `lora_json` differs from that entry's pin and `adapter.lora` (`catalog.get("voice", model_key)`,
  the same on every profile) → `usable=False, problem="revision"`. Else `usable=True`. **Changed (rework, t39):** a pending or deleted
  training take does not change `usable` (t31 item 6 is superseded). `languages` = `voice_model_takes` grouped by `language`. `changes` as
  `ModelChanges` says; the fingerprints make `edited` exact.
- `film_model(conn, owner_id, voice_id, language)`: the voice gone or pending → `"deleting"`; `voice_model_for` not usable → its problem; `language`
  not in `languages` → `"language"`; else `FilmModel(model, None)`. `castable_models` (changed, rework, t40: a new name, D89) returns
  `(voice, model)` for every non-pending voice whose `film_model` has no problem, sorted as `list_voices`.
- `save_voice_model`: `_write_voice_file` (`feed/saved_voices.py:465-508`) with key `content_key(sha256, ".safetensors")`. A `voice_models` row with
  this `bake_key` exists → return it and store nothing (t31). `status["dataset_sha256"] != dataset_hash` → `VoiceError("The voice model does not
  match the takes that were sent.")`. **Changed (rework, t39; D79):** `parent_ok` = the voice exists and is not pending; a take deleted or
  edited during the bake no longer drops the result (the page then counts it as a change since the new version). `write_row`: `version` =
  the stored row's version + 1, else 1; `INSERT … ON CONFLICT(voice_id) DO UPDATE SET …` (no `REPLACE`, so no foreign-key action fires on the
  takes); `DELETE` then `INSERT` the voice's `voice_model_takes` from the snapshot (example id, language, seconds, fingerprint); `end_bake_run`;
  an old `storage_key` that differs goes to `voice_file_deletions` (`deletion_id` NULL). `None` = the voice went away; nothing is stored.
  **Changed 2026-11-02 (rework, t40; D88):** in the same `BEGIN IMMEDIATE`, after the `bake_key` check, the run row decides: a row with
  `cancel_requested_at` set, or no row at all (the run already ended), → `BakeCancelled`, and nothing is written. Because `write_row` ends the
  run row in the transaction that stores the model, a cancel that comes after the store finds no open run (`request_bake_cancel` → False):
  the store wins and the model stays.
- `request_bake_cancel(conn, bake_key)`: one `BEGIN IMMEDIATE`; `UPDATE voice_bake_runs SET cancel_requested_at = <now> WHERE bake_key = ? AND
  cancel_requested_at IS NULL`; returns whether an open run row exists (changed, rework, t40).
- `save_test`: `_write_voice_file` with key `content_key(sha256, ".wav")`. `parent_ok` = the voice exists and is not pending, AND its
  `voice_models` row still has the `model_version` and `model_sha256` the test used (a defence: a bake and a test never overlap, D74). `write_row`: an upsert by
  `voice_id`; `example_id` is written only when that take row still exists, else NULL (the foreign key would refuse it); an old
  `storage_key` that differs goes to `voice_file_deletions`. `None` → the caller's `PreviewFailed`.
- Drain: `_cascade_one` (`feed/saved_voices.py:369-410`) adds, for a voice delete, the `storage_key` of the voice's `voice_models` and
  `voice_tests` rows (the rows cascade with the voice). For an example delete it adds nothing new: the model and its `voice_model_takes`
  stay, and `voice_tests.example_id` becomes NULL. `_remove_one`'s ref check (`:425-439`) adds `UNION ALL SELECT 1 FROM voice_models WHERE
  storage_key = ?` and the same for `voice_tests`. The `voice_conditionings` and `voice_previews` parts stay (D81). The docstring names voice
  models and tests.
- **Changed 2026-11-02 (ASR, t42) — lane 31c:** in the API above, `set_transcript`, `BakeSnapshot.unresolved`, `ModelChanges.edited` and
  `ModelChanges.unresolved` are replaced by the K2b names (`heard_words`, `takes_to_hear`, `set_heard_words`, `BakeSnapshot.waiting` /
  `.unusable`, `ModelChanges.unusable`). `read_bake_run` returns lane 31's `BakeRun(voice_id, takes, dataset_hash, cancel_requested_at)`
  (D96), in place of the tuple written above. `bake_snapshot`'s `takes` are the heard takes only. The snapshot JSON, `voice_model_takes`
  and `_fingerprint` keep the word `transcript`: it now holds the heard words.

K4. The bake in `app` — `feed/voice_bake.py` (new; thin client), `web/voices_api.py` (D74, D83).

```python
class BakeBlocked(VoiceError): ...    # unresolved takes, or no take: the page lists them
class BakeFailed(VoiceError): ...     # the reason the page shows
class BakeBusy(VoiceError): ...       # the service runs a bake this job did not start; carries its status (changed, rework, D83)

def run_bake(conn, client, voices: Storage, owner_id: int, voice_id: int, *, voice_lock=None, job_lock=None, progress=None, cancelled=None,
             attach: dict | None = None, wait_for_other: bool = False) -> VoiceModel
# job_lock: the VoiceBakeJobs lock — the publication barrier (t31), as in feed/voice_preview.py:185-200
```

- `run_bake`, in this order: `snapshot = bake_snapshot(...)`; no take or any unresolved take → `BakeBlocked` with the list. `with voice_turn(voice_lock,
  progress, cancelled, 0.0):` (`feed/build.py:170`) holds the voice lock for the WHOLE bake, so films and tests wait ("waiting for the voice model")
  and never meet the service's 409. "sending the takes" (0.01): `bake_key = start_bake_run(conn, owner_id, voice_id, snapshot)` (the immutable
  snapshot, t31), read each take, `client.bake_start(clips, tag={"source": "app", "owner_id", "voice_id", "bake_key"})` (**changed, rework:
  `source`, D83**); or, with `attach`, adopt that running bake and read its snapshot back with `read_bake_run` (never rebuilt from the current
  rows; **Changed 2026-11-02 (rework, t40):** the attach path skips the take checks above, see the split below). A 409 from `bake_start` → `BakeBusy(client.bake_current())`, unless `wait_for_other`: then "waiting for the other bake to stop" (0.01),
  `bake_current()` every 1 s until no bake runs, Cancel works, then `bake_start` again. Then every 1 s `bake_status`: progress mirrors the
  service's stage, with `0.02 + 0.95 * fraction`. `cancelled()` → one `bake_cancel`, and the loop goes on until ANY end state (`cancelled`,
  `done` or `failed`); a cancelled run then raises `Cancelled` and stores nothing (t31). `failed` → `BakeFailed(f"The bake failed ({error}).")`;
  404 → `BakeFailed("The models service restarted, so the bake stopped. Bake again.")`. `done` → "storing the voice model" (0.98):
  `bake_adapter` → sha256 check → under `job_lock`: `cancelled()` → `Cancelled`; `save_voice_model(…, bake_key=…)`; `None` →
  `BakeFailed("The voice was deleted during the bake, so the voice model was dropped.")` (**changed, rework:** a deleted or edited take no
  longer drops it, D79). `finally`: `bake_delete` on every end of a bake this run started or adopted (an error is logged) and `end_bake_run`.
  No model call and no fetch run while `job_lock` is held: only the run check, one SQLite write and one file write.
- **Changed 2026-11-02 (rework, t40): the new start and the attach split BEFORE any take check.**
  - New start (`attach` is None): `snapshot = bake_snapshot(...)`; no take or any unresolved take → `BakeBlocked`; then the voice turn,
    `start_bake_run`, the read of each take's audio and `bake_start`, as above.
  - Attach (`attach` given, from `reattach`): no `bake_snapshot`, no unresolved check, no read of any take's words or audio. `read_bake_run(conn,
    owner_id, attach["tag"]["bake_key"])` gives the voice id, the stored snapshot takes and the dataset hash; no row → the run ended or was
    never stored: no store, the bake is cleaned up (`reattach` below). Then the voice turn, the status loop and the publication with the stored
    snapshot only. So an attach still stores the model when every take was deleted and drained meanwhile, or when a new unresolved take was
    added (D79): the model learned from the snapshot, not from the current takes.
  - In both paths the publication goes through `save_voice_model`, which refuses a run with a cancel request (`BakeCancelled` → `Cancelled`,
    nothing stored; D88).
- `VoiceBakeJobs(context, voice_lock, drain)` in `web/voices_api.py`: the `VoicePreviews` pattern (`lock`, `run`, `epoch`, `state`, daemon thread,
  `snapshot()`, `join()`; §2.16 E). One bake per process. `cancel`, `start(replace=yes)` and the voice-delete cancel take the job lock, so a run
  that was cancelled, replaced or deleted stores nothing (t31). A replacement starts only after the old run is torn down: the new thread first
  joins the old one, whose `finally` has deleted its bake.
  **Changed 2026-11-02 (rework, t40; D88):** Cancel, `replace=yes` and the voice-delete cancel each call `request_bake_cancel(conn, bake_key)`
  for the running app bake while they hold the job lock, BEFORE they bump the run and answer. So the request outlives an app restart. When
  it returns False, the publication already stored the model under the same lock: the reply says that the bake ended with v<N> (the store
  wins). A reload between the request and the run's end leaves the row with the flag, and `reattach` cleans it up.
- `reattach()` runs in `register()` and on the first idle `GET /voices/bake` of a process. **Changed (rework, D83):** it acts ONLY on a bake whose
  tag has `source == "app"` and this `owner_id`: with a `voice_bake_runs` row for `tag.bake_key` → running: adopt it on a thread (`attach=`);
  done: store it (at most once, by `bake_key`) and delete it; failed or cancelled: show its reason once and delete it; with no row → cancel
  it if it runs, and delete it. Any other bake (the §6.5 check script's, tag `{"check": NAME}`) is left alone: never adopted, cancelled or
  deleted on its own.
  **Changed 2026-11-02 (rework, t40; D88):** a run row with `cancel_requested_at` set is never adopted for a store: a running bake → one
  `bake_cancel`, then a cleanup thread waits for ANY end state and calls `bake_delete`; an ended bake → `bake_delete`; in both cases
  `end_bake_run` and nothing is stored, also when the service reports `done`. An attach of a run with no flag uses only the stored snapshot
  (above). After a replace, the old run's row carries the flag and the new run has its own row, so a restart stores at most the new one.
- Routes (literal paths before `/voices/{voice_id}`):

  | Method | Path | Does | Reply |
  |---|---|---|---|
  | POST | `/voices/{voice_id}/bake` | start a bake of every take (form `replace=yes` after the question) | `_voice_bake.html`; 409 with the question when another bake runs |
  | GET | `/voices/bake` | the bake job (polled every 1 s while a bake runs) | `_voice_bake.html`, `Cache-Control: no-store` |
  | POST | `/voices/bake/cancel` | cancel the job's bake | `_voice_bake.html` |
  | POST | `/voices/{voice_id}/examples/{example_id}/transcript` | form `text`: confirm or correct the words | `_voice_body.html` |

  The 409 question: another voice of this app is being baked → "«<name>» is being baked (N%). Bake «<this>» instead?"; a bake this app did not
  start → "The models service is baking a voice for the bake check (N%). Stop it and bake «<this>» instead?" (**changed, rework, D83**). The page
  shows the question with a confirm button that posts `replace=yes`. `replace=yes` on an app bake cancels it as before; on a foreign bake it
  calls `bake_cancel` on it (never `bake_delete`: its starter deletes it) and starts the new job with `wait_for_other=True`. Codes: 404 unknown
  or foreign voice; 409 pending, `NOT_AVAILABLE` ("Voice copy is not available on this device."), `BAKE_NOT_OFFERED` ("Baking a voice is not
  offered on this device yet.", t31), `BakeBlocked` (with the list), a bake of the same voice; 400 a `VoiceError`; 507 the disk guard.
- A voice delete request cancels a running bake of that voice (logical cancel + `bake_cancel`). **Changed 2026-11-02 (rework, t40):** it first writes
  the cancel request under the job lock (D88).
- Removed routes (**changed, rework, t39**): `POST /voices/{voice_id}/film-example`, `POST /voices/{voice_id}/examples/{example_id}/preview`,
  `GET /voices/{voice_id}/examples/{example_id}/preview/audio`. They answer 404 or 405, as every unknown path does.
- **Changed 2026-11-02 (ASR, t42; D92, D94, D95) — lane 32a (the job), lane 32c (the routes): the new start gains a transcription stage.**
  The attach path does not change: it reads only the stored snapshot, which holds the heard words, and hears nothing.
  New start, in this order (it replaces the new-start steps above):
  1. Before the voice lock: `snapshot = bake_snapshot(...)`. No take at all → `BakeBlocked("No take yet. Record takes on Recordings.")`;
     takes, but none `heard` or `waiting` → `BakeBlocked(f"No take can be learned from: {not_used_text(snapshot.unusable)}. Record or
     upload a take where you speak.")` (**changed, ASR plan check, t44:** each take with its own reason; before: "no speech was heard in
     any take"). Unusable takes never block. The route answers both with 409 before a job starts.
  2. `with voice_turn(voice_lock, progress, cancelled, 0.0):` for the whole job, as before (films and tests wait).
  3. A bake in the service (`bake_current()` with `state == "running"`): this app's → the route already asked (K4 routes); another one →
     `BakeBusy(current)`, unless `wait_for_other`: "waiting for the other bake to stop" (0.01) until it ends, Cancel works. This check
     moves BEFORE the stage below, so no take is heard while a foreign bake holds the gate (a transcribe would answer 409); the 409 of
     `bake_start` still maps to `BakeBusy` as before.
  4. "transcribing the takes · take n of m" (`0.01 + 0.14 * n / m`): `takes = takes_to_hear(...)`, read again under the lock; no take →
     the stage is skipped. For each take, by id: `cancelled()` → `Cancelled`; `audio = read_example_audio(voices, take)`; `reply =
     client.transcribe(audio, take.language)`; `set_heard_words(conn, owner_id, voice_id, take.id, text=reply["text"],
     model=f"{reply['model']}@{reply['revision']}")` (`None`: the take went away or was heard meanwhile; go on). A 409 →
     `BakeBusy(client.bake_current())`; a 503 → `BakeFailed("Hearing the words of a take is not offered on this device yet.")`; any other
     error → `BakeFailed(f"Could not hear the words of take #{take.id} ({reason}).")`. Words stored before the error stay; the next Bake
     hears only the rest. The call in flight is never interrupted: Cancel is checked between takes.
     **Changed 2026-11-02 (ASR plan check, t44): a take deleted during the stage.** Before each `read_example_audio`, the job reads the
     row again with `take_to_hear(...)`; `None` (pending, gone, or heard meanwhile) → skip it, no call. When `read_example_audio` raises,
     the job reads the row again: `None` → skip it (it was deleted and drained); a take that is still there → `BakeFailed(f"The audio of
     take #{take.id} could not be read ({reason}).")`. A take deleted while its call runs is caught by `set_heard_words`, which writes
     nothing for a pending or gone take; the job goes on. The guard of `set_heard_words` stays.
  5. (Superseded by the t44 paragraph at the end of this step.) Under `job_lock` (only SQLite writes under it, t40): `cancelled()` →
     `Cancelled`; `snapshot = bake_snapshot(...)`; no `heard` take → `BakeBlocked("No speech was heard in any take, so the voice model has nothing to learn from. Record or upload a take where you
     speak.")`; `bake_key = start_bake_run(conn, owner_id, voice_id, snapshot)`, and the job records `bake_key` before it leaves the lock.
     So a Cancel, a `replace=yes` or a voice delete either comes first (no run row: nothing starts) or finds the `bake_key` and writes
     `cancel_requested_at` (D88). A take added after this point is not in the snapshot and counts as "added since v<N>".
     **Changed 2026-11-02 (ASR plan check, t44): the snapshot gap.** `add_example` takes no lock (`web/voices_api.py:514-548`), so a take
     can arrive while step 4 runs. Under `job_lock`: `cancelled()` → `Cancelled`; `snapshot = bake_snapshot(...)`; when `snapshot.waiting`
     is not empty, the job releases `job_lock` and goes back to step 4 for those takes, then takes `job_lock` again and reads the snapshot
     again. `job_lock` is never held across an HTTP call. Only when the read under `job_lock` finds no `waiting` take does the job go on:
     no `heard` take → `BakeBlocked(f"No take can be learned from: {not_used_text(snapshot.unusable)}. Record or upload a take where you
     speak.")` (before: "No speech was heard in any take, …"); else `start_bake_run(...)` in that same hold of the lock. That final read is
     the cutoff: a take added after it is not in the snapshot, is not heard in this job, and counts as "added since v<N>". The bar keeps its
     highest value when a round adds takes to m.
  6. "sending the takes" (0.15): read the audio of each snapshot take; `bake_start(clips, tag=…)` with each clip's `text` = its stored heard
     words, exactly (the service strips them and hashes them, so `dataset_sha256` equals `snapshot.dataset_hash`). Then the status loop as
     above, with progress `0.16 + 0.81 * fraction`; "storing the voice model" stays 0.98.
  - The job's done text names what was left out: "Voice model v<N> is stored. Not used: take #9, #12 — no speech was heard in them."
    (only when `unusable` is not empty). **Changed 2026-11-02 (ASR plan check, t44):** the text is `f"Voice model v{N} is stored. Not
    used: {not_used_text(snapshot.unusable)}."`, each take with its own reason (`NO_WORDS` or `TOO_LONG`).
  - Restart during the stage: no run row and no service bake exist yet, so `reattach` finds nothing; the words already stored stay.
  - A voice delete during the stage: the job's logical cancel (no `bake_key` yet, so no row to flag); `set_heard_words` writes nothing for
    a pending voice, and `start_bake_run` refuses it (`VoiceDeleting`).
  - Kept as written (t44): a take is heard once; Cancel acts between calls; nothing retries a take; no language detection — the take's
    language label is the decoding language (D77, kept per t43).
  - The route `POST /voices/{voice_id}/examples/{example_id}/transcript` of the routes table above is never built: it answers 404 or 405.
    The add route (`POST /voices/{voice_id}/examples`) reads no `transcript` field.
  - `BakeBlocked` carries a text, never a list of takes.

K5. Films and tests (D76-D78, D80, D86). **Changed 2026-11-02 (rework, t39): rewritten. The earlier K5 (the model beside the film take, one
take without a model, the training-take checks at publication) is superseded.**

Films (`feed/build.py`):

- `_saved_cast` (`:584-617`): the chosen voice → `film_model(conn, owner_id, voice.id, language)`; a problem → `_fail` with the film text below,
  before the story pass and any model call. `castable_models` (voices with a model of the language; changed, t40: a new name, D89) feeds
  `saved_for_prompt` in place of `castable`, so the story pass offers bracket voices that can narrate only.
- The film's one `voice_turn` (`:411-447`) first resolves each used voice AGAIN with `film_model` (a Rebake that ran while the film waited for
  the voice lock has stored a new version, and the drain may already have removed the old file), then reads
  `model_reference(model, read_voice_model(voices, model), language)`. A problem there → `_fail` with its text; a read error →
  `_fail(f"The voice model of «{name}» could not be read ({reason}).")`. While the film holds the lock, no bake can store a version (a bake
  holds the same lock for its whole run, D74). One `client.speak(…, references=…)` call, as today, inside `_Ticking`. No conditioning store:
  `_keep_conditioning` (`:620-633`) goes.
- The reply check (D86): for every used token, `reply["adapters"][token]` must exist with `state` `loaded` or `kept` and the sent sha256; else
  `_fail(f"The models service did not speak «{name}» through its voice model.")`.
- `decisions["voice"]` (`:455-465`): `conditioning` goes; `models = {token: {"version", "sha256", "state"}}` is added. The cast gets
  `narrator_model = f"voice model v{version}"`.
- The ready save (`:557-581`): one `BEGIN IMMEDIATE` checks that every used VOICE still exists and is not pending (`_deleted_voice` without
  its take part), then the WAVs and the row, as today. A take deleted while the film is made does not fail it, and neither does a Rebake
  stored meanwhile: the film used the version it read and records it.
- The texts, one set for the film failure, the Prepare 409 and the picker:

  | problem | film failure and Prepare 409 | picker option (disabled) |
  |---|---|---|
  | `none` | "«<name>» has no voice model yet. Bake it under Voices, or choose another narrator." | "«<name>» — bake required" |
  | `revision` | "The voice model of «<name>» was made for another model revision. Rebake it under Voices, or choose another narrator." | "«<name>» — Rebake required (another model revision)" |
  | `language` | "The voice model of «<name>» learned no <Language>. Record <Language> takes and Rebake under Voices, or choose another narrator." | "«<name>» — its voice model learned no <Language>" |
  | `deleting` | "The saved voice «<name>» is being deleted. Choose another narrator." (as today) | "«<name>» — being deleted" (as today) |

- "This film" (`feed/store.py:108`, `film_narrator` → a new `FilmNarrator.model`): a saved-voice film with `cast.narrator_model` adds
  "· voice model v<N>" ("narrated by me (saved voice) — your choice · voice model v3"); a saved-voice film with no `models` in its
  decisions (made from one take before Slice B) adds "· made from one take, before voice models"; a preset film adds nothing.

The `/feed` picker (`web/app.py:944-1038`, `web/templates/feed.html:22-56`; D78):

- `picker["saved"]` = the voices of `castable_models` (changed, t40: in place of `castable`). New `picker["saved_blocked"]` = every other non-pending saved voice, as a disabled option
  with the picker label of its problem (the table above). A stored choice that cannot narrate stays the selected "unavailable" option with the
  same label (`:982-992` take labels → the model labels).
- Under the picker, when the selected saved voice has the problem `none`, `revision` or `language`: `<p class="feed-note">«<name>» needs a voice
  model for <Language> films: <a href="/voices/<id>?tab=model">open «<name>» in Voices</a>.</p>` (a plain link, the grid→leaf navigation of
  §13.1 from another section; the leaf's close then goes to `/voices`).
- `_store_narrator` (`:1029-1038`): the take texts → the film texts of the table, 409.

Tests — one Test per person (`feed/voice_preview.py`, `web/voices_api.py`; D80, D85):

```python
TEXT_KINDS = ("sentence", "take", "custom")
@dataclass(frozen=True)
class PreviewText:
    kind: str                 # "sentence" | "take" | "custom"
    text_id: str | None       # the sentence id when kind == "sentence"
    example_id: int | None    # the take when kind == "take"
    text: str
@dataclass(frozen=True)
class PreviewJob:
    voice_id: int
    language: str
    text: PreviewText
    revision: str             # cloning_entry_for(profile, language).source.revision at the start
def preview_text(voice: SavedVoice, language: str, kind: str, text_id: str, custom: str) -> PreviewText
def run_preview(conn, client, voices, owner_id, job, *, voice_lock=None, job_lock=None, on_saved=None, progress=None, cancelled=None) -> VoiceTest
```

- `preview_text`: `sentence` → `PREVIEW_SENTENCES[language][text_id]`, else "Unknown test sentence for <Language>: '<id>'."; `take` → `text_id`
  is the id of a non-pending take of this voice in `language` (else "Unknown take.") that has words (else "Take #<id> has no words. Add them on
  Recordings."); its text is the take's transcript; `custom` → 1..`CUSTOM_TEXT_MAX` (300) after strip ("Type a text to test." / "A test text
  can have at most 300 characters."); any other kind → "Unknown kind of test text.". The old `reading` kind goes.
- `run_preview`, under `voice_turn(voice_lock, …)`: "reading the voice model" (0.05): `film_model(…, job.language)`; a problem →
  `PreviewFailed` with the test text ("Bake the voice model first." / "Rebake the voice model first: it was made for another model revision." /
  "The voice model learned no <Language>." / "This voice is being deleted."); it uses the version it finds here (a Rebake that ran while the
  test waited for the voice lock gives the new version, and no bake can store one while the test holds the lock, D74); read the file →
  `model_reference`. "generating the voice"
  (0.10-0.97, `_Ticking`); the cancel check right before the call (t26); one `client.speak(job.text.text, job.language, token, None, None,
  references={token: reference})`; the reply check of D86. No conditioning store. "saving the test" (0.97) under `job_lock`: the cancel <!-- pragma: allowlist secret -->
  check; the revision check (as today, `feed/voice_preview.py:188-189`); `save_test(…)`; `None` → `PreviewFailed("The voice model changed while
  the test ran, or the voice is being deleted. Generate it again.")`; `on_saved`.
- Cancel, replace, `epoch`, `completed` and the reconnect stay as §2.16 D-E and D51-D53; only the stored row changes.
- `POST /voices/{voice_id}/preview` (form `language`, `text_kind`, `text_id`, `text`) → `_voice_preview.html`. Codes: 404 "Unknown saved voice.";
  409 "This voice is being deleted.", `NOT_AVAILABLE` when no cloning entry speaks `language`, and the three model texts above; 400 the
  `preview_text` refusals; 507 the disk guard. `GET /voices/{voice_id}/preview/audio` → the stored test WAV (`audio/wav`), else 404 "This voice
  has no test yet.". `VoicePreviews.snapshot()` carries `voice_id` and `language` in place of `example_id`. The route refuses at once, with the
  same 409 texts, a voice whose `film_model` has a problem when the request comes, so Generate never starts a job that must fail.
- **Changed 2026-11-02 (ASR, t42; D101) — lane 32b:** `preview_text`'s `take` kind takes only a `heard` take: a non-pending take of this
  voice in `language` (else "Unknown take."); `waiting` → "The words of take #<id> are heard at the next bake."; `unusable` → "Take
  #<id> has no words to speak: <transcript_error>." (**changed, ASR plan check, t44:** its own reason, `NO_WORDS` or `TOO_LONG`; before:
  "No speech was heard in take #<id>."); its text is the heard words. A take stored before t42 is `waiting`, so its shown text is never
  spoken as its words. The Test tab's option reads "Your words in take #<id>" and is listed only for `heard` takes (K6).

K6. The page in Slice B. **Changed 2026-11-02 (rework, t39): rewritten.** The figures, takes and dates in these wireframes are EXAMPLES, not
your stored voice (t31).

Header — every tab (`_voice_body.html`):

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│ Upload   Library   Chat   Organize   [Voices]                     RAM 18/32 GB  ⚙ │
├───────────────────────────────────────────────────────────────────────────────────┤
│ Voices › me                                                           [⋯]   [×]  │
│ Voice model v3 · ready · films and tests of «me» speak through it                 │
│ 2 takes added, 1 deleted since v3 — Rebake to include them                        │
│                                                                                   │
│ [ Recordings ]   Voice model    Test                                              │
└───────────────────────────────────────────────────────────────────────────────────┘
  other states of the first line:
│ No voice model yet. «me» cannot narrate a film until you bake one.  Bake…         │
│ Baking v4 · training · step 240 of 600 · 40%  [████████░░░░░░░░░░]  [Cancel]      │
│ Voice model v3 was made for another model revision. Rebake it: «me» cannot        │
│ narrate until then.                                                               │
```

Recordings tab (kit rows of K1 and D68; no Test button):

```
│ [ Recordings ]   Voice model    Test                                              │
│ ───────────────────────────────────────────────────────────────────────────────── │
│ Russian · 14 of 26 kit texts · 2:51 of words checked                              │
│  Normal — Read in your normal voice.                                              │
│   ru-kitchen “Утренний свет медленно…”  #3 [▶ 0:13] words checked  Record again  Delete │
│   ru-market  “По воскресеньям рынок…”   #4 [▶ 0:12] check the words: Confirm · Edit  Delete │
│  Slow — Read slowly and clearly, with short pauses.                               │
│   ru-slow-1  “Каждое утро пекарь…”      [Record]                                  │
│   ┌ recorder — opens at this row ───────────────────────────────────────────┐    │
│   │ Slow — Read slowly and clearly, with short pauses.                      │    │
│   │ “Каждое утро пекарь открывает лавку в шесть часов…”   0:07  [Stop]      │    │
│   └─────────────────────────────────────────────────────────────────────────┘    │
│  Your uploads                                                                     │
│   #9 [▶ 0:22] uploaded · needs its words [________________________] [Save words]  │
│ Add a take   Language [Russian ▾]   [x] This is my voice, or I have this person's │
│   permission to copy it.   [Record]  [Upload a file]                              │
│   Record clean, normal speech in a quiet room. Films and tests of «me» speak      │
│   through its voice model, baked from all its takes. A new take is used after     │
│   the next Rebake.                                                                │
```

Voice model tab:

```
│ Recordings   [ Voice model ]   Test                                               │
│ ───────────────────────────────────────────────────────────────────────────────── │
│ Voice model v3 · made 2026-11-05 · used by every film and test of «me»            │
│ Learned from 31 takes · 6:12 · Russian 24 takes 4:50 · English 7 takes 1:22       │
│ Since v3: 2 takes added · 1 deleted · words of 1 take edited                      │
│ [Rebake]  trains v4 from all 32 current takes; v3 stays in use until v4 is stored │
│                                                                                   │
│ Your checked words: Russian 4:50 · English 1:22. The package authors suggest      │
│ 5-10 minutes of audio for one speaker. Guidance only.                             │
└───────────────────────────────────────────────────────────────────────────────────┘
  other states of the same block:
│ No voice model yet. Films and tests of «me» speak only through a voice model      │
│ baked from all its takes.   [Bake voice model]  from 12 takes · 2:31 · Russian    │
│ Cannot bake yet: 2 takes need their words (#9, #4).   [Rebake] (disabled)         │
│ Baking v4 · training · step 240 of 600 · 40%  [████████░░░░░░░░░░]  [Cancel]      │
│ «Grandma» is being baked (40%). Bake «me» instead?   [Stop it and bake «me»]      │
│ The models service is baking a voice for the bake check (40%).                    │
│ Stop it and bake «me» instead?   [Stop it and bake «me»]                          │
│ The bake failed (<reason>).   [Rebake]                                            │
```

Test tab:

```
│ Recordings   Voice model   [ Test ]                                               │
│ ───────────────────────────────────────────────────────────────────────────────── │
│ Language  [Russian ▾]        the languages voice model v3 learned                 │
│ Text      [RU-1 — В субботу рано утром мы поехали к озеру… ▾]                     │
│             RU-1 … RU-5 · Words of take #5 (checked) · Words of take #4           │
│             (not checked) · … · Your own text                                     │
│           “В субботу рано утром мы поехали к озеру, пока туман ещё не…”           │
│           [textarea — only for "Your own text", at most 300 characters]           │
│ [Generate]  through voice model v3 only                                           │
│ Testing «me» (Russian) · generating the voice · 42%  [████░░░░░░]  [Cancel]       │
│ ───────────────────────────────────────────────────────────────────────────────── │
│ Generated  Russian · “Весь день по крыше стучал дождь…” · voice model v3 ·        │
│            voxcpm2 @ 32279ef · made in 18.2 s · 2026-11-05                  [▶]   │
│ Original   take #5: “Весь день по крыше стучал дождь…”                      [▶]   │
│            (only when the test spoke the words of a take that still exists)      │
└───────────────────────────────────────────────────────────────────────────────────┘
  with no usable model:
│ Bake the voice model first. Tests speak only through it.   [Bake voice model]     │
│ [Generate] (disabled)                                                             │
  after a Rebake:
│ Generated  … · made with voice model v3; «me» now has v4 — generate it again      │
```

Cards on `/voices`:

```
│ ┌───────────────────────────────┐  ┌───────────────────────────────┐              │
│ │ me                            │  │ Grandma                       │              │
│ │ Russian · 4:50 · 24 takes     │  │ English · 1:12 · 6 takes      │              │
│ │ English · 1:22 · 7 takes      │  │ No voice model yet            │              │
│ │ Voice model v3 · 3 changes    │  └───────────────────────────────┘              │
│ └───────────────────────────────┘  ┌───────────────────────────────┐              │
│ ┌───────────────────────────────┐  │ Mom · deleting…               │              │
│ │ Dad                           │  └───────────────────────────────┘              │
│ │ English · 0:48 · 4 takes      │                                                 │
│ │ Baking · 40%                  │                                                 │
│ └───────────────────────────────┘                                                 │
```

- Header, first line (every tab, from `voice_model_for` and the bake job): no model → "No voice model yet. «<name>» cannot narrate a film until
  you bake one." with "Bake…" (switches to the Voice model tab, no history entry); usable → "Voice model v<N> · ready · films and tests of
  «<name>» speak through it"; `revision` → "Voice model v<N> was made for another model revision. Rebake it: «<name>» cannot narrate until
  then."; this voice baking → "Baking v<N+1> · <stage> · N% · Cancel". Second line, only with a model and a change: "<counts> since v<N> —
  Rebake to include them", where the counts read "2 takes added", "1 deleted", "words of 1 take edited", joined by ", ". The ⋯ menu and ×
  stay as in V3.
- Recordings: the kit rows of K1 with their takes; per take "#id ▶ m:ss · recorded|uploaded · date · words state · Record again (kit rows) ·
  Delete". No Test button. Delete asks nothing: the voice model stays (D87). The tip: "Record clean, normal speech in a quiet room. Films and
  tests of «<name>» speak through its voice model, baked from all its takes. A new take is used after the next Rebake."
- Voice model: "Voice model v<N> · made <date> · used by every film and test of «<name>»"; "Learned from <T> takes · m:ss · <Language> <n>
  takes m:ss · …" (from `languages`); "Since v<N>: …" with each count, or "No changes since v<N>."; the button "Bake voice model" before the
  first model, "Rebake" after (D87), with "trains v<N+1> from all <T> current takes; v<N> stays in use until v<N+1> is stored"; disabled with
  "Cannot bake yet: <n> takes need their words (#…)." or "No take yet. Record takes on Recordings."; the job line with stage, percent, bar and
  Cancel; the question of K4 with its confirm button; a failed or cancelled end shows its reason once. The meter: "Your checked words:
  <Language> m:ss · …. The package authors suggest 5-10 minutes of audio for one speaker. Guidance only." from `adapter.guidance_minutes`; it
  never says "enough". On a profile with no voice copy: "Voice copy is not available on this device."; with voice copy and no bake: "Baking a
  voice is not offered on this device yet."
- Test: "Language" = the model's languages that voice copy speaks here; "Text" = per language the sentences "<id> — <sentence>", then "Words
  of take #<id> (checked)" / "(not checked)" for each non-pending take with words (newest first), then "Your own text"; the words of the
  chosen sentence or take show under the select; the textarea shows only for "Your own text". Generate, then the job line "Testing «<name>»
  (<Language>): “<text>”", the bar, "<stage> · N%" and Cancel; a test of another voice reads "Testing «<name>» · N% ·
  Cancel". The comparison shows the stored test: "Generated — <Language> · “<text>” · voice model v<N> · <model> @ <first 7 of revision> ·
  made in <s> s · <date>" ▶, plus "· made with voice model v<N>; «<name>» now has v<M> — generate it again" when the model changed; and,
  when the test spoke a take's words, "Original — take #<id>: “<words>”" ▶, or "Original — the take was deleted". With no usable model the tab
  shows "Bake the voice model first. Tests speak only through it." with the Bake button, and Generate is disabled. The failure texts are the
  ones of K5, plus "The models service could not make the test (<reason>)." and "The test failed (<reason>)." (as today).
- Cards: per language with takes "<Language> · m:ss · N takes"; then one model line: "No voice model yet" | "Voice model v<N>" | "Voice model
  v<N> · <n> changes" (added + deleted + edited) | "Baking · N%" | "Rebake required". The grid polls `/voices/list` every 2 s while a card
  shows "Baking" or "deleting…".

Page code:

- `_voice_body.html` (`#voices`): the header from `voice_model_for` and the job; the Recordings rows without the Test button (`:78-81`) and
  without the film-take confirm (`:84`); the new tip (`:107`); the Voice model tab = the bake block (`:118-149` go); the Test tab = a
  `.test-language` select, a `.test-text` select and one `<template class="test-compare">` with the stored test (`:152-188` go). It carries
  `data-model-ready="1"` when the voice can be tested, plus `data-epoch` and `data-completed` as today (t27). The film lines `:35-42` go.
- `voice.html`: the panel keeps `.preview-reading` (the chosen words), `.preview-custom`, Generate, the job include and `.preview-compare`; the text
  select leaves the panel for `#voices` as `.test-text` (`:47-55` go), because its take options change with the takes and `carryInputs` keeps its
  value across a swap. The recorder stays as it is.
- `_voice_preview.html`: `data-language` in place of `data-example-id`; the job and done texts above.
  `_voice_bake.html` (new): the bake job line with `data-state`, `data-voice-id` and the question; polled every 1 s while a bake runs.
  `_voice_cards.html`: the model line in place of `:18-31`'s film lines.
- `web/static/voices.js`: `carryInputs` keeps `['.voice-language', '.test-language', '.test-text']` (`:430-431`); `testTarget` returns `{voiceId,
  language, kind, textId, words, ready}` from `.test-language`, `.test-text` and `#voices[data-model-ready]` (`:790-799`); `syncTest` shows only the
  options of the chosen language, the chosen words, the textarea for "custom", and disables Generate when the voice is not ready
  (`:811-838`); `selectTake` → `selectTest(language)`, which `mountLeaf` calls for a running or failed test of this voice (`:995-998`);
  `generatePreview` posts `/voices/{id}/preview` with `language`, `text_kind`, `text_id`, `text` (`:952-963`); the `.example-preview` click
  (`:1070-1071`) and the `.film-take-select` change (`:1079`) go. New: the kit rows, the words box (K2) and the bake poller of
  `/voices/bake`. `previewSync`, the t27 certificates and the t36 ownership rules stay, and every new async site follows t36 (capture `own`
  before the first await; check `own.alive()` before each DOM effect, retry or recovery read).

**Changed 2026-11-02 (ASR, t42; D101) — lane 32c.** The page shows no words to type, check or confirm. These wireframes replace the
Recordings and Voice model blocks above; the header, the Test tab and the cards keep their layout with the text changes listed after
them. The figures are EXAMPLES. **Changed 2026-11-02 (ASR plan check, t44):** every text that names a left-out take gives that take's
own reason (`not_used_text`: `NO_WORDS` "the speech recognition found no words in it" or `TOO_LONG` "it holds more than 1000
characters of words"); no text says "no speech".

Recordings tab:

```
│ [ Recordings ]   Voice model    Test                                              │
│ ───────────────────────────────────────────────────────────────────────────────── │
│ Russian · 14 of 26 kit texts · 2:51 recorded                                      │
│  Normal — Read in your normal voice.                                              │
│   ru-kitchen “Утренний свет медленно…”  #3 [▶ 0:13]  Record again  Delete         │
│  Slow — Read slowly and clearly, with short pauses.                               │
│   ru-slow-1  “Каждое утро пекарь…”      [Record]                                  │
│   ┌ recorder — opens at this row ───────────────────────────────────────────┐    │
│   │ Slow — Read slowly and clearly, with short pauses.                      │    │
│   │ “Каждое утро пекарь открывает лавку в шесть часов…”   0:07  [Stop]      │    │
│   │ Read it, or say anything else: the bake hears what you say.             │    │
│   └─────────────────────────────────────────────────────────────────────────┘    │
│  Other takes                                                                      │
│   #9  [▶ 0:22] uploaded   Delete                                                  │
│   #12 [▶ 0:18] uploaded · not used by the voice model: the speech recognition     │
│       found no words in it   Delete                                               │
│ Add a take   Language [Russian ▾]   [x] This is my voice, or I have this person's │
│   permission to copy it.   [Record]  [Upload a file]                              │
```

Voice model tab:

```
│ Recordings   [ Voice model ]   Test                                               │
│ ───────────────────────────────────────────────────────────────────────────────── │
│ Voice model v3 · made 2026-11-05 · used by every film and test of «me»            │
│ Learned from 31 takes · 6:12 · Russian 24 takes 4:50 · English 7 takes 1:22       │
│ Since v3: 2 takes added · 1 deleted                                               │
│ Not used: take #12 — the speech recognition found no words in it; take #4 — it   │
│ holds more than 1000 characters of words. Delete them, or keep them: they never   │
│ block a bake.                                                                     │
│ [Rebake]  trains v4 from all 32 current takes; v3 stays in use until v4 is stored │
│                                                                                   │
│ Your recordings: Russian 4:50 · English 1:22. The package authors suggest 5-10    │
│ minutes of audio for one speaker. Guidance only.                                  │
└───────────────────────────────────────────────────────────────────────────────────┘
  other states of the same block:
│ No voice model yet. Films and tests of «me» speak only through a voice model      │
│ baked from all its takes.   [Bake voice model]  from 12 takes · 2:31 · Russian    │
│ Baking v4 · transcribing the takes · take 3 of 5 · 9%  [██░░░░░░░░░]  [Cancel]    │
│ Baking v4 · training · step 240 of 600 · 40%  [████████░░░░░░░░░░]  [Cancel]      │
│ No take yet. Record takes on Recordings.   [Bake voice model] (disabled)          │
│ No take can be learned from: take #12 — the speech recognition found no words in  │
│ it. Record or upload a take where you speak.   [Bake voice model] (disabled)      │
│ Voice model v4 is stored. Not used: take #12 — the speech recognition found no    │
│ words in it.                                                                      │
```

- Header (every tab): the second line's counts are "N takes added" and "N deleted" only; "words of N takes edited" goes. Unusable takes
  are not a change: they show on their rows and on the Voice model tab.
- Recordings: the language line reads "<Language> · N of 26 kit texts · m:ss recorded". A take row is "#id ▶ m:ss · recorded|uploaded ·
  date · Record again (kit rows) · Delete", plus "· not used by the voice model: <transcript_error>" for an `unusable` take; a `waiting`
  or `heard` take shows no words text. "Your uploads" → "Other takes" (uploads and recordings with no `kit_id`). The recorder adds the
  line "Read it, or say anything else: the bake hears what you say." under the text. Both sources end in a preview with **Save** and
  **Discard** only: no "What you said" box; Save posts `language`, `source`, `consent` and `prompt_id`. The rest of Add a take (the
  consent box, "Different text", the kit row's language) does not change.
- Voice model: "Since v<N>: <n> takes added · <n> deleted" (each count that is not zero), or "No changes since v<N>."; then, when a take
  is `unusable`, f"Not used: {not_used_text(...)}." (each take with its own reason; changed, ASR plan check, t44) and "Delete it, or
  keep it: it never blocks a bake."; "Bake voice model" / "Rebake" with "from <T> takes" or "trains v<N+1> from all <T> current takes", T =
  the `heard` + `waiting` takes; disabled only with "No take yet. Record takes on Recordings." or f"No take can be learned from:
  {not_used_text(...)}. Record or upload a take where you speak." (changed, ASR plan check, t44); the job's stages are "waiting for the voice model", "waiting for the
  other bake to stop", "transcribing the takes · take n of m", "sending the takes", the service's stages and "storing the voice model";
  the done line names the takes left out (K4). The meter: "Your recordings: <Language> m:ss · …" over the `heard` + `waiting` takes,
  then the authors' sentence, unchanged. "Your checked words" goes.
- Test: the take options read "Your words in take #<id>" for `heard` takes only (newest first); the words under the select are the
  heard words. "(checked)" / "(not checked)" go.
- Page code: `_voice_body.html` renders the row texts and the Voice model lines above from `words_state`, `transcript_error` and
  `voice_model_for(...).changes`; `web/static/voices.js` has no words box and posts no `transcript`; the bake poller shows the
  transcription stage like any other stage; `web/voices_api.py` registers no transcript route.

K7. Error paths (Slice B). **Changed 2026-11-02 (rework, t39): rewritten.**

| Case | Where | Result |
|---|---|---|
| ~~unresolved takes, or no take~~ changed 2026-11-02 (ASR, t42): no take, or no take that is `heard` or `waiting` | Bake | 409 "No take yet. Record takes on Recordings." or "No take can be learned from: <each take with its own reason>. …" (changed, ASR plan check, t44); the button shows the same text (before: the list of unresolved takes) |
| no voice copy on this profile | Bake | 409 `NOT_AVAILABLE` "Voice copy is not available on this device." |
| voice copy, but no bake offered on this profile (cloud for now) | Bake | 409 `BAKE_NOT_OFFERED` "Baking a voice is not offered on this device yet." (t31) |
| another voice of this app is being baked | Bake | 409 with the question; `replace=yes` cancels it and starts this one |
| the models service runs a bake this app did not start (the §6.5 check) | Bake | 409 with the question; `replace=yes` calls `bake_cancel` on it, never `bake_delete`, waits until it ends, then starts (D83) |
| the app restarts while the §6.5 check bakes | `reattach` | the foreign bake is left alone (D83) |
| a film or a test during an app bake | `app` | waits ("waiting for the voice model"), Cancel works |
| a film or a test during a foreign bake | `/speak` | fails with the service's 409 text (§2.17 H) |
| the voice is deleted during its bake | job | the bake is cancelled; nothing is stored |
| a take is deleted or its words change during the bake | job | the model is stored; the page counts the change since the new version (D79) |
| Cancel, `replace=yes` or a voice delete while the result is being stored | job | the job lock orders them: the store wins and the model stays, or the cancel wins and nothing is stored (t31) |
| a done or failed end after Cancel | job | ends the wait like `cancelled`; nothing is stored; the bake is deleted (t31) |
| a voice with no voice model | film / Prepare / picker / test | the `none` texts of K5; no speak call; never a take, never Automatic or a preset |
| a model of another revision or LoRA shape | film / Prepare / picker / test | the `revision` texts of K5; the page says "Rebake" |
| a model that learned no takes in the film's language | film / Prepare / picker / test | the `language` texts of K5 |
| a take of the model is deleted, before, during or after a film or a test | film / test | nothing fails; the model stays and is used; the page counts "1 deleted since v<N>" |
| takes added or words edited after the bake | page / films | "… since v<N> — Rebake to include them"; films and tests keep the model |
| a Rebake ends while a test waits for the voice model | test | the test uses the new version and records it (a bake and a test never overlap: both hold the voice lock, D74) |
| a Rebake is stored while a film is made | film | the film is made with the version it read and records it |
| the reply lacks the model, or reports another sha256 | film / test | failed: "The models service did not speak «<name>» through its voice model." (D86) |
| more than 4 saved voices in one film | `/speak` | 422 (`MAX_ADAPTERS`); the film fails with that reason |
| the voice model file cannot be read | film / test | failed: "The voice model of «<name>» could not be read (<reason>)." |
| the app restarts during a bake | `app` | the job reattaches to the running app bake, reads its stored snapshot back and stores it when it ends |
| Changed 2026-11-02 (rework, t40): every take of the bake is deleted and drained, or a new unresolved take (changed, ASR, t42: a new take) is added, before the reattach | `reattach` | the attach reads only the stored snapshot: no `BakeBlocked`; the model is stored; the page counts the changes since it (D79) |
| Changed 2026-11-02 (rework, t40): the app restarts after a Cancel, before the run ended | `reattach` | the run row carries `cancel_requested_at`: the bake is cancelled if it runs, deleted when it ends, and nothing is stored, also on `done` (D88) |
| Changed 2026-11-02 (rework, t40): the app restarts after `replace=yes` | `reattach` | the old run's row carries the flag and is cleaned up with no store; the new run is adopted and stored when it ends |
| Changed 2026-11-02 (rework, t40): Cancel, replace or a voice delete after the model was stored | job | `request_bake_cancel` finds no open run: the store wins, the model stays, and the reply says the bake ended with v<N> |
| Changed 2026-11-02 (rework, t40): a publication meets a run with a cancel request | store | `BakeCancelled`; nothing is written; the run ends as `cancelled` |
| the app restarts after the store and before `bake_delete` | `app` | the reattach finds the `bake_key` stored: no second version; it deletes the bake (t31) |
| the models service restarts | job | `failed` "The models service restarted, so the bake stopped. Bake again." |
| §6.5 fails on this Mac | user | saved voices cannot narrate here until that is solved; the main session reports the blocker; no one-take path comes back |
| ~~blank or long words; an unknown or pending take~~ changed 2026-11-02 (ASR, t42): any post to `…/examples/{id}/transcript` | web | 404 or 405: the route is never built |
| an old per-take route (`/film-example`, `/examples/{id}/preview`, `…/preview/audio`) | web | 404 or 405 |
| Changed 2026-11-02 (ASR, t42): a take whose heard words hold no letter, or more than 1000 characters (**changed, ASR plan check, t44:** no letter and no digit — blank or symbols only, `NO_WORDS`; "1 2 3 4 5" is usable) | job (transcription) | stored as `unusable` with its own reason (`NO_WORDS` or `TOO_LONG`); shown on its row and on the Voice model tab; left out of the bake and named in the done line; the bake goes on |
| Changed 2026-11-02 (ASR, t42): every take the job heard is `unusable` | job | `BakeBlocked` "No take can be learned from: <each take with its own reason>. …" (changed, ASR plan check, t44; before: "No speech was heard in any take, …"); no run row, no `bake_start`; the words stay stored |
| Changed 2026-11-02 (ASR, t42): `/voice/transcribe` answers 409 (a foreign bake started) | job | `BakeBusy` with the question of K4; words stored so far stay |
| Changed 2026-11-02 (ASR, t42): `/voice/transcribe` answers 503 | job | `failed` "Hearing the words of a take is not offered on this device yet." |
| Changed 2026-11-02 (ASR, t42): any other transcribe error (weights not on disk, a child death, a timeout, an identity mismatch, a models-service restart) | job | `failed` "Could not hear the words of take #N (<reason>)."; words stored before stay; the next Bake hears only the rest |
| Changed 2026-11-02 (ASR, t42): Cancel during "transcribing the takes" | job | the call in flight ends first; no other take is heard; no run row, no bake, no model; the stored words stay |
| Changed 2026-11-02 (ASR, t42): Cancel, `replace=yes` or a voice delete during the hand-off to `start_bake_run` | job | `job_lock` orders them: no run row (cancel first), or a run row with `cancel_requested_at` (cancel after, D88) |
| Changed 2026-11-02 (ASR, t42): the app restarts during "transcribing the takes" | `app` | the job is gone; no run row and no service bake, so `reattach` finds nothing; the next Bake hears only the takes still `waiting` |
| Changed 2026-11-02 (ASR, t42): the voice is deleted during the stage | job | `set_heard_words` writes nothing for a pending voice; `start_bake_run` refuses it; nothing is stored |
| Changed 2026-11-02 (ASR, t42): a take is added or deleted during the stage | job | the snapshot, written under `job_lock` after the stage, holds the `heard`, non-pending takes of that moment; a later take counts as added since the new version |
| Changed 2026-11-02 (ASR plan check, t44): a take is added while the stage runs (also during its last call) | job | the read under `job_lock` finds it `waiting`; the job releases the lock, hears it, and reads again; the snapshot holds it |
| Changed 2026-11-02 (ASR plan check, t44): a take is added after the final snapshot read under `job_lock` | job | not heard in this job, not in the snapshot; the page counts it as added since the new version |
| Changed 2026-11-02 (ASR plan check, t44): a queued take is deleted (pending, or drained) before its turn | job | `take_to_hear` returns `None`: skipped, no transcribe call, no read |
| Changed 2026-11-02 (ASR plan check, t44): a take is deleted while its transcribe call runs | job | `set_heard_words` writes nothing; the job goes on |
| Changed 2026-11-02 (ASR plan check, t44): `read_example_audio` fails for a take | job | the row is read again: deleted → skipped; still there → `failed` "The audio of take #N could not be read (<reason>)." |
| Changed 2026-11-02 (ASR plan check, t44): the request that loads Whisper also evicts another unit | models service log | the line gives the net `free_mb` change of the request and names the evicted units; it is never read as Whisper's cost (D103) |
| Changed 2026-11-02 (ASR, t42): ASR hears wrong words in a take | — | the bake trains on them; nobody corrects them (the user chose no editor); `## Open items outside scope` |

**L. Binding rules (Phase 5)**

- `ivms777/CLAUDE.md:120-141`: training and adapter inference run only in models-service children; `app`, `worker` and the scripts are HTTP
  clients. `feed/voice_bake.py` imports no model, and the one-model gate scans `feed` (`tests/test_one_model_process_gate.py:39-42`).
- `:163-204`, `:223-232`: no test loads a model; the user's Slice A run is the only proof; no time, memory, likeness or mood claim before it.
- `:264-274`: a bake shows stage, percent and Cancel; a second bake request offers to replace, never a dead end.
- `:151-161`, D33: voice model files leave only through the voice outbox drain.
- `:236-251`: the LoRA shape, the training cost and the authors' guidance are declared in the catalog.
- `:51-118`: `/voices` is a grid and `/voices/{id}` its one leaf; tabs never push; Esc unwinds the recorder first; the user never loses a take to a
  nav click without a question.
- `:3-36`: docs first in every slice (§2.14 Phase 5).
- Changed 2026-11-02 (rework, t39): `:206-216` — a change under `modelsvc audio models embedding captioning inference` reloads the models
  service and ends a running bake, so only lane 31b touches those folders, and only when no §6.5 bake runs (D84). `:271-274` — a bake that
  `app` did not start is stopped only after a question, never left as a dead end (D83). D33 and D38 — a saved narrator that cannot speak
  fails with its reason, never Automatic, a preset or one take.
- Changed 2026-11-02 (ASR, t42): `:120-141` — ASR runs only in a models-service child; `app` sends audio over HTTP and imports no model
  library (`feed/voice_bake.py` stays in the one-model gate). `:234-251` — the ASR model, its revision, languages and cost are declared in
  `models/catalog.py` (`TRANSCRIBERS`). `:163-204`, `:223-232` — no test and no agent loads Whisper; the user's §6.7 run is the proof that
  it runs, and no claim about how well it hears is made before it. `:264-274` — the transcription stage shows a stage, a percent and Cancel. `:206-216` — lane
  31d edits the reload folders and waits like 31b (D84).

**M. Rejected (Phase 5)**

Full fine-tuning (SFT); training in `app`, `worker`, the models-service parent or the voice child; two VoxCPM2 copies at once; one blocking
`train` call (`TorchWorker` would hold its lock for the whole bake, and an eviction would hold the governor lock with it, §1.8); TensorBoard; an
adapter cache in the models service (it outlives a delete, as D15 found for blobs); clip concatenation or averaging (t24, t28); ASR to fill in the
words; dropping a take without words; a likeness score; a time or memory promise; a cloud bake before a cloud run is measured.
**Changed 2026-11-02 (rework, t39):** also rejected — a prompt take beside the adapter (variant A) in films or tests; any film-take select
or pin; a one-take fallback for a voice with no usable model, also when §6.5 fails; a per-take test; retiring the model when a take it
learned from is deleted; an unused take sent only to satisfy the `/speak` contract (D76); a voice model speaking a language it learned no
take in (D77); cancelling or deleting a bake that `app` did not start without a question (D83).
**Changed 2026-11-02 (ASR, t42):** "ASR to fill in the words" and "dropping a take without words" leave this list: ASR now hears the words,
and a take with no usable words is left out with its reason shown (K2, D94). Rejected now: a words box, a confirm step or any typed words;
the kit text stored or trained as a take's words; a take blocking a bake for its words; ASR inside the training child, the voice child or
`app`; hearing a take when it is saved (D92); a second ASR model per language; hearing a take again once it was heard; a new package for
ASR (D90).

### 2.18 Automatic may pick a saved voice (Phase 6; user 2026-11-03, pair t50)

Goal: with Automatic, the app chooses the narrator itself, from the preset voices AND the user's saved voices whose voice model can narrate
the film's language here. D17 holds: one film speaks presets only or saved voices only. A narrator the user chose reads the whole film
(pair t50; this section does not change it).

**Facts (working tree, read 2026-11-03).** The t50 builder, the t49 lane and lane 32c edit some of these files now, so the line numbers move.
- `feed/build.py:272-284`: `chosen = read_narrator(…)`; `saved_film = chosen is not None and chosen.preset_speaker is None`. Only a chosen
  saved narrator reads `_saved_cast` (`:600-626`) and fills `saved` / `saved_cast` (`SavedCast(token, name)`).
- `feed/build.py:314`: `speakers=catalogue_for_prompt(language) if chosen is None else ""`. Automatic shows the story pass presets only.
- `feed/build.py:356-369`: a chosen narrator → `passage_voices = []`. Automatic → `resolve(language, story.narrator) or
  pick_speaker(language, shot_signals)`, then `_passage_voices` (`:655-685`), which resolves each name with `resolve` only (`:679`).
- `feed/build.py:385-386`: `narrator_chosen_by` is "user", "story" (the story named a real voice) or "mood"; `:394` `offered`.
- `feed/build.py:403-430`: for every used voice of `saved`: `film_model` again under `voice_turn`, a problem → `_fail(_cannot_narrate(…))`,
  `model_reference` (D76); `:452-458` the D86 reply check; `:469-481` `voice.saved`, `voice.models`, `cast.narrator_model`; `:573-590` the
  ready save with `_deleted_voice` in one transaction. All of it reads `saved_film`.
- `feed/saved_voices.py:1239-1248` `castable_models(conn, owner_id, language)`: the voices not being deleted whose `film_model` finds no
  problem (D77). Voice names are unique per owner by `casefold()` (`_check_name_free`, `:391-403`).
- `feed/voice.py:75-77` `saved_for_prompt`: "  <name> — a saved voice: this person's own voice". After t50 only
  `tests/test_feed_voice.py:154-161` calls it. `feed/voice.py:147-166` `pick_speaker` chooses by the catalog's `register` (`:137-144`).
- `feed/story.py:1230-1240`: the cast block prints `speakers` verbatim under "THE VOICES YOU MAY CAST, and what each sounds like:", then
  "Use these names EXACTLY". The cast retry (`:299-313`) and the score (`:784`) only test that `speakers` is not empty.
- `web/app.py:965-966`: the picker offers saved voices only when `cloning_entry_for(ctx.settings.profile, language)` is set, then
  `castable_models`. The build call (`web/app.py:1229-1233`) passes no profile.
- `modelsvc/backends/composite.py:123-124`, `:139-144`: a `/speak` with `references` goes whole to the cloning entry, so a preset cannot
  speak in it (D17).
- `feed/store.py:120-139` `film_narrator` and `web/templates/feed.html:107-112` ("This film") already print a saved narrator as "<name>
  (saved voice)" with "· voice model v<N>", for any `chosen_by`.

**A. The build — `feed/build.py`** (lane 36a)

```python
def build_presentation(conn, client, audio, owner_id, group_id, *, language=DEFAULT, planner_model="gemma4-E2B",
                       base_seconds=4.5, max_shots=40, progress=None, cancelled=None,
                       voices: Storage | None = None, voice_lock: threading.Lock | None = None,
                       profile: str | None = None) -> Presentation
def _passage_voices(story, language: str, film_voice: str | None, bracketed: set[int] | None = None,
                    saved: Sequence[SavedCast] | None = None) -> list[str | None]
```

1. `saved_chosen = chosen is not None and chosen.preset_speaker is None` (today's `saved_film`). `_saved_cast` runs for it as today.
2. Automatic (`chosen is None`): when `profile` and `voices` are set and `cloning_entry_for(profile, language)` is not None, `saved =
   [(voice, model) for voice, model in castable_models(conn, owner_id, language) if resolve(language, voice.name) is None]`; else `saved =
   []`. `saved_cast` is built from `saved` as today (D105).
3. `speakers=automatic_for_prompt(language, saved_cast) if chosen is None else ""` (B).
4. `saved_pick = None` before the chosen / Automatic branches (`:356-369`), so step 5 never reads an unset name when the user chose a
   preset or a saved voice. The Automatic narrator (D107): `preset_pick = resolve(language, story.narrator)`; `saved_pick = None if
   preset_pick else resolve_saved(story.narrator, saved_cast)`. With `saved_pick`: `speaker = saved_pick`, `passage_voices = _passage_voices(story,
   language, speaker, bracketed, saved=saved_cast)`. Else: `speaker = preset_pick or pick_speaker(language, shot_signals)` and
   `_passage_voices(story, language, speaker, bracketed)`. `narrator_asked_was_real = bool(preset_pick or saved_pick)`.
5. `saved_film = saved_chosen or saved_pick is not None`, set right after the branches of step 4. Every later use reads it (`:394`, `:412`,
   `:469-481`, `:573`), so a saved pick gets the model-only references, the D86 check, the labels and the ready save of a saved-voice film
   (D109). Under the voice turn, a pick for which film_model now reports no usable model for the film's language fails with the
   `_cannot_narrate` text; a usable Rebake completed before the voice turn supplies the new version; a Rebake after synthesis does not
   invalidate the film, which records the version actually spoken (`:407-427`). No new model-change failure.
6. `offered`: a chosen saved narrator → the saved names (as today); otherwise the preset names, then the names of `saved_cast`.
7. `_passage_voices`: with `saved`, each name resolves by `resolve_saved(name, saved)` only; else by `resolve(language, name)` only. A name
   of the other kind is None, so that passage reads in the narrator's voice. The bracket gate does not change.
8. The docstring of `voices`: "A film narrated by a saved voice, chosen or picked by Automatic, needs it."

**B. The table the story pass sees — `feed/voice.py`** (lane 36a; D106)

```python
def automatic_for_prompt(language: str, saved: Sequence[SavedCast]) -> str
```

- `saved` empty → exactly `catalogue_for_prompt(language)`. Else: those lines, then the line "  SAVED VOICES — the user's own recordings of
  real people. How each one sounds is not described. Choose one when it suits the memory, for example when the memory is about that
  person or the notes ask for them. Never mix the two kinds: a saved narrator takes saved passage voices only, a preset narrator presets
  only.", then `saved_for_prompt(saved)`.
- No gender, register or sound note for a saved voice. `feed/story.py` does not change.
- The module docstring (`:28-29`) adds: "Automatic shows the story pass both kinds (`automatic_for_prompt`), and the narrator's kind
  decides the kind of every passage voice."

**C. The page — `web/app.py`, `web/templates/feed.html`** (lane 36b; D110)

- `web/app.py:1229-1233`: the build call adds `profile=ctx.settings.profile`. Nothing else in `web/app.py` changes.
- The Automatic option: "Automatic — the app picks a voice for this memory".
- The always-hint: when `picker.voice_copy` → "The story picks the voice: a preset voice or one of your saved voices with a voice model for
  <Language>. Otherwise the memory's mood picks a preset voice."; else → "The story picks a preset voice; otherwise the memory's mood
  guides the choice."
- The Automatic bracket line: "A [voice] note can request another voice for one passage: another preset when a preset narrates, another
  saved voice when a saved voice narrates." The chosen-narrator line stays.
- "This film" and the card do not change (D108): "narrated by Grandma (saved voice) — picked automatically by the story · voice model v2";
  the card "Grandma (auto)".

**D. Not changed:** `feed/story.py`, `feed/store.py`, `feed/saved_voices.py`, `pick_speaker`, `_saved_cast`, `_narrator_picker`'s options and
`_store_narrator`, `memory_narrators`, the models service, the chosen-narrator rules of t50.

**E. Docs first (lane 35).** Anchors read 2026-11-03; they move while t49 and t50 run.
- `docs/design.md` §11.1:
  - the film mermaid, node 3 (`:2089`): "Automatic only: casts the narrator + a per-passage voice from the preset table" → "… from the
    preset table and the saved voices with a voice model for the language"; node 4a (`:2090`): "Automatic: a [voice] bracket may give ONE
    passage another preset" → "… another voice of the narrator's kind";
  - "A model CASTS the film" (`:2101-2108`): Automatic also shows the story pass the saved voices whose voice model can narrate the language
    here, by name, with no description of their sound;
  - "**Automatic picks a preset voice only, never a saved voice.**" (`:2116`) → "**Automatic may pick a saved voice.**": the candidates
    (D105), the mood fallback picks a preset (D108), `narrator_chosen_by` keeps "story" / "mood";
  - "One voice engine per film … only presets speak." (`:2129-2132`): the narrator's kind decides every passage voice (D107); no saved voice
    → presets only;
  - "A saved narrator makes the film a saved-voice film" (`:2142`) and "**A saved narrator never degrades to a preset.**" (`:2158`): chosen
    or picked by Automatic; the references hold the narrator's model and each saved passage voice's model (D109).
- `docs/design.md` §11.3, the `voice` row (`:2624`): "in an Automatic film, that PASSAGE alone is spoken by another preset voice" → "… by
  another voice of the narrator's kind: a preset in a preset film, a saved voice in a saved-voice film".
- `docs/design.md` §13, the `/feed` bullet (`:2781`): the option text of C.
- `docs/ui.md`: "This film" (`:462-471`) adds the example of C; the Automatic option (`:482`); the always-hint (`:512-513`), both
  variants of C; the Automatic bracket line (`:516-518`).
- Validate the edited §11.1 mermaid block with the script in `<task-skill>/references/mermaid.md`. It
  must print `OK`.
- Closing check (step 37, after 36b): `grep -rn "never picked automatically\|preset voice for this memory\|picks a preset voice only"
  docs/ web/templates/ tests/` finds nothing.

Rejected: a new picker entry beside Automatic (t50); a saved voice in the mood fallback; a description of a saved voice's sound that nobody
measured or declared; a film with both kinds; a preset after a failed saved pick.

### 2.19 Editable narration (Phase 7; user 2026-11-03, pair t62)

Goal: a finished film's narration shows in "Tell the story better" as timed, editable rows, by default. The user changes the words, and
↻ Regenerate speaks that version as written, also on every later ↻ Regenerate, until the user chooses "Write a new story". The pair
settled the approach (t62, `<base>/.pair/files/t62-edit-narration-agreed.md`) and corrected it at the plan check (t63,
`<base>/.pair/threads/t63/decision.md`, `<base>/.pair/files/t63-merge2-writer.md`; t63 point 1 supersedes t62 point 5); this section
plans how. §2.19 I also fixes the existing shot/audio order defect (the user's "Fix in this task", pair t64): after it, a passage's words,
its photograph, its voice and its time are one thing in one order.

**Facts (working tree, read 2026-11-03).** The t60 and t61 fixers edit `docs/ui.md` and `docs/design.md` now, so the doc lines move.
- `feed/story.py:217-220`: `Story.chunks` is the texts of `story.script` in TELLING order. `feed/build.py:463-473` hands them to `/speak`
  as the chunks, so segment `i` of the reply is chunk `i`. `audio/segments.py:74-78` gives each segment `text`, `start` and `end`.
- `feed/director.py:243-263` `with_script` re-sorts the passages into memory order (`_in_memory_order`, `:266-290`) and blanks `line`.
  So shot `i` is NOT passage `i` when the story is out of memory order. `with_narration_timing` (`:293-317`) gives shot `i` the start of
  segment `i`, and pairs only when the counts are equal (`:306-308`). §2.19 I removes the re-sort: after it, shot `i` IS passage `i`.
- `feed/director.py:161-195` `Edit.as_dict` keeps the joined `narration` and a `line` (a title card) per shot. No passage text, no
  passage boundary. `feed/build.py:340-348` `decisions["story"]` keeps counts, `narrator_asked`, `score_plan` and `music_prompt`; no cards,
  no passage texts.
- `feed/story.py:887-939` `_script` always returns a script for a told story (`_even` is its fallback), so every ready film has
  passages. `:925-926`: a bad photo index takes the previous passage's photo. `:1000-1020` `_score`: 0-based passage indexes.
  `:222-229` `lines_for` drops a card past the last shot; `feed/director.py:414-426` `with_lines` takes a full-length list.
- `feed/story.py:271-278` strips every `[bracket]` from the told sentences before speech. Nothing else strips one.
- `feed/build.py:853-878` `_fail` upserts a `failed` row with `edit.as_dict()` and the decisions so far over the ready row
  (`feed/store.py:38-62`, `ON CONFLICT … DO UPDATE`). `feed/build.py:581-626`: only a saved-voice film saves under `BEGIN IMMEDIATE`
  (`:602-622`); a preset film calls `save_ready()` alone. `:359-363`: a film with no narration fails.
- `feed/build.py:372-426`: the bracket gate `_bracketed_passages(viewer_notes, previous_seconds, len(story.script))` is positional; a
  chosen narrator reads every passage (`passage_voices = []`); Automatic resolves `story.narrator` and `_passage_voices` on `story.voices`.
- `feed/notes.py:8-10`: notes are INPUT to the story pass, never output. `:22` `MAX_LENGTH = 600`, `:49` cuts at it. `feed/story.py:37`
  `WORDS_PER_NOTE`, `:44-57` `narration_words`, `:60-63` `_note_count` counts the `[at …]` lines of the notes only.
- `web/app.py:1075-1118` the note routes answer JSON; an edit is delete-then-add. `:905-912` `notes_json` in a
  `<script type="application/json">`. `:670-708` `_film_block` writes the card's retell text (`:699-703`); `:747` calls it.
- `web/app.py:1120-1155` ↻ Retell memory → `albums/memories_build.py:63-92` `rebuild_memory` → `albums/memory_store.py:82-104`
  `rewrite_memory`: it deletes the memory's presentations (`:103`) and keeps its notes and narrators. A failed retelling writes nothing.
  `albums/memory_store.py:42-79` `replace_memories` deletes the memory groups, and every `ON DELETE CASCADE` row goes with them.
- `web/static/feed-notes.js`: `:32-40` reads `#notes`; `:50-52` `markStale`; `:57-64` Enter saves; `:69-76` auto-grow; `:83-93` `saveNote`
  (no post when the text did not change); `:97-115` the fold, folded on a first load; `:117-153` `render`; `:182-195` `window.feedNotes`;
  `:203` the panel opens on the Prepare screen. `web/static/feed-gl.js:1370`, `:1392` float every entry of `window.feedNotes.all()` as a
  bubble; `:1603` sets `seek`; `:1248`, `:1314`, `:1671` hide or show the panel on play, pause and the title card.
- `web/templates/feed.html:72-73` the body carries `data-key` and `data-lang`; `:125-126` the Regenerate button and its title; `:227` the
  Prepare button; `:246` `#notes`; `:250-292` the panel.
- `db/connection.py:100-104`: every `migrate` creates the missing tables of `db/schema.sql`. A new table needs no version bump (D39).
- `tests/test_feed_build.py:176-182` and `tests/test_web_feed.py:393-396` pin the timeline keys; `tests/test_feed_director.py:197-199`
  pins `as_dict`, which does not change.
- Added (plan check, t63): `feed/build.py:559-573` calls the conductor with `keep_timing=aligned`, and `feed/director.py:343-357` then
  keeps every shot's `at` and `seconds`: when the speech matches the shots, a timing note changes nothing. `feed/build.py:715-718`
  keeps a passage voice only at a passage a current bracket points at; `:733-734`: no bracket note → no passage voice at all.
- Added (t63): `feed/score.py:72` `SCORE_MAX_SECONDS = 180.0`, applied at `:110`: the score never exceeds 180 s, whatever the script.
- Added (t63): `web/templates/feed.html:164` is `<audio id="narration">`, and `web/static/feed-gl.js:26` reads it by that id.
- Added (t63): `web/templates/feed.html:328-355` posts both prepare forms (`:107-127`, `:224-228`) from a submit handler with no wait;
  `web/static/feed-notes.js:119` rebuilds every field on each render, and `:140` saves on blur, without a wait.
- Added (t63): `web/templates/feed.html:59-69` the picker lines ("A [voice] note can request another voice for one passage: …");
  `:267-283` the legend ("the words become narration, the bracket becomes an instruction").

**A. The narration of every new film — `timeline.passages` and `timeline.script`** (lane 40b; the pure builder is lane 40a's)

```python
# feed/narration.py
def timeline_script(story: Story, segments: Sequence[dict], passage_voices: Sequence[str | None], speaker: str,
                    photo_ids: Sequence[int], *, token: str) -> tuple[list[dict], dict]
```

1. One passage per entry of `story.script`, in TELLING order, the order of `story.chunks` and of the audio. Passage `i`:
   `{"n": i, "text": story.chunks[i], "start": seg[i]["start"], "end": seg[i]["end"], "photo": story.script[i][0],
   "photo_id": photo_ids[story.script[i][0]], "voice_asked": story.voices[i] or "", "voice_used": passage_voices[i] or speaker}`
   (with no `passage_voices`, `voice_used` is `speaker`). The passages come from `story.script` and the segments, never from
   `edit.shots`; after §2.19 I, shot `i` shows passage `i`'s photograph.
2. `seg` = the segments that are dicts with a `start`. When their count is not `len(story.chunks)`, every `start` and `end` is null and
   the texts stay (D112).
3. `script = {"token": token, "narrator_asked": story.narrator, "narrator_used": speaker, "cards": [{"n": n, "text": text}, …],
   "score_plan": [{"from": index, "brief": brief}, …], "music_prompt": story.music_prompt}`.
   **Changed (t63, t64; D125):** a card names its PASSAGE. After §2.19 I, `story.cards` holds `(i, text)` with `i` the 0-based sentence
   (= passage) number, and shot `i` is passage `i`, so `timeline_script` stores `{"n": i, "text": text}` as it is. A card past the last
   passage is dropped, as `lines_for` drops it (`feed/story.py:222-229`). Nothing sorts or converts it.
4. `feed/build.py`: `photo_ids = [shot.photo_id for shot in edit.shots]` right after `direct(…)` (`:274`), before `with_script` replaces
   `edit`. After `with_narration_timing` (`:496`): `passages, script = timeline_script(story, segments, passage_voices, speaker,
   photo_ids, token=secrets.token_hex(16))`. `save_ready` (`:585-598`) stores `timeline={**edit.as_dict(), "passages": passages, <!-- pragma: allowlist secret -->
   "script": script}`.
5. `_fail` does not change: a failed row has no passages and no script. The draft (B) holds the words of a retry.
6. `Edit` and `as_dict` do not change, and the A-H work does not change `feed/director.py`; lane 39b changes it under §2.19 I (G,
   **changed, t65**). `/feed/{key}/timeline.json` serves the two new keys as stored; the player reads neither.

**B. The draft — `narration_drafts` in `db/schema.sql`, the store in `feed/narration.py` (new)** (lane 40a; D113, D122)

**Changed (plan check, t63):** a successful film REBASES its draft and never deletes it; a fresh `draft_token` names each seeded draft;
`changed` replaces `from_edits`.

```sql
-- The user's edited narration of one memory's film in one language (design §6, §11.3). No row = the film's own words.
-- Seeded with ALL passages of the ready film on the first change, with every replay input, so a retry never needs that film.
-- A film made from it REBASES it in the transaction of its ready save: the rows take that film's passages, times and script; the
-- words stay. Only "Write a new story", a retelling and the memory's own deletion remove it.
CREATE TABLE IF NOT EXISTS narration_drafts (
  group_id      INTEGER NOT NULL REFERENCES groups(id) ON DELETE CASCADE,
  language      TEXT NOT NULL,                          -- feed.language.LANGUAGES
  draft_token   TEXT NOT NULL,                          -- 32 hex, new at every seeding; a rebase keeps it
  source_token  TEXT NOT NULL,                          -- timeline.script.token of the film whose times the rows carry
  revision      INTEGER NOT NULL CHECK (revision >= 1), -- 1 at seeding, +1 on every change and on every rebase
  script_json   TEXT NOT NULL,                          -- narrator_asked, narrator_used, cards, score_plan, music_prompt
  passages_json TEXT NOT NULL,                          -- [{n, text, start, end, photo, photo_id, voice_asked, voice_used}], telling order, never empty
  updated_at    TEXT NOT NULL,
  PRIMARY KEY (group_id, language)
);
```

```python
MAX_SCRIPT_CHARS = 10_000                # a provisional app limit (D115)

@dataclass(frozen=True)
class Passage:
    n: int                      # its number in the film whose times it carries; a change never renumbers it
    text: str
    start: float | None         # seconds in that film's audio; None = the voice did not pair the segments
    end: float | None
    photo: int                  # the story's photograph for it: an index of that film's photo_ids
    photo_id: int | None
    voice_asked: str            # the story pass's voice for it ("" = the narrator)
    voice_used: str | None      # what spoke it in that film (a preset name or a saved-voice token)

@dataclass(frozen=True)
class Draft:
    group_id: int
    language: str
    draft_token: str
    source_token: str
    revision: int
    script: dict
    passages: tuple[Passage, ...]

@dataclass(frozen=True)
class NarrationRows:            # what the panel shows
    state: str                  # "film" | "draft" | "legacy" | "none"
    source: str | None          # the film token (film) or the draft's source_token (draft); None otherwise
    draft: str | None           # the draft's draft_token; None for every other state
    revision: int               # the draft's revision; 0 for the film's own rows
    seekable: bool              # the times locate the ready film of this language
    changed: bool               # a draft whose words are not the words of the ready film (D122); False for every other state
    rows: tuple[Passage, ...]
    def as_json(self) -> dict

class NarrationRefused(ValueError):  # .code: "stale" | "empty" | "bracket" | "too_long" | "missing"; .message: the text of D

def read_draft(conn, group_id: int, language: str) -> Draft | None
def narration_rows(conn, group_id: int, language: str) -> NarrationRows
def save_passage(conn, group_id: int, language: str, n: int, text: str, *, source: str, draft: str,
                 revision: int) -> NarrationRows
def remove_passage(conn, group_id: int, language: str, n: int, *, source: str, draft: str, revision: int) -> NarrationRows
def discard_draft(conn, group_id: int, language: str, *, source: str, draft: str, revision: int) -> NarrationRows
def draft_languages(conn, group_id: int) -> list[str]           # in LANGUAGES order
def draft_story(draft: Draft, photo_ids: Sequence[int]) -> Story  # pure (C)
def rebase_consumed(conn, draft: Draft, passages: list[dict], script: dict) -> bool  # the caller holds the transaction (C)
```

1. `narration_rows`: a draft → state `draft`, its rows, `source = source_token`, `draft = draft_token`, `revision`, `seekable` = the
   ready film's `timeline.script.token` equals `source_token`, and `changed` = NOT (`seekable` and the draft's `[(n, text)]` equals the
   ready film's `[(n, text)]`) (D122). No draft and a ready film (`status == "ready"` and a `narration_key`, as `web/app.py:873-875`)
   with `timeline.passages` → state `film`, `source` = its token, `draft` None, revision 0, `seekable` True, `changed` False. A ready
   film without `passages` → `legacy`. Anything else → `none` (source and draft None).
2. Every change runs under `BEGIN IMMEDIATE`: read, check, write, `COMMIT`; `ROLLBACK` on any error (the pattern of `feed/build.py:608-622`).
3. The stale rule: the posted `(source, draft, revision)` must equal `narration_rows`'s `(source, draft or "", revision)`, read inside
   that transaction. Else `stale`. State `none` never matches. So another window, a new film, a failed build that replaced the film, a
   rebase, "Write a new story" and a retelling all read as `stale` (D114). A draft seeded again from the same film after "Write a new
   story" has a new `draft_token`, so a page of the old draft, or a build that used it, never writes into the new one.
4. Seeding: with no draft, the change copies ALL passages of the ready film and its `timeline.script` (without `token`) into a new row,
   with `draft_token = secrets.token_hex(16)`, `source_token` = the film's token and revision 1, and applies the change in the same
   write.
5. The text (D115): `" ".join(text.split())`. A text equal to the row's text writes nothing: no seed, no new revision; the answer is the
   current rows. An empty text removes passage `n`; removing the last one is `empty`. A text with `[` or `]` is `bracket`. A script whose
   texts sum to more than `MAX_SCRIPT_CHARS` characters is `too_long`. No row has its own limit, and nothing is cut. `n` not in the rows is
   `missing`.
6. `remove_passage` is `save_passage` with the text "". `discard_draft` ("Write a new story"): the stale rule, then `DELETE` the row.
7. `rebase_consumed(conn, draft, passages, script)`: `UPDATE narration_drafts SET passages_json = ?, source_token = ?, script_json = ?,
   revision = revision + 1, updated_at = … WHERE group_id = ? AND language = ? AND draft_token = ? AND revision = ?`, with the new film's
   passages (`n` from 0, the new times), the new film's token and its `script` without `token`. It answers True when it changed a row.
   A newer revision of the same draft, a draft with another `draft_token`, or no row → nothing changes.
8. No function reads or writes `memory_notes`. A narration row is never a note: no `MAX_LENGTH` cut, no `WORDS_PER_NOTE` room, no
   `notes_for_prompt` line.

**C. A build with a draft — `feed/build.py`** (lane 40b; D116, D117)

1. The snapshot: at `step("writing the narration")` (`:301`), before `write_story`, `draft = read_draft(conn, group_id, language)`.
   No draft → today's path, unchanged (`:314-349`), plus A.
2. With a draft, `story = draft_story(draft, photo_ids)`. `write_story` is not called, so `_tell`, the translation, the expansions, the
   note weaving and `_dress` (`feed/story.py:259-284`) never run, and `_context_tokens` is not called for a story. The Story:
   - `script` = `((photo index, text), …)` in the draft's order. The photo index is the position of `photo_id` in `photo_ids`; when it is
     gone (a photo deleted from the library), the previous passage's index, else 0 (the rule of `feed/story.py:925-926`).
   - `narration = " ".join(texts)`; `voices` = the `voice_asked` of each passage; `narrator = script["narrator_asked"]`;
     `music_prompt = script["music_prompt"]`.
   - `cards` (**changed, t63, t64; D125**): for each stored `{"n", "text"}` whose passage `n` is still in the draft, `(k, text)`, where
     `k` is the position of passage `n` in the draft. A card whose passage was removed is dropped. Shot `k` is passage `k` (§2.19 I), so
     a card stays on its own passage when another passage goes.
   - `memory_order` (**added, t64; D127**): "kept" when the resolved photo indexes introduce the photographs in their numbered order
     (`story.introduces_in_order`), else "broken". No model call and no relabel: every passage keeps the photograph stored with it.
   - `score`: each `{"from": f, "brief": b}` moves to the position of the first passage of the draft whose `n` ≥ `f`; a second entry on
     the same position is dropped; an entry with no such passage is dropped.
3. `decisions["story"]` with a draft: `{"source": "draft", "draft_revision": draft.revision, "context_tokens": None, "passages":
   len(story.script), "narrator_asked": story.narrator, "score_plan": […], "music_prompt": …, "notes_uncovered": uncovered_notes(…),
   "memory_order": story.memory_order}`. A build with no draft writes today's keys plus `memory_order` (§2.19 I).
4. Everything after `:350` runs as today on that Story. The rules of Phases 3-6 apply unchanged:
   - a chosen preset or saved narrator reads every passage; the `voice_asked` values are ignored (t50; `:374-382`);
   - Automatic: `narrator_asked` resolves as a preset, then as a saved candidate of `castable_models` (D105, D107), else `pick_speaker`
     ("mood", D108). `_passage_voices` takes the `voice_asked` values through the bracket gate of the CURRENT notes
     (`_bracketed_passages`, `:372`; a stored voice stays only at a passage a current bracket points at, `:715-718`, and with no
     bracket note there is none, `:733-734`). A new `[voice]` note recasts nothing, because no story pass reads it; when no passage is
     cast, the film keeps the notice "a note asked for a different voice and none was cast" (`:428-431`);
   - a saved-voice film: the voice turn, the D86 reply check, D109 and the deleted-voice check of the ready save, as today;
   - `/speak` gets `story.narration` and `story.chunks`: the draft's texts, exactly. The timing, the score (`plan_sections` with the moved
     plan and the stored `music_prompt`), the beats and the conductor (`directions` = the notes) run as today. When the speech matches
     the shots, the conductor keeps every shot's time (`:559-573`; `feed/director.py:343-357`), so a timing note changes nothing; an
     effect note still reaches it. The stages, the progress and Cancel do not change.
5. A builds the new film's passages from the new audio: new times, `n` from 0.
6. The save (D117): no draft and not `saved_film` → `save_ready()` as today (`:602-603`). Otherwise ONE `BEGIN IMMEDIATE`: the
   deleted-voice check (a saved-voice film only) → `save_ready()` → `rebase_consumed(conn, draft, passages, script)` when there is a
   draft → `COMMIT`. The consumed draft then holds the same words with the new film's times, token and script, at revision + 1: state
   `draft`, `seekable`, not `changed`. A newer edit saved during the build survives as it is (its revision differs): it keeps the
   earlier film's times, so it is not `seekable`, and it is `changed`. A deleted voice rolls back and fails as today (`:617-621`), and
   the draft stays.
7. `_fail`, `Cancelled` and every other way a build ends without a ready save leave the draft as it is. Nothing touches it at the start.

**D. The routes — `web/app.py`** (lane 40c; D114). JSON answers, like the note routes (`:1075-1118`). `lang` goes through `_language`.

| Route | Form fields | 200 answer |
|---|---|---|
| `GET /feed/{key}/narration?lang=` | — | `narration_rows(…).as_json()` |
| `POST /feed/{key}/narration/{n:int}` | `lang`, `source`, `draft`, `revision`, `text` | `{"saved": true, **as_json}` (an empty text removes the passage) |
| `POST /feed/{key}/narration/{n:int}/delete` | `lang`, `source`, `draft`, `revision` | the same, through `remove_passage` |
| `POST /feed/{key}/narration/discard` | `lang`, `source`, `draft`, `revision` | `{"saved": true, **as_json}` after "Write a new story": the ready film's own rows (state `film`), else state `none` or `legacy` |

- The two `{n}` routes use the path convertor `{n:int}`, so `POST …/narration/discard` never matches them. The film's own rows post
  `draft` as "".
- `as_json()` = `{"state", "source", "draft", "revision", "seekable", "changed", "rows": [{"n", "text", "start", "end", "stamp"}]}`;
  `stamp` is `m:ss` (the `Note.stamp` rule, `feed/notes.py:33-37`) or "–:––" for no time.
- A refusal answers `{"saved": false, "error": <code>, "message": <text>}`:
  - `stale`, 409: "Not saved: this narration changed in another window, or a new film replaced it."
  - `missing`, 404: "This passage is no longer in the narration."
  - `empty`, 400: "A film needs narration. Keep at least one passage."
  - `bracket`, 400: "The narration is read aloud word for word. Put a [bracket] instruction in a note."
  - `too_long`, 400 (**changed, t63**): "The narration is longer than this app allows: <N> characters, at most 10000."
  - an unknown memory: 404 from `_memory_group`, as the note routes.
- `feed()` (`:840-937`): `"narration": rows` and `"narration_json": json.dumps(rows.as_json(), ensure_ascii=False).replace("</", "<\\/")`
  (the `notes_json` rule, `:905-912`).
- `_film_block(key, states, drafts)` (`:670-708`): with drafts, the retell text adds "Deletes your edited narration in: <Language>, …."
  The caller (`:747`) passes `draft_languages(ctx.conn, group_id)`. Nothing else in `web/app.py` changes; the build call does not.

**E. The panel — `web/templates/feed.html`, `web/static/feed-notes.js`, `web/static/feed.css`** (lane 40c; D119, D120, D123, D124)

- `feed.html`:
  - `<script id="narration-data" type="application/json">{{ narration_json | safe }}</script>` beside `#notes` (`:246`). Never
    `id="narration"`: that is the audio element (`:164`, read by `feed-gl.js:26`). Every id on the page stays unique.
  - Inside `#notes-body`, above `#notes-list` (`:262`): one `<p class="feed-narration-state" data-state="…" hidden>` per state line
    below; the disclosure line of every draft with the button "Write a new story"; `<p class="feed-narration-error" role="alert"
    hidden>` with a hidden button "Load the current narration".
  - With a draft, the Regenerate button (`:125-126`) and the Prepare button (`:227`) get the title "Speak your edited narration, and
    score this memory again".
  - The draft variants (D124): the picker's Automatic bracket line (`:66-69`) and the legend's first lead (`:269-273`) each get a twin
    marked `data-narration="draft"`; the originals are marked `data-narration="no-draft"`. `feed-notes.js` shows the twin of the
    payload's state on load and after every reply, so the page stays right when the first change makes a draft. The chosen-narrator
    line ("The chosen narrator reads the whole film.") has no twin.
  - The prepare submit handler (`:328-355`) waits for `await window.feedNotes.settled()` before its `fetch`. False → it posts nothing,
    and its refusal line (the `.feed-refused` line of `:343-351`) reads "Not started: an edit to the narration is not saved. See Tell
    the story better." (D123).
  - The panel keeps its hide, open and fold rules (`feed-notes.js:97-115`, `:203`; `feed-gl.js:1248`, `:1314`, `:1671`).
- The state lines (the template holds them; the script shows the one of the payload's state):
  - `film`: "The film's narration is in this list with your notes. Change its words, and ↻ Regenerate speaks them as written. With no
    change, ↻ Regenerate writes a new story."
  - `draft`, `changed`: "Your edited narration. ↻ Regenerate speaks it as written."
  - `draft`, not `changed`: "This film speaks your edited narration. ↻ Regenerate speaks it again as written." (D120)
  - under either draft line: "Notes cannot rewrite these words or provide a new voice or music interpretation. Existing voice choices
    still pass current note gating and validation. Effect instructions reach the conductor; matched speech fixes shot timing." and the
    button "Write a new story".
  - `legacy`: "This older film has no timed narration text. ↻ Regenerate makes a new film whose narration you can edit here."
  - `none`: no line.
- The draft twins (D124):
  - the picker, Automatic: "With your edited narration, a [voice] note cannot ask for a new voice: it keeps the voice the story already
    gave the passage it points at."
  - the legend lead: "With your edited narration, a note's words are not narrated and cannot change its words, voices or music. Effect
    brackets still reach the conductor; matched speech fixes shot timing."
- `feed-notes.js`:
  1. It reads `#narration-data` like `#notes` (`:32-40`, loud on a parse error) and keeps the payload apart from `notes`.
     `window.feedNotes.all()` still returns the notes only, so a passage never floats as a bubble (`feed-gl.js:1370`, `:1392`).
  2. `render` (`:117-153`) draws one list, keyed by passage `n` and note id. A passage row is `<li class="is-narration">`: the stamp
     button, `<span class="feed-note-tag">narration</span>`, the auto-grown textarea, and ×. A re-render never replaces a row whose
     field has the focus or holds text that differs from its saved text (an unsaved row): that row keeps its element, its text, its focus
     and its caret. The rows sort by time (a passage's `start`, a note's `at`); on a tie the passage comes first. Passages with no time
     come first, in telling order, then the notes. The count (`:118`) is the number of rows: passages plus notes.
  3. The times map to the cut that plays: with a player and `seekable`, a passage stamp seeks to its `start`
     (`window.feedNotes.seek`). A draft row keeps the time of the film it came from, also after its text changed, because that film's
     audio did not change. Without `seekable` (a draft of an earlier film: after a failed build, or a newer edit that a successful build
     did not use), the stamp is disabled with the title "From the film these words were taken from". With no player (the Prepare screen),
     it is disabled with the notes' title "Where this lands once the film is made" (`:130-134`). A passage with no time shows "–:––",
     disabled. A film made from a draft gives the rebased draft its own new times (C.6).
  4. A passage saves on blur and on Enter (`:57-64`, `:138-141`); shift+Enter is a newline, which the server joins to a space. × posts
     the delete route. An unchanged text posts nothing (`:84`). Passage saves run one at a time, in order: each waits for the reply of the
     one before, and posts the `source`, `draft` and `revision` of the latest payload. A 200 replaces the payload and re-renders (rule
     2). Every 200 that changed a row calls `markStale()` (`:50-52`). At load, `markStale()` runs only for a `changed` draft.
  5. A 400, a 404 or a network failure (D123): the error line shows the message ("Not saved: the app did not answer. Your text stays in
     its row." for a network failure); every save queued behind it is dropped, never posted; those rows stay unsaved with their text.
     The next blur of an unsaved row posts it again, as a new save. A × that has no 200 of its own (refused, failed, or dropped behind
     a failure) leaves its row and counts as unsaved: `settled()` answers false. Click × again, or "Load the current narration", which the
     error line then offers, with no freeze. Only that row's own delete 200 or a successful load clears it; a 200 on another row does
     not. The load waits for the queue and posts no passage save while it loads; a failed load clears nothing.
  6. A 409 (D123): every save queued behind it is dropped, never posted, and the panel posts no more passage saves. Every passage field
     becomes read-only and keeps its text, so nothing typed is lost. The error line shows the stale message, then "Copy any text you want
     to keep, then load the current narration.", and the button "Load the current narration". That button reads `GET
     /feed/{key}/narration` and re-renders every row: the one re-render that replaces unsaved rows. Then the panel saves again.
  7. `window.feedNotes.settled()` → `Promise<boolean>` (D123): it blurs the panel's focused field (a passage blur queues its save),
     waits for the queue, and answers True only when no passage row is unsaved (a × with no 200 of its own counts, rule 5) and no 409
     froze the panel. It does not wait for a note's save (`## Open items outside scope`).
  8. "Write a new story": it waits for `settled()`; False → it posts nothing. Else `confirm("Stop using your edited narration? The next
     Prepare or Regenerate writes a new story.")`, then the discard route; the reply re-renders.
  9. On load and after every reply, it shows the state line of the payload and the matching `data-narration` twins.
- `feed.css`: `.feed-note-tag`, `.feed-narration-state`, `.feed-narration-error`, `.feed-narration-undo` only.

Wireframes (part of the gate; the texts are those above, the numbers are an example):

```
A ready film, on the title card or a pause (state "film"):
┌ Tell the story better ───────────────────────────────────────────── 5 ▾ ┐
│ The film's narration is in this list with your notes. Change its words, │
│ and ↻ Regenerate speaks them as written. With no change, ↻ Regenerate   │
│ writes a new story.                                                     │
│                                                                         │
│ [0:00] narration │ A warm afternoon by the water with nowhere at   │ ×  │
│                  │ all to be.                                      │    │
│ [0:00]           │ This is Julia                                   │ ×  │  ← a note: no tag
│ [0:04] narration │ The light came off the surface of it in long    │ ×  │
│                  │ sheets.                                         │    │
│ [0:09] narration │ We stayed there until the whole day went cold.  │ ×  │
│ [0:10]           │ [glitch] here                                   │ ×  │
│                                                                         │
│ ▸ Use [brackets] to direct it                                           │
│ [0:07] │ Who is this? What happened here?                       │ [Add]   │
└─────────────────────────────────────────────────────────────────────────┘
  [m:ss] seeks the film to where that passage is spoken.

After one change (state "draft", changed), and a save that another window beat (409):
┌ Tell the story better ───────────────────────────────────────────── 5 ▾ ┐
│ Your edited narration. ↻ Regenerate speaks it as written.               │
│ Notes cannot rewrite these words or provide a new voice or music        │
│ interpretation. Existing voice choices still pass current note gating   │
│ and validation. Effect instructions reach the conductor; matched speech │
│ fixes shot timing.                                [Write a new story]   │
│ ! Not saved: this narration changed in another window, or a new film    │
│   replaced it. Copy any text you want to keep, then load the current    │
│   narration.                              [Load the current narration]  │
│ [0:00] narration │ A warm afternoon by the water with nowhere at   │ ×  │  ← read-only now
│                  │ all to be.                                      │    │
│   …                                                                     │
└─────────────────────────────────────────────────────────────────────────┘
  The header's ↻ Regenerate is lit (is-stale) from the page load on, because the draft is changed.

After ↻ Regenerate made a film from that draft (state "draft", not changed):
┌ Tell the story better ───────────────────────────────────────────── 5 ▾ ┐
│ This film speaks your edited narration. ↻ Regenerate speaks it again as │
│ written.                                                                │
│ Notes cannot rewrite these words or provide a new voice or music        │
│ interpretation. … matched speech fixes shot timing. [Write a new story] │
│ [0:00] narration │ A warm afternoon by the water with nowhere at   │ ×  │
│   …                                                                     │
└─────────────────────────────────────────────────────────────────────────┘
  ↻ Regenerate is not lit at load; the times seek in the new film.

A film made before Phase 7 (state "legacy"):
┌ Tell the story better ───────────────────────────────────────────── 2 ▾ ┐
│ This older film has no timed narration text. ↻ Regenerate makes a new   │
│ film whose narration you can edit here.                                 │
│ [0:00]           │ This is Julia                                   │ ×  │
│   …                                                                     │
└─────────────────────────────────────────────────────────────────────────┘
```

**F. Retell and the other resets** (lane 40a for the store, lane 40c for the card; D118)
- ↻ Retell memory stays the reset of every language: `rewrite_memory` (`albums/memory_store.py:82-104`) also runs `DELETE FROM
  narration_drafts WHERE group_id = ?` beside `:103`. The card says so (D). A failed retelling writes nothing, so its drafts stay.
- Rebuild memories (`replace_memories`): the group rows go, and the drafts go with them by the cascade.
- "Write a new story" on `/feed` (`discard_draft`) is the one reset of one draft. After it, the rows are the ready film's own (state
  `film`) when a ready film exists; after a failed build there is none (`feed/build.py:853-878`), so the state is `none`. The next
  Prepare or ↻ Regenerate runs the story pass.
- Not a reset: a narrator change, a note added, edited or deleted, the other language, a saved voice deleted or baked again, a library
  change. A photo deleted from the library is handled by C.2.
- A successful film made from a draft rebases it (C.6): the rows stay a draft, not `changed`, with that film's times, and every later
  ↻ Regenerate speaks the same words until "Write a new story". A failed or cancelled build keeps the draft as it is (C.7).

**G. Not changed by A-H:** `feed/story.py`, `feed/director.py`, `feed/notes.py`, `feed/store.py`, `feed/voice.py`,
`feed/saved_voices.py`, the note routes, `notes.MAX_LENGTH`, `WORDS_PER_NOTE`, `_bracketed_passages`, `_passage_voices`,
`web/static/feed-gl.js`, the narrator picker's options, the models service, `tests/test_migrate.py` (D39). §2.19 I changes
`feed/story.py`, `feed/director.py` and `feed/build.py` (lane 39b); `feed/narration.py` imports no private name of them.

**H. Docs first (lane 39).** Anchors read 2026-11-03; they move while the t60 and t61 fixers run.
- `docs/design.md` §6: the `memory_presentations` bullet (`:937-943`): the `Edit` JSON also carries the film's `passages` (text, time,
  photograph and voice per passage, in telling order) and its `script` (the token and the replay inputs); a new **`narration_drafts`**
  bullet after `memory_narrators` (`:991-998`): one edited narration per memory and language, seeded with all passages and every replay
  input at the first change, its own `draft_token`, a revision, the source film's token, the cascade; rebased (never deleted) by the
  ready save of a film made from it; removed by "Write a new story" and by a retelling (**changed, t63**).
- `docs/design.md` §11.1:
  - the film mermaid (`:2087-2102`), node 3 (`:2092`): add the line "an edited narration (narration_drafts): NO model call — its words are
    the script"; node 6 (`:2098`): add "timeline.passages + script · a used draft is rebased in the same transaction" (**changed, t63**);
  - a new paragraph "**Edited narration is spoken as written.**" after the "A model CASTS the film" paragraphs: C.1-C.7 in prose, with the
    narrator rules of C.4 and what a note can still do;
  - the retell paragraph (`:2331-2335`): a retelling also deletes the memory's edited narration, and the card says so.
- `docs/design.md` §11.3: the panel paragraph (`:2546-2549`): the panel also lists the film's narration as rows tagged "narration"; a new
  paragraph after `:2567-2572` on the narration rows (E, the stale rule, the bracket refusal, no cut, no bubble, never a note; changed
  and unchanged drafts, "Write a new story", the save barrier before Prepare and ↻ Regenerate, and what a note can still do: the
  disclosure line of E, word for word); the "Editing a note to nothing deletes it" paragraph (`:2616-2618`): an emptied passage is
  removed, the last one never.
- `docs/design.md` §13: the `/feed` bullet (`:2800-2819`): "Tell the story better" lists the film's narration as editable timed rows;
  the memory card (`:2737-2744`): the retell text names the edited narration.
- `docs/data-model.md`: the `narration_drafts` DDL of B after `memory_narrators` (`:306-318`).
- `docs/ui.md`: the card's retell text (`:191-194`); a new sub-bullet **Tell the story better** under `/feed` after `:511-515`, with every
  text of D and E (the state lines, the draft twins of the picker line and the legend lead, the "Not started" line), the row look, the
  sort, the times, the save barrier, the refusal and network rules and the 409 freeze.
- The order-fix docs of §2.19 I.7 (design.md §11.1 `:1862-1864`, `:2338-2349`, mermaid node 3 `:2092`).
- Validate the edited §11.1 mermaid block with the script in `<task-skill>/references/mermaid.md`. It
  must print `OK`.
- Closing check (step 41): `grep -n "narration_drafts" docs/design.md docs/data-model.md` finds the §6 bullet and the DDL; `grep -n
  "timeline.passages\|Edited narration is spoken as written" docs/design.md` finds §11.1 and §11.3; `grep -rn "re-sorts\|out of
  sequence\|stable sort that leaves repeats\|_in_memory_order" docs/ feed/` finds nothing (**changed, t65**: "re-sorts the story pass"
  spans two lines of design.md, `:2343-2344`, so a line grep never matched it; today the new pattern hits design.md `:2343`, `:2344`,
  `:2348` and director.py `:257`, `:266`).

Rejected: a passage field on `Shot` (the writer's first view: it puts Phase 7 data into `Edit` and `director.py`, and before §2.19 I the
shots were in memory order); the passages as `memory_notes` (no language,
the 600 cut, the story pass rewrites them, `WORDS_PER_NOTE`); rows split out of the joined narration of an older film (invented times);
one row per edited passage (t62: one draft with all passages); a stage-2-only story pass over the draft (it picks photographs and cards
again; t62 skips `write_story` whole); a draft cleared when a build starts; **changed (t63):** a draft deleted by its film's ready save
(t62 point 5, superseded by t63 point 1: a later ↻ Regenerate would rewrite the user's words); a draft named by `(source_token,
revision)` alone (an Undo and a new seed of the same film repeat it); `id="narration"` for the JSON block (the audio element has it).

**I. Shot/audio order fix** (lane 39b; the user's "Fix in this task" in `## Decisions`, pair t64: `<base>/.pair/threads/t64/decision.md`,
`<base>/.pair/files/t64-merge-writer.md`, `t64-merge2-writer.md`, and `t64-merge3-writer.md`, which is the last word; D126-D130)
**Changed 2026-11-04 (pair t74):** `_every_photograph`, the coverage and order clauses of the stage-2 prompt, `order_fix`, I.3's corrective call and "retried" are gone; `_match_photographs` matches each sentence after stage 2, votes stand, `memory_order` is a diagnostic, and every photograph is best effort (design §11.1).

Goal: ONE order, the told sentence order, for the words, the `/speak` chunks, the shots, the passage voices, the bracket gate, the score
plan's `from`, the title cards, the conductor's briefs and `timeline.passages`. No code sorts passages, and nothing relabels a photograph.
The memory's photo order is asked for, checked, and corrected at most once; when the correction fails, the story's own match wins. A
passage's words and its photograph never separate.

Facts (working tree, read 2026-11-03):
- The defect has two parts. `with_script` (`feed/director.py:243-263`) re-sorts ONLY the shots (`_in_memory_order`, `:266-290`), while
  `/speak` gets `story.chunks` in told order (`feed/build.py:463-473`) and `with_narration_timing` (`feed/director.py:293-317`) pairs shot
  `i` with segment `i`. And the sort key `(photo, first)` (`:283-289`) moves every return next to its first appearance, so a legal story
  `[0, 1, 0]` shows `[0, 0, 1]`: photograph 0 under the words of passage 1.
- The passage voices (`_passage_voices`), the bracket gate (`_bracketed_passages`), the score plan and the cards stay in told order
  (`feed/build.py:372`, `:390-394`, `:521-524`; `feed/story.py:222-229`).
- `_voices` (`feed/story.py:872-884`) reads raw item `i`, while `_script` (`:887-939`) skips non-dict, invalid and duplicate items,
  inserts missing sentences and keeps the model's list order. So after any skip, voice `i` belongs to another sentence.
- Cards: the prompt asks for a "shot number" (`:1242-1243`), while in the script `shot` means a photograph number (`:1222`); `_cards` is
  bounded by the photo count (`:857`, `:979-997`), but `lines_for` uses the number as a shot position (`:222-229`).
- `_attempt` turns an exception or a non-dict into `payload = {}` (`:845-849`), and `_script` then returns `_even` (`:900`). `_even`
  (`:970-976`) is non-decreasing, so a fallback always looks "in order". **Changed (t65):** `[]` and an all-malformed list do NOT give
  `_even` today: the missing-sentence fill (`:928-933`) runs before the empty check (`:936-937`) and fills every sentence, and
  `_every_photograph` then repairs it, so 4 sentences and 3 photos give `[0, 1, 0, 2]` (in order, every photo shown) where `_even`
  gives `[0, 0, 1, 2]`. A garbage corrective answer would then pass every check of I.3.
- `_dress` (`:287-314`) asks stage 2 once, then once more when a bracket asked for a voice and none was cast (`:298-313`), and it may
  select that second answer because of its cast (`:312`). `_every_photograph` (`:942-967`) can itself introduce a photograph out of
  order: three photos and `[0, 0, 0, 1]` give `[0, 0, 2, 1]`.

1. **`feed/director.py`** — `with_script` keeps the given order:
   `shots = tuple(replace(edit.shots[index], line="") for index, _text in script if 0 <= index < len(edit.shots))`, with its docstring
   saying the story's sentence order is the film's order. `_in_memory_order` is deleted. `with_narration_timing` does not change.
   **Changed (t65):** `with_arrangement(edit, entries, *, keep_timing=False, keep_order=False)`: the order is positional
   (`list(range(len(edit.shots)))`) when `keep_order` OR `keep_timing`, else `_resolve_order` as today (`:341-343`); the times follow
   `keep_timing` alone (`:356-360`). Without it, the conductor's monotonic times stand, as today. Today a film whose segments do not
   pair takes `_resolve_order`, so the conductor may reorder its shots: shot numbers `[2, 4, 1, 3]` give `[1, 3, 0, 2]`.
2. **`feed/story.py`** — the parse (D128):
   - `_script(raw, shot_count, told) -> tuple[script, voices, matched]` parses each item once: its sentence number, its photograph and
     its voice. With `told`, the output is in told sentence order: sentence `i` is passage `i`. The first accepted duplicate of a
     sentence wins. No item accepted (`raw` not a list, `[]`, or every item malformed) → `_even`, with `matched` False; nothing is
     filled. Only when at least one item is accepted, a sentence no item gave takes the photograph of the sentence before it (the first
     one: the next accepted sentence's) and the voice "" (**changed, t65**). `_every_photograph` then runs as today; it changes
     photographs only, so the voices stay on their sentences. `matched` is True exactly when at least one item was accepted.
   - Without `told` (no story pass wrote prose), the items stay in their list order, each with its own voice, as today.
   - `_voices` is deleted. `_attempt` takes `script` and `voices` from the one parse, and `narration` stays the script joined.
   - Cards (D130): `STORY_SCHEMA` card items become `{"sentence": integer, "text": string}`, required `["sentence", "text"]`.
     `_cards(raw, len(script))`: 1-based sentence in, 0-based passage out, at most `MAX_TITLE_CARDS`, a number past the last passage
     dropped. The prompt line (`:1241-1245`) asks for `{"sentence": <1-based sentence number>, "text": "..."}` "on the sentence where the
     film turns".
   - The prompt: `_assemble` adds, right after "Use EVERY photograph at least once …" (`:1227-1228`): "Introduce the photographs in
     their numbered order: photograph N first appears only after photographs 1 to N-1 have; a return to an earlier one is always
     allowed."
   - `introduces_in_order(photos: Sequence[int]) -> bool` (pure, D127): `first = list(dict.fromkeys(photos))`; True when `first ==
     sorted(first)`.
   - `Story` gains two fields with defaults (D127): `memory_order: str = "kept"` ("kept" | "retried" | "broken") and `matched: bool =
     True` (False = the photographs came from `_even`, not from the model).
3. **`_dress`, the one bounded correction** (D129). After the voice-cast logic has chosen its Story (`:293-313`), on that Story:
   - in order (`introduces_in_order` of its photographs) → `memory_order = "kept"`, and no extra call;
   - out of order → ONE corrective stage-2 call: `_attempt(…, order_fix=True)` with the same told sentences, briefs, notes, directions
     and speakers. `order_fix` (beside `insist`, through `_prompt` and `_assemble`) puts first in the prompt: "CRITICAL: your previous
     answer introduced the photographs out of their numbered order. Give each sentence a photograph again, and introduce them in
     their numbered order; a return to an earlier one is always allowed."
   - The candidate is accepted only when: it did not raise and `matched` is True (not the `{}` fallback, not `_even`); it has the same
     passage count and the same texts as the chosen Story; every photo index is valid (`0 <= p < len(edit.shots)`); it shows every
     photograph the chosen Story shows; and `introduces_in_order` is True.
   - Accepted → `replace(story, script=tuple(zip(candidate photos, story texts)), memory_order="retried")`: ONLY its photo indexes go
     in. The chosen Story's texts, narration, narrator, voices, cards, score and music prompt stay, so a voice cast by the voice retry
     survives.
   - Otherwise (an exception, the fallback, a malformed answer, lost coverage, still out of order) → the chosen Story unchanged, with
     `memory_order = "broken"`, and a warning in the log.
   - No new tell, rewrite, expansion, translation, sort or relabel.
4. **The draft path** (§2.19 C.2): no model call and no relabel. The stored `photo_id`s stay on their passages after the missing-photo
   rule of C.2; `draft_story` sets `memory_order` to "kept" or "broken" from that mapping (D127). Removing the passage that introduced a
   photograph can make a later return to it the first appearance: the film then shows it out of the memory's order and records
   "broken". The words, their order and their photographs stay as the user left them.
5. **`feed/build.py`**: `decisions["story"]["memory_order"] = story.memory_order` on both paths (§2.19 C.3), and (**changed, t65**)
   the conductor call (`:559-573`) passes `keep_order=bool(story.script)` beside `keep_timing=aligned`. Nothing else changes.
   `with_script`, `_passages` (`:822-831`), the conductor's `shots` (`:566-567`), `_bracketed_passages` and `_passage_voices` already
   walk `story.script`, so they now share the shots' order.
6. **What the film then is.** Shot `i` shows `photo_ids[story.script[i][0]]` with passage `i`'s voice; the score turns at passage
   `from`; a card sits on its sentence's shot; `timeline.passages[i]` is that passage. A return stays where the story put it. This
   ORDER holds for every film, also when the segments do not pair (`keep_order`). Exact sync, where shot `i` starts on segment `i`,
   needs paired segments (`keep_timing`); without them the conductor's monotonic times stand and every passage time stays null
   (§2.19 A.2) (**changed, t65**).
7. **Docs first (lane 39)**, beside H:
   - `docs/design.md` §11.1 "THE PHOTO ORDER BELONGS TO THE MEMORY" (`:2338-2349`): drop "`director.with_script` re-sorts the story
     pass's passages so a photograph can never be introduced out of sequence" and "applied as a stable sort that leaves repeats where
     the story put them". New text: the passages are never re-sorted; the story's sentence order is the film's order for the words,
     the voice and the pictures; the memory's order is asked for in stage 2, checked, and corrected by ONE more stage-2 call that may
     change only the photographs; when that fails, the story's own match wins and the film records `memory_order` "broken"; a
     passage's words and its photograph never separate; an edited narration is only checked and recorded. **Changed (t65):** it also
     says that the conductor never reorders the shots of a film with a script, and it tells the kept order apart from exact sync,
     which needs paired segments. The new text uses none of the closing-grep phrases of H ("re-sorts", "out of sequence", "stable sort
     that leaves repeats", `_in_memory_order`).
   - The film mermaid node 3 (`:2092`): replace "passages ≥ photos · EVERY photo carries ≥ 1 passage" with "passages ≥ photos · EVERY
     photo carries ≥ 1 passage · told order = film order · memory order checked, corrected once".
   - §11.1 step 3 (`:1862-1864`): a title card names a sentence.
8. **Rejected** (t64): moving the words with their photograph into memory order (it undoes `_in_place`, `feed/story.py:639-672`, the
   note order, design.md:2440-2444, and "THE STORY … FINAL", `:1216-1218`); `_even` as the repair (it drops every photo choice of the
   film); the relabel `in_memory_order` (sigma) of merges 1-2 (it shows a photograph under words the model did not match it to, merge 3);
   cutting the narration WAV and joining it in memory order (it breaks the prosody and the segment times); taking the whole corrective
   answer (it could drop the voice retry's cast, merge 3 point 1).

## 3. Dependency step — `voxcpm==2.0.3` (main session; nothing is installed)

All work happens in `S=<base>/.scratch/deps-voxcpm/`. The repo does not change until the gate passes. Never run `uv sync`, `uv add` (it syncs), `uv pip install`,
`pip`, `make models` or `make up`. Never download weights. Never import `voxcpm` or `torch`.

1. Inputs:
   ```bash
   B=<base>; S=$B/.scratch/deps-voxcpm
   mkdir -p $S/proj $S/base $S/scripts $S/src
   cp $B/ivms777/pyproject.toml $B/ivms777/uv.lock $S/proj/; cp $B/ivms777/pyproject.toml $S/base/
   # $S/proj/pyproject.toml: add "voxcpm==2.0.3" as the last entry of [project.optional-dependencies].models
   # $S/proj/pyproject.toml: also add a [tool.uv] table with override-dependencies = ["funasr; sys_platform == 'never'"]
   #   (user decision, ## Decisions; it removes funasr → kaldiio)
   ```
2. Resolve with builds enabled (sdists such as `reverse-geocoder` build in uv's isolated build env; nothing goes into a venv).
   The two aarch64-apple-darwin compiles (proj and base) run with `MACOSX_DEPLOYMENT_TARGET=14.0`.
   torch>=2.13.0 ships only macosx_14_0 wheels, and at uv's 13.0 default the base fails the same way.
   ```bash
   cd $S/proj && UV_CACHE_DIR=$S/uv-cache uv lock > $S/lock.log 2>&1
   for p in aarch64-apple-darwin x86_64-unknown-linux-gnu; do
     for d in proj base; do
       ( cd $S/$d || exit 1
         if [ $p = aarch64-apple-darwin ]; then export MACOSX_DEPLOYMENT_TARGET=14.0; fi
         UV_CACHE_DIR=$S/uv-cache uv pip compile pyproject.toml --extra models \
           --python-version 3.12 --python-platform $p -o $S/$d.$p.txt > $S/$d.$p.log 2>&1 )
     done
   done
   ```
3. New packages = lines in `proj.<p>.txt` that are not in `base.<p>.txt`, per platform.
4. Python 3.12:
   - The resolve above is for 3.12.
   - Wheel coverage: `$S/scripts/wheels.py` reads `https://pypi.org/pypi/<name>/<version>/json` for every new package and marks
     "wheel for cp312/abi3/py3 on macOS arm64 / manylinux x86_64" or "sdist only (source build at install)".
   - Syntax: fetch the `voxcpm-2.0.3` wheel or sdist into `$S/src/` (data only) and run `uv run --no-project --python 3.12 python -I -m compileall -q $S/src/<dir>` — it compiles and runs nothing.
   - Runtime import on 3.12 is proven only in the user's stack (§6.1).
5. Licences: `$S/scripts/licences.py` reads `info.license_expression`, `info.license` and the `License ::` classifiers of every new package from the PyPI JSON API.
   It writes `$S/licences.md`. When the metadata is empty or custom, it reads the LICENSE file in the fetched wheel or sdist.
   - Pass: MIT, BSD-*, Apache-2.0, ISC, PSF-2.0, Zlib, Unlicense, CC0-1.0, 0BSD, BSL-1.0, HPND, MIT-CMU.
   - Anything else (MPL, LGPL, GPL, AGPL, SSPL, BUSL, any NC clause, proprietary, unknown) → **stop and list it for the user**. No silent acceptance.
   - Approved exceptions (user decision, ## Decisions; exact name, version and licence only):
     `soxr==1.1.0` (`LGPL-2.1-or-later`), `orjson==3.13.0` (`MPL-2.0 AND (Apache-2.0 OR MIT)`), `cffi==2.1.1` (`MIT-0`),
     `pyahocorasick==2.3.1` (`BSD-3-Clause and Public-Domain`), `torchcodec==0.17.0` (BSD + its bundled third-party licences).
   - `licences.py` reads the licence payloads of the exceptions from the exact wheels uv selects: cp312/abi3/py3 on macOS arm64 and on manylinux x86_64.
     It writes each exception and its obligations to `$S/exceptions.md`.
   - The torchcodec 0.17.0 manylinux wheel bundles `libnvjpeg` under the NVIDIA CUDA EULA (`third_party/LICENSE.libnvjpeg-NVIDIA-CUDA-EULA.txt`).
     Pair decision t14: it is a proprietary NVIDIA redistributable that allows commercial use under that EULA (§1.1.1; obligations §1.1.2; supplement §2.1/§2.6).
     It is part of the approved torchcodec exception. Never call it permissive. No new user approval is claimed.
     `docs/models.md` and the PR licence table list it. The hand-over (§4 step 11) names it in one line, so the user can object at the commit gate.
   - `kaldiio` is never approved. Any other non-pass licence still stops.
   - Weights: Apache-2.0 (findings [A21]). Packages already in the baseline are not re-judged here (Silero and MusicGen weights: `## Open items outside scope`).
6. Gate: PASS = `uv lock` and all four compiles exit 0, and every new package passes step 5.
   - PASS also needs `funasr` and `kaldiio` absent from both `proj.*.txt`. In `proj/uv.lock`, uv's universal lock keeps both entries;
     the only edge to `funasr` must be voxcpm's, with the marker `sys_platform == 'never'`, and the only edge to `kaldiio` must be funasr's.
   - PASS also needs the remaining STOP set to be exactly the five approved names and versions of step 5.
   - The torchcodec `libnvjpeg` payload (NVIDIA CUDA EULA) is part of the approved torchcodec exception (step 5, pair t14). It is not a FAIL.
   - FAIL → stop. Report uv's own message, or the licence list, to the user.
   - Never relax an ivms777 pin (`torch>=2.13.0`, `transformers>=5.15.0`, `numpy>=2.5.2`, `pyproject.toml:17`, `:31-32`).
   - Never add `override-dependencies`, `--prerelease`, or another engine without the user.
     The funasr override of step 1 is user-approved (## Decisions, 2026-11-01).
7. Apply (after PASS): in `<base>/ivms777/pyproject.toml`, add `"voxcpm==2.0.3",` with a two-line comment (§4.1 voice copy; pinned; mac/cloud) after the `en-core-web-sm` entry.
   Also add a new `[tool.uv]` table (the file has none) with `override-dependencies = ["funasr; sys_platform == 'never'"]`
   and a one-line comment: kaldiio (via funasr) has an evaluation-only licence; voxcpm 2.0.3 never imports funasr.
   Run `cd <base>/ivms777 && uv lock` (no sync). `cmp <base>/ivms777/uv.lock $S/proj/uv.lock` must report no difference. Keep `$S/*.md` for the PR text and `docs/models.md`.
   Applied (2026-11-01): `<base>/ivms777/pyproject.toml` has `"voxcpm==2.0.3"` and the `[tool.uv]` funasr override; `uv lock` exit 0;
   `cmp` with `$S/proj/uv.lock` reports no difference.
8. Phase 5 (2026-11-02): no new package, so no gate run. Every import of `voxcpm.training` and of the trainer is locked already: `datasets` 3.6.0,
   `argbind` 0.3.9, `pyyaml` 6.0.3, `soundfile` 0.14.0, `safetensors` 0.8.0, `librosa` 1.0.0 (`uv.lock`, §1.8), and each is in
   `$S/proj.aarch64-apple-darwin-14.txt` and `$S/proj.x86_64-unknown-linux-gnu.txt` (read 2026-11-02). The trainer uses no TensorBoard.
   `tests/test_voxcpm_trainer.py` pins its imports. A change that needs another package stops and runs steps 1-7 for it.

## 4. Implementation order and lanes

Lanes are builder lanes (tier builder). Each lane owns only its listed files, reads each file's `git diff` first (§1.5), runs its targeted tests plus
`uv run ruff check <its files>`, and reports `GREEN` or `RED: <reason>`. No lane commits, calls a running service, or loads a model.
Contracts between lanes are §2.2-§2.8. A lane that needs a contract change stops and reports.
Note (2026-11-01): the lanes docs-engine, 3a, 3b and 8 (store) ran in parallel (no shared files). The docs of each phase still land in the same change before the hand-over.

### Phase 1 — engine

| Step | Lane | Files | After |
|---|---|---|---|
| 1 | main: deps | `<base>/.scratch/deps-voxcpm/*`; then `pyproject.toml`, `uv.lock` | — (stop on FAIL) |
| 2 | docs-engine | `docs/design.md` §4.1, §5.1, §8.1 only; `docs/models.md` | 1 |
| 3a | catalog-routing | `models/catalog.py`, `modelsvc/slots.py`, `modelsvc/downloads.py`, `scripts/fetch_models.py`; `tests/test_catalog.py`, `tests/test_modelsvc_slots.py`, `tests/test_downloads.py`, `tests/test_fetch_models.py` | 2 |
| 3b | child | `audio/voxcpm_voice.py` (new), `tests/test_voxcpm_voice.py` (new) | 2 (parallel with 3a) |
| 4a | speak-wire | `modelsvc/app.py`, `modelsvc/backends/base.py`, `modelsvc/backends/composite.py`, `modelsvc/backends/voice_backend.py`, `modelsvc/backends/fake.py`, `inference/models_client.py`; `tests/test_modelsvc_api.py`, `tests/test_modelsvc_backends.py`, `tests/test_models_client.py` | 3a |
| 4b | check-script | `scripts/voice_check.py` (new), `tests/test_voice_check.py` (new) | 3a (parallel with 4a) |
| 5 | main | `uv run pytest -q`, `uv run ruff check .` (one runner each); fixer lane on RED | 3b, 4a, 4b |
| 6 | main | no stop (pair t14, D32): phase 2 starts; §6.1 moves to the one hand-over at step 11 | 5 |

A §6.1 FAIL (thresholds §6.1 step 8) stops the task for the user's decision. Report the report.json numbers to the user. No other engine and no workaround without the user. A re-plan goes through the planner.

### Phase 2 — product

| Step | Lane | Files | After |
|---|---|---|---|
| 7 | docs-product | `docs/design.md` §5 (+ mermaid), §6, §11.1 (+ mermaid), §11.3, §13, §13.1; `docs/data-model.md`; `docs/ui.md`; `docs/ingest-pipeline.md` (`:111-115`); mermaid validation | 6 |
| 8 | store | `db/schema.sql` (both outbox tables, `AUTOINCREMENT` ids), `db/connection.py` (`:8` only), `config.py`, `web/deps.py`, `feed/saved_voices.py` (new: store, WAV check, write-ahead file writes, enqueue, `process_voice_deletions`), `tests/test_saved_voices.py` (new) | 7 |
| 9a | casting | `feed/voice.py`, `feed/build.py` (incl. the audio writes moved to "saving the film" and the ready-save check); `tests/test_feed_voice.py`, `tests/test_feed_build.py` | 8 |
| 9b | web | `web/voices_api.py` (new; routes + `VoiceDeletionDrain`), `web/app.py`, `web/templates/voices.html` (new), `web/templates/_voices_list.html` (new), `web/templates/organize.html`, `web/templates/feed.html`, `web/static/voices.js` (new), `web/static/app.css`; `tests/test_web_voices.py` (new), `tests/test_web_feed.py` | 8 (parallel with 9a); the `voices=ctx.voices` line and the `voice_drain.wake()` call in `_start_film` land after 9a is GREEN |
| 9c | worker | `ingest/pipeline.py` (group 1 only, `:50-62`), `tests/test_pipeline_pass.py` (append). `ingest/cli.py` does not change. | 8 (parallel with 9a, 9b) |
| 10 | main | full suite + ruff; mermaid check of `docs/design.md`; docs-vs-code check of every §2.14 item; `sadensmol-code-review` over the diff, every P0/P1 fixed | 9a, 9b, 9c |
| 11 | **user** | one hand-over: §6.1, then §6.2. Text: "implemented, offline tests green, ready for your testing — run §6.1 then §6.2". One line names the torchcodec `libnvjpeg` (NVIDIA CUDA EULA, §3 step 5). No claim about GPU quality or speed before the user's results (`ivms777/CLAUDE.md:223-227`) | 10 — **stop until the user reports**; a §6.1 FAIL stops the task for the user's decision |
| 12 | main | commit (own hunks only), push, PR, Linear comment (§7) | 11 PASS (§6.1 and §6.2); **changed 2026-11-02:** also step 17 PASS (§6.3). Phase 3 ships in the same PR. **Changed 2026-11-02 (Phase 4):** also step 22 PASS (§6.4); Phase 4 ships in the same PR. **Changed 2026-11-02 (Phase 5):** also steps 29a, 29b and 34 PASS; Phase 5 ships in the same PR (D66). **Changed 2026-11-02 (rework, t39):** a 29a FAIL goes to the user's decision (§7 item 5); no one-take path is added to make it pass |

### Phase 3 — organize and film clarity (§2.15; added 2026-11-02, pair t17 + t18)

Starts only after the user approves this amendment (gate). Same lane rules as above (§1.5, the `git diff` read first, targeted tests + ruff, `GREEN`/`RED`).

| Step | Lane | Files | After |
|---|---|---|---|
| 13 | docs-ux | `docs/design.md` §6 (`:763`, `:772-776`), §11.1 (`:1751-1752`, `:1769-1784` mermaid, `:1786-1799`, `:1801-1826`, `:1846-1861`), §11.3 (`:2150`), §13 (`:2233-2254`), §13.1 (`:2344-2346`); `docs/ui.md` (`:165-177`, `:186`, `:189`, `:203-205`, `:206-230`); `docs/data-model.md` (`:174`, `:196-204`); the comment `web/app.py:506-507` only; mermaid validation (§2.14 Phase 3) | the gate |
| 14 | narrator-store | `db/schema.sql` (`memory_narrators` only), `db/connection.py` (`_ADDED_COLUMNS` `:62-69` only), `feed/saved_voices.py` (`NarratorChoice`, `read_narrator`, `set_narrator`; `RECORDING_PROMPTS` and `add_example(…, prompt_id=)`, §2.15 I), `feed/store.py` (`FilmState`, `FilmNarrator`, `film_states`, `film_narrator`), `feed/voice.py` (`VoiceLabel`, `voice_label`, docstring `:1-4`); `tests/test_saved_voices.py`, `tests/test_feed_store.py`, `tests/test_feed_voice.py` (append) | 13 |
| 15a | casting-preset | `feed/build.py`; `tests/test_feed_build.py` (append) | 14 |
| 15b | web-ux | `web/app.py` (`organize()` `:659-717`, `feed()` `:785-880`, `_narrator_picker` / `_store_narrator` `:882-940`), `web/templates/_album_card.html`, `web/templates/organize.html`, `web/templates/feed.html`, `web/static/app.css`, `web/static/feed.css`, `web/static/chat.js` (`:80-83` only), `web/templates/voices.html` (the recorder panel), `web/templates/_voices_list.html` (the tip only, t21), `web/static/voices.js` (the preparation panel and Esc), `web/voices_api.py` (the `prompt_id` field); `tests/test_web_feed.py` (append + the adapted tests of §5 Phase 3), `tests/test_web_chat.py` (append), `tests/test_web_voices.py` (append) | 14 (parallel with 15a) |
| 16 | main | **changed 2026-11-02 (t25, D56):** the targeted tests of the Phase 3 files and of their direct callers (`tests/test_feed_store.py`, `tests/test_feed_voice.py`, `tests/test_saved_voices.py`, `tests/test_feed_build.py`, `tests/test_feed_metadata.py`, `tests/test_web_feed.py`, `tests/test_web_chat.py`, `tests/test_web_voices.py`, `tests/test_web_organize.py`, `tests/test_migrate.py` run only) + `uv run ruff check <changed files>`; mermaid check of `docs/design.md`; docs-vs-code check of every §2.14 Phase 3 item. No `sadensmol-code-review` here: it runs only inside `task finish` | 15a, 15b |
| 17 | **user** | one hand-over: §6.3. Text: "implemented, offline tests green, ready for your testing — run §6.3". Also name any §6.1/§6.2 step the user has not reported yet. No claim about how the voices sound | 16 — **stop until the user reports** |

- Lane 15b uses only lane 14's names (`film_states`, `FilmState`, `film_narrator`, `FilmNarrator`, `NarratorChoice`, `set_narrator(…, preset_speaker=)`,
  the nested `RECORDING_PROMPTS`, `add_example(…, prompt_id=)`); it does not wait for 15a.
  Lane 15a writes the cast keys that lane 14's `film_narrator` reads (`narrator_name`, `narrator_label`, `narrator_chosen_by`, `passages[].used_label`, §2.15 A, E).
- No lane edits `tests/test_migrate.py` (§1.5 rule 4; D39), the user's hunks of §1.6, or `models/catalog.py`.

### Phase 4 — voice workbench (§2.16; added 2026-11-02, pair t24)

Starts only after the user approves this amendment and answers the "Use for films" question (`## Open items outside scope`). It does not wait for
§6.3 (t24). Same lane rules as above (§1.5, the `git diff` read first, targeted tests + ruff, `GREEN`/`RED`). File ownership is disjoint per lane.

| Step | Lane | Files | After |
|---|---|---|---|
| 18 | docs-workbench | `docs/design.md` §5 (`:510`, `:515`, `:529`, `:532` mermaid; `:546`), §6 (`:761-799`), §11.1 (`:1773-1776`, `:1786` mermaid, `:1834-1853` + the new "Testing a saved voice" paragraph), §13 (`:2267-2271`), §13.1 (`:2337-2339`, `:2388-2391`); `docs/ui.md` (`:203-237`); `docs/data-model.md` (`:185-194`, `:223-229`); mermaid validation (§2.14 Phase 4) | the gate |
| 19 | workbench-store | `db/schema.sql` (the two new tables and the `voice_file_deletions` comment only), `feed/saved_voices.py` (`FilmExample`, `film_example`, `set_film_example`, `castable`, `reference_for`, `VoicePreview`, `save_preview`, `previews`, `read_preview_audio`, the drain's key query and ref check, the docstring); `tests/test_saved_voices.py` (append) | 18 |
| 20a | casting-workbench | `feed/build.py` (`reference_for`, the `_saved_cast` texts, `voice_turn` with its two cancel checks and `finally` release, `voice_lock=` with one turn over the references read, the speak block and `_keep_conditioning`, §2.16 C); `tests/test_feed_build.py` (append) | 19 |
| 20b | preview-job | `feed/voice_preview.py` (new; `run_preview` with the `job_lock` publication barrier, `on_saved`, and the `cancelled()` check just before `client.speak`, outside the speak handler, §2.16 D), `tests/test_voice_preview.py` (new) | 20a |
| 20c | web-workbench | `web/voices_api.py` (`VoicePreviews` with the 32-hex `epoch`, `completed`, `job_lock=self.lock` and `on_saved`; `Cache-Control: no-store` on GET preview; the five routes; the list and page context), `web/app.py` (`:221-243` the `voice_calls` lock, `:1191-1195` the `voice_lock=` argument, `:961-976` and `:1009-1012` the texts only), `web/templates/_voices_list.html`, `web/templates/voices.html`, `web/templates/_voice_preview.html` (new), `web/static/voices.js` (`previewVerdict`, `completionKey`, `refreshLoop`, the generation and reconciliation rules, §2.16 G; changed 2026-11-02, t25 round 2), `web/static/app.css`; `tests/test_web_voices.py` (append + the adapted tests of §5 Phase 4), `tests/test_web_feed.py` (append) | 19 (parallel with 20a); the test routes, `_voice_preview.html`, the panel JS and the `voice_lock=` line land after 20a and 20b are GREEN |
| 21 | main | **changed 2026-11-02 (t25, D56):** the targeted tests of the Phase 4 files and of their direct callers (`tests/test_saved_voices.py`, `tests/test_voice_preview.py`, `tests/test_feed_build.py`, `tests/test_feed_metadata.py`, `tests/test_feed_voice.py`, `tests/test_pipeline_pass.py`, `tests/test_web_voices.py`, `tests/test_web_feed.py`, `tests/test_one_model_process_gate.py`, `tests/test_migrate.py` run only) + `uv run ruff check <changed files>`; mermaid check of `docs/design.md`; docs-vs-code check of every §2.14 Phase 4 item. No `sadensmol-code-review` here: it runs only inside `task finish` | 20a, 20b, 20c |
| 22 | **user** | one hand-over: §6.4. Text: "implemented, offline tests green, ready for your testing — run §6.4". Also name §6.3 when the user has not reported it yet. No claim about likeness or speed | 21 — **stop until the user reports** |

- Lane 20c uses only lane 19's names, and later 20a's `voice_turn` and 20b's `PreviewText`, `PreviewJob`, `PreviewFailed`, `preview_text`,
  `run_preview`, `PREVIEW_SENTENCES`, `CUSTOM_TEXT_MAX`, exactly as §2.16 B-D pins them.
- No lane edits `modelsvc/`, `audio/`, `models/catalog.py`, `inference/models_client.py`, `scripts/voice_check.py`, `db/connection.py` or
  `tests/test_migrate.py` (§2.16 A; D39).

### Phase 5 — baked voice models and the Voices page (§2.17; added 2026-11-02, pair t28 + t30)

Starts only after the user approves this amendment. Slice A and Slice V run side by side; Slice B waits for both user checks and the gate answer
(D66). Same lane rules as above (§1.5, the `git diff` read first, targeted tests + ruff, `GREEN`/`RED`). File ownership is disjoint per lane.
Tiers: builder lanes, main steps, user steps.
**Changed 2026-11-02 (rework, t39):** the amendment was approved; steps 23-28b are done (Slices A and V are implemented and green in the working
tree); steps 29a and 29b are with the user. Slice B (steps 30-34) is re-planned below. It starts after the user approves this rework and does
not wait for 29a or 29b. What the old rows 30-34 said is summarised below the table.

| Step | Lane | Files | After |
|---|---|---|---|
| 23 | docs-phase5 (A + V) | `docs/design.md` §4.1 (`:384-400`), §5 mermaid (`:508`, `:515`), §5.1 (`:620-658`), §8.1 (`:984-994`), §11.1 (`:1886`), §13 (`:2321-2346` + the `/voices` bullet), §13.1 (`:2400`, `:2411-2413`, `:2422-2425`, `:2462-2467`); `docs/models.md` (the new section before `:238`); `docs/ui.md` (`:176`, `:203-297`, `:345`, `:394-400`); mermaid validation (§2.14 Phase 5, Slices A and V) | the gate — **done 2026-11-02** |
| 24a | catalog-train (A) | `models/catalog.py` (`AdapterSpec`, `ModelEntry.adapter`, the voxcpm2 spec and its note, `adapter_entry_for`), `modelsvc/slots.py` (`TRAIN_UNIT`, `_register_voice_train` registering once, the LoRA arg of the voice child), `modelsvc/governor.py` (`lease`, `leased`, `UnitLeased`); `tests/test_catalog.py`, `tests/test_modelsvc_slots.py`, `tests/test_governor.py` (append) | 23 — **done** |
| 24b | trainer (A) | `audio/voxcpm_trainer.py` (new; text ids by `add_column`, `while True: yield from loader`), `tests/test_voxcpm_trainer.py` (new) | 23 (parallel with 24a) — **done** (with the t33 fix) |
| 24c | adapter-child (A) | `audio/voxcpm_voice.py` (the `lora` arg, `_use_adapter` with the names-and-shapes pre-check and the reset + cleared sha on any failure, `adapter_mismatch`, `tensor_mismatch`, `load_mismatch`, `use_clip`, the reply's `adapters`), `tests/test_voxcpm_voice.py` (append) | 23 (parallel with 24a, 24b) — **done** |
| 25 | bake-service (A) | `modelsvc/voice_bakes.py` (new; `VoiceGate`, `VoiceBakes`, `BakeNotOffered`, `BakeOnly`), `modelsvc/app.py` (the six routes, `VoiceAdapter`, `use_clip`, `adapters`, the 409s of speak and ensure, the 503), `modelsvc/backends/base.py`, `composite.py` (bake methods, the gate in `speak`, `model_ensure` checks, `model_unload` through the governor), `fake.py` (scripted bake), `backends/__init__.py` (build + `sweep`), `inference/models_client.py` (bake methods, adapter in references); `tests/test_voice_bakes.py` (new), `tests/test_modelsvc_api.py`, `tests/test_modelsvc_backends.py`, `tests/test_models_client.py` (append) | 24a, 24b, 24c — **done** (with the t32, t34, t35 and t37 fixes) |
| 26 | bake-check (A) | `scripts/voice_bake_check.py` (new), `tests/test_voice_bake_check.py` (new) | 25 — **done** (with the t38 fix) |
| 27 | voices-ui (V) | `web/templates/base.html`, `organize.html` (`:23-25`), `voices.html`, `voice.html` (new), `_voice_cards.html` (new), `_voices_list.html` → `_voice_body.html`, `_voice_preview.html`, `feed.html` (`:52`), `web/static/voices.js`, `web/static/app.css`, `web/voices_api.py`, `web/app.py` (`:1035-1039` texts only), `feed/build.py` (`:607-616` texts only); `tests/test_web_voices.py` (paths + the tests of §5 Phase 5 V), `tests/test_web_feed.py`, `tests/test_web_organize.py`, `tests/test_feed_build.py` (the three texts) | 23 (parallel with 24-26) — **done** (with the t36 fix) |
| 28a | main (A) | the targeted tests of the Slice A files and their direct callers (`tests/test_catalog.py`, `tests/test_modelsvc_slots.py`, `tests/test_governor.py`, `tests/test_scheduler.py`, `tests/test_registry.py`, `tests/test_voxcpm_trainer.py`, `tests/test_voxcpm_voice.py`, `tests/test_voice_bakes.py`, `tests/test_modelsvc_api.py`, `tests/test_modelsvc_backends.py`, `tests/test_models_client.py`, `tests/test_voice_bake_check.py`, `tests/test_voice_check.py`, `tests/test_downloads.py`, `tests/test_fetch_models.py`) + `uv run ruff check <changed files>`; mermaid check; docs-vs-code check of §2.14 Phase 5 Slice A (D56) | 26 — **done, GREEN** |
| 28b | main (V) | the targeted tests of the Slice V files and their direct callers (`tests/test_web_voices.py`, `tests/test_web_feed.py`, `tests/test_web_organize.py`, `tests/test_web_chat.py`, `tests/test_feed_build.py`, `tests/test_saved_voices.py`) + ruff; mermaid check; docs-vs-code check of §2.14 Phase 5 Slice V (D56) | 27 — **done, GREEN** |
| 29a | **user** | hand-over: §6.5. Text: "Slice A is implemented, offline tests green — run §6.5 (the bake check on this Mac)". No claim about time, memory or likeness | 28a — **pending with the user.** **Changed (rework, t39):** it no longer stops Slice B. A FAIL means saved voices cannot narrate on this Mac: the main session reports it as a blocker for the user's decision and never adds a one-take path |
| 29b | **user** | hand-over: §6.6. Text: "Slice V is implemented, offline tests green — run §6.6 (the Voices page)" | 28b — **pending with the user**; steps 2, 3, 9 and 10 of §6.6 move to §6.7 (§6.6 note) |
| 30 | docs-bake (B) | `docs/design.md` §5 mermaid (`:538`, `:543`, `:557`, `:560`), §5.1 (`:673-698`, `:708-717`), §6 (`:897-929`), §11.1 (`:1960-1974`, the film mermaid `:1984`, `:1989`, `:2026-2075` + the new "Baking a saved voice" paragraph), §13 (`:2521-2545`), §13.1 (`:2655-2665`, a check only); `docs/ui.md` (`:201-345`, `:376-410`); `docs/data-model.md` (`:180-216`); `docs/models.md` (`:239-370`); mermaid validation (§2.14 Phase 5 Slice B, rewritten) | the gate of this rework — **done** in the working tree, with the t41 fixes; its K2 texts are reworked by 30b |
| 30b | docs-asr (B) — **added 2026-11-02 (ASR, t42)** | `docs/design.md` §4.1 (after `:403-420`), §5 mermaid (`:540`, `:560`), §5.1 (`:766`, `:784-797` + the `/voice/transcribe` bullet), §6 (`:905-913`), §8.1 (after `:1161-1195`), §11.1 (`:2096-2101`, `:2124-2154`), §13 (`:2612-2632`), §13.1 (`:2700`); `docs/ui.md` (`:229-363`, the items of §2.14 "Slice B, ASR rework"); `docs/data-model.md` (`:174-186`, `:238`); `docs/models.md` (`:9-44`, `:239-395` + the new Whisper section before `:396`, `:499-503`); mermaid validation; the closing `grep` of §2.14 | the gate of the ASR amendment |
| 31 | bake-store (B) | `db/schema.sql` (the K2 columns in the `voice_examples` CREATE; `voice_bake_runs`, `voice_models`, `voice_model_takes` with `language` and `seconds`, `voice_tests`; the "no longer used" comments on `voice_conditionings`, `voice_film_examples`, `voice_previews`; the `voice_file_deletions` comment), `db/connection.py` (`_ADDED_COLUMNS` only), `feed/saved_voices.py` (K1 kit, K2 `add_example(…, transcript=)` and `set_transcript`, the K3 API with `film_model`, `castable_models`, `model_reference`, `save_test`, `voice_test`, `read_test_audio`, `request_bake_cancel`, `BakeCancelled`, the drain; **changed, t40: it only ADDS — every name that `feed/build.py`, `feed/voice_preview.py`, `web/app.py` and `web/voices_api.py` import stays, and `castable` keeps its take pairs**; `voice_bake_runs.cancel_requested_at`), `scripts/voice_bake_check.py` (`speak_variants` takes the newest take, its imports and docstring; D82); `tests/test_saved_voices.py`, `tests/test_voice_bake_check.py` (append and adapt) | 30 — **done in the working tree** (`<base>/.pair/files/lane31-report.md`: its checks were green before the user's "stop tests"; no test has run since; its K2 parts are frozen as typed at t42 and reworked by 31c; its five builder decisions are D96-D100) |
| 31c | asr-store (B) — **added 2026-11-02 (ASR, t42)** | `db/schema.sql` (the `voice_examples` CREATE: `transcript_model`, `transcript_error`, the comments of K2b; `transcript_confirmed_at` stays), `db/connection.py` (`_ADDED_COLUMNS` only: the two new entries; the lane-31 entries stay), `feed/saved_voices.py` (K2b: `add_example` without `transcript=` and with `kit_id` NULL for no `prompt_id`; `VoiceExample` with `transcript_model`, `transcript_error`, `words_state`; `heard_words`, `takes_to_hear`, `set_heard_words`, `NO_WORDS`, `TOO_LONG`; **changed (ASR plan check, t44):** digits count in `heard_words`, plus `take_to_hear` and `not_used_text`, and the state `unusable`; `BakeSnapshot` with `waiting` and `unusable`; `ModelChanges(added, deleted, unusable)`; `voice_model_for`; remove `set_transcript`, `unresolved` and `edited` after a `grep -rn` over `feed web scripts tests` finds no user outside this file and `tests/test_saved_voices.py`, else RED), `scripts/voice_bake_check.py` (the `--list` labels, D104); `tests/test_saved_voices.py`, `tests/test_voice_bake_check.py` (adapt: the K2 tests of lane 31 are replaced by those of §5 "Slice B, ASR") | 30b |
| 31b | speak-model-only (B) | `modelsvc/app.py` (`VoiceReference.audio` / `sha256` optional, `_decode_references`; §2.17 F), `inference/models_client.py` (`_reference_payload`; §2.17 G), **changed, t40:** `modelsvc/backends/fake.py` (`FakeBackend.speak` for a `use_clip=False` token: no conditioning, `adapters[token] = {"state": "loaded", "sha256": <sent>}`); `tests/test_modelsvc_api.py`, `tests/test_models_client.py` (with the HTTP test with no audio), `tests/test_voxcpm_voice.py`, `tests/test_modelsvc_backends.py` (append) | 30, AND the user's §6.5 report (any result) or the user's answer at step 33a (D84) |
| 31d | asr-service (B) — **added 2026-11-02 (ASR, t42)** | `models/catalog.py` (`TRANSCRIBERS`, `transcriber_for`; §2.17 B), `modelsvc/slots.py` (`ASR_UNIT`, `_register_asr` once), `audio/whisper_asr.py` (new; K2c), `modelsvc/voice_bakes.py` (`governor.unload(ASR_UNIT)` in "waiting for the GPU"), `modelsvc/backends/base.py` (`transcribe`, `TranscribeNotOffered`, `TranscribeRefused`), `modelsvc/backends/composite.py` (`transcribe`, the gate on `asr` ensures, the D103 log line), `modelsvc/backends/fake.py` (`transcribe`), `modelsvc/app.py` (`POST /voice/transcribe`), `inference/models_client.py` (`transcribe`), `scripts/fetch_models.py` (`wanted`); `tests/test_catalog.py`, `tests/test_modelsvc_slots.py`, `tests/test_whisper_asr.py` (new), `tests/test_voice_bakes.py`, `tests/test_modelsvc_backends.py`, `tests/test_modelsvc_api.py`, `tests/test_models_client.py`, `tests/test_fetch_models.py` (append) | 30b and 31b GREEN (it shares `modelsvc/app.py`, `inference/models_client.py`, `fake.py` and their tests with 31b), AND the same §6.5 report or 33a answer (D84) |
| 32a | bake-job (B) | `feed/voice_bake.py` (new; the immutable snapshot, the tag with `source: "app"`, `BakeBusy` and `wait_for_other`, any end state ends the wait, the `job_lock` publication barrier, `bake_delete` in `finally`; **changed, t40:** the new start and the attach split before any take check, and `BakeCancelled` → `Cancelled`), `tests/test_voice_bake.py` (new); **changed (ASR, t42):** the new start of §2.17 K4 "Changed (ASR, t42)": the foreign-bake check before the stage, "transcribing the takes" through `client.transcribe` and `set_heard_words`, the hand-off to `start_bake_run` under `job_lock`, the clips with the stored heard words, the new progress split, the new `BakeBlocked` texts (D92, D94, D95); **changed (ASR plan check, t44):** the stage repeats until the snapshot read under `job_lock` finds no `waiting` take (never holding `job_lock` across HTTP), `take_to_hear` before each audio read and after a failed read, the per-take reasons of `not_used_text` | 31, **and 31c (ASR, t42)** |
| 32b | casting-model (B) | `feed/build.py` (`_saved_cast` through `film_model`, the model references, the D86 reply check, `decisions["voice"]["models"]`, `cast.narrator_model`, the ready save on voices only; `_keep_conditioning` goes), `feed/voice_preview.py` (one test per person: `PreviewText`, `PreviewJob`, `preview_text`, `run_preview`), `feed/store.py` (`FilmNarrator.model`, `film_narrator`); `tests/test_feed_build.py`, `tests/test_voice_preview.py`, `tests/test_feed_store.py` (append and adapt); **changed (ASR, t42):** `preview_text`'s `take` kind takes `heard` takes only, with the K5 texts | 31 (parallel with 32a); **changed (ASR, t42):** the casting work may run now; the `take` branch, its texts and its tests wait for 31c GREEN |
| 32c | web-bake (B) | `web/voices_api.py` (`VoiceBakeJobs` with the job lock on cancel, replace and delete, the reattach by `source` and `bake_key`, the bake, transcript and test routes, the removed routes, `BAKE_NOT_OFFERED`, the leaf and card context), `web/app.py` (`_narrator_picker`, `_store_narrator`, and the "This film" context of `feed()` only), `web/templates/voice.html`, `_voice_body.html`, `_voice_cards.html`, `_voice_preview.html`, `_voice_bake.html` (new), `feed.html` (the picker group, the link line, "This film"), `web/static/voices.js`, `web/static/app.css`; `tests/test_web_voices.py`, `tests/test_web_feed.py` (append and adapt). **Changed, t40:** `VoiceBakeJobs` writes `request_bake_cancel` under the job lock on Cancel, replace and voice delete before it answers; `reattach` cleans up a run with the flag; 32c imports none of the names lane 32d removes, in its code or its tests. **Changed (ASR, t42):** no words box, no transcript route, the add route posts no `transcript`; the K6 "Changed (ASR, t42)" texts (the not-used row text, "Other takes", the recorder line, the Voice model lines and states, the transcription stage in the job line, the meter, "Your words in take #N") | 31 (parallel with 32a, 32b); **changed, t40:** the bake routes and the bake JS switch on only after BOTH 32a and 32b are GREEN; the test route and the Test tab JS after 32b is GREEN. **Changed (ASR, t42):** the lane starts after 31c GREEN; the bake routes and the bake JS switch on only after 32a, 32b AND 31d are GREEN (before 31d the client has no `transcribe`, so a Bake could only fail) |
| 33a | **user** (only when needed) | when 31b has not started because §6.5 has no report yet. Text: "Slice B is ready except one models-service change (the `/speak` reference without a take). It reloads the models service, and a reload ends a running bake. Tell me when no bake runs." **Changed (ASR, t42):** when 31b or 31d has not started. Text: "Slice B is ready except two models-service changes: the `/speak` reference without a take, and the speech recognition that hears your takes before a bake. They reload the models service, and a reload ends a running bake. Tell me when no bake runs." | 31, 32a, 32b, 32c, 32d GREEN — **stop until the user answers**. **Changed (ASR, t42):** also 31c GREEN; 32c without its bake parts |
| 32d | store-cleanup (B) — **added 2026-11-02 (rework, t40)** | `feed/saved_voices.py` (remove `FilmExample`, `film_example`, `set_film_example`, `castable`, `save_conditioning`, `VoicePreview`, `save_preview`, `previews`, `read_preview_audio`; keep `example_for`, `conditionings`, `reference_for` for the check script, D82), `tests/test_saved_voices.py` (remove the tests of those names; add the `hasattr` test of §5). Before it edits, `grep -rn` over `feed web scripts tests` finds no import of a removed name outside these two files; a hit stops the lane with RED | 32b and 32c GREEN. **Changed (ASR, t42):** 32c GREEN without its bake parts is enough (they import no name 32d removes) |
| 33 | main (B) | the targeted tests of the Slice B files and their direct callers (`tests/test_saved_voices.py`, `tests/test_voice_bake_check.py`, `tests/test_voice_bake.py`, `tests/test_feed_build.py`, `tests/test_feed_metadata.py`, `tests/test_feed_store.py`, `tests/test_voice_preview.py`, `tests/test_web_voices.py`, `tests/test_web_feed.py`, `tests/test_modelsvc_api.py`, `tests/test_modelsvc_backends.py`, `tests/test_models_client.py`, `tests/test_voxcpm_voice.py`, `tests/test_pipeline_pass.py`, `tests/test_one_model_process_gate.py`, `tests/test_migrate.py` run only) + `uv run ruff check <changed files>`; mermaid check; docs-vs-code check of §2.14 Phase 5 Slice B (D56). **Changed (ASR, t42):** also `tests/test_catalog.py`, `tests/test_modelsvc_slots.py`, `tests/test_whisper_asr.py`, `tests/test_voice_bakes.py`, `tests/test_fetch_models.py`, `tests/test_governor.py`; the docs-vs-code check also covers §2.14 "Slice B, ASR rework". The lane-31 tests that never ran after the user's "stop tests" run here too | 31, 31b, 32a, 32b, 32c, **32d (changed, t40)**; **changed (ASR, t42):** also 30b, 31c, 31d and 32c's bake parts |
| 34 | **user** | hand-over: §6.7. Text: "Slice B is implemented, offline tests green — run §6.7. A saved voice narrates a film only after you bake it in the app." Also name §6.5 and §6.6 when the user has not reported them. When §6.5 failed: one line that saved voices cannot narrate on this Mac until that is solved. No claim about likeness or speed. **Changed (ASR, t42):** one more line: "Run `make models` once first: it downloads the speech-recognition model (about 1.6 GB). You can record or upload anything; the bake hears your words." No claim about how well ASR hears or how much memory it takes | 33 — **stop until the user reports** |

- Lane 25 uses only 24a-24c's names exactly as §2.17 B-F pin them; lane 26 only lane 25's client methods. Lane 27 uses no Slice A name.
- Lanes 32b and 32c use only lane 31's names; 32c uses 32a's `run_bake`, `BakeBlocked`, `BakeFailed`, `BakeBusy` once 32a is GREEN, and 32b's
  `PreviewText`, `PreviewJob`, `PreviewFailed`, `preview_text`, `run_preview`, `PREVIEW_SENTENCES`, `CUSTOM_TEXT_MAX` once 32b is GREEN, exactly
  as §2.17 K4-K5 pin them.
- No Slice A or V lane edits `db/`, `feed/saved_voices.py`, `feed/voice_preview.py` or `tests/test_migrate.py`. No lane edits `scripts/voice_check.py`.
- **Changed 2026-11-02 (rework, t39):**
  - Lane 31 owns the store AND the check script, so the script never imports a name the store removed (D82).
  - **Changed 2026-11-02 (rework, t40):** `app` runs with `--reload`, so the tree must load after every lane. Lane 31 only adds names; lanes
    32b and 32c move their code and their tests off the K0 names; lane 32d removes those names last, and only after a `grep` finds no other
    importer (D82, D89). Lane 32c's bake routes and bake JS switch on only when 32a AND 32b are GREEN, so no bake can store a model while the
    film and test code still sends one take.
  - Lane 31b is the ONLY Slice B lane that edits a file under `modelsvc`, `audio`, `models`, `inference`, `embedding` or `captioning`, and it waits
    as its row says (D84). Lanes 32b and 32c build the model-only reference and test it against fakes, so they do not wait for 31b. The
    models service needs 31b before a real film or test can use the reference without a take, so step 33 and the hand-over wait for it.
  - Who waits for what: 30 → the gate of this rework; 31 → 30; 32a, 32b, 32c → 31; 31b → 30 and the §6.5 report or 33a; 33 → 31, 31b, 32a-c;
    34 → 33. 29a and 29b run beside all of it, at the user's pace. **Changed 2026-11-02 (rework, t40):** 32d → 32b and 32c GREEN; 33 → also 32d; 32c's bake parts
    → 32a and 32b GREEN.
  - **Changed 2026-11-02 (rework, t40):** lane 31b also owns `modelsvc/backends/fake.py` and `tests/test_modelsvc_backends.py`; `fake.py` is under `modelsvc`, so
    it waits with the rest of 31b (D84).
  - Lane 32c shares files only with lane 27, which is done; "32c starts only after 29b PASS" no longer applies.
  - No lane edits `tests/test_migrate.py` or `models/catalog.py`.
- **Changed 2026-11-02 (ASR, t42; D102):**
  - Order: the ASR gate → 30b (docs first) → 31c → 32a, 32b's `take` branch and 32c (32c's bake parts also wait for 31d) → 32d → 33a
    when needed → 31b → 31d → 32c's bake parts → 33 → 34. 32b's casting work and 31b do not wait for the ASR gate (Gate). 29a and 29b
    still run beside all of it.
  - Lanes 31b and 31d are the only Slice B lanes that edit a file under `modelsvc`, `audio`, `models`, `inference`, `embedding` or
    `captioning`. Both wait for the user's §6.5 report or the 33a answer (D84). They run one after the other in that window (31d shares
    `modelsvc/app.py`, `inference/models_client.py`, `modelsvc/backends/fake.py` and their tests with 31b), so the models service reloads
    twice and no bake may run during either.
  - "No lane edits `models/catalog.py`" no longer holds for 31d, which adds `TRANSCRIBERS` and `transcriber_for` only; `CATALOG`, `SLOTS`
    and `DEFAULTS` do not change. No lane edits `tests/test_migrate.py`.
  - File ownership is disjoint per running lane: 30b owns only `docs/`; 31c owns `db/schema.sql`, `db/connection.py`,
    `feed/saved_voices.py`, `scripts/voice_bake_check.py` and their two tests until it is GREEN, then 32d owns `feed/saved_voices.py` and
    `tests/test_saved_voices.py`; 32a, 32b and 32c keep their files; 31d owns the files of its row.
  - The app-side lanes import no new catalog name: the job takes the ASR identity from the transcribe reply (`model`, `revision`), and the
    page decides "bake offered" with `adapter_entry_for` as before. So `app` keeps loading under `--reload` while 31d waits (D102).
  - Lane 31c removes `set_transcript`, `VoiceExample.unresolved`, `transcript_confirmed_at` on the dataclass and `ModelChanges.edited` /
    `.unresolved` only after a `grep -rn` over `feed web scripts tests` finds no user outside `feed/saved_voices.py` and
    `tests/test_saved_voices.py` (§1.10 found none on 2026-11-02); a hit stops the lane with RED (D89's rule).

Superseded rows (kept for the record; **changed 2026-11-02, rework, t39**): old 30 waited for "29a PASS, 29b PASS, the gate answer on deleted
takes"; old 31 had `saved_voices.last_model_version`, `VoiceModelState` with `"deleting"`, `reference_for(…, model=)` and
`save_preview(…, model_take_ids=)`; old 32b had "the training-take check in the ready transaction" and `model_take_ids`; old 32c had no picker
and no test route. All are replaced by the rows above.

### Phase 6 — Automatic may pick a saved voice (§2.18; added 2026-11-03, pair t50)

Starts only after the user approves this amendment. Same lane rules as above. File ownership is disjoint per running lane (D111).

| Step | Lane | Files | After |
|---|---|---|---|
| 35 | docs-auto | `docs/design.md` §11.1 (the film mermaid nodes 3 and 4a, `:2101-2108`, `:2116-2124`, `:2129-2132`, `:2142`, `:2158`), §11.3 (the `voice` row `:2624`), §13 (`:2781`); `docs/ui.md` (`:462-471`, `:482`, `:512-513`, `:516-518`); mermaid validation (§2.18 E) | the gate of this amendment; the t50 builder GREEN and the t49 lane GREEN (both edit these two files) |
| 36a | auto-cast | `feed/voice.py` (`automatic_for_prompt`, the docstring), `feed/build.py` (`build_presentation`: `profile=`, the Automatic candidates, the narrator resolution, `saved_film` after it, `offered`, the `voices` docstring; `_passage_voices(…, saved=)`); `tests/test_feed_voice.py`, `tests/test_feed_build.py` (append and adapt) | 35; lane 32b, the t50 builder and the fixer of the 2026-11-03 bracket tests GREEN (they hold `feed/build.py` or `tests/test_feed_build.py`) |
| 36b | auto-web | `web/app.py` (the `build_presentation` call at `:1229-1233` only), `web/templates/feed.html` (the Automatic option, the always-hint, the Automatic bracket line); `tests/test_web_feed.py` (adapt and append) | 35 and 36a GREEN; lane 32c GREEN without its bake parts (they edit none of these three files, row 32c); the t50 picker texts GREEN |
| 37 | main (auto) | the targeted tests `tests/test_feed_voice.py`, `tests/test_feed_build.py`, `tests/test_feed_story.py`, `tests/test_feed_store.py`, `tests/test_web_feed.py`, `tests/test_one_model_process_gate.py` (run only) + `uv run ruff check <changed files>`; mermaid check; the closing `grep` of §2.18 E; docs-vs-code check of §2.18 (D56) | 36a, 36b. When step 33 has not run yet, step 33 also runs these files |
| 38 | **user** | hand-over: §6.8. Text: "Automatic can now pick one of your baked saved voices as the narrator — run §6.8 after §6.7." No claim about how often the story picks a saved voice. One message with step 34 when both are ready | 37 and 34 (§6.8 needs a saved-voice film to speak: lane 31b in the models service and a baked voice) — **stop until the user reports** |

- Lane 36a adds `profile` with the default `None`. Until 36b passes it, Automatic stays presets only, so `app` loads and behaves as before
  after each lane (`--reload`).
- No Phase 6 lane edits a file under `modelsvc`, `audio`, `models`, `inference`, `embedding` or `captioning` (no D84 wait), or
  `feed/story.py`, `feed/store.py`, `feed/saved_voices.py`, `web/voices_api.py`, `web/static/voices.js`, `web/templates/voice.html`,
  `tests/test_web_voices.py`, `docs/data-model.md` or `docs/models.md`.
- Who waits for what: 35 → the gate, t50, t49; 36a → 35, 32b, t50, the bracket-test fixer; 36b → 35, 36a, 32c (without its bake parts),
  t50's picker texts; 37 → 36a, 36b; 38 → 37 and 34.

### Phase 7 — editable narration (§2.19; added 2026-11-03, pair t62)

Starts only after the user approves this amendment. Same lane rules as above. File ownership is disjoint per running lane (D121).

| Step | Lane | Files | After |
|---|---|---|---|
| 39 | docs-narration | `docs/design.md` §6 (`:937-943`, the new `narration_drafts` bullet after `:991-998`), §11.1 (the film mermaid nodes 3 `:2092` and 6 `:2098`, the new "Edited narration is spoken as written" paragraph, the retell paragraph `:2331-2335`), §11.3 (`:2546-2549`, the new narration-rows paragraph after `:2567-2572`, `:2616-2618`), §13 (`:2737-2744`, `:2800-2819`); `docs/data-model.md` (the DDL after `:306-318`); `docs/ui.md` (`:191-194`, the new "Tell the story better" sub-bullet after `:511-515`); the order-fix text of §2.19 I.7 (§11.1 `:1862-1864`, `:2338-2349`, mermaid node 3 `:2092`); mermaid validation (§2.19 H) | the gate of this amendment; the t60 and t61 fixers GREEN (they hold `docs/ui.md` and `docs/design.md`) |
| 39b | story-order | `feed/story.py` (§2.19 I.2-3: `_script` → `(script, voices, matched)`, `_voices` deleted, `_cards` by sentence, `STORY_SCHEMA` cards, the `_assemble` rule and `order_fix`, `introduces_in_order`, `Story.memory_order` and `Story.matched`, the corrective call in `_dress`), `feed/director.py` (`with_script` keeps the given order; `_in_memory_order` deleted; `with_arrangement(…, keep_order=)`, t65), `feed/build.py` (`decisions["story"]["memory_order"]` and `keep_order=bool(story.script)` in the conductor call, §2.19 I.5); `tests/test_feed_story.py`, `tests/test_feed_director.py`, `tests/test_feed_build.py` (append and adapt; the card key of the fake at `:86`), `tests/test_web_feed.py` (the card key of the fake at `:64` ONLY; it runs `test_the_story_is_a_narration_with_sparse_title_cards`, `:408-413`) | 39; any lane that holds `feed/build.py`, `tests/test_feed_build.py` or `tests/test_web_feed.py` (row 36b) GREEN |
| 40a | narration-store | `db/schema.sql` (`narration_drafts` only), `feed/narration.py` (new: §2.19 A `timeline_script`, B, C.2 `draft_story`, B.7 `rebase_consumed`), `albums/memory_store.py` (`rewrite_memory`: one `DELETE`, §2.19 F); `tests/test_feed_narration.py` (new), `tests/test_albums.py` (append) | 39b (it imports `Story`, `introduces_in_order` and the new card shape); any lane that holds `db/schema.sql`, `albums/memory_store.py` or `tests/test_albums.py` (rows 31, 31c) GREEN |
| 40b | narration-build | `feed/build.py` (§2.19 A.4, C); `tests/test_feed_build.py` (append and adapt) | 40a and 39b (both edit `feed/build.py` and `tests/test_feed_build.py`); any lane that holds `feed/build.py` or `tests/test_feed_build.py` GREEN |
| 40c | narration-web | `web/app.py` (`feed()`'s context, the four routes of §2.19 D, `_film_block` and its caller `:747`), `web/templates/feed.html` (the `#narration-data` block, the state lines, the error line and its button, the two titles, the `data-narration` twins of the picker line and the legend lead, the prepare handler's `settled()` wait, §2.19 E), `web/static/feed-notes.js`, `web/static/feed.css`; `tests/test_web_feed.py` (append and adapt) | 40a (parallel with 40b); any lane that holds one of these files GREEN |
| 41 | main (narration) | the targeted tests `tests/test_feed_narration.py`, `tests/test_albums.py`, `tests/test_memories_build.py`, `tests/test_feed_build.py`, `tests/test_feed_story.py`, `tests/test_feed_store.py`, `tests/test_feed_notes.py`, `tests/test_feed_director.py`, `tests/test_feed_metadata.py`, `tests/test_web_feed.py`, `tests/test_web_organize.py`, `tests/test_db.py`, `tests/test_migrate.py` and `tests/test_one_model_process_gate.py` (the last two run only) + `uv run ruff check <changed files>`; mermaid check; the closing `grep` of §2.19 H; docs-vs-code check of §2.19, I included (D56) | 39b, 40a, 40b, 40c |
| 42 | **user** | hand-over: §6.9. Text: "Your film's narration now shows as timed rows in Tell the story better. Change the words, then ↻ Regenerate speaks them as written — run §6.9." One message with steps 34 and 38 when they are ready. No claim about how the voice sounds | 41 — **stop until the user reports** |

- Lane 40a uses only lane 39b's names (`Story.memory_order`, `introduces_in_order`, the card shape of §2.19 I.2) exactly as §2.19 I pins
  them. Lane 40b uses only lane 40a's names (`read_draft`, `draft_story`, `rebase_consumed`, `timeline_script`); lane 40c only `narration_rows`,
  `NarrationRows`, `save_passage`, `remove_passage`, `discard_draft`, `draft_languages`, `NarrationRefused`, `MAX_SCRIPT_CHARS`, exactly as
  §2.19 B pins them. Lane 40c's tests write presentations with `timeline.passages` themselves, so 40c does not wait for 40b.
- Until 40b lands, no film has `passages`, so the page shows the legacy line for every film; `app` loads and works after each lane
  (`--reload`).
- No Phase 7 lane edits a file under `modelsvc`, `audio`, `models`, `inference`, `embedding` or `captioning` (no D84 wait), or
  `feed/notes.py`, `feed/store.py`, `feed/voice.py`, `feed/saved_voices.py`, `web/voices_api.py`, `web/static/feed-gl.js`,
  `tests/test_migrate.py` or `docs/models.md`. Only lane 39b edits `feed/story.py` and `feed/director.py`. Lane 39b edits one line of
  `tests/test_web_feed.py` (`:64`) before lane 40c owns that file.
- Who waits for what: 39 → the gate, t60, t61; 39b → 39 and the holders of its files (36b); 40a → 39b and the holders of its files
  (31, 31c); 40b → 40a and 39b; 40c → 40a; 41 → 39b, 40a, 40b, 40c; 42 → 41 (**changed, t65**).
- **Changed 2026-11-03 (pair t64):** lane 39b is the order fix of §2.19 I. It lands before 40a and 40b, so lane 40a builds the
  timeline and the draft on the told order and the card-by-sentence shape (D121).

### Phase 8 — ask AI to rephrase the narration (added 2026-11-04, pair t73)

The user asked: "add in tell the story better chat field - so i can ask AI to rephrase it, when rephrasing it should have access to all the details used by original video composed, music, timeline etc."

What the viewer gets:

```
┌ Tell the story better ───────────────── 6 ┐
│ 0:00 narration [Это был день, когда…]   × │
│ 0:06 narration [Мурлыканье tabby…]      × │
│ …                                         │
│ ─ Ask AI to rephrase ───────────────────  │
│ you: сделай теплее, без слова tabby       │
│ AI:  Changed 2 of 6 passages.             │
│   0:06 old → new                          │
│   0:21 old → new                          │
│   [Use this] [Dismiss]                    │
│ Rephrasing… 40% (estimate)      [Cancel]  │
│ [ Ask for a change…               ] [Send]│
└───────────────────────────────────────────┘
```

- The AI proposes. Nothing changes until **Use this**. Then ↻ Regenerate speaks the new words as written, by the existing draft path.
- Rephrase changes only words. Photos, voices and timing of each passage stay.
- Limit: the app stores no snapshot of the build's inputs. `timeline` and `decisions` are the exact record of what was composed: shots, effects, cards, narrator, music prompt and sections. The title, description, notes and photo details are today's values, and the prompt labels them "as they are today".
- Limit: over the model's context, the request is refused with a reason. Nothing is cut.
- The conversation lives in the page session only. There is no chat table.

Agreed detail (pair t73):


- Job, not a sync POST: `/text/complete` is a sync route with no disconnect check (modelsvc/app.py:464-471), so an abort stops nothing. CLAUDE.md "Long jobs" needs stage, percent, Cancel. Pattern: VoicePreviews (web/voices_api.py:278-312): run counter, a new start replaces the old, cancel invalidates and never waits. Percent is a labelled time estimate that holds below 100 (`_Ticking`, feed/build.py:110-123). Stages: "reading the film", "writing new words", "checking the words".
- Routes: POST /feed/{key}/narration/rephrase (lang, source, draft, revision, request, history, proposal) -> 202 + job; GET .../rephrase/{job}; POST .../rephrase/{job}/cancel; POST /feed/{key}/narration/rewrite. No clash with `/narration/{n:int}` (web/app.py:1160). Planner slot (web/app.py:216), json_schema, timeout 300 s.
- Save barrier `settled()` (feed-notes.js:420) before Send AND before Use this. Rows, triple and context read together; no DB lock held during inference. Refused when frozen, legacy, none, or building/queued (web/app.py:946): legacy and none already fail `_check_fresh` (narration.py:287-289).
- Follow-up works on the displayed candidate, posted as `proposal`, labelled apart from saved rows. History bounded: last 6 messages; request <= 1000 chars.
- Context: your full list, with `beats`, `params`, photo anchors of notes. Automatic tags labelled as guesses (ShotSignal.brief, signals.py:55-56). No audio, no vector, no `effects`/`grades` tables (director.py:164-169). Film facts come only from the READY row of that language; a draft with no ready film has script_json only (db/schema.sql:365) and the prompt says so.
- Fit: REFUSE, no cutting at all. I drop my cut: mac and cloud serve 32768 (config.py:54, Makefile:52), so a refusal is rare and a narrowing path is code for nothing. Output reserve comes from the rows' size. `text_context` None = no budget, as the story pass.
- Whole rejection with a reason on any bad item: unknown or repeated n, empty, bracket, wrong language (`_is_in_language`, story.py:187), total > MAX_SCRIPT_CHARS, unparseable JSON (after `_loads`, story.py:1069). I drop my "drop and name".
- Apply: new `rewrite_passages(conn, group_id, language, changes, *, source, draft, revision)` in feed/narration.py, ONE `_locked_change`, new code `empty_text` (in save_passage an empty text removes, narration.py:179). No difference = returns None, no draft. 409 = existing freeze; preview and text stay. `render(true)` (feed-notes.js:251).
- Freshness checked on publishing a result too: a stale result shows as outdated, never as current.

Decisions inside it:


1. "Exactly one text per current n" vs changes only: WRITER stays. Schema {reply, changes:[{n,text}]}: only the passages it changes; every other passage keeps its words BY IDENTITY. Your objection was concealment. Fix: the job result states "Changed 8 of 20 passages", and the preview lists each one old -> new, so nothing is hidden. Why not all rows: the model is gemma4-E2B (config.py:39). "Fix passage 3" with all rows demands 19 verbatim copies, and every slip is a change nobody asked for, passing every validation. Output also grows with the narration, up to 10 000 chars per turn. Missing n is therefore not an error. Zero changes + a reply is a valid turn: reply only, no Use this.

2. Snapshot: NOT added, and no exact-input promise is made (your condition). timeline + decisions are the exact record of what was composed. Title, description, notes and photo details are labelled "as they are today" in the prompt (signals.py:138-174, notes.py:70-80). A note corrected after the film should win in a rephrase. PLAN.md states this limit at the gate, so the user can ask for it.

3. "Discard after departure": kept page state instead, from CLAUDE.md "never lose the user's place". Transcript, candidate, base triple and job id live in sessionStorage per memory + language, in try/catch as feed-notes.js:132. A return re-polls the job; a 404 says "did not finish, ask again"; the freshness rule covers the rest. Language change = another key, so no late result crosses languages. Still no table.

4. Added: a running rephrase is a level of its own (CLAUDE.md layers rule 7): `window.feedNotes.escape()` runs before `feedFilm.escape` (feed.html:335-340) and cancels the job.

5. UI: own form under #notes-list, transcript above it, Enter sends (`submitOnEnter`), Use this + Dismiss. Hidden in legacy/none, disabled when frozen.

Docs first: design §11.1, §11.3, the §11 mermaid, `docs/ui.md`. Tests: `tests/test_feed_rephrase.py`, `tests/test_feed_narration.py`, `tests/test_web_feed.py`, with the two-turn cumulative case and typing during Use this.
Hand-over: the user tries a request on a real film (a live model is needed, so tests cannot prove the wording).

## 5. Tests (no model loads; fakes and stubs only)

Targeted runs: `cd <base>/ivms777 && uv run pytest -q tests/<file>.py`. At steps 5 and 10: `uv run pytest -q` and `uv run ruff check .` in `<base>/ivms777`.
**Changed 2026-11-02 (t25, D56):** at steps 16 and 21 the main step runs only the targeted tests of the changed files, the tests of their direct
callers (the lists in §4) and `uv run ruff check <changed files>`. `sadensmol-code-review` runs only inside `task finish`.
A failure in a file of the user's uncommitted work is reported, not fixed (§1.5).

- `tests/test_catalog.py` (append): `voxcpm2` is an `HfSource` with the 40-hex revision; `profiles == ("mac", "cloud")`; `clones`; `languages == ("en", "ru")`; no speakers;
  estimate flag; `cloning_entry_for("jetson", "en") is None`; `cloning_entry_for("mac", "ru").key == "voxcpm2"`; `cloning_languages("cloud") == ("en", "ru")`;
  every non-cloning voice entry declares speakers. The existing budget test keeps passing for mac and cloud.
- `tests/test_modelsvc_slots.py` (append): preset routing never picks `voxcpm2` (also after a cloning swap); `ensure_voice_for("en", cloning=True)` on mac → `voxcpm2`,
  child args `(repo, device, revision)`; on jetson → False; every catalog voice key has a `VOICE_TARGETS` entry; an unknown key raises (no Kokoro fallback).
- `tests/test_downloads.py` (append): a pinned entry snapshots, probes and sizes at its revision; its source id carries the revision; unpinned entries unchanged.
- `tests/test_fetch_models.py` (append at the end): mac wants `openbmb/VoxCPM2@32279ef…` under `voice`; jetson does not; `_fetch("repo@rev")` calls `snapshot_download(repo, revision=rev)` (fake module).
- `tests/test_voxcpm_voice.py` (new; fake tts, pickle save/load): CPU refused; revision mismatch refused; missing weights message; built then restored;
  stale blob (other revision, other version, other sha) → built with the reason; unknown token → `ValueError`; per-passage tokens map to their caches;
  controls passed; ≤ 500-character pieces; sample rate from the model; reply shape; the module source has no top-level torch/voxcpm import (AST).
- `tests/test_modelsvc_api.py` (append): references decoded to bytes; sha mismatch, unknown token, mixed, > 8 → 422; `CloningUnavailable` → 503 with the reason;
  reply carries `revision` and base64 blobs; a request without references calls the backend exactly as today.
- `tests/test_modelsvc_backends.py` (append): references → `ensure_voice_for(cloning=True)`, voice unit, BATCH; unavailable → `CloningUnavailable`; identity mismatch raises;
  `VoiceBackend` sends the fifth argument only when given; `FakeBackend` skips cloning entries for presets.
- `tests/test_models_client.py` (append): payload encoding; 1800 s timeout with references, 300 s without; reply decoding of `revision` and blobs.
- `tests/test_voice_check.py` (new; fake client): WAV metrics; monotonic check; VOID marking; pass 2 sends the pass-1 blob and reads `restored`; A/B order recorded in `key.json`; unload called between passes;
  the clip contract of §2.13 step 1 refuses a 44100 Hz clip, a stereo clip, a 24-bit clip and a truncated clip before any client call.
- `tests/test_saved_voices.py` (new): tables exist after `migrate`, `SCHEMA_VERSION == 14`; create, rename, duplicate by `casefold()` (Latin and Cyrillic), blank, too long;
  recorded transcript = prompt, uploaded = `None`; `example_for` = newest in the language; `castable` filters by language; narrator set, read, clear;
  group delete cascades the narrator; conditioning save/read/upsert; owner scoping.
  - WAV (the test builds each clip with stdlib `wave`, or with `struct` for a hand-made header): 5.0 s and 30.0 s accepted; 4.99 s and 30.01 s refused;
    8-bit, 24-bit, 32-bit int; a float header (format 3) and a compressed header (format 2); stereo and 6 channels; 44100 Hz and 16000 Hz;
    garbage bytes, an empty payload, a header with 0 frames; a truncated data chunk → each a `VoiceError` with its text (the test asserts the type, not the `wave` cause);
    the stored file equals the posted bytes and `content_hash == sha256(posted bytes)`; every other `add_example` check.
  - Enqueue only: `request_voice_deletion` and `request_example_deletion` insert one intent and leave every row and file; a second call adds nothing; unknown or foreign → `False`.
  - Pending: `list_voices` shows `deleting=True`; `rename_voice`, `add_example`, `create_voice` with its name → `VoiceDeleting`; `castable` and `example_for` skip it
    (an older example of the same voice in that language is used); `save_conditioning` → `False` and writes no file.
  - Later cascade: `process_voice_deletions` deletes the voice, its examples and conditionings, sets `memory_narrators.voice_id` to NULL (`read_narrator` → `NarratorChoice(None)`),
    removes the files, ends the intent and returns 0. An example intent removes only that example and its blobs.
  - Kept files: a WAV shared with another voice stays; a clip added again (same bytes) before the file step stays and its key row is acknowledged.
  - Crash before unlink (`storage.delete` raises): rows cascaded, key row and intent stay, return > 0; a re-run removes the file and ends the intent.
    Crash after unlink (a storage fake that unlinks, then raises): the key row stays; a re-run finds no file, acknowledges it and ends the intent.
  - Two drainers (two connections to one file) on one outbox: every key is removed once; no error. One bad key never stops the pass for the other keys.
  - `save_conditioning` that replaces a blob puts the old key in `voice_file_deletions` with `deletion_id` NULL; the drain removes it when no row holds it.
  - A write that fails after `storage.write` (the insert raises) rolls back and leaves the write-ahead row; the drain removes the file.
  - Process death mid-write: close the writer's connection after `storage.write` and before `COMMIT` → the write-ahead row stays; the drain removes the file.
  - A drain between the write-ahead `COMMIT` and the writer's `BEGIN IMMEDIATE` (a hook runs `process_voice_deletions` there) acknowledges the row;
    the writer finds it gone, commits a new one, and ends with the file, its row and no write-ahead row.
  - A valid shared key is not rewritten: a storage fake counts writes; adding the same bytes to a second voice writes 0 times.
  - Ids: delete the newest voice, drain, create a voice → a new id, not the old one; the same for an example. A delete of the new voice then gets its own intent.
- `tests/test_feed_voice.py` (append): `offered` never yields a cloning entry; `resolve_saved` by name (case-folded) and by token; `saved_for_prompt` lines.
- `tests/test_feed_build.py` (append; a fake client whose `speak` accepts `references`): a saved narrator → the story prompt lists saved voices, not presets;
  `speak` gets `voice=token` and the references; a bracket cast to a second saved voice adds its reference; `story.narrator` is ignored; no example → `failed`, no speak call;
  a 503 from speak → `failed` with that reason and exactly one speak call; `built` blobs saved, `restored` not saved; `decisions["voice"]` recorded;
  no narrator row → the existing tests pass unchanged; a narrator being deleted → `failed` "is being deleted"; a deleted narrator (`voice_id` NULL) → `failed` "was deleted";
  in both cases no speak call and no Automatic film.
  Delete during a build (a fake `speak` enqueues and drains the deletion): of the narrator → `failed` "was deleted while this film was made"; of a bracket voice → the same;
  with a counting fake audio storage: delete the narrator after synthesis, a bracket voice after synthesis, and the narrator once after music →
  each `failed`, zero film-audio writes and zero deletes, and the audio of another ready film and of another in-flight film stays, also with identical bytes.
  Publish, then delete → the film stays `ready` and both WAVs stay. An Automatic film writes both WAVs at "saving the film".
  `tests/test_feed_build.py:157-161` (both clips stored, ready) keeps passing unchanged.
- `tests/test_pipeline_pass.py` (append): one `drain_pass` cascades a pending voice intent and removes its files, also when the embedder is down (pattern `:41`).
- `tests/test_web_voices.py` (new): the page renders voices, prompts, consent text, close fallback `/organize`; create/rename return the fragment;
  example upload stores it; 400/404/409/413/507 cases; jetson profile → notice and 409; audio GET returns the WAV; the Organize link exists and sits outside the picker form;
  each WAV format case of `test_saved_voices.py` → 400 with its text.
  - Enqueue only (`inline_drain` False): voice and example delete → 200, the fragment shows "deleting…", every row and file stays, no drain thread starts.
    A second delete → the same fragment, one intent. Rename / add example on the pending voice → 409. Audio of a pending example → 404.
    After `process_voice_deletions`: `GET /organize/voices/list` has no voice, and a delete → 404.
  - Standalone schedule (`inline_drain` True): with `process_voice_deletions` stubbed to wait on a `threading.Event`, the delete reply arrives while the event is unset
    (the request does not wait); after `set()` and `app.state.await_voice_drain()`, the rows and files are gone. Two deletes during one drain → one thread; both intents end.
  - Start-up: a library with a pending intent, then `create_app` with `inline_drain` True → `await_voice_drain()` → the voice is gone, with no request.
    `inline_drain` False → no thread; the intent stays for the worker.
  - Retry: a storage whose `delete` fails once, `RETRY_SECONDS` patched to 0 → the drain thread runs again, removes the file and ends the intent.
  - Exit window: the test holds the drain's lock while a stubbed pass returns 0, calls `schedule()` from a second thread, then releases the lock →
    exactly one more pass runs, whichever thread takes the lock first.
  - Wake: an add-example request with a pending key → a drain runs; the add-example route with no pending row starts no thread.
- `tests/test_web_feed.py` (append): the picker lists only castable voices of the language; hidden with none and on jetson (**superseded 2026-11-02**: always shown, see Phase 3 below); Prepare stores the choice and Regenerate preselects it;
  `""` clears it; a POST with no `narrator` field keeps the stored row; invalid → 404/409 and no build starts. A stored narrator that is being deleted, deleted,
  or has no example in the language → one more option, `selected`, with that label; Prepare with it → 409 with the reason, and no build starts.
  A pending voice is never in the castable list. On jetson with a stored saved choice → the picker shows "Automatic" and "«<name>» — voice copy is not available
  on this device", `selected`. A finished saved-voice film still plays after its voice is deleted and drained (`GET /feed/{key}/audio/narration` → 200).
  After a film build, `app.state.await_feed_build()` then `await_voice_drain()` → a replaced-blob key left by the build is gone.

Phase 3 (§2.15; D41 says where each test lives):

- `tests/test_feed_voice.py` (append): `voice_label("af_heart", "en") == VoiceLabel("af_heart", "af_heart (female, warm)")`;
  a token in `cast` → `VoiceLabel("Grandma", "Grandma (saved voice)")`; a `saved:` token not in `cast` → `VoiceLabel("a saved voice", "a saved voice")`;
  an unknown name → `VoiceLabel(name, name)`; `None` and `""` → `None`.
- `tests/test_feed_store.py` (append):
  - `film_states` holds only the languages with a row, in `LANGUAGES` order; `ready` is True for a ready row with narration,
    False for a ready row with no `narration_key`, a `failed` row (with its `error`) and a `running` row.
  - `film_narrator` (t18 read rules), one case each: stored `narrator_name` / `narrator_label` win; `chosen_by` `"user"`, `"story"`, `"mood"` read back;
    `"story"` + `narrator_asked_was_real=False` → `"mood"`; no `narrator_chosen_by` (a HEAD row) + flag True → `"story"`, + flag False → `"mood"`, no flag → `"auto"`;
    an old row with `narrator_used="af_heart"` and no label → `"af_heart (female, warm)"`; `"saved:3"` with no label → `"a saved voice"`;
    no `cast`, `narrator_used=None`, `{}` and `None` → `None`.
  - `others`: only `passages` entries with `used` set and ≠ `narrator_used`, numbered from 1, with `used_label` (or the `voice_label` of an old row);
    an entry with `asked="am_adam"` and `used` = the narrator (a rejected or gated cast) is not listed; `model_cast` is never read.
- `tests/test_saved_voices.py` (append):
  - fresh schema: `memory_narrators` has `preset_speaker`; an `INSERT` with both columns set raises `sqlite3.IntegrityError`.
  - old table: a file with the pre-Phase-3 `memory_narrators` DDL (`db/schema.sql:245-250` before Phase 3) and one row → `migrate` → the column exists, the row stays with `preset_speaker` NULL.
  - `set_narrator(…, preset_speaker="af_heart")` → `read_narrator == NarratorChoice(None, "af_heart")`; then a saved id → `NarratorChoice(id)` (preset NULL);
    then a preset again → `voice_id` NULL. Both given → `VoiceError`, the row unchanged; a blank preset → `VoiceError`; `set_narrator(…, None)` deletes a preset row.
  - On the old table (no CHECK), the same calls never leave a row with both columns set.
  - Draining a deleted saved voice leaves a preset row of another memory unchanged; the narrator row of the deleted voice reads `NarratorChoice(None)`, preset `None`.
- `tests/test_feed_build.py` (append; the existing fakes):
  - a stored preset `af_bella`, a story that asks for `af_heart` → `speak` gets `voice="af_bella"` and no `references` keyword;
    `cast.narrator_used == "af_bella"`, `narrator_chosen_by == "user"`, `narrator_name == "af_bella"`, `narrator_label == "af_bella (female, bright)"`.
  - the same speaker two ways (t18): a stored preset `af_bella` → `"user"`; Automatic with a story that names `af_bella` → `"story"`; both speak `af_bella`.
  - story vs mood (t18): Automatic with a story narrator `""` or an invented name → `narrator_used == pick_speaker(...)` and `"mood"`.
  - the story prompt of a preset film lists the presets (`catalogue_for_prompt`), not saved voices; the film is `ready` with `voices=None`.
  - a bracketed passage in a preset film is cast to another preset; its `passages` entry has `used_label`. A story voice at a passage no bracket points at,
    or an invented one, leaves `used` = the narrator, so `film_narrator(...).others` is empty.
  - a stored preset that `resolve` rejects (`"xenia"` on an English film) → `failed` with the §2.15 F reason, no speak call.
  - Saved film: `narrator_chosen_by == "user"`, `narrator_label == "<name> (saved voice)"`.
  - the existing saved-film and Automatic tests pass unchanged, except `test_an_automatic_film_records_its_voice_too` (`tests/test_feed_build.py:1123-1128`):
    its fake story names no narrator (`:84`), so `pick_speaker` casts it and the assert becomes `narrator_chosen_by == "mood"`.
- `tests/test_web_feed.py`, the Organize card (append beside `:206-213`):
  - before any film: `/organize?by=memories` has `href="/feed/memory-0?lang=en"` "Prepare English film" and `?lang=ru` "Prepare Russian film", each "not made yet";
    no unlabelled `>▶ Play</a>`.
  - after `_prepare` (en; the fake story names no narrator, `tests/test_web_feed.py:62`): "▶ Play English film" with "ready · <cast speaker> (auto)";
    Russian stays "Prepare Russian film". After a Prepare with `preset:af_bella`: "ready · af_bella (your choice)".
    A row whose cast says `"user"` while `memory_narrators` now holds Automatic still reads "(your choice)" (the stored cast wins, t18).
  - `feed_building = ("memory-0", "ru")` → "Open Russian film" "being made…"; `feed_queued` → "waiting to be made…"; building wins over a ready row of that language.
  - stored rows: `failed` → "the last try failed"; `running` → "interrupted"; ready with no narration → "no voice"; a ready row with `decisions_json='{}'` → "ready · voice not recorded".
  - Retell: the button reads "↻ Retell memory"; the note ends "No film to delete." before any film and "Deletes the films in: English." after the en film;
    the global "Rebuild memories" is still on the page.
  - the chat card: `_album_card.html` rendered with only `album` and `ctx` (as `web/app.py:1590-1592`) has no `album-films` and no `album-retell`.
  - film block (t19): each memory card has exactly one `<div class="album-film-block" id="films-memory-0">`, holding the rows, the retell form and its note
    but not the notes list; it carries `hx-get` and `hx-trigger` only while
    `feed_building` or `feed_queued` names that memory (any language), and neither attribute when nothing builds; the page has one `__filmBlocksWired` script.
    The bfcache return itself is a browser path: §6.3 step 6 checks it.
  - each row reads "<label> · <state>" (`</a> · <span class="album-film-state">`).
  - adapt `test_a_memory_can_be_retold_on_its_own` (`:724-741`): assert "↻ Retell memory" instead of "Rebuild".
- `tests/test_web_feed.py`, `/feed` (append beside `:863-1080`):
  - "Film language" is on the page; "English ✓" only after the en film is ready; "Русский" without ✓; while en is building, no ✓ on English.
  - "This film", one row per case written with `save_presentation`: `user` → "This film: English · narrated by af_bella (female, bright) — your choice";
    `story` → "… — picked automatically by the story"; `mood` → "… — picked automatically using the memory's mood"; a HEAD-shaped cast with no flag →
    "… — picked automatically"; `decisions_json='{}'` → "This film: English · Voice not recorded".
  - "Other voices": a cast with one bracket passage (`passage: 2`, `used: "am_adam"`) → "Other voices: passage 3 → am_adam (male, neutral)";
    an entry whose `used` is the narrator (a rejected cast, `asked="count_dracula"`) → no "Other voices" line and no "count_dracula" on the page.
  - after an Automatic Prepare, the Regenerate form still selects "Automatic — the app picks a preset voice for this memory" (t18), and the page shows no
    predicted voice before the first Prepare (no "narrated by" on the Prepare card).
  - the hint "The story picks the voice; otherwise the memory's mood guides the choice. Saved voices are never picked automatically." is on the page.
  - the bracket line: Automatic or a stored preset → "A [voice] note can request another preset voice for one passage."; a stored saved narrator →
    "A [voice] note can request another saved voice for one passage."
  - jetson, no saved voice: the select exists in the Prepare form with the Automatic option, the "Preset voices" group with every `offered("en")` speaker as
    `preset:<name>` and `"<name> — <gender>, <register>: <note>"`, and the disabled "Voice copy not available on this device"; no "Add a voice" hint.
  - mac, no castable voice: the select with Automatic and the presets, no "Saved voices" group, and the hint "Add a voice under Organize → Saved voices.";
    the page has no `href="/organize/voices"`. The Russian page lists the Silero presets only.
  - Prepare `preset:af_bella` → 303; `read_narrator == NarratorChoice(None, "af_bella")`; Regenerate preselects `preset:af_bella`. `preset:AF_BELLA` stores `"af_bella"`.
  - `preset:xenia` on en → 409 with "«xenia» cannot narrate in English. Choose another narrator.", no build; `preset:nobody` and `preset:` → 404 "Unknown preset voice.", no build.
  - a stored saved choice, then a Prepare with `preset:af_bella` → `voice_id` NULL; a stored preset, then a saved id → `preset_speaker` NULL.
  - a stored preset that the catalog rejects (written with `set_narrator` directly) → one more option in "Preset voices", `selected`, "«<name>» — not offered for English films".
  - adapt the whole-list picker tests (`:863-873`, `:889-901`, `:961-1014`) to compare the Automatic option (its t18 text) plus the "Saved voices" group only
    (the `_options` helper gains a group filter).
  - ready-screen structure (t20): the page has one `<header class="feed-head">`; inside it `.feed-actions` holds the `.feed-lang` form, `id="download"` and
    `.feed-close`, in that order; `.feed-settings` comes after `.feed-actions` and holds the `.feed-regen` form, which holds "This film", the select and the
    help lines (none inside a `<details>`). On the Prepare and building screens the header holds `.feed-lang` and `.feed-close`, and no `.feed-settings`.
    The ids `download`, `stage`, `start`, `notes-panel` and the history script stay (`location.replace` is still in the page).
- `tests/test_web_chat.py` (append, the source-reading style of `tests/test_web_feed.py:695`): `web/static/chat.js` keeps `window.__chatPageshowWired`,
  and its `pageshow` handler calls `location.reload()` only under `e.persisted` and `document.getElementById("chat-q")`
  (a regex over the listener statement). The Chat → Organize → film → Close path itself is the §6.3 check.
- `tests/test_web_voices.py` (append): the page and the list fragment (`GET /organize/voices/list`) show "Record clean, normal speech in a quiet room." and
  "Only the newest example in each language is used." inside each available `.voice-add`, before Record (t21).
- Reading texts (§2.15 I):
  - `tests/test_saved_voices.py` (append): `RECORDING_PROMPTS` has 4 texts per language; the first ids are `en-kitchen` / `ru-kitchen` with today's texts;
    every id starts with its language; no text contains a voice_check sentence of `scripts/voice_check.py:SENTENCES`.
    `add_example(source="recorded", prompt_id="ru-rain")` → `transcript` = that text; `prompt_id=None` → the first text; `"ru-nope"` and `"en-rain"` on a Russian
    example → `VoiceError` and no row; `source="uploaded", prompt_id="ru-rain"` → `transcript is None`.
    Adapt `:205` and `:210` (`RECORDING_PROMPTS["ru"]` → `RECORDING_PROMPTS["ru"]["ru-kitchen"]`).
  - `tests/test_web_voices.py` (append): every text is on the page with its `data-lang` and `data-prompt-id`; `.recorder-next` "Different text" and
    `.recorder-start` "Start recording" sit inside `#voice-recorder`, outside `#voices`; POST examples with `source=recorded&prompt_id=ru-rain` → the stored
    `transcript` is that text; `prompt_id=en-rain` with `language=ru` → 400 with its text; no `prompt_id` → the first text.
    `web/static/voices.js` (source reading): Save appends `prompt_id`; the Esc handler tests the live recording, the take, the prepared panel and a focused field in that order.
    Adapt `:105` to loop over the nested texts.
    Replace `test_the_picker_is_hidden_with_no_castable_voice` and `test_the_picker_is_hidden_on_jetson` (`:875-887`) with the two always-shown tests above.

Phase 4 (§2.16; D55 says where each test lives). Fakes only: a fake client whose `speak` accepts `references`, records each call and returns a
fixed reply (`model`, `revision`, `conditioning`); no model load, no running service.

- `tests/test_saved_voices.py` (append):
  - schema: `voice_film_examples` and `voice_previews` exist after `migrate`; `SCHEMA_VERSION == 14`; a file with the Phase 3 schema (no new tables)
    and one voice row → `migrate` → both tables exist and the row stays; a `voice_previews` insert with `text_kind='other'` or `conditioning='x'`
    raises `sqlite3.IntegrityError`.
  - `film_example`: no row → the newest, `chosen=False`; `set_film_example(older)` → that example, `chosen=True`, also after a newer example is
    added; the chosen example pending → `(None, True, "deleting")` and `castable` skips the voice (no older or newer example in its place); drained →
    the row holds `example_id` NULL → `(None, True, "deleted")`; a new example in that language → still `"deleted"`; `set_film_example(new)` →
    chosen; the voice drained → no `voice_film_examples` row is left; no example at all → `(None, False, "none")`.
  - `set_film_example`: an example of another voice, or of another owner → `VoiceError`; another language → `VoiceError`; a pending voice or example
    → `VoiceDeleting`; an unknown language → `VoiceError`; each refusal writes no row.
  - `reference_for(conn, store, example)` == `{"audio", "sha256", "language", "transcript", "conditioning"}` with the stored blobs (the old
    `_reference` shape).
  - `save_preview`: the row and the file; a second save of the same example replaces the row, puts the old key in `voice_file_deletions`
    (`deletion_id` NULL), and the drain removes the old file; a pending example → `None`, no row, and the drain removes the written file;
    `previews(conn, owner)` holds only the owner's tests.
  - drain: a voice delete and an example delete each remove the test WAV; a test WAV whose bytes equal another example's WAV stays while that
    example holds the key; a key held only by `voice_previews` is kept by the ref check.
- `tests/test_voice_preview.py` (new):
  - `PREVIEW_SENTENCES == {lang: dict(pairs) for lang, pairs in scripts.voice_check.SENTENCES.items()}`; no sentence appears in any
    `RECORDING_PROMPTS` text.
  - `preview_text`: recorded + `"reading"` → the transcript; uploaded + `"reading"` → `VoiceError`; `"sentence"`, `"EN-2"` on an English example →
    that text; `"RU-2"` on an English example and `"EN-9"` → `VoiceError`; custom `"  "` → `VoiceError`; 301 characters → `VoiceError`; 300
    characters → accepted and stripped; kind `"other"` → `VoiceError`.
  - `run_preview` (a `FakeClient`): one `speak` call with `(text, example.language, "saved:<voice_id>", None, None)` and `references ==
    {"saved:<voice_id>": reference_for(conn, voices, example)}`; a `built` blob is stored with `save_conditioning`; the returned `VoicePreview` and
    the stored row carry the text, model, revision, `"built"`, the elapsed time and the audio length.
  - restored reuse: with a stored blob, the request carries it; the fake answers `restored` → no new conditioning row, `conditioning == "restored"`.
  - cancel: `cancelled` turns True during `speak` → `Cancelled`; no `voice_previews` row; a counting storage writes the conditioning blob and no
    test WAV; the conditioning row exists.
  - cancel while waiting: the test holds the lock and `cancelled` returns True → `Cancelled` at the next acquire timeout (the test only guards a
    hang with a join timeout; changed 2026-11-02, t25 round 2: no time promise), `speak` never called, progress showed "waiting for the voice model".
  - added 2026-11-02 (t25, round 2) — cancel just before speak: the fake storage `read` (in step 2) flips `cancelled` to True → `Cancelled`, `speak`
    never called, the voice lock is free afterwards, no `PreviewFailed`.
  - stale: the fake answers `revision="other"` for `job.revision="rev"` → `PreviewFailed` "The voice model changed …"; no row.
  - deleted during the test: the fake's `speak` enqueues and drains the example's deletion → `PreviewFailed` "The example was deleted …"; no row;
    no file is left after the drain.
  - a fake `speak` that raises an `httpx.HTTPStatusError` with `{"detail": "voice copy is not available: …"}` → `PreviewFailed` with that text;
    a missing example WAV → `PreviewFailed` "The example could not be read …".
  - casting stays: `memory_narrators` and `voice_film_examples` rows are equal before and after a test.
  - progress: the stages come in the order of §2.16 D, the fractions never go down, and the last one before the return is "saving the test".
  - added 2026-11-02 (t25) — publication barrier: the test holds `job_lock` while `run_preview` reaches step 5, sets `cancelled` to True, then
    releases → `Cancelled`, no `voice_previews` row, `on_saved` never called. With `cancelled` False, a storage fake whose test-WAV write waits on
    a `threading.Event`: `job_lock.acquire(blocking=False)` from the test fails while the write waits (the lock is held through the save);
    after `set()`, the row exists and `on_saved` ran once.
  - added 2026-11-02 (t25) — restored reuse, film then test: `build_presentation` of a film narrated by the voice (the fake answers `built`)
    → then `run_preview` on the same example → the test's reference carries the film's blob; the fake answers `restored` → no new conditioning row.
    The same while the film's fake `speak` waits on an Event and the test waits for the shared `voice_lock` in another thread.
- `tests/test_feed_build.py` (append; the existing fakes):
  - resolver parity: a chosen older example → the film's `references[token]["sha256"] == older.content_hash` and equals
    `film_example(...).example.content_hash` (the badge); with no choice → the newest.
  - the chosen example drained → `failed` "The English example of «<name>» chosen for films was deleted. …", no speak call; being deleted →
    "… is being deleted. …", no speak call; never another example.
  - the bake: `run_preview` on an example, then a film narrated by that voice → the film's reference carries the blob the test stored; the fake
    answers `restored` → no new conditioning row.
  - `voice_lock`: a held lock → progress shows "waiting for the voice model" and `speak` runs only after release, for a preset film and a saved
    film; a cancel during the wait → `Cancelled`, no `failed` row, no speak call.
  - added 2026-11-02 (t25) — `voice_turn`: `cancelled` True before the call → `Cancelled`, and the lock is still free afterwards
    (`lock.acquire(blocking=False)` succeeds); `cancelled` returns False on its first call and True on the next one (the check after a free acquire) → `Cancelled`, and the lock is free; a body
    that raises → the lock is free; a body that returns → the lock is free.
  - added 2026-11-02 (t25) — the film's turn: a fake `speak` that records whether `voice_lock.locked()` is True during the references read (a
    fake storage `read`), the speak call and `save_conditioning` → True in all three; a `_fail` in the references read releases the lock.
  - added 2026-11-02 (t25) — restored reuse, test then film, is the "bake" test above; it also runs with the test waiting on the shared lock while
    the film holds it, and then the reverse order.
- `tests/test_web_voices.py` (append; the fake client through `Settings.build_models_client`, as `tests/test_web_feed.py:78-97`):
  - the page: each example shows `#<id>`, "Original" with its `<audio>`, and a `.example-preview` "Test this example" button with its data; the newest
    English example reads "used for films (newest)" AND has a "Use for films" button; the older one reads "not used for films" with the button;
    after POST film-example with the older one → "used for films (your choice)" on it with NO button, and "not used for films" with the button on
    the newer one (changed 2026-11-02, t25); POST film-example with the newest (the pin) → "used for films (your choice)" on it, and it stays
    chosen after a newer example is added; the panel `#voice-preview` sits outside `#voices`, with the reading option, the five sentences of each
    language and "Your own text".
  - film-example route: 200 + fragment; an unknown voice, a foreign voice, or an example of another voice → 404; another language → 400; a pending
    example → 409; jetson → 409.
  - the chosen example deleted and drained → the fragment holds "The English example chosen for films was deleted."; the Delete form of a chosen
    example carries the §2.16 F `data-confirm`; the Delete form of a non-chosen example has none.
  - the test route: 200 + `_voice_preview.html`; after `app.state.await_voice_preview()`, GET `/organize/voices/preview` reads `done`; the list
    shows "Generated" with the text, the fake model, "made in", "conditioning built"; GET the test audio → 200 `audio/wav`; a second test of the
    same example → one `voice_previews` row.
  - refusals: an unknown kind, `"reading"` on an upload, `"sentence:RU-2"` on an English example, a blank or a 301-character text → 400 with the
    text; an unknown example → 404; a pending example → 409; jetson → 409; low disk → 507.
  - cancel: a fake `speak` that waits on a `threading.Event`; POST cancel → `data-state="idle"` while the event is unset (the reply does not wait);
    `set()`, `await_voice_preview()` → no `voice_previews` row; the conditioning row exists.
  - replace: a test of example A while the fake waits, then a test of example B → after both finish, a `voice_previews` row for B only.
  - added 2026-11-02 (t25) — same-example replacement: a test of example A with EN-1 whose fake `speak` waits; a test of A with EN-2; then the
    EN-1 call returns LAST → after both finish, one `voice_previews` row for A with the EN-2 text; the state is `done` for the second run.
  - added 2026-11-02 (t25) — publication barrier through the routes: the fake test-WAV write waits on an Event inside step 5; POST cancel from a
    second thread does not return while the write waits; after `set()`, the cancel returns `idle`, and the row exists (the test was published
    before the cancel). A POST cancel that returns BEFORE step 5 → no row after `await_voice_preview()`.
  - reload: GET `/organize/voices` while a test runs → `#voice-preview-job` has `data-state="running"` and the example id.
  - the test leaves `memory_narrators` and `voice_film_examples` unchanged.
  - the tip: adapt `TIP` (`:548-549`) to the §2.16 F text; the existing tip tests keep passing.
  - `web/static/voices.js` (source reading, `_js_function` as `:629-632`): `startRecording` pauses every `audio` before `getUserMedia`;
    `pollPreview` fetches `/organize/voices/preview`; `generatePreview` appends `text_kind`, `text_id` and `text`; the Esc handler tests live,
    take, prepared, the focused field, then `previewTarget && !previewBusy()`, in that order, before `leave()` (extend
    `test_the_script_sends_the_prompt_id_and_unwinds_esc_in_order`, `:616-626`).
  - changed 2026-11-02 (t25, round 2) — ordering tests under node, not regex: the test extracts `previewVerdict`, `completionKey` and `refreshLoop`
    with `_js_function` and runs scripted vectors with `subprocess.run(["node", "-e", …])`; it is skipped when `shutil.which("node")` is `None`
    (no npm package, no JS runner added; the user's mac has node v26.12.1). Each vector asserts the calls to the injected `replaceList` and
    `markRefreshed`:
    - old epoch e1 / run 1 done, then new epoch e2 / run 1 done → both refresh (different keys);
    - an old-epoch status reply after the reconciliation → `ignore`; an unseen third epoch is never adopted (`reconcile`, never `apply`);
    - same run: `running` after `done` → `ignore`; a lower run in the same epoch → `ignore`;
    - a deferred typing retry, and a delayed list body, after a Cancel, a new run or a restart → no `replaceList`, no marker;
    - a failed refresh retries only while its key and generation are current;
    - publication wins, then Cancel → the idle reply's `data-completed` refreshes once;
    - the reviewer's case A: `completed=1` deferred while typing; Generate bumps the generation and gets a 400; the no-store status still reports
      `completed=1`; after the blur → exactly one successful refresh;
    - the reviewer's case B: an old publication whose refresh is delayed; adopt the new epoch, `idle`, `completed=0`; deliver the old body → only the
      restart-sync refresh replaces the list; the new process's later `completed=1` refreshes separately;
    - the restart text: the panel showed `running`, the adopted snapshot is `idle` → "The app restarted, so this test stopped. Generate it again."
  - added 2026-11-02 (t25, round 2), Python: `completed` is set by `on_saved` and carried by every snapshot and fragment (`data-completed`); GET
    `/organize/voices/preview` answers `Cache-Control: no-store`; the epoch is 32 hex characters, and two `create_app` calls make two epochs; while a
    cancelled test's fake `speak` blocks on an Event, `app.state.voice_calls.locked()` is True and a second test reads "waiting for the voice
    model", then runs after `set()`. Every blocked-speak, publication-wins and cancel-wins, reuse and outbox test above stays.
- `tests/test_web_feed.py` (append):
  - a stored saved narrator whose chosen English example was drained → one more option, `selected`, "«<name>» — the English example chosen for films
    was deleted"; Prepare with it → 409 with the §2.16 C text, no build; being deleted → "… is being deleted".
  - a voice whose chosen example is pending is not in the "Saved voices" options.
  - `app.state.voice_calls` exists; with it held, a Prepare's progress fragment reads "waiting for the voice model"; after release the film is ready.

Phase 5 (§2.17; D75 says where each test lives). Fakes only: no torch, no voxcpm, no datasets import, no model load, no running service. A fake
model, a fake data loader and a fake optimizer drive the trainer; a fake `TorchWorker` drives the service job; a fake client drives the scripts and
the app job.

Slice A:

- `tests/test_catalog.py` (append): `voxcpm2.adapter.lora` equals the dict of §2.17 B; `train_cost_measured is False`; `train_profiles == ("mac",)`;
  `guidance_minutes == (5.0, 10.0)`; no other entry has an adapter; `adapter_entry_for("mac").key == "voxcpm2"`; `adapter_entry_for("cloud")` and
  `adapter_entry_for("jetson")` are `None`.
- `tests/test_modelsvc_slots.py` (append): on mac `voice_train` is registered with the trainer target, args `(repo, device, revision, <sorted LoRA
  JSON>, <data_dir>/voice-bakes)` and cost 14000; `model_cost_mb={"voice_train": 9000}` overrides it; on jetson no `voice_train`; a cloning swap
  gives the voxcpm2 child the LoRA JSON as a fourth arg; Kokoro and Silero args do not change. Added (t31): with the lease held and the
  `voice_train` worker running, `apply({"music": <another key>})` leaves `voice_train` registered and its worker running (no `unload`, no `stop`).
- `tests/test_governor.py` (append): with `lease("b")` held, `acquire(["c"])` under pressure evicts `a` but never `b`, and raises
  `InsufficientMemory` when only `b` could make room; `unload("b")` raises `UnitLeased`; after the `with`, `b` is evictable again; `leased()`.
- `tests/test_voxcpm_trainer.py` (new):
  - import scope: importing the module imports no `torch`, `voxcpm`, `datasets` (`sys.modules` before and after, as `tests/test_voxcpm_voice.py`);
    an AST scan of every import in the module finds only `torch`, `voxcpm` (and its submodules), `safetensors`, `huggingface_hub`, `numpy` and
    the standard library (no `tensorboard`, no `accelerate`, no `peft`).
  - `default_steps(1) == 200`, `default_steps(25) == 500`, `default_steps(80) == 1000`; `train_config` accepts the defaults and refuses `steps=0`,
    `steps=5001`, `learning_rate=0`, `learning_rate=0.02`, `batch_size=9`, each naming the field.
  - `check_trainable(["base_lm.layers.0.self_attn.q_proj.lora_A"])` passes; `[]` and `["base_lm.layers.0.mlp.weight"]` raise.
  - `run_training` with a fake `train_step` returning 3.0, 2.5, …: `steps_done == steps`, `loss_first`, `loss_last`; `report` sees the steps in order
    with a rising fraction; `cancelled` turning True after step 3 → `state == "cancelled"`, `steps_done == 3`, and `train_step` is not called again;
    a `nan` loss → `RuntimeError` "…not a number at step N"; `peak_device_bytes` is the largest `memory()` value; `seconds_per_step` from a fake clock.
  - added (t31): the fake dataset records `add_column("text_ids", …)` with the fake tokenizer's ids and records no `map` call; the batch
    generator runs a second pass over a 2-batch fake loader (the loader's iterator is asked for twice), and no `itertools.cycle` is used (AST).
  - `VoxCPMTrainer.loaded(fake_model, …)`: `start(job)` returns at once and the thread reaches `done`; `status()` while it runs never waits for a
    step (the fake `train_step` blocks on an Event, `status` returns); `save` gets every `lora_` tensor and `adapter_metadata` with format, model,
    revision, LoRA JSON, dataset sha256, config and steps; a second `start` raises; `cancel` during training → `cancelled` and `save` never called;
    an exception in the fake data step → `failed` with the type and message.
  - device: `VoxCPMTrainer("repo", "cpu", "rev", "{}", "/tmp")` raises before any import; the accelerator-device check refuses a fake accelerator
    whose device type is `cpu` for `device="mps"`; the env helper sets `HF_DATASETS_CACHE` and `HF_DATASETS_OFFLINE` before the `datasets` import (a
    fake import hook records the order).
- `tests/test_voxcpm_voice.py` (append; the `loaded` seam with a fake tts that records `reset_lora_weights`, `load_lora_weights`, generations):
  - no adapter on a fresh child → no reset, state `"none"`; adapter X → reset then load, `"loaded"`; X again → no reset, no load, `"kept"`; then no
    adapter → reset, `"none"`; two tokens with adapters X and Y in one 4-passage call `A B A B` → the loads alternate in speaking order and every
    generation runs with the adapter of its token.
  - `adapter_mismatch`: another revision, another LoRA JSON, a foreign format → each its reason; `load_mismatch`: a skipped key, a missing key →
    each its reason; the child then resets and raises `ValueError`, and nothing is generated.
  - added (t31): `tensor_mismatch` with an extra base key (`base_lm.layers.0.self_attn.q_proj.weight`) → refused BEFORE `load_lora_weights` is
    called, and the fake's base weight is unchanged; a wrong shape → refused before the load; a fake `load_lora_weights` that copies one tensor
    and then raises → reset called and the kept sha cleared. After each failure, a plain request reports `"none"` and the adapter that was
    loaded before reports `"loaded"` (never `"kept"`), and both generate.
  - added (t31): with no adapter, `load_lora_weights` and `reset_lora_weights` are never called, and `generate_with_prompt_cache` gets the same
    arguments as on a child built without `lora`.
  - `use_clip=False` → `generate_with_prompt_cache(prompt_cache=None, …)`, no `build_prompt_cache`, no conditioning entry for the token.
  - a child built without `lora` and a request with an adapter → `ValueError("this voice child was built without LoRA")`.
  - the temporary adapter file ends in `.safetensors` and is gone after the call, also after a failure.
- `tests/test_voice_bakes.py` (new; fake registry, governor, scheduler, worker, clock):
  - `start` order: `gate.close()` → `gate.drain()` → `governor.unload("voice")` → `lease("voice_train")` → `reserve(["voice_train"], BATCH)` →
    `call("start", job)`; `finally`: lease released, `unload("voice_train")`, the cache folder removed, `active()` False, then `gate.open()` —
    after done, failed, cancelled and an exception.
  - added (t31), the gate: a speak held inside the backend (a fake that blocks on an Event) → the bake stays in "waiting for the voice calls to
    end" and calls no `unload("voice")` until the speak returns; a speak that entered the gate and waits in a fake `_admit` is counted the same; a
    speak after `close()` → `BakeRunning` with no `ensure_voice_for` call; `model_ensure("voice")` during a bake and `model_ensure("voice_train")`
    outside one → 409 through the route; after the teardown a speak runs again.
  - the work folder holds one WAV per clip and `train.jsonl` with `{"audio": <absolute path>, "text"}` lines; `dataset_sha256` is order-independent.
  - status mapping: the stages and fractions of §2.17 E (load fraction from the fake clock, capped at 0.099; training `0.12 + 0.83 * step/steps`);
    `min_free_mb` is the smallest fake `free_mb`; `peak_device_bytes` the largest.
  - cancel before the reserve → no `call("start")`; cancel during training → exactly one `call("cancel")`, status `cancelled` only after the child says so.
  - a second `start` while one runs → `BakeRunning`; `adapter(id)` before done → `BakeRunning`; `delete(id)` while running → `BakeRunning`;
    after done → the folder is gone; `sweep()` removes a leftover folder.
  - refusals (`BakeRefused`): 0 clips, 301 clips, a 5 MiB clip, an sha mismatch, a stereo or 44.1 kHz WAV, blank text, 1001 characters, `language="de"`,
    a bad config, a 70 KiB tag; each writes no folder.
  - the child dies (a fake `call` raises `RuntimeError("… worker died during status")`) → `failed` with that text, and cleanup runs.
- `tests/test_modelsvc_api.py` (append): the six routes and their codes; `GET /voice/bakes/current` is not taken by `{bake_id}`; `/speak` during a
  bake → 409 with the percent; adapter base64 errors, sha mismatch, 5 adapters, a 65 MiB adapter, `use_clip=false` without an adapter → 422; the
  reply's `adapters` passes through; `POST /models/voice_train/unload` during a bake → 409.
- `tests/test_modelsvc_backends.py` (append): `CompositeBackend.speak` raises `BakeRunning` while the `VoiceGate` is closed and calls `leave()` after a speak that raised (t31); `model_unload` goes through the
  governor; the fake backend's scripted bake walks running → done and returns its bytes.
- `tests/test_models_client.py` (append): every bake method's path, method, timeout and decoding (`bake_current` 404 → `None`); `speak` sends the
  adapter as base64 with `sha256` and `use_clip`.
- `tests/test_voice_bake_check.py` (new; a temp DB made by `migrate`, a temp voices store, a fake client):
  - `--list` prints each take with its words and the "reading text, not checked" / "NO WORDS" labels, and starts no bake;
  - added (t31): Ctrl-C, then the fake reports `done` (or `failed`) instead of `cancelled` → the wait ends, `bake_adapter` is never called, no
    `adapter.safetensors` is written, `bake_delete` is called, exit 130; a `failed` run with no Ctrl-C → exit 1 and `bake_delete`; the report
    labels B's RTF "today's films on the LoRA-built child";
  - a take with no words, or no `--confirm-texts` → exit 2, the take named, `bake_start` never called; `--text 9=…` supplies the words;
  - the run: `bake_start` gets every take with its words; Ctrl-C (a `KeyboardInterrupt` from the fake sleep) → `bake_cancel`, then the report and
    exit 130; done → `adapter.safetensors` written and `bake_delete` called; three `speak` calls per sentence (A adapter + clip, B clip, C
    `use_clip=False`) with random X/Y/Z names and a `key.json` that maps them; `report.json` holds the fields of §2.17 I step 7;
  - the DB file's bytes are equal before and after (read-only), and the voices store is unchanged.
- Added 2026-11-02 — the tests of the pair fixes, already in the working tree (t32-t38; this plan only records them):
  - t32 (`tests/test_modelsvc_backends.py`, `tests/test_modelsvc_api.py`): a manual `voice` ensure enters the gate; with a real concurrency-1
    scheduler and the ensure queued in `_admit`, the voice load ends before the bake unloads `voice` and loads the trainer, with no overlap; a
    held load; a closed gate → 409 and no scheduler or registry call; a load failure releases the count, also in the registry-only branch.
  - t33 (`tests/test_voxcpm_trainer.py`): `run_training` with `steps=1` and the cancel set inside `train_step`, and set in the final report →
    `cancelled`, `steps_done == 1`, the measurements kept; the trainer with `steps` 1 and 5 and a cancel during training → `cancelled`, one
    forward, no save, no adapter, no `adapter.safetensors`.
  - t34 (`tests/test_voice_bakes.py`): with `gate.open` paused, a `start` from another thread has not returned, the old work folder exists and
    the gate is closed; `rmtree` sees the previous bake in an end state (done, failed, cancelled); after the second close, only the second
    bake's thread opens the gate.
  - t35 (`tests/test_modelsvc_backends.py`, `tests/test_modelsvc_api.py`): `voice_train` ensure → `BakeOnly` and 409 with no bakes, an inactive
    bake and an active one (open gate, closed gate, teardown), on the scheduler path and the registry path, with no scheduler or registry call.
  - t37 (`tests/test_voice_bakes.py`, `tests/test_modelsvc_api.py`): 480 declared frames with no payload, a zero-frame WAV, and 480 frames less
    one byte → 422 with the indexed reason; no bake, no folder, `active()` False, the gate open, no unload or training call.
  - t38 (`tests/test_voice_bake_check.py`): stdout lists every X/Y/Z file with no variant, adapter state or per-row RTF; an error row prints
    the sentence, the file and "ERROR" with no reason; `report.json` and `key.json` keep the mapping and the diagnostics; the mean RTF of A,
    B and C is printed.

Slice V:

- `tests/test_web_voices.py` (rewrite every `/organize/voices` path to §2.17 V1; every existing behaviour test keeps its assertions):
  - `GET /organize/voices` → 308 with `Location: /voices`; the old POST paths answer 404 or 405.
  - `GET /voices/list` and `GET /voices/preview` answer their fragments (not 422); a foreign or unknown id → 404 on every `/voices/{id}/…` route.
  - the grid: no `.voices-close`; each card is `<a class="voice-card" href="/voices/<id>">` with no `hx-boost`, the per-language minutes and takes,
    and the film line; a pending voice has no link and "deleting…"; New voice sits before the cards; create answers the cards with `data-created`
    on the new one; no voice copy → the notice and no New voice.
  - the leaf: "Voices" link and the name in the header; `.voice-menu` holds Rename and Delete voice with today's confirm; three `role="tab"`
    buttons; `?tab=model` renders the model tab active, `?tab=nope` renders Recordings; `#voice-recorder` and `#voice-preview` sit outside `#voices`.
  - the film-take select: one per language with takes; "(newest)" on the newest; the stored choice selected; after `film-example` the choice is
    selected; a drained chosen take → a disabled selected "— the chosen take was deleted —" option and the §2.16 F line; no "newest" reset option.
  - the Test tab: the take select lists the usable takes; the comparison block names the take and its text.
  - the t27 certificate tests and the Phase 4 test-job tests pass on the new paths.
  - `voices.js` under node (`_js_function`, skipped without node, as Phase 4): `cameFromGrid("http://h/voices", "http://h")` True,
    `("http://h/voices/3", …)`, `("http://x/voices", "http://h")` and `("", …)` False; `tabFrom("?tab=test") == "test"`, `tabFrom("?tab=x") ==
    "recordings"`; `leaveNeedsConfirm` True for live and take, False for prepared and idle; `lifecycle` with fakes: `mount` twice → one set of
    timers; `unmount` stops every fake track and clears every timer; a fetch reply after `unmount` changes nothing. Added (t31): `pageshow` with
    `persisted: true` after `pagehide` mounts again; the sequences native Back, native Forward and htmx restore each end with exactly one poller
    on the current nodes; `pageshow` with `persisted: false` after a fresh load does not mount twice.
  - `voices.js` evaluated twice in a node `vm` with a fake `document` and `window`: each document listener is added once; nothing calls
    `getElementById` at evaluation when `#voices-page` is absent.
  - `voices.js` source: the `htmx:confirm` handler checks `nav.nav` and skips `#settings-open`; Esc tests live, take, prepared, the focused field,
    then close, in that order; close never reads `history.length`.
  - Added 2026-11-02 (t36, in the code): the real lifecycle and page callbacks with a held body: the headers resolve, the body is held,
    `unmount`, a second leaf mounts with the same epoch, `completed` and generation, then the old body resolves and, separately, rejects →
    no swap, no job update, no error, no recovery read, no timer change; for the deleting poll, `send` (detail and success), `mutate` (body and
    detail), `pollPreview`, and the old `previewSync` reconcile and repair; the grid and a page with no leaf as the second destination; an
    alive-owner control that does apply.
  - Changed 2026-11-02 (rework, t39): the film-take select tests and the per-take Test tests above describe Slice V as built; Slice B
    replaces them (Slice B list below).
- `tests/test_web_organize.py` (append): `/organize` has no `/organize/voices` and no `/voices` link in its head; the nav in `base.html` lists
  Upload, Library, Chat, Organize, Voices in that order, all inside `nav.nav`.
- `tests/test_web_feed.py` (adapt): the hint reads "Add a voice under Voices." with `href="/voices"`; the 409 texts say "under Voices".
- `tests/test_feed_build.py` (adapt): the three `_saved_cast` texts say "under Voices".

Slice B (**changed 2026-11-02, rework, t39: rewritten**; fakes only — a fake client, a temp DB made by `migrate`, a temp voices store; no
model, no service):

- `tests/test_saved_voices.py` (lane 31; append and adapt):
  - schema: after `migrate`, `voice_bake_runs`, `voice_models`, `voice_model_takes` (with `language` and `seconds`) and `voice_tests` exist; an
    existing DB gains `voice_examples.transcript_confirmed_at` and `kit_id`; a DB seeded with one row each in `voice_film_examples`,
    `voice_previews` and `voice_conditionings` still holds those rows and files after `migrate` (D81); `SCHEMA_VERSION == 14`; no
    `saved_voices.last_model_version`.
  - the kit: 26 texts per language; ids unique and `<lang>-…`; the first text of each language is unchanged; `RECORDING_PROMPTS` is derived from
    it; every `style` is a `KIT_STYLES` key; no text holds a word of the voice_check list; every text has 80-140 characters.
  - `add_example(…, prompt_id="ru-slow-1", transcript="…")` → confirmed, `kit_id` stored; no `transcript` → the kit text, not confirmed; an upload
    with words → confirmed; without → NULL; 1001 characters → `VoiceError`. `set_transcript`: confirms; corrects; blank → `VoiceError`; a pending
    take → `VoiceDeleting`; a foreign take → `VoiceError`.
  - `bake_snapshot`: `unresolved` lists the takes with no words and with unconfirmed words; `snapshot_hash` ignores order and changes with words.
    `start_bake_run` / `read_bake_run` give back the same takes (with language and seconds) after the current rows change.
  - `save_voice_model`: version 1, then version 2 in place with the old file key in `voice_file_deletions`; `voice_model_takes` holds the
    snapshot ids, languages, seconds and fingerprints; the same `bake_key` twice → the same row, no version 3, no second file; a
    `dataset_sha256` that differs → `VoiceError`; a take deleted (pending or drained) during the bake → STORED, version up; a take whose words
    changed during the bake → STORED; the voice pending → `None` and no row.
  - `voice_model_for`: no row → `problem == "none"`; usable; `changes` after an added take, an edited word, a deleted take (also while it is
    only pending) and an unresolved take, each counted once; another revision or LoRA JSON → `problem == "revision"`; `languages` from the
    snapshot; a deleted training take keeps `usable` True and `languages` unchanged.
  - `film_model` and `castable_models` (changed, t40): `none`, `revision`, `language` (a model that learned only `ru` → `en` is `language`),
    `deleting`; `castable_models` lists only voices with a usable model of that language, as `(voice, model)`; a voice with takes and no model
    is not in it; `castable` still returns today's take pairs (lane 31 only adds).
  - `model_reference` → exactly `{"language", "adapter": {"blob", "sha256"}, "use_clip": False}`: no `audio`, `sha256`, `transcript` or
    `conditioning` key.
  - `save_test`: stores with the model version and sha256; a second test replaces it and sends the old key to `voice_file_deletions`; a model
    version or sha256 that differs from the stored row → `None`; the voice pending → `None`; a `take` test whose take was cascaded meanwhile →
    stored with `example_id` NULL; a later take delete sets `example_id` NULL and keeps the row and its file.
  - drain: a voice delete removes the model file and the test WAV; an example delete of a trained take keeps the `voice_models` row, its
    file and its `voice_model_takes` row; the ref check keeps a key held only by `voice_models` or only by `voice_tests`; the files of
    `voice_previews` and `voice_conditionings` still leave with their take.
  - added (t40): every name that `feed/build.py`, `feed/voice_preview.py`, `web/app.py` and `web/voices_api.py` import from
    `feed.saved_voices` still exists after lane 31 (an import of each module succeeds). `voice_bake_runs.cancel_requested_at` exists and is
    NULL after `start_bake_run`; `request_bake_cancel` sets it once and returns True, and returns False when the run row is gone;
    `save_voice_model` for a run with the flag → `BakeCancelled`, no row, no file kept (the write-ahead row stays for the drain); for a
    `bake_key` with no run row and no model row → `BakeCancelled`; a cancel after a store → False, and the model stays (store wins).
- `tests/test_saved_voices.py` (lane 32d; added t40): the removed names are gone — `film_example`, `set_film_example`, `FilmExample`,
  `castable`, `save_conditioning`, `save_preview`, `previews`, `read_preview_audio`, `VoicePreview` (`hasattr` is False); `example_for`,
  `conditionings` and `reference_for` remain; the tests of the removed names leave the file.
- `tests/test_voice_bake_check.py` (lane 31; adapt): A and B take the newest take of the language, also when `voice_film_examples` holds an older
  pin for it (seeded by SQL, never through `set_film_example`, which lane 32d removes; changed, t40); `report.json`'s `prompt_takes` reads `{"example_id", "rule": "newest"}`; C still sends `use_clip=False`; the t38 blind-output
  tests and the read-only test pass unchanged.
- `tests/test_modelsvc_api.py`, `tests/test_models_client.py`, `tests/test_voxcpm_voice.py` (lane 31b; append): a reference with
  `use_clip=false`, an adapter and no `audio` → 200, and the backend gets a reference with no `audio` key; `use_clip` true (or absent) with no
  `audio` → 422 "the reference of '<token>' needs its audio"; `use_clip=false` with `audio` → its sha256 is checked as today; every existing
  test passes. The client sends no `audio` and no `sha256` for a reference without `audio`, and the same body as today for one with it. The
  child, through the `loaded` seam, speaks a `use_clip=False` reference that has no `audio` key: no `build_prompt_cache`, no conditioning
  entry, the adapter `loaded`. Added (t40): `tests/test_modelsvc_backends.py` — `FakeBackend.speak` with a `use_clip=False` reference and no
  audio returns no conditioning entry for that token and `adapters[token] == {"state": "loaded", "sha256": <sent>}`, and a plain reference
  still gets today's `built` entry; `tests/test_models_client.py` — `ModelsClient.speak` with `references={"saved:1": {"language": "ru",
  "adapter": {"blob": b"…", "sha256": …}, "use_clip": False}}` against `create_models_app(FakeBackend())` → 200, audio, no `conditioning`
  entry and that `adapters` entry.
- `tests/test_voice_bake.py` (lane 32a; new; a fake client): the tag holds `source: "app"`, `owner_id`, `voice_id`, `bake_key`; cancel while the
  fake publication waits on an Event under `job_lock` → the cancel waits and the model stays stored; cancel before the publication → nothing
  stored; Cancel and then a `done` or `failed` end → the wait ends, nothing is stored, `bake_delete` called; `bake_delete` on done, failed,
  cancelled and a 404; `attach=` reads the snapshot from `voice_bake_runs` after the current takes changed and makes no `bake_start`
  (**changed 2026-11-03, t48:** "a replacement start joins the old run first" moved to `tests/test_web_voices.py`, `VoiceBakeJobs`); unresolved → `BakeBlocked` and no `bake_start`; the voice lock is held from the first stage to
  the end and released after done, failed and cancelled; a held lock → "waiting for the voice model" and Cancel works; progress mirrors the
  fake service's stages; cancel → one `bake_cancel`, `Cancelled` only after the fake reports `cancelled`; 404 → the restart text; done → the
  model stored and `bake_delete` called; a take deleted during the bake → the model IS stored (D79). A 409 from `bake_start` → `BakeBusy`
  with the fake's current status; with `wait_for_other=True` → "waiting for the other bake to stop", `bake_current` polled until it ends,
  then one `bake_start`; Cancel during that wait → no `bake_start`; `bake_delete` is never called on the other bake.
  Added (t40): `attach=` after EVERY take of the snapshot was deleted and drained → no `BakeBlocked`, no take read, the model stored with
  the snapshot's `voice_model_takes`; `attach=` after a new unresolved take was added → the same; a new start with that unresolved take →
  `BakeBlocked` and no `bake_start`; a run row with `cancel_requested_at` set when the fake reports `done` → `Cancelled`, nothing stored,
  `bake_delete` called; a cancel request written while the publication holds `job_lock` → it waits, the store wins, the model stays.
- `tests/test_feed_build.py` (lane 32b; append and adapt): a baked voice → the reference is exactly `model_reference` (no `audio`,
  `transcript` or `conditioning`); one `speak` call; no `save_conditioning`; `decisions["voice"]["models"]` and `cast.narrator_model ==
  "voice model v1"`; a model with changes since it is still sent; no model → `failed` with the `none` text and no speak call; another
  revision → the `revision` text; a model with no English takes and an English film → the `language` text; never a preset or Automatic speak;
  a bracket voice with no model is not offered to the story pass; a training take deleted after the references were read → the film is
  READY; the voice deleted after the references were read → `failed` at the ready save with no film audio; a reply whose `adapters` lacks the
  token, reports `none`, or has another sha256 → `failed` with the D86 text and no film audio.
- Lane 32b (added t40): `tests/test_feed_build.py`, `tests/test_voice_preview.py` and `tests/test_feed_store.py` import none of the names lane
  32d removes; `feed/build.py` and `feed/voice_preview.py` neither.
- `tests/test_feed_store.py` (lane 32b; append): `film_narrator` gives `model == "voice model v3"` from `cast.narrator_model`; a saved-voice
  film with no `models` → "made from one take, before voice models"; a preset film → no model text.
- `tests/test_voice_preview.py` (lane 32b; rewrite the per-take tests): `preview_text` — a sentence of the language, an unknown sentence id, a
  take's words, a take of another language or a pending take ("Unknown take."), a take with no words, own text (blank, 301 characters), an
  unknown kind, the old `reading` kind refused; `run_preview` sends `model_reference` and stores a `voice_tests` row with the version and
  sha256; no model → `PreviewFailed("Bake the voice model first.")` and no speak; another revision and a missing language → their texts; a
  model stored as v2 while the test waited for the voice lock → the test speaks through v2 and stores `model_version == 2`; `save_test`
  answering `None` → `PreviewFailed` with its text; cancel before the call → no call; a cancel during the publication waits (job lock); the
  D86 reply check; a `take` test whose take is deleted during the call → stored with `example_id` NULL.
- `tests/test_feed_build.py` (lane 32b) also: a model stored as v2 while the film waited for the voice lock (the v1 file already removed) →
  the film re-resolves, reads v2 and records "voice model v2".
- `tests/test_web_voices.py` (lane 32c; append and adapt): **(t48)** a replacement start joins the old run first (`VoiceBakeJobs`, a bake part); the leaf and the cards hold no `.example-preview`, no `.film-take-select`, no
  `template.take-compare`, and none of the texts "films use take", "Films: take", "one example per language", "chosen for films"; the old
  routes `/voices/{id}/film-example`, `/voices/{id}/examples/{eid}/preview` and `…/preview/audio` → 404 or 405; the header for no model, a
  usable model, changes ("2 takes added, 1 deleted since v3 — Rebake to include them"), another revision, and a bake of this voice; the card
  line for each state; Voice model: the snapshot line, the changes line, "Bake voice model" vs "Rebake", disabled with the unresolved list, the
  question for another app bake and for a foreign bake, the meter with the authors' figure and never the word "enough"; Test: the language
  options are the model's languages, the text options per language (sentences, "Words of take #N (checked)" / "(not checked)", "Your own
  text"), no model → "Bake the voice model first." with the Bake form and a disabled Generate; `POST /voices/{id}/preview` and its codes;
  `GET /voices/{id}/preview/audio` and its 404; the comparison names the stored test's model version and, for a take's words, the take's
  ▶ Original or "the take was deleted"; the kit rows, the words box, the transcript route; the bake routes and codes (`NOT_AVAILABLE`,
  `BAKE_NOT_OFFERED`, `BakeBlocked`, `replace=yes`); `VoiceBakeJobs.reattach` adopts a fake running bake with `source: "app"` and leaves a
  fake current bake with tag `{"check": "me"}` alone (no adopt, no cancel, no delete); `replace=yes` on a foreign bake → `bake_cancel` on it and
  never `bake_delete`; a voice delete cancels its bake; the t27 certificate tests pass with the new job fields.
  Added (t40): Cancel, `replace=yes` and a voice delete each set `cancel_requested_at` on the running run's row before their reply returns
  (read the row inside a fake `bake_cancel`); an app restart after a Cancel (a new `VoiceBakeJobs` on the same DB, the fake service reports
  `running` and then `done`) → `reattach` calls `bake_cancel` and `bake_delete`, stores nothing, ends the run row; an app restart after
  `replace=yes` → the old run (flagged) is cleaned up with no store, and the new run is adopted and stored once; store wins — the cancel
  arrives while the publication holds the job lock → the reply names the new version, no flag is written, and a later restart finds the
  `bake_key` stored and makes no second version. The bake routes answer only once 32a and 32b are GREEN (the lane order; no test).
  `tests/test_web_voices.py` and `tests/test_web_feed.py` import none of the names lane 32d removes.
  - `voices.js` under node (`_js_function`): `testTarget` and `syncTest` with fake nodes — the options are filtered by language, Generate is
    disabled without `data-model-ready`, the words show for a sentence and a take, the textarea only for own text; `generatePreview` posts
    `/voices/<id>/preview` with `language`, `text_kind`, `text_id`, `text`; `carryInputs` keeps `.test-language` and `.test-text`. Source checks:
    no `.example-preview` or `.film-take-select` handler; the bake poller captures `own` before its first await and checks `own.alive()`
    before each DOM effect (t36).
- `tests/test_web_feed.py` (lane 32c; append and adapt): the picker — a castable voice enabled; an unbaked voice as a disabled "«me» — bake
  required" option; a stored unbaked choice selected with that label and the line with `href="/voices/<id>?tab=model"`; another revision and
  a missing language with their labels; Prepare with an unbaked voice → 409 "«me» has no voice model yet. Bake it under Voices, or choose
  another narrator." and no build; "This film" shows "· voice model v2", and an old one-take film "· made from one take, before voice
  models".
- `tests/test_one_model_process_gate.py` (run only): `feed/voice_bake.py` imports no model library. `tests/test_migrate.py` (run only).

Slice B, ASR (**added 2026-11-02, ASR, t42**; fakes only — no Whisper, no torch, no transformers, no scipy import in a test, no model load,
no running service, no download). These items replace the Slice B lines above that test confirmed words: lane 31's `add_example(…,
transcript=…)` confirmed / not confirmed / 1001 characters, `set_transcript`, `bake_snapshot`'s `unresolved`, the `edited` and
`unresolved` counts of `voice_model_for`; lane 32a's "unresolved → `BakeBlocked`" and "a new start with that unresolved take →
`BakeBlocked`"; lane 32c's "disabled with the unresolved list", "the words box, the transcript route" and "Words of take #N (checked) /
(not checked)"; lane 32b's "a take with no words" refusal (now the K5 texts).

- `tests/test_saved_voices.py` (lane 31c):
  - schema: after `migrate`, `voice_examples` has `transcript_model` and `transcript_error`, and still has `transcript_confirmed_at` and
    `kit_id`; an existing DB made before them gains the two columns; `_ADDED_COLUMNS` lists them; `SCHEMA_VERSION == 14`.
  - `add_example` has no `transcript` parameter (`inspect.signature`); a recording with `prompt_id="ru-slow-1"` → `kit_id == "ru-slow-1"`,
    `transcript`, `transcript_model`, `transcript_error` NULL; a recording with no `prompt_id` → `kit_id` NULL; an unknown `prompt_id` →
    `VoiceError` as today; an upload → `kit_id` NULL; `words_state == "waiting"` for all of them.
  - `heard_words`: "  Мои   слова. " → ("Мои слова.", None); "", "   ", "♪ ♪", "..." → (None, `NO_WORDS`); 1001 letters → (None,
    `TOO_LONG`); 1000 letters → kept. **Changed (ASR plan check, t44):** "1 2 3 4 5" → ("1 2 3 4 5", None); `NO_WORDS` reads "the speech
    recognition found no words in it" and no reason holds "speech was heard"; `not_used_text` lists each `unusable` take by id with its
    own reason (`NO_WORDS` and `TOO_LONG` in one text); `take_to_hear` returns the take while it is `waiting` and not pending, and `None`
    for a pending take, a pending voice, a drained take and a heard one.
  - `set_heard_words`: words → `heard` with the model string; "♪" → `unusable` with `NO_WORDS`; a take already heard (words or error) →
    `None` and the row unchanged; a pending take or voice → `None`; a take stored with the shown text and `transcript_model` NULL (seeded by
    SQL) → its `transcript` replaced by the heard words.
  - `takes_to_hear`: only non-pending takes with `transcript_model` NULL, by id; a legacy take is in it.
  - `bake_snapshot`: `takes` = the heard, non-pending takes; `waiting` and `unusable` as their states; `seconds`, `by_language` and
    `dataset_hash` over `takes` only; `snapshot_hash` changes with the heard words and ignores order.
  - `voice_model_for`: no row → `added` = heard + waiting, `unusable` counted apart; with a row: a waiting take → `added`; an unusable take →
    `unusable` only, never `added`; a deleted trained take → `deleted`; `ModelChanges` has no `edited` field.
  - `hasattr(saved_voices, "set_transcript")` is False; `VoiceExample` has no `unresolved` and no `transcript_confirmed_at`.
- `tests/test_voice_bake_check.py` (lane 31c): `--list` prints "heard by <model>@<7>", "shown text, not heard", "no usable words: <reason>"
  and "NO WORDS" for the four kinds of take; a take with no words still needs `--text` (exit 2 without it); the t38 and read-only tests
  pass unchanged.
- `tests/test_catalog.py` (lane 31d): `TRANSCRIBERS` holds one entry, key `whisper-large-v3-turbo`, slot `asr`, `HfSource("openai/whisper-
  large-v3-turbo", revision="41f01f3fe87f28c78e2fbf8b568835947dd65ed9")`, `cost_measured is False`, `profiles == ("mac",)`, `languages ==
  ("en", "ru")`; `"asr" not in SLOTS`; no `CATALOG` entry has slot `asr`; `transcriber_for("mac")` is that entry, `transcriber_for("cloud")`
  and `("jetson")` are `None`; its `cost_mb` + the headroom fits the mac budget (the rule of `test_every_model_fits_its_profile_budget`).
- `tests/test_modelsvc_slots.py` (lane 31d): on mac `asr` is registered with target `audio.whisper_asr:WhisperASR`, args `(repo, device,
  revision)` and cost 4000; `model_cost_mb={"asr": 3500}` overrides it; with its worker running, `apply({"music": <another key>})` leaves it
  registered and running; on jetson and cloud no `asr` unit.
- `tests/test_whisper_asr.py` (lane 31d; new): importing the module imports no `torch`, `transformers`, `huggingface_hub` or `scipy`
  (`sys.modules` before and after); `take_samples` refuses stereo, 44.1 kHz, 8-bit, zero frames, a cut payload and 30.5 s, each with its
  reason, and returns float32 samples of the right length for a 6 s take; `WhisperASR("repo", "cpu", "rev")` raises before any import;
  through `loaded` with a fake model, processor and resample: the processor gets the resampled samples and `sampling_rate=16000`,
  `generate` gets `language="ru"` and `task="transcribe"`, the decoded text comes back unchanged with the revision and the device;
  `language="de"` → `ValueError` and no `generate` call.
- `tests/test_voice_bakes.py` (lane 31d; append): the start order adds `governor.unload("asr")` right after `governor.unload("voice")` and
  before `lease("voice_train")`; with no `asr` registered the bake runs as before.
- `tests/test_modelsvc_backends.py` (lane 31d; append): `transcribe` with the gate closed → `BakeRunning` and no scheduler call; it runs
  `["asr"]` at `Priority.BATCH`; a child reply with another revision → `RuntimeError` "asr identity mismatch"; no `asr` unit →
  `TranscribeNotOffered`; `language="de"` → `TranscribeRefused`; a call that loads the child logs one line with both `free_mb` values, a
  call to a resident child logs none (**changed, ASR plan check, t44:** the line reads "loaded by this request" and "net change <N> MB";
  with a fake governor that evicts `music` during the call, it ends "evicted: music", and with no eviction "evicted: none"; it never
  holds "cost", "drop" or "measured"); `model_ensure("asr")` enters the gate and answers 409 while it is closed; `FakeBackend.transcribe`
  returns its scripted text.
- `tests/test_modelsvc_api.py` (lane 31d; append): `POST /voice/transcribe` → 200 with `text`, `model`, `revision`, `seconds`, `elapsed_s`;
  409 during a bake; 422 for bad base64, an sha mismatch, a 5 MiB clip, a stereo WAV, a 31 s WAV and `language="de"`, with no backend
  call; 503 with the "not offered" text.
- `tests/test_models_client.py` (lane 31d; append): `transcribe` posts base64 audio, its sha256 and the language to `/voice/transcribe`
  with timeout 600 and decodes the reply; one HTTP test through `create_models_app(FakeBackend())`; a 409 raises `httpx.HTTPStatusError`.
- `tests/test_fetch_models.py` (lane 31d; append): on mac `wanted` lists `("asr", "openai/whisper-large-v3-turbo@41f01f3fe87f28c78e2fbf8b568835947dd65ed9")`;
  on jetson and cloud it does not.
- `tests/test_voice_bake.py` (lane 32a; a fake client with `transcribe`): the `waiting` takes are sent one by one, by id, each with its
  language, and each reply is stored through `set_heard_words`; a `heard` or `unusable` take is never sent; a reply "♪" → that take
  `unusable`, the bake goes on without it, and the done text names it with its own reason; every take `unusable` → `BakeBlocked` with the
  "No take can be learned from: …" text naming each take and its reason (changed, t44), no `start_bake_run`, no `bake_start`; Cancel before the second call → no second call, `Cancelled`, the first take's words stored, no run
  row, no `bake_start`; a fake 409 from `transcribe` → `BakeBusy` with the fake's current status; a foreign bake running before the stage →
  `BakeBusy` and no `transcribe`; with `wait_for_other=True` → "waiting for the other bake to stop", then the stage; a 503 → the "not
  offered" text; any other error → "Could not hear the words of take #N (…)" and the earlier takes' words stay; the clips sent to
  `bake_start` carry the stored heard words exactly, and the fake service's `dataset_sha256` computed the service's way equals
  `snapshot.dataset_hash`, so `save_voice_model` stores the model; a Cancel sent while the job holds `job_lock` for `start_bake_run` waits,
  and afterwards either no run row exists or the row carries `cancel_requested_at` — never a row without the flag; progress: 0.01-0.15
  during the stage, then `0.16 + 0.81 * fraction`; the voice lock is held from the stage to the end; `attach=` makes no `transcribe` call;
  no take at all → `BakeBlocked("No take yet. …")` before the lock.
  **Changed (ASR plan check, t44):** a take added while the last transcribe call runs (the fake client adds it inside `transcribe`) →
  the read under `job_lock` finds it `waiting`, the job releases the lock, sends it, and the snapshot and `bake_start` hold it; a take
  added after the final snapshot read (inside a fake `start_bake_run` wrapper, after the read) → never sent to `transcribe`, not in the
  snapshot, and `voice_model_for` counts it as added after the store; `job_lock` is never held during a `transcribe` or `bake_start` call
  (the fake asserts the lock is free); a queued take deleted and drained before its turn → no `transcribe` call and no audio read for it,
  the bake goes on; a take deleted while its call runs (the fake requests the deletion inside `transcribe`) → `set_heard_words` writes
  nothing, the take is not in the snapshot, the bake goes on; a fake storage whose read raises for a take that still exists →
  `BakeFailed("The audio of take #N could not be read (…).")`, and for a take whose row is gone → skipped.
- `tests/test_voice_preview.py` (lane 32b): `preview_text("take")` — a `heard` take → its heard words; a `waiting` take → "The words of take
  #N are heard at the next bake."; an `unusable` take → "Take #N has no words to speak: <its reason>." (changed, t44; for `NO_WORDS` and `TOO_LONG`); a take stored with the shown text and no
  `transcript_model` → the waiting text, never its shown text.
- `tests/test_web_voices.py` (lane 32c): the leaf holds none of "What you said", "words checked", "check the words", "needs its words",
  "need their words", "Save words", "(checked)", "not checked", "Your checked words"; `POST /voices/{id}/examples/{eid}/transcript` → 404 or
  405; the add route stores no transcript even when the form carries a `transcript` field; an `unusable` take's row reads "not used by the
  voice model: the speech recognition found no words in it", and a `TOO_LONG` take's row its own reason (changed, t44); no text of the
  leaf holds "no speech"; "Other takes" holds the uploads and a recording with no `kit_id`; the recorder holds "Read it, or say anything
  else: the bake hears what you say."; the Voice model tab's "Not used: take #N — <reason>; …" line with each take's own reason, the two disabled texts and the
  enabled button with only `unusable` takes beside heard ones; the job line "Baking v<N> · transcribing the takes · take 2 of 5 · N%"; the
  meter "Your recordings:"; Test options "Your words in take #N" for heard takes only; the header counts never say "edited".
  The bake parts land after 31d (the lane order; no test).
- `tests/test_one_model_process_gate.py` (run only): `feed/voice_bake.py` and `web/voices_api.py` import no model library.

Phase 6 (§2.18; added 2026-11-03). Fakes only: the fake client of `tests/test_feed_build.py` whose `speak` accepts `references`, and its
`bake` helper (t46: fake ASR words, `save_voice_model` with fake adapter bytes). No model, no service.

- `tests/test_feed_voice.py` (lane 36a): `automatic_for_prompt("en", [])` equals `catalogue_for_prompt("en")`; with
  `[SavedCast("saved:1", "Grandma")]` it is the preset lines, then the SAVED VOICES line of §2.18 B, then "  Grandma — a saved voice: this
  person's own voice", in that order.
- `tests/test_feed_build.py` (lane 36a; append and adapt; `profile="mac"` unless the item says otherwise):
  - Automatic, «Grandma» baked with English takes, an English film: the story pass's `speakers` hold every preset line and the Grandma line.
    The story names "Grandma" → one `speak` with speaker `saved:<id>` and `references == {"saved:<id>": model_reference(…)}` (no `audio`);
    `decisions["cast"]`: `narrator_chosen_by == "story"`, `narrator_label == "Grandma (saved voice)"`, `narrator_model == "voice model v1"`,
    `offered` = the preset names, then "Grandma"; `decisions["voice"]["saved"] is True`.
  - The same film, and the reply has no adapter entry for the token → the film fails with "The models service did not speak «Grandma»
    through its voice model." (D86).
  - A saved Automatic narrator and a bracketed passage: its story voice is a preset → that passage reads as the narrator, and the call has no
    preset name; its story voice is a second baked voice «Dad» → that passage speaks `saved:<dad>`, and `references` hold both models.
  - A preset Automatic narrator and a bracketed passage whose story voice is "Grandma" → that passage reads as the preset narrator; no
    `references`; no voice model file is read.
  - The story names no real voice, with «Grandma» baked → `pick_speaker`'s preset, `narrator_chosen_by == "mood"`, no `references`.
  - Presets only: `speakers == catalogue_for_prompt(language)` exactly, and a story that names "Grandma" gets the mood preset, for each
    of: no saved voice; «Grandma» baked with Russian takes only, for an English film; «Grandma» with no voice model; `profile=None`;
    `profile="jetson"`; `voices=None`.
  - A baked voice named "af_heart" is not in the `speakers` of an English film, and the story's "af_heart" is the preset.
  - The picked voice is deleted after the story pass, before the speak (the fake client deletes it in its story reply) → the film fails
    with the text of `_cannot_narrate(…, "deleting")`, and no `speak` call runs. Deleted after the speak → "The saved voice «Grandma» was
    deleted while this film was made.", and no WAV is written.
  - Beside `test_a_model_rebaked_while_the_film_waited_is_the_one_it_speaks_through` (`tests/test_feed_build.py:1644-1668`), the same
    waiting-lock Rebake for an Automatic film whose story picked «Grandma»: a successful replacement → the film speaks through v2 and
    records `voice.models[…]["version"] == 2` and "voice model v2"; a replacement that loses the film's language → the film fails with
    the `language` text of `_cannot_narrate`, and no `speak` call runs. No other model-change failure is tested or added.
  - Full-build regressions for a chosen preset and a chosen saved voice (the branches of `feed/build.py:356-369`, with `saved_pick =
    None`): each builds `ready`, and `decisions["voice"]["saved"]` is False for the preset and True for the saved voice.
  - The t50 chosen-narrator tests stay as they are (`speakers == ""`).
- `tests/test_web_feed.py` (lane 36b; adapt and append): `AUTOMATIC` is "Automatic — the app picks a voice for this memory";
  `test_the_picker_hints_how_the_voice_is_chosen` reads the voice-copy hint and the new Automatic bracket line of §2.18 C; on jetson
  (`voiced_jetson`) the hint is "The story picks a preset voice; otherwise the memory's mood guides the choice."; no page holds "Saved
  voices are never picked automatically."; a Prepare passes `profile` equal to the app's profile to `build_presentation` (a stub on
  `web.app.build_presentation` records its keywords).

Phase 7 (§2.19; added 2026-11-03; **changed 2026-11-03, plan check t63**). Fakes only: `FakeModels` of `tests/test_feed_build.py` and
the app's fake models of `tests/test_web_feed.py`. No model, no service. The JS items run under node, skipped when node is missing (the
`_node` and `needs_node` pattern of `tests/test_web_voices.py:1790-1802`), with a small fake DOM and a `fetch` stub whose replies the test
releases one by one.

- Lane 39b, the order fix (§2.19 I; pair t64, merge 3 point 6). An expected photograph is always the one of the SELECTED mapping (the
  chosen Story's, or the accepted candidate's), never a relabelled one.
  - `tests/test_feed_story.py` (append and adapt):
    - `introduces_in_order`: `[0, 1, 0, 2]` True; `[2, 0, 2, 1]` False; `[0, 0, 2, 1]` False; `[]` True.
    - `_script` with shuffled, duplicate, malformed and missing sentence numbers, each item with its own voice: the output is in told
      order; each voice stays on its own sentence; the first accepted duplicate wins; a missing sentence takes the previous sentence's
      photo (the first one: the next accepted sentence's) and the voice ""; `matched` is True. A non-list `script`, `[]` and an
      all-malformed list each give `_even` (4 sentences, 3 photos: `[0, 0, 1, 2]`, not `[0, 1, 0, 2]`) with `matched` False (t65).
    - `_cards`: `{"sentence": 3, "text": …}` → `(2, …)`; a card on a repeated passage lands on that passage; a sentence number past the
      last passage is dropped; at most `MAX_TITLE_CARDS`. `STORY_SCHEMA`'s card items require `["sentence", "text"]`; adapt the payloads of
      `:97`, `:468`, `:476` to `sentence`. The stage-2 prompt holds the first-appearance rule and asks for the card's sentence number.
    - `_dress`, a valid script with a return (`[0, 1, 0]`): one stage-2 call, no second one, `memory_order == "kept"`.
    - `_dress`, an answer `[2, 0, 2, 1]` (4 sentences, 3 photos) and an ordered candidate `[0, 1, 0, 2]`: the second prompt holds the
      CRITICAL order line and the same numbered sentences; the result has the candidate's photographs, the first answer's texts,
      voices, cards, score, narrator and music prompt, and `memory_order == "retried"`.
    - `_dress`, the same answer with each of: a candidate call that raises; a non-JSON candidate (the `{}` fallback, `_even`); a
      candidate that no longer shows a photograph the first answer shows; a candidate still out of order → the result `==
      replace(selected_story, memory_order="broken")` (t65: `memory_order` defaults to "kept", so the plain Story is not equal).
    - `_dress` with a `[voice]` note: the voice retry's answer is chosen for its cast and is out of order; an accepted order correction
      keeps that cast.
  - `tests/test_feed_director.py` (append): `with_script` keeps the given order (`[(2, a), (0, b), (2, c), (1, d)]` → the shots show
    photos 2, 0, 2, 1) and drops an index out of range; `feed.director` has no `_in_memory_order`. `with_arrangement` with a shuffled
    conductor answer (shot numbers `[2, 4, 1, 3]`): with `keep_order=True` and `keep_timing=False` the shots keep their positions and
    the conductor's times are monotonic; with both False the order is `_resolve_order`'s, as today; with `keep_timing=True` the times
    are the shots' own (t65).
  - `tests/test_feed_build.py` (append and adapt):
    - A fake whose stage-2 answer is out of order and whose corrective answer is still out of order: `models.segments` are the told
      sentences in order; shot `i`'s `photo_id` is `photo_ids[script[i][0]]` of the chosen answer; the score section that starts at a
      passage's `from` starts at that passage's segment; a bracket-gated voice speaks its own passage; the conductor's brief `i` is the
      brief of passage `i`'s photograph; `decisions["story"]["memory_order"] == "broken"`.
    - The same fake with an ordered corrective answer: the shots follow the candidate's photographs, the words do not move, and
      `memory_order == "retried"`.
    - Adapt `test_the_film_follows_the_memorys_photo_order` (`:184-193`): the in-order fake gives `memory_order == "kept"`; its comment no
      longer claims a guarantee. `test_every_shot_starts_when_its_own_words_are_spoken` (`:203-209`) also asserts the `photo_id`s.
    - No segment pairing (t65): a fake whose speak returns no segments, and one whose segments are one fewer than the passages, each with
      a shuffled conductor answer: the shots keep the selected passages' photographs in told order, a card stays on its sentence's
      shot, the shot times are monotonic, and every `timeline.passages` time is None.
    - The fake's card at `:86` becomes `{"sentence": 1, "text": "The water"}`.
  - `tests/test_web_feed.py` (the one line `:64`, t65): the fake's card becomes `{"sentence": 1, "text": "The water"}`; run
    `test_the_story_is_a_narration_with_sparse_title_cards` (`:408-413`), which expects `["The water", "", ""]`.

- `tests/test_feed_narration.py` (new, lane 40a):
  - `timeline_script` pairs chunk `i` with segment `i`: for a script `[(2, "A"), (0, "B"), (1, "C")]` and segments
    0-2, 2-4, 4-6 s, the passages are A, B, C with starts 0, 2, 4 and photos 2, 0, 1, and their `photo_id`s are `photo_ids[2]`,
    `photo_ids[0]`, `photo_ids[1]`. `voice_asked` and `voice_used` come from `story.voices` and `passage_voices` (or the speaker).
  - Two segments for three chunks → every `start` and `end` is None, and the three texts stay.
  - `timeline_script` names each card by its passage: a card `(1, …)` is stored as `{"n": 1, …}`; a card `(5, …)` of a 3-passage
    script is dropped. Nothing sorts it (§2.19 I).
  - `narration_rows`: a ready film with passages → `film`, revision 0, the film's token, `draft` None, `seekable`, not `changed`; a ready
    film without → `legacy`; a `failed` or `running` row, or no row → `none`.
  - The first `save_passage` seeds ALL passages and the film's script (without `token`) at revision 1, with a new 32-hex `draft_token`
    and the film's token as `source_token`, and changes only passage `n`; the rows are `changed`. The next save makes revision 2. A save
    that puts the old text back makes the draft not `changed` again.
  - The stale rule: a wrong `source`, a wrong `draft`, a wrong `revision`, revision 0 when a draft exists, and any save in state `none`
    raise `NarrationRefused("stale")` and write nothing.
  - "Write a new story" and a new seed of the same film: `discard_draft`, then a save on the film rows → a new `draft_token` at revision
    1; a save that still posts the old `draft_token` at revision 1 raises `stale`.
  - A 1 000-character passage is stored whole (no 600 cut). The sum over 10 000 characters raises `too_long` with "The narration is
    longer than this app allows: <N> characters, at most 10000.". "[whisper] it was late" raises `bracket`. "  two\n\nlines  " is stored
    as "two lines". An unchanged text writes nothing: no row before → still no row.
  - An empty text removes passage `n` and keeps the others' `n`; removing the last one raises `empty`; an unknown `n` raises `missing`.
  - `discard_draft` deletes the row: with a ready film the film rows come back (state `film`); with a `failed` row the state is `none`.
  - Languages are separate: an `en` draft leaves `narration_rows(…, "ru")` as it was, and the reverse.
  - Note isolation: after saves and removes, `memory_notes` has no row, and `notes_for_prompt(read_notes(…))` holds no passage text.
  - `draft_story`: the Story's `narration`, `script`, `voices`, `narrator` and `music_prompt` are the draft's; a `photo_id` no longer in
    `photo_ids` takes the previous passage's photo (the first passage: 0); over a draft of passages `n` = 0, 1, 3 (passage 2 removed), the
    plan `[{from 0}, {from 2}, {from 4}]` becomes `((0, …), (2, …))`: position 2 is passage 3, and `from 4` is dropped.
  - `draft_story` cards: when passage 1 is removed, the card of passage 3 moves to position 2, where passage 3 now is; the card of a
    removed passage is dropped.
  - `draft_story` `memory_order`: a draft `[1, 0]` (the passage that introduced photograph 0 was removed) keeps its photographs and
    gives "broken"; an in-order draft gives "kept".
  - `rebase_consumed`: with the same `draft_token` and revision, the row takes the new passages (`n` from 0, new times), the new
    `source_token` and the new script, at revision + 1, and keeps its `draft_token` and its texts; with revision + 1 already stored, or
    another `draft_token`, it changes nothing and answers False.
  - The table cascades: deleting the memory's `groups` row deletes its drafts. `migrate` on a database without the table creates it
    (connect, drop the table, migrate again).
- `tests/test_albums.py` (append, lane 40a): `rewrite_memory` deletes the drafts of the memory in every language and keeps its notes and
  its narrators; another memory's draft stays.
- `tests/test_feed_build.py` (lane 40b; append and adapt):
  - Adapt `test_the_edit_replaces_the_old_timeline_shape` (`:176-182`): the keys add `passages` and `script`.
  - A film's `timeline.passages` match the speak call: each `text` is `models.segments[i]` and each `start`/`end` is the fake's segment
    `i`, in telling order. With a fake whose stage-2 reply puts the sentences on photos 3, 1, 2 and whose corrective reply is still out
    of order, passage 0 has the first sentence, start 0.0, `photo` 2 and `photo_id` 3, and `timeline.shots[0]` shows photo 3 too: shot
    `i` and passage `i` name the same photograph (§2.19 I).
  - A draft `[1, 0]` after a removed introduction: zero story calls; every `photo_id`/text pair stays; `decisions["story"]
    ["memory_order"] == "broken"`.
  - `timeline.script` holds a 32-hex token, the cards by passage, the score plan and the music prompt of the story; two builds give two
    tokens.
  - With a draft (seeded through `save_passage`): no prompt holds "WHAT THE DAY WAS" or "EACH PHOTOGRAPH" (the story pass did not run);
    the conductor prompt ("THE PHOTOGRAPHS available to you") is sent; `models.spoken[-1][0] == " ".join(draft texts)` and
    `models.segments == draft texts`, exactly; `decisions["story"]["source"] == "draft"` with the draft's revision.
  - The rebase: after that build the draft row still exists, with the same `draft_token` and texts, revision + 1, the new film's token
    and passages; `narration_rows` reads state `draft`, `seekable`, not `changed`. A second build with no change speaks the same words
    again and runs no story pass.
  - With a draft and a chosen preset narrator: `spoken_voice` is that preset and `spoken_voices == []`. Automatic with
    `narrator_asked == "am_adam"`: that preset narrates and `narrator_chosen_by == "story"`. Automatic with `narrator_asked` "Grandma" and
    «Grandma» baked for the language: a saved-voice film with the model reference (the Phase 6 fake). The same with «Grandma» deleted:
    the mood preset narrates.
  - With a draft and a `[voice]` note at the second passage, and `voice_asked` "am_adam" there: that passage speaks `am_adam`. With the
    note at another passage: no passage is cast, and `error` holds "a note asked for a different voice and none was cast".
  - Cards: a draft that removes the passage before a card's passage → the card is on that passage's shot in the new film; a draft that
    removes the card's own passage → no shot of the new film has that card's `line`.
  - Failed: a draft, then a build whose speak raises → a `failed` row with no `passages`, and the draft row is unchanged (`draft_token`,
    `source_token`, `revision`, `script_json`, `passages_json`). A retry with a working fake builds `ready` from the draft alone: the speak
    text is the draft's, the card's `line` is on its passage's shot, `decisions["story"]["score_plan"]` is the stored plan, and the music
    brief holds the stored `music_prompt`.
  - Cancelled: a `cancelled` that turns True after the speak call → `Cancelled`; the draft row is unchanged and the ready film stays.
  - A newer revision survives: a fake whose `speak` calls `save_passage` (revision R → R + 1) → the build is `ready`, it spoke revision
    R, and the draft row is exactly the R + 1 row (the earlier film's token and times: not `seekable`, `changed`).
  - An old build after "Write a new story" and a new seed of the same film: a fake whose `speak` calls `discard_draft` and then
    `save_passage` on the film rows → the build is `ready`, and the new draft (new `draft_token`, revision 1) is untouched.
  - A saved-voice film with a draft whose voice is deleted before the save → it fails with the existing text, and the draft stays.
  - No draft: the story pass runs as today, and a film with passages plus one note asks for `narration_words(3, 1)` words (the told
    prompt holds "Write <low>-<high> words"): no passage counts as a note.
- `tests/test_web_feed.py` (lane 40c; append and adapt):
  - Adapt `test_the_edit_carries_camera_moves_and_transitions` (`:393-396`): the keys add `passages` and `script`.
  - Automatic rows on load: after `_prepare(client)`, the page's `#narration-data` JSON parses (the `#notes` test pattern), state
    `film`, the rows' texts are the film's passages in order with their stamps, and the page holds the `film` state text. No click and
    no import.
  - Unique ids: the ready page has exactly one element with `id="narration"` (the audio) and one with `id="narration-data"`, and no id
    appears twice.
  - A ready row with no passages (`_film`) → state `legacy` and the legacy text; no row → state `none` and no state text shown.
  - The save route: 200 seeds a draft and answers state `draft`, revision 1, a `draft` token and `changed`; then the page shows "Your
    edited narration. ↻ Regenerate speaks it as written.", the disclosure line, "Write a new story", both titles "Speak your edited
    narration, and score this memory again", and the draft twins of the picker line and the legend lead.
  - Each refusal answers its status and its exact `message`: stale 409 (an old revision, a second tab's source, an old `draft` token),
    empty 400, bracket 400, too long 400 (the app-limit text), missing 404; an unknown memory 404.
  - The delete route removes a passage; the discard route brings the film rows back; `GET /feed/{key}/narration` answers the payload.
  - `lang=ru` saves leave the English rows as they were.
  - A narration save adds no `memory_notes` row, and the `#notes` JSON stays `[]`.
  - With a draft, the memory card's retell text ends "Deletes your edited narration in: English."; without one it is as today
    (`:308-314`).
  - Under node, `feed-notes.js` and the prepare handler of `feed.html`:
    - `window.feedNotes.all()` returns the notes only when `#narration-data` has rows;
    - a `changed` draft marks the Regenerate button `is-stale` at load; a not-`changed` draft and the film rows do not;
    - two quick saves post the second with the `draft` and revision of the first reply;
    - a delayed save and an immediate Regenerate: the prepare `fetch` runs only after the save's reply; a refused save → no prepare
      `fetch`, and the "Not started" line shows;
    - typing in row B while row A saves: A's reply re-renders, and B keeps its element, its text and its focus;
    - a conflict with queued edits: A answers 409 while B and C wait → B and C are never posted, every passage field is read-only and
      keeps its text, and only "Load the current narration" re-renders the rows;
    - a network failure: the save's `fetch` rejects → the row keeps its text, the queue behind it is dropped, `settled()` answers False,
      and a prepare submit posts nothing;
    - a reset during pending saves: "Write a new story" pressed while a save waits → the discard route is posted after that save's
      reply, with its revision; after a refused save it is not posted.
- `tests/test_one_model_process_gate.py` (run only): `feed/narration.py` imports no model library.

## 6. Manual checks for the user (mac, the user's own stack)

The user's `make up` runs from the source checkout. The user decides how to run the branch: for example, stop that stack and run `make up` in `<base>/ivms777`,
or check out the branch in the source checkout. The first `make models` installs `voxcpm` 2.0.3 into that checkout's venv and downloads `openbmb/VoxCPM2@32279ef…` (≈ 4.7 GB).

### 6.1 Checkpoint after phase 1 — GPU validation in the models service (findings §10 subset)

1. Record a clip of 12-15 s in a quiet room (QuickTime → New Audio Recording). Read the en prompt of §2.7 once. Convert it:
   `afconvert -f WAVE -d LEI16@48000 -c 1 in.m4a ~/voice-en.wav`. Do the same with the ru prompt → `~/voice-ru.wav`.
2. `echo "$PYTORCH_ENABLE_MPS_FALLBACK"` prints an empty line in the shell that runs `make up`.
3. `cd <base>/ivms777 && uv run --no-sync python -m scripts.voice_check --wav ~/voice-en.wav --language en --cross`, then the same with `voice-ru.wav --language ru --cross`.
4. Expected in `report.json` and the printed table:
   - every reply: `model = voxcpm2`, `revision = 32279effe8c19989596f05d353d1447f51d9e915`, `sample_rate = 48000`; no VOID;
   - duration > 0, RMS above silence, no NaN; segments monotonic for the 5-passage call;
   - pass 1 `built`, pass 2 `restored`.
5. Expected in the models service log: `Running on device: mps, dtype: float32` (`model/voxcpm2.py:176`) and no CPU-fallback warning.
   The `voxcpm` import and GPU generation work without `funasr` installed: no `ImportError` in the models service log.
6. Record the numbers: RTF per call, the resident-cost drop, the first-load time.
7. Listen to each `A`/`B` pair before opening `key.json`. Score similarity to your voice (1-5), naturalness (1-5), wrong or missing words per sentence, and "same voice in A and B".
8. PASS needs (findings §10 / D12, mac part): same-language mean similarity ≥ 4 and naturalness ≥ 4 in en and in ru; ≤ 1 wrong word per sentence;
   A and B judged the same voice; RTF ≤ 1.0. Cross-language results are reported only.
9. Tell the main session PASS or FAIL with the report path. FAIL stops the task for your decision.

### 6.2 Final check after phase 2

1. `/organize` (any view) shows "Saved voices" beside, not inside, the "Organize photos" selector. Open it. Esc returns to the same Organize view and scroll.
2. Create "Me". A second "me" is refused with a message.
3. Add an English example with Record: the consent box is required; the en prompt shows; Esc during recording cancels it and the page stays; record again, Stop, preview, Save → listed as en, ~12 s, recorded.
   Esc with an unsaved take discards it.
4. Add a Russian example with "Upload a file" (an `.m4a` or `.wav`) → listed as ru, uploaded. ▶ plays both examples.
5. Open a memory film (English). The picker shows "Automatic" and "Me". Choose "Me" → Prepare. The bar moves with a countdown during "recording the voice". The film speaks in your voice.
6. Regenerate: "Me" is preselected. Switch to Русский: "Me" is offered (ru example). Prepare → Russian narration in your voice.
7. Bracket: create "Me 2" with an English example. In an English film narrated by "Me", add a note `[Me 2's voice]` at one passage and Regenerate → that passage uses "Me 2".
8. Delete the Russian example. Its line shows "deleting…" at once (no ▶, no Delete), then it disappears without a reload.
   In Russian the picker shows "«Me» — no Russian example", selected. Prepare with it is refused with that reason under the form, and the page stays.
   Prepare with "Automatic" narrates with Silero.
9. Regression: an English film with "Automatic" narrates with Kokoro as before; Rebuild on a memory card works as before.
   (**Changed 2026-11-02:** after Phase 3 the card button reads "↻ Retell memory"; §6.3 step 7 checks it.)
10. Restore evidence: `sqlite3 ~/.ivms777/ivms777.db "select group_id, language, json_extract(decisions_json,'$.voice') from memory_presentations"`
    shows `voxcpm2`, the revision, `built` on the first saved-voice film and `restored` after a Regenerate.
11. Note the file keys of "Me" (WAVs and conditioning blobs):
    `sqlite3 ~/.ivms777/ivms777.db "select e.storage_key from voice_examples e join saved_voices v on v.id = e.voice_id where v.name = 'Me' union select c.storage_key from voice_conditionings c join voice_examples e on e.id = c.example_id join saved_voices v on v.id = e.voice_id where v.name = 'Me'"`.
    Delete "Me". The confirm says "Films already narrated with this voice keep their sound until you regenerate them." Confirm.
    The voice shows "deleting…" at once, with no rename, no Delete and no examples controls. Then it disappears without a reload.
12. Those files are gone: `ls -R ~/.ivms777/voices` lists none of the keys from step 11. "Me 2" still plays.
    `sqlite3 ~/.ivms777/ivms777.db "select count(*) from voice_deletions; select count(*) from voice_file_deletions"` prints `0` and `0`.
13. Open the English film of step 5. It still plays in your voice (a finished film keeps its sound until you regenerate it).
    The picker shows "A deleted saved voice", selected. Prepare with it is refused with "The saved voice chosen for this film was deleted…".
    Choose "Me 2" → Prepare → the film speaks in that voice.

### 6.3 Check after phase 3 — organize and film clarity

1. `/organize?by=memories`: each memory card lists "English" and "Russian" lines. A ready film reads "▶ Play English film · ready · <voice> (auto)"
   or "(your choice)"; a missing one reads "Prepare Russian film · not made yet". There is no unlabelled "▶ Play".
2. Click "Prepare Russian film". `/feed` opens in Russian: "Film language" shows "Русский" as current, and the page has no ✓ on it yet.
3. The select "Narrator for the next film" lists "Automatic — the app picks a preset voice for this memory", "Preset voices" (5 Russian voices, each with
   gender, register and a description) and "Saved voices" (only voices with a Russian example). Under it: the Automatic hint, "Add a voice under Organize → Saved voices."
   and "A [voice] note can request another preset voice for one passage." No voice is predicted before Prepare.
4. Leave Automatic → Prepare. When it is ready: "This film: Russian · narrated by <voice> (<gender>, <register>) — picked automatically by the story"
   (or "… using the memory's mood"), "Русский ✓", and "Automatic" is still selected.
5. Choose a preset (for example `aidar`) → ↻ Regenerate. The page reads "This film: Russian · narrated by aidar (male, deep) — your choice".
6. Esc → back to Organize at the same scroll. The card reads "▶ Play Russian film · ready · aidar (your choice)" without a page reload; the English line is unchanged.
   Check the cached return (t19) after a Prepare and after a narrator change; and during a build: "being made…", then "ready · …" by itself.
   Each time: the same scroll, an open notes list stays open, and after several Organize visits through the nav one refresh still runs, not several.
   Optional: a note `[low man's voice]` in that film → after Regenerate, "Other voices: passage N → <voice>" lists the passage that changed.
7. On the card, "↻ Retell memory" has the line "… Makes no film. Deletes the films in: English, Russian." Press it. When it is done, both lines read
   "Prepare … film · not made yet", and `/feed?lang=ru` still preselects the narrator you chose last.
8. A film made before this change still says who narrated it: a preset name, or "a saved voice", with the reason when it was recorded.
   `sqlite3 ~/.ivms777/ivms777.db "select language, json_extract(decisions_json,'$.cast.narrator_label'), json_extract(decisions_json,'$.cast.narrator_chosen_by') from memory_presentations"`
   shows the stored labels and `user` / `story` / `mood`.
9. Layout (t20): at window widths 360, 800 and 1280 px, check the ready screen, paused, during a build and during a ↓ Download export, with long labels
   (the Russian preset list, a long saved-voice name) and with a refused Prepare line. Nothing overlaps; the controls wrap; the settings region scrolls
   inside itself; the Download button stays fully visible while it exports. Before ▶ Play and while paused, the settings region scrolls, and Regenerate
   and the select work; a click on an empty part of the header reaches ▶ Play. Check the paused screen with the notes sheet expanded and collapsed (t21).
10. Chat → Organize → film → Close (t19, "Fix now"): from Chat, open Organize through the nav, open a memory's notes list, open a film, then Close.
    Organize is not reloaded: the same scroll, and the notes stay open.
11. `/organize/voices` → each voice's "Add an example" shows "Record clean, normal speech in a quiet room. Only the newest example in each language is used."
    before Record and Upload, also after the list refreshes (for example after a delete).
    (**Changed 2026-11-02:** after Phase 4 the second sentence reads "Films use one example per language: the newest, unless you choose one with
    Use for films."; §6.4 step 1 checks it.)
12. Reading texts (§2.15 I): tick consent, press Record. The panel shows a text, "Different text" and "Start recording", and the microphone is not on yet.
    Press "Different text" four times: four texts, then the first again. Press "Start recording": the text stays, "Different text" is gone, the timer runs.
    Stop: the same text stays over the preview. Esc discards the take and closes the panel. With no focused text field, the next Esc closes the page.
    Separately, press Record without Start, then Esc: the prepared panel closes without opening the microphone.
    Separately (pair t22), press Start and leave the browser's microphone prompt open: Record and Upload do nothing; Stop, and separately Esc, close the panel;
    a late allow or deny then gives no take and no error, the page stays, and a fresh Record → Start works.
    Separately (pair t23), upload a long file (or Stop a recording) and, while "Preparing the example…" shows, press Record and Upload: nothing happens;
    Esc closes the panel and the page stays; the late result gives no take and no error, and a fresh Record works.
    Record and Save one take, then press Record again: the panel opens on the next text.
    `sqlite3 ~/.ivms777/ivms777.db "select language, transcript from voice_examples order by id desc limit 1"` shows the text you read.
13. Tell the main session PASS, or what looked wrong.

### 6.4 Check after phase 4 — voice workbench

How a voice sounds is your judgment: the page shows no likeness score, and nothing here promises likeness or speed.

Changed 2026-11-02 (t25): steps 1-8 below (steps 6 and 8 in round 2).

1. `/organize/voices`: each example line shows `#<id>`, "Original" ▶ and "Test this example". In each language one example reads
   "used for films (newest)" and still offers "Use for films"; every other example reads "not used for films" and offers "Use for films".
   The tip reads "… Films use one example per language: the newest, unless you choose one with Use for films."
2. Record a second English example with another text. The badge moves to it. Press "Use for films" on the older one: it reads
   "used for films (your choice)" and has no button; the newer one reads "not used for films" and offers "Use for films".
3. "Test this example" on the example that reads "used for films". The panel offers EN-1 to EN-5 and "Your own text", and, for a recorded example,
   "The text you were asked to read (not checked against the recording)" with the text. Choose the reading text (EN-1 for an uploaded example) →
   Generate. A stage and a percentage move, with Cancel. Then under the example: "Generated" ▶ and the line "“<text>” · voxcpm2 @ 32279ef ·
   made in N s · conditioning built · <date>" — or "conditioning restored" when a film or a test already used this example. Play Original, then
   Generated.
4. Test the same example again with EN-2: the line ends "conditioning restored", because step 3 left a stored conditioning.
5. An uploaded example offers no reading text. Type or paste more than 300 characters in "Your own text": the box stops at 300 characters
   (`maxlength=300` blocks the 301st). A short sentence works.
6. Start a test with EN-5, then press Cancel: the panel is free at once. Cancel does not stop the model: a generation that already started may
   finish in the background. The old "Generated" line stays. Start another test at once: it may read "waiting for the voice model" until the
   earlier generation ends, then it runs. When Cancel lands just as a test is stored, that test's new "Generated" line stays: it was stored first.
7. A film and a test at the same time: Prepare an English film narrated by this voice; while its bar reads "recording the voice", start a test with
   EN-5. The panel reads "waiting for the voice model". Press Cancel: the panel is free at once. The film goes on and becomes ready; it does not
   fail. Then start a test with EN-5 and, while it runs, press ↻ Regenerate on that film: the film's bar reads "waiting for the voice model" until
   the test ends, then the film becomes ready.
8. App restart (changed 2026-11-02, t25 round 2): keep the page open and type a text in "Your own text"; finish a test. In the checkout your
   `make up` runs from, run `touch web/voices_api.py` (the app hot-reloads). Test another example: its "Generated" line appears without a page
   reload, and the typed text stays. Start a test and run the same `touch` while it runs: the panel reads "The app restarted, so this test
   stopped. Generate it again."
9. Start a test. While it runs, delete another example: the list shows "deleting…" and the panel keeps its progress. Press Esc with no text field
   focused: the page closes. Open Saved voices again: the panel shows the same test, and it finishes.
10. Open an English film of a memory, choose this voice → Prepare. Then
   `sqlite3 ~/.ivms777/ivms777.db "select json_extract(decisions_json,'$.voice.conditioning') from memory_presentations where language='en'"`
   shows `restored` for the voice: the film used the badged example that a test already baked.
11. Delete the chosen example. The confirm names films. After it is gone, the voice reads "The English example chosen for films was deleted. …".
    On `/feed` the picker shows "«<name>» — the English example chosen for films was deleted", selected; Prepare is refused with the reason.
    Press "Use for films" on another English example; Prepare then works.
12. After steps 3-10 the badge is still where step 2 or step 11 put it: a test never changes it.
13. Play a "Generated" clip, then press Record → Start recording: the playback pauses.
14. Esc order: open the test panel with no test running and focus "Your own text". Esc blurs the field; the next Esc closes the panel; the next
    closes the page.
15. Delete the voice. After "deleting…" ends, `ls -R ~/.ivms777/voices` lists none of the test WAV keys you noted before with
    `sqlite3 ~/.ivms777/ivms777.db "select storage_key from voice_previews"`.
16. Tell the main session PASS, or what looked wrong.

### 6.5 Check after Phase 5 Slice A — the feasibility bake on this Mac (D64, D65)

Nothing here promises a time, a memory figure or a likeness. The numbers come from your run.

1. Optional, before the run: record more takes of «me» on `/organize/voices` (or `/voices` once Slice V is in), with any reading text, at
   different paces and moods. The check bakes every take the voice has. The package authors suggest 5-10 minutes for one speaker; that is their
   guidance, not a promise.
2. During the bake: start no film and no voice test (they fail with "a voice is being baked"), and edit no file under `modelsvc`, `audio`,
   `models`, `embedding`, `captioning` or `inference` in the checkout your `make up` runs from (a hot reload ends the bake).
3. `cd <base>/ivms777 && uv run --no-sync python -m scripts.voice_bake_check --voice me --list`. Read each take's words against what you said.
   For a take whose words differ, note `--text <id>=<the words you said>`.
   **Changed 2026-11-02 (ASR, t42):** you need not type words if you run this check after your first app bake of «me» (§6.7): `--list`
   then shows the words ASR heard ("heard by whisper-large-v3-turbo@41f01f3") for each take. Before that, a take shows "shown text, not
   heard", and `--text` stays the way to give its real words.
4. `uv run --no-sync python -m scripts.voice_bake_check --voice me --confirm-texts [--text <id>=…]`. The line shows the stage, the percent, the
   step and the loss. Ctrl-C cancels: the line then reads "cancelled at step N" (or the end the service reached first), no adapter is kept, and
   the bake folder is deleted.
5. Expected in the models service log: `Running on device: mps, dtype: float32` (`model/voxcpm2.py:176`) for the training child, no CPU warning,
   and no `ImportError`.
6. **Changed 2026-11-02 (pair t38):** write down from the summary the script PRINTS (not from `report.json`, which reveals the mapping): the
   stage seconds, `seconds_per_step`, the steps, `free_mb_before` and `min_free_mb`, `peak_device_bytes`, the adapter size, and the mean RTF of
   A, B and C. B is the prompt take alone on the LoRA-built child (its LoRA matmuls run with B = 0; their cost was never measured, t31), so
   compare B's RTF with your §6.1 RTF. Watch Activity Monitor → Memory → "Memory Pressure" during training and note its colour. No speed
   promise.
7. **Changed 2026-11-02 (pair t38):** listen to each `X` / `Y` / `Z` file of a sentence and score it BEFORE you open `key.json` or
   `report.json`: similarity to your voice (1-5), naturalness (1-5), wrong or missing words.
8. PASS needs every item (D65). **Changed 2026-11-02 (rework, t39):** C alone decides; A and B only help to diagnose.
   - the automatic checks in `report.json` are all true (bake `done` on `mps`; the adapter `loaded` or `kept` in every A and C reply; every file has
     sound, no NaN, `voxcpm2` @ `32279effe8c19989596f05d353d1447f51d9e915`);
   - `min_free_mb` ≥ 1024, and macOS ended no process;
   - `seconds_per_step` ≤ 6.0;
   - C (the adapter alone, `use_clip=False` — the way films and tests will speak): mean similarity ≥ B's mean + 0.5, mean naturalness ≥ 3.5,
     ≤ 1 wrong word per sentence. (Before: "the better of A and C".)
9. **Changed 2026-11-02 (rework, t39):** also tell, as a diagnosis only: does A sound more like you than C? Films never use A: they speak
   through the voice model alone. (Before: "this decides whether a baked voice keeps its prompt take".) The prompt take of A and B is the
   newest take of the language once lane 31 lands (D82); before that it is the film take of Slice V.
10. Tell the main session PASS or FAIL with the report path. **Changed 2026-11-02 (rework, t39):** a FAIL no longer stops Slice B's code, which
    runs beside this check. A FAIL means saved voices cannot narrate a film on this Mac until it is solved. The main session reports that as a
    blocker for your decision (record more takes and run again, try other `--steps`, or stop baking). It never adds a one-take path.
11. **Added 2026-11-02 (rework, t39; D84):** while Slice B is built, the main session edits no file under `modelsvc`, `audio`, `models`,
    `inference`, `embedding` or `captioning` before you report this check, or before you say that no bake runs. If your app (with Slice B) asks
    "The models service is baking a voice for the bake check (N%). Stop it and bake «…» instead?", answering yes ends this check's bake.

### 6.6 Check after Phase 5 Slice V — the Voices page (t30)

**Changed 2026-11-02 (rework, t39):** this check describes Slice V as built. Slice B replaces four of its steps: step 2's film take on the
card, step 3's film take in the header, step 9 (the film-take select and its tombstone) and step 10 (a test of one take). §6.7 checks the
new card, header, Voice model tab and Test tab. The other steps (nav, grid, create, tabs, recorder, Esc, close, htmx and native history, the
question before a nav, the `/feed` hint) still hold and are still to report. When Slice B lands before you run this check, skip steps 2's
and 3's film-take parts, 9 and 10.

1. The nav reads Upload, Library, Chat, Organize, Voices. Organize has no "Saved voices" link. Settings (⚙) still opens as an overlay.
2. `/voices`: a card per voice with its per-language minutes and takes and the film take; no × on the page; "New voice" at the top. Create "Test
   person": its page opens at once on Recordings. Open the old bookmark `/organize/voices`: it lands on `/voices`.
3. Open «me» from its card. The header reads "Voices › me" with the film take per language; ⋯ holds Rename and Delete voice. Switch the tabs: the
   URL changes `?tab=` and Back does not walk through tabs; a reload keeps the tab.
4. Recordings: tick consent, Record. The recorder opens right under "Add a take", with a text, "Different text" and "Start recording". Record,
   Stop, then let the list refresh (delete another take meanwhile): the take stays. Save it.
5. Esc order: during a live recording Esc discards it; with an unsaved take Esc discards it; with a prepared recorder Esc closes it; with a focused
   field Esc blurs it; then Esc closes the page to `/voices`, which keeps its scroll.
6. Close rules: open a voice from the grid and press × → back to the grid (Back history, not a new entry). Paste `/voices/<id>` into a new tab and
   press × → `/voices`, and Back does not return to the voice page.
7. htmx history: on a voice page click Library, then the browser Back. The voice page works: one click on Delete removes ONE take, one Generate
   starts ONE test, Record opens ONE recorder. Forward, then Back again: the same. Native history (t31): delete a take on a voice page, press ×
   at once (back to the grid), then Forward: the voice page shows the take gone without a reload, and one Generate still starts ONE test.
8. With a live recording (or an unsaved take), click Chat in the nav: a question asks before it discards the recording. Cancel keeps you on the
   page with the recording. Click ⚙: no question; the recording goes on.
9. Voice model tab: one select per language; the newest reads "(newest)"; choose an older take → the header and the card show it. Delete the chosen
   take: the select shows "— the chosen take was deleted —" with the line; `/feed` refuses that voice with the "under Voices" text.
10. Test tab: choose a take and a text, Generate. Stage, percent and Cancel show beside the button; Original and Generated play side by side and name
    the take and its text. Leave for Library while a test runs, come back: the test shows again and ends. During the test, delete another take: the
    test goes on.
11. `/feed`: the hint reads "Add a voice under Voices." and the link opens `/voices`.
12. Tell the main session PASS, or what looked wrong.

### 6.7 Check after Phase 5 Slice B — kit, words, bake, the voice model as the only way a saved voice speaks

**Changed 2026-11-02 (rework, t39): rewritten.** How a voice sounds is your judgment; nothing here promises likeness or speed. The figures in
your page are your own; the wireframes of §2.17 K6 only show the layout. Before you start, note the output of
`sqlite3 ~/.ivms777/ivms777.db "select voice_id, version, storage_key from voice_models"` (empty at first).
**Changed 2026-11-02 (ASR, t42):** steps 0, 1, 2, 4, 5, 7 and 8 are rewritten, and steps 15-17 are added. Nowhere does the page ask you to
type or confirm words.

0. Added (ASR, t42): once, `cd <base>/ivms777 && make models`. It downloads the speech-recognition model
   (`openai/whisper-large-v3-turbo` at `41f01f3`, about 1.6 GB). Without it a bake stops at "transcribing the takes" with "… not on disk:
   run `make models`".
1. Recordings: per language the kit groups (Normal, Slow, Fast, Calm, Happy, Sad, Serious, Excited, Questions, Emphasis, Articulation, Numbers and
   names) with their instruction. No take row has a Test button and no row has a words box. The tip under "Add a take" names the voice
   model and says nothing about "one example per language". Record a slow text from its row: the recorder opens at that row with the
   instruction and the line "Read it, or say anything else: the bake hears what you say." Say something else than the text, Stop, Save:
   there is no "What you said" box; the take sits on that kit row. (Before: a "What you said" box and "words checked".)
2. Upload a WAV of you speaking about anything (not a kit text): Save asks for no words, and the take shows under "Other takes". Your older
   takes show no "check the words" or "needs its words" text. (Before: Confirm · Edit, and words before Save.)
3. Before any bake: the header of the voice reads "No voice model yet. «<name>» cannot narrate a film until you bake one."; its card reads "No
   voice model yet". The Test tab reads "Bake the voice model first." and Generate is disabled. On `/feed`, the picker shows this voice as a
   disabled "«<name>» — bake required" option; when it is the stored narrator, it stays selected with that label, and the line under the picker
   links to the voice's Voice model tab. Prepare with it: the refusal reads "«<name>» has no voice model yet. Bake it under Voices, or choose
   another narrator." No film is made from a take.
4. Voice model tab: "Bake voice model" is enabled with no word check. The meter reads "Your recordings: <Language> m:ss · …" and the
   authors' 5-10 minutes, called guidance. (Before: disabled while a take was unresolved; "Your checked words".)
5. Bake. The job first reads "transcribing the takes · take N of M" with the percent and Cancel: every take is heard, also your older takes
   and the upload of step 2. Then the service stages follow. The header and the tab show the stage, the percent and Cancel; the card on
   `/voices` shows "Baking · N%". Prepare a film narrated by this voice during this FIRST bake: it fails at once with the "no voice model
   yet" text, because there is no model to wait for.
6. When the bake is done: "Voice model v1 · ready · films and tests of «<name>» speak through it"; the Voice model tab shows "Learned from <T>
   takes · m:ss · <Language> …". Prepare a film: it is made, and "This film" reads "… · voice model v1".
7. Test tab: the Language select lists the languages the model learned. Choose a fixed sentence and Generate: stage, percent and Cancel show;
   then "Generated — … · voice model v1 · …" with ▶. Choose "Your words in take #<id>" (the take of step 1): the words under the select
   are what you said, as ASR heard them, not the kit text. Generate: ▶ Generated and ▶ Original (that take) sit side by side. Type your own
   text and Generate. Leave for Library while a test runs and come back: the test shows again and ends.
8. Record one more take: the header reads "1 take added since v1 — Rebake to include them"; films still say "voice model v1". (Before:
   "Edit the words of one take".)
9. Delete a take the model learned from: the model stays. The header adds "1 deleted"; a new film still reads "voice model v1". After the drain,
   the `select … from voice_models` of the start still lists the same `storage_key`, and `ls -R ~/.ivms777/voices` still lists that file.
10. Rebake. Prepare a film narrated by this voice during the Rebake: its bar reads "waiting for the voice model" until the bake ends, then it
    is made, and "This film" reads "voice model v2". Press Cancel on a later Rebake: it stops at the next step ("cancelled"), and the model you
    had stays in use. After v2: the change line goes. A test made with v1 reads "made with voice model v1; «<name>» now has v2 — generate it
    again". After the drain, the v1 file is gone from `~/.ivms777/voices` and the v2 file is there.
11. While a bake runs, run `touch web/voices_api.py` in the checkout your `make up` runs from (the app reloads): the page shows the bake again and it
    ends as v<N>.
12. Start a bake, press Cancel just as it reads "saving the voice model": either the new version appears, or nothing changes — never a model
    after the page said "cancelled". Start a bake of another voice while one runs: the page asks first; yes stops the first bake and then starts
    the second. If you run the §6.5 script while you press Bake in the app: the page asks "The models service is baking a voice for the bake
    check (N%). Stop it and bake «<name>» instead?"; No keeps the check running; after an app restart the check's bake is never touched.
13. A film made before Slice B from one take still plays, and "This film" reads "… · made from one take, before voice models". Regenerate it:
    it is made through the voice model.
14. Delete the voice: its model file and its test WAV leave `~/.ivms777/voices` after the drain.
15. Added (ASR, t42), on another voice or before step 14: upload a 5-30 s WAV with music or a hum and no speech, and Rebake. The bake goes
    on, and the new version is stored from the other takes. **Changed 2026-11-02 (ASR plan check, t44):** two results are possible; tell
    the main session which one you saw:
    - an empty result: ASR returned no words (blank or symbols only). The take's row reads "not used by the voice model: the speech
      recognition found no words in it"; the Voice model tab lists it under "Not used" with that reason; the done line names it.
    - a hallucinated result: ASR returned words that nobody said. The take shows no "not used" text and is trained with those words; the
      Test tab's "Your words in take #<id>" shows them. Note the words.
    No text on the page says "no speech". Delete the take: nothing else changes.
16. Added (ASR, t42): record two new takes and press Rebake; press Cancel during "transcribing the takes". The job stops after the take in
    hand; the model you had stays. Rebake again: the stage hears only the take it had not heard. `touch web/voices_api.py` during the stage
    (the app reloads): the job is gone, nothing breaks, and the next Rebake hears only the rest.
17. Added (ASR, t42): in the models service log, find the line that starts `asr whisper-large-v3-turbo@41f01f3 loaded by this request`
    and tell the main session the whole line. **Changed 2026-11-02 (ASR plan check, t44):** its "net change" is the free-memory change of
    that request, other units it evicted included (the line names them); it is not Whisper's cost, and the catalog's 4000 MB stays an
    estimate (`cost_measured=False`). During "training", `curl -s localhost:9000/models` lists no `asr` in `resident`.
18. Tell the main session PASS, or what looked wrong.

### 6.8 Check after Phase 6 — Automatic may pick a saved voice

Run it after §6.7, with a voice «<name>» whose voice model learned English (or Russian). Which voice the story picks is the story pass's
judgment; nothing here promises how often it picks a saved voice. To read what each film decided:
`sqlite3 ~/.ivms777/ivms777.db "select group_id, language, json_extract(decisions_json,'$.cast.narrator_chosen_by'), json_extract(decisions_json,'$.cast.narrator_label'), json_extract(decisions_json,'$.cast.offered') from memory_presentations"`.

1. On `/feed`, the first option of the picker reads "Automatic — the app picks a voice for this memory". The line under it reads "The story
   picks the voice: a preset voice or one of your saved voices with a voice model for <Language>. Otherwise the memory's mood picks a
   preset voice." No text says "Saved voices are never picked automatically."
2. Choose Automatic on a memory about «<name>» (its title names them, or a note says so), in a language the model learned, and Prepare.
   The query lists the presets and «<name>» under `offered`. When the story picks «<name>»: "This film" reads "… narrated by <name> (saved
   voice) — picked automatically by the story · voice model v<N>", the "recording the voice" bar moves on an estimate, and the film speaks
   in that voice. An "Other voices" line, if shown, names saved voices only. When the story picks a preset, "Other voices" names presets
   only. Regenerate a few times: no film has both kinds.
3. Prepare Automatic in a language that the model did not learn: `offered` holds presets only, and a preset narrates.
4. Choose «<name>» yourself and Regenerate: "— your choice", and it reads the whole film (t50).
5. Tell the main session PASS, or what looked wrong.

### 6.9 Check after Phase 7 — editable narration

**Changed 2026-11-03 (plan check, t63):** steps 3, 6, 8, 9 and 10 are rewritten; steps 4b and 8b are added. **Changed (pair t64):**
step 2b checks the shot/audio order fix (§2.19 I).
How the voice reads your words is your judgment; nothing here promises how it sounds. To read the stored state:
`sqlite3 ~/.ivms777/ivms777.db "select group_id, language, draft_token, revision, source_token, length(passages_json) from narration_drafts"` and
`sqlite3 ~/.ivms777/ivms777.db "select language, json_array_length(timeline_json,'$.passages'), json_extract(timeline_json,'$.script.token') from memory_presentations"`.

1. Open a film made before this change. Pause it (or stay on the title card) and open "Tell the story better": the first line reads
   "This older film has no timed narration text. ↻ Regenerate makes a new film whose narration you can edit here." Your notes are there
   as before.
2. ↻ Regenerate it. When it is ready, open the panel: every passage of the narration is a row tagged "narration", with its time, among
   your notes by time. You pressed nothing to add them. The count on the folded bar counts passages and notes. Press a passage's time:
   the film jumps to where that passage is spoken. Play: passages never float up as bubbles; notes still do.
2b. Order (§2.19 I): pick a film whose story returns to an earlier photograph (a passage tagged "narration" says something about a
   person or a place an earlier passage showed). Play it: while each passage is spoken, the picture is the one its words are about,
   and at the return the earlier photograph comes back under those words, not next to its first appearance. Press the time of the
   returning passage: the same photograph shows. Then run `sqlite3 ~/.ivms777/ivms777.db "select language,
   json_extract(decisions_json,'$.story.memory_order') from memory_presentations"`: it prints `kept`, `retried` or `broken` for every
   film made after this change. A `broken` film still shows each picture under its own words. **Changed (t65):** the order holds also
   for a film whose rows show "–:––" (the voice did not report a time per passage): its pictures come in the order of the rows, though
   a cut may not land exactly where the words start; exact sync needs the times.
3. Change a few words of one passage and press Enter. The panel reads "Your edited narration. ↻ Regenerate speaks it as written.", then
   "Notes cannot rewrite these words or provide a new voice or music interpretation. Existing voice choices still pass current note
   gating and validation. Effect instructions reach the conductor; matched speech fixes shot timing.", with "Write a new story", and
   ↻ Regenerate lights up. Under the picker (Automatic), the bracket line now reads "With your edited narration, a [voice] note cannot
   ask for a new voice: …", and the bracket legend starts "With your edited narration, a note's words are not narrated …". Reload the
   page: your words are still there, and ↻ Regenerate is lit again. The first query lists one row at revision 1 (or more after more
   changes).
4. Clear one passage (or press its ×): it goes. Clear the others until one is left, then clear it: "A film needs narration. Keep at least
   one passage." and the row stays. Type "[whisper]" into a passage: "The narration is read aloud word for word. Put a [bracket]
   instruction in a note." Put the bracket in a note instead.
4b. Type a change into a passage and, without leaving the field, press ↻ Regenerate at once: the build starts only after the change is
   saved, and the film speaks it. Then make a passage refuse (type "[x]" into it) and press ↻ Regenerate: no build starts, and the line
   under the form reads "Not started: an edit to the narration is not saved. See Tell the story better."
5. Paste a long text (more than 1 000 characters) into one passage: it is kept whole after a reload.
6. Open the same film in a second tab. Change a passage there. Then change a passage in the first tab: "Not saved: this narration changed
   in another window, or a new film replaced it. Copy any text you want to keep, then load the current narration." Your text stays in
   its row, and the fields can no longer be edited. Press "Load the current narration": the rows show the second tab's change, and you
   can edit again.
7. Switch to Русский: its rows are its own, and your English edits are not there. Back to English: they are.
8. Press ↻ Regenerate, then Cancel at once: the draft stays (the first query still lists it). Press ↻ Regenerate and let it finish.
   Listen: the film speaks your words exactly, in the order of the rows. The rows now carry the new film's times (press one: it seeks),
   the panel reads "This film speaks your edited narration. ↻ Regenerate speaks it again as written.", and ↻ Regenerate is not lit
   after a reload. The first query still lists the draft, with the same `draft_token`, one revision more and the new `source_token`.
8b. Change the narrator in the picker and press ↻ Regenerate with no change in the rows: the new narrator reads your same words. No
   new story is written.
9. Change a passage in the first tab. Then open the film in a second tab: it shows your change. In the first tab, press ↻ Regenerate.
   While it builds, change another passage in the second tab (it saves; a tab opened after the build started shows only the progress).
   When the film is ready, reload the first tab: the film speaks the first change; the panel shows your second change as "Your edited
   narration. …", and its times are disabled ("From the film these words were taken from").
10. Press "Write a new story". The question reads "Stop using your edited narration? The next Prepare or Regenerate writes a new
    story." Confirm: the panel shows the film's own rows with the line "… With no change, ↻ Regenerate writes a new story.", and the
    first query lists no draft for that film. Press ↻ Regenerate: the story pass writes new words.
11. Make a change again, then go to Organize: the card's "↻ Retell memory" line ends "Deletes your edited narration in: English."
    Optional: press it. When it is done, the films and the draft are gone, and your notes stay.
12. `sqlite3 ~/.ivms777/ivms777.db "select count(*) from memory_notes"` gives the same number before and after steps 3-10: an edit of the
    narration never becomes a note.
13. Tell the main session PASS, or what looked wrong.

## 7. Definition of done

1. `docs/design.md` (§4.1, §5 + mermaid, §5.1, §6, §8.1, §11.1 + mermaid, §11.3, §13, §13.1), `docs/models.md`, `docs/data-model.md`, `docs/ui.md`, `docs/ingest-pipeline.md` match the code. Every mermaid block renders (`mmdc`).
   (**Changed 2026-11-02, Phase 5:** also the "Voice baking" section of `docs/models.md` and the `/voices` entries of `docs/ui.md`.)
   (**Changed 2026-11-02, rework, t39:** no doc says that a film or a test uses one take, a film take, a pinned take or a stored conditioning;
   `grep -n -i "film take\|films use take\|one example per language\|Use for films" docs/` finds only text that calls them retired.)
   (**Changed 2026-11-02, ASR, t42:** no doc says that the user types, checks or confirms the words of a take; the closing `grep` of §2.14
   "Slice B, ASR rework" finds nothing about takes; `docs/models.md` documents the bake's Whisper with its repo, revision, licence and
   estimated cost, and `models/catalog.py` declares it in `TRANSCRIBERS`.)
   (**Changed 2026-11-03, Phase 7:** `docs/design.md` §6, §11.1 + mermaid, §11.3 and §13, `docs/data-model.md` and `docs/ui.md` describe
   the film's passages, `narration_drafts`, the panel's narration rows and the retell text; the closing `grep` of §2.19 H finds them.
   **Changed (pair t64):** §11.1 also states the told order and the one bounded memory-order correction of §2.19 I.7.)
2. Every new module has tests (§5). `uv run pytest -q` and `uv run ruff check .` are green in `<base>/ivms777`.
3. `sadensmol-code-review` over the diff; every P0/P1 fixed. (**Changed 2026-11-02, t25:** it runs inside `task finish`, not in a main step, D56.)
4. The dependency gate (§3) passed; its licence table is in the PR body and in `docs/models.md`.
   The licence results include the five approved exceptions with their obligations and the funasr removal (D35).
5. The user approved checkpoint §6.1, the final check §6.2 and the Phase 3 check §6.3 (finish Step 6b; §6.3 added 2026-11-02).
   The Phase 3 docs of §2.14 match the code, and the §11.1 mermaid block renders. The user answered the films question at the gate (`## Open items outside scope`);
   an answer other than the recommendation sends the plan back to the planner before any code.
   **Changed 2026-11-02 (Phase 4):** the user also approved the Phase 4 check §6.4; the Phase 4 docs of §2.14 match the code, and the edited §5 and
   §11.1 mermaid blocks render; the user answered the "Use for films" question at the gate (an answer other than the recommendation sends Phase 4
   back to the planner before any Phase 4 code).
   **Changed 2026-11-02 (Phase 5):** the user also approved §6.5 (the Slice A bake check, PASS), §6.6 (Slice V) and §6.7 (Slice B); the Phase 5
   docs of §2.14 match the code; every edited mermaid block renders; the user answered the deleted-take question at the gate. A §6.5 FAIL ends
   Phase 5 at the user's decision, and this item then counts only what ships.
   **Changed 2026-11-02 (rework, t39):** the deleted-take question is answered (keep the model, Rebake). The user approved the Slice B rework at
   its gate. Films and tests of a saved voice speak only through its voice model (no take, no one-take fallback), as §6.7 shows. The rewritten
   Slice B docs of §2.14 match the code. A §6.5 FAIL does not make the code add a one-take path: it is reported as a blocker, and the user
   decides what ships.
   **Changed 2026-11-02 (ASR, t42):** the user approved the ASR amendment at its gate, and §6.7 shows a bake that heard free speech and an
   unrelated upload, a take with no speech shown as "not used", and the measured drop of the Whisper load (step 17). The bake's ASR is
   claimed to run only after that report. **Changed 2026-11-02 (ASR plan check, t44):** "a take with no speech shown as 'not used', and
   the measured drop" → "step 15's result for a take with no speech (an empty result shown as 'not used' with its reason, or a
   hallucinated one noted), and the log line of the request that loaded Whisper (step 17; a net change, not a cost)".
   **Changed 2026-11-03 (Phase 7):** the user approved the narration amendment at its gate, and reported §6.9 PASS: a film's narration
   shows as timed rows on load, a ↻ Regenerate speaks the edited words exactly, a failed or cancelled build keeps the edits, an older
   film says why it has no rows, and an edit never becomes a note. **Changed (t63):** also: the edited words stay for every later
   ↻ Regenerate until "Write a new story"; no Prepare or ↻ Regenerate starts before a pending edit is saved. **Changed (pair t64):**
   also §6.9 step 2b PASS: each passage's picture stays under its own words, a return keeps its place, every new film records
   `memory_order`, and the closing `grep` of §2.19 H finds nothing in `docs/` and `feed/`. **Changed (t65):** the order also holds
   for a film whose segments do not pair (`keep_order`); exact sync is claimed only for paired segments.
6. Commit: only our hunks; the pyproject/uv.lock question is answered. PR title `IVMS777-21 Voice copy with VoxCPM2: saved voices and narrator picker`, base `main`, assignee `@me`, no reviewer.
   (**Changed 2026-11-02, Phase 5:** the title becomes `IVMS777-21 Voice copy with VoxCPM2: saved voices, narrator picker, Voices page and baked
   voice models`.)
   Body: `## Summary`, `## Top-level changes`, the §-list of `docs/design.md` changes, then the `🤖 Generated with` line. No local paths.
7. Linear: one comment on IVMS777-21 with the PR URL. The status stays In Progress.

## Model decisions

D1–D14 are the entries of the delivered r&d plan (History). This plan relies on their content as restated in §2 and §6. New entries start at D15.

### D15 — conditioning blobs live with the user's data in `app` (2026-11-01, PLAN, planner)
- what: the child returns a built conditioning as an opaque blob; `app` stores it in `voice_conditionings` + `data_dir/voices` per example + catalog key + revision, and sends it back with the clip; the child restores it only when key, revision, `voxcpm` version and clip sha256 match. A deleted or replaced blob leaves the disk only through the voice outbox drain (D33).
- why: a deleted voice must take its examples and their conditionings with it (cascade; the delete rule of `ivms777/CLAUDE.md:151-161` applied to voices, outbox included); films finished before the delete keep their narration until regenerated (the recommended answer to the gate question; D33); the models service holds no database (`docs/design.md:458`) and needs no new store, volume or "forget" call.
- rejected: a cache folder in the models service keyed by clip hash (latents outlive a delete; needs a forget endpoint and a volume).
- where: `feed/saved_voices.py:save_conditioning`, `process_voice_deletions`; `audio/voxcpm_voice.py:VoxCPMVoice._conditioning`; `db/schema.sql:voice_conditionings`

### D16 — VoxCPM2 reference mode for every example; no transcript field (2026-11-01, PLAN, planner)
- what: `build_prompt_cache(reference_wav_path=…)` only; a recorded example still stores its fixed prompt as `transcript`; the upload form has no transcript field.
- why: reference mode needs no transcript (`model/voxcpm2.py:681-740`); t4: "typed transcript only if the chosen model needs one"; one behaviour for recorded and uploaded examples.
- rejected: continuation / ref_continuation mode for recorded examples (two behaviours; no measured gain).
- where: `audio/voxcpm_voice.py:VoxCPMVoice._conditioning`; `feed/saved_voices.py:add_example`
- changed (2026-11-02, Phase 5): the working tree already clones a take WITH its words in `ref_continuation` mode and one without in
  `reference` mode (§1.8; `docs/models.md:128-139`), so "reference mode for every example" no longer describes the code. Phase 5 also
  reverses "no transcript field": Slice B confirms the words of every take and asks an upload for its words (D71), because a bake needs them
  (t28). This entry claims no new reason for ref_continuation; Slice A's blind listening compares the baked variants with that path (D64).
- changed (2026-11-02, rework, t39): films and tests no longer send any take, so no cloning mode applies to them: they speak through the voice
  model alone (D76). Only the check script's diagnostic variants A and B still send a take (D82).
- changed (2026-11-02, ASR, t42): no take stores its shown text as words any more, and the user confirms nothing; ASR hears a take's
  words before a bake (D91, D92). The check script's A and B send the stored `transcript`, which is the heard words once a bake heard
  the take.

### D17 — one voice engine per film; saved voices and presets never mix (2026-11-01, PLAN, planner)
- what: Automatic narrator → presets only, as today; a saved narrator → every passage is a saved voice of that language (token `saved:<id>`), and a `[voice]` bracket can cast another saved voice; `/speak` rejects a mixed request (422).
- why: one voice unit is resident at a time (`modelsvc/slots.py:202-218`); VoxCPM2 cannot speak Kokoro or Silero names; a mid-film swap loads two large models.
- rejected: per-run engine switching inside one `/speak`; a second voice unit; brackets that cast saved voices into a preset-narrated film.
- where: `feed/build.py:build_presentation`; `modelsvc/app.py:speak`

### D18 — the narrator choice is stored per memory and language (2026-11-01, PLAN, planner)
- what: Prepare and ↻ Regenerate post `narrator`; `memory_narrators(group_id, language, voice_id)`; the build reads it; the picker preselects it, on every profile; a POST without the field keeps the stored choice, `""` means Automatic (`Form(None)`; today `lang: str = Form(default="")`, `web/app.py:978`).
- why: the user's choice must survive navigation (`ivms777/CLAUDE.md:51-66`); `feed_building` / `feed_queued` stay `(key, language)` (`web/app.py:987-1006`).
- rejected: carrying the narrator only in the queued tuple; one owner-wide default narrator; `narrator=Form("")` (an absent field would clear the choice).
- where: `db/schema.sql:memory_narrators`; `web/app.py:feed_prepare`

### D19 — same-language examples only; the newest one wins (2026-11-01, PLAN, planner)
- what: a film in language L uses a saved voice only through an example in L; the picker lists only those voices; with several, the newest example in L is used.
- why: en↔ru cloning is not published for VoxCPM2 (findings §4.2, §7); the check script measures it, reported only.
- rejected: cross-language reuse in films; a user-picked "main" example.
- where: `feed/saved_voices.py:example_for`, `castable`
- changed (2026-11-02, Phase 4, pair t24; only after the user's "yes" at the gate, `## Open items outside scope`): same-language examples only still
  holds, and the newest stays the default. "Use for films" per language overrides the newest (D46-D48). This reverses the rejected "a user-picked
  'main' example". Without that "yes", this line does not apply and Phase 4 goes back to the planner.
- changed (2026-11-02, Phase 5, user + pair t28): a baked voice learns from ALL its takes in every language (D58, D73), which reverses
  "examples never combine" for a baked voice (the user: "USE ALL THE SAMPLES TO BAKE IT INTO THE PERSON'S MODEL"). The same-language rule stays
  for the prompt take, and a voice with no usable model still speaks from one take.
- changed (2026-11-02, rework, t39): superseded for films and tests. A saved voice narrates only through its voice model; the same-language rule
  moves to the model: it can narrate a language only when it learned takes in it (D77). The newest take survives only as the check script's
  diagnostic prompt take (D82).

### D20 — Saved voices is a leaf under Organize at `/organize/voices` (2026-11-01, PLAN, planner)
- what: a plain link in the Organize head, outside the photos selector; actions via `fetch`, no history push; Esc unwinds recording → unsaved take → text field → close (`history.back()`, fallback `/organize`).
- why: t4 placement; §13.1 `[grid, leaf]`; `ivms777/CLAUDE.md:112-115`; htmx 2.0.3 does not swap 4xx replies, so one `fetch` path shows every error.
- rejected: a new nav item; a section inside Settings (an overlay, `docs/design.md:2116-2128`); a route under `/upload`.
- where: `web/voices_api.py`; `web/templates/voices.html`; `web/static/voices.js`
- changed (2026-11-02, Phase 5, user request, pair t30): reversed. Voices is a top-level nav item: `/voices` is a grid and `/voices/{id}` its leaf
  (§2.17 V, D67); the Organize link goes; `GET /organize/voices` answers 308. This is the user's request ("add Voices menu item, move all voices
  stuff there", `## Decisions`), not a model decision.

### D21 — the browser makes every example a mono PCM-16 WAV at 48 kHz (2026-11-01, PLAN, planner)
- what: MediaRecorder or a picked file → `decodeAudioData` in `AudioContext({sampleRate: 48000})` → mono → WAV; the server accepts only an uncompressed, 16-bit, mono WAV at exactly 48000 Hz with every frame present, 5-30 s, ≤ 4 MiB, not silent. `add_example` checks the header with stdlib `wave` before `decode_wav`, and stores and hashes the checked bytes as posted. The check script applies the same contract.
- why: the app has no ffmpeg and no audio library beyond `wave` + numpy (`docs/design.md:1645-1646`, `storage/wav.py:1-8`); findings §10 clip spec (48 kHz, mono, 16-bit); `decode_wav` reads any sample width as int16 and downmixes any channel count (`storage/wav.py:49-57`), so it cannot be the check; a truncated file reads short with no error (§1.3); the only producers make 48 kHz (§2.9, §6.1 `afconvert`).
- rejected: server-side decoding of webm/opus/m4a; storing the browser's original file; a 16000-48000 Hz range (no producer makes those rates); a downmix of stereo; a stricter `decode_wav` (shared with `audio/` in the models image, `Dockerfile.models:42-47`, and with `feed/beats.py:83`, `feed/build.py:527`).
- where: `web/static/voices.js`; `feed/saved_voices.py:add_example`; `scripts/voice_check.py`

### D22 — a consent statement per example (2026-11-01, PLAN, planner)
- what: Record and Upload need a checked "This is my voice, or I have this person's permission to copy it."; the row stores `consent_at`.
- why: the app copies real voices for commercial use (`## Decisions`); the r&d findings left consent to the product.
- rejected: no consent step.
- where: `feed/saved_voices.py:add_example`; `web/templates/voices.html`

### D23 — a gender-neutral Russian recording prompt (2026-11-01, PLAN, planner)
- what: the ru prompt avoids first-person past-tense verbs (§2.7); the en prompt is the findings §10 prompt.
- why: the findings ru prompt uses masculine forms ("налил", "открыл", "слушал"); a woman reading it would read an unnatural text or change it.
- rejected: the findings ru prompt; a prompt per gender.
- where: `feed/saved_voices.py:RECORDING_PROMPTS`

### D24 — how the child loads VoxCPM2 (2026-11-01, PLAN, planner)
- what: `snapshot_download(repo, revision, local_files_only=True)`; folder name must equal the revision; `VoxCPM.from_pretrained(<local path>, load_denoiser=False, optimize=False, device=<embed_device>)`; mps or cuda only; float32 on MPS as the package forces; missing weights → a clear load error, no download inside a film.
- why: a Hub id loads `main`, not the pin (`core.py:157-161`); `optimize` compiles only on CUDA + triton and adds a warm-up (`model/voxcpm2.py:274-295`, `core.py:96-101`); GPU only (`docs/design.md:118-121`); `make models` / `POST /models/download` fetch the weights.
- rejected: `from_pretrained("openbmb/VoxCPM2")`; `optimize=True` on cloud before it is measured; an on-demand 4.7 GB download inside `/speak`.
- where: `audio/voxcpm_voice.py:VoxCPMVoice.__init__`

### D25 — catalog figures for `voxcpm2` (2026-11-01, PLAN, planner)
- what: key `voxcpm2`; `size_mb=4728` (4,957,031,714 B); `cost_mb=10500`, estimate; `languages=("en", "ru")`.
- why: 2,290,004,544 params × 4 B (fp32 on MPS) + 376,951,122 B AudioVAE = 9095 MB, plus ≈ 15% runtime margin; the app films only in en and ru (`feed/language.py:10`), and only those two are validated.
- rejected: the bf16 cloud figure (under-counts mac); declaring all 30 published languages.
- where: `models/catalog.py:CATALOG`

### D26 — strict voice child map and a gateway identity check (2026-11-01, PLAN, planner)
- what: `_register_voice` uses `VOICE_TARGETS[key]` (no Kokoro fallback); after a reference call the gateway compares reply `model` / `revision` with the routed entry and raises on a mismatch.
- why: findings gaps 6 and 8 (`modelsvc/slots.py:224`, `modelsvc/app.py:194-200`); t8: never a preset for an explicit stored identity; findings §10 "Declared identity".
- rejected: keep the silent fallback; trust the child's reply.
- where: `modelsvc/slots.py:_register_voice`; `modelsvc/backends/composite.py:speak`

### D27 — the child splits long passages into ≤ 500-character sentence pieces (2026-11-01, PLAN, planner)
- what: `speak_one` splits with `audio.silero_voice.split_text` and joins the pieces directly.
- why: a film narration is 200-400 words (`audio/silero_voice.py:27-31`); one long autoregressive call per run has no measured behaviour on VoxCPM2; Silero already works this way (`audio/silero_voice.py:103-110`).
- rejected: one `generate` per same-voice run.
- where: `audio/voxcpm_voice.py:VoxCPMVoice.speak`

### D28 — longer timeout and a moving bar for saved-voice narration (2026-11-01, PLAN, planner)
- what: `ModelsClient.speak` uses 1800 s when it sends references; the build wraps that call in `_Ticking` with `characters ÷ 14 × CLONE_RTF_ESTIMATE` (1.0, display only, replaced after §6.1 measures RTF).
- why: VoxCPM2 RTF on MPS fp32 is unknown; 300 s (`inference/models_client.py:56`) can expire on a 2-3 min narration plus a first load; a static bar is forbidden (`ivms777/CLAUDE.md:264-269`).
- rejected: keep 300 s and a static 30%.
- where: `inference/models_client.py:speak`; `feed/build.py:build_presentation`

### D29 — where the new modules live (2026-11-01, PLAN, planner)
- what: the store is `feed/saved_voices.py`; the routes are `web/voices_api.py`.
- why: `feed/` and `web/` are already in the one-model gate (`tests/test_one_model_process_gate.py:39-42`) and in the wheel (`pyproject.toml:72`); a new top-level package edits both; routes outside `web/app.py` keep clear of the user's uncommitted hunks.
- rejected: a new top-level `voices/` package; all routes inside `web/app.py`.
- where: PLAN.md

### D30 — validation through a script the user runs (2026-11-01, PLAN, planner)
- what: `scripts/voice_check.py` turns the findings §10 subset into one command: fixed sentences, 5-segment call, unload + restore pass, blind A/B files, `report.json`; it takes a WAV file, not a saved voice.
- why: t4 includes validation; Claude may not call the stack (`ivms777/CLAUDE.md:163-181`); a WAV input lets the user validate at §6.1 before the UI exists; the D12 jetson thresholds do not apply.
- rejected: a curl-only protocol; a `--voice` mode that reads the database.
- where: `scripts/voice_check.py`

### D31 — pinned fetch targets as `repo@revision` (2026-11-01, PLAN, planner)
- what: `fetch_models.wanted` yields `"openbmb/VoxCPM2@<revision>"`; `_fetch` splits on `@`.
- why: keeps the `fetch(target)` injection shape of the tests (`tests/test_fetch_models.py`); Hub ids contain no `@`.
- rejected: a `(slot, repo, revision)` tuple (breaks every existing caller and test).
- where: `scripts/fetch_models.py:wanted`, `_fetch`

### D32 — two phases with a user checkpoint between them (2026-11-01, PLAN, planner)
- what: phase 1 = deps, engine docs, catalog/routing, child, `/speak` wire, check script → the user's GPU validation (§6.1); phase 2 = product docs, store, casting, UI.
- why: t8 puts GPU validation first; the UI and casting are worth building only on an engine that passed; docs still change before the code of each phase (`ivms777/CLAUDE.md:8-19`).
- rejected: build everything, then validate once at the end.
- where: PLAN.md §4
- changed (2026-11-01, pair t14): no stop at step 6. Steps 1-10 run in one pass. §6.1 moves to one hand-over with §6.2 at step 11.
  The §6.1 FAIL rule and its thresholds stay: a §6.1 FAIL stops the task for the user's decision. Docs-first, the offline checks and the review/commit gate stay.
  This change reverses the `rejected` line above.
- why (change): the user's words (## Decisions, 2026-11-01): "what a crap!!!?????? I SAID YOU IMPLMEMENT THIS FEATURE AND LET ME KNOW SO I COULD TEST!!!"

### D33 — saved-voice deletion runs through an outbox and a drain (2026-11-01, PLAN, planner; pair t9, t10)
- what: the delete routes only insert an owner-scoped, idempotent intent into `voice_deletions` and reply with the row "deleting…"; a pending voice or example rejects every mutation (409) and every new cast; a film whose chosen narrator is pending, or was deleted (`memory_narrators.voice_id` → NULL), fails with that reason and is never made Automatic or a preset; `process_voice_deletions` deletes the rows and records their file keys in `voice_file_deletions` in one transaction, then per key runs `BEGIN IMMEDIATE` → ref check → unlink → ack → `COMMIT`, then ends the intent; `add_example` and `save_conditioning` first commit a write-ahead cleanup row, then write the file and its row in one `BEGIN IMMEDIATE` that re-reads that row and skips a valid existing file; `saved_voices` and `voice_examples` ids use `AUTOINCREMENT`; the drain is the only code that removes a voice file, replaced blobs included; it runs in `drain_pass` group 1 (worker), and with `inline_drain` on, on one background thread in `app` (a `running` flag under its lock) after each delete, film build and add-example request, and at start, retried every 10 s; a film finished before the delete keeps its sound, and a film still being made fails at its ready save, before any film audio is written (D34).
- why: `ivms777/CLAUDE.md:159-161` (the endpoint records the intent and returns; a worker does the cascade; reliable across restarts; never blocks the UI); D15 applies that delete rule to voices; voice files are content-addressed (`storage/keys.py:8-17`), so one file can be shared or added again, and the ref check must hold the same write lock as the writes; `drain_now` runs only after an upload (`web/upload_api.py:157-163`), so a no-worker run needs its own schedule; a plain `INTEGER PRIMARY KEY` reuses ids (§1.3), so a new delete could meet an old intent; `LocalStorage.write` truncates in place (`storage/local.py:32-35`).
- rejected: an inline delete plus a start-up orphan sweep (it breaks the rule, and it is a different recovery contract nobody approved); rows deleted in the request with a file-only outbox (the request would do the cascade); a worker-only drain (a bare `uvicorn` run has no worker, `web/app.py:239-253`, so rows would stay "deleting…" forever); `drain_now` from the delete route (a synchronous full pass inside the request); `ON DELETE CASCADE` for `memory_narrators.voice_id` (the next build would make an explicit saved narrator Automatic); leaving the mid-write crash window open (a voice clip nothing would ever remove); `Thread.is_alive()` for the drain state (a `schedule()` in the exit window would be lost); removing finished films' narration on delete (a re-plan, only if the user asks at the gate).
- where: `db/schema.sql:voice_deletions`, `voice_file_deletions`, `memory_narrators`, `saved_voices`, `voice_examples`; `feed/saved_voices.py:request_voice_deletion`, `request_example_deletion`, `process_voice_deletions`, `add_example`, `save_conditioning`; `feed/build.py:build_presentation`; `ingest/pipeline.py:drain_pass`; `web/voices_api.py:VoiceDeletionDrain`

### D34 — film audio is written only at "saving the film" (2026-11-01, PLAN, planner; pair t11)
- what: both `_store_audio` calls (narration `feed/build.py:307`, music `:352`) move to the "saving the film" stage for every film; the earlier steps use the bytes only; a saved film runs the voice check → the WAV writes → the ready `save_presentation` in one `BEGIN IMMEDIATE`, with no model call inside; a failed check rolls back and calls the unchanged `_fail`, and no film audio is written; no failure path prunes audio.
- why: `prune_orphan_audio` deletes every key no presentation references (`feed/store.py:94-103`); after a cancel the old thread runs on (`web/app.py:1086-1097`) while a new Prepare starts at once (`:981-983`), so a prune on a failure path can delete another running build's narration; a key comes from the bytes alone (`feed/build.py:641-646`; music seed `:547-551`), so it does not tell which build owns it.
- rejected: a global prune on failure (the race above); a per-build delete of its own keys (another film's identical bytes share the key); `_fail(…, audio=)` (stopped by the pair in t11).
- where: `feed/build.py:build_presentation`

### D35 — drop funasr; five licence exceptions (2026-11-01, IMPL, main)
- what: `[tool.uv] override-dependencies = ["funasr; sys_platform == 'never'"]` removes funasr and its `kaldiio` from every install; `soxr==1.1.0`, `orjson==3.13.0`, `cffi==2.1.1`, `pyahocorasick==2.3.1` and `torchcodec==0.17.0` are approved licence exceptions, with their obligations in `docs/models.md` and the PR; the Mac gate compiles target macOS 14 (§3).
- why: the user chose it (## Decisions, 2026-11-01); `kaldiio` 2.18.1 has an NTT evaluation-only licence; a grep over the unpacked voxcpm-2.0.3 wheel finds no funasr import; torch>=2.13.0 ships only macosx_14_0_arm64 wheels.
- rejected: also overriding gradio, modelscope, datasets, argbind, matplotlib and spaces (more overrides, risk of a lazy import at the GPU run; orjson is approved anyway); another engine; pausing.
- where: `pyproject.toml:[tool.uv]`; PLAN.md §3; `docs/models.md`
- added (2026-11-01, pair t14): the torchcodec 0.17.0 manylinux wheel bundles `libnvjpeg` under the NVIDIA CUDA EULA (`third_party/LICENSE.libnvjpeg-NVIDIA-CUDA-EULA.txt`).
  It is a proprietary NVIDIA redistributable that allows commercial use under that EULA (§1.1.1; obligations §1.1.2; supplement §2.1/§2.6), and part of the approved torchcodec exception.
  It is never called permissive. No new user approval is claimed. `docs/models.md` and the PR licence table list it; the hand-over names it in one line, so the user can object at the commit gate.

### D36 — the card's film rows: "Open" only while a film is being made (2026-11-02, PLAN, planner; pair t17)
- what: per language, first match wins: building → "Open <Language> film · being made…"; queued → "Open … · waiting to be made…"; ready (status ready + `narration_key`) → "▶ Play <Language> film · ready · <name> (auto)" or "(your choice)" (t18); every other state → "Prepare <Language> film" with "not made yet", "the last try failed", "interrupted" or "no voice"; the ✓ on `/feed` uses the same ready rule; `<Language>` is the English name (`language_name`).
- why: t17 named the three labels but not which states take "Open"; the target page plays a film only on the same rule (`web/app.py:852-854`), and every non-ready page has a Prepare button, so the link says what the user will do there; a build in progress is something to watch, not to prepare.
- rejected: "Open" for failed and interrupted rows (the page there offers Prepare); the failure reason on the card (it can be long; `/feed` shows it); the language's own name ("Русский") in an English sentence.
- where: `web/app.py:organize`; `web/templates/_album_card.html`

### D37 — how a voice is named on the page, and films made before Phase 3 (2026-11-02, PLAN, planner; t18)
- what: a preset's full label is "<catalog name> (<gender>, <register>)" (e.g. "af_heart (female, warm)") and its short name is the catalog name; a saved voice is "<name> (saved voice)" / "<name>"; picker options use the story-prompt format "<name> — <gender>, <register>: <note>"; the build stores `cast.narrator_name`, `cast.narrator_label` and `passages[].used_label`; an older film falls back to `voice_label(narrator_used)`, and an older saved-voice token reads "a saved voice".
- why: t18's "Heart (female, warm)" fixes the shape, but the catalog declares no display name, only `name`, `gender`, `register`, `note` (`models/catalog.py:113-116`); "Heart" from "af_heart" would be a name guessed at the call site, which `ivms777/CLAUDE.md:236-239` forbids; the card's short form ("af_heart (auto)") needs the name apart from the label; the strings are stored with the film, so a later delete or rename does not change them; an old token cannot be named without a DB read inside a pure reader; t18 keeps "Voice not recorded" for a missing cast or `narrator_used` only.
- rejected: deriving "Heart" by stripping the `af_` prefix; a new display-name field in the catalog (a catalog change outside t17/t18); a DB lookup of old `saved:<id>` tokens; showing the raw token.
- where: `feed/voice.py:voice_label`; `feed/store.py:film_narrator`; `feed/build.py:build_presentation`

### D38 — a preset the catalog does not offer is refused, never replaced (2026-11-02, PLAN, planner)
- what: at Prepare, `preset:<name>` of another language → 409, an unknown name or an empty one → 404, and no build starts; a stored preset that `resolve` rejects stays selected as "«<name>» — not offered for <Language> films"; the build fails with the reason and no speak call; it never becomes Automatic.
- why: §2.10's 404/409 split (unknown → 404; real but cannot narrate here → 409); D33's rule that an explicit choice never silently changes; `resolve` is the existing gate for preset names (`feed/voice.py:77-90`).
- rejected: falling back to Automatic or to `pick_speaker`; validating the name in the store (it holds no catalog rule today).
- where: `web/app.py:_store_narrator`, `_narrator_picker`; `feed/build.py:build_presentation`

### D39 — no schema version bump; no CHECK on an existing table; `test_migrate.py` untouched (2026-11-02, PLAN, planner)
- what: `SCHEMA_VERSION` stays 14; the CHECK is only in the CREATE; an existing table gets `preset_speaker` through `_ADDED_COLUMNS` with no CHECK, and `set_narrator` enforces the exclusion everywhere; the old-table migration test goes into `tests/test_saved_voices.py`.
- why: `_ensure_columns` runs on every migrate, not gated on the version (`db/connection.py:56-69`, `:99`); version 14 itself is not merged yet; SQLite's `ALTER TABLE ADD COLUMN` takes no table CHECK, and a table rebuild of user data is more risk than an enforced write path; §1.5 rule 4 keeps `tests/test_migrate.py` for the user's hunks.
- rejected: `SCHEMA_VERSION` 15; a copy-and-swap table rebuild to add the CHECK; a test appended to `tests/test_migrate.py`.
- where: `db/connection.py:_ADDED_COLUMNS`; `db/schema.sql:memory_narrators`; `feed/saved_voices.py:set_narrator`

### D40 — what sits around the picker (2026-11-02, PLAN, planner; t18)
- what: the "Saved voices" group shows only when it has an option, or with the disabled "Voice copy not available on this device"; "Add a voice under Organize → Saved voices." shows whenever voice copy speaks the language here, with or without saved voices; the t18 bracket line follows the STORED choice when the page renders (saved → "another saved voice", else "another preset voice"); all the lines live in the picker macro, so each page shows them once.
- why: t17 fixed the hint and the disabled line but not when each shows; an empty `<optgroup>` reads as broken; the hint helps also when the user wants one more voice; t18 gives the bracket line per narrator but not which narrator, and after each Prepare the page reloads with the new stored choice (`feed.html:291-292`).
- rejected: a link in the hint (t17: text only); an empty "Saved voices" group; a script that swaps the bracket line when the select changes (more code for one sentence).
- changed (2026-11-02, t18): the earlier line "the option 'Automatic' keeps its text" is reversed; it reads "Automatic — the app picks a preset voice for this memory", and the picker tests compare that text.
- where: `web/templates/feed.html`; `web/app.py:_narrator_picker`
- changed (2026-11-02, Phase 5, pair t30): the hint reads "Add a voice under Voices." and links to `/voices` (t30: "its hints link to Voices").
  This reverses the rejected "a link in the hint (t17: text only)". The other lines stay.

### D41 — where the Phase 3 tests and the extra doc lines go (2026-11-02, PLAN, planner)
- what: the card tests go into `tests/test_web_feed.py` beside the existing card tests; `film_states` / `film_narrator` into `tests/test_feed_store.py`; `voice_label` into `tests/test_feed_voice.py`; the docs list adds §11.3's `voice` row (`docs/design.md:2150`) and the comments `feed/voice.py:3-4`, `feed/build.py:300-301` to t17's list.
- why: `tests/test_web_feed.py` already has the memory fixture and `FakeModels` (`:78-97`) and the card tests (`:206-213`, `:724-765`), and `tests/test_web_organize.py` has no memory; every new function needs a test (`ivms777/CLAUDE.md:276-281`); those three lines would otherwise say that only Automatic casts presets, or that the user never picks the speaker (`ivms777/CLAUDE.md:17-19`).
- rejected: a new organize test file; leaving those lines stale.
- where: PLAN.md §2.14, §5

### D42 — how the Automatic choice is shown (2026-11-02, PLAN, planner; t18)
- what: the card says only "(auto)" or "(your choice)"; `/feed` says who chose it: "your choice", "picked automatically by the story", "picked automatically using the memory's mood", or "picked automatically" when the row cannot tell; "Other voices" is one line, entries "passage N → <label>" joined with "; "; the reader (`film_narrator`) is pure and lives in `feed/store.py`, beside `film_states`, so the card and `/feed` read one rule.
- why: t18 fixed the words and the read rules for old rows, not the layout of several passages or where the rule lives; one line keeps the ready screen calm; one reader keeps the card and the page from disagreeing.
- rejected: one line per passage; the reason on the card (t18 gives the card the short form); reading the reason from the current `memory_narrators` row (t18 forbids it).
- where: `feed/store.py:film_narrator`; `web/templates/feed.html`; `web/templates/_album_card.html`

### D43 — the `/feed` header details that t20 left open (2026-11-02, PLAN, planner; pair t20)
- what: `<header class="feed-head">` is the only positioned element (`position: fixed; top: 1rem; left: 1rem; right: 1rem`); `.feed-actions` and `.feed-settings` are its two children; the settings region is `max-height: 45vh; overflow-y: auto` and at most `min(36rem, 100%)` wide; the header has `z-index: 10` and `pointer-events: none`, and the action controls and `.feed-settings` have `pointer-events: auto`; the region is left-aligned; under 66rem the notes sheet is capped at `calc(55vh - 7rem)` when the settings region exists.
- changed (2026-11-02, pair t21): the `z-index`, the `pointer-events` on `.feed-settings`, the left alignment and the notes cap were added after the plan check; `.feed-start` (`feed.css:49-62`) would cover the header, a region without pointer events cannot scroll, and the notes sheet (`feed.css:391-404`) overlapped the region.
- why: t20 fixed the structure, the order and the "bounded, scrolls inside" rule, not the names or the values; a fixed full-width strip would otherwise take the clicks meant for the film under its empty parts; 45vh leaves the title card and the picture visible on a 360 px phone; the width cap keeps long option text readable on a wide screen.
- rejected: per-control fixed offsets (the cause, t20); a `<details>` around the help lines (D40, t20); a header that pushes the film down (the canvas is `position: fixed; inset: 0`, `feed.css:16`).
- where: `web/templates/feed.html`; `web/static/feed.css`

### D44 — where the t19 guard test and the t20 tip go (2026-11-02, PLAN, planner)
- what: the `chat.js` guard is tested by reading `web/static/chat.js` in `tests/test_web_chat.py`; the recording tip is one `<p class="voices-tip muted">` inside each available `.voice-add` of `_voices_list.html`, before Record and Upload, tested on the page and on the list fragment in `tests/test_web_voices.py`.
- changed (2026-11-02, pair t21): the tip moved from `voices.html` to `_voices_list.html:57-72`; the "Add an example" area lives there, and a list refresh (`voices.js:41-48`, `:89`) re-renders it.
- why: the repo has no JS test runner and already tests JS by reading the source (`tests/test_web_feed.py:695`); `tests/test_web_chat.py` has no uncommitted hunks; the tip matters at the moment of recording, not in the page head.
- rejected: a browser test (the user drives the UI, `ivms777/CLAUDE.md:163-181`); the tip in the page head.
- where: `tests/test_web_chat.py`; `web/templates/_voices_list.html`; `tests/test_web_voices.py`
- changed (2026-11-02, Phase 4): the tip's second sentence becomes "Films use one example per language: the newest, unless you choose one with Use
  for films." (§2.16 F). Its place and class stay.
- changed (2026-11-02, Phase 5 Slice V): the tip moves with the "Add a take" row into `_voice_body.html` (§2.17 V3); its text stays.

### D45 — reading-text ids and the small rules t20 left open (2026-11-02, PLAN, planner; pair t20)
- what: the eight texts of §2.15 I; ids `<language>-<theme>` (`en-kitchen`, `en-market`, `en-rain`, `en-train`, the same themes in `ru`); the prompt check runs after the `source` check and before the size and WAV checks, with the text "Unknown reading text for <Language>: '<id>'."; an uploaded example ignores a posted `prompt_id`; "the last one used" is the last SAVED take in the page session; the panel binds the voice and the language at the Record click; "Different text" wraps to the first text; the box and buttons are `.recorder-prepare`, `.recorder-next`, `.recorder-start`.
- why: t20 asked the planner for the texts, and it refuses "an id of another language", which needs ids that name their language; plain numbers would read like voice_check's EN-1..5; a cheap check first refuses a bad id before the WAV work; an upload has no reading text (t20: `uploaded → None`); a discarded take was not a recording the user kept, so it does not move the text on; binding at the click keeps the text and the language together.
- rejected: shared ids across languages (the other-language refusal could not work); `1..4` ids; a 400 for a `prompt_id` on an upload (old or odd clients would fail for a field that changes nothing); counting a discarded take as used.
- where: `feed/saved_voices.py:RECORDING_PROMPTS`, `add_example`; `web/static/voices.js`; `web/templates/voices.html`

### D46 — "Use for films" is a row per voice and language, with a tombstone (2026-11-02, PLAN, planner; t24)
- what: a new table `voice_film_examples(voice_id, language, example_id, chosen_at)`, primary key `(voice_id, language)`; `voice_id` cascades with
  the voice; `example_id` is `ON DELETE SET NULL`, so a deleted chosen example leaves a row with NULL (the tombstone); no row = the newest; an index
  on `example_id`; no schema version bump.
- why: t24 needs "a deleted chosen example shows as unavailable … never a silent substitute"; a flag on `voice_examples` disappears with the example,
  and the newest would then take over in silence; the same tombstone shape already works for `memory_narrators` (D33); a new table reaches an
  existing DB through `migrate` alone (§1.7), so D39's rules hold.
- rejected: a `used_for_films` column on `voice_examples` (no tombstone; also an `_ADDED_COLUMNS` entry); a column per language on `saved_voices`;
  clearing the choice on delete (the silent substitute t24 forbids); `SCHEMA_VERSION` 15.
- where: `db/schema.sql:voice_film_examples`; `feed/saved_voices.py:film_example`, `set_film_example`
- changed (2026-11-02, Phase 5, t28 + t30): the table stays. For a baked voice its take is the prompt take; for a voice with no usable model it
  is still the take films copy. Slice V shows it as one select per language (D69).
- changed (2026-11-02, rework, t39): superseded. Films no longer use a take, so "Use for films" and its tombstone go. The table
  `voice_film_examples` stays with its rows and is no longer read or written (D81).

### D47 — one resolver for films, the picker, the badge and the test (2026-11-02, PLAN, planner; t24)
- what: `film_example(conn, owner_id, voice_id, language) -> FilmExample(example, chosen, problem)` decides which example films use; `castable` calls
  it, so the build, the `/feed` picker and the Prepare check follow it; the page badge reads it; `reference_for` (moved from `feed/build.py:_reference`)
  turns an example into the `/speak` reference for a film and for a test alike; a chosen example that is deleted or being deleted makes the voice
  not castable in that language, with its own text on the page, in the picker, at Prepare (409) and in the build (`failed`).
- why: t24 asks for one resolver shared by films and tests; one function cannot disagree with itself, so the badge always names the example a film
  will send; sharing `reference_for` makes a test send exactly what a film sends, so the stored conditioning is the same cache (the bake); a pending
  chosen example must not fall back either, or a delete in progress would be a silent substitute.
- rejected: a separate resolver in the page; a test that resolves "the film example" instead of the example the user pressed (the user compares
  examples one by one); falling back to the newest when the chosen one is pending.
- where: `feed/saved_voices.py:film_example`, `castable`, `reference_for`; `feed/build.py:_saved_cast`; `web/app.py:_narrator_picker`, `_store_narrator`
- changed (2026-11-02, Phase 5): `film_example` stays the one TAKE resolver; `voice_model_for` (D73) is the one MODEL resolver beside it, and
  `reference_for` carries both.
- changed (2026-11-02, rework, t39): superseded for films, the picker and the test: `film_model` (D77) is the one resolver, on voice models.
  `film_example` and `set_film_example` leave the code; `reference_for` stays only for the check script (D82).

### D48 — no "back to the newest" control; a confirm on deleting the chosen example (2026-11-02, PLAN, planner)
- what: "Use for films" on an example is the only control; a tombstone ends only when the user chooses another example; the Delete form of the
  chosen example asks "This example is used for films in <Language>. Films narrated by «<name>» in <Language> will fail until you choose another
  example. Delete it?"; other example deletes keep no confirm.
- why: t24 leaves "asks for a new choice" to the planner; one control keeps the page small; a delete that makes films fail must say so before it
  happens; deleting a non-chosen example changes no film choice, and the newest default moving is the documented D19 rule.
- rejected: a "Use the newest" reset (more UI for a rare case; the user can choose the newest with the same button); auto-choosing a new example
  when the tombstone's language gets a new one (a silent substitute); a confirm on every example delete.
- where: `web/templates/_voices_list.html`; `feed/saved_voices.py:set_film_example`
- changed (2026-11-02, pair t25): "Use for films" shows on every usable example except the chosen one, the default newest included, so the user
  can pin the newest before recording another; every example that films do not use reads "not used for films". The rejected "Use the newest"
  reset stays rejected: pinning covers it.
- changed (2026-11-02, Phase 5, t30): the per-row "Use for films" button becomes one select per language (D69). There is still no reset
  option, and the confirm on deleting the chosen take stays.
- changed (2026-11-02, rework, t39): superseded. There is no film take, so no confirm guards a take delete (D87).

### D49 — the test texts (2026-11-02, PLAN, planner; t24)
- what: three kinds: the reading text of a recorded example, labelled "not checked against the recording"; the fixed sentences EN-1..5 / RU-1..5,
  copied from `scripts/voice_check.py:47-71` into `feed/voice_preview.py:PREVIEW_SENTENCES` with a test that compares the two; the user's own text,
  1-300 characters after strip; the language is always the example's; an uploaded example has no reading text.
- why: t24 asks for "fixed unseen same-language test sentences"; the §6.1 sentences are unseen by design (no reading text uses them, §2.15 I), so a
  page test and the user's voice-check run use the same words; app code cannot import `scripts` (`pyproject.toml:75`); 300 characters is about
  21 s of speech at 14 characters a second (D28), a short test and not a film; t24: no ASR, so an upload gets only fixed or own text.
- rejected: new sentences (two fixed sets to keep apart); importing `scripts.voice_check` at runtime; a language select for the test (cross-language
  is reported only, D19); a longer own text.
- where: `feed/voice_preview.py:PREVIEW_SENTENCES`, `preview_text`; `web/templates/voices.html`
- changed (2026-11-02, Phase 5 Slice B): a take whose words the user confirmed offers "The words of this take (checked)" instead of the
  "not checked against the recording" label (D71); an upload with confirmed words offers them too. The fixed and own texts stay.
- changed (2026-11-02, rework, t39): the texts belong to the person's test (D80): a fixed sentence of the chosen language, the words of any take
  of that language that has words ("Words of take #<id> (checked)" / "(not checked)"), or the user's own text (≤ 300). The `reading` kind of one
  take goes.
- changed (2026-11-02, ASR, t42): a take's option is "Your words in take #<id>", offered only for a take whose words ASR heard; a take
  stored with the shown text is not offered until a bake hears it (§2.17 K5, D101).

### D50 — one stored test per example, owned by the voice outbox (2026-11-02, PLAN, planner; t24)
- what: `voice_previews` keeps the LAST test of each example (primary key `example_id`): text kind, sentence id, text, model, revision, built or
  restored, the call's wall time, the audio length, and a content-addressed WAV in `data_dir/voices`; a new test replaces the row and sends the old
  key to `voice_file_deletions`; the drain's cascade and ref check include the table; a test of an older revision stays playable and says so.
- why: t24 wants Original vs Generated side by side and test files owned by the voice outbox; one row per example bounds the disk use and is all a
  side-by-side needs; the write-ahead file write (§2.7) already handles a crash and a delete during a test; showing the revision keeps a test bound
  to the model that made it.
- rejected: a test kept only in memory (lost on a restart, and nothing to compare later); a history of every test (unbounded files for no asked
  need); hiding an older-revision test (the user may still want to hear it); film-audio storage (`data_dir/audio` is pruned by the film code, D34).
- where: `db/schema.sql:voice_previews`; `feed/saved_voices.py:save_preview`, `previews`, `_cascade_one`, `_remove_one`
- changed (2026-11-02, rework, t39): superseded by D80: one stored test per VOICE in `voice_tests`, with the model version. `voice_previews` stays
  with its rows and files and is no longer read or written (D81).

### D51 — the test job: the film pattern, no queue; Cancel drops the test but keeps the conditioning (2026-11-02, PLAN, planner; t24)
- what: one test at a time per process; `VoicePreviews` keeps a run id, a state dict and a daemon thread, as the film does (`web/app.py:1170-1280`);
  a new Generate during a test bumps the run and starts a new thread at once (the old one is dropped at its next check); Cancel bumps the run and
  frees the page at once; the model call already running ends in the background; a built conditioning is stored also after a cancel; the test WAV
  is not; stages "waiting for the voice model" (2%), "reading the example" (5%), "generating the voice" (10-97%, `_Ticking` on
  `len(text) / 14 × CLONE_RTF_ESTIMATE`), "saving the test" (97%).
- why: `ivms777/CLAUDE.md:264-274` (stage, percent, Cancel; a second request replaces, never a dead end) and t24 ("reuse the pattern"); the models
  service has no cancel route (§1.7), so the honest Cancel is the film's; the voice lock (D52) already orders the model calls, so the film's
  `feed_queued` is not needed; a film keeps the conditioning of a cancelled build too (`feed/build.py:403-405` runs before the next stage check),
  and the blob is a cache the child checks itself (§2.5), so keeping it cannot make a wrong voice.
- rejected: a queue slot like `feed_queued` (the lock does it); dropping the conditioning on cancel (it throws away an encode for nothing); a
  models-service cancel (a change to the one gateway and its children, not asked).
- where: `web/voices_api.py:VoicePreviews`; `feed/voice_preview.py:run_preview`
- changed (2026-11-02, pair t25): the "Not claimed" exception is gone. Step 5 of `run_preview` (the run check, `save_preview` and `on_saved`,
  which marks the state `done`) runs while the `VoicePreviews` job lock is held, and `start` and `cancel` take the same lock; no model call runs
  under it. So a superseded test is never stored after its `cancel()` or `start()` returned. Why: the window was small but real, and the barrier
  costs one lock around one SQLite write and one file write. Rejected: a DB flag for cancel (the run id lives in memory, and the barrier is enough).
- changed (2026-11-02, pair t25 round 2): Cancel is logical, not a GPU stop (§2.16 D). The cancel route never waits for synthesis and may wait for
  a publication in progress; `run_preview` checks `cancelled()` once more just before `client.speak`, outside the speak call's broad handler;
  a call already sent runs to its reply (`modelsvc/torch_process.py:169-186`), and `voice_calls` stays held until the client call returns or
  raises, plus the conditioning store; the 1800 s is the HTTP timeout of `ModelsClient` (`inference/models_client.py:15`, `:101-107`), not a GPU
  deadline; no "within 1 s" promise. Why: the earlier text implied a stop the code cannot make. Rejected: `model_unload` or `stop` as a cancel (it
  waits for the same call, and it would also end the resident model for the next film).
- changed (2026-11-02, rework, t39): the job pattern, Cancel and the publication barrier stay; a test builds and stores no conditioning (no take
  is sent), so "keeps the conditioning" no longer applies. A test resolves the voice model under the voice lock and uses the version it finds.

### D52 — one voice call at a time per `app` process (2026-11-02, PLAN, planner)
- what: `app.state.voice_calls = threading.Lock()`; the film's speak call and the test's speak call each run inside `voice_turn(lock, …)`; a waiting
  call shows "waiting for the voice model" and still honours Cancel; `build_presentation(…, voice_lock=None)` keeps every existing caller and test.
- why: a test can now run beside a film; when the two need different voice models, `ensure_voice_for` swaps the unit outside `_run`
  (`modelsvc/slots.py:211-230`). **Corrected 2026-11-02 (t25, round 2):** a swap does not cut a running call (`stop` waits for it,
  `modelsvc/torch_process.py:169-186`); the failure is a swap between one request's `ensure_voice_for` and its call, which sends that call to the
  other model's child (§1.7), where it fails (a preset call on VoxCPM2 has no references; a saved call fails the identity check, D26). A lock in
  `app` covers both callers and lets each page say why it waits.
- rejected: a lock inside the models service (it would also cover the user's own `voice_check` run, but it changes Phase 1 code and the page could
  not say why it waits); refusing a test while a film builds (a dead end, `ivms777/CLAUDE.md:271-274`); waiting for the whole film (music and the
  conductor do not use the voice unit).
- where: `feed/build.py:voice_turn`, `build_presentation`; `web/app.py:_start_film`; `web/voices_api.py:VoicePreviews`
- changed (2026-11-02, pair t25): `voice_turn` checks `cancelled()` before the first acquire and again after an acquire succeeds, and releases
  the lock in `finally`. The film's turn covers the references read (`feed/build.py:370-379`), the speak block and `_keep_conditioning`
  (`:402-405`), not the speak block alone. Why: a stale test must not take the voice unit after it was cancelled; a film that read its
  references before the wait could miss a blob a test stored during that wait, and the conditioning must be stored before the next caller
  reads it, so film and test restore each other's blob in both orders. The cast and its examples are fixed before the wait.

### D53 — the test panel lives outside the list; a running test is not an Esc level (2026-11-02, PLAN, planner)
- what: `#voice-preview` sits in `voices.html` outside `#voices`, as the recorder panel does; its job line is a server fragment polled every 1 s; Esc
  order: live recording → unsaved take → prepared panel → focused text field → the test panel when no test runs → close the page; a running test
  is left running when the page closes and is shown again on return; Start recording pauses every `<audio>` of the page.
- why: `replaceList` and the deleting poll swap all of `#voices` (§1.7), so a panel inside it would lose its text and state (t24: "survives the
  list refresh"); a test runs in `app` like a film build, and `/feed` leaves a running build alone on Esc (`web/templates/feed.html:293-298`); the
  existing order and its tests stay, with the new level added just before close; with echo cancellation off (`voices.js:193`), a playing clip
  would be recorded into the take.
- rejected: the panel inside each example row; Esc that cancels a running test (a minute of GPU work lost to a key meant to go up); blocking Generate
  while recording (no shared resource).
- where: `web/templates/voices.html`; `web/templates/_voice_preview.html`; `web/static/voices.js`
- changed (2026-11-02, pair t25 + planner): `previewRun` counts as used only after a successful list refresh; typing, a failed fetch or a
  non-ok reply keeps the completion pending and tries again in 1 s; a fragment of an older run is ignored. The planner added `data-epoch`
  (a per-process token): the app hot-reloads (`ivms777/CLAUDE.md:206-216`) and its run counter then starts at 0 again, so a run number alone
  would make the page ignore every fragment of the new process. Rejected: comparing runs without the epoch.
- changed (2026-11-02, pair t25 round 2; `<base>/.pair/files/merge-preview-epoch-cancel-t25.md`): the epoch is `secrets.token_hex(16)` per
  `VoicePreviews` instance, an identity and never an order; runs are compared only within one epoch. `previewSeen = {epoch, run, state}` is seeded
  from the page's own fragment. An unknown epoch is never applied: the page bumps its `generation` and GETs the status with `no-store`; only that
  reply adopts the epoch, the old one goes into `retiredEpochs`, and a mutation is never replayed. Completion identity is `{epoch, completed}`
  (`VoicePreviews.completed`, set by `on_saved`, carried by every fragment as `data-completed`); one guarded refresh loop serves it, and
  `previewRun` is renamed `previewRefreshed`. Every confirmed epoch change runs one restart-sync refresh (`{newEpoch, 0}`). A failed mutation keeps
  its error and GETs the status with `no-store`. A restart during a running test shows "The app restarted, so this test stopped. Generate it
  again." Why: 16 random bytes make a repeat across reloads practically impossible, while a clock epoch fails when the clock steps back (a Generate whose reply is
  ignored is a button that does nothing, `ivms777/CLAUDE.md:266-269`); a `done`-state trigger misses a completion when Cancel lands after the
  publication, or when a refused Generate drops a deferred refresh, and the list would then show the old text for the new WAV; a reload kills the
  daemon thread silently (`ivms777/CLAUDE.md:176-178`). Rejected: a wall-clock epoch; a `done`-state completion trigger; comparing runs across
  epochs; replaying a mutation after a restart.
- changed (2026-11-02, Phase 5, t30): the test panel lives in the Test tab of `/voices/{id}`, still outside `#voices`; it is no longer an Esc
  level (tabs are none); the Esc order is §2.17 V4.
- changed (2026-11-02, rework, t39): the text select moves into `#voices` (as `.test-text`, kept across a swap by `carryInputs`), because its take
  options change with the takes; the textarea, Generate, the job line and the comparison stay in the panel outside `#voices`.

### D54 — routes, status codes, and no workbench without voice copy (2026-11-02, PLAN, planner)
- what: the five routes of §2.16 E, registered before the `/organize/voices/{voice_id}/…` routes; a test and "Use for films" answer 409
  `NOT_AVAILABLE` when no cloning entry speaks the language on this profile; 404/400/409/507 as §2.16 E; the job fragment is a Jinja template.
- why: the same split as §2.9 (unknown → 404, a real target that cannot act now → 409, bad input → 400) and as `_store_narrator`; a test writes a
  file, so it takes the upload disk guard; the film's progress fragment is hand-built HTML (`web/app.py:1249-1266`), and a template escapes every
  field by itself.
- rejected: allowing "Use for films" on a profile with no voice copy (no film there can use it, and the picker refuses saved voices there); one
  combined "test and choose" button (a test must never change casting, t24).
- where: `web/voices_api.py`
- changed (2026-11-02, Phase 5, t30): every route moves to `/voices/*` (§2.17 V1); the codes stay.
- changed (2026-11-02, rework, t39): the per-take routes `POST /voices/{id}/examples/{eid}/preview`, `GET …/preview/audio` and `POST
  /voices/{id}/film-example` go; `POST /voices/{id}/preview` and `GET /voices/{id}/preview/audio` come (D85); a test of a voice that cannot speak
  through a model answers 409 with the K5 text.

### D55 — where the Phase 4 code and tests live (2026-11-02, PLAN, planner)
- what: the store in `feed/saved_voices.py`; the job logic in a new `feed/voice_preview.py`, which imports `CLONE_RTF_ESTIMATE`, `Cancelled`,
  `_Ticking`, `_reason` and `voice_turn` from `feed/build.py`; the job state and routes in `web/voices_api.py`; the job fragment in a new
  `_voice_preview.html`; tests in `tests/test_saved_voices.py`, `tests/test_voice_preview.py` (new), `tests/test_feed_build.py`,
  `tests/test_web_voices.py`, `tests/test_web_feed.py`; lane order 18 → 19 → 20a → 20b, with 20c beside 20a.
- why: `feed/` is in the wheel and the one-model gate (D29); reusing the film's ticking bar, cancel type and reason reader keeps one behaviour;
  `feed/voice_preview.py` imports `feed/build.py`, so `voice_turn` must live in `feed/build.py` to avoid an import cycle; every new module needs
  tests (`ivms777/CLAUDE.md:276-281`); disjoint files per lane.
- rejected: the job inside `feed/build.py` (a film module would hold page logic); a copy of `_Ticking`; a new top-level package.
- where: PLAN.md §2.16, §4, §5

### D56 — what the main step runs, and where the code review runs (2026-11-02, PLAN, planner; pair t25)
- what: the main steps 16 and 21 run only the targeted tests of the changed files, the tests of their direct callers (listed per step in §4) and
  `uv run ruff check <changed files>`, plus the mermaid check and the docs-vs-code check; `sadensmol-code-review` runs only inside `task finish`,
  never in a main step; §7 item 3 is met there.
- why: the task skill's test-scope rule and its rule that the review belongs to `task finish` (pair t25); the direct-caller lists come from
  `grep` over the imports of the changed modules (2026-11-02): `feed/build.py` → `tests/test_feed_metadata.py`; `ingest/pipeline.py` →
  `tests/test_pipeline_pass.py`; `castable` → `web/app.py` (`tests/test_web_feed.py`); the schema → `tests/test_migrate.py` (run, never edited,
  §1.5 rule 4); a new `feed` module → `tests/test_one_model_process_gate.py`.
- rejected: the full suite at each main step (outside the test-scope rule); a code review in a main step (the review is `task finish`'s gate).
- where: PLAN.md §4 steps 16 and 21; §5; §7 item 3

### D57 — a bake runs in a `voice_train` child under a governor lease; one VoxCPM2 copy (2026-11-02, PLAN, planner; t28)
- what: a new torch unit `voice_train` hosts `audio.voxcpm_trainer:VoxCPMTrainer`; a bake first unloads `voice`, then takes `governor.lease("voice_train")`
  and `scheduler.reserve(["voice_train"], BATCH)` for its whole length; the child trains on a thread of its own and answers `status` and `cancel`
  between steps; at the end the unit is unloaded; every voice request answers 409 while a bake runs.
- why: t28 keeps training in a models-service child under the scheduler (`ivms777/CLAUDE.md:120-141`); training needs another model state than
  inference (`voxcpm2.py:1143-1182`), so it cannot reuse the voice child; one `call` holds the `TorchWorker` lock until it returns (§1.8), so a
  blocking `train` call would hide all progress and make an eviction wait the whole bake inside the governor lock; the governor evicts anything
  outside the current op's keep set (`governor.py:51-64`), so without a lease a music load could end a bake; one copy of a 9 GB model is the budget
  rule of §8.1.
- rejected: training in the voice child; a thread in the models-service parent (it imports no torch); a single blocking `train` call; progress
  through files beside a blocking call; letting `/speak` load the voice child next to the trainer (two copies).
- where: `modelsvc/slots.py:_register_voice_train`; `modelsvc/governor.py:lease`; `modelsvc/voice_bakes.py`; `audio/voxcpm_trainer.py`
- changed (2026-11-02, pair t31): one admission barrier covers the whole voice path. `VoiceGate`: `speak` enters before `ensure_voice_for`; a bake
  closes the gate, drains every accepted speak (also one waiting in `_admit`), then unloads `voice`, trains, and opens the gate after the teardown.
  `model_ensure("voice")` → 409 during a bake, `model_ensure("voice_train")` → 409 outside one. `_register_voice_train` registers only once. Why:
  a speak that passed an `active()` check could load `voice` after the unload (§1.8), and then only the budget sums kept one copy;
  `_register_torch_unit` unloads first, which a lease does not guard. Rejected: the `active()` check alone; trusting the budget sums.
- changed (2026-11-02, pair t32 and t35, in the code): `model_ensure("voice")` enters the counted gate like a speak, so a bake's drain also waits
  for manual voice ensures; `model_ensure("voice_train")` ALWAYS raises `BakeOnly` (409), before the gate, the scheduler and the registry. Why: a
  manual ensure that passed a closed-check could load `voice` beside the trainer; only a bake may load `voice_train`. Rejected: a second gate; a
  scheduler change; checking `active()` (t32, t35).

### D58 — the training recipe: the package's LoRA defaults and the planner's loop defaults (2026-11-02, PLAN, planner)
- what: LoRA = `LoRAConfig` defaults with `enable_lm` and `enable_dit` on (r 8, alpha 16, dropout 0, q/k/v/o; `voxcpm2.py:134-147`, `core.py:48-55`);
  the loop = AdamW over the LoRA weights, learning rate 1e-4, batch 1, weight decay 0.01, gradient clip 1.0, seed 42, loss = `loss/diff + loss/stop`,
  steps = `min(1000, max(200, 20 × takes))`; the request may set each, within the `train_config` ranges; every value is stored with the result;
  one adapter per voice from its takes in every language.
- why: the package pins the LoRA shape but ships no loop, no config and no script (§1.8); the loop values are a common starting point that no
  source here pins, so they are labelled the planner's and Slice A measures them; batch 1 keeps the activation memory smallest on an unmeasured
  MPS run; 20 passes over each take with a floor and a ceiling bounds the time for 3 takes and for a full kit; one adapter per person is what the
  user asked ("the person's model").
- rejected: copying values from the online guide (not in the pinned source); larger batches; per-language adapters (two bakes, two files);
  projections on (`enable_proj`, not the package's own choice); full fine-tuning.
- where: `models/catalog.py:AdapterSpec`; `audio/voxcpm_trainer.py:TrainConfig`, `default_steps`, `train_config`, `run_training`
- changed (2026-11-02, pair t31): text ids come from `ds.add_column("text_ids", …)` with `tok = model.text_tokenizer`, never from a `ds.map`
  closure over the model (the `datasets` fingerprint pickler would copy every tensor, §1.8; read from the code, not seen in a run); the batches are
  `while True: yield from loader`, so each pass shuffles again. Rejected: `ds.map(…, new_fingerprint=…)` (one more rule to keep right);
  `itertools.cycle` (it keeps every batch).

### D59 — the bake request carries the takes by value; the service owns a work folder and sweeps it (2026-11-02, PLAN, planner)
- what: `POST /voice/bakes` takes every clip as base64 with its sha256, words and language (≤ 300 clips, ≤ 4 MiB each, ≤ 192 MiB in all); the
  service writes them and `train.jsonl` into `<data_dir>/voice-bakes/<bake_id>/`, keeps the HF `datasets` cache in `<data_dir>/voice-bakes/hf-cache`
  (offline), removes the cache after each bake, the previous finished folder at the next start, a folder on `DELETE`, and everything at start.
- why: D15's rule — the models service holds no database and no store of user data; the same contract as `/speak` references; a full kit at
  48 kHz is tens of MB, which one local request carries; the folder holds copies of a person's voice, so nothing may outlive a bake or a restart.
- rejected: reading `data_dir/voices` from the models service (it would read `app`'s store and need its key mapping; the models image has no
  `storage/local.py`, §1.8); multipart (a second request shape); a lasting dataset cache.
- where: `modelsvc/voice_bakes.py:VoiceBakes.start`, `sweep`; `inference/models_client.py:bake_start`

### D60 — the adapter file, its identity, and `/speak` by value with a switch per run (2026-11-02, PLAN, planner)
- what: the adapter is a safetensors file of `get_lora_state_dict()` with string metadata (format, model, revision, voxcpm, LoRA JSON, dataset hash,
  config, steps); `/speak` references may carry it (≤ 4 per request, ≤ 64 MiB each, sha256 checked); the child keeps one loaded adapter by sha256
  and, per run, keeps it, resets, or resets then loads and checks it; a metadata mismatch or a missing or skipped key fails the call with the reason.
- why: `load_lora_weights` copies matching keys and skips the rest in silence (`voxcpm2.py:1233-1241`), so a wrong file would speak a part of a
  model; a reset first makes every switch start from zero B (`:1249-1252`); by value keeps the voice data with `app` and its outbox (D15); one child
  speaks one generation at a time, so a switch never races.
- rejected: an adapter cache in the models service keyed by sha256 (outlives a delete, needs a forget call); a path into `app`'s store (D59);
  trusting `load_lora_weights` without the key check; the voxcpm version as a refusal reason (the keys check covers the shape).
- where: `audio/voxcpm_trainer.py:adapter_metadata`; `audio/voxcpm_voice.py:_use_adapter`, `adapter_mismatch`, `load_mismatch`; `modelsvc/app.py:VoiceAdapter`
- changed (2026-11-02, pair t31): the names and shapes of the file are checked against the exact LoRA set BEFORE `load_lora_weights`
  (`tensor_mismatch`); on ANY failure the child resets the LoRA weights and clears the kept sha256. Why: `load_lora_weights` copies into any
  parameter with a matching name, base weights included, and can raise halfway through (`voxcpm2.py:1229-1241`); a reset never restores a base
  weight; a kept sha after a failure would answer "kept" for a half-loaded model. Rejected: checking only after the load.

### D61 — the voice child is always built with the catalog's LoRA config (2026-11-02, PLAN, planner; t28)
- what: the `voice` child of an entry with an `AdapterSpec` gets the LoRA JSON as a fourth arg and builds `VoxCPM.from_pretrained(…, lora_config=…)`;
  with no adapter loaded every B is zero. The LoRA layers (about 18 MB in float32, an estimate) are declared inside `cost_mb` 10500 in the catalog
  note (changed 2026-11-02, pair t31).
- why: `load_lora` needs a config at build (`core.py:317-335`), and LoRA layers exist only when the model is built with one (`voxcpm2.py:246-247`);
  a zero B adds no adapter contribution (`lora.py:58-65`); the extra LoRA matmuls still run and are not measured, so §6.5 reports B's RTF ("today's
  films on the LoRA-built child") and nothing promises a speed (changed 2026-11-02, pair t31); the conditioning cache is VAE-only
  (`voxcpm2.py:681-740`), so stored blobs stay valid.
- rejected: rebuilding the child to load an adapter (a full model load per switch); a second inference child for baked voices (two copies).
- where: `modelsvc/slots.py:_register_torch_unit`; `audio/voxcpm_voice.py:VoxCPMVoice.__init__`

### D62 — `AdapterSpec` in the catalog; bakes offered on mac only until a cloud run is measured (2026-11-02, PLAN, planner)
- what: `ModelEntry.adapter: AdapterSpec | None` with `lora`, `train_cost_mb=14000` (an estimate), `train_cost_measured=False`,
  `train_profiles=("mac",)` and `guidance_minutes=(5, 10)`; `adapter_entry_for(profile)`; adapters are used wherever the entry runs.
- why: a model's metadata is declared beside it (`ivms777/CLAUDE.md:236-251`); the governor needs a cost for the new unit; the user chose "This
  Mac first (Recommended)" (`## Decisions`); the meter needs the authors' figure as data, not a literal in a template.
- rejected: constants in the trainer and the child (two copies of one shape); offering a cloud bake before any cloud run; a cost of 0 or of the
  inference figure (the governor would overcommit).
- where: `models/catalog.py`

### D63 — the service bake job: stages, percent, what Cancel stops, one bake at a time (2026-11-02, PLAN, planner)
- what: one bake per service; the stages and fractions of §2.17 E; the load stage moves on `LOAD_ESTIMATE_S` (display only); Cancel is checked
  before the reserve and after the load, and inside the child at each step boundary; a load in progress runs to its end first; the status reads
  `cancelled` only when the child says so; `finally` always ends the child and releases the lease.
- why: `ivms777/CLAUDE.md:264-274` (named stage, percent, Cancel); a training loop CAN stop between steps, unlike a running speak call (§2.16 D), so
  this Cancel is real at step granularity; `TorchWorker.start` holds its lock through a load (`torch_process.py:124-167`), so a load cannot be cut;
  the honest text says both.
- rejected: a "within N seconds" promise; killing the child to cancel (it would also drop a nearly saved model, and the `finally` does it anyway);
  a queue of bakes.
- where: `modelsvc/voice_bakes.py`
- changed (2026-11-02, pair t31): what ENDS a bake for a caller: a caller that cancelled waits for any end state (`cancelled`, `done`, `failed`),
  keeps no adapter, and deletes the bake; every caller deletes the bake on every end, because its folder holds copies of the takes (D59); the
  check script follows the same rules. A request on a profile with no bake answers "baking a voice is not offered on this device yet", apart
  from "voice copy is not available". Why: a done can arrive after a Cancel, and a wait for `cancelled` alone would hang; a folder left behind is a
  copy of a person's voice.
- changed (2026-11-02, pair t33 and t34, in the code): a cancel during the LAST training step saves nothing; a bake publishes its end, opens the
  gate and clears `active` in one lock block, so a next start sees all three or none. Why: a cancel seen after the last step was lost and the
  adapter was saved; a next start could remove the folder of a bake whose end was not yet published.

### D64 — Slice A's check reads the saved voice read-only and compares three variants blind (2026-11-02, PLAN, planner)
- what: `scripts/voice_bake_check.py --voice NAME` opens the database with `mode=ro`, lists the takes and their words, needs `--confirm-texts` and
  words for every take (`--text ID=WORDS` for this run), bakes through the service, keeps the adapter only in its report folder, and speaks the
  voice_check sentences in three variants (A adapter + prompt take, B prompt take alone, C adapter alone) under random names with a key.
- why: the user's request is to bake an EXISTING voice; D30's reason for a WAV input (validation before the UI existed) does not apply to a voice the
  user already recorded; read-only means the check cannot change the user's data; B is today's film path, so A and C are judged against what the
  user hears now; C answers whether the prompt take helps (t28: explain its role); the user confirms each take's words (t28).
- rejected: a page button (Slice A builds no UI); storing the adapter as the voice's model in Slice A (that is Slice B, with its outbox); a WAV list
  input (the takes are in the app already); judging likeness by a score.
- where: `scripts/voice_bake_check.py`
- changed (2026-11-02, rework, t39): C (the adapter alone) is the film path, so it is what the check judges (D65); A and B stay as diagnostics,
  and their prompt take is the newest take of the language (D82). Added (pair t38, in the code): the console output is blind, and `key.json` and
  `report.json` reveal the mapping, so the user listens first.

### D65 — Slice A PASS / FAIL thresholds (2026-11-02, PLAN, planner)
- what: PASS needs every automatic check (the bake ends `done` on `mps`; every A and C reply reports the adapter `loaded` or `kept`; every file has
  sound, no NaN, the identity `voxcpm2` @ the pin), `min_free_mb` ≥ 1024 during the bake, `seconds_per_step` ≤ 6.0, and, scored blind before the
  key, the better baked variant (A or C) at mean similarity ≥ B's mean + 0.5 with mean naturalness ≥ 3.5 and ≤ 1 wrong word per sentence. Any miss
  is a FAIL, which stops Slice B for the user's decision.
- why: the user decides on measured numbers (`ivms777/CLAUDE.md:223-232`); 1 GB of free memory above the governor's 512 MB headroom keeps the Mac
  usable; 6 s a step keeps a 600-step bake within an hour; a baked voice that does not sound clearly more like the person than one take is not worth
  Slice B's UI; the 1-5 scales and the word count are §6.1's.
- rejected: no thresholds (no decision point); a likeness promise; thresholds on the loss (no meaning for a listener).
- where: PLAN.md §6.5; `scripts/voice_bake_check.py` (the automatic checks)
- changed (2026-11-02, rework, t39): PASS needs C on its own: mean similarity ≥ B's mean + 0.5, mean naturalness ≥ 3.5, ≤ 1 wrong word per
  sentence, plus the automatic, memory and speed checks; A and B only help to diagnose (`scripts/voice_bake_check.py:59-65`). A FAIL no longer
  stops Slice B's code: it means saved voices cannot narrate on this Mac, which the main session reports as a blocker; no one-take path is added.
  Rejected: "the better of A and C" (films never use A, t39).

### D66 — three slices: A beside V, then B; each with its own user check (2026-11-02, PLAN, planner; t28 + t30)
- what: Slice A (engine, §6.5) and Slice V (the Voices page with today's features, §6.6) start after the gate and touch disjoint files except the
  docs, which one lane writes first; Slice B starts after §6.5 PASS, §6.6 PASS and the gate answer on deleted takes; all of Phase 5 ships in this PR.
- why: t28's first slice is a feasibility bake and the user chose the Mac; t30 runs V beside A; B needs the measured bake and V's page; the user tests
  each part as it lands (D32's change, the user's 2026-11-01 words).
- rejected: one slice for everything; V after B (the user wants the better page now); a second PR (the task holds Phase 5, t28).
- where: PLAN.md §4 Phase 5
- changed (2026-11-02, rework, t39): A and V are implemented; B starts after the gate of this rework and does not wait for §6.5 PASS, §6.6 PASS or
  a gate answer (the user: do the rework now). Only lane 31b waits, for the reload reason of D84.

### D67 — the Voices grid and leaf: cards, tabs, close, the deleting state, the texts that move (2026-11-02, PLAN, planner; t30)
- what: §2.17 V; a card is a plain link with name, per-language minutes, takes and film take; a pending card has no link; create opens the new leaf;
  `?tab` is `recordings` (default) | `model` | `test` through `replaceState`; close = `history.back()` only with `data-from-grid` (set from a
  same-origin `document.referrer` whose path is `/voices`), else `location.replace("/voices")`; a deleted voice's leaf closes after the reply; the
  three "Organize → Saved voices" texts in `feed/build.py:607-616` change with V (text only).
- why: t30 fixes the structure and leaves these details; a card link is a full navigation (§13.1), so the referrer names the grid exactly when
  the user came from it, and a DOM attribute survives bfcache and the htmx cache; `replace` keeps depth two (§13.1 rule 3); without the build texts
  a failed film would send the user to a page that is gone; t30 limits V to `web/`, docs and tests for LOGIC, and these three strings are no logic.
- rejected: `history.length > 1` (t30); a `ctx` (one grid, no state, t30); `location.assign` for the fallback (one more entry); leaving the build
  texts until Slice B.
- where: `web/voices_api.py`; `web/templates/voices.html`, `voice.html`, `_voice_cards.html`, `_voice_body.html`; `web/static/voices.js`; `feed/build.py:_saved_cast`
- changed (2026-11-02, rework, t39): a card's "Films: take …" line and the header's per-language film-take lines become the voice model state
  (§2.17 K6, D87).

### D68 — `voices.js` lifecycle: once-guard, delegated listeners, mount and unmount, a question before a nav, the recorder kept (2026-11-02, PLAN, planner; t30)
- what: §2.17 V5; also: `replaceList` detaches `#voice-recorder` before it swaps `#voices` and puts it back under the row it belongs to (the "Add a
  take" row in V, the kit row in B), or at the end of the takes when that row is gone.
- why: htmx 2.0.3 runs page scripts again on a history restore (t30, §1.8), so a second run must not add listeners or act on detached nodes; a nav
  click must not throw away a take without a question; the recorder must open where the user reads (t30) and still survive a list refresh (D53).
- rejected: a script that runs only on full load; per-node listeners; `beforeunload` for boosted navs (htmx never fires it); moving the recorder
  inside `#voices` with no detach (a refresh would destroy a live take).
- where: `web/static/voices.js`
- changed (2026-11-02, pair t31): `mount()` also runs on `pageshow` with `e.persisted` (pattern `organize.html:64-68`), because a native bfcache
  restore runs no script again and the page would come back without its poller; native Back and Forward and the htmx restore each end with
  exactly one poller on the current nodes.
- changed (2026-11-02, pair t36, in the code): every async site owns its reply through body decoding: a guarded `session.fetch`, `own` captured
  before the first await, `own.alive()` before each DOM effect, retry or recovery read. Slice B's bake poller follows the same rule.

### D69 — one film-take select per language, with its tombstone (2026-11-02, PLAN, planner; t30)
- what: the Voice model tab holds one `<select>` per language with takes; its options are the usable takes ("#id · m:ss · “text” (newest)" on the
  newest); the stored choice, else the newest, is selected; a change posts `film-example` (it pins the take, D48's change of t25); a tombstone shows
  a disabled, selected option "— the chosen take was deleted —" with the §2.16 F line; there is still no "back to the newest" option.
- why: t30 asks for one selector per person and language; a select is one control instead of a button per row; D48's reasons against a reset still
  hold; the tombstone must stay visible and fixable in the same place (t24).
- rejected: per-row "Use for films" buttons (t30); a "newest (automatic)" option (D48); hiding the tombstone.
- where: `web/templates/_voice_body.html`; `web/static/voices.js`
- changed (2026-11-02, rework, t39): superseded. The Voice model tab holds the voice model, not a film-take select; there is no film take.

### D70 — the recording kit (2026-11-02, PLAN, planner; t28)
- what: 26 texts per language: the four reading texts of §2.15 I as "normal" plus two for each of slow, fast, calm, happy, sad, serious, excited,
  question, emphasis, articulation, numbers and names; ids `<lang>-<style>-<n>`; an English instruction per style; `RECORDING_PROMPTS` is derived
  from the kit, so `add_example`'s check and D45 stay.
- why: the user asked for texts that cover slow and fast reading, moods and articulation; t28 adds questions, emphasis, numbers and names; 26 takes
  of 6-10 s are about 3-4 minutes per language, so both languages, or a second take of each text, reach the authors' 5-10 minutes; the app's
  UI is English; the voice_check sentences stay unseen (§2.15 I); D23 holds for Russian.
- rejected: one long text per style (a take over 30 s is refused, §2.7); instructions inside the words (they would be read aloud); a kit of 10
  texts (far below the guidance).
- where: `feed/saved_voices.py:KitText`, `KIT_STYLES`, `RECORDING_KIT`, `RECORDING_PROMPTS`
- changed (2026-11-02, ASR, t42): the kit texts are suggestions; `kit_id` records the text shown and is never the take's words; the
  recorder says "Read it, or say anything else: the bake hears what you say." (D99, D101).

### D71 — transcripts: the user confirms or corrects each take; uploads need their words; unresolved takes block the bake (2026-11-02, PLAN, planner; t28) — superseded (ASR, t42)
- what: `voice_examples.transcript_confirmed_at` and `kit_id`; a take's Save posts the words shown in an editable box (prefilled with the kit text)
  and confirms them; `set_transcript` confirms or corrects any take; old takes and uploads without words are unresolved, shown on their rows and
  listed beside the Bake button, which they block.
- why: t28's rules ("the user confirms or corrects the text of each take, and uploads need a transcript; unresolved clips are shown and block the
  bake; never dropped silently"); the reading text was never checked against the speech (§2.16 D); a training pair with wrong words teaches the
  wrong alignment; ASR stays out of scope.
- rejected: treating every recorded take as confirmed; leaving unconfirmed takes out of the bake; ASR; a CHECK or a version bump (D39).
- where: `db/connection.py:_ADDED_COLUMNS`; `feed/saved_voices.py:add_example`, `set_transcript`, `bake_snapshot`; `web/voices_api.py`
- changed (2026-11-02, rework, t39): unchanged, except that the Test tab now offers the words of every take of the chosen language (D49's change).
- superseded (2026-11-02, ASR, t42): the user rejected typed and confirmed words ("i can say ANYTHING, OR UPLOAD ANYTHING! … YOU SHOULD USE
  IT") and accepted local speech recognition ("Yes, add it (Recommended)", `## Decisions`). ASR hears the words (D90-D95); no take blocks a
  bake for its words; `transcript_confirmed_at` stays in the schema, unused (D91).

### D72 — the stored voice model: one row per voice, versioned, bound to the revision, owned by the voice outbox (2026-11-02, PLAN, planner)
- what: `voice_models` (one row per voice: version, model key, revision, LoRA JSON, config, dataset hash, takes, seconds, a content-addressed
  safetensors file, metrics) and `voice_model_takes`; a new bake replaces the row with version + 1 and sends the old file to the drain; the store
  keeps a result only when the voice and every snapshot take still exist and are not pending; the cascade and the ref check include the table.
- why: t28 asks for a versioned adapter and for the version films and tests use; one row gives one answer and bounds the disk use (≈ 18 MB each,
  §1.8); the write-ahead file write and the drain (D33) already cover a crash and a delete; a model of a deleted take must not appear after the bake.
- rejected: keeping every version (unbounded files, no asked need); storing the file in the models service (D59); a model row that outlives its voice.
- where: `db/schema.sql:voice_models`, `voice_model_takes`; `feed/saved_voices.py:save_voice_model`, `voice_model_for`, `_cascade_one`, `_remove_one`
- changed (2026-11-02, pair t31): an immutable bake snapshot (`voice_bake_runs`: take ids, content hashes, languages, words, kit ids and
  per-take fingerprints) is written before the bake starts, and a reattach reads it back; `voice_model_takes` stores the fingerprints, so the
  "edited" count is computable; the adapter's `dataset_sha256` must equal the snapshot's hash; `voice_models.bake_key` is unique, so one bake is
  stored at most once (also after a restart between the save and `bake_delete`); `saved_voices.last_model_version` keeps the version rising
  after a retired model. Why: a snapshot rebuilt from the current rows after a restart would not be what was trained; without fingerprints a
  word edit is invisible; a second store would add a version for one bake. Rejected: the snapshot inside the service tag (up to 300 takes of
  1000 characters exceed `TAG_MAX_BYTES`); restarting versions at 1.
- changed (2026-11-02, rework, t39): the model stays when takes are added, edited or deleted, and when a take is deleted or edited DURING its bake
  (D79); `save_voice_model`'s `parent_ok` checks only the voice; a Rebake updates the row in place with version + 1; `voice_model_takes` gains
  `language` and `seconds` and keeps its rows after a take is deleted; `saved_voices.last_model_version` and the index by take are dropped
  (nothing is retired now). `voice_previews.model_version` / `model_sha256` are dropped: tests move to `voice_tests` (D80).

### D73 — what films and tests use: the voice model with the film take, else one take, always shown; stale and retired rules (2026-11-02, PLAN, planner; t28)
- what: `voice_model_for` decides; usable (same key, revision and LoRA shape) → the reference carries the model, and the film take (D47) stays the
  prompt take; stale (takes added, words edited) → still used, and the page counts the changes; another revision or shape → not used, the page says
  "bake again"; a deleted take that the model learned from retires the model in the drain (the recommendation to the gate question); with no usable
  model the voice speaks as in Phase 4, and the film and the page say "one take (no voice model)"; a model that the child refuses fails the film.
- why: t28 leaves the exact rule to the planner ("films keep the last good adapter until a rebake"), and this plan keeps a stale model in use; how
  it sounds waits for the user's check (`ivms777/CLAUDE.md:223-232`); a
  delete in this app removes what was made from the deleted data (`ivms777/CLAUDE.md:151-161` for folders, D15 for voices); the user must hear and
  see which path a film took (D33's "never a silent substitute"); a broken model must not be spoken in part (D60).
- rejected: failing films of a stale model (a dead end for one new take); keeping a model after a take it learned from is deleted (the user decides
  at the gate); silently falling back with no label.
- where: `feed/saved_voices.py:voice_model_for`; `feed/build.py`; `feed/voice_preview.py`; `web/app.py:feed`
- changed (2026-11-02, pair t31): a model with a training take that is pending is not usable at once, before the drain runs ("deleting"); films
  and tests carry the model's identity and its training take ids, and the film's ready transaction and `save_preview` refuse a pending or deleted
  training take; finished films stay (D33). The earlier quality sentence about stale models is removed: keeping a stale model is the
  chosen rule, not a quality claim. Why: between a delete request and the drain, a film or a test could still publish a model made from the
  deleted take.
- changed (2026-11-02, rework, t39): superseded. Films and tests use ONLY the voice model with `use_clip=False`: no film take beside it, no
  one-take fallback, no "one take (no voice model)" path (D76, D77). Without a usable model of the film's language, the film, Prepare, the picker
  and the test refuse with a "bake required" text (K5, D78). A deleted training take no longer retires the model or makes it unusable, and
  publications no longer re-check the trained take ids (t31 item 6 superseded by the user's answer; D79).

### D74 — the bake job in `app`: it holds the voice lock, reattaches after a restart, replaces only after a question (2026-11-02, PLAN, planner)
- what: `run_bake` holds `app.state.voice_calls` through `voice_turn` for the whole bake; films and tests wait with "waiting for the voice model";
  `VoiceBakeJobs` follows the `VoicePreviews` pattern and adopts a running or finished bake of this owner after an `app` restart through
  `bake_current`; a bake request for another voice while one runs answers 409 with a question, and `replace=yes` cancels the running bake and starts
  the new one; a voice delete cancels its bake.
- why: one voice unit (D52), and a film that met the service's 409 would fail; the models service keeps the bake when `app` hot-reloads
  (`ivms777/CLAUDE.md`, "Everything hot-reloads"), so `app` must find it again; `ivms777/CLAUDE.md:271-274` asks to cancel the other and run this
  one, and an hour of training deserves a question first.
- rejected: a bake that does not hold the lock (films would fail on 409); losing a finished bake on a restart; a silent replace; a queue.
- where: `feed/voice_bake.py:run_bake`; `web/voices_api.py:VoiceBakeJobs`
- changed (2026-11-02, pair t31): a publication barrier: the run check and `save_voice_model` run under the `VoiceBakeJobs` job lock, which
  `cancel`, `replace=yes` and a voice delete also take (the pattern of `feed/voice_preview.py:185-200`), so a cancelled, replaced or deleted run
  stores nothing; `done` and `failed` after a Cancel also end the wait; `bake_delete` runs in `finally` on every end; a replacement starts only
  after the old run is torn down; the reattach reads the stored snapshot by `bake_key` and stores at most once. Why: a done that lands after a
  Cancel must not appear as a new model; a restart must not train against one set of takes and publish against another.
- changed (2026-11-02, rework, t39): the bake's tag carries `source: "app"`; `reattach` acts only on such a bake of this owner; a Bake request
  during a bake that `app` did not start asks first and, on yes, cancels it without deleting it (D83).
- changed (2026-11-02, rework, t40): `run_bake` splits the new start from the attach BEFORE any take check: an attach reads only the stored
  snapshot in `voice_bake_runs` and never the current takes, their words or their audio. Cancel, `replace=yes` and a voice delete write
  `cancel_requested_at` under the job lock before they answer; `save_voice_model` refuses a run with it; `reattach` cleans such a run up and
  stores nothing (D88). Why: an attach that ran the take checks failed when takes changed during the bake; an in-memory cancel was lost on an app
  reload, and the reattach then stored a bake the user had cancelled.
- changed (2026-11-02, ASR, t42): the new start runs a transcription stage under the same voice lock before `start_bake_run`, checks for a
  foreign bake before that stage, and writes the run row under `job_lock` with the cancel check (D92, D95). The attach path does not
  change.

### D75 — where the Phase 5 code and tests live (2026-11-02, PLAN, planner)
- what: the trainer in `audio/voxcpm_trainer.py` (new), the service job in `modelsvc/voice_bakes.py` (new), the check in
  `scripts/voice_bake_check.py` (new), the app job in `feed/voice_bake.py` (new); tests in `tests/test_voxcpm_trainer.py`, `tests/test_voice_bakes.py`,
  `tests/test_voice_bake_check.py`, `tests/test_voice_bake.py` (all new) and appends to the existing test files of §5 Phase 5; the lanes of §4
  Phase 5 own disjoint files, and one docs lane writes the Slice A and Slice V docs first.
- why: `audio/` holds the models-service children and is copied into the models image (`Dockerfile.models:46`); `modelsvc/` holds the gateway;
  `feed/` is in the one-model gate and the wheel (D29); every new module needs tests (`ivms777/CLAUDE.md:276-281`).
- rejected: the bake job inside `composite.py` (one file for two concerns); the app job inside `web/voices_api.py` (page code with model-flow logic, D55).
- where: PLAN.md §2.17, §4, §5
- changed (2026-11-02, rework, t39): lane 31 also owns `scripts/voice_bake_check.py` and its test (D82); lane 32b also owns `feed/store.py` and
  `tests/test_feed_store.py`; lane 31b owns `modelsvc/app.py`, `inference/models_client.py` and their tests (D84); `voice_tests` is a new table in
  `db/schema.sql`.
- changed (2026-11-02, rework, t40): a new lane 32d owns the removals in `feed/saved_voices.py` and `tests/test_saved_voices.py`; lane 31b also
  owns `modelsvc/backends/fake.py` and `tests/test_modelsvc_backends.py`.
- changed (2026-11-02, ASR, t42): the Whisper child in `audio/whisper_asr.py` (new) with `tests/test_whisper_asr.py` (new), beside the
  other `audio/` children (copied into the models image, `Dockerfile.models:36-47`); the route in `modelsvc/app.py`, the call in
  `modelsvc/backends/composite.py`, the unit in `modelsvc/slots.py`; the stage in `feed/voice_bake.py`; the words in
  `feed/saved_voices.py` (D102).

### D76 — films and tests send the voice model alone; `/speak` takes a reference with no take when `use_clip` is false (2026-11-02, PLAN, planner; t39)
- what: the reference of a saved voice in a film or a test is `{"language", "adapter": {"blob", "sha256"}, "use_clip": false}`;
  `VoiceReference.audio` and `sha256` become optional, and `_decode_references` needs them only when `use_clip` is true (else 422 "the
  reference of '<token>' needs its audio"); audio sent with `use_clip=false` is still checked and not used; the client sends `audio` and
  `sha256` only when the reference has them; the voice child does not change.
- why: t39 — no prompt take beside the adapter; today the service requires and checks a take's audio even when the child never uses it
  (`modelsvc/app.py:80-92`, `:309-313`; `audio/voxcpm_voice.py:283-285`), so a model-only film would have to choose and send a take, which
  brings back a per-language take rule and its delete cases.
- rejected: sending an unused take (a hidden film take); a separate `adapters` field beside `references` (a second shape for one voice);
  refusing audio with `use_clip=false` (the check script's variant C sends it, `scripts/voice_bake_check.py:257-259`).
- where: `modelsvc/app.py:VoiceReference`, `_decode_references`; `inference/models_client.py:_reference_payload`;
  `feed/saved_voices.py:model_reference`
- changed (2026-11-02, rework, t40): `FakeBackend.speak` follows the contract: no conditioning for a `use_clip=False` token, and `adapters[token]
  = {"state": "loaded", "sha256": <sent>}`; an HTTP test sends a reference with no audio through `ModelsClient` and
  `create_models_app(FakeBackend())`. Where: also `modelsvc/backends/fake.py:speak`.

### D77 — a voice narrates a language only through a usable model that learned it; one resolver (2026-11-02, PLAN, planner; t39)
- what: `film_model(conn, owner_id, voice_id, language) -> FilmModel(model, problem)`, problem `none` | `revision` | `language` | `deleting`;
  `castable_models` (changed, t40: a new name; `castable` keeps its take pairs until lane 32d removes it) keeps the voices with no problem;
  the build, the picker, the Prepare check and the test call it; the model's languages are the
  languages of its training snapshot (`voice_model_takes`); the film and the test resolve again under the voice lock.
- why: one function cannot disagree with itself (D47's reason, kept for models); en↔ru cloning is not published for VoxCPM2 (D19's reason),
  and no run here measured a voice model on a language it learned no take in; the snapshot does not change when a take is deleted, so a
  deleted take never removes a language before the next Rebake; a Rebake that ran while a film waited for the voice lock may have removed the
  old file, so the resolve must happen under the lock.
- rejected: any usable model for every language (cross-language speech never measured); the languages of the CURRENT takes (a new English
  take would open an untrained English path); one resolver per caller; resolving only before the wait.
- where: `feed/saved_voices.py:film_model`, `castable_models`; `feed/build.py:_saved_cast`, `build_presentation`; `web/app.py:_narrator_picker`,
  `_store_narrator`; `feed/voice_preview.py:run_preview`

### D78 — the picker shows every saved voice; one that cannot narrate is disabled with its reason and a link (2026-11-02, PLAN, planner; t39)
- what: `picker["saved"]` lists the voices of `castable_models` (changed, t40); `picker["saved_blocked"]` lists every other non-pending voice as a disabled option
  ("«me» — bake required", …); a stored choice that cannot narrate stays the selected unavailable option with the same label; when the
  selected saved voice needs a model, a line under the picker links to `/voices/<id>?tab=model`; the Prepare 409 and the film failure carry
  the same text (§2.17 K5).
- why: after Slice B no voice narrates until it is baked, so a picker that hides unbaked voices would hide the user's own voice with no
  reason; t39 asks for "bake required" as a clear text with a link to the voice page; D33 and D38 — an explicit choice that cannot speak fails
  clearly and never becomes Automatic.
- rejected: hiding unbaked voices (today's rule for "no example", `web/app.py:944-995`); enabling them and failing only at Prepare; a link
  inside the stored failure text (a film row keeps plain text).
- where: `web/app.py:_narrator_picker`, `_store_narrator`; `web/templates/feed.html`

### D79 — the voice model stays; "changes since" comes from its snapshot; a take changed during a bake does not drop the result (2026-11-02, PLAN, planner; user + t39)
- what: no drain, delete or edit retires a model; `voice_model_takes` keeps one row per trained take with its language, seconds and
  fingerprint, with no FK to `voice_examples`; the page counts added, deleted, edited and unresolved takes against it; Rebake trains version
  + 1 from all current takes and updates the row in place; `save_voice_model` stores a finished bake while the voice exists, also when a take
  was deleted or edited during the bake; `saved_voices.last_model_version` and the index by take leave the plan.
- why: the user — "why do you need to delete model??? why not add a button to rebake model???" (`## Decisions`); t39 keeps the model after
  additions, word edits and deletions and asks for the version, the snapshot and the changes since; dropping a finished bake for an edit made
  during it would break the same rule; with no retirement, the stored row always holds the last version, so no extra counter is needed.
- rejected: retiring the model on a deleted take (the earlier D73 and the old gate recommendation); marking it unusable while a training take
  is pending (t31 item 6); dropping a result whose takes changed during the bake; keeping every version.
- where: `db/schema.sql:voice_models`, `voice_model_takes`; `feed/saved_voices.py:voice_model_for`, `save_voice_model`, `_cascade_one`
- changed (2026-11-02, rework, t40): an attach after an app restart stores the model from the stored snapshot also when every take was deleted and
  drained, or a new unresolved take was added, before the attach: the model learned from the snapshot, and the page then counts the changes since
  it.
- changed (2026-11-02, ASR, t42): the page counts added and deleted takes only; "edited" and "unresolved" go (a heard take's words never
  change, and nothing is unresolved); a take ASR could not use is counted apart as `unusable` and is never "added" (D100).

### D80 — one Test per person, stored in a new `voice_tests` row per voice (2026-11-02, PLAN, planner; user + writer view)
- what: the Test tab tests the person: a language (the model's languages), a text (a fixed sentence, the words of a take of that language, or
  the user's own text ≤ 300), one Generate, through the model only; `voice_tests` keeps the LAST test of each voice with its language, text
  kind, sentence id, take id (`ON DELETE SET NULL`), text, model version, model sha256, model key, revision, wall time, length and a WAV; a
  test uses the version it finds under the voice lock and records it; ▶ Original plays the take whose words the test spoke.
- why: the user asked for "1 test button per person" with a chosen or typed text; films speak only through the model (t39), so a test of a
  take tests nothing a film does; one row per voice bounds the files and is all the comparison needs (D50's reason); a take id with SET NULL
  keeps the test when its take is deleted, as the model stays.
- rejected: reusing `voice_previews` (its key is the take, `db/schema.sql:254-268`); a row per language (the user asked for one test); a
  history of tests; a take picker (t39); dropping a test because a Rebake ended while it waited (a bake and a test never overlap, D74).
- where: `db/schema.sql:voice_tests`; `feed/saved_voices.py:save_test`, `voice_test`; `feed/voice_preview.py`; `web/voices_api.py`

### D81 — the one-take data is kept and ignored; no destructive migration (2026-11-02, PLAN, planner; t39)
- what: `voice_film_examples`, `voice_previews` and `voice_conditionings` keep their CREATE statements and their rows; films, tests and the page
  no longer read or write them (the check script still reads conditionings for A and B, D82); the drain still collects their files when a
  take or a voice is deleted; films made from one take keep their sound until regenerated, and "This film" says "made from one take, before
  voice models".
- why: D39 (no destructive schema change); the rows are the user's data and small; the drain already owns their files, so they leave with
  their take; a per-take test was made from one take, not by a model, so moving it into `voice_tests` would show a result the model never made;
  D33 keeps finished films.
- rejected: DROP TABLE or a row delete at migrate (destructive, with no ask); moving per-take tests into `voice_tests`; deleting the
  conditioning files now (a cache the check script still reads).
- where: `db/schema.sql`; `feed/saved_voices.py:_cascade_one`, `_remove_one`; `feed/store.py:film_narrator`

### D82 — the check script's diagnostic prompt take is the newest take (2026-11-02, PLAN, planner)
- what: `scripts/voice_bake_check.py` takes the prompt take of A and B from `example_for` (the newest take of the language) instead of
  `film_example`; `report.json`'s `prompt_takes` records `{"example_id", "rule": "newest"}`; C, the thresholds and the blind output (t38) do
  not change; `example_for`, `conditionings` and `reference_for` stay in `feed/saved_voices.py` for the script only; lane 31 changes the store
  and the script together.
- why: `film_example` and its table leave films (t39), and the script imports it (`scripts/voice_bake_check.py:32-40`, `:239`); A and B only
  help to diagnose (t39), so the newest take is enough; one lane for both files means the script never imports a removed name, also while the
  user runs §6.5 beside Slice B.
- rejected: keeping `film_example` for the script (a stale pin would still steer a number the user reads); dropping A and B (t39 keeps them
  as diagnostics).
- where: `scripts/voice_bake_check.py:speak_variants`; `feed/saved_voices.py:example_for`, `reference_for`
- changed (2026-11-02, rework, t40): lane 31 only ADDS: it switches the script to `example_for` and keeps `film_example` and every other name that
  `app` imports, because `app` runs with `--reload` and must load after every lane; lane 32d removes `film_example` and the other K0 names after
  32b and 32c are GREEN. The script's test seeds an old pin by SQL, never through `set_film_example`.

### D83 — app bakes carry `source: "app"`; `app` never cleans up a bake it did not start, and stops one only after a question (2026-11-02, PLAN, planner)
- what: `run_bake`'s tag is `{"source": "app", "owner_id", "voice_id", "bake_key"}`; `reattach` adopts, stores, cancels or deletes only a bake
  with that source and this owner; a Bake request while the models service runs another bake (the §6.5 script's, tag `{"check": NAME}`)
  answers 409 with "The models service is baking a voice for the bake check (N%). Stop it and bake «<name>» instead?"; `replace=yes` calls
  `bake_cancel` on it (never `bake_delete`: its starter deletes it) and starts the new job with `wait_for_other=True`.
- why: Slice B now runs while the user may run §6.5 (`## Decisions`); the earlier reattach rule ("no row for the key → cancel it") could end
  the user's check after an app reload; `ivms777/CLAUDE.md:271-274` asks to cancel the other and queue this one, never a dead end, and an hour
  of training deserves a question first (D74); the script handles an end it did not ask for (exit 1, `scripts/voice_bake_check.py:369-370`)
  and deletes its own bake (`:219-224`).
- rejected: a dead-end refusal ("wait, or cancel it elsewhere"); cancelling a foreign bake without a question; deleting it from `app`;
  treating every bake of the service as the app's.
- where: `feed/voice_bake.py:run_bake`; `web/voices_api.py:VoiceBakeJobs`

### D84 — only one Slice B lane edits a models-service file, and it waits until no §6.5 bake runs (2026-11-02, PLAN, planner)
- what: lane 31b (the `/speak` contract of D76) is the only Slice B lane that edits a file under `modelsvc`, `audio`, `models`, `inference`,
  `embedding` or `captioning`; it starts after the docs lane and after the user's §6.5 report (any result); when the other lanes are GREEN and
  no report has come, the main session tells the user that one change reloads the models service and ends a running bake, and waits for the
  user's word that no bake runs (step 33a); the other lanes do not wait for §6.5.
- why: §6.5 needs the uncommitted Slice A code, which exists only in `<base>/ivms777`, so its models service runs from there (§1.9); `make up`
  reloads that service on a change in those folders, and a reload ends a bake (`ivms777/CLAUDE.md:206-216`, §1.8); the user asked not to wait
  for §6.5, so only the one lane that must not run beside it waits.
- rejected: editing those files while §6.5 may run (it can end the user's bake); holding all of Slice B until §6.5 ends (against "do the
  rework NOW"); avoiding the contract change (D76).
- where: PLAN.md §4 Phase 5 steps 31b, 33a
- changed (2026-11-02, rework, t40): lane 31b also owns `modelsvc/backends/fake.py` (under `modelsvc`, so in the reload scope) and
  `tests/test_modelsvc_backends.py`; it waits as before.
- changed (2026-11-02, ASR, t42): lane 31d (the bake's ASR) also edits those folders and waits the same way, after 31b; the 33a question
  names both changes (D102).

### D85 — the test job keeps its names and its polling routes; only the start and audio routes change (2026-11-02, PLAN, planner)
- what: `VoicePreviews`, `run_preview`, `PreviewJob`, `PreviewText`, `PreviewFailed`, `GET /voices/preview`, `POST /voices/preview/cancel`,
  `_voice_preview.html`, `previewSync` and the t27 certificates stay; the start moves to `POST /voices/{id}/preview` and the audio to
  `GET /voices/{id}/preview/audio`; the page says "Test".
- why: the epoch, completion and ownership rules of t25, t27 and t36 are tested code (`web/voices_api.py:157-271`, `web/static/voices.js:147-325`);
  a rename would touch each of them for no change in behaviour; the user sees "Test", never the internal name.
- rejected: renaming the job to "test" everywhere; a second job for the person test beside the take test.
- where: `web/voices_api.py`; `feed/voice_preview.py`; `web/static/voices.js`

### D86 — every film and test checks that the reply spoke through the voice model it sent (2026-11-02, PLAN, planner; t39)
- what: after `client.speak`, every used token must report `adapters[token].state` `loaded` or `kept` with the sent sha256; else the film fails
  with "The models service did not speak «<name>» through its voice model." and the test with the same reason; a film records
  `decisions["voice"]["models"]` and the cast's `narrator_model`.
- why: the user's rule is that films use the model; the reply's `adapters` entry (`audio/voxcpm_voice.py:294-296`) is the only proof of which
  model spoke, and a missing or other entry means a broken contract, which must fail, never pass in silence; "This film" then shows the
  version it used.
- rejected: trusting the reply; checking it only in tests.
- where: `feed/build.py:build_presentation`; `feed/voice_preview.py:run_preview`; `feed/store.py:film_narrator`

### D87 — the page texts: the model state line, "Bake voice model" and "Rebake", no confirm on a take delete (2026-11-02, PLAN, planner; t39)
- what: the header line and the card line of §2.17 K6 come from `voice_model_for` and the bake job; the button reads "Bake voice model" before
  the first model and "Rebake" after; Rebake is offered also with no change; a take delete asks nothing; the Recordings tip names the voice
  model; the changes line reads "<counts> since v<N> — Rebake to include them".
- why: t39 asks for the version, the snapshot and the changes since; the user asked for "a button to rebake"; a take delete no longer changes
  what films use (the model stays), so D48's confirm has no reason left; a Rebake after a failed or doubtful bake must not need a fake change.
- rejected: a confirm on deleting a trained take; hiding Rebake when nothing changed; one "Bake" label for both cases.
- where: `web/templates/_voice_body.html`, `_voice_cards.html`; `web/voices_api.py`

### D88 — a cancel request is stored with the bake run; the store and the cancel meet in the database (2026-11-02, PLAN, planner; t40)
- what: `voice_bake_runs.cancel_requested_at` (nullable); `request_bake_cancel(conn, bake_key)` sets it on an OPEN run and returns False when
  the run row is gone; Cancel, `replace=yes` and a voice delete call it under the job lock before they answer; `save_voice_model` checks the
  run row in its own `BEGIN IMMEDIATE`: a flag, or no run row and no model row with that `bake_key`, → `BakeCancelled` and nothing written;
  the store ends the run row in the same transaction, so a later cancel finds no open run and the reply says the bake ended with v<N>;
  `reattach` cancels (when it runs) and deletes a flagged bake and stores nothing.
- why: t40 — the cancel lived only in memory, so an app reload between a Cancel and the bake's end let `reattach` store a bake the user had
  cancelled or replaced; one row checked inside the store's own transaction orders a cancel and a store also across processes and restarts;
  a missing run row means the run already ended, so storing then would publish a result nobody waits for.
- rejected: a flag only in `VoiceBakeJobs` (lost on a reload); a separate cancel table; deleting the run row on cancel (the reattach would
  then not know the bake was cancelled, and the reattach rule "no row → cancel and delete" would hide the reason); returning `None` for a
  cancel (it would read like "the voice went away").
- where: `db/schema.sql:voice_bake_runs`; `feed/saved_voices.py:request_bake_cancel`, `save_voice_model`, `BakeCancelled`;
  `feed/voice_bake.py:run_bake`; `web/voices_api.py:VoiceBakeJobs`

### D89 — the store API changes by addition: `castable_models` beside `castable`; removals last, behind a grep (2026-11-02, PLAN, planner; t40)
- what: lane 31 adds `castable_models(conn, owner_id, language) -> list[(SavedVoice, VoiceModel)]` and leaves `castable` (take pairs) and every
  other name `app` imports unchanged; 32b and 32c switch to `castable_models` and `film_model`; lane 32d removes the K0 names only after a
  `grep -rn` over `feed web scripts tests` finds no importer outside `feed/saved_voices.py` and `tests/test_saved_voices.py`; the new name stays
  after 32d.
- why: t40 — `app` runs with `--reload`, so a store that changed `castable`'s return type in lane 31 would break the film build and the
  picker at runtime until 32b and 32c land (`feed/build.py:405-416` and `web/app.py:952`, `:1029` unpack take pairs); a rename back to
  `castable` in 32d would need edits in files 32d does not own.
- rejected: changing `castable` in place in lane 31; a rename back to `castable` in lane 32d; removing names in lane 31.
- where: `feed/saved_voices.py:castable_models`, `castable`; PLAN.md §4 steps 31, 32d

### D90 — the bake's ASR model: `openai/whisper-large-v3-turbo` at `41f01f3`, float32 on MPS, mac only, declared in `TRANSCRIBERS` (2026-11-02, PLAN, planner; ASR, t42)
- what: one ASR entry, `whisper-large-v3-turbo` = `openai/whisper-large-v3-turbo` @ `41f01f3fe87f28c78e2fbf8b568835947dd65ed9`, `size_mb`
  1547, `cost_mb` 4000 (`cost_measured=False`), `profiles=("mac",)`, `languages=("en", "ru")`, loaded with `transformers` 5.15.0 in float32
  on MPS; declared in a new `catalog.TRANSCRIBERS` tuple with `transcriber_for(profile)`, outside `CATALOG` and `SLOTS`.
- why: the user's two decisions (commercial models only, 2026-11-01; local ASR, 2026-11-02); the Hub reads MIT, not gated, en and ru, and
  808,878,080 parameters (§1.10); `transformers` is already locked and holds Whisper, so no package and no licence exception is added (D35
  stays); a take is ≤ 30 s, Whisper's one window; turbo has 4 decoder layers against large-v3's 32 at "a minor quality degradation"
  (README `:124-125`) and about half the memory of large-v3 (1.54 B parameters); float32 because no float16 run on MPS was measured here,
  and 3086 MiB fits beside the voice child under the mac budget; outside `CATALOG` because it is no slot, so the settings popup, `DEFAULTS`
  and `tests/test_catalog.py:33-35` stay as they are; mac only, as bakes (D62).
- rejected: `openai/whisper-large-v3` (twice the parameters for a 5-30 s take); `whisper-medium` / `whisper-small` (less memory, but no
  source read here compares their Russian words with turbo's, and turbo's estimate already fits beside the voice child);
  `faster-whisper` / CTranslate2, NeMo models, GigaAM (a new package each); funasr and kaldi-based stacks (excluded, D35); an `asr` slot in `SLOTS` (a user choice nobody asked for, a
  `DEFAULTS` row per profile, a settings row); float16 before a measurement.
- where: `models/catalog.py:TRANSCRIBERS`, `transcriber_for`; `audio/whisper_asr.py`; PLAN.md §1.10, §2.17 B

### D91 — the heard words live in `voice_examples.transcript`, with `transcript_model` and `transcript_error`; the kit text is never words (2026-11-02, PLAN, planner; ASR, t42)
- what: two new nullable columns (`transcript_model` = `<asr key>@<revision>`, `transcript_error` = why no usable words); `transcript`
  holds the heard words; `add_example` never stores the kit text as `transcript` and loses its `transcript` parameter; `kit_id` stays as the
  text shown; a take stored before t42 keeps its shown text until its first bake replaces it with the heard words;
  `transcript_confirmed_at` stays in the schema, unused.
- why: one words field keeps every reader as it is (the snapshot JSON, `_fingerprint`, `snapshot_hash`, the service's `dataset_sha256`, the
  check script's `reference_for`); the shown text stored as words is the bug of t42, so nothing must read it as words; `transcript_model`
  tells heard words from a shown text with no guess; the user's database already holds `transcript_confirmed_at` (§1.10) and D39 allows no
  drop.
- rejected: a separate `asr_text` column (two words columns, every reader must choose, and the wrong one stays under the old name); a data
  migration that clears old transcripts (D39, D81); dropping `transcript_confirmed_at` from the schema (the user's DB keeps it anyway).
- where: `db/schema.sql:voice_examples`; `db/connection.py:_ADDED_COLUMNS`; `feed/saved_voices.py:add_example`, `VoiceExample`,
  `set_heard_words`

### D92 — ASR runs as a stage of the app's bake job, one take per `POST /voice/transcribe`, before the snapshot (2026-11-02, PLAN, planner; ASR, t42)
- what: `run_bake`'s new start hears every `waiting` take through `client.transcribe` (one short call per take, under the voice lock),
  stores the words with `set_heard_words`, then writes the snapshot and starts the service bake as before; the service's bake request and
  its checks do not change; the models service holds no database and no words.
- why: the pair's design asks to store the words and freeze the snapshot before training (`modelsvc/voice_bakes.py:474-477` refuses empty
  words, `:252-256` hashes them, `save_voice_model` checks the hash); words stored per take survive a Cancel, a failure and a restart, so a
  take is heard once; one call per take gives the stage a percent and a Cancel between takes with no new job API; a stage of the same job
  keeps one bake job, one lock and one progress line for the user.
- rejected: ASR inside the service's bake (the app would learn the words only after training, and the snapshot and its hash would not hold
  them); a separate ASR job in the service with its own status and cancel routes (more API for calls of ≤ 30 s of audio); hearing a take
  when it is saved (Whisper loaded during every recording session and beside films, for words nobody reads before a bake).
- where: `feed/voice_bake.py:run_bake`; `modelsvc/app.py:/voice/transcribe`; `inference/models_client.py:transcribe`
- changed (2026-11-02, ASR plan check, t44): the stage repeats until the snapshot read under `job_lock` finds no `waiting` take; that read
  is the cutoff. `job_lock` is released for every transcribe round and never held across HTTP. Why: `add_example` takes no lock
  (`web/voices_api.py:514-548`), so a take can arrive while the stage runs; without the loop it would sit `waiting` outside the snapshot
  with no reason shown, or the snapshot would hold a take with no words. Rejected: taking the voice lock in `add_example` (a recording
  would wait for a whole bake); leaving a late take out with no check.

### D93 — the `asr` unit: registered once, one Whisper child, in the voice gate, BATCH, unloaded by a bake before its lease (2026-11-02, PLAN, planner; ASR, t42)
- what: `ASR_UNIT = "asr"`, registered once in `apply()` like `voice_train`; the target `audio.whisper_asr:WhisperASR`; every transcribe and
  every `POST /models/asr/ensure` enters the voice gate (409 during a bake; a bake's drain waits for them); the call runs at
  `Priority.BATCH`; a bake unloads `asr` right after `voice`, before `lease("voice_train")`; the reply's revision must equal the
  catalog's.
- why: one model process (`ivms777/CLAUDE.md:120-141`); the gate is the one barrier that keeps a load from starting beside the training
  child (t31, t32); the lease alone would not evict a 4000 MB unit that fits the budget, so the bake unloads it by name and Whisper never
  shares memory with training; a registration unloads its unit first, so registering once keeps a slot switch from ending a transcribe;
  D26's identity rule.
- rejected: Whisper inside the training child (it loads VoxCPM2 at start, and a bake would hold both); inside the voice child (another
  model under the voice's identity); a lease of `asr` (nothing must keep it); INTERACTIVE priority (a bake is background work).
- where: `modelsvc/slots.py:ASR_UNIT`, `_register_asr`; `modelsvc/backends/composite.py:transcribe`, `model_ensure`;
  `modelsvc/voice_bakes.py:_run`

### D94 — a take ASR cannot use stays, shows why, is left out and never blocks a bake; a failed call fails the job; a take is heard once (2026-11-02, PLAN, planner; ASR, t42)
- what: heard words with no letter → `unusable` "no speech was heard in it" (changed, t44: see below); over 1000 characters → `unusable` "it holds
  more than 1000 characters of words"; the take keeps its audio, shows "not used by the voice model: <reason>" on its row and under "Not used" on the
  Voice model tab, and is named in the done line; a bake goes on without it; only "no take" and "no heard take" stop a bake; a transcribe
  call that fails ends the job and writes nothing for that take; a take with `transcript_model` set is never heard again.
- why: the pair's design (t42): "the take stays in the voice and shows a failure; it is never dropped silently"; the user: "YOU SHOULD USE
  IT", so a music upload must not hold the voice back; the service refuses text outside 1-1000 characters (§1.10); a call error is not a
  fact about the take; one result per take keeps the model's fingerprints exact.
- rejected: blocking the bake on such a take (a dead end for one upload); leaving it out with no word; training it with the kit text (t42's
  bug); hearing it again at every bake (the same model gives the same result; the user can delete it); cutting long words to 1000
  characters (a wrong pairing of audio and words).
- where: `feed/saved_voices.py:heard_words`, `set_heard_words`, `bake_snapshot`; `feed/voice_bake.py:run_bake`; `web/templates/_voice_body.html`
- changed (2026-11-02, ASR plan check, t44): words with a letter OR a digit are usable ("1 2 3 4 5"); blank or symbol-only words get
  `NO_WORDS` "the speech recognition found no words in it", because the model reports no "no speech" result and the page must not
  claim one; `TOO_LONG` keeps its own reason; `not_used_text` names each left-out take with its own reason in the blocked, done, "Not
  used" and Test texts; the state is `unusable` (was `unheard`). A take deleted during the stage is skipped: `take_to_hear` is read
  before each audio read, and a failed read is checked against the row again (deleted → skip; still there → the job fails with the
  read error); `set_heard_words` keeps its guard. Kept: a take is heard once; nothing retries a take. Rejected: "no speech" texts;
  failing the whole job for a take the user deleted on purpose.

### D95 — the transcription stage: its progress, Cancel, restart and the hand-off to the run row (2026-11-02, PLAN, planner; ASR, t42)
- what: "transcribing the takes · take n of m" at 0.01-0.15 of the job bar, then the service's stages at `0.16 + 0.81 * fraction`; Cancel
  is checked before each call, and the call in flight ends first; a foreign bake is checked before the stage; the snapshot and
  `start_bake_run` run under `job_lock` with the cancel check, and the job records `bake_key` before it leaves the lock; a restart during
  the stage leaves no run row and no service bake; the words already stored stay.
- why: `ivms777/CLAUDE.md:264-274` (stage, percent, Cancel); a 30 s take is a short call, so Cancel between takes is the honest promise;
  the t40 rule (only SQLite writes under `job_lock`) and D88 need the Cancel and the run row ordered, or a Cancel could fall between the
  check and the row and a restart would store a cancelled bake; D74's voice lock covers the stage, so films and tests wait.
- rejected: interrupting a running transcribe (one `TorchWorker` call holds its lock to the end, §1.8); a run row before the stage (a
  restart would reattach to a bake that never started); dropping the heard words on Cancel (they are facts of the takes).
- where: `feed/voice_bake.py:run_bake`; `web/voices_api.py:VoiceBakeJobs`
- changed (2026-11-02, ASR plan check, t44): the hand-off reads the snapshot under `job_lock`; `waiting` takes (arrived during the stage)
  send the job back to the stage with the lock released; only a read with no `waiting` take writes the run row in the same hold of the
  lock. The bar keeps its highest value when a round raises m. Kept: Cancel acts between calls.

### D96 — lane 31 builder decision 1: `read_bake_run` returns the `BakeRun` NamedTuple — kept (2026-11-02, PLAN, planner; ASR, t42)
- what: `BakeRun(voice_id, takes, dataset_hash, cancel_requested_at)`, as lane 31 wrote it (`feed/saved_voices.py:1254-1258`), replaces the
  plain tuple of §2.17 K3.
- why: it carries the D88 flag that `reattach` needs, by name; the plain tuple of K3 had no place for it.
- rejected: the three-element tuple of K3 (the flag would need a second read).
- where: `feed/saved_voices.py:BakeRun`, `read_bake_run`

### D97 — lane 31 builder decision 2: the 80-140 character kit test covers only the 44 added texts — kept (2026-11-02, PLAN, planner; ASR, t42)
- what: the length test runs over the 22 added texts per language; the four reading texts of §2.15 I keep their own lengths.
- why: those four keep their ids, order and words (§2.17 K1), so a length rule must not force a change; the kit is now a suggestion, so a
  text's length only guides the user.
- rejected: changing the four reading texts to fit 80-140 characters.
- where: `tests/test_saved_voices.py`

### D98 — lane 31 builder decision 3: a blank posted transcript counts as none — dropped (2026-11-02, PLAN, planner; ASR, t42)
- what: `add_example` takes no `transcript` any more, so the rule has nothing to act on.
- why: the user types no words (t42).
- rejected: keeping the parameter for a caller that does not exist.
- where: `feed/saved_voices.py:add_example`

### D99 — lane 31 builder decision 4: a recording with no `prompt_id` stores `kit_id` NULL — changed (2026-11-02, PLAN, planner; ASR, t42)
- what: lane 31 stored the language's first kit text as `kit_id`; now a recording posted with no `prompt_id` stores NULL and is listed
  under "Other takes".
- why: `kit_id` records the text the user was shown; with no `prompt_id` nothing says a text was shown, and a guessed text would place the
  take on a kit row it never belonged to; the recorder always posts its text's id (`web/voices_api.py:520`, `:544`).
- rejected: the first text (a guess shown as a fact).
- where: `feed/saved_voices.py:add_example`

### D100 — lane 31 builder decision 5: the change counts — changed to `ModelChanges(added, deleted, unusable)` (2026-11-02, PLAN, planner; ASR, t42)
- what: lane 31 counted a new unresolved take in both `added` and `unresolved`; now a `waiting` take counts in `added` only, an `unusable`
  take in `unusable` only, and `edited` and `unresolved` go.
- why: "N takes added — Rebake to include them" must be true: a Rebake hears and trains a waiting take but never an unusable one; nothing is
  unresolved; a heard take's words never change, so `edited` would always be 0.
- rejected: counting an unusable take as added (the header would ask for a Rebake that changes nothing); keeping `edited` as a dead field.
- where: `feed/saved_voices.py:ModelChanges`, `voice_model_for`

### D101 — the page after ASR: no words UI, "Other takes", the recorder line, "Not used", "Your recordings", "Your words in take #N" (2026-11-02, PLAN, planner; ASR, t42)
- what: the texts of §2.17 K6 "Changed (ASR, t42)": no words box, words state, Confirm, Edit or "What you said" preview; an unusable row
  reads "not used by the voice model: <reason>"; "Your uploads" → "Other takes"; the recorder adds "Read it, or say anything else: the
  bake hears what you say."; the Voice model tab lists "Not used" takes and its two disabled texts; the meter reads "Your recordings";
  the Test option reads "Your words in take #<id>" for heard takes only.
- why: the user's two decisions (no typing, no confirming, any speech); the user's five takes are recordings stored before the kit with no
  `kit_id` (§1.10), so "Your uploads" would mislabel them; the failure must show (D94); the Test option compares the generated speech with
  what the user really said.
- rejected: showing the heard words on each row (a words view the user did not ask for, and an invitation to edit); keeping "(checked)" /
  "(not checked)"; backfilling `kit_id` from the old text (a data migration, D39).
- where: `web/templates/_voice_body.html`, `voice.html`; `web/static/voices.js`; `web/voices_api.py`; `feed/voice_preview.py:preview_text`

### D102 — the lanes of the ASR amendment: docs first, a store rework, one models-service lane in the D84 window (2026-11-02, PLAN, planner; ASR, t42)
- what: 30b (docs) → 31c (store, check-script labels) → 32a, 32b's `take` branch, 32c → 32d; 31d (catalog, slots, Whisper child, route,
  bake unload, client, fetch script) after 31b, in the same §6.5 / 33a window; 32c's bake parts after 31d; steps 33 and 34 after all of
  them; the app-side code imports no new catalog name.
- why: docs first (`ivms777/CLAUDE.md:3-36`); every file in `modelsvc`, `audio`, `models` and `inference` reloads the models service and
  ends a bake (D84), and `models/catalog.py` and `inference/models_client.py` are there; `app` must keep loading under `--reload` while 31d
  waits, so the job takes the ASR identity from the reply; before 31d the client has no `transcribe`, so a Bake could only fail, and the
  bake parts wait.
- rejected: one lane for the store and the service (it would hold the app work until §6.5); a catalog edit outside the window (it ends a
  §6.5 bake); `getattr` checks for a missing client method.
- where: PLAN.md §4 Phase 5

### D103 — the ASR cost is an estimate; a load logs the net free-memory change of its request; §6.7 reads it (2026-11-02, PLAN, planner; ASR, t42; changed t44)
- what: `cost_mb` 4000 with `cost_measured=False`; when a transcribe call loads the child, `CompositeBackend.transcribe` logs `asr
  <key>@<rev7> loaded: free_mb <before> → <after> (drop <N> MB)` from `governor.state().free_mb`; §6.7 step 17 asks the user for N; a
  pinned figure goes through `IVMS777_MODEL_COST_MB {"asr": N}` or a later catalog change.
- why: `ivms777/CLAUDE.md:223-232` and the catalog's own rule (`models/catalog.py:16-20`: a cost is measured as a drop in free memory
  across a load); the agent cannot load the model; one log line costs no API.
- rejected: a figure stated as measured; a check script only for ASR (the bake of §6.7 already loads it); storing the figure in the voice
  model's metrics (it belongs to the model service, not to a voice).
- where: `modelsvc/backends/composite.py:transcribe`; `models/catalog.py:TRANSCRIBERS`; PLAN.md §6.7 step 17
- changed (2026-11-02, ASR plan check, t44): `governor.acquire` evicts other units when it needs room (`modelsvc/governor.py:58-81`), so the
  `free_mb` change across the call includes those evictions. The line is `asr <key>@<rev7> loaded by this request: free_mb <before> →
  <after>, net change <N> MB; evicted: <units, or none>` (the evicted units = `governor.state().resident` before minus after). It is
  the net change of that request, never the model's cost, and never uses "cost", "drop" or "measured"; `cost_measured=False`
  stays; §6.7 step 17 and §7 item 5 drop "measured". A pinned cost still needs a load with no eviction, read the way §8.1 requires.
  Rejected: calling the change a cost (an eviction would make it look larger or smaller than Whisper).

### D104 — the check script names where each take's words come from; §6.5 may use the heard words (2026-11-02, PLAN, planner; ASR, t42)
- what: `--list` labels "heard by <model>@<rev7>", "shown text, not heard", "no usable words: <reason>", "NO WORDS"; the run rules
  (`--confirm-texts`, `--text`) and the read-only rule do not change; the script never calls `/voice/transcribe`; §6.5 step 3 says the
  user can run the check after a first app bake to use the heard words.
- why: "reading text, not checked" (`scripts/voice_bake_check.py:79`, `:132-135`) would label heard words wrongly; the script reads the
  database read-only, so it cannot store words, and a transcribe call from it would need lane 31d, which waits for §6.5 itself.
- rejected: ASR inside the script (a circular wait with D84); dropping `--text` (the user may run §6.5 before any app bake).
- where: `scripts/voice_bake_check.py`; PLAN.md §2.17 I, §6.5 step 3

### D105 — Automatic's saved candidates: `castable_models` of the language, only where voice copy speaks it here (2026-11-03, PLAN, planner; t50)
- what: with Automatic, when `profile` and `voices` are set and `cloning_entry_for(profile, language)` is not None, the candidates are
  `castable_models(conn, owner_id, language)` minus every voice whose name `resolve(language, name)` accepts; else there are none, and
  Automatic is presets only, as before. `build_presentation` gains `profile: str | None = None`; `web/app.py` passes `ctx.settings.profile`.
- why: the user's answer names the baked saved voices of the film's language; `castable_models` is the one resolver of D77, and the
  picker uses the same two calls (`web/app.py:965-966`), so Automatic can pick exactly the enabled saved options; the build gets no profile
  today (`web/app.py:1229-1233`); a saved name equal to a preset name would make the story's answer ambiguous.
- rejected: every saved voice (an unbaked one cannot narrate, and there is no one-take fallback, D77); a call to the models service (the
  picker decides offline); reading the profile from `config` inside `feed/build.py` (a hidden global); a preset-first rule for equal names
  (the saved voice would be listed but never reachable).
- where: `feed/build.py:build_presentation`; `web/app.py` (the build call)

### D106 — the story pass sees a saved voice's name and that its sound is not described; `feed/story.py` does not change (2026-11-03, PLAN, planner; t50)
- what: `feed/voice.py:automatic_for_prompt(language, saved)` gives the preset lines, then the SAVED VOICES line of §2.18 B, then
  `saved_for_prompt(saved)`; with no saved voice it is `catalogue_for_prompt(language)`. The build passes it as `speakers`.
- why: `ivms777/CLAUDE.md` "A model's own metadata is DECLARED": a saved voice declares only its name (`SavedCast` is a token and a name,
  `feed/voice.py:54-59`); `_assemble` prints `speakers` verbatim (`feed/story.py:1230-1240`), so the line reaches the model with no edit
  of `feed/story.py` while the t50 builder holds it; the retry and the score only need `speakers` not empty.
- rejected: a gender, register or sound note guessed from the name or the takes; a description field the user writes (not asked; a
  column and a UI); the token `saved:<id>` in the table (a name tells the model who it is, and `resolve_saved` takes the name); a prompt
  change in `feed/story.py`.
- where: `feed/voice.py:automatic_for_prompt`; `feed/build.py:build_presentation`

### D107 — one engine per Automatic film: the narrator's kind decides every passage voice (2026-11-03, PLAN, planner; t50)
- what: the narrator resolves as a preset (`resolve`), else as a candidate (`resolve_saved`), else by `pick_speaker`. A saved narrator →
  `_passage_voices(…, saved=saved_cast)` resolves every passage name among the candidates only; a preset narrator → among the presets only.
  A name of the other kind is dropped, so that passage reads in the narrator's voice. A saved pick makes the film a saved-voice film:
  `saved_film` is set after this step.
- why: D17; a `/speak` with `references` goes whole to the cloning entry (`modelsvc/backends/composite.py:123-124`, `:139-144`), so a
  preset cannot speak in it; the code for saved passage voices is in place (`feed/build.py:403-430`); the bracket gate stays positional.
- rejected: a saved narrator with preset passages (two engines in one film); no passage voices in a saved Automatic film (a bracket would
  do nothing in the one mode where brackets cast); trusting the table's kind rule with no check in the code.
- where: `feed/build.py:build_presentation`, `_passage_voices`

### D108 — the mood fallback stays presets only; `narrator_chosen_by` keeps "story" and "mood"; the film labels do not change (2026-11-03, PLAN, planner; t50)
- what: `pick_speaker` does not change. A saved narrator that the story picked has `narrator_chosen_by == "story"` and `narrator_label`
  "<name> (saved voice)"; "This film" reads "narrated by <name> (saved voice) — picked automatically by the story · voice model v<N>"; the
  card reads "<name> (auto)". `feed/store.py` and the "This film" template do not change.
- why: `pick_speaker` chooses by the declared `register` (`feed/voice.py:137-166`), and a saved voice declares none; "story" says who
  chose, and the label and the model version say what; `film_narrator` (`feed/store.py:120-139`) and `feed.html:107-112` already print both.
- rejected: a mood pick among saved voices (a person's voice picked by a seed, for no reason); a new value such as "story-saved" (a reader
  rule and an old-row rule for no new fact).
- where: `feed/voice.py:pick_speaker`; `feed/store.py:film_narrator`

### D109 — an Automatic saved pick follows every saved-film rule and never falls back to a preset (2026-11-03, PLAN, planner; t50)
- what: the picked voice is resolved again under the voice lock, like a chosen one. When it is deleted or being deleted, or film_model
  now reports no usable model for the film's language, the film fails with the existing `_cannot_narrate` texts or "The saved voice
  «<name>» was deleted while this film was made."; a failed speak fails the film. No second speak call runs. A usable Rebake completed
  before the voice turn supplies the new version. A Rebake after synthesis does not invalidate the film; record the version actually
  spoken. No new model-change failure (`feed/build.py:407-427`; `docs/design.md:2149-2151`, `:2169-2170`; §2.17 error table "a Rebake
  is stored while a film is made").
- why: D17 and §11.1 ("never a second call with a preset"); the window is a race, because the voice was castable at the story pass;
  ↻ Regenerate makes a new pick; one code path serves every saved-voice film.
- rejected: a fallback to the mood preset inside the voice turn (a second engine load and a second code path); new failure texts only for
  Automatic.
- where: `feed/build.py:build_presentation`

### D110 — the picker texts of Automatic (2026-11-03, PLAN, planner; t50)
- what: the option "Automatic — the app picks a voice for this memory"; the always-hint by `picker.voice_copy` (the two texts of §2.18 C);
  the Automatic bracket line "A [voice] note can request another voice for one passage: another preset when a preset narrates, another
  saved voice when a saved voice narrates."; the chosen-narrator line stays. This replaces D40's t18 option text and the hint "Saved
  voices are never picked automatically.".
- why: the old texts (`web/templates/feed.html:26`, `:58`, `:65`) would be false; the hint branches on a flag the page already has
  (`web/app.py:983`), so it stays true where voice copy does not speak the language.
- rejected: a hint that lists this memory's baked voices (one more query and test for one sentence); a second "auto" option (t50: it
  duplicates Automatic).
- where: `web/templates/feed.html:narrator_picker`; `docs/ui.md`; `docs/design.md` §13

### D111 — the Phase 6 lanes: docs, then `feed/`, then `web/`, each after the lanes that hold its files (2026-11-03, PLAN, planner; t50)
- what: 35 (docs) → 36a (`feed/voice.py`, `feed/build.py`, their tests) → 36b (the build call in `web/app.py`, `feed.html`,
  `tests/test_web_feed.py`) → 37 (main) → 38 (user, §6.8, with step 34). 35 waits for the t50 builder and the t49 lane; 36a for 32b,
  t50 and the bracket-test fixer; 36b for 32c without its bake parts and t50's picker texts.
- why: docs first (`ivms777/CLAUDE.md:3-36`); one owner per file at a time (t49 and t50 edit both docs, t50 and 32b edit
  `feed/build.py:356-394`, 32c holds `web/app.py`, `feed.html` and `tests/test_web_feed.py`); `app` reloads after every lane, so the
  keyword exists (36a) before a caller passes it (36b); no models-service file, so no D84 wait; §6.8 needs a saved-voice film to speak,
  so it needs lane 31b in the models service and a bake (§6.7).
- rejected: one lane for `feed/` and `web/` (it would hold the build change until 32c); a start before t50 (both rewrite the same lines
  and paragraphs).
- where: PLAN.md §4 Phase 6

### D112 — every new film keeps its passages in telling order, with no times when the segments do not pair, and a random token (2026-11-03, PLAN, planner; t62; changed t63)
- what: the build writes `timeline.passages` (`n`, `text`, `start`, `end`, `photo`, `photo_id`, `voice_asked`, `voice_used`) and
  `timeline.script` (`token` = `secrets.token_hex(16)`, `narrator_asked`, `narrator_used`, `cards` as `{"n", "text"}` (D125),
  `score_plan`, `music_prompt`) through the pure `narration.timeline_script`. A photograph is named by its index AND its `photo_id`.
  When the segments do not pair one to one, every time is null and the texts stay. A failed row carries neither key. `Edit` and
  `director.py` do not change for this (**changed, t65**: `Edit` and `as_dict` stay as they are; lane 39b changes `director.py` under
  §2.19 I).
- why: segment `i` is chunk `i` (`feed/story.py:217-220`, `feed/build.py:463-473`), and the shot list is no source of passages (it was
  re-sorted, `feed/director.py:257-290`, until §2.19 I); "pair one to one" is the rule `with_narration_timing` already uses (`:306-308`); `decisions["story"]`
  holds no cards and no texts (`feed/build.py:340-348`); the row is upserted in place (`feed/store.py:38-62`), so only a token per saved
  film tells which film a page showed; a random token never repeats, even when two films have the same audio and the same key.
- rejected: a passage field on `Shot` (it puts Phase 7 data into `Edit`; before §2.19 I it followed memory order); the keys in `decisions` (a failed row overwrites them,
  `feed/build.py:853-878`); no passages when the times are missing (a new film would read as an older one); the narration audio key as
  the token.
- where: `feed/narration.py:timeline_script`; `feed/build.py:build_presentation`

### D113 — one draft row per memory and language in `narration_drafts`, named by its own `draft_token`, changed only under a write lock (2026-11-03, PLAN, planner; t62; changed t63)
- what: the table of §2.19 B: `(group_id, language)` key, `ON DELETE CASCADE`, `draft_token` (32 hex, new at every seeding, kept by a
  rebase), `source_token` (the film whose times the rows carry), `revision` from 1, `script_json`, `passages_json`. No schema version
  bump (D39). Every change runs read, check and write under `BEGIN IMMEDIATE`. The store and the pure helpers live in a new
  `feed/narration.py`.
- why: t62 point 4 and t63 point 2 (an Undo and a new seed of the same film gave the same `(source_token, revision 1)`); `migrate`
  creates a missing table on every start (`db/connection.py:100-104`); request threads each own a connection and write at once
  (`db/connection.py:17-24`), so a read-check-write needs the lock that the ready save takes (`feed/build.py:608`); `feed/notes.py` holds
  the input of the story pass (`:8-10`) and `feed/store.py` derived films, and a draft is neither.
- rejected: a column on `memory_presentations` (a failed build upserts that row, `feed/build.py:853-878`); a row per passage (t62:
  seeded with ALL passages, one revision); the code in `feed/notes.py` or `feed/store.py`; the source token as the draft's name.
- where: `db/schema.sql`; `feed/narration.py`

### D114 — four JSON routes, and one stale rule on `(source, draft, revision)` for every "the rows you saw are gone" (2026-11-03, PLAN, planner; t62; changed t63)
- what: `GET /feed/{key}/narration`, `POST /feed/{key}/narration/{n:int}` (an empty text removes), `POST …/{n:int}/delete`,
  `POST /feed/{key}/narration/discard` ("Write a new story"), each POST with `lang`, `source`, `draft` and `revision`, each answering
  the whole payload. The film's own rows have revision 0 and post `draft` "". A change is accepted only when the posted triple equals
  the current rows'; else 409 `stale`. The GET serves "Load the current narration" after a 409 (D123). The texts and codes are those of
  §2.19 D.
- why: the note routes answer JSON so the film keeps its place (`web/app.py:1075-1091`); one comparison covers another window, a new
  film, a failed build that replaced the film, a rebase, "Write a new story", a new seed and a retelling; the whole payload lets the
  panel replace its state in one step.
- rejected: an id per row in a table of its own; a 404 for "no film" (to the page it is the same stale case); a redirect answer like
  `/note/{id}/delete` (`web/app.py:1114-1118`); a silent last-write-wins.
- where: `web/app.py`; `feed/narration.py:save_passage`, `remove_passage`, `discard_draft`

### D115 — what a passage may hold; 10 000 characters is a provisional app limit (2026-11-03, PLAN, planner; t62; changed t63)
- what: whitespace is joined to single spaces; no limit per row and no cut; a script over 10 000 characters in all is refused with
  "The narration is longer than this app allows: <N> characters, at most 10000."; a text with `[` or `]` is refused with "… Put a
  [bracket] instruction in a note."; an unchanged text writes nothing; an emptied passage is removed and the last one is refused. There
  is no "add a passage". Photographs and passage voices are not editable here; cards follow their passages (D125).
- why: t62 forbids the 600 cut (`feed/notes.py:22`, `:49`). The limit is a provisional app policy, not a measured one: the score stops at
  180 s whatever the script (`feed/score.py:72`, `:110`), so a long script costs narration time (about 14 characters a second,
  `feed/build.py:104`, `:470`), a longer `/speak` call and a film whose voice outlasts its music; 10 000 characters is about 12 minutes of
  speech, far above the story pass's 400-word ceiling (`feed/story.py:41`). Only `write_story` strips brackets
  (`feed/story.py:271-278`), so the voice would read a bracket in a draft aloud. The user asked to edit the words.
- rejected: stripping brackets in silence (it changes the user's words); a cap per row; letting the voice read a bracket; a control to
  add a passage (a passage carries a photograph and a voice; not asked); "too long to speak" (the limit is the app's, t63 point 7).
- where: `feed/narration.py:save_passage`, `MAX_SCRIPT_CHARS`

### D116 — the replay: the draft becomes the Story; photographs by `photo_id`; the score plan follows its passages (2026-11-03, PLAN, planner; t62; changed t63)
- **Changed 2026-11-04 (pair t74):** a draft build now runs `story.match_photographs` after `draft_story`, so a wrong stored photograph is corrected. Words, order, voices, narrator and score stay, `draft_story` stays model-free, no vote keeps the stored photograph, and the rebase stores the final photographs. `context_tokens` is the real value and `votes` has one entry per passage. In `feed/narration.py:_write`, a row whose text changed loses its card, at the seed and at each later change. No blanket card clear and no migration (design §11.1).
- what: `draft_story` builds `script`, `narration`, `voices`, `narrator`, `cards` (D125), `music_prompt` and `score` from the draft
  alone (§2.19 C.2). A photograph no longer in the memory takes the previous passage's; a score entry moves to the first remaining
  passage at or after it. `decisions["story"]` adds `"source": "draft"` and `"draft_revision"`, and `context_tokens` is None. The stage
  names do not change.
- why: t62 point 5 skips `write_story` whole (`feed/story.py:259-284`); `_score` counts 0-based passages (`feed/story.py:1000-1020`);
  `_script` already falls back to the previous photograph (`:925-926`); within a draft's life a photo can only leave the memory, because
  a retelling and a rebuild remove the draft (D118).
- rejected: stage 2 (`_dress`) over the draft (it picks photographs and cards again); failing a film when a photograph is gone; dropping
  the whole score plan when a passage is removed; a new stage label (the step takes milliseconds).
- where: `feed/narration.py:draft_story`; `feed/build.py:build_presentation`

### D117 — the ready save rebases the consumed draft, by `draft_token` and revision; a preset film with a draft takes that transaction (2026-11-03, PLAN, planner; t62; changed t63)
- what: no draft and no saved voice → `save_ready()` as today. Otherwise one `BEGIN IMMEDIATE`: the deleted-voice check (saved films) →
  `save_ready()` → `rebase_consumed` (`UPDATE … SET passages_json, source_token, script_json, revision = revision + 1 WHERE group_id,
  language, draft_token, revision`) → `COMMIT`. The rebased draft keeps its `draft_token` and its words and takes the new film's
  passages, times, token and script. A newer revision or another draft is never touched. Nothing happens at the start of a build, on a
  failure or on a Cancel.
- why: t63 point 1 supersedes t62 point 5: a deleted draft let the next plain ↻ Regenerate rewrite the user's words; today only a
  saved-voice film holds a transaction (`feed/build.py:602-622`); `draft_token` AND the revision make sure that an older build never
  writes into a newer edit or into a draft seeded again after "Write a new story" (t63 point 2).
- rejected: deleting the draft (t62's rule; the words would be lost); rebasing after the commit (a crash between leaves the draft with
  the old film's times); the revision alone, or `(source_token, revision)`.
- where: `feed/build.py:build_presentation`; `feed/narration.py:rebase_consumed`

### D118 — a retelling deletes the drafts and the card says so; "Write a new story" is the one reset on `/feed` (2026-11-03, PLAN, planner; t62; changed t63)
- what: `rewrite_memory` deletes the memory's drafts in every language beside its presentations; the card's retell text adds "Deletes
  your edited narration in: <Language>, …." when a draft exists; `replace_memories` removes them by the cascade; a failed retelling
  keeps them. "Write a new story" (after the save barrier, with the confirm "Stop using your edited narration? The next Prepare or
  Regenerate writes a new story.") calls the discard route; after it the panel shows the ready film's own rows, or none when a failed
  build replaced the film. A narrator change, a note, the other language, a saved-voice delete or bake and a library change leave a
  draft as it is.
- why: a retelling rewrites the title, the description and the photo order (`albums/memory_store.py:82-104`), and the draft's
  photographs, cards and score plan were made for the old telling; the card already says what a retelling deletes (`web/app.py:699-703`,
  `docs/ui.md:191-194`); with a draft, every ↻ Regenerate skips the story pass (D120), so the user needs a way back to a new story; after
  a failed build `_fail` has replaced the film (`feed/build.py:853-878`), so "the film's own words come back" would be false there (t63).
- rejected: keeping drafts across a retelling (a Prepare would speak the old telling over the new order); deleting without saying so;
  resetting a draft on a narrator change or a note; the name "Undo my edits".
- where: `albums/memory_store.py:rewrite_memory`; `web/app.py:_film_block`; `feed/narration.py:discard_draft`

### D119 — the panel: one keyed list, notes stay the only bubbles, one save at a time, the texts in the template, `narration-data` (2026-11-03, PLAN, planner; t62; changed t63)
- what: §2.19 E: passages and notes in one list keyed by passage `n` and note id, sorted by time, a passage first on a tie; the tag
  "narration"; the count of all rows; `window.feedNotes.all()` stays the notes; stamps seek only with a player and `seekable`; one passage
  save at a time with the latest `source`, `draft` and revision; a re-render never replaces a focused or unsaved row; Regenerate is lit
  at load only for a `changed` draft; the state lines live in `feed.html`, one per state, and the script shows the matching one; the
  JSON block is `#narration-data`; the fold rules do not change.
- why: t62 point 3 (the same list); the player floats every entry of `all()` (`web/static/feed-gl.js:1370`, `:1392`), and the
  narration is already heard; overlapping saves would refuse the user's own second save; `render` rebuilds every field today
  (`web/static/feed-notes.js:119`), which would drop the text and the caret of the row being typed in (t63 point 4); `<audio
  id="narration">` exists (`web/templates/feed.html:164`, `web/static/feed-gl.js:26`, t63 point 3); the tests read the page HTML
  (`tests/test_web_feed.py`); the pair kept the fold (t62 VIEW, t63), so the folded bar's count must show that the rows are there.
- rejected: passages as bubbles; a separate section or panel; unfolding the sheet when rows exist (it overrides the remembered choice,
  `web/static/feed-notes.js:95-115`); a full re-render on every reply; `id="narration"`.
- where: `web/static/feed-notes.js`; `web/templates/feed.html`; `web/static/feed.css`

### D120 — the edited words stay for every later ↻ Regenerate; "Write a new story" ends them; clean and changed drafts read differently (2026-11-03, PLAN, planner; t63; rewritten)
- what: a film made from a draft rebases it (D117), so the next ↻ Regenerate speaks the same words again, with any narrator. A `changed`
  draft reads "Your edited narration. ↻ Regenerate speaks it as written." and lights ↻ Regenerate at load; a draft that is not `changed`
  reads "This film speaks your edited narration. ↻ Regenerate speaks it again as written." and does not light it. Both carry the t63
  disclosure line and "Write a new story". The `film` line says that a ↻ Regenerate with no change writes a new story. The old
  `from_edits` line ("change a passage first") is gone.
- why: the user asked to change the words "and rebuild of course later" (`## Decisions`); t63 point 1 settled the rebase; a lit
  ↻ Regenerate means "the film does not say this yet" (`web/static/feed-notes.js:47-52`), which is false for a rebased, unchanged draft.
- rejected: deleting the draft when its film is saved, with a warning (the plan before t63); a dummy edit to keep the words; lighting
  ↻ Regenerate for every draft.
- where: `web/templates/feed.html`; `web/static/feed-notes.js`; `feed/narration.py:narration_rows` (`changed`)

### D121 — the Phase 7 lanes: docs, then the order fix, then the store, then the build beside the web (2026-11-03, PLAN, planner; t62; changed t63, t64, t65)
- what: 39 (docs, with the §2.19 I.7 text) → 39b (the order fix: `feed/story.py`, `feed/director.py` with `with_arrangement`'s
  `keep_order`, `feed/build.py`'s decisions key and `keep_order` argument, `tests/test_feed_story.py`, `tests/test_feed_director.py`,
  `tests/test_feed_build.py`, and the one card line `:64` of `tests/test_web_feed.py`) → 40a (`db/schema.sql`, `feed/narration.py`,
  `albums/memory_store.py`, `tests/test_feed_narration.py`, `tests/test_albums.py`) → 40b (`feed/build.py`, `tests/test_feed_build.py`)
  beside 40c (`web/app.py`, `feed.html`, `feed-notes.js`, `feed.css`, `tests/test_web_feed.py`) → 41 (main) → 42 (user, §6.9, one
  message with steps 34 and 38). 39 waits for the t60 and t61 fixers. **Changed (t65):** 39b also waits for any holder of
  `tests/test_web_feed.py` (row 36b) to be GREEN; 40a also waits for any holder of `db/schema.sql`, `albums/memory_store.py` or
  `tests/test_albums.py` (rows 31, 31c) to be GREEN.
- why: docs first (`ivms777/CLAUDE.md:3-36`); t64 puts the order fix in a lane of its own after lane 39 and before lanes 40a and 40b;
  39b and 40b both edit `feed/build.py` and `tests/test_feed_build.py`, and 40a builds on 39b's `Story` fields and card shape; the
  web fake's `{"shot": 1, …}` card (`tests/test_web_feed.py:64`) would be dropped by D130 and turn `:408-413` RED, so 39b changes that
  line and runs that test (t65); one owner per file at a time, so every lane waits for the earlier holders of its files (t65); 40b and
  40c share no file, and 40c's tests write presentations with passages themselves; no models-service file, so no D84 wait; the t60
  and t61 fixers hold `docs/ui.md` and `docs/design.md` (`<base>/.pair/decisions.md`, t60 DOCS and t61); `app` reloads after each
  lane, and with 40a alone every film reads as legacy.
- rejected: one lane for the build and the web (the web would wait for the build); the order fix inside lane 40b (it would hold the
  fix until the store lands); a start before t60 and t61 end.
- where: PLAN.md §4 Phase 7

### D122 — `changed` is computed, never stored (2026-11-03, PLAN, planner; t63)
- what: `narration_rows` sets `changed` = NOT (the draft's `source_token` is the ready film's token AND the draft's `[(n, text)]` equals
  that film's passages). There is no column for it. A rebased draft is not `changed`; a draft whose last change was put back to the
  film's text is not `changed` either; a draft of an earlier film (after a failed build, or a newer edit a build did not use) always is.
- why: t63 point 1 asks for a clean/dirty flag; both inputs are already in the two rows read for the payload (§2.19 B.1), so the flag can
  never disagree with them.
- rejected: a `changed` column set by each save and cleared by the rebase (a second fact to keep right, and wrong after a put-back
  edit); comparing the texts only (a draft of an earlier film would read as clean while no film speaks it).
- where: `feed/narration.py:narration_rows`

### D123 — the save barrier, and what a refusal, a network failure and a 409 do in the panel (2026-11-03, PLAN, planner; t63)
- what: `window.feedNotes.settled()` blurs the panel's focused field, waits for the passage queue, and answers True only when no
  passage row is unsaved and no 409 froze the panel. Both prepare forms wait for it and post nothing on False, with the line "Not
  started: an edit to the narration is not saved. See Tell the story better."; "Write a new story" waits for it too. A 400, 404 or
  network failure drops the saves queued behind it (never posted); their rows stay unsaved, and their next blur posts them again. A
  × that has no 200 of its own (refused, failed, or dropped behind a failure) leaves its row and counts as unsaved: `settled()` answers
  False. Click × again, or "Load the current narration", which the error line then offers. A
  409 drops the queue, stops every passage save, makes the passage fields read-only with their text kept, and offers "Load the current
  narration" (the GET), the only re-render that replaces unsaved rows.
- why: t63 point 4; the prepare handler posts at once today (`web/templates/feed.html:328-355`) while a blur save is still in flight
  (`web/static/feed-notes.js:140`), so a build could read the draft before the last edit; after a 409 the row numbers may belong to a
  rebased draft (`n` from 0, §2.19 B.7), so a replayed or a later automatic save could write a text into the wrong passage.
- rejected: replaying the queued saves after a 409 (they would land on rows the user never saw); re-rendering from the GET at once
  after a 409 (it would replace the typed text and the field in focus); waiting for note saves too (not asked; `## Open items outside
  scope`).
- where: `web/static/feed-notes.js`; `web/templates/feed.html` (the prepare handler)

### D124 — the picker line and the legend lead have draft twins, shown by `data-narration` (2026-11-03, PLAN, planner; t63)
- what: with a draft, the Automatic bracket line reads "With your edited narration, a [voice] note cannot ask for a new voice: it keeps
  the voice the story already gave the passage it points at.", and the legend lead reads "With your edited narration, a note's words
  are not narrated and cannot change its words, voices or music. Effect brackets still reach the conductor; matched speech fixes shot
  timing." Each original and its twin carry `data-narration="no-draft"` / `"draft"`; `feed-notes.js` shows the right one on load and
  after every reply.
- why: t63 point 5: "A [voice] note can request another voice for one passage" (`web/templates/feed.html:66-69`) and "the words become
  narration" (`:269-273`) are false with a draft, because no story pass runs and the stored voices only pass the current gate
  (`feed/build.py:372`, `:715-718`, `:733-734`), and matched speech keeps every shot's time (`feed/build.py:559-573`,
  `feed/director.py:343-357`); the first change makes a draft without a page load, so a server-only switch would stay wrong until a
  reload.
- rejected: one sentence that fits both cases (it would be vague in both); changing the legend rows built from `directive_legend()`
  (they describe what each kind does in a story pass); a server-only switch.
- where: `web/templates/feed.html:narrator_picker`, the legend; `web/static/feed-notes.js`

### D125 — a title card names its passage, in the told order (2026-11-03, PLAN, planner; t63; changed t64)
- what: `timeline.script.cards` is `[{"n", "text"}]`, where `n` is the card's passage (its sentence, §2.19 I.2): `timeline_script`
  stores each `(i, text)` of `story.cards` as it is, and drops one past the last passage. `draft_story` turns each kept card into the
  position of its passage in the draft; a removed passage's card is dropped. Nothing calls a sort.
- why: t63 point 6 (a removed passage before a card moved that card to another shot); after §2.19 I shot `i` IS passage `i`, and a card
  names a sentence (t64 point 6), so the position of a passage is all the remap needs.
- rejected: the conversion through `director._in_memory_order` of the t63 plan (t64 deletes that function); cards by photograph (a story
  may show one photograph twice).
- where: `feed/narration.py:timeline_script`, `draft_story`

### D126 — one told order for the film; `with_script` keeps the given order; nothing relabels a photograph (2026-11-03, PLAN, planner; t64)
- what: `director.with_script` expands the script in its own order and keeps its index filter; `_in_memory_order` is deleted.
  `feed/build.py` changes only by the `memory_order` key and (**changed, t65**) the conductor call's `keep_order=bool(story.script)`:
  every consumer of `story.script` (`/speak`, the voices, the gate, the score, the conductor's briefs, the timeline) already walks it in
  told order, so they now share the shots' order. `director.with_arrangement` gets `keep_order`: the order is positional when
  `keep_order` or `keep_timing`, and the times follow `keep_timing` alone. So the order holds also without paired segments; exact
  sync still needs them.
- why: t64 (merge 1 points 1-3, merge 3); the defect was the shots alone being sorted (`feed/build.py:350` against `:463-473`) and the
  sort key moving every return (`feed/director.py:283-289`); a call site per consumer would be a second place to drift. t65: with
  `keep_timing` False (`aligned` needs equal counts, `feed/build.py:559`) `with_arrangement` takes `_resolve_order`
  (`feed/director.py:341-343`), so the conductor could reorder a film whose segments do not pair; `keep_timing=True` is no fix there,
  because the craft pass leaves `at` at 0 (`:128-129`).
- rejected: moving the words with their photograph (it undoes `_in_place` and the note order, merge 1 point 1); `_even` as the
  repair; the relabel `in_memory_order` (merge 3: a photograph under words it was not matched to); keeping `_in_memory_order` for
  another caller (it has none outside `with_script`); `keep_timing=True` without paired segments (every shot at 0 s).
- where: `feed/director.py:with_script`, `with_arrangement`; `feed/build.py:build_presentation`

### D127 — `memory_order` and `matched` are `Story` fields; one pure check serves both paths (2026-11-03, PLAN, planner; t64)
- **Amended 2026-11-04 (pair t74):** no "retried": `memory_order` is "kept" or "broken", a diagnostic of the final photographs; `Story.votes` is added, and `matched` is read only by `decisions["story"]`.
- what: `Story` gains `memory_order: str = "kept"` and `matched: bool = True`, with defaults, so every existing `Story(…)` still builds.
  `introduces_in_order(photos)` (pure, in `feed/story.py`) is the one check: `_dress` uses it on the chosen Story and on the candidate,
  `draft_story` on the resolved draft mapping. `feed/build.py` copies `story.memory_order` into `decisions["story"]`.
- why: `write_story` returns one `Story` (`feed/story.py:232-284`), so the status must travel in it to reach `build.py`; the draft path
  builds a `Story` too (§2.19 C.2); one check keeps "kept" meaning the same on both paths.
- rejected: a second return value of `write_story` (every caller and test changes); a check in `feed/build.py` (it would have to know
  how `_dress` tried); a check copied into `feed/narration.py`.
- where: `feed/story.py:Story`, `introduces_in_order`; `feed/narration.py:draft_story`; `feed/build.py`

### D128 — one parse per item: `_script` returns `(script, voices, matched)` in told order (2026-11-03, PLAN, planner; t64)
- what: §2.19 I.2: one parse of sentence, photograph and voice per item; the first accepted duplicate wins; no item accepted (not a
  list, `[]`, all malformed) → `_even` with `matched` False and no fill; only after at least one accepted item does a missing sentence
  take the previous sentence's photograph (the first one: the next accepted sentence's) and the voice ""; `_voices` is deleted; with
  no told sentences the list order stays.
- why: t64 merge 1 point 5 (`_voices` reads raw item `i` while `_script` skips and inserts items, `feed/story.py:872-884`, `:903-933`);
  merge 3 point 2 needs to tell a real model match from `_even`, and only `_script` knows which one it returned. Today only a non-list
  gives `_even` (`:900`): `[]` and an all-malformed list are filled first (`:928-933`) and repaired by `_every_photograph`, so they
  give an in-order, full-coverage `[0, 1, 0, 2]` that would pass every check of a corrective answer (t65; traced through `:903-967`).
- rejected: comparing a candidate with `_even(told, n)` (a real answer can be equal to it); a separate re-parse to find the fallback
  (two parsers to keep equal); keeping `_voices` beside the new parse.
- where: `feed/story.py:_script`, `_attempt`

### D129 — the corrective call: an `order_fix` header, the five checks, only the photographs taken (2026-11-03, PLAN, planner; t64)
- **Superseded 2026-11-04 (pair t74):** no corrective call and no `order_fix`; `story._match_photographs` matches each sentence after stage 2 and its votes stand (§2.19 I; `<base>/.pair/threads/t74/decision.md`).
- what: `_attempt`, `_prompt` and `_assemble` take `order_fix: bool = False`, beside `insist`; True puts first the CRITICAL line of
  §2.19 I.3. `_dress` makes the call only when the chosen Story is out of order, checks the candidate (no exception and `matched`; the
  same count and texts; valid indexes; every photograph of the chosen Story still shown; first appearances in order), and takes only
  its photo indexes. Else the chosen Story stays, "broken", with a warning in the log.
- why: t64 merge 3 points 1-3; `insist` is the prompt's existing way to say "your previous answer broke a rule" (`feed/story.py:1134-1139`);
  the voice retry passes its correction through `notes` (`:300-311`), which places it under "what the person told you" (`:1151-1197`),
  and an order rule is not the person's words.
- rejected: the correction text in `notes`; a second corrective call; taking the whole candidate (merge 3 point 1); accepting a
  candidate only because its first appearances are in order (merge 3 point 2).
- where: `feed/story.py:_dress`, `_attempt`, `_prompt`, `_assemble`

### D130 — cards name a sentence in the stage-2 schema (2026-11-03, PLAN, planner; t64)
- what: `STORY_SCHEMA` card items are `{"sentence", "text"}`; the prompt asks for the 1-based sentence number; `_cards(raw,
  len(script))` returns 0-based passage positions, at most `MAX_TITLE_CARDS`, and drops a number past the last passage.
- why: t64 merge 1 point 6: the prompt asked for a "shot number" (`feed/story.py:1242-1243`) where "shot" means a photograph (`:1222`),
  `_cards` was bounded by the photo count (`:857`), and `lines_for` reads a position (`:222-229`); design.md:1833 puts a card on "the
  exact narration sentence".
- rejected: keeping `shot` with a new meaning (one word, two meanings in one prompt); cards by photograph (a photograph can carry two
  passages).
- where: `feed/story.py:STORY_SCHEMA`, `_cards`, `_assemble`

## Decisions

- 2026-11-01, reply to "set IVMS777-21 to Done and create the follow-up tickets?": "let me test if first"
- 2026-11-01: "probably it should be on organize screen - saved voices, upload new voice, register it to the persona etc?"
- 2026-11-01: "okay please start with implementation within this task. go ahead and make no mistakes!"
- 2026-11-01, reply to "XTTS-v2 weights allow only non-commercial use — acceptable, or mac-only first with VoxCPM2 (Apache-2.0)?": "no we will use it for commercial so please use models for commercial use"
- 2026-11-01, reply to "May we scope this task to VoxCPM2 voice copy on mac/cloud, with separate Jetson and Silero/MusicGen replacement follow-ups?": "please skip jetson for now, we stop supporting it."
- 2026-11-01, reply to "The voxcpm 2.0.3 dependency gate failed … How do we go on?" (A: drop funasr with override-dependencies "funasr; sys_platform == 'never'", approve the 5 exact exceptions with their obligations in docs/models.md and the PR, macOS 14 as the Mac gate target, re-run the gate; B: pause; C: another engine): "A: drop funasr (Recommended)"
- 2026-11-01: "what a crap!!!?????? I SAID YOU IMPLMEMENT THIS FEATURE AND LET ME KNOW SO I COULD TEST!!!"
- 2026-11-02: "can you please improve organize window. it's not clean what language i use to create video, not clear the memory already have video what language in? when we click rebuild on the video - what language it gets rebuilt into? what voices it uses? what if i like to use specific voice? where is a selection! please address all these issues, make it user friendly!"
- 2026-11-02: "don't forget about auto mode - when you choose voices on your own"
- 2026-11-02, reply to "Chat → Organize → film → Close fully reloads Organize … Fix it in this task, or in a follow-up?": "Fix now (reviewer)"
- 2026-11-02: "how it's possible? [screenshot: the /feed ready screen, the Narrator select under ↓ Download] buttons are overlapped. i used "me" voice for the last attempt, recorded my voice but the resulting video's voice isn't similar to my. we need more source recording maybe? or some tweaks?"
- 2026-11-02, reply to "Should Phase 3 add a test of the high-likeness mode (ref_continuation A/B in voice_check)?": "Not now"
- 2026-11-02: "i can save multiple voices it's okay, but why you give me the same narration always?i guess different texts make more sense when you want to copy something???"
- 2026-11-02: "when i saved multiple voices to the person, with different text, different intonations etc. all of them are used for the voice generation? should it be a pre bake process on the same screen - when i - write voices firsts - then click some bake button - and model generates my own voice from these examples - then button for testing - where i could test final with some example test? or probably the same text i used to record i could use to check - and having button along with the recording - like listen original, listen generated. so i don't like the quality right now, so want to concentrate on the voices screen/view to make this generation more reliable. please give me more tools for that! pre baking also could decimprove total memory generation speed!"
- 2026-11-02, reply to "Where should the voice workbench (compare original vs generated, test sentences, 'Use for films' per language) go?": "Phase 4 in this task (Recommended)"
- 2026-11-02, reply to "You approved the plan. To confirm the open gate question: may "Use for films" replace "the newest example wins" (D19)? …": "Yes (Recommended)"
- 2026-11-02, after the reviewer's STOP on the epoch rule and the Cancel guarantee (this t25 fix): "approved go ahead"
- 2026-11-02 (testing Phase 4, screenshot of «me» with Russian #3/#4 "not used for films", #5 "used for films (newest)"): "what this means: used for films?????/ you fucking idiot should use ALL THE SAMPLES! YOU SHOULD PROVIDE ME DIFFERENT TEXTS WHICH COVERS DIFFERENT ASPECTS OF THE VOICE - SLOW READING, FAST, DIFFENT MOOD ARTICULATIONS! THEN YOU SHOULD USE ALL THE SAMPLES TO BAKE IT INTO THE PERSON'S MODEL .... WHAT THE FUCK YOU JUST DID???????? IT'S SHIT!!!!!! YOU GET IT????"
- 2026-11-02 (screenshot of the test panel): "isn't this looks like a shit as well??????/" — fixed in pair t29 (CSS only).
- 2026-11-02, reply to "To use ALL your samples, the next phase records a guided kit (slow, normal, fast, moods, questions, emphasis) and trains a LoRA voice model per person from every take … Where should the first bake be verified?": "This Mac first (Recommended)"
- 2026-11-02: the Phase 5 amendment approved at the plan gate ("The user approved.").
- 2026-11-02, reply to the gate question "A baked voice model is trained from your recordings. Say you later delete one recording … What should happen to the model?" (options: delete the model too / keep it, mark it out of date): "idiot why do you need to delete model??? why not add a button to rebake model???" — so the model is NOT deleted when a take it learned from is deleted; a Rebake button trains it again. Slice B is written the other way (delete), so Slice B goes back to the planner for this rule before any Slice B code; Slices A and V are not affected.
- 2026-11-02 (while Phase 5 was being planned): "add Voices menu item, move all voices stuff there. improve UX/UI what we have now is total shit!"
- 2026-11-02 (screenshot of the new /voices/{id} page whose header reads "Russian: films use take #5 (newest)"): "once again films should not use any specific recording. BUT ALL RECORDINGS! YOU DON'T UNDERSTAND IT FUCKING IDIOT?????"
- 2026-11-02: "based on this it should be probably 1 test button per person. where you could choose what text to generate or provide your own text to be generated!"
- 2026-11-02 (screenshot of the Recordings tab with a Test button on every take and the hint "Films use one example per language: the newest, unless you choose one on the Voice model tab."): "I ASKED YOU TO REWORK THIS SHIT WHY IT'S NOT DONE????????" — meaning: do the rework now; do not wait for the §6.5 Mac bake check.
- 2026-11-02: "you understand that text is given just for some example of something to say, you should not link to this test. i can say ANYTHING, OR UPLOAD ANYTHING! NOT RELATED TO THE TEXT AND YOU SHOULD USE IT!!!!!! IDIOT!"
- 2026-11-02, reply to "The voice model needs the words of each recording to train. May I add local speech recognition (in the models service), so any recording or upload trains your voice, and you type and confirm nothing? The shown text stays only a suggestion.": "Yes, add it (Recommended)"
- 2026-11-03: the ASR amendment (t42, with the t44 plan-check corrections) approved at the plan gate ("The user approved."), after a guided change view with no annotations.
- 2026-11-03, reply to "Another session edits feed/build.py: `_bracketed_passages` now returns set() when no note has a bracket; it breaks 9 cast/bracket tests in tests/test_feed_build.py. How do I handle it?": "Keep it, adapt tests" (keep that change; a fixer adapts the 9 tests; that session stops editing build.py while lanes run).
- 2026-11-03: "for voice recording please add a pause button. so we could make pauses during recording. for re reading text for example and other purposes."
- 2026-11-03: "for now if i choose one voice , then you should use it for the whole movie."
- 2026-11-03: "also add auto mode - where you could chose appropriate voice on your own"
- 2026-11-03, reply to "Auto mode: the /feed picker already has "Automatic", but it picks only a preset voice (a saved voice is never picked automatically). Should Automatic also be allowed to pick one of your baked saved voices as the narrator?": "Yes, presets + saved" (Automatic may pick a preset or one of the user's baked saved voices of the film's language; one voice engine per film stays; the design and PLAN are amended first).
- 2026-11-03, reply to "docs/design.md:2272-2274 says the nearest passage that shows the bracketed photo is spoken by the other voice; `_bracketed_passages` uses only the note's position. Should I fix that doc text in this task, or leave it for a separate one?": "go ahead"
- 2026-11-03: the Automatic amendment (Phase 6, §2.18, D105-D111, with the t54 plan-check corrections) approved at the plan gate ("The user approved."), after a guided change view with no annotations.
- 2026-11-03, reply to the 33a question ("Slice B is ready except two models-service changes … Tell me when no bake runs."): "sorry what answer idiot. go ahead" (no bake runs; lanes 31b then 31d start).
- 2026-11-03: "when we generate a video - we have ability to edit the narration right?" — answered: no; only memory notes (`feed/notes.py`, `web/static/feed-notes.js`, `web/app.py:1075-1118`) steer the story pass, and Prepare/Regenerate rewrites the words.
- 2026-11-03: "can you please make when narration is done it gets added to the list of text with timelines, so i can just edit it without creating everything from scrach. do you understand what i'm talking to you about? [screenshot: the 'Tell the story better' notes panel on /feed — title, 'Use [brackets] to direct it', 0:00 stamp, the textarea 'Who is this? What happened here?', the Add button] this should be shows with the generated story by default! so i can change it (and rebuild of course later)"
- 2026-11-03: the pair (writer + reviewer) settled the approach of that request in t62 (`<base>/.pair/files/t62-edit-narration-agreed.md`): the film's narration as timed, editable rows in the notes panel by default; a draft per memory and language; ↻ Regenerate speaks it as written. This is Phase 7 (§2.19, D112-D121). The user's plan-gate approval of this amendment is pending.
- 2026-11-03: the pair's plan check of Phase 7 (t63, `<base>/.pair/threads/t63/decision.md`) revised it: the consumed draft is rebased and kept (it supersedes t62 point 5), a `draft_token`, `#narration-data`, the save barrier, the corrected texts, the card remap and the app limit (§2.19 A-H, D112-D125). The user's plan-gate approval is still pending.
- 2026-11-03, reply to "Should we fix it in this task or in a follow-up task?" (the existing shot/audio order defect: `feed/director.py:257-317`, `with_script` re-sorts the shots into memory order while the `/speak` segments are in telling order): "Fix in this task". The pair is settling how to fix it; the plan holds §2.19 I as a placeholder until that decision.
- 2026-11-03: the pair settled how to fix it in t64 (`<base>/.pair/threads/t64/decision.md`; `<base>/.pair/files/t64-merge3-writer.md` is the last word): one told order for the words, the voice and the pictures; no sort and no relabel; the memory's photo order checked and corrected by one stage-2 call, else the story's match wins (`memory_order` kept | retried | broken). This is §2.19 I, lane 39b, D126-D130. The user's plan-gate approval is still pending.
- 2026-11-03: the Phase 7 amendment (§2.19 A-I, D112-D130, with the t63 and t65 plan-check corrections) approved at the plan gate ("The user approved."), after a guided diff view with no annotations.
- 2026-11-04: "bro the text and images aren't synced. please check generation logs. you show single cat (2nd time) but discussing previous image with 2 cats"; "check logs, fix this shit! also video sometimes has only audio track and animation is frozen, fix that too!"
- 2026-11-04: "there is no scrollbar ... how i supposed to scroll through them ???"
- 2026-11-04: "add in tell the story better chat field - so i can ask AI to rephrase it, when rephrasing it should have access to all the details used by original video composed, music, timeline etc."
- 2026-11-04: Phase 8 and the t74 photo match amendment approved at the plan gate ("The user approved."), after a guided diff view with no annotations.

## Open disagreements

### Where this PLAN.md lives — the user settles it at the gate
- writer: PLAN.md at `<base>`, outside the repo, never committed — `work_sadensmol/skills/task/SKILL.md:38` ("PLAN.md — outside the repo — never committed").
- reviewer: `docs/plans/NN-*.md` in the repo — `ivms777/CLAUDE.md:45-49` ("Every plan lives in `docs/plans/`, no exceptions … Never write a plan anywhere else — not a scratch dir … not the skill's own default location").
- If the user picks the repo: this plan goes into the PR as `docs/plans/26-voice-copy.md` (the highest existing plan is `docs/plans/25-memory-score.md`).

- The `chat.js` bfcache reload (pair t19) was settled by the user on 2026-11-02: "Fix now (reviewer)" (`## Decisions`). Applied in §2.15 B, lane 15b, §5 and §6.3 step 10.

## Open items outside scope

- **Added 2026-11-02 (ASR, t42) — no work in this task:**
  - Words that ASR hears wrongly are trained as heard; nothing corrects them (the user chose no editor), and no accuracy figure is claimed.
    A take that ASR heard is never heard again, also after a later ASR revision.
  - `scipy` reaches the models service only through `librosa`, a `voxcpm` dependency (`uv.lock:2055-2070`); if `voxcpm` drops `librosa`,
    the Whisper child's resample must move to another library.
  - No ASR on cloud or jetson: it follows bakes, which are mac only until a cloud run is measured (D62).
  - Float16 Whisper on MPS (about half the memory) was not measured; float32 stays until a run shows float16 gives the same words.
  - Found during the t44 corrections, not planned (it is in the approved t39/t40 design, not in the ASR stage): a snapshot take that is
    deleted and drained between `start_bake_run` and "sending the takes" has no file left, because the drain's ref check does not read
    `voice_bake_runs` (§2.17 K3 drain); that new start then fails on the audio read. A fix needs a decision outside this amendment.
- **Answered 2026-11-02:** "idiot why do you need to delete model??? why not add a button to rebake model???" (`## Decisions`). The model is
  kept; Rebake trains a new version. Slice B is re-planned to it (§2.17 K0, K3, K7; D79). The recommendation below was not taken.
  **Question at the gate (Phase 5, blocks Slice B, not Slices A and V):** "When you delete a take that a baked voice model learned from, should
  the app delete that model too (films then speak from one take until you bake again), or keep using it until you bake again?"
  Recommendation: delete it — a delete removes what was made from the deleted take (D73). Slice B is written that way (§2.17 K3, K7). The other
  answer sends Slice B's drain and state rules back to the planner.
- **Baking on the cloud profile (follow-up, D62).** Bakes are offered on mac only until a cloud run is measured; a model baked on the mac is used
  wherever `voxcpm2` runs at the same revision. No ticket, no work in this task.
- **Close rule of the other leaves (follow-up, t30).** `photo.html:181` and `feed.html:289` still close on `history.length > 1`; the voice leaf
  uses the known-grid rule (D67). Aligning them is outside this task.
- **Question at the gate (blocks the commit, not the work):** `pyproject.toml` and `uv.lock` carry your uncommitted `en-core-web-sm` change. `uv lock` puts the `voxcpm` entries next to it,
  so the two cannot be split cleanly. May this PR include your `en-core-web-sm` change, or will you commit it first?
- **Question at the gate (product, blocks the code):** "Deleting a saved voice keeps films already made with it until you regenerate them; a film still being made fails.
  OK, or should deleting also remove those films' narration?" Recommendation: OK. This plan is written that way (§2.8, D33). Removal is a re-plan through the planner.
- **Answered 2026-11-02: "Yes (Recommended)"** (`## Decisions`). D19's Phase 4 "changed" line applies.
  **Question at the gate (Phase 4, blocks the Phase 4 code; t24):** "Phase 4 adds 'Use for films': per language you choose which example of a saved
  voice films use. This replaces the rule 'the newest example wins' (D19); the newest stays the default when you choose nothing. When you delete the
  chosen example, films with that voice in that language fail with the reason until you choose another example; the app never picks another one by
  itself. OK?" Recommendation: yes. Phase 4 is written that way (§2.16 A-C, D46-D48). Any other answer sends Phase 4 back to the planner before any
  Phase 4 code.
- **Silero and MusicGen licences.** Silero v4 ru (`audio/silero_voice.py:14`; `models/catalog.py:379`, `:392-393`) and MusicGen small and medium
  (`models/catalog.py:404-407`, `:420-429`) carry CC-BY-NC weights. Automatic Russian narration and every score are therefore not commercial.
  Voice copy does not change that, and Silero is not a commercial fallback. No ticket, no work in this task.
- **Friendly preset names (follow-up, pair t19).** Names such as "Heart" for `af_heart` need a display field declared in the catalog (`models/catalog.py:113-116`,
  `ivms777/CLAUDE.md:234-239`). Not in this task; D37 keeps the catalog name.
- ref_continuation (VoxCPM2 'Ultimate Cloning') A/B in voice_check — the user said 'Not now' (2026-11-02); D16 changes only after a measured listening result.
  **Changed 2026-11-02 (Phase 5):** the working tree already uses ref_continuation for a take with words (§1.8, D16); Slice A's blind listening
  compares the baked variants with that path (D64).
- **Jetson removal.** Removing the jetson profile from code and docs is not in this task. The generic capability rule (§2.11) covers the profile while it exists.
- Noticed, not changed: `docs/design.md` §4.1 (`:366-374`) and `docs/models.md` describe an `off` voice/music entry that the code does not have (`modelsvc/slots.py:181-184`);
  this task removes `off` only from the rows it rewrites. The §5 diagram puts "feed audio" in `worker`; the code builds films in `app` (`web/app.py:995-1043`).
- Noticed, not changed: with no worker, a folder deletion waits for the next upload. The route only enqueues (`web/app.py:1747-1753`), and `drain_now`'s only caller
  is upload finish (`web/upload_api.py:158`). No fix in this task.
- **Answered 2026-11-03: "Fix in this task"** (`## Decisions`); fixed by §2.19 I (pair t64, lane 39b). Found during Phase 7
  planning: `director.with_script` re-sorts the shots into memory order (`feed/director.py:257-290`), while the voice speaks the passages
  in telling order (`feed/build.py:463-473`) and `with_narration_timing` gives sorted shot `i` the start of passage `i`
  (`feed/director.py:306-317`). When the story pass puts its passages out of memory order, a photograph shows under another passage's
  words. Phase 7's rows follow the audio, so their times are right either way.
- Noticed, not changed (2026-11-03, t63): a note saves on blur without a wait (`web/static/feed-notes.js:83-93`), and the save barrier
  of §2.19 E waits for passage saves only. A note typed just before ↻ Regenerate may miss that build. No fix in this task.
