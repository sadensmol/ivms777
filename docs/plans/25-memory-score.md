# Memory Score Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the film's single looped 30-second clip with a **score** — two to four sections that follow the story's own turns, generated as one continuous piece, mixed under the voice, ending instead of looping.

**Architecture:** The story pass already writes the passages, so it also plans the score: 2-4 sections, each pinned to a passage. `feed/score.py` (new, pure) turns that plan plus the memory's own signals into a brief per section — place, season, light, hour, occasion, energy and a section ROLE (open / lift / turn / close) that moves tempo and instrumentation. `POST /music` takes the whole list; the MusicGen child generates section one from text and each next section as a **continuation** of the previous one's tail, so the result is one piece that changes rather than four clips glued together. Section boundaries are the narration's own passage times, so the music turns where the story turns.

**Tech Stack:** Python 3.13 · MusicGen via `transformers` 5.15 in a `TorchWorker` child · numpy for the splice · WebGL2 player (`web/static/feed-gl.js`).

**Spec:** `docs/design.md` §11.1 (the film pipeline) and §5.1 (`POST /music`), rewritten by Task 1 of this plan, which is therefore the spec.

## Global Constraints

- **One model process (CLAUDE.md, §5.1).** `feed/` stays a thin client: no `torch`, no `transformers`. Only `modelsvc/` and `audio/` may import them. `tests/test_one_model_process_gate.py` scans `feed`.
- **`feed/score.py` is PURE** — no HTTP, no models, no clock, no randomness. Same memory in, same brief out.
- **Narration is mandatory; music is not (§11.1).** Every failure in this plan degrades to today's behaviour or to a silent film — never to a failed prepare.
- **No "off" option (CLAUDE.md).** The `music` slot names a real model on every profile.
- **A model's metadata is DECLARED (CLAUDE.md).** A new catalog entry carries its measured cost, not a guess.
- **Verified, not assumed.** MusicGen continuation was proven to run on this stack before this plan was written: `processor(audio=tail, sampling_rate=32000, text=[brief])` → `generate()` returns the prompt tail **plus** the new audio, so the splice must trim that tail.
- **`ruff` line-length is 100.** `make lint` before each commit.
- **Test command:** `uv run pytest tests/<file>.py -v`. Full suite: `make test`.
- **Never run `git commit` / `git add`** — the user commits. Where a task says "Commit", report the task is ready to commit instead.

---

## File Structure

**New — pure, thin-client, gate-scanned:**

| File | Responsibility |
|---|---|
| `feed/score.py` | `signals + story plan + passage times → [ScoreSection]`. All brief composition. Pure. |
| `tests/test_feed_score.py` | Its tests. |

**Modified:**

| File | Change |
|---|---|
| `feed/story.py` | `STORY_SCHEMA` + prompt gain `score`; `Story.score`; `_score()` repair. |
| `feed/mood.py` | Per-role score vocabulary the sections draw on; `music_prompt` keeps its single-brief job as the fallback. |
| `feed/build.py` | Build the sections after the narration, ask for the whole score in one call, stop clamping to 30 s. |
| `audio/musicgen.py` | Chained continuations, the splice, `temperature`/`guidance`/`seed`. |
| `modelsvc/backends/{base,composite,fake,music_backend}.py`, `modelsvc/app.py`, `inference/models_client.py` | `POST /music` takes `sections`, `seed`, `temperature`, `guidance`; returns `segments`. |
| `models/catalog.py` | `musicgen-medium` entry; mac default (**subject to Task 0's measurement**). |
| `web/static/feed-gl.js` | Music no longer loops; duck under narration by segment. |
| `docs/design.md` | §5.1 `/music`, §11.1 the score. |

---

### Task 0: Measure before choosing the model

A bigger model is only worth it if a film still prepares in a reasonable time. `musicgen-medium` is ~4x the parameters of `small`; MusicGen is autoregressive, so cost is linear in the SECONDS of audio asked for, and the score now covers the whole film rather than 30 s.

- [ ] **Step 1: Benchmark both on this machine**

Run a throwaway script that loads `facebook/musicgen-small` and `facebook/musicgen-medium` on `mps` and times 15 s of audio from each. Record seconds-of-compute per second-of-audio.

- [ ] **Step 2: Decide the mac default from the number**

Budget: the score for a 90-second film. If `medium` puts that over ~4 minutes of generation, mac stays on `small` and the plan says so out loud — the arc, the briefs and the sampling are the larger win either way. Whatever is chosen, Task 7 records the MEASURED `cost_mb` and the measured rate in the catalog note.

---

### Task 1: The design docs, first

`docs/design.md` describes the current state and changes before the code (CLAUDE.md).

**Files:**
- Modify: `docs/design.md` §5.1 (`POST /music`), §11.1 (the score), `docs/models.md`

**Interfaces:**
- Produces: the vocabulary every later task uses — `ScoreSection(brief, seconds, role)`, `POST /music {sections, seed, temperature, guidance}` → `{audio, sample_rate, model, segments}`.

- [ ] **Step 1: Rewrite the music paragraph of §11.1**

The film's score is 2-4 sections, planned by the story pass, briefed from the memory's own signals, generated as one continuous piece by chained continuation, cut on narration passage boundaries, ducked under the voice and ended rather than looped. State the fallbacks: no plan → one section from `mood.music_prompt`; a failed call → a silent film (music has always been optional).

- [ ] **Step 2: Extend the `POST /music` bullet in §5.1**

`{prompt | sections, seconds, seed, temperature, guidance}` → `{audio, sample_rate, model, segments}`. `segments` are the section boundaries actually produced, because a continuation's length is not exactly what was asked for.

- [ ] **Step 3: `docs/models.md`** — the music slot's entry gains whatever Task 0 chose, with its measured rate.

---

### Task 2: `feed/score.py` — the sections

**Files:**
- Create: `feed/score.py`
- Test: `tests/test_feed_score.py`

**Interfaces:**
- Consumes: `feed.signals.ShotSignal`, `feed.mood.energy`.
- Produces:
  - `ScoreSection(role: str, brief: str, seconds: float, at: float)` — frozen dataclass.
  - `plan_sections(signals, script, passage_times, runtime, energy, asked="", model_plan=()) -> list[ScoreSection]`
  - `ROLES = ("open", "lift", "turn", "close")`

- [ ] **Step 1: Write the failing tests**

```python
def test_a_short_film_gets_two_sections():
    sections = plan_sections(signals, script, times, runtime=24.0, energy="calm")
    assert len(sections) == 2

def test_a_long_film_gets_four():
    ...
    assert len(sections) == 4

def test_sections_cover_the_whole_runtime_without_gaps():
    total = sum(s.seconds for s in sections)
    assert abs(total - runtime) < 0.01
    assert sections[0].at == 0.0

def test_a_boundary_lands_on_a_passage_start():
    assert all(s.at in set(passage_times) for s in sections[1:])

def test_the_brief_names_what_is_in_those_photographs():
    # snow + evening + Borjomi in the second half only -> only that section says so
    assert "snow" in sections[1].brief and "snow" not in sections[0].brief

def test_the_role_moves_the_tempo():
    assert "BPM" in sections[0].brief
    assert sections[0].brief != sections[-1].brief

def test_the_last_section_ends_rather_than_vamps():
    assert "ending" in sections[-1].brief or "resolving" in sections[-1].brief

def test_the_viewers_request_leads_every_section():
    sections = plan_sections(..., asked="fun chord")
    assert all(s.brief.startswith("fun chord") for s in sections)

def test_the_models_plan_is_used_when_it_is_sane():
    sections = plan_sections(..., model_plan=[{"from": 0, "brief": "solo cello"},
                                              {"from": 3, "brief": "brass, loud"}])
    assert "solo cello" in sections[0].brief and "brass" in sections[1].brief

def test_a_nonsense_plan_is_repaired_not_obeyed():
    # out-of-range passage, one section, empty briefs
    sections = plan_sections(..., model_plan=[{"from": 99, "brief": ""}])
    assert 2 <= len(sections) <= 4 and all(s.brief for s in sections)
```

- [ ] **Step 2: Run them and watch them fail** — `uv run pytest tests/test_feed_score.py -v`

- [ ] **Step 3: Implement**

Section count from the runtime: `< 30 s → 2`, `< 75 s → 3`, else `4`. Boundaries: the passage start nearest each even division of the runtime, so a section never begins mid-sentence. Roles: `open`, then `lift`/`turn` for the middle, `close` last — each contributing a tempo nudge and an instruction (`"opening, sparse, let it breathe"`, `"lift, add percussion, more movement"`, `"a turn, change the instrument"`, `"ending, resolve and fade"`), on top of the energy sentence from `feed/mood.py`. The section's own stretch of `signals` supplies the concrete nouns: place, `season_weather`, `light`, the hour of day, `occasion`. The viewer's `[music]` request leads (§11.3), the model's brief for that section follows, the machine's sentence last.

- [ ] **Step 4: Run the tests** — all pass.

- [ ] **Step 5: Commit** (report ready; the user commits).

---

### Task 3: The story pass plans the score

**Files:**
- Modify: `feed/story.py`
- Test: `tests/test_feed_story.py`

**Interfaces:**
- Produces: `Story.score: tuple[tuple[int, str], ...]` — `(passage index, brief)`, in order, at most 4.

- [ ] **Step 1: Write the failing tests**

```python
def test_the_score_plan_is_parsed():
    payload = {"script": [...], "score": [{"from": 0, "brief": "solo piano"},
                                          {"from": 4, "brief": "strings swell"}], ...}
    story = write_story(FakeClient(json.dumps(payload)), ...)
    assert story.score == ((0, "solo piano"), (4, "strings swell"))

def test_a_score_entry_pointing_outside_the_script_is_dropped():
    ...
    assert story.score == ((0, "solo piano"),)

def test_more_than_four_sections_are_cut_to_four():
    assert len(story.score) == 4

def test_a_missing_score_is_not_an_error():
    assert story.score == ()

def test_the_prompt_asks_for_the_score():
    assert '"score"' in prompt and "changes" in prompt
```

- [ ] **Step 2: Run them, watch them fail.**

- [ ] **Step 3: Implement** — `STORY_SCHEMA` gains

```python
"score": {
    "type": "array",
    "items": {"type": "object",
              "properties": {"from": {"type": "integer"}, "brief": {"type": "string"}},
              "required": ["from", "brief"]},
},
```

a `_score(raw, passages)` repair beside `_cards` (drop out-of-range `from`, drop empty briefs, sort by `from`, cap at 4), and a prompt bullet: two to four moments where the music should CHANGE, each naming what it changes to, in English (the audio model only reads English), pinned to the passage it starts at.

- [ ] **Step 4: Tests pass.**
- [ ] **Step 5: Commit.**

---

### Task 4: `/music` takes a score

**Files:**
- Modify: `audio/musicgen.py`, `modelsvc/backends/{base,composite,fake,music_backend}.py`, `modelsvc/app.py`, `inference/models_client.py`
- Test: `tests/test_modelsvc_api.py`, `tests/test_models_client.py`, `tests/test_modelsvc_backends.py`

**Interfaces:**
- Produces: `music(prompt=None, seconds=30.0, seed=None, *, sections=None, temperature=None, guidance=None) -> {audio, sample_rate, model, segments}` where `sections` is `[{"brief": str, "seconds": float}]` and `segments` is `[{"start": float, "end": float}]`.

- [ ] **Step 1: Write the failing tests** (fake backend + API round trip)

```python
def test_music_accepts_a_score_and_reports_its_sections():
    resp = tc.post("/music", json={"sections": [{"brief": "a", "seconds": 4},
                                                {"brief": "b", "seconds": 4}]})
    segments = resp.json()["segments"]
    assert len(segments) == 2 and segments[0]["end"] == segments[1]["start"]

def test_music_still_accepts_one_prompt():
    assert tc.post("/music", json={"prompt": "warm", "seconds": 4}).status_code == 200

def test_a_score_and_a_prompt_together_is_a_422():
    ...
```

- [ ] **Step 2: Run them, watch them fail.**

- [ ] **Step 3: Implement the child**

In `audio/musicgen.py`: generate section 1 from text; for each later section call the processor with `audio=<tail of the previous section>, sampling_rate=self._sample_rate, text=[brief]`, generate, then **drop the leading `len(tail)` samples** from the result (verified: the model returns the prompt audio followed by the continuation) and crossfade `CROSSFADE_S = 0.35` into the previous section. `seed` seeds `torch.manual_seed` once for the whole score so a rebuild of the same memory sounds the same. `temperature` and `guidance` reach `generate()`; keep today's `guidance_scale=3.0` as the default so an unchanged caller is unchanged. `MAX_SECONDS` becomes a per-SCORE cap (`SCORE_MAX_SECONDS = 180.0`) with the per-section cap at 40 s.

- [ ] **Step 4: Tests pass.**
- [ ] **Step 5: Commit.**

---

### Task 5: `feed/build.py` asks for the score

**Files:**
- Modify: `feed/build.py`
- Test: `tests/test_feed_build.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_the_music_call_carries_one_section_per_planned_change():
    # fake client records the call
    assert len(fake.music_calls[0]["sections"]) == len(story.score)

def test_the_sections_add_up_to_the_films_runtime():
    ...

def test_a_film_with_no_score_plan_still_gets_music():
    assert fake.music_calls[0]["sections"]  # from mood, not from the model

def test_music_that_fails_still_leaves_a_watchable_film():
    assert stored.status == "ready" and stored.music_key is None
```

- [ ] **Step 2: Run them, watch them fail.**

- [ ] **Step 3: Implement** — after the narration (so passage times exist), call
`score.plan_sections(shot_signals, story.script, passage_times, runtime, energy, asked=directives.music, model_plan=story.score)` and pass the result to `client.music(...)` with a seed derived from the memory's signature so a rebuild is reproducible. Drop `min(runtime, MUSIC_MAX_SECONDS)`.

- [ ] **Step 4: Tests pass.**
- [ ] **Step 5: Commit.**

---

### Task 6: The player stops looping and ducks

**Files:**
- Modify: `web/static/feed-gl.js`, `web/templates/feed.html` if the audio element carries `loop`

- [ ] **Step 1: Remove the loop** — the score now covers the film; a loop can only produce a seam.
- [ ] **Step 2: Duck under the narration** — the existing gain node drops to `DUCK = 0.45` while a narration segment is playing and returns to `1.0` between passages, with a 250 ms ramp so the change is not a click.
- [ ] **Step 3: Check the film still stops on its own last frame** — the "music loops" comments at `feed-gl.js:453`, `:664` and `:869` describe behaviour this task removes; correct them.
- [ ] **Step 4: Commit.**

---

### Task 7: The catalog entry

**Files:**
- Modify: `models/catalog.py`
- Test: `tests/test_catalog.py`

- [ ] **Step 1: Write the failing test** — the chosen model is in the catalog, is on the `music` slot, and is the mac default.
- [ ] **Step 2: Implement** — the entry carries the MEASURED `cost_mb` and a note stating its measured generation rate (Task 0), the licence (CC-BY-NC 4.0), and why it is or is not the mac default.
- [ ] **Step 3: `uv run python -m scripts.fetch_models`** pulls the weights.
- [ ] **Step 4: Tests pass. Commit.**

---

## Self-Review

**Spec coverage:** every point of the approved design has a task — the story pass's plan (3), the composed briefs (2), chained continuation with real sampling controls (4), boundaries on narration passages (2, 5), no loop plus ducking (6), the model (0, 7), the docs (1).

**Placeholders:** none. Task 0's outcome is an explicit decision point, not a TBD — the plan states the budget that decides it.

**Type consistency:** `ScoreSection(role, brief, seconds, at)` is produced in Task 2 and consumed in Task 5; `Story.score` is `tuple[tuple[int, str], ...]` in Task 3 and is `model_plan` in Task 2; `/music` takes `sections=[{"brief", "seconds"}]` and returns `segments=[{"start", "end"}]` in Tasks 4 and 5.
