"""The score plan (design §11.1) — WHERE the music changes, and what it changes to.

PURE: signals and passage times in, section briefs out. No model, no I/O, no clock.

It exists because one clip is not a score. A single 30-second brief generated once and
looped under the whole film is the same texture from the first photograph to the last:
it cannot open, lift, turn or end, so however good the narration and the effects are,
the film sounds like a slideshow with a bed under it.

Three things decide a section's brief, and all three are already known before a note is
generated:

- **the memory's own energy and occasion** (`feed/mood.py`) — a birthday is not scored
  like a hospital visit, and that judgement is measured from the tags, not guessed;
- **the photographs in THAT STRETCH of the film** — snow in the last third belongs in
  the last third's brief and nowhere else. This is the part a single prompt cannot do at
  all: it can only ever describe the average of the whole memory;
- **the role the section plays** — an opening is sparse, a lift adds movement, a turn
  changes the instrument, an ending resolves. Without this the sections are four
  variations on one idea rather than a piece that goes somewhere.

The story pass may also plan the score itself (`Story.score`), because it wrote the
passages and knows where the film turns. Its plan decides WHERE the changes fall and
adds its own words to each brief; it never replaces the measured half, since a small
model reaches for "warm ambient pads, gentle, unobtrusive" whatever it is shown.
"""

from dataclasses import dataclass

from feed import mood
from feed.signals import ShotSignal
from feed.signals import _common as common_labels

# In running order. A film with fewer sections uses the first and the last — an opening
# and an ending are the two that carry the shape; a lift with no ending is a fade-out.
ROLES: tuple[str, ...] = ("open", "lift", "turn", "close")

# What each role asks the audio model to DO differently. Stated as an instruction about
# this section rather than as an adjective, because an adjective is absorbed into the
# same average every other adjective lands in.
_ROLE_BRIEF: dict[str, str] = {
    "open": "This is the opening: start sparse and unhurried, one instrument, let it "
            "breathe.",
    "lift": "This section lifts: add percussion and movement, a little faster and "
            "fuller than before.",
    "turn": "This section turns: change the lead instrument and the feel, keep the "
            "same key.",
    "close": "This is the ending: bring it down, resolve, and fade out — it must end, "
             "not loop.",
}

# How many changes a film of a given length can carry. Two sections under half a minute
# is already a change every fifteen seconds; four under it would be four stings.
_COUNT_BY_RUNTIME: tuple[tuple[float, int], ...] = ((30.0, 2), (75.0, 3))
_MAX_SECTIONS = 4

# How far a section boundary may be dragged to reach the start of a sentence. Mirrors
# `conductor.SNAP_WINDOW`: beyond this the boundary is no longer the one that was
# planned, and snapping stops serving the score and starts deciding it — which is how a
# film whose passages all fall in its first ten seconds ended up with one section.
SNAP_WINDOW = 2.5

# A section shorter than this is a sting, not a piece of music — MusicGen needs a few
# seconds before it has established anything at all.
MIN_SECTION_SECONDS = 5.0

# The whole score, capped. MusicGen is autoregressive and MEASURED at ~4.9x realtime for
# `musicgen-small` on this mac (15 s of audio in 73 s), so every second here is five
# seconds the user waits on the progress bar. Three minutes of score is already ~15
# minutes of generation; past that a film is better served by ending its music early
# than by doubling a wait nobody asked for.
SCORE_MAX_SECONDS = 180.0

# The tags worth naming to an audio model. `season_weather` and `light` are what make a
# score sound like the day it is under; `setting` and `occasion` are what make it sound
# like the place. The rest of the taxonomy (composition, quality, palette) describes the
# PHOTOGRAPH, not the day, and reads as noise in a music brief.
_SCENE_DIMENSIONS: tuple[tuple[str, str], ...] = (
    ("season_weather", ""), ("light", ""), ("setting", "in a "),
)


@dataclass(frozen=True)
class ScoreSection:
    """One stretch of the film and the music that plays over it."""

    role: str
    brief: str
    at: float
    seconds: float


def plan_sections(
    signals: list[ShotSignal],
    passages: list[tuple[float, int]],
    runtime: float,
    energy: str,
    model_plan: tuple[tuple[int, str], ...] = (),
    hint: str = "",
    title: str = "",
) -> list[ScoreSection]:
    """The film's score, as 2-4 briefs with their own start and length.

    `passages` is `(start second, shot index)` per narration passage — the voice's own
    timings, so a section can only ever start where a sentence starts. `model_plan` is
    the story pass's `(passage index, brief)` plan, used where it is sane and repaired
    where it is not. `hint` is its film-wide sentence (`Story.music_prompt`), which
    colours any section its plan said nothing about.
    """
    runtime = max(MIN_SECTION_SECONDS, min(float(runtime), SCORE_MAX_SECONDS))
    starts = _starts(passages, runtime, model_plan)
    plan = _plan_by_start(passages, model_plan)
    roles = _roles(len(starts))
    sections = []
    for index, (at, role) in enumerate(zip(starts, roles)):
        end = starts[index + 1] if index + 1 < len(starts) else runtime
        sections.append(ScoreSection(
            role=role,
            brief=_brief(signals, passages, at, end, energy, role,
                         plan.get(at, "") or hint, title),
            at=at,
            seconds=round(end - at, 3),
        ))
    return sections


def _count(runtime: float) -> int:
    """How many changes this film can carry, and how many it can AFFORD: a section must
    be long enough to be music (`MIN_SECTION_SECONDS`)."""
    count = _MAX_SECTIONS
    for limit, allowed in _COUNT_BY_RUNTIME:
        if runtime < limit:
            count = allowed
            break
    return max(1, min(count, int(runtime // MIN_SECTION_SECONDS)))


def _roles(count: int) -> list[str]:
    """`open` and `close` first — the shape of a piece is its beginning and its end;
    the middle roles fill in as the film gets long enough to need them."""
    if count <= 1:
        return ["open"]
    return [ROLES[0], *ROLES[1:count - 1], ROLES[-1]]


def _starts(
    passages: list[tuple[float, int]], runtime: float,
    model_plan: tuple[tuple[int, str], ...],
) -> list[float]:
    """Where each section begins. Always on a passage boundary when there is one: a
    change of music mid-sentence is heard as a mistake rather than as a turn."""
    times = [at for at, _ in passages if 0.0 < at < runtime]
    planned = [passages[index][0] for index, _ in model_plan
               if 0 <= index < len(passages) and 0.0 < passages[index][0] < runtime]
    wanted = _count(runtime)
    if planned:
        # The model decided where the film turns. Keep its boundaries, in order, and
        # only as many as the runtime can carry.
        starts = sorted(dict.fromkeys(planned))[: wanted - 1]
    else:
        # An even split, each division dragged to the nearest sentence start.
        starts = []
        for step in range(1, wanted):
            target = runtime * step / wanted
            nearest = min(times, key=lambda t: abs(t - target)) if times else target
            if abs(nearest - target) > SNAP_WINDOW:
                nearest = target
            if nearest not in starts:
                starts.append(nearest)
    starts = [0.0, *starts]
    return _spaced(starts, runtime)


def _spaced(starts: list[float], runtime: float) -> list[float]:
    """Drop any boundary that would leave a section too short to be music."""
    kept = [starts[0]]
    for start in starts[1:]:
        if start - kept[-1] >= MIN_SECTION_SECONDS and runtime - start >= MIN_SECTION_SECONDS:
            kept.append(start)
    return kept


def _plan_by_start(
    passages: list[tuple[float, int]], model_plan: tuple[tuple[int, str], ...]
) -> dict[float, str]:
    """The model's briefs, keyed by the second they start at."""
    return {passages[index][0]: brief for index, brief in model_plan
            if 0 <= index < len(passages) and brief}


def _brief(
    signals: list[ShotSignal], passages: list[tuple[float, int]],
    at: float, end: float, energy: str, role: str, asked: str, title: str,
) -> str:
    """One section's brief: what the memory is, what is on screen HERE, what this
    section does, and whatever the story pass asked for."""
    here = _signals_between(signals, passages, at, end)
    measured = mood.music_prompt(title, here or signals, hint=asked, energy_label=energy)
    parts = [measured, _scene(here or signals), _ROLE_BRIEF[role]]
    return " ".join(part for part in parts if part).strip()


def _signals_between(
    signals: list[ShotSignal], passages: list[tuple[float, int]], at: float, end: float
) -> list[ShotSignal]:
    """The photographs on screen during this section — the point of sectioning at all.

    Falls back to the whole memory when there are no passage timings (a film whose
    voice reported no segments), which is the old single-brief behaviour.
    """
    indexes = [shot for start, shot in passages if at <= start < end]
    return [signals[i] for i in indexes if 0 <= i < len(signals)]


def _scene(signals: list[ShotSignal]) -> str:
    """What these photographs actually show, as words an audio model can use.

    Sentence, not a tag list: `season_weather=snow` means nothing to MusicGen, "snow, a
    winter evening" does.
    """
    words: list[str] = []
    for dimension, prefix in _SCENE_DIMENSIONS:
        labels = common_labels([s.tags.get(dimension) for s in signals], limit=1)
        if labels:
            words.append(f"{prefix}{labels[0]}")
    places = common_labels(
        [s.place or (s.facets or {}).get("place_city") for s in signals], limit=1
    )
    if places:
        words.append(f"in {places[0]}")
    return f"It is under photographs of {', '.join(words)}." if words else ""
