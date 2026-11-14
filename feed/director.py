"""The director (design §11.1) — signals in, an `Edit` out.

PURE and DETERMINISTIC. No model, no HTTP, no database, no clock, no randomness.
Everything is derived from the shot's signals and its position, so the same memory
always cuts the same way and every rule below is a unit test.

The split this file exists to enforce: **craft is rules, story is the model**. How long
a shot holds, which way the camera travels, where the cuts land, what the film is
graded — all of that is measurable from the library and belongs here. What the film
SAYS is taste, and lives in `feed/story.py`.
"""

from dataclasses import dataclass, replace
from datetime import datetime

from feed import effects, mood
from feed.signals import ShotSignal
from search import signals as gates

# Camera bounds. Below ~1.05 the move is invisible; past ~1.35 a 1600 px thumbnail
# softens on a 1080p screen.
# Gentler since the renderer switched to CONTAIN framing: the photo is shown whole,
# so any zoom past ~1.12 starts eating into it again — which is how heads got cut off.
_REST = 1.0
_PUSH = 1.12
_HERO_PUSH = 1.07          # heroes move LESS — the slower it is, the more it reads
# The OPENING shot pulls back instead of pushing in: a reveal is how a film starts, and
# one bookend is not an oscillation.
_OPEN_PULL = 1.14

# THERE IS NO LATERAL DRIFT, AND NO ALTERNATION. Both used to exist — neighbouring
# shots deliberately travelled opposite ways, landscapes swept sideways and portraits
# vertically — and together they made the film rock like a boat: left, right, left,
# right, every few seconds, for the whole runtime. A camera that changes its mind on
# every cut is not a style, it is a metronome.
#
# What replaces them is the oldest rule in the book: ONE slow push, the same direction
# for the whole film, easing to a stop. The variation lives in HOW MUCH each shot
# moves — a hero barely at all, a montage frame a hair — not in WHICH WAY.

# A burst is the same frame shot twice — a retake, not a scene. It must stay RARE.
#
# The first version used `STRICT.image` (0.80) with a 90 s window, both borrowed from
# the similar-photos strip, and it swallowed 36 of 40 shots on a real memory. The
# mistake was the frame of reference: 0.80 is "unusually alike FOR THE WHOLE LIBRARY",
# but a memory is time-and-place bounded BY CONSTRUCTION, so almost every adjacent
# pair inside one clears it. A heuristic for finding needles was applied to a haystack
# that had already been sorted into needles.
#
# So the bar is now RELATIVE to this memory: a pair is a burst only if it is far more
# alike than that memory's own typical neighbouring pair. Plus a hard 12 s window
# (retakes are seconds apart, not minutes) and a cap on how much of a film may be
# montage, because the burst path is the one that turns the camera off.
_BURST_SECONDS = 12.0
# How far above the memory's own median neighbour-similarity a pair must sit.
_BURST_MARGIN = 0.06
# Never below this, however uniform the memory: two shots really must look alike.
_BURST_FLOOR = gates.STRICT.image
# A memory's "norm" cannot be measured from one or two samples — the median IS the
# sample, so the margin makes the bar unreachable. Below this many candidate pairs,
# fall back to a plain high bar: unmistakably-alike, no statistics claimed.
_BURST_MIN_SAMPLE = 4
_BURST_ABSOLUTE = 0.92
# At most this share of a film may be montage. Past it, the memory is simply uniform
# (one room, one evening) and deserves a paced edit, not a flick-book.
_BURST_MAX_SHARE = 0.25
_BURST_HOLD = 1.4

# A chapter break: a real gap in the day, or a different city.
_CHAPTER_GAP_SECONDS = 3 * 3600.0

_HERO_SHARP = 0.80
_HERO_SUBJECT = 0.60
_HERO_HOLD = 1.7

_DISSOLVE = effects.transition_seconds("dissolve")
_DIP = effects.transition_seconds("dip")
_CUT = effects.transition_seconds("cut")
# No shot is ever shorter than this, whatever a model asks for: below about a second a
# photograph is a flicker, not a shot.
_MIN_HOLD = 0.9

# Grade presets — what the fragment shader applies: lift (added), gamma (curve), gain
# (multiplied), per RGB channel.
GRADES: dict[str, dict] = {
    "neutral": {"lift": [0.00, 0.00, 0.00], "gamma": [1.00, 1.00, 1.00],
                "gain": [1.00, 1.00, 1.00]},
    "warm":    {"lift": [0.02, 0.01, 0.00], "gamma": [0.98, 1.00, 1.03],
                "gain": [1.06, 1.01, 0.95]},
    "cool":    {"lift": [0.00, 0.01, 0.02], "gamma": [1.03, 1.00, 0.97],
                "gain": [0.95, 1.01, 1.07]},
    "pastel":  {"lift": [0.04, 0.04, 0.05], "gamma": [1.05, 1.05, 1.05],
                "gain": [0.97, 0.97, 0.99]},
    "muted":   {"lift": [0.02, 0.02, 0.02], "gamma": [1.02, 1.02, 1.02],
                "gain": [0.93, 0.94, 0.95]},
    "night":   {"lift": [0.00, 0.01, 0.03], "gamma": [1.10, 1.08, 1.00],
                "gain": [0.88, 0.92, 1.05]},
    "bright":  {"lift": [0.01, 0.01, 0.01], "gamma": [0.95, 0.95, 0.95],
                "gain": [1.05, 1.05, 1.04]},
}


@dataclass(frozen=True)
class Move:
    scale_from: float
    scale_to: float
    x_from: float
    x_to: float
    y_from: float
    y_to: float
    ease: str          # "in-out" | "linear"


@dataclass(frozen=True)
class Shot:
    photo_id: int
    seconds: float
    move: Move
    transition: str            # a key of `effects.TRANSITIONS`
    transition_seconds: float
    grade: str
    line: str
    hero: bool
    # WHERE this shot starts, in seconds from the film's t=0 — which is also the audio's
    # t=0. Absolute, not accumulated, because the player's clock IS the soundtrack's
    # `currentTime` (§11.1): a duration-chain drifts against audio within seconds, and
    # a drifting cut is exactly the "three things playing at once" this design exists
    # to kill. The craft pass leaves these at 0 and the conductor fills them in.
    at: float = 0.0
    # Which entry of `feed/effects.py` plays over this shot, and the four numbers it
    # is driven by — a focus point, an amount, a direction. Packed positionally per
    # `effects.PARAM_ORDER`, which is the shader's only contract.
    effect: str = effects.DEFAULT_SHOT
    params: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)
    # The shot's effect SEQUENCE — `{effect, params, from, to}` over the shot's own
    # length. One treatment held for eight seconds is a screensaver; a shot is a little
    # arc, so effects arrive, hand over and leave (§11.1).
    beats: tuple[dict, ...] = ()


@dataclass(frozen=True)
class Edit:
    shots: tuple[Shot, ...]
    grade: str
    # The spoken script. It IS the film's story (§11.1), so it belongs in the edit and
    # in the JSON the player fetches — an earlier version kept it only inside the
    # narration WAV, which made the story invisible and untestable.
    narration: str = ""

    @property
    def seconds(self) -> float:
        """The film's length.

        Once the conductor has placed the shots this is simply where the last one ends;
        before that it is the sum of the craft pass's nominal holds.
        """
        if self.shots and any(shot.at for shot in self.shots):
            return round(max(shot.at + shot.seconds for shot in self.shots), 3)
        return round(sum(s.seconds + s.transition_seconds for s in self.shots), 3)

    def as_dict(self) -> dict:
        return {
            "grade": self.grade,
            "grades": GRADES,
            "seconds": self.seconds,
            "narration": self.narration,
            # Name -> shader id, so the renderer switches on a number it was TOLD,
            # never on a table duplicated in JavaScript and left to drift.
            "effects": effects.payload(),
            "shots": [
                {
                    "photo_id": s.photo_id,
                    "at": s.at,
                    # `effect`/`params` are the FIRST beat, kept so anything reading the
                    # older single-effect shape still works; `beats` is what the
                    # renderer actually plays.
                    "effect": s.effect,
                    "params": list(s.params),
                    "beats": [dict(beat) for beat in s.beats],
                    "seconds": s.seconds,
                    "transition": s.transition,
                    "transition_seconds": s.transition_seconds,
                    "grade": s.grade,
                    "line": s.line,
                    "hero": s.hero,
                    "move": {
                        "scale_from": s.move.scale_from, "scale_to": s.move.scale_to,
                        "x_from": s.move.x_from, "x_to": s.move.x_to,
                        "y_from": s.move.y_from, "y_to": s.move.y_to,
                        "ease": s.move.ease,
                    },
                }
                for s in self.shots
            ],
        }


def direct(
    signals: list[ShotSignal], *, base_seconds: float = 4.5, max_shots: int = 40
) -> Edit:
    """Turn per-shot signals into a cut film."""
    chosen = signals[:max_shots]
    if not chosen:
        return Edit(shots=(), grade="neutral")
    bursts = _burst_flags(chosen)
    grade = _grade(chosen)
    # The memory's energy governs WHICH effects are reachable and HOW FAST they hand
    # over (§11.1). A birthday and a hospital visit cannot share a pace.
    energy = mood.energy(chosen)
    shots: list[Shot] = []
    for index, signal in enumerate(chosen):
        hero = _is_hero(signal) and not bursts[index]
        seconds = _hold(index, len(chosen), base_seconds, hero=hero, burst=bursts[index])
        default_beats = _default_beats(index, signal, energy,
                                       hero=hero, burst=bursts[index])
        transition, transition_seconds = _transition(
            index, signal, chosen[index - 1] if index else None, bursts[index]
        )
        shots.append(
            Shot(
                photo_id=signal.photo_id,
                seconds=round(seconds, 3),
                move=_move(index, signal, hero=hero, burst=bursts[index]),
                transition=transition,
                transition_seconds=transition_seconds,
                grade=grade,
                line="",
                hero=hero,
                effect=default_beats[0]["effect"],
                params=tuple(default_beats[0]["params"]),
                beats=tuple(default_beats),
            )
        )
    return Edit(shots=tuple(shots), grade=grade)


def with_narration(edit: Edit, narration: str) -> Edit:
    """Attach the spoken script, so it travels with the edit rather than only inside
    the WAV."""
    return replace(edit, narration=narration)


def with_script(edit: Edit, script: tuple[tuple[int, str], ...]) -> Edit:
    """Rebuild the film's shot list from the STORY (design §11.1).

    Until now a shot was a photograph. It is not: a shot is a **passage of the story**,
    and a photograph is what illustrates it. So the same picture may appear twice
    because the story came back to it, and a picture the story never needed simply does
    not appear (§11.1). Each new shot inherits its photograph's move, grade and effect arc.

    The story's sentence order is the film's order: shot `i` is passage `i`, because
    the voice speaks the passages in that order. The memory's photo order is only
    recorded in the story pass, never imposed here, so a passage's words and its
    photograph never separate.

    An empty script leaves the edit alone — a film with no words still shows its
    pictures in library order.
    """
    if not script:
        return edit
    shots = tuple(
        replace(edit.shots[index], line="")
        for index, _text in script
        if 0 <= index < len(edit.shots)
    )
    return replace(edit, shots=shots) if shots else edit


def with_narration_timing(
    edit: Edit, segments: list[dict], total_seconds: float
) -> Edit:
    """Start every shot exactly when its own passage is spoken.

    This is the sync, and it is arithmetic rather than judgement: segment `i` of the
    narration IS the words for shot `i`, because the script's passages were handed to
    the voice as the chunks to speak. No model is asked to guess which sentence goes
    with which picture, so none can guess wrong.

    A mismatch in length means the voice did not chunk the way we asked; the edit is
    returned untouched rather than half-aligned.
    """
    usable = [s for s in segments if isinstance(s, dict) and "start" in s]
    if len(usable) != len(edit.shots) or not usable:
        return edit
    starts = [max(float(s["start"]), 0.0) for s in usable]
    span = max(total_seconds, starts[-1] + _MIN_HOLD)
    shots = []
    for index, shot in enumerate(edit.shots):
        at = starts[index]
        end = starts[index + 1] if index + 1 < len(starts) else span
        shots.append(replace(shot, at=round(at, 3),
                             seconds=round(max(end - at, _MIN_HOLD), 3)))
    return replace(edit, shots=tuple(shots))


def with_arrangement(
    edit: Edit, entries: list[dict], *, keep_timing: bool = False, keep_order: bool = False
) -> Edit:
    """Apply the conductor's plan.

    Each entry is `{shot, at, seconds, beats, transition}`.

    With `keep_order` the running order came from the STORY, so the conductor never
    reorders the shots. With `keep_timing` — the normal path — the clock also came from
    the VOICE, so the conductor only chooses the effects and how each shot arrives;
    `keep_timing` implies `keep_order`. Without either (a film with no script), `shot`
    is a 1-based index into this edit and the plan may reorder the film. Without
    `keep_timing` the conductor's times stand.

    The plan comes from a model, so nothing here trusts it:

    - a `shot` that is out of range or missing is repaired, and with `keep_order` the
      running order is already fixed by the story, so only the effects move;
    - times are forced monotonic, so two shots can never be on screen at once;
    - effect and transition names are validated against the library;
    - a plan of the wrong LENGTH is dropped whole and the craft pass stands.
    """
    if len(entries) != len(edit.shots):
        return edit
    # The STORY already fixed the running order. Letting the conductor reorder here
    # would move a shot away from the words it was written for.
    order = (list(range(len(edit.shots))) if keep_order or keep_timing
             else _resolve_order(entries, len(edit.shots)))
    shots: list[Shot] = []
    cursor = 0.0
    for position, (source, entry) in enumerate(zip(order, entries)):
        shot = edit.shots[source]
        transition = "none" if position == 0 else effects.transition(
            entry.get("transition"), fallback=shot.transition
        )
        # Monotonic by construction: a start earlier than where the previous shot ends
        # is a model error, and letting it through would play two shots at once.
        # When the narration already placed this shot, its timing is not up for
        # discussion: the voice IS the clock (§11.1).
        if keep_timing:
            at, seconds = shot.at, shot.seconds
        else:
            at = max(_number(entry.get("at"), cursor), cursor)
            seconds = max(_number(entry.get("seconds"), shot.seconds), _MIN_HOLD)
        effect = effects.shot_effect(entry.get("effect"), fallback=shot.effect)
        # The plan may give a whole SEQUENCE; `entry["effect"]` is only the fallback for
        # a model that answered in the older one-effect shape.
        beats = effects.pack_beats(
            entry.get("beats"), effect, dict(zip(effects.PARAM_ORDER[effect],
                                                 shot.params)))
        shots.append(replace(
            shot,
            at=round(at, 3),
            seconds=round(seconds, 3),
            effect=beats[0]["effect"],
            params=tuple(beats[0]["params"]),
            beats=tuple(beats),
            transition=transition,
            transition_seconds=effects.transition_seconds(transition),
        ))
        cursor = at + seconds
    return replace(edit, shots=tuple(shots))


def _resolve_order(entries: list[dict], total: int) -> list[int]:
    """The 0-based source index for each position, every photo used exactly once.

    A small model asked for a permutation will sometimes return a duplicate or skip a
    number. Rather than reject the whole plan for that, the duplicates are dropped and
    the unused photographs fill the holes in their original order — the arrangement it
    meant survives, and no memory loses a picture.
    """
    seen: set[int] = set()
    resolved: list[int | None] = []
    for entry in entries:
        try:
            index = int(entry.get("shot", 0)) - 1
        except (TypeError, ValueError):
            index = -1
        if 0 <= index < total and index not in seen:
            seen.add(index)
            resolved.append(index)
        else:
            resolved.append(None)
    spare = [i for i in range(total) if i not in seen]
    return [index if index is not None else spare.pop(0) for index in resolved]


def _number(value, fallback: float) -> float:
    """A float from whatever the model put in the field. Models emit "3.5", null and
    the occasional "3.5s"; none of those may take the film down."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)


def with_lines(edit: Edit, lines: list[str]) -> Edit:
    """Attach the story pass's per-shot lines.

    A wrong-length list is DROPPED rather than zipped short: a half-captioned film
    reads as a bug, whereas a wordless-but-moving one reads as a choice. The model is
    asked for exactly one line per shot; when it disobeys, the film still plays.
    """
    if len(lines) != len(edit.shots):
        return edit
    return replace(
        edit,
        shots=tuple(replace(s, line=line) for s, line in zip(edit.shots, lines)),
    )


def _is_hero(signal: ShotSignal) -> bool:
    """Sharp AND containing a person. Either alone is not a hero: a tack-sharp photo
    of a wall is not the shot to hold on."""
    sharp = signal.scores.get("quality", 0.0) if signal.tags.get("quality") == "sharp" else 0.0
    person = signal.scores.get("subject", 0.0) if signal.tags.get("subject") == "person" else 0.0
    return sharp >= _HERO_SHARP and person >= _HERO_SUBJECT


def _hold(index: int, total: int, base: float, *, hero: bool, burst: bool) -> float:
    if burst:
        return _BURST_HOLD
    if hero:
        return base * _HERO_HOLD
    if index == 0:
        return base * 1.35            # open slow: let the first frame land
    if index == total - 1:
        return base * 1.45            # close slow: let it end, don't just stop
    return base * 0.85                # tighten through the middle


def _move(index: int, signal: ShotSignal, *, hero: bool, burst: bool) -> Move:
    """One push, always inward, always eased. Never a pan (see the camera bounds).

    `signal` is unchanged input for now — orientation no longer picks an axis, because
    there is no lateral axis left to pick.
    """
    if burst:
        # A montage frame is brief, so the move is small — but it is NEVER static.
        # A frozen frame is what makes an edit read as a slideshow, and this branch
        # used to freeze every shot it touched.
        return Move(_REST, _REST + 0.03, 0.0, 0.0, 0.0, 0.0, "linear")
    if hero:
        # The shot the film is ABOUT. It moves least, so the face stays still and the
        # length does the work.
        return Move(_REST, _HERO_PUSH, 0.0, 0.0, 0.0, 0.0, "in-out")
    if index == 0:
        # Open by pulling back into the frame — the one shot that travels outward.
        return Move(_OPEN_PULL, _REST, 0.0, 0.0, 0.0, 0.0, "in-out")
    return Move(_REST, _PUSH, 0.0, 0.0, 0.0, 0.0, "in-out")


# The craft pass ROTATES through the library by shot index. Deterministic — no
# randomness, the same memory always cuts the same way — but consecutive shots never
# get the same treatment. Without this, a memory whose photographs share one light tag
# received one identical bed on every shot, and the film read as "two effects": a bit
# of light and a bit of movement.
# Which effects a memory may reach for, BY ENERGY. `glitch` and `pulse` belong to a
# party; over a quiet or sad day they are simply wrong, and no amount of good timing
# rescues them. This is the same judgement `feed/mood.py` applies to the score, so the
# picture and the music always agree about what kind of day it was.
# The no-model film. These pools are what the viewer sees when the conductor never
# runs, so every family in the library is represented here too — an effect that only a
# working model can reach is an effect most films never show.
_ARRIVALS: dict[str, tuple[str, ...]] = {
    mood.HIGH: ("glitch", "rack_focus", "pixelate", "bloom", "zoom_punch", "weave",
                "flash", "prism"),
    mood.CALM: ("rack_focus", "weave", "bloom", "freeze_ripple", "rack_focus",
                "halftone"),
    mood.LOW:  ("rack_focus", "weave", "soft_focus", "rack_focus", "film_burn"),
}
_DEPARTURES: dict[str, tuple[str | None, ...]] = {
    # A lively day leaves on something every time; a sad one mostly just settles.
    mood.HIGH: ("light_leak", "pulse", "anamorphic_flare", "light_leak", "pulse",
                "strobe_cut"),
    mood.CALM: ("light_leak", None, "pulse", None, "god_rays", None),
    mood.LOW:  (None, "light_leak", None, "spotlight", None, None),
}
_BEDS_LOW: tuple[str, ...] = ("bokeh", "breeze", "weave", "echo", "bokeh", "spotlight")
_BEDS_BRIGHT: tuple[str, ...] = ("bloom", "breeze", "weave", "god_rays", "bloom",
                                 "parallax")
_BEDS_PLAIN: tuple[str, ...] = ("breeze", "weave", "bloom", "dolly_swing", "breeze",
                                "tilt")
_BRIGHT_LIGHT = ("bright", "backlit", "golden hour", "harsh midday")


def _default_beats(
    index: int, signal: ShotSignal, energy: str, *, hero: bool, burst: bool
) -> list[dict]:
    """The craft pass's own effect ARC for one shot, from measurable signals alone.

    This is what the film looks like when no model runs — the story pass can fail and
    the edit must still be worth watching. The conductor replaces it shot by shot.

    The shape is always the same and it is the reason a long hold stays alive: a beat
    that ARRIVES (something happens at the cut), a bed that runs UNDER the whole shot,
    and often a beat that LEAVES (something happens on the way out). Overlapping, so
    each hands over to the next instead of snapping.
    """
    light = signal.tags.get("light")
    # The bed suits the PHOTOGRAPH; which of the suitable ones is used rotates, so two
    # neighbours are never treated the same way.
    if light == "low light":
        beds = _BEDS_LOW
    elif light in _BRIGHT_LIGHT:
        beds = _BEDS_BRIGHT
    else:
        beds = _BEDS_PLAIN
    bed = beds[index % len(beds)]

    if burst:
        # A montage frame is over before anything can develop, so it gets one thing.
        return effects.pack_beats([{"effect": "weave", "from": 0.0, "to": 1.0}], "weave")

    arrivals, departures = _ARRIVALS[energy], _DEPARTURES[energy]
    pace = mood.PACE[energy]
    arrival = arrivals[index % len(arrivals)]
    if hero:
        # The shot the film is ABOUT always arrives on the face — that is the whole
        # reason it is a hero.
        arrival = "rack_focus"
    if arrival == bed:                      # never hand over to yourself
        arrival = "rack_focus" if bed != "rack_focus" else "weave"
    departure = departures[index % len(departures)] if pace["beats"] >= 3 else None

    # A lively memory hands over EARLY and often; a sombre one is allowed to sit still.
    hold, handover = pace["hold"], pace["handover"]
    beats = [
        {"effect": arrival, "from": 0.0, "to": hold},
        {"effect": bed, "from": hold - handover,
         "to": 1.0 if departure is None else 0.86},
    ]
    if departure is not None:
        beats.append({"effect": departure, "from": 0.74, "to": 1.0,
                      "params": {"strength": 0.32} if departure == "light_leak"
                      else {"depth": 0.28}})
    return effects.pack_beats(beats, bed)


def _transition(
    index: int, signal: ShotSignal, previous: ShotSignal | None, burst: bool
) -> tuple[str, float]:
    if index == 0:
        return "none", 0.0
    if burst:
        return "cut", _CUT
    if previous is not None and _is_chapter_break(previous, signal):
        return "dip", _DIP
    return "dissolve", _DISSOLVE


def _is_chapter_break(previous: ShotSignal, signal: ShotSignal) -> bool:
    if previous.place and signal.place and previous.place != signal.place:
        return True
    gap = _gap_seconds(previous.shot_at, signal.shot_at)
    return gap is not None and gap > _CHAPTER_GAP_SECONDS


def _burst_flags(signals: list[ShotSignal]) -> list[bool]:
    """Mark retakes — pairs far more alike than this memory's own norm.

    BOTH members of a matching pair are marked, so a three-shot burst becomes three
    fast cuts rather than one slow shot followed by two fast ones.

    Two guards keep this rare. The bar is relative to the memory's OWN median
    neighbour similarity, so a uniform memory does not become one long montage; and
    the result is capped at `_BURST_MAX_SHARE` of the film, keeping only the most
    alike pairs when more qualify. Without the cap, the montage path — which turns the
    camera off — could still claim the whole edit.
    """
    pairs: list[tuple[float, int]] = []
    for index in range(1, len(signals)):
        previous, current = signals[index - 1], signals[index]
        gap = _gap_seconds(previous.shot_at, current.shot_at)
        if gap is None or gap > _BURST_SECONDS:
            continue
        cosine = _cosine(previous.vector, current.vector)
        if cosine > 0:
            pairs.append((cosine, index))
    flags = [False] * len(signals)
    if not pairs:
        return flags
    if len(pairs) < _BURST_MIN_SAMPLE:
        bar = _BURST_ABSOLUTE
    else:
        scores = sorted(cosine for cosine, _ in pairs)
        median = scores[len(scores) // 2]
        bar = max(median + _BURST_MARGIN, _BURST_FLOOR)
    qualifying = sorted((c for c in pairs if c[0] >= bar), reverse=True)
    budget = max(1, int(len(signals) * _BURST_MAX_SHARE))
    for _, index in qualifying[:budget]:
        flags[index - 1] = flags[index] = True
    return flags


def _grade(signals: list[ShotSignal]) -> str:
    """ONE grade for the whole film — a film that re-grades every shot looks broken,
    not styled. Low light wins outright; otherwise the most common palette does."""
    lows = sum(1 for s in signals if s.tags.get("light") == "low light")
    if lows * 2 > len(signals):
        return "night"
    counts: dict[str, int] = {}
    for signal in signals:
        palette = signal.tags.get("palette")
        if palette in GRADES:
            counts[palette] = counts.get(palette, 0) + 1
    if not counts:
        return "neutral"
    # Most common, and by name on a tie, so the choice is deterministic.
    return min(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]


def _gap_seconds(first: str | None, second: str | None) -> float | None:
    if not first or not second:
        return None
    try:
        return abs((_parse(second) - _parse(first)).total_seconds())
    except ValueError:
        return None


def _parse(stamp: str) -> datetime:
    # `fromisoformat` reads a trailing Z natively since 3.11, which is the floor here.
    return datetime.fromisoformat(stamp)


def _cosine(first: list[float] | None, second: list[float] | None) -> float:
    if not first or not second or len(first) != len(second):
        return 0.0
    dot = sum(a * b for a, b in zip(first, second))
    norm = (sum(a * a for a in first) ** 0.5) * (sum(b * b for b in second) ** 0.5)
    return dot / norm if norm else 0.0
