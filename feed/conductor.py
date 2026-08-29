"""The conductor (design §11.1) — ONE model call that lines the whole film up.

The director owns craft that is measurable: how long a shot can hold, what a burst is,
which grade the palette asks for. The conductor owns the thing that is not measurable —
**where every shot, effect and cut belongs relative to the music and the voice.** That
is an editing decision made while looking at all the facts at once, and it is exactly
the kind of judgement a model is better at than a rule.

So this module does not arrange anything itself. It BRIEFS the model with everything
the machine already knows —

- what is in each photograph, and how long the director thinks it can hold;
- **where every narration sentence starts and ends**, to the millisecond, from the
  voice model's own segment timings;
- **where every musical downbeat falls**, from `feed/beats.py`, plus the tempo;
- the full effect library with what each one feels like and when to use it;

— and asks for one plan: for each shot, when it starts, how long it lasts, which effect
plays over it with which parameters, and how it arrives.

Then it REPAIRS the answer. A model that returns nine entries for ten shots, a start
time before the previous shot ended, or an effect it invented, must not cost the film.
Validation lives in `director.with_arrangement`; the fallback plan lives here.

NEVER raises. If the call fails, `plain_arrangement` lays the shots evenly across the
narration and snaps them to the beat grid — which is already better than the fixed
timer this replaced.
"""

import json
import logging

from feed import effects
from feed.beats import snap
from feed.director import Edit

logger = logging.getLogger(__name__)

# How far a cut may be dragged to reach a downbeat. Beyond this the cut is no longer
# the one that was planned, and sync stops serving the edit and starts overruling it.
SNAP_WINDOW = 0.35

# Mirrors `director._MIN_HOLD`: below about a second a photograph is a flicker, not a
# shot, so the arrangement reserves this much for every shot still to come.
_MIN_HOLD = 0.9
# A film shorter than this is not a film. Guards the arithmetic when a caller passes 0.
_MIN_RUNTIME = 3.0

PLAN_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "shots": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "shot": {"type": "integer"},
                    "at": {"type": "number"},
                    "seconds": {"type": "number"},
                    "transition": {"type": "string"},
                    "beats": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "effect": {"type": "string"},
                                "params": {"type": "object"},
                                "from": {"type": "number"},
                                "to": {"type": "number"},
                            },
                            "required": ["effect", "from", "to"],
                        },
                    },
                },
                "required": ["shot", "at", "seconds", "beats", "transition"],
            },
        },
    },
    "required": ["shots"],
}


def arrange(
    client, model: str, *, title: str, description: str, edit: Edit,
    shots: list[str], segments: list[dict], bars: list[float], bpm: float,
    total_seconds: float, digest: str = "", directions: str = "",
    energy: str | None = None,
) -> list[dict]:
    """The plan, one entry per shot, already snapped to the beat grid.

    Returns `plain_arrangement(...)` when the model cannot be reached or answers with
    the wrong number of shots — a film always gets a timeline.
    """
    fallback = plain_arrangement(edit, bars, total_seconds)
    total_seconds = max(float(total_seconds or 0.0), _MIN_RUNTIME)
    if not edit.shots:
        return fallback
    try:
        raw = client.text_complete(
            model,
            [{"role": "user", "content": _brief(
                title, description, edit, shots, segments, bars, bpm, total_seconds,
                digest, directions, energy)}],
            json_schema=PLAN_SCHEMA,
            # Low, not zero: this is a judgement call, but a wandering one produces a
            # film that cuts in the wrong places.
            temperature=0.4,
            max_tokens=3000,
        )
        plan = _loads(raw).get("shots")
    except Exception:
        logger.exception("conductor call failed; falling back to an even arrangement")
        return fallback
    if not isinstance(plan, list) or len(plan) != len(edit.shots):
        logger.warning("conductor returned %s entries for %d shots; using the fallback",
                       len(plan) if isinstance(plan, list) else type(plan).__name__,
                       len(edit.shots))
        return fallback
    # NOT sorted by `shot`: the array order IS the running order, and `shot` says which
    # photograph plays there. Sorting would throw away the arrangement — the whole
    # reason for asking a model rather than a rule (§11.1).
    return _snap_all(plan, bars, total_seconds)


def plain_arrangement(edit: Edit, bars: list[float], total_seconds: float) -> list[dict]:
    """An even, beat-snapped timeline using the craft pass's own choices.

    This is the film with NO model in it: the shots share the runtime in proportion to
    the holds the director computed, every boundary is pulled onto the nearest downbeat,
    and each shot keeps the effect its signals earned.
    """
    if not edit.shots:
        return []
    weights = [shot.seconds for shot in edit.shots]
    span = max(total_seconds, sum(weights) * 0.5) or float(len(edit.shots))
    scale = span / sum(weights)
    plan: list[dict] = []
    cursor = 0.0
    for index, shot in enumerate(edit.shots):
        seconds = shot.seconds * scale
        plan.append({
            "shot": index + 1,
            "at": cursor,
            "seconds": seconds,
            "effect": shot.effect,
            "params": list(shot.params),
            "beats": [dict(beat) for beat in shot.beats],
            "transition": shot.transition,
        })
        cursor += seconds
    return _snap_all(plan, bars, total_seconds)


def _snap_all(plan: list[dict], bars: list[float], total_seconds: float) -> list[dict]:
    """Pull every start onto a downbeat, then re-derive the durations from the starts.

    Durations are DERIVED rather than snapped independently, so no rounding can leave a
    gap of black or an overlap between two shots.

    **The film ends WITH its soundtrack.** The last shot is stretched or trimmed to land
    exactly on `total_seconds`, and no start is allowed past it. A model asked to fill
    sixty seconds will happily return a plan that runs to ninety, and a film that
    outlives its own audio plays its final shot in silence — which looks like a hang.
    """
    span = max(float(total_seconds or 0.0), _MIN_RUNTIME)
    snapped: list[dict] = []
    for entry in plan:
        moved = dict(entry)
        try:
            at = float(entry.get("at", 0.0))
        except (TypeError, ValueError):
            at = 0.0
        moved["at"] = round(snap(at, bars, window=SNAP_WINDOW), 3)
        snapped.append(moved)
    if not snapped:
        return snapped
    # The film always starts at zero — a beat-snapped first shot would open on silence.
    snapped[0]["at"] = 0.0
    # Forced monotonic and inside the runtime, leaving room for every shot still to
    # come: clamping without that room would pile the tail onto one instant.
    count = len(snapped)
    for index in range(1, count):
        ceiling = span - (count - index) * _MIN_HOLD
        snapped[index]["at"] = round(
            min(max(snapped[index]["at"], snapped[index - 1]["at"] + _MIN_HOLD),
                max(ceiling, snapped[index - 1]["at"] + _MIN_HOLD)), 3)
    for index in range(count - 1):
        snapped[index]["seconds"] = round(
            snapped[index + 1]["at"] - snapped[index]["at"], 3)
    snapped[-1]["seconds"] = round(max(span - snapped[-1]["at"], _MIN_HOLD), 3)
    return snapped


def _brief(title, description, edit, shots, segments, bars, bpm, total_seconds,
           digest="", directions="", energy=None) -> str:
    """Everything the machine knows, written for an editor who cannot see the film.

    The shot lines are `ShotSignal.brief()` — the full description, every taxonomy
    label, the place, the time, the camera settings. All of it is here so the model can
    answer the one question that matters: WHICH photograph is this sentence about?
    """
    shot_lines = "\n".join(
        f"{index + 1}. [{'HERO' if shot.hero else 'shot'}, holds well for about "
        f"{shot.seconds:.1f}s] "
        f"{shots[index] if index < len(shots) and shots[index] else '(no description)'}"
        for index, shot in enumerate(edit.shots)
    )
    voice_lines = "\n".join(
        f"  {seg['start']:.2f}s – {seg['end']:.2f}s: {seg['text']}"
        for seg in segments
    ) or "  (no narration timings available)"
    beat_line = (
        ", ".join(f"{bar:.2f}" for bar in bars[:48]) + ("…" if len(bars) > 48 else "")
        if bars else "(no music, so no beat grid — pace it on the narration alone)"
    )
    return (
        "You are editing a short film made from one person's own photographs. The "
        "pictures, the music and the narration already exist. Your ONLY job is to line "
        "them up so they feel like one thing rather than three.\n\n"
        f"FILM: {title}\n"
        f"ABOUT: {description}\n"
        f"RUNTIME: {total_seconds:.1f} seconds. Tempo: {bpm:.0f} BPM.\n\n"
        + (f"WHAT THIS MEMORY IS, from the library's own labels — the weather, the "
           f"hour and the occasion are what an effect should MATCH:\n{digest}\n\n"
           if digest else "")
        + 
        f"THE PHOTOGRAPHS available to you, numbered. Each line is what is IN the "
        f"picture, then its labels, where and when it was taken, and how it was shot:\n"
        f"{shot_lines}\n\n"
        "WHAT THE VOICE SAYS, AND EXACTLY WHEN. **This is the spine of the edit.** "
        "Whatever the narration is describing at a given second, THAT is the "
        "photograph that must be on screen at that second — if the voice says 'the "
        "cat', show the cat, not the car. Read each sentence, find the picture it is "
        "about, and put it there. A cut belongs in the GAP between two sentences, "
        "never in the middle of one:\n"
        f"{voice_lines}\n\n"
        "WHERE THE MUSIC'S DOWNBEATS FALL (seconds). Cuts that land on these feel "
        "intentional; cuts that land between them feel like a mistake:\n"
        f"  {beat_line}\n\n"
        + (f"### WHAT THE PERSON WHOSE MEMORY THIS IS ASKED FOR\n"
           f"Written while watching an earlier cut, each marked with the moment it "
           f"applies to. These are explicit INSTRUCTIONS and they OUTRANK your own "
           f"judgement and every rule below — everything else in this brief is "
           f"inference, and this is not.\n"
           f"- ANYTHING IN [SQUARE BRACKETS] IS AN INSTRUCTION. It may be in either "
           f"language and any wording — read it and do what it asks. The rest of the "
           f"note is what they said about the memory; it is not for you.\n"
           f"- An EFFECT they named goes on the shot covering that moment, with "
           f"whatever parameters suit it. Do not substitute a different one because it "
           f"would look better.\n"
           f"- A LENGTH they named ('hold 5s') sets that shot's `seconds`, and the "
           f"shots around it absorb the difference.\n"
           f"- 'skip' means that photograph does not appear; 'title' means put a "
           f"title card on it.\n"
           f"{directions}\n\n" if directions else "")
        + f"{effects.catalogue_for_prompt(energy)}\n\n"
        "Return JSON: {\"shots\": [{\"shot\": <1-based number>, \"at\": <seconds "
        "from the start of the film>, \"seconds\": <how long it holds>, "
        "\"transition\": <one name from the transition list>, \"beats\": [{"
        "\"effect\": <one name from the list>, \"params\": {<its parameters>}, "
        "\"from\": <0-1>, \"to\": <0-1>}]}]}\n\n"
        "ABOUT `beats` — THIS IS WHAT MAKES A LONG SHOT WATCHABLE. A photograph held "
        "for eight seconds under one unchanging treatment is a screensaver. So each "
        f"shot gets a SEQUENCE of up to {effects.MAX_BEATS} beats, timed as fractions "
        "of that shot's own length: `from: 0, to: 0.4` runs over the first 40% of it.\n"
        "- OVERLAP them. A beat that ends at 0.5 and one that starts at 0.35 hand over "
        "smoothly; back-to-back beats snap.\n"
        f"- At most {effects.MAX_ACTIVE} play at once — one thing you notice and one "
        "thing you feel. More than that and the frame stops being a photograph.\n"
        "- The shape that works: something ARRIVES at the cut (rack_focus, glitch), a "
        "bed runs UNDER most of the shot (breeze, weave, bloom, bokeh), and on a long "
        "hold something LEAVES near the end (light_leak, pulse).\n"
        "- A SHORT shot (under ~3s) wants one beat. A LONG shot (over ~6s) needs three "
        "or it will feel stuck.\n\n"
        "RULES:\n"
        f"- Exactly {len(edit.shots)} entries. The ORDER of the array is the running "
        f"order of the film; `shot` says which photograph plays there. You may put the "
        f"photographs in ANY order — that is the main thing you are for.\n"
        f"- Use every photograph exactly once.\n"
        f"- Cover the whole {total_seconds:.1f} seconds with no gaps and no overlaps.\n"
        "- The first entry starts at 0.\n"
        "- MATCH THE PICTURE TO THE WORDS. This matters more than every other rule "
        "here: a beautiful effect on the wrong photograph is a worse film than a plain "
        "cut on the right one.\n"
        "- Prefer starts that sit ON a downbeat listed above.\n"
        "- Hold a shot while the voice is describing it.\n"
        "- Vary the effects BETWEEN shots as well as within them. Repeating one for "
        "every shot is worse than using none, "
        "and the strong ones (glitch, burn, whip, light_leak) lose all their power if "
        "used more than once or twice in the whole film.\n"
        "- Match the effect to the PHOTOGRAPH and to the moment, using the "
        "'use when' and 'avoid when' notes. A pulse over a quiet moment, or a bloom on "
        "a flat grey interior, is a worse choice than plain.\n"
        f"- For `pulse`, set bpm to {bpm:.0f}.\n"
        "- No shot shorter than 1 second."
    )


def _loads(raw: str) -> dict:
    """Parse the model's JSON, tolerating the ```json fence small models emit."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1] if "\n" in text else text
        text = text.rsplit("```", 1)[0]
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return {}
        payload = json.loads(text[start : end + 1])
    return payload if isinstance(payload, dict) else {}
