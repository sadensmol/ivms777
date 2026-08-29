"""How energetic a memory is (design §11.1) — one word that the whole film obeys.

PURE: tags in, a label out. No model, no I/O.

It exists because the same treatment cannot serve a birthday and a hospital visit. The
mood decides three things, and they must agree with each other or the film contradicts
itself:

- **the score** — a celebration scored with warm ambient pads is the single most
  jarring thing this app can produce, and it is what a generic fallback prompt gives
  you every time;
- **which effects** are reachable — `glitch` and `pulse` belong to a party, `bokeh` and
  `breeze` to a quiet evening;
- **the PACE** — how many beats a shot gets and how fast they hand over. A fun day
  changes quickly; a sombre one is allowed to sit still.

The labels come from the taxonomy the library already produces (`vibe`, `emotion`,
`occasion`), so this adds no analysis pass — it reads what ingest already knew.
"""

from feed.signals import ShotSignal

HIGH = "high"
CALM = "calm"
LOW = "low"

# Labels that mean "this was a lively day". `occasion` is weighted double: a birthday is
# a birthday even in photographs where nobody happens to be smiling.
_HIGH_VIBE = {"festive", "playful", "energetic", "joyful", "happy", "lively", "chaotic"}
_HIGH_OCCASION = {"birthday", "party", "wedding", "celebration", "holiday", "concert",
                  "christmas", "new year", "festival", "graduation"}
_LOW_VIBE = {"melancholy", "moody", "sombre", "somber", "lonely", "tense", "gloomy"}
_LOW_EMOTION = {"sad", "tired", "bored", "anxious", "angry"}


def energy(signals: list[ShotSignal]) -> str:
    """`HIGH`, `CALM` or `LOW` for a whole memory.

    A vote across every photograph, not a single tag: one grinning face in a quiet
    afternoon does not make it a party, and one bored expression at a wedding does not
    make it a wake.
    """
    if not signals:
        return CALM
    high = low = 0
    for signal in signals:
        vibe = signal.tags.get("vibe", "")
        emotion = signal.tags.get("emotion", "")
        occasion = signal.tags.get("occasion", "")
        if occasion in _HIGH_OCCASION:
            high += 2
        if vibe in _HIGH_VIBE or emotion in {"happy", "joyful", "excited", "surprised"}:
            high += 1
        if vibe in _LOW_VIBE:
            low += 1
        if emotion in _LOW_EMOTION:
            low += 1
    if high > low and high >= len(signals):
        return HIGH
    if low > high and low >= len(signals):
        return LOW
    return CALM


# What the score should sound like when the story pass does not supply a prompt of its
# own. NOT one generic sentence: "warm ambient pads, slow, gentle" over a birthday is
# exactly the funeral-music complaint, and a fallback that only ever fits sad memories
# is a fallback that is wrong most of the time.
# Tempo is stated in BPM and instruments are named, because "warm ambient pads,
# gentle, unobtrusive" is what a text-to-audio model produces for almost any prompt —
# and it is funeral music over a birthday.
_MUSIC = {
    HIGH: ("Upbeat celebratory instrumental at about 120 BPM, bright major key, "
           "acoustic guitar, hand claps, light drums and warm brass, joyful and "
           "energetic, party atmosphere, no vocals."),
    CALM: ("Gentle cinematic instrumental at about 85 BPM, warm piano and soft "
           "strings, unhurried and hopeful, no vocals."),
    LOW: ("Slow, spare instrumental at about 60 BPM, quiet sustained strings and "
          "single piano notes, still and reflective, no percussion, no vocals."),
}
# What the occasion tag alone is worth saying out loud to the audio model.
_OCCASION_HINT = {
    "birthday": "a birthday party",
    "wedding": "a wedding",
    "christmas": "Christmas",
    "new year": "New Year's Eve",
    "party": "a party",
    "concert": "a concert",
    "graduation": "a graduation",
    "holiday": "a holiday celebration",
    "travel": "a journey",
}


def _occasion(signals: list[ShotSignal]) -> str:
    counts: dict[str, int] = {}
    for signal in signals:
        label = signal.tags.get("occasion", "")
        if label in _OCCASION_HINT:
            counts[label] = counts.get(label, 0) + 1
    if not counts:
        return ""
    return _OCCASION_HINT[min(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]]


def music_prompt(
    title: str, signals: list[ShotSignal], hint: str = "", asked: str = "",
    energy_label: str | None = None,
) -> str:
    """The score brief sent to the audio model.

    Three sources, in strict order of authority:

    1. what the VIEWER asked for (`asked`, from `[music]` directions, §11.3) — an
       instruction from the only person who was there, so it leads;
    2. the memory's own mood and occasion — measured, and not optional, because a small
       model reaches for "warm ambient pads, gentle, unobtrusive" whatever it is shown,
       which is funeral music over a birthday;
    3. the story pass's sentence (`hint`) — kept last, as colour rather than as brief.

    MusicGen has no notion of a timeline, so a direction written at 0:12 becomes a
    characteristic of the whole piece. Crude, but "a bright major chord lands here"
    still pulls the clip the way they wanted.
    """
    occasion = _occasion(signals)
    lead = f"Music for {occasion}. " if occasion else ""
    # `energy_label` is the WHOLE FILM's energy (§11.1): a score section covering three
    # quiet photographs in the middle of a party is still party music, so a section
    # brief states the film's mood rather than re-measuring its own stretch.
    label = energy_label if energy_label in _MUSIC else energy(signals)
    parts = [asked.strip(), f"{lead}{_MUSIC[label]}", hint.strip()]
    return " ".join(part for part in parts if part).strip()


# How the score is SAMPLED, per energy. MusicGen's own defaults are one setting for
# every prompt, which is a large part of why every film came back sounding like the
# same safe average: `guidance` is how hard the brief is enforced, `temperature` how far
# the model will wander from its safest note. A party is worth pushing on both; a quiet
# afternoon is not.
SAMPLING = {
    HIGH: {"guidance": 4.5, "temperature": 1.15},
    CALM: {"guidance": 3.0, "temperature": 1.0},
    LOW: {"guidance": 2.5, "temperature": 0.9},
}


# How busy the picture is allowed to be. `beats` is how many effects a shot cycles
# through; `hold` is how much of the shot the opening beat keeps before handing over —
# a smaller number means the film changes sooner.
PACE = {
    HIGH: {"beats": 3, "hold": 0.34, "handover": 0.24},
    CALM: {"beats": 3, "hold": 0.50, "handover": 0.30},
    LOW: {"beats": 2, "hold": 0.62, "handover": 0.42},
}
