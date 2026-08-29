"""What a viewer can ask for in a note, as a LEGEND — not as a parser (design §11.3).

A note may be plain prose, an instruction in square brackets, or both in one line:
`[страшным хриплым голосом] это был вечер перед Хэллоуином`.

**Nothing here reads them.** There was a matcher: a table of phrases and word stems that
classified each bracket and routed it. It is gone, and it was wrong in a way no amount of
table-filling fixes — `[страшных хриплый мужкской голос]` fell straight through it,
because it knew "хриплый голос" and another word sat between the two. People write
instructions in their own words, in their own order, in either language, with typos. A
table can only ever recognise the phrasings someone thought to add to it.

So notes go to the models VERBATIM (`feed/notes.py`), and each pass is told that
brackets are instructions to obey rather than sentences to narrate. The story pass reads
them and casts the narrator, any per-passage voice, and the score; the conductor reads
them and places effects and lengths. Interpreting "a scary hoarse man's voice" into a
speaker is judgement, and judgement is what a model is for.

What stays here is the LEGEND: examples shown on screen so a viewer knows the brackets
do something and roughly what can be asked for. Examples, not a vocabulary — anything
sensible works, because a model is reading it.

PURE data. No model, no I/O, no matching.
"""

from feed import effects

VOICE = "voice"
MUSIC = "music"
EFFECT = "effect"
TIMING = "timing"


def legend() -> list[tuple[str, str, list[str]]]:
    """`(category, what it does, example phrasings)` for the on-screen legend.

    The effect names come from the library itself, so the legend cannot advertise an
    effect the renderer does not have. Everything else is an EXAMPLE: the model reads
    the bracket, so a phrasing that is not listed works just as well.
    """
    return [
        (VOICE, "how the narrator reads this part", [
            "low man's voice", "низкий мужской голос", "страшным хриплым голосом",
            "whisper", "шёпот", "excited", "sad voice",
        ]),
        (MUSIC, "what the score does here", [
            "fun chord", "весёлая музыка", "drums", "барабаны", "swell", "big finish",
        ]),
        (EFFECT, "how the picture is treated here",
            sorted(effects.SHOT_EFFECTS)[:6] + ["slide", "dip to black", "засветка"]),
        (TIMING, "how long this part takes", [
            "hold 5s", "держать 5с", "quick", "pause", "skip", "title",
        ]),
    ]
