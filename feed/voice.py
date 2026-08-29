"""Which speaker narrates a memory, and which narrates one passage (design §11.1, §11.3).

The LANGUAGE picks the model — that routing lives in the models service, and the user
never chooses it. This is about the speaker WITHIN that model.

**No list of names lives here.** Every speaker is declared in `models/catalog.py` beside
the model that offers it, with its gender, its register and what it sounds like
(CLAUDE.md § "A model's own metadata is DECLARED"). This module reads that table, so a
voice added to the catalog is castable the moment it is declared, and a name that is not
in the catalog cannot be reached from here at all.

Casting is the STORY PASS's job: it is shown the same table and picks the film's
narrator, plus a different voice for any passage the viewer asked to sound different.
What lives here is the two things a model must not be trusted with —

- `resolve()`, the gate: a cast the model invented would raise inside the TTS child and
  fail the whole film, because narration is mandatory (§11.1);
- `pick_speaker()`, the fallback: what the film sounds like when no model runs at all,
  derived from the memory's own mood.

There is deliberately **no keyword reader** for the viewer's `[voice]` brackets. One
lived here and it needed the phrase written its way — `[страшных хриплый мужкской
голос]` fell straight through it, because another word sat between the two it knew.
People do not write to a table. The model reads the bracket itself, in whatever
language and word order it was typed, and casts from the speakers above.

PURE: catalog and signals in, a name out. No model, no I/O.
"""

from feed import mood
from feed.signals import ShotSignal
from models import catalog


def offered(language: str) -> tuple[catalog.Speaker, ...]:
    """Every speaker that can read `language`, as declared beside its model."""
    for entry in catalog.CATALOG:
        if entry.slot == "voice" and language in entry.languages:
            return entry.speakers
    return ()


def resolve(language: str, name: str | None) -> str | None:
    """`name` if it is a real speaker for this language, else None.

    The gate between a model's CHOICE and the voice service. A cast the story pass
    invented would raise inside the TTS model and fail the whole film, so it is checked
    against the catalog before it can get that far.
    """
    if not name:
        return None
    wanted = str(name).strip().lower()
    for speaker in offered(language):
        if speaker.name.lower() == wanted:
            return speaker.name
    return None


def catalogue_for_prompt(language: str) -> str:
    """The speakers as the story pass sees them, so it can cast the film itself."""
    return "\n".join(
        f"  {s.name} — {s.gender}, {s.register}: {s.note}" for s in offered(language)
    )


def _by(language: str, **wanted: str) -> tuple[str, ...]:
    """Speaker names matching every given field, in catalog order."""
    return tuple(
        speaker.name
        for speaker in offered(language)
        if all(getattr(speaker, field) == value for field, value in wanted.items())
    )


def _band(language: str, energy: str) -> tuple[str, ...]:
    """The fallback's shortlist for a mood, derived from the catalog's own `register` —
    so there is no second list of names to keep in step with the first."""
    if energy == mood.HIGH:
        return _by(language, register="bright") or _by(language, register="warm")
    if energy == mood.LOW:
        return _by(language, register="deep") or _by(language, register="neutral")
    return _by(language, register="warm") or _by(language, register="neutral")


def pick_speaker(language: str, signals: list[ShotSignal]) -> str | None:
    """The film's narrator when nothing cast one — derived from the memory itself.

    Two things choose it, both deterministic, so the same memory is always read by the
    same voice:

    - the memory's **energy** (`feed/mood.py`) picks the register, so the voice agrees
      with the score, which is chosen from the same signal;
    - the memory's own **photographs** pick which voice of that register, so two quiet
      afternoons do not sound like the same narrator reading twice.

    None for an unknown language on purpose: a wrong speaker name makes the voice model
    raise, and narration is mandatory (§11.1), so guessing would fail the whole film.
    """
    if not offered(language):
        return None
    options = _band(language, mood.energy(signals)) or tuple(
        speaker.name for speaker in offered(language)
    )
    return options[_seed(signals) % len(options)]


def _seed(signals: list[ShotSignal]) -> int:
    """A stable number for one memory, from the photographs it is made of.

    Deterministic and content-derived rather than random: re-preparing a film must not
    change its narrator, and the director is pure by contract (§11.1).
    """
    return sum(signal.photo_id for signal in signals) + len(signals)
