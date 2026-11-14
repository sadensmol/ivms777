"""Which speaker narrates a memory (design §11.1) — pure, no model.

Every name asserted here was verified to load and produce audio before it was wired in:
narration is mandatory, so an unknown speaker fails the film rather than degrading it.
"""

import pathlib

import pytest

from feed import mood
from feed.signals import ShotSignal
from feed.voice import offered, pick_speaker

# The speakers each model actually ships, read from the loaded models.
SILERO = {"aidar", "baya", "kseniya", "xenia", "eugene", "random"}
KOKORO = {"af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky",
          "am_adam", "am_michael", "am_puck", "bf_emma", "bm_george"}


def sig(pid, **tags):
    return ShotSignal(photo_id=pid, width=1, height=1, shot_at=None, caption="",
                      description="", place=None, tags=tags, scores={}, vector=None)


PARTY = [sig(i, occasion="birthday", vibe="festive", emotion="happy") for i in range(1, 6)]
GRIEF = [sig(i, vibe="melancholy", emotion="sad") for i in range(1, 6)]
QUIET = [sig(i, vibe="calm") for i in range(1, 6)]


@pytest.mark.parametrize("language,known", [("ru", SILERO), ("en", KOKORO)])
def test_every_offered_speaker_exists_in_the_model(language, known):
    for band in [[sp.name for sp in offered(language)]]:
        for name in band:
            assert name in known, f"{name} is not a real {language} speaker"


def test_the_register_follows_the_day():
    # A birthday must not be read in the register of a hospital visit.
    assert pick_speaker("ru", PARTY) != pick_speaker("ru", GRIEF)
    assert pick_speaker("en", PARTY) != pick_speaker("en", GRIEF)


def test_two_different_memories_do_not_share_one_narrator():
    # Every film in the library used to come out in the same voice, because almost
    # every memory routed to one default.
    first = [sig(i, vibe="calm") for i in range(1, 6)]
    second = [sig(i, vibe="calm") for i in range(20, 25)]
    assert pick_speaker("en", first) != pick_speaker("en", second)


def test_the_same_memory_always_gets_the_same_voice():
    # The director is pure by contract: re-preparing must not change the narrator.
    assert pick_speaker("ru", QUIET) == pick_speaker("ru", list(QUIET))


def test_an_unknown_language_lets_the_model_default():
    # Guessing a speaker name would make the voice model raise, and narration is
    # mandatory (§11.1) — so a wrong guess would fail the whole film.
    assert pick_speaker("de", QUIET) is None


def test_an_empty_memory_still_gets_a_speaker():
    assert pick_speaker("ru", []) in SILERO


def test_the_voice_and_the_score_agree_about_the_day():
    # Both are chosen from the SAME energy, so a film can never be read brightly over
    # a sombre score, or the reverse.
    assert mood.energy(PARTY) == mood.HIGH
    assert pick_speaker("ru", PARTY) in [sp.name for sp in offered("ru") if sp.register == "bright"]
    assert mood.energy(GRIEF) == mood.LOW
    assert pick_speaker("ru", GRIEF) in [sp.name for sp in offered("ru") if sp.register == "deep"]


def test_the_speaker_table_comes_from_the_catalog():
    # No list of names lives in `feed/voice.py`: a voice added to the catalog is
    # castable the moment it is declared (CLAUDE.md § "A model's own metadata").
    import feed.voice as module

    source = pathlib.Path(module.__file__).read_text()
    for name in ("aidar", "xenia", "am_michael", "af_bella"):
        assert name not in source, f"{name} is hardcoded in voice.py"


def test_a_cast_is_checked_against_the_catalog():
    # A name the model invented raises inside the TTS child, and narration is mandatory
    # (§11.1), so it never gets that far.
    from feed.voice import resolve

    assert resolve("ru", "aidar") == "aidar"
    assert resolve("ru", "  AIDAR  ") == "aidar"      # forgiving about form
    assert resolve("ru", "count_dracula") is None      # not about existence
    assert resolve("ru", "am_michael") is None         # right name, wrong language
    assert resolve("de", "aidar") is None


def test_the_prompt_lists_every_speaker_with_what_it_sounds_like():
    from feed.voice import catalogue_for_prompt, offered

    text = catalogue_for_prompt("ru")
    for speaker in offered("ru"):
        assert speaker.name in text
        assert speaker.gender in text and speaker.register in text
    assert "low and rough" in text                      # the note, so it can judge


def test_there_is_no_keyword_reader_left():
    # A matcher lived here and needed the phrase written its way; `[страшных хриплый
    # мужкской голос]` fell straight through it (§11.3). The model reads the bracket.
    import feed.voice as module

    assert not hasattr(module, "override_speaker")
    # The keyword tables themselves, not the comment explaining why they went.
    for gone in ("_ASKS_LOW", "_ASKS_HIGH", "_ASKS_MALE", "_ASKS_FEMALE"):
        assert not hasattr(module, gone), gone


def test_offered_never_yields_a_voice_copy_entry(monkeypatch):
    # A cloning entry has no preset speakers; even one declared FIRST, and with speakers,
    # is never what a preset film casts from.
    import dataclasses

    from models import catalog

    clone = next(entry for entry in catalog.CATALOG if entry.clones)
    decoy = dataclasses.replace(clone, key="decoy-clone", speakers=(
        catalog.Speaker("decoy", "female", "warm", "not a preset"),))
    monkeypatch.setattr(catalog, "CATALOG", (decoy, *catalog.CATALOG))
    for language in ("en", "ru"):
        names = {speaker.name for speaker in offered(language)}
        assert names and "decoy" not in names
    assert {s.name for s in offered("en")} <= KOKORO
    assert {s.name for s in offered("ru")} <= SILERO


def test_a_saved_voice_resolves_by_its_name_or_its_token():
    from feed.voice import SavedCast, resolve_saved

    cast = [SavedCast("saved:1", "Mum"), SavedCast("saved:2", "Дедушка Коля")]
    assert resolve_saved("Mum", cast) == "saved:1"
    assert resolve_saved("  mUM ", cast) == "saved:1"              # case-folded
    assert resolve_saved("ДЕДУШКА КОЛЯ", cast) == "saved:2"         # Cyrillic too
    assert resolve_saved("saved:2", cast) == "saved:2"              # the token itself
    assert resolve_saved("saved:3", cast) is None                  # not in this cast
    assert resolve_saved("af_heart", cast) is None                 # a preset never
    assert resolve_saved("", cast) is None and resolve_saved(None, cast) is None


def test_the_prompt_lists_each_saved_voice_on_its_own_line():
    from feed.voice import SavedCast, automatic_for_prompt

    text = automatic_for_prompt("en", [SavedCast("saved:1", "Mum"), SavedCast("saved:2", "Dad")])
    assert text.splitlines()[-2:] == [
        "  Mum — a saved voice: this person's own voice",
        "  Dad — a saved voice: this person's own voice",
    ]


def test_a_preset_is_labelled_with_its_gender_and_register():
    from feed.voice import VoiceLabel, voice_label

    assert voice_label("af_heart", "en") == VoiceLabel("af_heart", "af_heart (female, warm)")


def test_a_saved_voice_of_the_cast_is_labelled_with_its_name():
    from feed.voice import SavedCast, VoiceLabel, voice_label

    cast = [SavedCast("saved:1", "Grandma")]
    assert voice_label("saved:1", "en", cast) == VoiceLabel("Grandma", "Grandma (saved voice)")


def test_a_saved_token_outside_the_cast_reads_a_saved_voice():
    from feed.voice import SavedCast, VoiceLabel, voice_label

    expected = VoiceLabel("a saved voice", "a saved voice")
    assert voice_label("saved:3", "en") == expected
    assert voice_label("saved:3", "en", [SavedCast("saved:1", "Grandma")]) == expected


@pytest.mark.parametrize("name, language", [("narrator_x", "en"), ("af_heart", "ru")])
def test_a_name_the_language_does_not_offer_is_labelled_as_it_is(name, language):
    from feed.voice import VoiceLabel, voice_label

    assert voice_label(name, language) == VoiceLabel(name, name)


@pytest.mark.parametrize("empty", [None, ""])
def test_no_name_has_no_label(empty):
    from feed.voice import voice_label

    assert voice_label(empty, "en") is None


SAVED_VOICES_LINE = (
    "  SAVED VOICES — the user's own recordings of real people. How each one sounds is not "
    "described. Choose one when it suits the memory, for example when the memory is about "
    "that person or the notes ask for them. Never mix the two kinds: a saved narrator takes "
    "saved passage voices only, a preset narrator presets only.")


def test_automatic_with_no_saved_voice_shows_the_presets_alone():
    from feed.voice import automatic_for_prompt, catalogue_for_prompt

    assert automatic_for_prompt("en", []) == catalogue_for_prompt("en")


def test_automatic_shows_the_presets_then_the_saved_voices_by_name():
    from feed.voice import SavedCast, automatic_for_prompt, catalogue_for_prompt

    text = automatic_for_prompt("en", [SavedCast("saved:1", "Grandma")])
    assert text.splitlines() == [
        *catalogue_for_prompt("en").splitlines(),
        SAVED_VOICES_LINE,
        "  Grandma — a saved voice: this person's own voice",
    ]
