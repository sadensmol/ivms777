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
