"""`memory_presentations` — one prepared feed per memory (design §6, §11.1)."""

import pytest

from albums.memory_store import Memory, group_id_for_album_key, replace_memories
from feed.store import (
    FilmNarrator,
    Presentation,
    film_narrator,
    film_states,
    prune_orphan_audio,
    read_presentation,
    save_presentation,
)
from storage.local import LocalStorage
from tests.factories import add_photo


@pytest.fixture
def memory_group(conn):
    for pid in (1, 2, 3):
        add_photo(conn, photo_id=pid, content_hash=f"h{pid}", thumb_key=f"{pid}.jpg",
                  caption="x", shot_at=f"2025-07-12T1{pid}:00:00")
    replace_memories(conn, 1, [Memory("Beach day", "Warm afternoon.", [1, 2, 3], "sig")])
    return conn.execute("SELECT id FROM groups WHERE kind = 'memory'").fetchone()["id"]


def _presentation(group_id: int, **over) -> Presentation:
    fields = {
        "group_id": group_id,
        "language": "en",
        "signature": "sig",
        "timeline": {"title": "Beach day", "shots": [{"photo_id": 1}]},
        "music_key": "aa/bb/aabb.wav",
        "narration_key": "cc/dd/ccdd.wav",
        "status": "ready",
        "error": None,
    }
    fields.update(over)
    return Presentation(**fields)


def test_save_then_read_round_trips(conn, memory_group):
    save_presentation(conn, _presentation(memory_group))
    stored = read_presentation(conn, memory_group, "en")
    assert stored.timeline["title"] == "Beach day"
    assert stored.music_key == "aa/bb/aabb.wav"
    assert stored.status == "ready"


def test_read_returns_none_when_never_prepared(conn, memory_group):
    assert read_presentation(conn, memory_group, "en") is None


def test_saving_twice_replaces_rather_than_duplicates(conn, memory_group):
    save_presentation(conn, _presentation(memory_group))
    save_presentation(conn, _presentation(memory_group, signature="sig2"))
    assert read_presentation(conn, memory_group, "en").signature == "sig2"
    n = conn.execute("SELECT count(*) AS n FROM memory_presentations").fetchone()["n"]
    assert n == 1


def test_rebuilding_memories_cascades_the_presentation_away(conn, memory_group):
    # §11's atomic swap deletes the groups; the presentation must go with them, or a
    # new memory would inherit a stale feed keyed by a recycled row id.
    save_presentation(conn, _presentation(memory_group))
    replace_memories(conn, 1, [Memory("Other", "d", [1], "sig2")])
    n = conn.execute("SELECT count(*) AS n FROM memory_presentations").fetchone()["n"]
    assert n == 0


def test_group_id_for_album_key_is_the_inverse_of_album_key_for_group(conn, memory_group):
    assert group_id_for_album_key(conn, 1, "memory-0") == memory_group
    assert group_id_for_album_key(conn, 1, "memory-9") is None


def test_group_id_for_album_key_is_owner_scoped(conn, memory_group):
    assert group_id_for_album_key(conn, 2, "memory-0") is None


def test_prune_orphan_audio_deletes_only_unreferenced_blobs(conn, memory_group, tmp_path):
    audio = LocalStorage(tmp_path / "audio")
    audio.write("aa/bb/aabb.wav", b"kept")
    audio.write("ee/ff/eeff.wav", b"orphan")
    save_presentation(conn, _presentation(memory_group, narration_key=None))
    assert prune_orphan_audio(conn, audio) == 1
    assert audio.exists("aa/bb/aabb.wav")
    assert not audio.exists("ee/ff/eeff.wav")


def test_the_same_memory_can_hold_one_presentation_per_language(conn, memory_group):
    save_presentation(conn, _presentation(memory_group, language="en"))
    save_presentation(conn, _presentation(memory_group, language="ru"))
    assert read_presentation(conn, memory_group, "en").language == "en"
    assert read_presentation(conn, memory_group, "ru").language == "ru"
    n = conn.execute("SELECT count(*) AS n FROM memory_presentations").fetchone()["n"]
    assert n == 2


def test_a_rebuild_drops_every_language(conn, memory_group):
    save_presentation(conn, _presentation(memory_group, language="en"))
    save_presentation(conn, _presentation(memory_group, language="ru"))
    replace_memories(conn, 1, [Memory("Other", "d", [1], "sig2")])
    n = conn.execute("SELECT count(*) AS n FROM memory_presentations").fetchone()["n"]
    assert n == 0


def test_film_states_lists_only_languages_with_a_row_in_language_order(conn, memory_group):
    assert film_states(conn, memory_group) == {}
    save_presentation(conn, _presentation(memory_group, language="ru"))
    assert list(film_states(conn, memory_group)) == ["ru"]
    save_presentation(conn, _presentation(memory_group, language="en"))
    assert list(film_states(conn, memory_group)) == ["en", "ru"]


@pytest.mark.parametrize("status, narration_key, error, ready", [
    ("ready", "cc/dd/ccdd.wav", None, True),
    ("ready", None, None, False),
    ("failed", None, "The voice model refused.", False),
    ("running", None, None, False),
])
def test_film_states_reads_the_stored_row(conn, memory_group, status, narration_key, error, ready):
    save_presentation(conn, _presentation(memory_group, status=status, narration_key=narration_key,
                                          error=error))
    state = film_states(conn, memory_group)["en"]
    assert (state.language, state.status, state.ready, state.error) == ("en", status, ready, error)


def test_film_states_carries_the_narrator_of_the_stored_cast(conn, memory_group):
    decisions = {"language": "en", "cast": {"narrator_used": "af_heart", "narrator_chosen_by": "user"}}
    save_presentation(conn, _presentation(memory_group, decisions=decisions))
    assert film_states(conn, memory_group)["en"].narrator == FilmNarrator(
        "af_heart", "af_heart (female, warm)", "user", ())


def _decisions(language="en", **cast) -> dict:
    return {"language": language, "cast": {"narrator_used": "af_heart", **cast}}


def test_the_stored_name_and_label_win():
    narrator = film_narrator(_decisions(narrator_used="saved:7", narrator_name="Grandma",
                                        narrator_label="Grandma (saved voice)"))
    assert (narrator.name, narrator.label) == ("Grandma", "Grandma (saved voice)")


@pytest.mark.parametrize("chosen_by, asked_was_real", [("user", False), ("story", True), ("mood", True)])
def test_who_chose_the_voice_reads_back(chosen_by, asked_was_real):
    narrator = film_narrator(_decisions(narrator_chosen_by=chosen_by, narrator_asked_was_real=asked_was_real))
    assert narrator.chosen_by == chosen_by


def test_a_story_choice_the_story_did_not_really_ask_for_is_the_mood():
    narrator = film_narrator(_decisions(narrator_chosen_by="story", narrator_asked_was_real=False))
    assert narrator.chosen_by == "mood"


@pytest.mark.parametrize("cast, chosen_by", [
    ({"narrator_asked_was_real": True}, "story"),
    ({"narrator_asked_was_real": False}, "mood"),
    ({}, "auto"),
])
def test_a_film_that_did_not_record_who_chose_reads_the_flag(cast, chosen_by):
    assert film_narrator(_decisions(**cast)).chosen_by == chosen_by


def test_an_old_preset_film_is_labelled_from_the_catalog():
    narrator = film_narrator(_decisions())
    assert (narrator.name, narrator.label) == ("af_heart", "af_heart (female, warm)")


def test_an_old_saved_voice_film_reads_a_saved_voice():
    narrator = film_narrator(_decisions(narrator_used="saved:3"))
    assert (narrator.name, narrator.label) == ("a saved voice", "a saved voice")


@pytest.mark.parametrize("decisions", [
    None,
    {},
    {"language": "en"},
    {"language": "en", "cast": {"narrator_used": None}},
])
def test_no_recorded_narrator_is_none(decisions):
    assert film_narrator(decisions) is None


def test_other_voices_list_only_passages_cast_to_another_voice():
    narrator = film_narrator(_decisions(
        model_cast=["", "bm_george", "", "am_adam"],
        passages=[
            {"passage": 1, "asked": "bm_george", "used": "af_heart"},     # rejected or gated: the narrator read it
            {"passage": 2, "asked": "am_adam", "used": "am_adam", "used_label": "am_adam (male, neutral)"},
            {"passage": 4, "asked": "bm_george", "used": "bm_george"},   # an old row: no used_label
            {"passage": 5, "asked": "nobody", "used": None},
        ],
    ))
    assert narrator.others == ((3, "am_adam (male, neutral)"), (5, "bm_george (male, deep)"))


SAVED_CAST = {"narrator_used": "saved:3", "narrator_name": "me", "narrator_label": "me (saved voice)",
              "narrator_chosen_by": "user"}


def test_a_saved_voice_film_names_the_voice_model_that_spoke():
    decisions = {"language": "en", "cast": {**SAVED_CAST, "narrator_model": "voice model v3"},
                 "voice": {"saved": True, "models": {"saved:3": {"version": 3, "sha256": "ab", "state": "loaded"}}}}
    assert film_narrator(decisions).model == "voice model v3"


def test_a_saved_voice_film_made_before_voice_models_says_so():
    decisions = {"language": "en", "cast": SAVED_CAST,
                 "voice": {"saved": True, "conditioning": {"saved:3": "built"}}}
    assert film_narrator(decisions).model == "made from one take, before voice models"


@pytest.mark.parametrize("voice", [
    {"saved": False, "model": "kokoro"},
    None,
])
def test_a_preset_film_names_no_voice_model(voice):
    decisions = _decisions(narrator_chosen_by="user")
    if voice is not None:
        decisions["voice"] = voice
    assert film_narrator(decisions).model is None


def test_a_saved_voice_film_that_failed_before_it_spoke_names_no_voice_model():
    # The cast is stored before the voice turn; `decisions["voice"]` only after the speak.
    assert film_narrator({"language": "en", "cast": SAVED_CAST}).model is None
