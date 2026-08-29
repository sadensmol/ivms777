"""The score plan (design §11.1) — pure: signals and passage times in, section briefs
out. No model, no I/O."""

import itertools

from feed.mood import CALM, HIGH
from feed.score import ROLES, ScoreSection, plan_sections
from feed.signals import ShotSignal


def sig(pid, **tags):
    return ShotSignal(photo_id=pid, width=4000, height=3000, shot_at=None,
                      caption=f"caption {pid}", description=f"description {pid}",
                      place=tags.pop("place", None), tags=tags, scores={}, vector=None,
                      facets=tags.pop("facets", None))


# Ten photographs: the first half a bright afternoon indoors, the second half snow.
SIGNALS = ([sig(i, light="daylight", setting="cafe") for i in range(1, 6)]
           + [sig(i, season_weather="snow", light="low light") for i in range(6, 11)])
# One passage per photograph, one second apart, so the arithmetic is obvious.
PASSAGES = [(float(i), i) for i in range(10)]


def test_a_short_film_gets_two_sections():
    sections = plan_sections(SIGNALS, PASSAGES[:3], runtime=24.0, energy=CALM)
    assert len(sections) == 2


def test_a_longer_film_gets_three():
    sections = plan_sections(SIGNALS, PASSAGES, runtime=60.0, energy=CALM)
    assert len(sections) == 3


def test_a_long_film_gets_four():
    sections = plan_sections(SIGNALS, PASSAGES, runtime=120.0, energy=CALM)
    assert len(sections) == 4


def test_the_sections_cover_the_whole_runtime_without_a_gap():
    sections = plan_sections(SIGNALS, PASSAGES, runtime=60.0, energy=CALM)
    assert sections[0].at == 0.0
    assert abs(sum(s.seconds for s in sections) - 60.0) < 0.01
    for earlier, later in itertools.pairwise(sections):
        assert abs(earlier.at + earlier.seconds - later.at) < 0.01


def test_every_boundary_lands_where_a_passage_starts():
    # A section that begins mid-sentence is a cut the audience hears as a mistake.
    sections = plan_sections(SIGNALS, PASSAGES, runtime=10.0, energy=CALM)
    starts = {at for at, _ in PASSAGES}
    assert all(s.at in starts for s in sections[1:])


def test_each_section_describes_its_own_stretch_of_the_memory():
    sections = plan_sections(SIGNALS, PASSAGES, runtime=10.0, energy=CALM)
    assert "snow" in sections[-1].brief
    assert "snow" not in sections[0].brief


def test_the_roles_run_from_opening_to_ending():
    sections = plan_sections(SIGNALS, PASSAGES, runtime=120.0, energy=CALM)
    assert [s.role for s in sections] == list(ROLES)


def test_the_last_section_ends_rather_than_vamping():
    sections = plan_sections(SIGNALS, PASSAGES, runtime=60.0, energy=CALM)
    assert "ending" in sections[-1].brief.lower()


def test_every_brief_names_a_tempo():
    # "warm ambient pads, gentle" is what a text-to-audio model returns for anything.
    sections = plan_sections(SIGNALS, PASSAGES, runtime=60.0, energy=HIGH)
    assert all("BPM" in s.brief for s in sections)


def test_the_energy_reaches_the_brief():
    calm = plan_sections(SIGNALS, PASSAGES, runtime=60.0, energy=CALM)
    high = plan_sections(SIGNALS, PASSAGES, runtime=60.0, energy=HIGH)
    assert calm[0].brief != high[0].brief


def test_the_models_plan_is_used_where_it_is_sane():
    sections = plan_sections(SIGNALS, PASSAGES, runtime=60.0, energy=CALM,
                             model_plan=((0, "solo cello"), (5, "brass, loud")))
    assert "solo cello" in sections[0].brief
    assert any("brass, loud" in s.brief for s in sections[1:])


def test_the_films_own_sentence_colours_a_section_the_plan_said_nothing_about():
    sections = plan_sections(SIGNALS, PASSAGES, runtime=60.0, energy=CALM,
                             model_plan=((0, "solo cello"),), hint="a music box")
    assert "solo cello" in sections[0].brief          # the plan wins where it spoke
    assert "a music box" in sections[-1].brief        # the film's sentence fills in


def test_the_models_plan_decides_where_the_music_turns():
    sections = plan_sections(SIGNALS, PASSAGES, runtime=60.0, energy=CALM,
                             model_plan=((0, "solo cello"), (7, "brass")))
    assert sections[1].at == 7.0


def test_a_plan_pointing_past_the_end_is_repaired_not_obeyed():
    sections = plan_sections(SIGNALS, PASSAGES, runtime=60.0, energy=CALM,
                             model_plan=((0, "cello"), (99, "brass")))
    assert 2 <= len(sections) <= 4
    assert all(s.seconds > 0 for s in sections)


def test_an_empty_plan_falls_back_to_the_measured_memory():
    assert plan_sections(SIGNALS, PASSAGES, runtime=60.0, energy=CALM, model_plan=())


def test_a_film_with_no_passages_still_gets_a_score():
    sections = plan_sections(SIGNALS, [], runtime=45.0, energy=CALM)
    assert abs(sum(s.seconds for s in sections) - 45.0) < 0.01


def test_a_memory_with_no_photographs_still_gets_a_score():
    sections = plan_sections([], [], runtime=20.0, energy=CALM)
    assert sections and all(s.brief for s in sections)


def test_no_section_is_too_short_to_be_music():
    # Four sections over twelve seconds is four stings, not a score.
    sections = plan_sections(SIGNALS, PASSAGES, runtime=12.0, energy=CALM)
    assert all(s.seconds >= 5.0 for s in sections)


def test_a_very_long_film_does_not_generate_for_ever():
    from feed.score import SCORE_MAX_SECONDS

    sections = plan_sections(SIGNALS, PASSAGES, runtime=600.0, energy=CALM)
    assert sum(s.seconds for s in sections) <= SCORE_MAX_SECONDS


def test_sections_are_frozen():
    import dataclasses

    import pytest

    section = plan_sections(SIGNALS, PASSAGES, runtime=60.0, energy=CALM)[0]
    assert isinstance(section, ScoreSection)
    with pytest.raises(dataclasses.FrozenInstanceError):
        section.brief = "no"
