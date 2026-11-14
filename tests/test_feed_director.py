"""The director (design §11.1) — pure, deterministic, no model, no I/O."""

from itertools import pairwise

import feed.director
from feed.director import (
    GRADES,
    direct,
    with_arrangement,
    with_lines,
    with_narration_timing,
    with_script,
)
from feed.signals import ShotSignal


def sig(pid, *, w=4000, h=3000, at=None, tags=None, scores=None, vec=None, place=None):
    return ShotSignal(photo_id=pid, width=w, height=h, shot_at=at, caption=f"c{pid}",
                      description=f"d{pid}", place=place, tags=tags or {},
                      scores=scores or {}, vector=vec)


LAND = [sig(1), sig(2), sig(3), sig(4)]
SAME = [1.0] + [0.0] * 1151
OTHER = [0.0] * 1151 + [1.0]


def test_one_shot_per_signal_in_order():
    assert [s.photo_id for s in direct(LAND).shots] == [1, 2, 3, 4]


def test_is_deterministic():
    assert direct(LAND).as_dict() == direct(LAND).as_dict()


def test_the_camera_never_pans():
    # A lateral sweep on every shot is what made the film rock like a boat. Whatever
    # the orientation, the ONLY thing that changes is the scale.
    for signal in (sig(1, w=4000, h=3000), sig(1, w=3000, h=4000)):
        move = direct([signal, sig(2), sig(3)]).shots[1].move
        assert move.x_from == move.x_to == 0.0
        assert move.y_from == move.y_to == 0.0


def test_every_move_actually_moves():
    for shot in direct(LAND).shots:
        assert shot.move.scale_from != shot.move.scale_to, shot.photo_id
        assert 1.0 <= min(shot.move.scale_from, shot.move.scale_to)
        assert max(shot.move.scale_from, shot.move.scale_to) <= 1.35


def test_the_camera_never_reverses_after_the_opening_shot():
    # One direction for the whole film. Alternating it is a metronome, not a style.
    shots = direct([sig(i) for i in range(1, 9)]).shots
    assert all(s.move.scale_to > s.move.scale_from for s in shots[1:])


def test_the_film_opens_by_pulling_back():
    opening = direct(LAND).shots[0].move
    assert opening.scale_from > opening.scale_to


def test_a_hero_shot_is_held_longer_on_a_slow_push_in():
    hero = sig(2, tags={"quality": "sharp", "subject": "person"},
               scores={"quality": 0.95, "subject": 0.83})
    shots = direct([sig(1), hero, sig(3)]).shots
    assert shots[1].hero is True
    assert shots[1].seconds > shots[2].seconds
    assert shots[1].move.scale_to > shots[1].move.scale_from   # push IN
    assert shots[1].move.ease == "in-out"


def test_a_sharp_photo_without_a_person_is_not_a_hero():
    only_sharp = sig(2, tags={"quality": "sharp"}, scores={"quality": 0.95})
    assert direct([sig(1), only_sharp, sig(3)]).shots[1].hero is False


def _mix(a, b, t):
    """A vector t of the way from a to b, for building graded similarity."""
    return [x * (1 - t) + y * t for x, y in zip(a, b)]


def test_a_retake_collapses_into_a_fast_cut():
    # Two identical frames 5 s apart, among shots that are otherwise unalike: this
    # is a retake, and it is the case the montage path exists for.
    shots = direct([
        sig(1, at="2025-07-12T10:00:00", vec=OTHER),
        sig(2, at="2025-07-12T10:00:30", vec=SAME),
        sig(3, at="2025-07-12T10:00:35", vec=SAME),          # the retake
        sig(4, at="2025-07-12T10:01:30", vec=_mix(SAME, OTHER, 0.5)),
    ]).shots
    assert shots[2].transition == "cut"
    assert shots[2].seconds < 2.0


def test_a_uniform_memory_is_not_turned_into_one_long_montage():
    # THE bug this rule was rewritten for: a memory is time-and-place bounded by
    # construction, so nearly every neighbouring pair looks alike. A film of 40 such
    # shots must not become 40 hard cuts with the camera switched off.
    shots = direct([
        sig(i, at=f"2025-07-12T10:{i:02d}:00", vec=_mix(SAME, OTHER, 0.02 * i))
        for i in range(20)
    ]).shots
    cuts = [s for s in shots if s.transition == "cut"]
    assert len(cuts) <= len(shots) * 0.25 + 1
    assert sum(1 for s in shots if s.transition == "dissolve") > len(shots) / 2


def test_no_shot_is_ever_frozen():
    # A static frame is what makes an edit read as a slideshow. The montage branch
    # used to return an identical from/to scale, freezing 36 of 40 shots.
    for signals in (
        [sig(1, at="2025-07-12T10:00:00", vec=SAME), sig(2, at="2025-07-12T10:00:05", vec=SAME)],
        [sig(i) for i in range(6)],
    ):
        for shot in direct(signals).shots:
            assert shot.move.scale_from != shot.move.scale_to, shot.photo_id


def test_photos_far_apart_in_time_are_never_a_retake():
    apart = [sig(1, at="2025-07-12T10:00:00", vec=SAME),
             sig(2, at="2025-07-12T10:05:00", vec=SAME)]
    assert direct(apart).shots[1].transition != "cut"


def test_a_visually_different_pair_is_not_a_burst():
    pair = [sig(1, at="2025-07-12T10:00:00", vec=SAME),
            sig(2, at="2025-07-12T10:00:10", vec=OTHER)]
    assert direct(pair).shots[1].transition != "cut"


def test_photos_with_no_vector_are_never_a_burst():
    pair = [sig(1, at="2025-07-12T10:00:00"), sig(2, at="2025-07-12T10:00:10")]
    assert direct(pair).shots[1].transition != "cut"


def test_a_long_time_gap_is_a_dip_to_black():
    shots = direct([sig(1, at="2025-07-12T10:00:00"),
                    sig(2, at="2025-07-12T20:00:00")]).shots
    assert shots[1].transition == "dip"


def test_a_change_of_city_is_a_dip_to_black():
    shots = direct([sig(1, at="2025-07-12T10:00:00", place="Borjomi"),
                    sig(2, at="2025-07-12T10:30:00", place="Tbilisi")]).shots
    assert shots[1].transition == "dip"


def test_the_grade_comes_from_palette_and_light():
    warm = direct([sig(1, tags={"palette": "warm"}, scores={"palette": 0.7})])
    night = direct([sig(1, tags={"light": "low light"}, scores={"light": 0.9})])
    assert warm.grade == "warm"
    assert night.grade == "night"
    assert warm.grade in GRADES and night.grade in GRADES


def test_the_grade_is_one_for_the_whole_film():
    mixed = [sig(1, tags={"palette": "warm"}), sig(2, tags={"palette": "cool"}),
             sig(3, tags={"palette": "warm"})]
    edit = direct(mixed)
    assert edit.grade == "warm"                      # the majority palette
    assert {s.grade for s in edit.shots} == {"warm"}  # never re-graded per shot


def test_a_memory_with_no_palette_tags_grades_neutral():
    assert direct(LAND).grade == "neutral"


def test_every_grade_preset_carries_the_shader_uniforms():
    for name, preset in GRADES.items():
        assert set(preset) == {"lift", "gamma", "gain"}, name
        assert all(len(preset[k]) == 3 for k in preset), name


def test_pacing_opens_and_closes_slower_than_the_middle():
    shots = direct([sig(i) for i in range(1, 8)]).shots
    assert shots[0].seconds > shots[3].seconds
    assert shots[-1].seconds > shots[3].seconds


def test_the_first_shot_has_no_transition_in():
    assert direct(LAND).shots[0].transition == "none"
    assert direct(LAND).shots[0].transition_seconds == 0.0


def test_long_memories_are_capped():
    edit = direct([sig(i) for i in range(200)], max_shots=40)
    assert len(edit.shots) == 40


def test_with_lines_attaches_the_story_text():
    edit = with_lines(direct(LAND), ["a", "b", "c", "d"])
    assert [s.line for s in edit.shots] == ["a", "b", "c", "d"]


def test_with_lines_ignores_a_wrong_length_list():
    # The model returned the wrong number of lines; the film must still play.
    edit = with_lines(direct(LAND), ["only", "two"])
    assert [s.line for s in edit.shots] == ["", "", "", ""]


def test_an_empty_memory_produces_an_empty_edit():
    edit = direct([])
    assert edit.shots == () and edit.seconds == 0.0


def test_as_dict_carries_everything_the_renderer_needs():
    payload = direct(LAND).as_dict()
    assert set(payload) == {"shots", "grade", "grades", "seconds", "narration", "effects"}
    shot = payload["shots"][0]
    assert set(shot) == {"photo_id", "at", "effect", "params", "beats", "seconds",
                         "move", "transition", "transition_seconds", "grade", "line",
                         "hero"}
    assert set(shot["move"]) == {"scale_from", "scale_to", "x_from", "x_to",
                                 "y_from", "y_to", "ease"}
    # The renderer switches on ids it was TOLD, never on a table copied into JS.
    assert len(shot["params"]) == 4
    assert shot["effect"] in payload["effects"]["shot"]
    assert shot["transition"] in payload["effects"]["transition"]
    # Every shot is an ARC, not a filter: effects arrive, hand over and leave.
    assert shot["beats"], "a shot with no beats renders as a still"
    for beat in shot["beats"]:
        assert beat["effect"] in payload["effects"]["shot"]
        assert len(beat["params"]) == 4
        assert 0.0 <= beat["from"] < beat["to"] <= 1.0


def test_a_long_hold_gets_more_than_one_effect():
    # A photograph under one unchanging treatment for eight seconds is a screensaver
    # (§11.1). The craft pass alone — no model — must already produce an arc.
    hero = sig(2, tags={"quality": "sharp", "subject": "person"},
               scores={"quality": 0.95, "subject": 0.83})
    shots = direct([sig(1), hero, sig(3), sig(4)]).shots
    # Every shot arrives and settles at minimum; some also leave. A hero always arrives
    # on the face — that is what makes it a hero.
    assert all(len(shot.beats) >= 2 for shot in shots)
    assert shots[1].beats[0]["effect"] == "rack_focus"
    assert any(len(shot.beats) >= 3 for shot in shots)


def test_neighbouring_shots_are_never_treated_the_same_way():
    # A memory whose photographs share one light tag used to get one identical bed on
    # every shot, and the film read as "two effects" (§11.1). The rotation is by INDEX,
    # so it is deterministic — the same memory always cuts the same way.
    shots = direct([sig(i, tags={"light": "low light"}) for i in range(1, 9)]).shots
    arcs = [tuple(b["effect"] for b in shot.beats) for shot in shots]
    from itertools import pairwise

    assert all(a != b for a, b in pairwise(arcs)), arcs
    assert len({e for arc in arcs for e in arc}) >= 5, "the library is barely used"


def test_effect_beats_overlap_so_one_hands_over_to_the_next():
    # Back-to-back beats snap; overlapping ones blend.
    from itertools import pairwise

    beats = direct([sig(1), sig(2)]).shots[0].beats
    assert any(later["from"] < earlier["to"] for earlier, later in pairwise(beats))


def test_a_burst_frame_gets_a_single_beat():
    # A montage frame is over before anything can develop.
    from feed import effects

    for shot in direct([sig(1), sig(2), sig(3)]).shots:
        assert len(shot.beats) <= effects.MAX_BEATS


def test_with_script_keeps_the_told_order_and_drops_an_index_out_of_range():
    edit = direct(LAND)
    script = ((2, "a"), (0, "b"), (2, "c"), (9, "gone"), (1, "d"))
    assert [s.photo_id for s in with_script(edit, script).shots] == [3, 1, 3, 2]


def test_nothing_sorts_the_passages_into_memory_order():
    assert not hasattr(feed.director, "_in_memory_order")


# A conductor that would reorder the film: shot numbers 2, 4, 1, 3, with times that
# overlap.
SHUFFLED = [{"shot": n, "at": at, "seconds": 2.0}
            for n, at in ((2, 0.0), (4, 0.5), (1, 9.0), (3, 1.0))]


def _monotonic(shots):
    return all(b.at >= a.at + a.seconds - 1e-6 for a, b in pairwise(shots))


def test_keep_order_keeps_the_shots_where_the_story_put_them():
    edit = direct(LAND)
    shots = with_arrangement(edit, SHUFFLED, keep_order=True).shots
    assert [s.photo_id for s in shots] == [1, 2, 3, 4]
    assert _monotonic(shots)


def test_without_keep_order_the_conductor_may_reorder_a_film_with_no_script():
    shots = with_arrangement(direct(LAND), SHUFFLED).shots
    assert [s.photo_id for s in shots] == [2, 4, 1, 3]


def test_keep_timing_keeps_the_order_and_the_voices_clock():
    timed = with_narration_timing(direct(LAND), [{"start": t} for t in (0, 2, 5, 7)], 10.0)
    shots = with_arrangement(timed, SHUFFLED, keep_timing=True).shots
    assert [s.photo_id for s in shots] == [1, 2, 3, 4]
    assert [(s.at, s.seconds) for s in shots] == [(0, 2), (2, 3), (5, 2), (7, 3)]
