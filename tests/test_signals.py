"""The similarity evidence model (§9) — gates, the entry ramp, and noisy-OR."""
import math

import pytest

from search import signals

# --- cosine gates + entry ramp -------------------------------------------------

def test_below_gate_is_absent_not_zero():
    # "absent" is None, so a caller skips the contribution entirely rather than
    # scoring a 0 — the graceful-degradation rule (§9.2).
    assert signals.cosine_strength(0.79, gate=0.80) is None
    assert signals.cosine_strength(0.0, gate=0.80) is None


def test_crossing_the_gate_is_worth_half_immediately():
    # The whole point of the ramp: a pair sitting AT the gate must not score ~0,
    # or admitting it buys nothing (design §9).
    assert signals.cosine_strength(0.80, gate=0.80) == pytest.approx(0.5)
    assert signals.cosine_strength(0.75, gate=0.75) == pytest.approx(0.5)


def test_ramp_reaches_full_at_one():
    assert signals.cosine_strength(1.0, gate=0.80) == pytest.approx(1.0)
    assert signals.cosine_strength(0.90, gate=0.80) == pytest.approx(0.75)


def test_impossible_gate_silences_the_signal():
    # The ablation switch (§9.3) passes a gate no cosine can reach.
    assert signals.cosine_strength(1.0, gate=signals.SIGNAL_OFF) is None


# --- tags ----------------------------------------------------------------------

def test_tag_below_its_gate_is_absent():
    # A `subject` the model only half-believes cannot qualify a pair — this is
    # exactly the mis-tag that made a Christmas tree "similar" to a teddy bear.
    assert signals.tag_strength(0.58, idf=0.41, gate=0.80) is None


def test_tag_keeps_a_floor_of_its_weight_when_common():
    # idf 0 (a label on every photo) must not zero the tag — it damps it to the
    # floor, so a common-but-confident match still reranks.
    assert signals.tag_strength(1.0, idf=0.0, gate=0.5) == pytest.approx(signals.IDF_FLOOR)
    assert signals.tag_strength(1.0, idf=1.0, gate=0.5) == pytest.approx(1.0)


def test_tag_strength_scales_with_agreement():
    assert signals.tag_strength(0.5, idf=1.0, gate=0.5) == pytest.approx(0.5)


# --- moment (time x place) ------------------------------------------------------

def test_same_instant_same_spot_is_a_perfect_moment():
    assert signals.moment_strength(0.0, 0.0) == pytest.approx(1.0)


def test_moment_decays_with_time_and_distance():
    same_hour_same_block = signals.moment_strength(1.0, 200.0)
    same_afternoon_nearby = signals.moment_strength(3.0, 500.0)
    next_day_across_town = signals.moment_strength(24.0, 5000.0)
    assert same_hour_same_block > same_afternoon_nearby > next_day_across_town
    assert same_hour_same_block > signals.STRICT.moment
    assert next_day_across_town < signals.STRICT.moment


def test_moment_without_gps_falls_back_to_time_alone():
    # Place unknown -> the signal is time-only and deliberately capped, because
    # "same hour" without "same place" is a much weaker claim.
    with_place = signals.moment_strength(1.0, 0.0)
    without_place = signals.moment_strength(1.0, None)
    assert without_place == pytest.approx(with_place * signals.MOMENT_NO_GPS)


def test_moment_uses_the_documented_decay_constants():
    assert signals.moment_strength(signals.MOMENT_TAU_HOURS, None) == pytest.approx(
        math.exp(-1.0) * signals.MOMENT_NO_GPS
    )


# --- noisy-OR composition -------------------------------------------------------

def test_no_evidence_scores_zero():
    assert signals.combine([]) == 0.0


def test_one_signal_scores_itself():
    assert signals.combine([0.6]) == pytest.approx(0.6)


def test_a_pile_of_weak_signals_cannot_beat_one_strong_one():
    # The bug this replaces: five faint style tags used to out-sum a real match.
    strong = signals.combine([0.74])
    pile = signals.combine([0.05] * 6)
    assert strong > pile


def test_score_is_bounded_and_monotonic():
    assert signals.combine([0.9, 0.9, 0.9]) < 1.0
    assert signals.combine([0.5, 0.3]) > signals.combine([0.5])


def test_combine_ignores_absent_signals():
    assert signals.combine([0.5, None, 0.0]) == pytest.approx(0.5)


# --- tier table -----------------------------------------------------------------

def test_every_taxonomy_dimension_has_exactly_one_tier():
    dimensions = [d for tier in signals.TAG_TIERS.values() for d in tier.dimensions]
    assert len(dimensions) == len(set(dimensions))


def test_subject_is_the_only_content_tag_tier():
    content = [name for name, tier in signals.TAG_TIERS.items() if tier.content]
    assert content == ["subject"]


def test_weights_follow_the_agreed_order():
    order = [
        signals.IMAGE_WEIGHT,
        signals.TAG_TIERS["subject"].weight,
        signals.CAPTION_WEIGHT,
        signals.TAG_TIERS["where"].weight,
        signals.MOMENT_WEIGHT,
        signals.TAG_TIERS["look"].weight,
        signals.TAG_TIERS["quality"].weight,
    ]
    assert order == sorted(order, reverse=True)


def test_a_lone_moment_belongs_behind_show_more():
    # `moment` is a CONTENT signal, so a visually different photo from the same time
    # and place always QUALIFIES. But measured on the real library, a moment-only
    # pair is a tire close-up matching a plastic container photographed the same
    # minute — same outing, unrelated object. It is worth offering, not worth putting
    # in the default strip, so it clears the LOOSE floor and not the STRICT one (§9).
    strength = signals.moment_strength(1.0, 200.0)
    lone = signals.combine([
        signals.MOMENT_WEIGHT * signals.cosine_strength(strength, signals.STRICT.moment)
    ])
    assert lone < signals.STRICT.score_min
    assert lone >= signals.LOOSE.score_min


def test_dimension_weights_expand_from_tier_weights():
    weights = signals.dimension_weights({"subject": 1.0, "where": 0.5})
    assert weights["subject"] == 1.0
    assert weights["setting"] == 0.5 and weights["occasion"] == 0.5
    # An unnamed tier keeps its built-in default.
    assert weights["palette"] == signals.TAG_TIERS["look"].weight


def test_the_subject_gate_is_separate_from_every_other_tag_gate():
    assert signals.STRICT.for_dimension("subject") == signals.STRICT.subject
    assert signals.STRICT.for_dimension("emotion") == signals.STRICT.tag
    assert signals.STRICT.for_dimension("setting") == signals.STRICT.tag


def test_loose_relaxes_every_gate_but_never_below_the_noise_floor():
    # "Looser" must still mean "measured", not "made up": a cosine gate under its
    # random-pair median would be scoring chance. Those medians are 0.558 (image)
    # and 0.621 (caption) on the reference library.
    for field in ("image", "caption", "moment", "subject", "tag", "score_min"):
        assert getattr(signals.LOOSE, field) < getattr(signals.STRICT, field), field
    assert signals.LOOSE.image > 0.558
    assert signals.LOOSE.caption > 0.621


def test_loose_and_strict_share_identical_weights():
    # Only gates relax. Changing weights would reshuffle results that are already
    # ranked below the strict ones and make the two passes incomparable.
    assert not hasattr(signals.LOOSE, "weight")
    assert signals.TAG_TIERS["subject"].weight == 0.75
