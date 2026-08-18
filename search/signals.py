"""The similarity evidence model (§9) — every signal, its gate, its weight, and
the one rule that composes them.

Each signal answers the same question — *how much does this one fact argue that two
photos belong together?* — and answers it in the SAME unit: an **evidence** value in
`[0, w]`, where `w` is the most that signal may ever claim on its own. Evidence is
then composed with a **noisy-OR** (`combine`), so the final score is a true 0–1 that
the UI can print as a percentage.

Two numbers describe a signal:

* **gate** — the admission bar. Below it the signal is **absent** (`None`), never a
  zero: an absent signal is skipped for that one candidate and never kills it (the
  §9.2 graceful-degradation ADR).
* **weight** — the cap. A perfect image match alone asserts 0.95; a perfect `quality`
  match alone asserts 0.03.

**Why cosines get an entry ramp.** SigLIP and text-embedding cosines do not start at
0 — measured over all 21 115 pairs of the 206-photo reference library, two RANDOM
photos already score 0.558 (image) and 0.621 (caption). So a cosine's gate must sit
well above that noise, and — this is the part that is easy to get wrong — a pair
sitting just ABOVE the gate must not score ~0, or admitting it buys nothing. The ramp
makes crossing the gate instantly worth **half** the signal's weight, rising to full
at a cosine of 1.0.

**The rule for choosing a gate: never below the library's random-pair median.** The
bug this model replaces broke exactly that — `similar_caption_min` was 0.60, *under*
the 0.621 noise median, so 64 % of all random pairs cleared it and the caption signal
was scoring pure chance.

Tag confidences need no ramp: they are already softmax probabilities spanning 0–1.
They are damped by `idf` instead, which is the tag world's equivalent noise floor.
"""

import math
from dataclasses import dataclass

# --- the seven signals ---------------------------------------------------------
# Ordered by weight, highest first. See docs/retrieval.md#similar-photo-scoring for
# the measured distribution behind every gate.

IMAGE_GATE = 0.80       # top 4% of pairs; random pair = 0.558
IMAGE_WEIGHT = 0.95

CAPTION_GATE = 0.75     # top 5% of pairs; random pair = 0.621
CAPTION_WEIGHT = 0.65

MOMENT_GATE = 0.20      # ~top 4% of pairs (see MOMENT_* below)
MOMENT_WEIGHT = 0.35

# A text query has no seed image vector, so its rank in the semantic+keyword fusion
# stands in for it (§9.2). Deliberately LIGHTER than the image signal: a fused rank
# says "this matched the query somehow", not "these two photos are alike", and the
# rank fraction moves in big steps on short candidate lists. At the image signal's
# 0.95 a one-place difference in the fusion swamped a confirmed `subject` match,
# which is backwards — the tag is the harder evidence. Below `subject` (0.75) so a
# confirmed tag can lift a candidate past a slightly better-ranked one.
RANK_WEIGHT = 0.60

# A cosine no real pair can reach — the ablation switch (§9.3). Passed as a gate it
# silences a signal in BOTH halves (recall union and scoring) without a flag.
SIGNAL_OFF = 2.0


@dataclass(frozen=True)
class TagTier:
    """One weight shared by several taxonomy dimensions.

    Per-dimension weights were collapsed into tiers because ten hand-tuned numbers
    could not be reasoned about, and `idf` already handles the part that actually
    varies — how rare a specific label is. Gates live in `Gates` below, not here, so
    a whole profile can be relaxed at once.
    """

    dimensions: tuple[str, ...]
    weight: float
    content: bool


TAG_TIERS: dict[str, TagTier] = {
    # What the photo IS — the only tag tier that can qualify a pair.
    "subject": TagTier(("subject",), 0.75, content=True),
    # WHERE / WHICH event. Reliable enough to reorder, never to qualify.
    "where": TagTier(("setting", "occasion"), 0.40, content=False),
    # How it LOOKS. Measured worst: `emotion` clears 0.5 on 28% of photos and
    # `surprised` lands on half the library, objects included.
    "look": TagTier(
        ("light", "season_weather", "palette", "vibe", "composition", "emotion"),
        0.12, content=False,
    ),
    # Sharpness says almost nothing about similarity — a pure tiebreak. `sharp` sits
    # on 201/206 photos, so idf reduces it to ~0.01 evidence in practice.
    "quality": TagTier(("quality",), 0.03, content=False),
}

# How much of a tag's weight survives when the label is on EVERY photo. Without a
# floor, idf would zero a common-but-confident match instead of merely damping it.
IDF_FLOOR = 0.4

# `moment` — one continuous stretch of time in one place. Named for Apple Photos'
# term, and it slots under what this app already has: a MEMORY is a curated story, a
# MOMENT is a single outing. Two photos of completely different things, taken minutes
# apart in the same spot, are part of the same experience — which is what a photo
# library is actually for.
#
# Measured on the reference library (184 photos carry both EXIF time and GPS, over 25
# distinct ~1 km places): the median pair is 886 hours and 119 km apart, and only
# 3.75% of pairs fall within 1 h / 200 m. Closeness in BOTH time and place is
# therefore about as rare as an image cosine of 0.80 — hence a comparable weight.
MOMENT_TAU_HOURS = 12.0   # time constant: same afternoon still counts, next day does not
MOMENT_SCALE_M = 1000.0   # distance constant: same block counts, across town does not
MOMENT_NO_GPS = 0.5       # place unknown -> time alone, capped: a much weaker claim

@dataclass(frozen=True)
class Gates:
    """One complete admission profile — every gate, plus the final score floor.

    Two profiles exist. **`STRICT`** is what the similar strip shows by default:
    every gate sits in the top few per cent of its measured distribution, so a result
    is there because something real matched. **`LOOSE`** is what the "Show more"
    button reveals — the same signals and the same weights, admitted at lower bars.

    Weights are deliberately IDENTICAL in both. Relaxing gates is what finds more
    photos; changing weights would only reshuffle results that are already ranked
    below the strict ones, and would make the two passes incomparable.

    `LOOSE` still obeys the one hard rule — no cosine gate may sit below its
    random-pair median (image 0.558, caption 0.621), because under that line the
    signal is measuring chance, not similarity. That is what makes "looser" honest
    rather than "made up".
    """

    image: float
    caption: float
    moment: float
    subject: float
    tag: float        # where / look / quality share one bar
    score_min: float

    def for_dimension(self, dimension: str) -> float:
        """The confidence a shared tag in `dimension` needs to count at all."""
        return self.subject if dimension in content_dimensions() else self.tag


STRICT = Gates(
    # top 4% of pairs (random pair 0.558) · top 5% (random pair 0.621) · ~top 4%
    image=0.80, caption=0.75, moment=0.20,
    # `subject`'s median top-score is 0.83 and 93% of photos clear 0.5, so a 0.5 bar
    # gates nothing: it admitted `subject: toy` @ 0.58 on a girl by a Christmas tree
    # (runner-up `person` @ 0.31) and called it a teddy bear's twin.
    subject=0.80, tag=0.50,
    # Measured against the old model over 22 seeds: at 0.25 the strip's tail filled
    # with moment-only pairs — a tire close-up matched a plastic container at 0.42
    # because they were photographed in the same minute. Same outing, unrelated
    # object. 0.45 keeps those out of the default strip; "Show more" is where they
    # belong, because sometimes that IS what you are looking for.
    score_min=0.45,
)

LOOSE = Gates(
    image=0.66,     # still well above the 0.558 random-pair median
    caption=0.68,   # still above the 0.621 random-pair median — the hard rule holds
    moment=0.08,    # roughly "same day, within a few km"
    subject=0.55,   # admits a subject the model is less sure of
    tag=0.35,
    score_min=0.12,
)

# How many extra photos "Show more" may reveal. Small on purpose: the point is a
# second look, not an infinite scroll of ever-weaker matches.
LOOSE_LIMIT = 6


def cosine_strength(raw: float, gate: float) -> float | None:
    """A cosine-like signal's strength, `0.5 … 1.0`, or None when it is absent.

    `None` below the gate — absent, not zero (§9.2). At the gate exactly the strength
    is 0.5, so admitting a pair is immediately worth half the signal's weight; it
    rises linearly to 1.0 at a raw value of 1.0.
    """
    if gate >= 1.0 or raw < gate:
        return None
    return 0.5 + 0.5 * (raw - gate) / (1.0 - gate)


def tag_strength(agreement: float, idf: float, gate: float) -> float | None:
    """A shared tag's strength, or None when the two photos do not agree enough.

    `agreement` is the WEAKER of the two photos' confidences — a 0.71 close-up
    matching a 1.00 close-up agree at 0.71, not 1.00. `idf` (0–1) damps a label the
    whole library carries, down to `IDF_FLOOR` rather than to nothing.
    """
    if agreement < gate:
        return None
    return agreement * (IDF_FLOOR + (1.0 - IDF_FLOOR) * idf)


def moment_strength(hours_apart: float, metres_apart: float | None) -> float:
    """How much two photos look like one `moment` — same stretch of time, same place.

    `exp(-Δt/τ) × exp(-Δd/scale)`, so both must be close: same hour + same block
    scores 0.75, same afternoon + 500 m scores 0.47, and next-day-or-5 km decays to
    ~0. `metres_apart=None` (either photo has no GPS) falls back to time alone,
    scaled by `MOMENT_NO_GPS`.
    """
    time_factor = math.exp(-abs(hours_apart) / MOMENT_TAU_HOURS)
    if metres_apart is None:
        return time_factor * MOMENT_NO_GPS
    return time_factor * math.exp(-abs(metres_apart) / MOMENT_SCALE_M)


def combine(evidence: list[float | None]) -> float:
    """Compose independent evidence into one 0–1 score — a **noisy-OR**.

    `1 - Π(1 - e)`: each signal takes a share of what is still unclaimed. This gives
    three properties the previous decayed sum did not:

    * the score is a real 0–1 the UI can show as "N% similar", instead of an
      unbounded number compared against a magic floor;
    * one strong signal dominates for free — no decay constant to tune;
    * weak signals can never STACK into a false positive, because each can only ever
      take a fraction of the remainder.

    `None` entries are absent signals and are skipped.
    """
    remaining = 1.0
    for value in evidence:
        if value is None or value <= 0.0:
            continue
        remaining *= 1.0 - min(value, 0.99)
    return 1.0 - remaining


def dimension_weights(tier_weights: dict[str, float] | None = None) -> dict[str, float]:
    """Expand per-tier weights to the per-dimension dict the scorer consumes.

    Tuning happens per TIER (four numbers, in `vocab.yaml`); every consumer still
    sees a plain `{dimension: weight}` map, so nothing downstream knows about tiers.
    A tier absent from `tier_weights` keeps its built-in default.
    """
    overrides = tier_weights or {}
    return {
        dimension: float(overrides.get(name, tier.weight))
        for name, tier in TAG_TIERS.items()
        for dimension in tier.dimensions
    }


def content_dimensions() -> tuple[str, ...]:
    """Dimensions whose tags can make two photos similar BY THEMSELVES (§9)."""
    return tuple(
        dimension
        for tier in TAG_TIERS.values()
        if tier.content
        for dimension in tier.dimensions
    )
