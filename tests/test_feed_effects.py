"""The effect library (design §11.1) — pure data, so the tests are about its contract."""

import re

from feed import effects


def test_every_shot_effect_has_a_stable_id():
    ids = [entry["id"] for entry in effects.SHOT_EFFECTS.values()]
    assert len(ids) == len(set(ids))
    assert effects.SHOT_EFFECTS[effects.DEFAULT_SHOT]["id"] == 0


def test_an_invented_effect_falls_back_rather_than_failing():
    # The story pass is a small model; it WILL invent a name. Losing the film over a
    # typo is not an option (§11.1).
    assert effects.shot_effect("cinematic_wow") == effects.DEFAULT_SHOT
    assert effects.shot_effect(None) == effects.DEFAULT_SHOT
    assert effects.transition("swoosh") == effects.DEFAULT_TRANSITION


def test_a_real_name_survives():
    assert effects.shot_effect("bloom") == "bloom"
    assert effects.transition("whip") == "whip"


def test_the_prompt_only_advertises_effects_that_exist():
    text = effects.catalogue_for_prompt()
    for name in effects.SHOT_EFFECTS:
        assert f"### {name}" in text
    # "none" is the first shot's non-transition; the model must never pick it.
    assert "### none" not in text


def test_every_effect_tells_the_model_what_it_looks_like_and_when_to_use_it():
    # The description IS the interface (§11.1). A model given only a name picks at
    # random, which is how a film ends up with a nightclub pulse over a quiet evening.
    for name, entry in effects.SHOT_EFFECTS.items():
        assert entry["feel"].strip(), name
        assert entry["use_when"].strip(), name
        assert entry["avoid_when"].strip(), name


def test_every_parameter_explains_both_ends_of_its_range():
    for name, entry in effects.SHOT_EFFECTS.items():
        for key, spec in entry["params"].items():
            low, high, default, note = spec
            assert low <= default <= high, (name, key)
            assert len(note) > 20, (name, key)


def test_the_prompt_names_every_parameter_with_its_range():
    text = effects.catalogue_for_prompt()
    for name, entry in effects.SHOT_EFFECTS.items():
        for key in entry["params"]:
            assert key in text, (name, key)


def test_transition_seconds_come_from_the_catalogue():
    assert effects.transition_seconds("cut") == 0.0
    assert effects.transition_seconds("dip") > effects.transition_seconds("dissolve")
    assert effects.transition_seconds("nonsense") == effects.transition_seconds("dissolve")


def test_the_payload_maps_every_name_to_its_shader_id():
    payload = effects.payload()
    assert payload["shot"] == {n: e["id"] for n, e in effects.SHOT_EFFECTS.items()}
    assert payload["transition"]["whip"] == effects.TRANSITIONS["whip"]["id"]


def test_a_sequence_is_repaired_rather_than_rejected():
    packed = effects.pack_beats(
        [{"effect": "made_up", "from": 0.9, "to": 0.1},
         {"effect": "bloom", "from": -3, "to": 99, "params": {"strength": 50}}],
        "weave",
    )
    assert [b["effect"] for b in packed] == ["bloom", "weave"]   # sorted by start
    for beat in packed:
        assert 0.0 <= beat["from"] < beat["to"] <= 1.0
        assert len(beat["params"]) == 4
    # Out-of-range parameters are clamped, never passed through.
    assert packed[0]["params"][1] <= 1.0


def test_an_empty_sequence_becomes_one_full_length_beat():
    packed = effects.pack_beats(None, "bokeh")
    assert len(packed) == 1
    assert packed[0]["effect"] == "bokeh"
    assert packed[0]["from"] == 0.0 and packed[0]["to"] == 1.0


def test_a_sequence_is_capped_so_a_shot_cannot_become_a_showreel():
    packed = effects.pack_beats(
        [{"effect": "weave", "from": i / 10, "to": i / 10 + 0.1} for i in range(9)],
        "plain",
    )
    assert len(packed) <= effects.MAX_BEATS


def test_a_beat_is_never_shorter_than_a_blink():
    packed = effects.pack_beats([{"effect": "glitch", "from": 0.5, "to": 0.5001}], "plain")
    assert packed[0]["to"] - packed[0]["from"] >= effects.MIN_BEAT_SPAN - 1e-6


# --- the catalogue and the shader must not drift (§11.1) -------------------------------


def _shader() -> str:
    import re  # noqa: F401 - used by the parity tests below
    from pathlib import Path

    return Path("web/static/feed-gl.js").read_text()


def _implemented(pattern: str) -> set[int]:
    import re

    return {int(n) for n in re.findall(pattern, _shader())}


def test_every_effect_the_catalogue_offers_is_implemented_in_the_shader():
    # The prompt is generated from this table, so an entry with no shader branch is an
    # effect the model can choose and the renderer will silently ignore.
    ids = _implemented(r"fx\s*==\s*(\d+)") | _implemented(r"fx\.[xy]\s*==\s*(\d+)")
    for name, entry in effects.SHOT_EFFECTS.items():
        if entry["id"] == 0:
            continue                    # `plain` is the absence of an effect
        assert entry["id"] in ids, f"{name} (id {entry['id']}) has no shader branch"


def test_every_transition_the_catalogue_offers_is_implemented_in_the_shader():
    ids = _implemented(r"uMode\s*==\s*(\d+)")
    shader = _shader()
    for name, entry in effects.TRANSITIONS.items():
        if entry["id"] == 0:
            continue                    # dissolve/none ARE the default mix
        if entry["id"] in ids:
            continue
        # A transition may instead ride a uniform of its own — `whip`, `zoom_blur` and
        # `match_zoom` are dissolves with `uWhip` smearing both layers — but then the
        # renderer must at least NAME it, or nothing ever sets that uniform.
        assert re.search(rf"\b{re.escape(name)}\b", shader), (
            f"{name} (id {entry['id']}) is offered but the renderer never mentions it")


def test_no_effect_carries_more_parameters_than_the_shader_can_receive():
    # Each effect is delivered as ONE vec4, so four is the hard ceiling.
    for name, entry in effects.SHOT_EFFECTS.items():
        assert len(entry["params"]) <= 4, name


def test_the_no_model_pools_only_name_effects_that_exist():
    # The deterministic film is what plays when the conductor fails; a typo here is a
    # silent fallback to `plain` on every shot.
    from feed import director

    pools = (*director._ARRIVALS.values(), *director._DEPARTURES.values(),
             director._BEDS_LOW, director._BEDS_BRIGHT, director._BEDS_PLAIN)
    for pool in pools:
        for name in pool:
            assert name is None or name in effects.SHOT_EFFECTS, name


def test_the_no_model_pools_respect_the_energy_they_are_filed_under():
    # A hospital visit must not get `kaleidoscope` just because no model ran.
    from feed import director

    for energy, pool in (*director._ARRIVALS.items(), *director._DEPARTURES.items()):
        for name in pool:
            if name is None:
                continue
            assert energy in effects.SHOT_EFFECTS[name]["energy"], (energy, name)


def test_the_shader_never_carries_a_backtick():
    # The GLSL lives inside a JS template literal, so ONE backtick in a pasted comment
    # ends the string and breaks the whole player. It has happened twice.
    src = _shader()
    at = src.index("const FRAG = `")
    start = src.index("`", at)
    end = src.index("`", start + 1)
    body = src[start + 1:end]
    assert "${" not in body
    # If a backtick had crept in, the literal would have ended early and the body would
    # be a fraction of its real size.
    assert body.count("\n") > 500, "the FRAG literal is terminating early"


def test_every_effect_declares_the_moods_it_suits():
    for name, entry in effects.SHOT_EFFECTS.items():
        assert entry.get("energy"), name
        assert set(entry["energy"]) <= {"high", "calm", "low"}, name


def test_the_shortlist_is_smaller_than_the_whole_library():
    # The conductor's brief already carries a line per photograph; the full table is
    # ~13 000 tokens (§11.1).
    for energy in ("high", "calm", "low"):
        assert len(effects.reachable(energy)) < len(effects.SHOT_EFFECTS)
        assert effects.reachable(energy), energy
