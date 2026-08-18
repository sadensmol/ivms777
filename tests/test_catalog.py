"""The model catalog (design §4.1) — the closed list of models a slot may hold.

Everything here is a shape/consistency check on plain data: the catalog is what
`app` renders, what the governor budgets against, and what the client resizes by,
so a malformed entry is a runtime failure in three different places.
"""

import inspect
from pathlib import Path

import pytest

from config import PROFILE_DEFAULTS, Settings
from inference.prompts import caption_messages
from ingest.jobs import STAGES
from models import catalog
from modelsvc.governor import MemoryGovernor

PROFILES = sorted(PROFILE_DEFAULTS)


def _governor_headroom_mb() -> int:
    return inspect.signature(MemoryGovernor.__init__).parameters["headroom_mb"].default


def test_keys_are_unique_per_slot():
    seen = set()
    for entry in catalog.CATALOG:
        assert (entry.slot, entry.key) not in seen, f"duplicate {entry.slot}/{entry.key}"
        seen.add((entry.slot, entry.key))


def test_every_entry_declares_a_real_slot_and_real_profiles():
    for entry in catalog.CATALOG:
        assert entry.slot in catalog.SLOTS, entry.key
        assert entry.profiles, f"{entry.key} is offered nowhere"
        for profile in entry.profiles:
            assert profile in PROFILE_DEFAULTS, f"{entry.key}: unknown profile {profile}"


def test_invalidates_names_real_stages():
    assert set(catalog.INVALIDATES) == set(catalog.SLOTS)
    for slot, rng in catalog.INVALIDATES.items():
        if rng is None:
            continue
        first, last = rng
        assert first in STAGES and last in STAGES, slot
        assert STAGES.index(first) <= STAGES.index(last), slot


@pytest.mark.parametrize("profile", PROFILES)
def test_every_slot_has_a_default_offered_on_that_profile(profile):
    for slot in catalog.SLOTS:
        key = catalog.default_key(slot, profile)
        assert key in [e.key for e in catalog.entries_for(slot, profile)]


def test_image_embed_entries_carry_dim_and_preprocess():
    for entry in catalog.entries_for("image_embed", "jetson"):
        assert entry.dim, entry.key
        assert entry.preprocess is not None, entry.key
        assert entry.preprocess.mode in ("squash", "native"), entry.key
        assert entry.preprocess.resample in ("bilinear", "bicubic"), entry.key
        assert entry.prompt_template is None, entry.key


def test_text_embed_entries_carry_dim_but_no_preprocess():
    for entry in catalog.entries_for("text_embed", "jetson"):
        assert entry.dim, entry.key
        assert entry.preprocess is None, entry.key


@pytest.mark.parametrize("slot", ["caption", "planner"])
def test_llm_entries_carry_a_usable_prompt_template(slot):
    for entry in catalog.entries_for(slot, "jetson"):
        assert entry.dim is None and entry.preprocess is None, entry.key
        assert entry.prompt_template, entry.key
        # `_SYSTEM_BY_MODEL` is empty today — every model falls back to the default
        # system prompt. The contract is that the key RESOLVES, not that it has a
        # bespoke template.
        messages = caption_messages(entry.prompt_template, "data:image/png;base64,x")
        assert messages and messages[0]["content"]


def test_defaults_reproduce_todays_behaviour():
    # The whole point of a default: an untouched install must behave exactly as it
    # did before slots existed.
    siglip = catalog.get("image_embed", catalog.default_key("image_embed", "jetson"))
    assert siglip.dim == 1152
    assert (siglip.preprocess.input_px, siglip.preprocess.mode) == (384, "squash")
    assert catalog.default_key("caption", "mac") == catalog.default_key("planner", "mac")
    assert catalog.default_key("caption", "jetson") == catalog.default_key("planner", "jetson")


def test_cloud_slots_are_not_switchable():
    for slot in catalog.SLOTS:
        assert catalog.is_switchable(slot, "cloud") is False
        assert catalog.is_switchable(slot, "jetson") is True


def test_get_rejects_an_unknown_key():
    with pytest.raises(KeyError):
        catalog.get("image_embed", "no-such-model")
    with pytest.raises(KeyError):
        # right key, wrong slot — slots are separate namespaces on purpose
        catalog.get("planner", catalog.default_key("image_embed", "mac"))


@pytest.mark.parametrize("profile", PROFILES)
def test_every_offered_entry_can_actually_load_on_that_profile(profile):
    # The governor refuses a load unless `cost + headroom <= ram_budget_mb`
    # (modelsvc/governor.py), so an entry that breaks it can NEVER load — offering
    # it would be offering a slot that is guaranteed to fail (design §8.1).
    settings = Settings(profile=profile, data_dir=Path("/tmp/pl"))
    headroom = _governor_headroom_mb()
    unsatisfiable = {
        entry.key: entry.cost_mb
        for slot in catalog.SLOTS
        for entry in catalog.entries_for(slot, profile)
        if entry.cost_mb + headroom > settings.ram_budget_mb
    }
    assert not unsatisfiable, (
        f"{profile}: {unsatisfiable} cannot ever load — each cost + {headroom} MB "
        f"headroom must fit ram_budget_mb={settings.ram_budget_mb}"
    )


def test_measured_costs_match_the_boards_numbers():
    # The four defaults are the only costs measured on the Jetson (design §8.1).
    # Everything else is an estimate and must say so, because the UI shows it.
    measured = {
        (e.slot, e.key): e.cost_mb for e in catalog.CATALOG if e.cost_measured
    }
    assert measured[("image_embed", "siglip2-so400m-384")] == 3400
    assert measured[("text_embed", "nomic-1.5")] == 2200
    assert measured[("planner", "gemma4-E2B")] == 3800
    assert measured[("caption", "gemma4-E2B")] == 4300
    # vision costs more than text-only, or the swap logic describes a model that
    # does not exist.
    assert measured[("planner", "gemma4-E2B")] < measured[("caption", "gemma4-E2B")]
