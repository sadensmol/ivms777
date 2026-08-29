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
        if slot not in catalog.AUTOMATIC_SLOTS:
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


def test_voice_and_music_are_real_slots():
    for slot in ("voice", "music"):
        assert slot in catalog.SLOTS
        assert catalog.INVALIDATES[slot] is None  # they store nothing in ingest


def test_no_slot_offers_a_do_nothing_model():
    # "No narration" / "No music" were options; they are not any more. A film without
    # a soundtrack is a broken film, not a configuration (§11.1).
    for entry in catalog.CATALOG:
        assert entry.key != "off", entry.slot


def test_every_profile_has_a_real_music_and_voice_model():
    for profile in PROFILES:
        assert catalog.default_key("music", profile).startswith("musicgen-")
        assert catalog.default_key("voice", profile) == "kokoro-82m"


def test_mac_scores_with_the_bigger_music_model():
    # Measured: medium generates at 4.75x realtime against small's 4.90x, so the only
    # thing it costs is resident size — which mac has and the 8 GB board does not.
    assert catalog.default_key("music", "mac") == "musicgen-medium"
    for profile in ("jetson", "cloud"):
        assert catalog.default_key("music", profile) == "musicgen-small"
    assert catalog.get("music", "musicgen-medium").profiles == ("mac",)


def test_the_voice_slot_is_never_offered_to_the_user():
    # The film's language and the memory's mood pick the voice (§11.1); nobody should
    # have to know Kokoro cannot speak Russian to get Russian narration.
    for profile in PROFILES:
        assert catalog.is_switchable("voice", profile) is False
    assert catalog.is_switchable("music", "mac") is True


def test_the_two_audio_models_are_hf_sources_with_no_dim_or_preprocess():
    for slot, key in (("voice", "kokoro-82m"), ("music", "musicgen-small")):
        entry = catalog.get(slot, key)
        assert isinstance(entry.source, catalog.HfSource)
        assert entry.dim is None and entry.preprocess is None
        assert entry.prompt_template is None
        assert entry.cost_measured is False  # neither is measured on the board yet


def test_voice_entries_declare_their_languages():
    assert catalog.get("voice", "kokoro-82m").languages == ("en",)
    assert catalog.get("voice", "silero-v4-ru").languages == ("ru",)


def test_non_voice_entries_declare_no_languages():
    for entry in catalog.CATALOG:
        if entry.slot != "voice":
            assert entry.languages == (), entry.key


def test_every_offered_language_has_a_voice_that_speaks_it():
    from feed.language import LANGUAGES

    for profile in PROFILES:
        spoken = {
            lang
            for entry in catalog.entries_for("voice", profile)
            for lang in entry.languages
        }
        assert set(LANGUAGES) <= spoken, profile


def test_silero_is_fetched_from_its_own_host_not_the_hub():
    # There is no `silero/*` repo on Hugging Face; an HfSource would make the
    # downloader probe a repo that does not exist and call the model permanently
    # absent.
    source = catalog.get("voice", "silero-v4-ru").source
    assert isinstance(source, catalog.UrlSource)
    assert source.url.startswith("https://models.silero.ai/")
    assert source.file == "v4_ru.pt"
