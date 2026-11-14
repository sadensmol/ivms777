"""`make models` pre-fetches the selected slots' weights (design §4.1)."""

from pathlib import Path

import pytest

from config import Settings
from scripts.fetch_models import main, wanted


def test_it_wants_every_fetchable_slot():
    settings = Settings(data_dir=Path("/tmp/pl"), profile="mac")
    by_slot: dict[str, list[str]] = {}
    for slot, target in wanted(settings):
        by_slot.setdefault(slot, []).append(target)
    assert by_slot["image_embed"] == ["google/siglip2-so400m-patch14-384"]
    assert by_slot["text_embed"] == ["nomic-ai/nomic-embed-text-v1.5"]
    assert by_slot["music"] == ["facebook/musicgen-medium"]   # the mac default
    assert "hexgrad/Kokoro-82M" in by_slot["voice"]


def test_gguf_slots_are_not_listed():
    # `caption`/`planner` are GGUF files fetched by `make llama-mac`, not snapshots.
    settings = Settings(data_dir=Path("/tmp/pl"), profile="mac")
    listed = dict(wanted(settings))
    assert "caption" not in listed and "planner" not in listed


def test_every_profile_fetches_a_music_model():
    # There is no "no music" option any more, so every profile pulls one — mac the
    # bigger one it can afford, the 8 GB board the small one (§4.1, §11.1).
    for profile, repo in (("mac", "facebook/musicgen-medium"),
                          ("jetson", "facebook/musicgen-small"),
                          ("cloud", "facebook/musicgen-small")):
        settings = Settings(data_dir=Path("/tmp/pl"), profile=profile)
        assert dict(wanted(settings))["music"] == repo


def test_a_url_backed_entry_is_listed_by_its_url():
    # Silero is not on the Hub at all (§4.1); it is fetched from its own host, so it
    # must be listed by URL rather than by a repo id that does not exist.
    from models import catalog
    from models.catalog import UrlSource

    entry = catalog.get("voice", "silero-v4-ru")
    assert isinstance(entry.source, UrlSource)
    assert entry.source.url.startswith("https://models.silero.ai/")


def test_a_stored_voice_choice_is_ignored_because_voice_is_automatic():
    # `voice` is not switchable: the film's language picks the model (§11.1), so a
    # leftover stored choice must not override it.
    from db.connection import connect, migrate
    from db.settings import set_setting

    settings = Settings(data_dir=Path("/tmp/pl-url"), profile="mac")
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    conn = connect(settings.db_path)
    migrate(conn)
    set_setting(conn, settings.owner_id, "model_slot.voice", "silero-v4-ru")
    # Both voices are fetched regardless, because the slot is automatic — the stored
    # choice simply has no say in it.
    voices = [t for slot, t in wanted(settings, conn) if slot == "voice"]
    assert "hexgrad/Kokoro-82M" in voices
    conn.close()


def test_a_failing_download_fails_the_target_after_trying_the_rest():
    # `make up` must not start an app that is missing a model it needs — a silently
    # skipped Silero is how Russian narration had no voice. Every model is still
    # tried, so one run reports every failure at once.
    calls = []

    def boom(target):
        calls.append(target)
        raise RuntimeError("Access denied. This repository requires approval.")

    assert main(["--profile", "mac"], fetch=boom) == 1
    assert len(calls) >= 3          # it kept going after the first failure


def test_a_clean_run_succeeds():
    assert main(["--profile", "mac"], fetch=lambda target: None) == 0


def test_it_reports_what_it_skipped_and_why(capsys):
    def fetch(target):
        if "musicgen" in target:
            raise RuntimeError("Access denied. This repository requires approval.")

    main(["--profile", "mac"], fetch=fetch)
    out = capsys.readouterr().out
    assert "musicgen-medium" in out
    assert "requires approval" in out
    assert "skipped" in out
    assert "make up stops" in out


def test_a_clean_run_reports_no_skips(capsys):
    main(["--profile", "mac"], fetch=lambda target: None)
    out = capsys.readouterr().out
    assert "skipped" not in out
    assert "pre-fetching" in out


def test_every_voice_is_prefetched_because_the_slot_is_automatic():
    # `voice` is routed at runtime by the film's language (§11.1), so pre-fetching
    # only the selected one leaves Russian narration with no model on disk — exactly
    # the failure this guards against.
    settings = Settings(data_dir=Path("/tmp/pl"), profile="mac")
    targets = [target for slot, target in wanted(settings) if slot == "voice"]
    assert "hexgrad/Kokoro-82M" in targets
    assert any(t.startswith("https://models.silero.ai/") for t in targets)


def test_a_non_automatic_slot_fetches_only_the_selected_model():
    settings = Settings(data_dir=Path("/tmp/pl"), profile="mac")
    embed = [target for slot, target in wanted(settings) if slot == "image_embed"]
    assert embed == ["google/siglip2-so400m-patch14-384"]


def test_mac_prefetches_voxcpm2_at_its_pinned_commit():
    # Voice copy is a `voice` entry, so the automatic slot pre-fetches it too, and at
    # the ONE commit the child is pinned to (§4.1).
    from models import catalog

    source = catalog.get("voice", "voxcpm2").source
    settings = Settings(data_dir=Path("/tmp/pl"), profile="mac")
    voices = [target for slot, target in wanted(settings) if slot == "voice"]
    assert f"openbmb/VoxCPM2@{source.revision}" in voices
    assert "hexgrad/Kokoro-82M" in voices          # unpinned: listed by repo id alone


def test_jetson_does_not_prefetch_voxcpm2():
    settings = Settings(data_dir=Path("/tmp/pl"), profile="jetson")
    assert not any("VoxCPM2" in target for _, target in wanted(settings))


def test_a_pinned_target_is_fetched_at_its_revision(monkeypatch):
    import sys
    import types

    from scripts.fetch_models import _fetch

    calls = []
    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = lambda repo, **kw: calls.append((repo, kw))
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    settings = Settings(data_dir=Path("/tmp/pl"), profile="mac")

    _fetch("openbmb/VoxCPM2@abc123", settings)
    _fetch("hexgrad/Kokoro-82M", settings)
    assert calls == [
        ("openbmb/VoxCPM2", {"revision": "abc123"}),
        ("hexgrad/Kokoro-82M", {}),
    ]


WHISPER = "openai/whisper-large-v3-turbo@41f01f3fe87f28c78e2fbf8b568835947dd65ed9"


@pytest.mark.parametrize("profile", ["mac", "cloud"])
def test_whisper_is_prefetched_at_its_pinned_commit_where_voices_clone(profile):
    settings = Settings(data_dir=Path("/tmp/pl"), profile=profile)
    assert ("asr", WHISPER) in wanted(settings)


def test_jetson_does_not_prefetch_whisper():
    settings = Settings(data_dir=Path("/tmp/pl"), profile="jetson")
    assert not any(slot == "asr" or "whisper" in target for slot, target in wanted(settings))


ECAPA = "speechbrain/spkrec-ecapa-voxceleb@0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"


def test_mac_prefetches_the_speaker_model_at_its_pinned_commit():
    settings = Settings(data_dir=Path("/tmp/pl"), profile="mac")
    assert ("speaker_embed", ECAPA) in wanted(settings)


def test_the_speaker_model_is_prefetched_on_mac_only():
    for profile in ("jetson", "cloud"):
        settings = Settings(data_dir=Path("/tmp/pl"), profile=profile)
        assert not any(slot == "speaker_embed" or "ecapa" in target
                       for slot, target in wanted(settings)), profile
