import inspect
from pathlib import Path

import pytest

from config import PROFILE_DEFAULTS, Settings
from modelsvc.governor import MemoryGovernor


def _governor_headroom_mb() -> int:
    # Read the real default rather than restating 512 — the invariant below must
    # track the governor, not a copy of it that can drift.
    return inspect.signature(MemoryGovernor.__init__).parameters["headroom_mb"].default


@pytest.mark.parametrize("profile", sorted(PROFILE_DEFAULTS))
def test_every_model_fits_its_profile_budget(profile):
    # The governor refuses a load unless `cost + headroom <= ram_budget_mb`
    # (modelsvc/governor.py). A model whose declared cost breaks that can NEVER
    # load, on any amount of free RAM — it is not a tight fit, it is an
    # unsatisfiable config. `gemma-vision` shipped at 5000 against a 5000 jetson
    # budget with a 512 headroom, so every caption raised InsufficientMemory and
    # the whole stage failed on an idle board with 5.6 GB free.
    settings = Settings(profile=profile, data_dir=Path("/tmp/pl"))
    headroom = _governor_headroom_mb()
    unsatisfiable = {
        name: cost
        for name, cost in settings.model_cost_mb.items()
        if cost + headroom > settings.ram_budget_mb
    }
    assert not unsatisfiable, (
        f"{profile}: {unsatisfiable} cannot ever load — each cost + {headroom} MB "
        f"headroom must fit ram_budget_mb={settings.ram_budget_mb}"
    )


def test_defaults_to_mac_profile():
    s = Settings(data_dir=Path("/tmp/pl"))
    assert s.profile == "mac"
    # One gemma4-E2B GGUF on llama-server serves text + vision (plan 16).
    assert s.caption_model == "gemma4-E2B"
    assert s.planner_model == "gemma4-E2B"
    assert s.embed_device == "cpu"


def test_jetson_profile_overrides_model_and_device():
    s = Settings(profile="jetson", data_dir=Path("/tmp/pl"))
    assert s.caption_model == "gemma4-E2B"
    assert s.planner_model == "gemma4-E2B"
    assert s.embed_device == "cuda"


def test_cloud_profile_overrides_model_and_device():
    s = Settings(profile="cloud", data_dir=Path("/tmp/pl"))
    assert s.caption_model == "qwen2.5vl:7b"
    assert s.embed_device == "cuda"


def test_explicit_value_beats_profile_default():
    s = Settings(profile="jetson", data_dir=Path("/tmp/pl"), caption_model="my-vlm:latest")
    assert s.caption_model == "my-vlm:latest"


def test_derived_paths_hang_off_data_dir():
    s = Settings(data_dir=Path("/tmp/pl"))
    assert s.db_path == Path("/tmp/pl/ivms777.db")
    assert s.thumb_dir == Path("/tmp/pl/thumbs")


def test_env_prefix_is_ivms777(monkeypatch):
    monkeypatch.setenv("IVMS777_PROFILE", "cloud")
    monkeypatch.setenv("IVMS777_DATA_DIR", "/tmp/other")
    s = Settings()
    assert s.profile == "cloud"
    assert s.data_dir == Path("/tmp/other")


def test_ram_budget_follows_profile_default(tmp_path):
    mac = Settings(profile="mac", data_dir=tmp_path, use_fake_embedder=True, use_fake_inference=True)
    jetson = Settings(profile="jetson", data_dir=tmp_path, use_fake_embedder=True, use_fake_inference=True)
    assert mac.ram_budget_mb == 24000
    # 5000, not 6000: app+worker+OS hold ~1.9 GB of the board's 7.4 GB (§8.1).
    assert jetson.ram_budget_mb == 5000


def test_conveyor_profile_defaults():
    jetson = Settings(profile="jetson", data_dir=Path("/tmp/pl"))
    assert jetson.gpu_concurrency == 1
    assert jetson.llm_managed is True
    assert jetson.llm_idle_ttl_s == 120
    # Measured on the board, not guessed (§8.1) — these MUST track reality or
    # the governor loads a model on top of one it should have evicted.
    assert jetson.model_cost_mb["siglip"] == 3400
    assert jetson.model_cost_mb["nomic"] == 2200
    # Text-only gemma is cheaper than the vision mode by the projector (§3.1).
    # Both measured on the board as the drop in available RAM: gemma 3606 MB,
    # gemma-vision 3936 MB (peaks 3374 / 4047). gemma-vision was 5000 — an
    # over-estimate that exceeded the 5000 budget and made captioning impossible.
    assert jetson.model_cost_mb["gemma"] == 3800
    assert jetson.model_cost_mb["gemma-vision"] == 4300
    assert jetson.model_cost_mb["gemma"] < jetson.model_cost_mb["gemma-vision"]

    mac = Settings(profile="mac", data_dir=Path("/tmp/pl"))
    assert mac.gpu_concurrency == 3
    assert mac.llm_managed is False            # host-native llama-server (make llama-mac), reused not spawned
    assert mac.llm_idle_ttl_s is None          # 32 GB: never idle-unload

    cloud = Settings(profile="cloud", data_dir=Path("/tmp/pl"))
    assert cloud.gpu_concurrency == 4
    assert cloud.llm_managed is False           # remote vLLM, not supervised


def test_conveyor_env_override():
    s = Settings(profile="jetson", data_dir=Path("/tmp/pl"), gpu_concurrency=2)
    assert s.gpu_concurrency == 2


def test_llm_health_url_uses_port():
    s = Settings(profile="jetson", data_dir=Path("/tmp/pl"))
    assert s.llm_health_url == "http://localhost:8080/health"
