from pathlib import Path

from config import Settings


def test_defaults_to_mac_profile():
    s = Settings(data_dir=Path("/tmp/pl"))
    assert s.profile == "mac"
    assert s.caption_model == "gemma4:26b-a4b"
    assert s.embed_device == "cpu"


def test_jetson_profile_overrides_model_and_device():
    s = Settings(profile="jetson", data_dir=Path("/tmp/pl"))
    assert s.caption_model == "qwen3-vl:4b"
    assert s.planner_model == "qwen3-vl:4b"
    assert s.embed_device == "cuda"


def test_cloud_profile_overrides_model_and_device():
    s = Settings(profile="cloud", data_dir=Path("/tmp/pl"))
    assert s.caption_model == "gemma4:26b-a4b"
    assert s.embed_device == "cuda"


def test_explicit_value_beats_profile_default():
    s = Settings(profile="jetson", data_dir=Path("/tmp/pl"), caption_model="gemma4:e4b")
    assert s.caption_model == "gemma4:e4b"


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
