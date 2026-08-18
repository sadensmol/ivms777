"""Slot resolution: stored choice → env override → profile default (design §4.1)."""


from config import Settings
from db.connection import connect, migrate
from db.settings import set_setting
from models import catalog, slots


def _conn(tmp_path):
    conn = connect(tmp_path / "t.db")
    migrate(conn)
    return conn


def _settings(tmp_path, **kw) -> Settings:
    return Settings(data_dir=tmp_path, **kw)


def test_nothing_stored_resolves_to_profile_defaults(tmp_path):
    conn = _conn(tmp_path)
    resolved = slots.resolve(conn, _settings(tmp_path, profile="jetson"))
    assert {slot: entry.key for slot, entry in resolved.items()} == {
        slot: catalog.default_key(slot, "jetson") for slot in catalog.SLOTS
    }


def test_env_override_beats_the_default(tmp_path):
    conn = _conn(tmp_path)
    settings = _settings(tmp_path, profile="jetson", caption_model="qwen3-vl-4b")
    assert slots.resolve_key(conn, settings, "caption") == "qwen3-vl-4b"
    # untouched slots keep their defaults
    assert slots.resolve_key(conn, settings, "planner") == "gemma4-E2B"


def test_stored_choice_beats_the_env_override(tmp_path):
    conn = _conn(tmp_path)
    settings = _settings(tmp_path, profile="jetson", caption_model="qwen3-vl-4b")
    set_setting(conn, settings.owner_id, "model_slot.caption", "gemma4-E2B")
    assert slots.resolve_key(conn, settings, "caption") == "gemma4-E2B"


def test_stored_choice_applies_to_every_slot(tmp_path):
    conn = _conn(tmp_path)
    settings = _settings(tmp_path, profile="mac")
    set_setting(conn, settings.owner_id, "model_slot.image_embed", "siglip2-so400m-512")
    set_setting(conn, settings.owner_id, "model_slot.text_embed", "embeddinggemma-300m")
    resolved = slots.resolve(conn, settings)
    assert resolved["image_embed"].preprocess.input_px == 512
    assert resolved["text_embed"].key == "embeddinggemma-300m"


def test_the_embedder_is_labelled_with_the_stored_slot_choice(tmp_path):
    # `photos.embedding_model` (the photo page's "Embedded for semantic search
    # (...)") is stamped with this label, so it must name the model the slot
    # actually holds — not a config constant.
    conn = _conn(tmp_path)
    settings = _settings(tmp_path, profile="mac")
    set_setting(conn, settings.owner_id, "model_slot.image_embed", "siglip2-so400m-512")
    assert settings.build_embedder(conn)[1] == "siglip2-so400m-512"


def test_the_embedder_label_falls_back_to_the_profile_default(tmp_path):
    settings = _settings(tmp_path, profile="mac")
    assert settings.build_embedder()[1] == catalog.default_key("image_embed", "mac")


def test_unknown_stored_key_falls_back_to_the_default(tmp_path):
    # A downgraded install (or a catalog entry that was removed) must still boot.
    conn = _conn(tmp_path)
    settings = _settings(tmp_path, profile="jetson")
    set_setting(conn, settings.owner_id, "model_slot.caption", "no-such-model")
    assert slots.resolve_key(conn, settings, "caption") == "gemma4-E2B"


def test_key_not_offered_on_this_profile_falls_back(tmp_path):
    # qwen3-vl-8b is mac-only: on jetson it must not resolve, because the governor
    # could never load it (design §8.1).
    conn = _conn(tmp_path)
    settings = _settings(tmp_path, profile="jetson")
    set_setting(conn, settings.owner_id, "model_slot.caption", "qwen3-vl-8b")
    assert slots.resolve_key(conn, settings, "caption") == "gemma4-E2B"


def test_unknown_env_override_falls_back_too(tmp_path):
    conn = _conn(tmp_path)
    settings = _settings(tmp_path, profile="jetson", planner_model="ollama-qwen2.5:3b")
    assert slots.resolve_key(conn, settings, "planner") == "gemma4-E2B"


def test_defaults_resolve_without_a_connection(tmp_path):
    # The `models` service holds no DB (design §4.1) — it must resolve the profile
    # defaults with `conn=None`.
    settings = _settings(tmp_path, profile="jetson")
    resolved = slots.resolve(None, settings)
    assert resolved["caption"].key == "gemma4-E2B"


def test_stored_keys_use_the_documented_prefix(tmp_path):
    assert slots.setting_key("caption") == "model_slot.caption"


def test_resolve_keys_returns_a_plain_slot_to_key_map(tmp_path):
    conn = _conn(tmp_path)
    settings = _settings(tmp_path, profile="mac")
    assert slots.resolve_keys(conn, settings) == {
        "image_embed": "siglip2-so400m-384",
        "text_embed": "nomic-1.5",
        "caption": "gemma4-E2B",
        "planner": "gemma4-E2B",
    }


def test_cloud_ignores_stored_choices(tmp_path):
    # Cloud slots are config-only: vLLM serves one model per container.
    conn = _conn(tmp_path)
    settings = _settings(tmp_path, profile="cloud")
    set_setting(conn, settings.owner_id, "model_slot.caption", "gemma4-E2B")
    assert slots.resolve_key(conn, settings, "caption") == "qwen2.5vl-7b"
