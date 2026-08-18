"""Switching a slot is a re-index (design §4.1): the stored choice, the vector
table and the requeued jobs move together, or not at all."""

import pytest

from config import Settings
from db.connection import connect, migrate
from db.settings import get_setting
from db.vectors import ensure_vec_dim, vec_dim
from embedding.store import read_vector, write_vector
from models import slots
from tests.factories import add_photo


def _conn(tmp_path):
    conn = connect(tmp_path / "t.db")
    migrate(conn)
    return conn


def _settings(tmp_path, profile="mac", **kw) -> Settings:
    return Settings(data_dir=tmp_path, profile=profile, **kw)


def _photo(conn, owner_id=1, name="a"):
    return add_photo(conn, owner_id=owner_id, content_hash=name * 8)


def _stages(conn, photo_id):
    return {
        row["stage"]: row["status"]
        for row in conn.execute("SELECT stage, status FROM jobs WHERE photo_id = ?", (photo_id,))
    }


# --- the vector table ------------------------------------------------------


def test_vec_dim_reads_the_declared_width(tmp_path):
    assert vec_dim(_conn(tmp_path)) == 1152


def test_ensure_vec_dim_is_a_noop_at_the_same_width(tmp_path):
    conn = _conn(tmp_path)
    pid = _photo(conn)
    write_vector(conn, pid, [0.5] * 1152)
    assert ensure_vec_dim(conn, 1152) is False
    assert read_vector(conn, pid) is not None  # vectors survive


def test_ensure_vec_dim_rebuilds_at_a_different_width(tmp_path):
    conn = _conn(tmp_path)
    pid = _photo(conn)
    write_vector(conn, pid, [0.5] * 1152)
    assert ensure_vec_dim(conn, 768) is True
    assert vec_dim(conn) == 768
    assert read_vector(conn, pid) is None  # not migrated — dropped
    write_vector(conn, pid, [0.25] * 768)  # and the new width is usable
    assert len(read_vector(conn, pid)) == 768


# --- switching -------------------------------------------------------------


def test_switching_the_image_embedder_requeues_embed_and_taxonomy(tmp_path):
    conn = _conn(tmp_path)
    settings = _settings(tmp_path)
    pid = _photo(conn)
    result = slots.switch(conn, settings, "image_embed", "siglip2-so400m-512")
    assert result.stages == ("embed", "taxonomy")
    assert result.photos_requeued == 1
    assert _stages(conn, pid) == {"embed": "pending", "taxonomy": "pending"}
    assert get_setting(conn, 1, "model_slot.image_embed") == "siglip2-so400m-512"
    assert slots.resolve_key(conn, settings, "image_embed") == "siglip2-so400m-512"


def test_a_same_width_switch_keeps_the_vector_table(tmp_path):
    conn = _conn(tmp_path)
    pid = _photo(conn)
    write_vector(conn, pid, [0.5] * 1152)
    result = slots.switch(conn, _settings(tmp_path), "image_embed", "siglip2-so400m-512")
    assert result.vectors_dropped is False
    assert vec_dim(conn) == 1152
    # The stale vectors stay until the requeued embed stage overwrites them: the
    # space is the same, so they are wrong-ish, not meaningless.
    assert read_vector(conn, pid) is not None


def test_switching_the_text_embedder_requeues_only_the_caption_vectors(tmp_path):
    conn = _conn(tmp_path)
    pid = _photo(conn)
    result = slots.switch(conn, _settings(tmp_path), "text_embed", "embeddinggemma-300m")
    assert result.stages == ("caption_embed",)
    assert _stages(conn, pid) == {"caption_embed": "pending"}
    assert result.vectors_dropped is False


def test_switching_the_caption_model_requeues_captions_and_their_vectors(tmp_path):
    conn = _conn(tmp_path)
    pid = _photo(conn)
    result = slots.switch(conn, _settings(tmp_path), "caption", "qwen3-vl-4b")
    assert result.stages == ("caption", "caption_embed")
    assert _stages(conn, pid) == {"caption": "pending", "caption_embed": "pending"}


def test_switching_the_planner_requeues_nothing(tmp_path):
    conn = _conn(tmp_path)
    pid = _photo(conn)
    result = slots.switch(conn, _settings(tmp_path), "planner", "qwen3-4b-2507")
    assert result.stages == ()
    assert result.photos_requeued == 0
    assert _stages(conn, pid) == {}  # the planner stores nothing
    assert get_setting(conn, 1, "model_slot.planner") == "qwen3-4b-2507"


def test_switching_to_the_active_model_is_a_noop(tmp_path):
    conn = _conn(tmp_path)
    pid = _photo(conn)
    result = slots.switch(conn, _settings(tmp_path), "caption", "gemma4-E2B")
    assert result.photos_requeued == 0
    assert _stages(conn, pid) == {}


def test_only_the_owners_photos_are_requeued(tmp_path):
    conn = _conn(tmp_path)
    mine, theirs = _photo(conn, 1, "a"), _photo(conn, 2, "b")
    result = slots.switch(conn, _settings(tmp_path), "caption", "qwen3-vl-4b")
    assert result.photos_requeued == 1
    assert _stages(conn, mine) and not _stages(conn, theirs)


def test_an_unknown_model_is_refused_and_writes_nothing(tmp_path):
    conn = _conn(tmp_path)
    pid = _photo(conn)
    with pytest.raises(ValueError):
        slots.switch(conn, _settings(tmp_path), "caption", "no-such-model")
    assert get_setting(conn, 1, "model_slot.caption") is None
    assert _stages(conn, pid) == {}


def test_a_model_not_offered_on_this_profile_is_refused(tmp_path):
    conn = _conn(tmp_path)
    with pytest.raises(ValueError):
        # qwen3-vl-8b is mac-only
        slots.switch(conn, _settings(tmp_path, profile="jetson"), "caption", "qwen3-vl-8b")


def test_cloud_slots_cannot_be_switched(tmp_path):
    conn = _conn(tmp_path)
    with pytest.raises(ValueError):
        slots.switch(conn, _settings(tmp_path, profile="cloud"), "caption", "qwen2.5vl-7b")


def test_a_failed_requeue_rolls_back_the_whole_switch(tmp_path, monkeypatch):
    # One transaction (design §4.1): a new model against the old model's vectors is
    # a state the design says never exists, so a partial switch must not survive.
    conn = _conn(tmp_path)
    pid = _photo(conn)
    write_vector(conn, pid, [0.5] * 1152)
    import ingest.jobs

    def boom(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(ingest.jobs, "reprocess", boom)
    with pytest.raises(RuntimeError):
        slots.switch(conn, _settings(tmp_path), "image_embed", "siglip2-so400m-512")
    assert get_setting(conn, 1, "model_slot.image_embed") is None
    assert read_vector(conn, pid) is not None
    assert _stages(conn, pid) == {}


# --- preview ---------------------------------------------------------------


def test_preview_reports_the_same_cost_without_touching_anything(tmp_path):
    conn = _conn(tmp_path)
    pid = _photo(conn)
    settings = _settings(tmp_path)
    preview = slots.preview(conn, settings, "caption", "qwen3-vl-4b")
    assert (preview.stages, preview.photos_requeued) == (("caption", "caption_embed"), 1)
    assert _stages(conn, pid) == {}
    assert get_setting(conn, 1, "model_slot.caption") is None
    switched = slots.switch(conn, settings, "caption", "qwen3-vl-4b")
    assert (switched.stages, switched.photos_requeued) == (
        preview.stages,
        preview.photos_requeued,
    )


def test_preview_of_the_active_model_costs_nothing(tmp_path):
    conn = _conn(tmp_path)
    _photo(conn)
    preview = slots.preview(conn, _settings(tmp_path), "caption", "gemma4-E2B")
    assert preview.photos_requeued == 0 and preview.stages == ()
