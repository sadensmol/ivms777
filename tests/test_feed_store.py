"""`memory_presentations` — one prepared feed per memory (design §6, §11.1)."""

import pytest

from albums.memory_store import Memory, group_id_for_album_key, replace_memories
from feed.store import (
    Presentation,
    prune_orphan_audio,
    read_presentation,
    save_presentation,
)
from storage.local import LocalStorage
from tests.factories import add_photo


@pytest.fixture
def memory_group(conn):
    for pid in (1, 2, 3):
        add_photo(conn, photo_id=pid, content_hash=f"h{pid}", thumb_key=f"{pid}.jpg",
                  caption="x", shot_at=f"2025-07-12T1{pid}:00:00")
    replace_memories(conn, 1, [Memory("Beach day", "Warm afternoon.", [1, 2, 3], "sig")])
    return conn.execute("SELECT id FROM groups WHERE kind = 'memory'").fetchone()["id"]


def _presentation(group_id: int, **over) -> Presentation:
    fields = {
        "group_id": group_id,
        "language": "en",
        "signature": "sig",
        "timeline": {"title": "Beach day", "shots": [{"photo_id": 1}]},
        "music_key": "aa/bb/aabb.wav",
        "narration_key": "cc/dd/ccdd.wav",
        "status": "ready",
        "error": None,
    }
    fields.update(over)
    return Presentation(**fields)


def test_save_then_read_round_trips(conn, memory_group):
    save_presentation(conn, _presentation(memory_group))
    stored = read_presentation(conn, memory_group, "en")
    assert stored.timeline["title"] == "Beach day"
    assert stored.music_key == "aa/bb/aabb.wav"
    assert stored.status == "ready"


def test_read_returns_none_when_never_prepared(conn, memory_group):
    assert read_presentation(conn, memory_group, "en") is None


def test_saving_twice_replaces_rather_than_duplicates(conn, memory_group):
    save_presentation(conn, _presentation(memory_group))
    save_presentation(conn, _presentation(memory_group, signature="sig2"))
    assert read_presentation(conn, memory_group, "en").signature == "sig2"
    n = conn.execute("SELECT count(*) AS n FROM memory_presentations").fetchone()["n"]
    assert n == 1


def test_rebuilding_memories_cascades_the_presentation_away(conn, memory_group):
    # §11's atomic swap deletes the groups; the presentation must go with them, or a
    # new memory would inherit a stale feed keyed by a recycled row id.
    save_presentation(conn, _presentation(memory_group))
    replace_memories(conn, 1, [Memory("Other", "d", [1], "sig2")])
    n = conn.execute("SELECT count(*) AS n FROM memory_presentations").fetchone()["n"]
    assert n == 0


def test_group_id_for_album_key_is_the_inverse_of_album_key_for_group(conn, memory_group):
    assert group_id_for_album_key(conn, 1, "memory-0") == memory_group
    assert group_id_for_album_key(conn, 1, "memory-9") is None


def test_group_id_for_album_key_is_owner_scoped(conn, memory_group):
    assert group_id_for_album_key(conn, 2, "memory-0") is None


def test_prune_orphan_audio_deletes_only_unreferenced_blobs(conn, memory_group, tmp_path):
    audio = LocalStorage(tmp_path / "audio")
    audio.write("aa/bb/aabb.wav", b"kept")
    audio.write("ee/ff/eeff.wav", b"orphan")
    save_presentation(conn, _presentation(memory_group, narration_key=None))
    assert prune_orphan_audio(conn, audio) == 1
    assert audio.exists("aa/bb/aabb.wav")
    assert not audio.exists("ee/ff/eeff.wav")


def test_the_same_memory_can_hold_one_presentation_per_language(conn, memory_group):
    save_presentation(conn, _presentation(memory_group, language="en"))
    save_presentation(conn, _presentation(memory_group, language="ru"))
    assert read_presentation(conn, memory_group, "en").language == "en"
    assert read_presentation(conn, memory_group, "ru").language == "ru"
    n = conn.execute("SELECT count(*) AS n FROM memory_presentations").fetchone()["n"]
    assert n == 2


def test_a_rebuild_drops_every_language(conn, memory_group):
    save_presentation(conn, _presentation(memory_group, language="en"))
    save_presentation(conn, _presentation(memory_group, language="ru"))
    replace_memories(conn, 1, [Memory("Other", "d", [1], "sig2")])
    n = conn.execute("SELECT count(*) AS n FROM memory_presentations").fetchone()["n"]
    assert n == 0
