import pytest

from ingest.jobs import stage_counts
from ingest.receive import (
    HashMismatchError,
    UnreadableImageError,
    known_hashes,
    link_existing,
    receive,
)
from storage.keys import content_key
from storage.local import LocalStorage
from tests.factories import add_upload
from tests.fixtures import jpeg_bytes, jpeg_bytes_with_exif, sha


@pytest.fixture
def originals(tmp_path):
    return LocalStorage(tmp_path / "originals")


def test_receive_stores_the_original_under_its_content_key(conn, originals):
    upload_id = add_upload(conn)
    data = jpeg_bytes()
    result = receive(
        conn, originals, owner_id=1, upload_id=upload_id,
        rel_path="Pictures/one.JPG", declared_hash=sha(data), data=data,
    )
    assert result.created is True
    assert originals.read(content_key(sha(data), ".jpg")) == data


def test_receive_records_the_local_path_it_came_from(conn, originals):
    upload_id = add_upload(conn)
    data = jpeg_bytes()
    result = receive(
        conn, originals, owner_id=1, upload_id=upload_id,
        rel_path="Pictures/holiday/one.jpg", declared_hash=sha(data), data=data,
    )
    row = conn.execute(
        "SELECT rel_path, filename FROM photo_sources WHERE photo_id = ?", (result.photo_id,)
    ).fetchone()
    assert row["rel_path"] == "Pictures/holiday/one.jpg"
    assert row["filename"] == "one.jpg"


def test_receive_extracts_exif_and_derives_facets(conn, originals):
    upload_id = add_upload(conn)
    data = jpeg_bytes_with_exif()
    result = receive(
        conn, originals, owner_id=1, upload_id=upload_id,
        rel_path="a.jpg", declared_hash=sha(data), data=data,
    )
    row = conn.execute(
        "SELECT shot_at, camera, width, height, bytes FROM photos WHERE id = ?",
        (result.photo_id,),
    ).fetchone()
    assert row["shot_at"].startswith("2025-07-12")
    assert row["camera"] == "TestCam"
    assert row["width"] == 64
    assert row["bytes"] == len(data)
    keys = {
        r["key"]
        for r in conn.execute(
            "SELECT key FROM photo_facets WHERE photo_id = ?", (result.photo_id,)
        )
    }
    assert "year" in keys


def test_receive_queues_a_thumbnail(conn, originals):
    upload_id = add_upload(conn)
    data = jpeg_bytes()
    receive(
        conn, originals, owner_id=1, upload_id=upload_id,
        rel_path="a.jpg", declared_hash=sha(data), data=data,
    )
    assert stage_counts(conn, "thumbnail")["pending"] == 1


def test_receive_queues_an_embed(conn, originals):
    upload_id = add_upload(conn)
    data = jpeg_bytes()
    receive(
        conn, originals, owner_id=1, upload_id=upload_id,
        rel_path="a.jpg", declared_hash=sha(data), data=data,
    )
    assert stage_counts(conn, "embed")["pending"] == 1


def test_the_same_bytes_from_two_paths_make_one_photo_and_two_sources(conn, originals):
    upload_id = add_upload(conn)
    data = jpeg_bytes()
    first = receive(
        conn, originals, owner_id=1, upload_id=upload_id,
        rel_path="Pictures/a.jpg", declared_hash=sha(data), data=data,
    )
    second = receive(
        conn, originals, owner_id=1, upload_id=upload_id,
        rel_path="Desktop/a copy.jpg", declared_hash=sha(data), data=data,
    )
    assert second.photo_id == first.photo_id
    assert second.created is False
    assert second.source_added is True
    assert conn.execute("SELECT count(*) FROM photos").fetchone()[0] == 1
    assert conn.execute("SELECT count(*) FROM photo_sources").fetchone()[0] == 2
    assert stage_counts(conn, "thumbnail")["pending"] == 1


def test_the_same_path_twice_is_not_recorded_twice(conn, originals):
    upload_id = add_upload(conn)
    data = jpeg_bytes()
    result = None
    for _ in range(2):
        result = receive(
            conn, originals, owner_id=1, upload_id=upload_id,
            rel_path="Pictures/a.jpg", declared_hash=sha(data), data=data,
        )
    assert result.source_added is False
    assert conn.execute("SELECT count(*) FROM photo_sources").fetchone()[0] == 1


def test_bytes_that_do_not_match_the_declared_hash_are_rejected(conn, originals):
    upload_id = add_upload(conn)
    data = jpeg_bytes()
    with pytest.raises(HashMismatchError):
        receive(
            conn, originals, owner_id=1, upload_id=upload_id,
            rel_path="a.jpg", declared_hash="00" * 32, data=data,
        )
    assert conn.execute("SELECT count(*) FROM photos").fetchone()[0] == 0
    assert list(originals.iter_keys()) == []


def test_a_file_that_is_not_an_image_is_rejected_and_stores_nothing(conn, originals):
    upload_id = add_upload(conn)
    data = b"this is not a jpeg"
    with pytest.raises(UnreadableImageError):
        receive(
            conn, originals, owner_id=1, upload_id=upload_id,
            rel_path="notes.jpg", declared_hash=sha(data), data=data,
        )
    assert conn.execute("SELECT count(*) FROM photos").fetchone()[0] == 0
    assert list(originals.iter_keys()) == []


def test_known_hashes_returns_only_what_this_owner_already_has(conn, originals):
    upload_id = add_upload(conn)
    data = jpeg_bytes()
    receive(
        conn, originals, owner_id=1, upload_id=upload_id,
        rel_path="a.jpg", declared_hash=sha(data), data=data,
    )
    assert known_hashes(conn, 1, [sha(data), "ff" * 32]) == {sha(data)}
    assert known_hashes(conn, 2, [sha(data)]) == set()


def test_link_existing_records_a_path_without_any_bytes(conn, originals):
    upload_id = add_upload(conn)
    data = jpeg_bytes()
    first = receive(
        conn, originals, owner_id=1, upload_id=upload_id,
        rel_path="Pictures/a.jpg", declared_hash=sha(data), data=data,
    )
    photo_id = link_existing(
        conn, owner_id=1, upload_id=upload_id,
        rel_path="Backup/a.jpg", content_hash=sha(data),
    )
    assert photo_id == first.photo_id
    assert conn.execute("SELECT count(*) FROM photo_sources").fetchone()[0] == 2
    assert link_existing(
        conn, owner_id=1, upload_id=upload_id, rel_path="x.jpg", content_hash="ff" * 32
    ) is None
