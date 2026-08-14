import io

import pytest
from PIL import Image

from ingest.thumbs import make_thumbnails, thumb_key
from storage.local import LocalStorage
from tests.fixtures import make_jpeg


def test_thumb_key_shards_by_hash_prefix():
    assert thumb_key("abcdef1234", 320) == "ab/abcdef1234_320.jpg"


def test_makes_both_sizes_and_returns_grid_key(tmp_path):
    source = make_jpeg(tmp_path / "src.jpg", size=(2000, 1000))
    derived = LocalStorage(tmp_path / "thumbs")

    key = make_thumbnails(source, "abcdef1234", derived, grid_px=320, detail_px=1600)

    assert key == thumb_key("abcdef1234", 320)
    assert derived.exists(thumb_key("abcdef1234", 320))
    assert derived.exists(thumb_key("abcdef1234", 1600))


def test_thumbnail_fits_inside_the_box_and_keeps_aspect(tmp_path):
    source = make_jpeg(tmp_path / "src.jpg", size=(2000, 1000))
    derived = LocalStorage(tmp_path / "thumbs")

    make_thumbnails(source, "hash1", derived, grid_px=320, detail_px=1600)

    with Image.open(io.BytesIO(derived.read(thumb_key("hash1", 320)))) as image:
        assert max(image.size) <= 320
        assert image.size == (320, 160)


def test_small_source_is_not_upscaled(tmp_path):
    source = make_jpeg(tmp_path / "small.jpg", size=(50, 40))
    derived = LocalStorage(tmp_path / "thumbs")

    make_thumbnails(source, "hash2", derived, grid_px=320, detail_px=1600)

    with Image.open(io.BytesIO(derived.read(thumb_key("hash2", 320)))) as image:
        assert image.size == (50, 40)


def test_unreadable_source_raises(tmp_path):
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"not an image")
    with pytest.raises(OSError):
        make_thumbnails(broken, "hash3", LocalStorage(tmp_path / "thumbs"), 320, 1600)


def test_backfill_thumbnails_enqueues_only_for_photos_without_one(conn):
    from ingest.jobs import stage_counts
    from ingest.thumbs import backfill_thumbnails
    from tests.factories import add_photo

    add_photo(conn, photo_id=1, content_hash="a")                  # no thumb_key -> needs one
    add_photo(conn, photo_id=2, content_hash="b", thumb_key="2.jpg")  # already has one
    assert backfill_thumbnails(conn) == 1
    assert stage_counts(conn, "thumbnail")["pending"] == 1
    backfill_thumbnails(conn)  # idempotent — no duplicate job
    assert stage_counts(conn, "thumbnail")["pending"] == 1
