from ingest.exif import read_exif
from tests.fixtures import make_jpeg, make_jpeg_with_exif


def test_reads_datetime_and_camera(tmp_path):
    facts = read_exif(make_jpeg_with_exif(tmp_path / "a.jpg"))
    assert facts.shot_at == "2025-07-12T14:30:00"
    assert facts.camera == "TestCam"


def test_reads_dimensions_even_without_exif(tmp_path):
    facts = read_exif(make_jpeg(tmp_path / "b.jpg", size=(100, 20)))
    assert (facts.width, facts.height) == (100, 20)
    assert facts.shot_at is None


def test_unreadable_file_returns_empty_facts(tmp_path):
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"not an image")
    facts = read_exif(broken)
    assert facts.shot_at is None
    assert facts.width is None
    assert facts.raw == {}


def test_raw_captures_every_tag_by_name(tmp_path):
    facts = read_exif(make_jpeg_with_exif(tmp_path / "c.jpg"))
    assert facts.raw["Model"] == "TestCam"
    assert facts.raw["DateTimeOriginal"] == "2025:07:12 14:30:00"


def test_raw_is_json_serialisable(tmp_path):
    import json

    facts = read_exif(make_jpeg_with_exif(tmp_path / "d.jpg"))
    assert json.loads(json.dumps(facts.raw))["Model"] == "TestCam"
