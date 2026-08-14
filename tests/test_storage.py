import pytest

from storage.local import IMAGE_EXTENSIONS, LocalStorage


@pytest.fixture
def library(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "one.jpg").write_bytes(b"one")
    (tmp_path / "two.PNG").write_bytes(b"two")
    (tmp_path / "notes.txt").write_bytes(b"nope")
    (tmp_path / ".hidden.jpg").write_bytes(b"hidden")
    return tmp_path


def test_iter_keys_finds_images_recursively_and_ignores_others(library):
    storage = LocalStorage(library, extensions=IMAGE_EXTENSIONS)
    assert sorted(storage.iter_keys()) == ["a/one.jpg", "two.PNG"]


def test_iter_keys_without_filter_returns_every_file(library):
    storage = LocalStorage(library)
    assert "notes.txt" in set(storage.iter_keys())


def test_read_returns_bytes(library):
    assert LocalStorage(library).read("a/one.jpg") == b"one"


def test_write_creates_parent_directories(tmp_path):
    storage = LocalStorage(tmp_path)
    storage.write("deep/nested/file.bin", b"data")
    assert (tmp_path / "deep" / "nested" / "file.bin").read_bytes() == b"data"


def test_exists_and_stat(library):
    storage = LocalStorage(library)
    assert storage.exists("a/one.jpg")
    assert not storage.exists("a/missing.jpg")
    assert storage.stat("a/one.jpg").size == 3


def test_local_path_resolves_under_root(library):
    assert LocalStorage(library).local_path("a/one.jpg") == library / "a" / "one.jpg"


def test_escaping_the_root_is_rejected(library):
    storage = LocalStorage(library)
    with pytest.raises(ValueError):
        storage.read("../outside.jpg")
    with pytest.raises(ValueError):
        storage.write("/etc/passwd", b"x")


def test_free_bytes_reports_something_plausible(tmp_path):
    storage = LocalStorage(tmp_path / "originals")
    assert storage.free_bytes() > 0
