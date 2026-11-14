"""espeak-ng drops a data path over 160 characters and exits the process (§11.1)."""

import os
from pathlib import Path

from audio.espeak_path import ESPEAK_PATH_LIMIT, short_data_path


def _deep_data(root: Path, marker: bytes = b"data") -> Path:
    deep = root / ("x" * 60) / ("y" * 60) / ("z" * 60) / "espeak-ng-data"
    deep.mkdir(parents=True)
    (deep / "phontab").write_bytes(marker)
    return deep


def test_a_short_path_is_used_as_is(tmp_path):
    data = tmp_path / "espeak-ng-data"
    data.mkdir()
    assert short_data_path(str(data), copy_dir=str(tmp_path)) == str(data)


def test_a_long_path_is_replaced_by_a_short_real_copy(tmp_path):
    # A worktree venv put the bundled data at 166 characters; espeak-ng fell back to
    # its build machine's path and called exit(1), killing the voice child. A symlink
    # is not enough — phonemizer resolve()s the path straight back to the long one.
    deep = _deep_data(tmp_path)
    copy_dir = tmp_path / "c"
    copy_dir.mkdir()

    path = short_data_path(str(deep), copy_dir=str(copy_dir))

    assert len(str(Path(path).resolve())) < ESPEAK_PATH_LIMIT
    assert not os.path.islink(path)
    assert (Path(path) / "phontab").read_bytes() == b"data"


def test_the_copy_is_made_once_and_reused(tmp_path):
    deep = _deep_data(tmp_path)
    copy_dir = tmp_path / "c"
    copy_dir.mkdir()

    first = short_data_path(str(deep), copy_dir=str(copy_dir))
    (Path(first) / "phontab").write_bytes(b"already copied")
    second = short_data_path(str(deep), copy_dir=str(copy_dir))

    assert second == first
    assert (Path(second) / "phontab").read_bytes() == b"already copied"


def test_two_venvs_never_share_a_copy(tmp_path):
    a = _deep_data(tmp_path / "a", b"a")
    b = _deep_data(tmp_path / "b", b"b")
    copy_dir = tmp_path / "c"
    copy_dir.mkdir()

    assert short_data_path(str(a), copy_dir=str(copy_dir)) != short_data_path(
        str(b), copy_dir=str(copy_dir)
    )
