"""A data path espeak-ng will actually accept (design §11.1).

espeak-ng copies its data path into a fixed 160-byte buffer (`N_PATH_HOME`). A longer
path is silently dropped, the library falls back to the path compiled into it — for
the `espeakng_loader` wheel, its build machine's `/Users/runner/…` — and `exit(1)`s
the whole process when `phontab` is not there. A worktree venv put the bundled data
at 166 characters, so the Kokoro child died mid-load with nothing but an EOFError.

A symlink does not help: phonemizer `resolve()`s the data path, which leads straight
back to the long one. So a long path is replaced by a real copy (~19 MB, made once).

Torch-free, so it is unit-tested directly.
"""

import hashlib
import os
import shutil
import tempfile
from pathlib import Path

# espeak-ng's N_PATH_HOME on non-Windows builds. Kept below it with room for the
# `/espeak-ng-data` suffix the library tries first.
ESPEAK_PATH_LIMIT = 160
_HEADROOM = len("/espeak-ng-data")


def short_data_path(data_path: str, copy_dir: str | None = None) -> str:
    """`data_path` itself when short enough, else a short real copy of it."""
    if len(str(Path(data_path).resolve())) + _HEADROOM < ESPEAK_PATH_LIMIT:
        return data_path
    # One copy per source: two venvs (a worktree and the main checkout) never share.
    digest = hashlib.sha1(data_path.encode()).hexdigest()[:10]
    copy = os.path.join(copy_dir or tempfile.gettempdir(), f"ivms777-espeak-{digest}")
    if os.path.isdir(copy):
        return copy
    # Copy beside it and rename into place: a concurrent child never sees half a copy.
    staging = f"{copy}.{os.getpid()}"
    shutil.rmtree(staging, ignore_errors=True)
    shutil.copytree(data_path, staging)
    try:
        os.rename(staging, copy)
    except OSError:  # another child won the race; its copy is complete
        shutil.rmtree(staging, ignore_errors=True)
    return copy
