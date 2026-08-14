# Photo Library Organizer — Plan 01: Foundation and Ingest

> **PARTLY SUPERSEDED — completed 2026-08-13.** Tasks 3, 4, 7, 8 and 12 built a host-mounted folder scanner and a folder-picker UI. `docs/design.md` §3.2b now specifies browser upload instead, and `docs/plans/02-upload-ingest.md` replaces those tasks. This document is a snapshot of what was built, kept for its reasoning; it is not a description of the current app. Everything else it built — EXIF capture, facets, thumbnails, the job queue, the library grid, facet filters and sorting — is still current.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Point the app at a folder and get a browsable thumbnail grid of every photo in it — deduplicated by content hash, filterable and sortable by 25 EXIF facets — with a resumable job pipeline ready for the model stages.

**Architecture:** Three containers (`app`, `worker`, `inference`) over one SQLite file with `sqlite-vec` and FTS5. The worker drains per-photo, per-stage job rows library-wide in stage order, so restarts resume exactly where they stopped. Storage and inference sit behind protocols with fakes, so nothing in this plan needs a model or a network.

**Tech Stack:** Python 3.12, uv, FastAPI, Jinja2, HTMX, SQLite + sqlite-vec, Pillow + pillow-heif, pytest, Docker Compose.

**Spec:** `docs/design.md`

**Covers:** Spec phases 0 and 1. Phases 2-6 (embeddings, search, captions, planner, groups, chat) get their own plans.

## Global Constraints

- Python 3.12. Dependencies managed by `uv` with a committed `uv.lock`.
- Package name is `ivms777`. Source lives in `ivms777/`, tests in `tests/`.
- **Never run `git commit`, `git add`, or any staging command.** The user commits. Every task ends with a checkpoint where you report what changed and stop.
- Every user-scoped query filters on `owner_id`. `owner_id` is the constant `settings.owner_id` in v1 — there is no auth, no login, no user table, no admin role.
- Original photo files are never modified, moved, or renamed. The app opens them read-only.
- All I/O paths come from `Settings`. No hardcoded paths outside `ivms777/config.py`.
- Deploy profiles are `mac`, `jetson`, `cloud`. Profile changes config only, never code.
- Tests must not download model weights or make network calls. Use the fakes.
- SQLite connections open with `journal_mode=WAL` and `busy_timeout=5000`.

---

### Task 1: Project skeleton, settings, and profiles

**Files:**
- Create: `pyproject.toml`
- Create: `ivms777/__init__.py`
- Create: `ivms777/config.py`
- Create: `tests/__init__.py`
- Create: `tests/test_config.py`
- Create: `.gitignore`

**Interfaces:**
- Consumes: nothing.
- Produces: `ivms777.config.Settings` (pydantic-settings `BaseSettings`) with fields `profile: Literal["mac","jetson","cloud"]`, `data_dir: Path`, `library_root: Path | None`, `db_path: Path`, `thumb_dir: Path`, `inference_base_url: str`, `caption_model: str`, `planner_model: str`, `embed_device: Literal["cpu","cuda","mps"]`, `owner_id: int`, `thumb_grid_px: int`, `thumb_detail_px: int`, `page_size: int`. Also `ivms777.config.get_settings() -> Settings` (cached).

- [ ] **Step 1: Write the failing test**

Create `tests/test_config.py`:

```python
from pathlib import Path

from ivms777.config import Settings


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ivms777'`

- [ ] **Step 3: Create the project files**

Create `pyproject.toml`:

```toml
[project]
name = "ivms777"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.32",
    "jinja2>=3.1",
    "python-multipart>=0.0.12",
    "pydantic>=2.9",
    "pydantic-settings>=2.6",
    "pillow>=11.0",
    "pillow-heif>=0.20",
    "sqlite-vec>=0.1.6",
    "httpx>=0.27",
    "pyyaml>=6.0",
]

[dependency-groups]
dev = [
    "pytest>=8.3",
    "pytest-asyncio>=0.24",
    "ruff>=0.7",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["ivms777"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["slow: needs real model weights, deselected by default"]
addopts = "-m 'not slow'"

[tool.ruff]
line-length = 100
```

Create `.gitignore`:

```
__pycache__/
*.pyc
.venv/
data/
.pytest_cache/
.ruff_cache/
```

Create empty `ivms777/__init__.py` and `tests/__init__.py`.

Create `ivms777/config.py`:

```python
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Profile = Literal["mac", "jetson", "cloud"]

PROFILE_DEFAULTS: dict[Profile, dict[str, object]] = {
    "mac": {
        "caption_model": "gemma4:26b-a4b",
        "planner_model": "gemma4:e4b",
        "embed_device": "cpu",
        "inference_base_url": "http://host.docker.internal:11434/v1",
    },
    "jetson": {
        "caption_model": "qwen3-vl:4b",
        "planner_model": "qwen3-vl:4b",
        "embed_device": "cuda",
        "inference_base_url": "http://inference:11434/v1",
    },
    "cloud": {
        "caption_model": "gemma4:26b-a4b",
        "planner_model": "gemma4:e4b",
        "embed_device": "cuda",
        "inference_base_url": "http://inference:8000/v1",
    },
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="IVMS777_", extra="ignore")

    profile: Profile = "mac"
    data_dir: Path = Path("/data")
    library_root: Path | None = None

    caption_model: str | None = None
    planner_model: str | None = None
    embed_device: Literal["cpu", "cuda", "mps"] | None = None
    inference_base_url: str | None = None

    owner_id: int = 1
    thumb_grid_px: int = 320
    thumb_detail_px: int = 1600
    page_size: int = Field(default=100, ge=1, le=500)

    @model_validator(mode="after")
    def _apply_profile_defaults(self) -> "Settings":
        for key, value in PROFILE_DEFAULTS[self.profile].items():
            if getattr(self, key) is None:
                object.__setattr__(self, key, value)
        return self

    @property
    def db_path(self) -> Path:
        return self.data_dir / "ivms777.db"

    @property
    def thumb_dir(self) -> Path:
        return self.data_dir / "thumbs"


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_config.py -v`
Expected: 6 passed.

Note: the profile-default fields are declared `| None` so "unset" is distinguishable from "explicitly set". The `db_path`/`thumb_dir` properties return `Path`, and the test compares against `Path`, so this passes.

- [ ] **Step 5: Checkpoint**

Report: files created, test output. Do not commit — the user commits.

---

### Task 2: Database schema, connection, and migrations

**Files:**
- Create: `ivms777/db/__init__.py`
- Create: `ivms777/db/schema.sql`
- Create: `ivms777/db/connection.py`
- Create: `tests/conftest.py`
- Create: `tests/test_db.py`

**Interfaces:**
- Consumes: `ivms777.config.Settings`.
- Produces: `ivms777.db.connection.connect(db_path: Path) -> sqlite3.Connection` (WAL, busy_timeout, `sqlite_vec` loaded, `row_factory = sqlite3.Row`, foreign keys on) and `ivms777.db.connection.migrate(conn: sqlite3.Connection) -> None` (idempotent). Tables per spec section 6.

- [ ] **Step 1: Write the failing test**

Create `tests/conftest.py`:

```python
from pathlib import Path

import pytest

from ivms777.config import Settings
from ivms777.db.connection import connect, migrate


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path, library_root=tmp_path / "library")


@pytest.fixture
def conn(settings: Settings):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    c = connect(settings.db_path)
    migrate(c)
    yield c
    c.close()
```

Create `tests/test_db.py`:

```python
import sqlite3

import pytest

from ivms777.db.connection import connect, migrate


def test_wal_and_busy_timeout_are_set(conn):
    assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000


def test_sqlite_vec_extension_is_loaded(conn):
    version = conn.execute("SELECT vec_version()").fetchone()[0]
    assert isinstance(version, str)


def test_expected_tables_exist(conn):
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view')").fetchall()
    names = {r[0] for r in rows}
    expected_tables = [
        "photos", "tags", "photo_tags", "photo_facets",
        "jobs", "groups", "group_photos", "scans",
    ]
    for expected in expected_tables:
        assert expected in names


def test_vector_table_accepts_and_returns_a_row(conn):
    conn.execute("INSERT INTO photo_vec(rowid, embedding) VALUES (1, ?)", (b"\x00" * (1152 * 4),))
    count = conn.execute("SELECT count(*) FROM photo_vec").fetchone()[0]
    assert count == 1


def test_migrate_is_idempotent(settings):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    c = connect(settings.db_path)
    migrate(c)
    migrate(c)
    assert c.execute("SELECT count(*) FROM photos").fetchone()[0] == 0
    c.close()


def test_owner_path_uniqueness_is_enforced(conn):
    args = ("a/b.jpg", "hash1", "2026-01-01T00:00:00", "2026-01-01T00:00:00")
    conn.execute(
        "INSERT INTO photos(owner_id, path, content_hash, created_at, updated_at)"
        " VALUES (1, ?, ?, ?, ?)",
        args,
    )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO photos(owner_id, path, content_hash, created_at, updated_at)"
            " VALUES (1, ?, ?, ?, ?)",
            args,
        )


def test_duplicate_of_points_at_another_photo(conn):
    for photo_id, path in ((1, "a.jpg"), (2, "copy-of-a.jpg")):
        conn.execute(
            "INSERT INTO photos(id, owner_id, path, content_hash, created_at, updated_at)"
            " VALUES (?, 1, ?, 'samehash', '2026-01-01', '2026-01-01')",
            (photo_id, path),
        )
    conn.execute("UPDATE photos SET duplicate_of = 1 WHERE id = 2")
    row = conn.execute("SELECT duplicate_of FROM photos WHERE id = 2").fetchone()
    assert row["duplicate_of"] == 1


def test_deleting_a_photo_cascades_to_jobs(conn):
    conn.execute(
        "INSERT INTO photos(id, owner_id, path, content_hash, created_at, updated_at)"
        " VALUES (7, 1, 'x.jpg', 'h', '2026-01-01', '2026-01-01')"
    )
    conn.execute(
        "INSERT INTO jobs(photo_id, stage, status, updated_at)"
        " VALUES (7, 'thumbnail', 'pending', '2026-01-01')"
    )
    conn.execute("DELETE FROM photos WHERE id = 7")
    assert conn.execute("SELECT count(*) FROM jobs").fetchone()[0] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_db.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ivms777.db'`

- [ ] **Step 3: Write the schema and connection module**

Create empty `ivms777/db/__init__.py`.

Create `ivms777/db/schema.sql`:

```sql
CREATE TABLE IF NOT EXISTS photos (
  id              INTEGER PRIMARY KEY,
  owner_id        INTEGER NOT NULL,
  path            TEXT NOT NULL,
  content_hash    TEXT NOT NULL,
  phash           TEXT,
  bytes           INTEGER,
  width           INTEGER,
  height          INTEGER,
  mtime           REAL,
  shot_at         TEXT,
  camera          TEXT,
  lens            TEXT,
  gps_lat         REAL,
  gps_lon         REAL,
  thumb_key       TEXT,
  caption         TEXT,
  caption_model   TEXT,
  embedding_model TEXT,
  missing_since   TEXT,
  exif_json       TEXT,
  duplicate_of    INTEGER REFERENCES photos(id) ON DELETE SET NULL,
  created_at      TEXT NOT NULL,
  updated_at      TEXT NOT NULL,
  UNIQUE(owner_id, path)
);
CREATE INDEX IF NOT EXISTS photos_owner_hash ON photos(owner_id, content_hash);
CREATE INDEX IF NOT EXISTS photos_owner_shot ON photos(owner_id, shot_at);
CREATE INDEX IF NOT EXISTS photos_duplicate_of ON photos(duplicate_of);

CREATE TABLE IF NOT EXISTS photo_facets (
  photo_id   INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
  key        TEXT NOT NULL,
  value_text TEXT,
  value_num  REAL,
  PRIMARY KEY (photo_id, key)
);
CREATE INDEX IF NOT EXISTS photo_facets_lookup ON photo_facets(key, value_text);
CREATE INDEX IF NOT EXISTS photo_facets_range ON photo_facets(key, value_num);

CREATE VIRTUAL TABLE IF NOT EXISTS photo_vec USING vec0(embedding float[1152]);

CREATE TABLE IF NOT EXISTS tags (
  id        INTEGER PRIMARY KEY,
  dimension TEXT NOT NULL,
  label     TEXT NOT NULL,
  UNIQUE(dimension, label)
);

CREATE TABLE IF NOT EXISTS photo_tags (
  photo_id INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
  tag_id   INTEGER NOT NULL REFERENCES tags(id),
  score    REAL NOT NULL,
  source   TEXT NOT NULL,
  PRIMARY KEY (photo_id, tag_id, source)
);
CREATE INDEX IF NOT EXISTS photo_tags_tag ON photo_tags(tag_id);

CREATE TABLE IF NOT EXISTS jobs (
  photo_id   INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
  stage      TEXT NOT NULL,
  status     TEXT NOT NULL,
  attempts   INTEGER NOT NULL DEFAULT 0,
  error      TEXT,
  updated_at TEXT NOT NULL,
  PRIMARY KEY (photo_id, stage)
);
CREATE INDEX IF NOT EXISTS jobs_pending ON jobs(stage, status);

CREATE TABLE IF NOT EXISTS groups (
  id         INTEGER PRIMARY KEY,
  owner_id   INTEGER NOT NULL,
  kind       TEXT NOT NULL,
  name       TEXT NOT NULL,
  params     TEXT,
  status     TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS group_photos (
  group_id INTEGER NOT NULL REFERENCES groups(id) ON DELETE CASCADE,
  photo_id INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
  rank     REAL,
  PRIMARY KEY (group_id, photo_id)
);

CREATE TABLE IF NOT EXISTS scans (
  id          INTEGER PRIMARY KEY,
  owner_id    INTEGER NOT NULL,
  root_path   TEXT NOT NULL,
  started_at  TEXT NOT NULL,
  finished_at TEXT,
  files_seen  INTEGER DEFAULT 0,
  files_new   INTEGER DEFAULT 0
);

CREATE VIRTUAL TABLE IF NOT EXISTS photo_fts USING fts5(caption, tags_text);
```

Create `ivms777/db/connection.py`:

```python
import sqlite3
from pathlib import Path

import sqlite_vec

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # FastAPI runs sync route handlers in a threadpool, so the connection is used
    # from threads other than the one that created it. Python's sqlite3 serializes
    # calls internally (default SQLITE_THREADSAFE=1), and every statement here is
    # autocommitted, so sharing one connection is safe at this scale.
    conn = sqlite3.connect(db_path, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def migrate(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_PATH.read_text())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_db.py -v`
Expected: 8 passed.

If `test_vector_table_accepts_and_returns_a_row` fails on the blob length, confirm `sqlite-vec` expects `1152 * 4` bytes for `float[1152]` and adjust only the test's byte count — never the schema dimension, which must match SigLIP 2 `so400m`.

- [ ] **Step 5: Checkpoint**

Report: test output, and the `vec_version()` string so the extension version is on record.

---

### Task 3: Storage protocol and local filesystem backend

**Files:**
- Create: `ivms777/storage/__init__.py`
- Create: `ivms777/storage/base.py`
- Create: `ivms777/storage/local.py`
- Create: `tests/test_storage.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `ivms777.storage.base.StorageStat` (dataclass: `size: int`, `mtime: float`), `ivms777.storage.base.Storage` (Protocol with `iter_keys() -> Iterator[str]`, `read(key) -> bytes`, `write(key, data) -> None`, `exists(key) -> bool`, `stat(key) -> StorageStat`, `local_path(key) -> Path | None`), and `ivms777.storage.local.LocalStorage(root: Path, extensions: frozenset[str] | None = None)`.

Two instances are used by later tasks: one rooted at `settings.library_root` for originals, one at `settings.thumb_dir` for derived files. Keys are POSIX-style paths relative to the root.

- [ ] **Step 1: Write the failing test**

Create `tests/test_storage.py`:

```python
import pytest

from ivms777.storage.local import IMAGE_EXTENSIONS, LocalStorage


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_storage.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ivms777.storage'`

- [ ] **Step 3: Write the storage modules**

Create empty `ivms777/storage/__init__.py`.

Create `ivms777/storage/base.py`:

```python
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class StorageStat:
    size: int
    mtime: float


class Storage(Protocol):
    def iter_keys(self) -> Iterator[str]: ...
    def read(self, key: str) -> bytes: ...
    def write(self, key: str, data: bytes) -> None: ...
    def exists(self, key: str) -> bool: ...
    def stat(self, key: str) -> StorageStat: ...
    def local_path(self, key: str) -> Path | None: ...
```

Create `ivms777/storage/local.py`:

```python
from collections.abc import Iterator
from pathlib import Path

from ivms777.storage.base import StorageStat

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".tif", ".tiff"})


class LocalStorage:
    def __init__(self, root: Path, extensions: frozenset[str] | None = None) -> None:
        self.root = root.resolve()
        self.extensions = extensions

    def _resolve(self, key: str) -> Path:
        candidate = (self.root / key).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise ValueError(f"key escapes storage root: {key!r}")
        return candidate

    def iter_keys(self) -> Iterator[str]:
        for path in sorted(self.root.rglob("*")):
            if not path.is_file() or path.name.startswith("."):
                continue
            if self.extensions is not None and path.suffix.lower() not in self.extensions:
                continue
            yield path.relative_to(self.root).as_posix()

    def read(self, key: str) -> bytes:
        return self._resolve(key).read_bytes()

    def write(self, key: str, data: bytes) -> None:
        target = self._resolve(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    def exists(self, key: str) -> bool:
        return self._resolve(key).is_file()

    def stat(self, key: str) -> StorageStat:
        info = self._resolve(key).stat()
        return StorageStat(size=info.st_size, mtime=info.st_mtime)

    def local_path(self, key: str) -> Path | None:
        return self._resolve(key)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_storage.py -v`
Expected: 7 passed.

- [ ] **Step 5: Checkpoint**

Report: test output. Confirm the root-escape test passes — that guard is what keeps a malformed key from reading outside the library.

---

### Task 4: Scanner — walk, hash, EXIF, facets, upsert

**Files:**
- Create: `ivms777/ingest/__init__.py`
- Create: `ivms777/ingest/exif.py`
- Create: `ivms777/ingest/facets.py`
- Create: `ivms777/ingest/scanner.py`
- Create: `tests/fixtures.py`
- Create: `tests/test_exif.py`
- Create: `tests/test_facets.py`
- Create: `tests/test_scanner.py`

**Interfaces:**
- Consumes: `LocalStorage`, `connect`/`migrate`, `Settings`.
- Produces (facets): `ivms777.ingest.facets.Facet` (dataclass: `key: str`, `value_text: str | None`, `value_num: float | None`), `ivms777.ingest.facets.derive_facets(facts: ExifFacts, width: int | None, height: int | None) -> list[Facet]`, `ivms777.ingest.facets.store_facets(conn, photo_id: int, facets: list[Facet]) -> None`, and `ivms777.ingest.facets.FACET_KEYS: tuple[str, ...]`.
- Produces: `ivms777.ingest.exif.ExifFacts` (dataclass: `shot_at: str | None`, `camera: str | None`, `lens: str | None`, `gps_lat: float | None`, `gps_lon: float | None`, `width: int | None`, `height: int | None`), `ivms777.ingest.exif.read_exif(path: Path) -> ExifFacts`, `ivms777.ingest.scanner.sha256_file(path: Path) -> str`, `ivms777.ingest.scanner.ScanResult` (dataclass: `scan_id: int`, `seen: int`, `added: int`, `moved: int`, `missing: int`, `duplicates: int`), and `ivms777.ingest.scanner.scan(conn, storage, owner_id: int) -> ScanResult`.

Behaviour `scan` must implement:
- new key, new hash → insert a **canonical** photo row and enqueue its `thumbnail` job
- known key, unchanged hash → leave the row alone
- known hash at a new key, **and the original file is gone** → a move: update `path` in place, clear `missing_since`
- known hash at a new key, **and the original file still exists** → an exact duplicate: insert a row with `duplicate_of` set to the canonical photo's id, and enqueue **no jobs** for it
- key gone from storage → set `missing_since`, keep the row and its tags
- a returning file → clear `missing_since`
- canonical goes missing while a duplicate of it is still present → promote the lowest-id live duplicate to canonical, repoint the rest at it, and enqueue its `thumbnail` job

Duplicate detection is by sha256 of file bytes, so it catches identical images
under different names in different folders. It is exact-match only; visually
similar but not byte-identical photos are near-duplicates and are handled by
perceptual hashing in plan 05.

Deduplicating work is the point: only canonical rows ever get job rows, so an
identical photo stored in five folders is embedded and captioned once, not five
times.

`scan` works in two passes — hash every file first, then reconcile — because
telling a move from a duplicate requires knowing the full set of live files
before deciding.

- [ ] **Step 1: Write the failing test**

Create `tests/fixtures.py`:

```python
from pathlib import Path

from PIL import Image


def make_jpeg(path: Path, size: tuple[int, int] = (64, 48), color: str = "red") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path, format="JPEG")
    return path


def make_jpeg_with_exif(path: Path, when: str = "2025:07:12 14:30:00", model: str = "TestCam") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (64, 48), "blue")
    exif = image.getexif()
    exif[0x0110] = model          # Model
    exif[0x9003] = when           # DateTimeOriginal
    exif[0x010F] = "TestMake"     # Make
    image.save(path, format="JPEG", exif=exif)
    return path
```

Create `tests/test_exif.py`:

```python
from ivms777.ingest.exif import read_exif
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
```

Create `tests/test_facets.py`:

```python
from ivms777.ingest.exif import ExifFacts
from ivms777.ingest.facets import derive_facets, store_facets


def facet_map(facets):
    return {f.key: (f.value_text, f.value_num) for f in facets}


def test_time_facets_come_from_shot_at():
    facts = ExifFacts(shot_at="2025-07-12T20:30:00")  # a Saturday evening
    m = facet_map(derive_facets(facts, width=None, height=None))

    assert m["year"] == (None, 2025.0)
    assert m["month"] == (None, 7.0)
    assert m["hour"] == (None, 20.0)
    assert m["weekday"] == ("Saturday", None)
    assert m["time_of_day"] == ("evening", None)
    assert m["is_weekend"] == ("yes", None)


def test_time_of_day_buckets():
    def bucket(hour: int) -> str:
        facts = ExifFacts(shot_at=f"2025-07-12T{hour:02d}:00:00")
        return facet_map(derive_facets(facts, None, None))["time_of_day"][0]

    assert bucket(2) == "night"
    assert bucket(6) == "dawn"
    assert bucket(10) == "morning"
    assert bucket(15) == "afternoon"
    assert bucket(20) == "evening"
    assert bucket(23) == "night"


def test_exposure_facets_are_numeric():
    facts = ExifFacts(
        raw={"FNumber": 1.8, "ExposureTime": 0.005, "FocalLength": 35.0, "ISOSpeedRatings": 3200}
    )
    m = facet_map(derive_facets(facts, None, None))

    assert m["aperture"] == (None, 1.8)
    assert m["shutter_speed"] == (None, 0.005)
    assert m["focal_length"] == (None, 35.0)
    assert m["iso"] == (None, 3200.0)


def test_categorical_exposure_settings_are_named_not_numbered():
    facts = ExifFacts(raw={"Flash": 1, "WhiteBalance": 0, "MeteringMode": 5})
    m = facet_map(derive_facets(facts, None, None))

    assert m["flash"] == ("fired", None)
    assert m["white_balance"] == ("auto", None)
    assert m["metering_mode"] == ("pattern", None)


def test_camera_facets():
    facts = ExifFacts(camera="X-T5", lens="XF33mmF1.4", raw={"Make": "FUJIFILM"})
    m = facet_map(derive_facets(facts, None, None))

    assert m["camera_make"] == ("FUJIFILM", None)
    assert m["camera_model"] == ("X-T5", None)
    assert m["lens"] == ("XF33mmF1.4", None)


def test_image_shape_facets():
    m = facet_map(derive_facets(ExifFacts(), width=4000, height=3000))
    assert m["aspect"] == ("landscape", None)
    assert m["megapixels"][1] == 12.0

    portrait = facet_map(derive_facets(ExifFacts(), width=3000, height=4000))
    assert portrait["aspect"] == ("portrait", None)

    square = facet_map(derive_facets(ExifFacts(), width=1000, height=1000))
    assert square["aspect"] == ("square", None)


def test_gps_presence_is_a_facet():
    with_gps = facet_map(derive_facets(ExifFacts(gps_lat=51.5, gps_lon=-0.1), None, None))
    assert with_gps["has_gps"] == ("yes", None)
    assert with_gps["gps_lat"] == (None, 51.5)

    without = facet_map(derive_facets(ExifFacts(), None, None))
    assert without["has_gps"] == ("no", None)
    assert "gps_lat" not in without


def test_missing_exif_yields_only_the_facets_that_are_knowable():
    m = facet_map(derive_facets(ExifFacts(), None, None))
    assert set(m) == {"has_gps"}


def test_store_facets_replaces_previous_values(conn):
    conn.execute(
        "INSERT INTO photos(id, owner_id, path, content_hash, created_at, updated_at)"
        " VALUES (1, 1, 'a.jpg', 'h', '2026-01-01', '2026-01-01')"
    )
    store_facets(conn, 1, derive_facets(ExifFacts(camera="A"), None, None))
    store_facets(conn, 1, derive_facets(ExifFacts(camera="B"), None, None))

    rows = conn.execute(
        "SELECT value_text FROM photo_facets WHERE photo_id = 1 AND key = 'camera_model'"
    ).fetchall()
    assert [row["value_text"] for row in rows] == ["B"]
```

Create `tests/test_scanner.py`:

```python
from ivms777.ingest.scanner import scan, sha256_file
from ivms777.storage.local import IMAGE_EXTENSIONS, LocalStorage
from tests.fixtures import make_jpeg, make_jpeg_with_exif


def make_storage(root):
    root.mkdir(parents=True, exist_ok=True)
    return LocalStorage(root, extensions=IMAGE_EXTENSIONS)


def test_sha256_is_stable_and_content_based(tmp_path):
    a = make_jpeg(tmp_path / "a.jpg", color="red")
    b = make_jpeg(tmp_path / "b.jpg", color="red")
    assert sha256_file(a) == sha256_file(b)


def test_scan_inserts_rows_and_enqueues_thumbnail_jobs(conn, tmp_path):
    root = tmp_path / "lib"
    make_jpeg(root / "one.jpg")
    make_jpeg_with_exif(root / "sub" / "two.jpg")

    result = scan(conn, make_storage(root), owner_id=1)

    assert (result.seen, result.added) == (2, 2)
    assert conn.execute("SELECT count(*) FROM photos").fetchone()[0] == 2
    pending = conn.execute(
        "SELECT count(*) FROM jobs WHERE stage='thumbnail' AND status='pending'"
    ).fetchone()[0]
    assert pending == 2


def test_scan_records_exif_facts(conn, tmp_path):
    root = tmp_path / "lib"
    make_jpeg_with_exif(root / "shot.jpg")
    scan(conn, make_storage(root), owner_id=1)
    row = conn.execute("SELECT shot_at, camera, width FROM photos").fetchone()
    assert row["shot_at"] == "2025-07-12T14:30:00"
    assert row["camera"] == "TestCam"
    assert row["width"] == 64


def test_scan_stores_raw_exif_json(conn, tmp_path):
    import json

    root = tmp_path / "lib"
    make_jpeg_with_exif(root / "shot.jpg")
    scan(conn, make_storage(root), owner_id=1)
    raw = json.loads(conn.execute("SELECT exif_json FROM photos").fetchone()["exif_json"])
    assert raw["Model"] == "TestCam"


def test_scan_writes_facet_rows(conn, tmp_path):
    root = tmp_path / "lib"
    make_jpeg_with_exif(root / "shot.jpg")
    scan(conn, make_storage(root), owner_id=1)

    facets = {
        row["key"]: (row["value_text"], row["value_num"])
        for row in conn.execute("SELECT key, value_text, value_num FROM photo_facets")
    }
    assert facets["camera_model"] == ("TestCam", None)
    assert facets["camera_make"] == ("TestMake", None)
    assert facets["year"] == (None, 2025.0)
    assert facets["time_of_day"] == ("afternoon", None)
    assert facets["aspect"] == ("landscape", None)


def test_duplicates_get_their_own_facet_rows(conn, tmp_path):
    root = tmp_path / "lib"
    make_jpeg_with_exif(root / "a-shot.jpg")
    make_jpeg_with_exif(root / "z-shot.jpg")
    scan(conn, make_storage(root), owner_id=1)

    photo_ids = {row["photo_id"] for row in conn.execute("SELECT DISTINCT photo_id FROM photo_facets")}
    assert len(photo_ids) == 2


def test_rescan_is_idempotent(conn, tmp_path):
    root = tmp_path / "lib"
    make_jpeg(root / "one.jpg")
    storage = make_storage(root)
    scan(conn, storage, owner_id=1)
    second = scan(conn, storage, owner_id=1)
    assert second.added == 0
    assert conn.execute("SELECT count(*) FROM photos").fetchone()[0] == 1


def test_moved_file_updates_path_instead_of_duplicating(conn, tmp_path):
    root = tmp_path / "lib"
    original = make_jpeg(root / "one.jpg", color="green")
    storage = make_storage(root)
    scan(conn, storage, owner_id=1)

    original.rename(root / "moved" / "one.jpg" if (root / "moved").mkdir(exist_ok=True) is None else root / "moved" / "one.jpg")
    result = scan(conn, storage, owner_id=1)

    assert result.moved == 1
    assert conn.execute("SELECT count(*) FROM photos").fetchone()[0] == 1
    assert conn.execute("SELECT path FROM photos").fetchone()["path"] == "moved/one.jpg"


def test_deleted_file_is_marked_missing_but_kept(conn, tmp_path):
    root = tmp_path / "lib"
    path = make_jpeg(root / "one.jpg")
    storage = make_storage(root)
    scan(conn, storage, owner_id=1)

    path.unlink()
    result = scan(conn, storage, owner_id=1)

    assert result.missing == 1
    row = conn.execute("SELECT missing_since FROM photos").fetchone()
    assert row["missing_since"] is not None


def test_returning_file_clears_missing_since(conn, tmp_path):
    root = tmp_path / "lib"
    path = make_jpeg(root / "one.jpg", color="purple")
    storage = make_storage(root)
    scan(conn, storage, owner_id=1)
    path.unlink()
    scan(conn, storage, owner_id=1)

    make_jpeg(root / "one.jpg", color="purple")
    scan(conn, storage, owner_id=1)

    assert conn.execute("SELECT missing_since FROM photos").fetchone()["missing_since"] is None


def test_identical_file_under_a_different_name_is_a_duplicate(conn, tmp_path):
    # iter_keys() is sorted, so "a-original.jpg" is scanned first and becomes canonical.
    root = tmp_path / "lib"
    make_jpeg(root / "a-original.jpg", color="teal")
    make_jpeg(root / "z-backup" / "some-copy.jpg", color="teal")

    result = scan(conn, make_storage(root), owner_id=1)

    assert result.added == 1
    assert result.duplicates == 1
    assert conn.execute("SELECT count(*) FROM photos").fetchone()[0] == 2
    canonical = conn.execute("SELECT id, path FROM photos WHERE duplicate_of IS NULL").fetchone()
    dup = conn.execute(
        "SELECT path, duplicate_of FROM photos WHERE duplicate_of IS NOT NULL"
    ).fetchone()
    assert canonical["path"] == "a-original.jpg"
    assert dup["duplicate_of"] == canonical["id"]
    assert dup["path"] == "z-backup/some-copy.jpg"


def test_duplicates_are_not_queued_for_processing(conn, tmp_path):
    root = tmp_path / "lib"
    make_jpeg(root / "a-original.jpg", color="teal")
    make_jpeg(root / "z-copy.jpg", color="teal")

    scan(conn, make_storage(root), owner_id=1)

    assert conn.execute("SELECT count(*) FROM jobs").fetchone()[0] == 1


def test_three_copies_produce_one_canonical_and_two_duplicates(conn, tmp_path):
    root = tmp_path / "lib"
    for name in ("a.jpg", "b/b.jpg", "c/c/c.jpg"):
        make_jpeg(root / name, color="olive")

    result = scan(conn, make_storage(root), owner_id=1)

    assert (result.added, result.duplicates) == (1, 2)
    assert conn.execute("SELECT count(*) FROM jobs").fetchone()[0] == 1


def test_rescanning_duplicates_does_not_re_add_them(conn, tmp_path):
    root = tmp_path / "lib"
    make_jpeg(root / "a.jpg", color="olive")
    make_jpeg(root / "b.jpg", color="olive")
    storage = make_storage(root)
    scan(conn, storage, owner_id=1)

    second = scan(conn, storage, owner_id=1)

    assert (second.added, second.duplicates) == (0, 0)
    assert conn.execute("SELECT count(*) FROM photos").fetchone()[0] == 2


def test_a_true_move_is_not_counted_as_a_duplicate(conn, tmp_path):
    root = tmp_path / "lib"
    original = make_jpeg(root / "a.jpg", color="maroon")
    storage = make_storage(root)
    scan(conn, storage, owner_id=1)

    (root / "moved").mkdir(exist_ok=True)
    original.rename(root / "moved" / "a.jpg")
    result = scan(conn, storage, owner_id=1)

    assert (result.moved, result.duplicates) == (1, 0)
    assert conn.execute("SELECT count(*) FROM photos").fetchone()[0] == 1


def test_deleting_the_canonical_promotes_a_live_duplicate(conn, tmp_path):
    root = tmp_path / "lib"
    canonical_path = make_jpeg(root / "a.jpg", color="navy")
    make_jpeg(root / "b.jpg", color="navy")
    storage = make_storage(root)
    scan(conn, storage, owner_id=1)

    canonical_path.unlink()
    scan(conn, storage, owner_id=1)

    survivor = conn.execute(
        "SELECT path FROM photos WHERE duplicate_of IS NULL AND missing_since IS NULL"
    ).fetchone()
    assert survivor["path"] == "b.jpg"
    queued = conn.execute(
        "SELECT count(*) FROM jobs j JOIN photos p ON p.id = j.photo_id WHERE p.path = 'b.jpg'"
    ).fetchone()[0]
    assert queued == 1


def test_scan_row_is_recorded(conn, tmp_path):
    root = tmp_path / "lib"
    make_jpeg(root / "one.jpg")
    result = scan(conn, make_storage(root), owner_id=1)
    row = conn.execute("SELECT files_seen, files_new, finished_at FROM scans WHERE id=?", (result.scan_id,)).fetchone()
    assert (row["files_seen"], row["files_new"]) == (1, 1)
    assert row["finished_at"] is not None
```

In `test_moved_file_updates_path_instead_of_duplicating`, replace the awkward inline rename with two plain statements — create the directory, then rename:

```python
    (root / "moved").mkdir(exist_ok=True)
    original.rename(root / "moved" / "one.jpg")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_exif.py tests/test_facets.py tests/test_scanner.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ivms777.ingest'`

- [ ] **Step 3: Write the exif module**

Create empty `ivms777/ingest/__init__.py`.

Create `ivms777/ingest/exif.py`:

```python
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import pillow_heif
from PIL import ExifTags, Image

pillow_heif.register_heif_opener()

_DATETIME_ORIGINAL = 0x9003
_MODEL = 0x0110
_LENS_MODEL = 0xA434
_GPS_IFD = 0x8825
_EXIF_IFD = 0x8769


@dataclass(frozen=True)
class ExifFacts:
    shot_at: str | None = None
    camera: str | None = None
    lens: str | None = None
    gps_lat: float | None = None
    gps_lon: float | None = None
    width: int | None = None
    height: int | None = None
    raw: dict[str, object] = field(default_factory=dict)


def _jsonable(value: object) -> object:
    """EXIF values include IFDRational, bytes, and tuples. Make them JSON-safe."""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")[:200]
    if isinstance(value, Fraction) or hasattr(value, "numerator"):
        try:
            return float(value)  # type: ignore[arg-type]
        except (ZeroDivisionError, TypeError, ValueError):
            return None
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)[:200]


def _collect_raw(exif: Image.Exif) -> dict[str, object]:
    collected: dict[str, object] = {}
    for tag_id, value in exif.items():
        name = ExifTags.TAGS.get(tag_id, str(tag_id))
        collected[name] = _jsonable(value)
    try:
        for tag_id, value in exif.get_ifd(_EXIF_IFD).items():
            name = ExifTags.TAGS.get(tag_id, str(tag_id))
            collected[name] = _jsonable(value)
    except Exception:
        pass
    collected.pop("ExifOffset", None)
    collected.pop("GPSInfo", None)
    return collected


def _parse_datetime(raw: object) -> str | None:
    if not isinstance(raw, str) or len(raw) < 19:
        return None
    date, _, time = raw.partition(" ")
    return f"{date.replace(':', '-')}T{time}"


def _to_degrees(value: object) -> float | None:
    try:
        degrees, minutes, seconds = (float(part) for part in value)  # type: ignore[misc]
    except (TypeError, ValueError):
        return None
    return degrees + minutes / 60 + seconds / 3600


def _read_gps(exif: Image.Exif) -> tuple[float | None, float | None]:
    try:
        gps = exif.get_ifd(_GPS_IFD)
    except Exception:
        return None, None
    if not gps:
        return None, None
    lat = _to_degrees(gps.get(ExifTags.GPS.GPSLatitude))
    lon = _to_degrees(gps.get(ExifTags.GPS.GPSLongitude))
    if lat is not None and gps.get(ExifTags.GPS.GPSLatitudeRef) == "S":
        lat = -lat
    if lon is not None and gps.get(ExifTags.GPS.GPSLongitudeRef) == "W":
        lon = -lon
    return lat, lon


def read_exif(path: Path) -> ExifFacts:
    try:
        with Image.open(path) as image:
            width, height = image.size
            exif = image.getexif()
    except Exception:
        return ExifFacts()

    raw = _collect_raw(exif)
    lat, lon = _read_gps(exif)
    return ExifFacts(
        shot_at=_parse_datetime(raw.get("DateTimeOriginal") or exif.get(_DATETIME_ORIGINAL)),
        camera=exif.get(_MODEL) or None,
        lens=raw.get("LensModel") or exif.get(_LENS_MODEL) or None,
        gps_lat=lat,
        gps_lon=lon,
        width=width,
        height=height,
        raw=raw,
    )
```

- [ ] **Step 4: Write the facets module**

These are the "external filters" — exact facts from EXIF, kept in their own
table so they are never confused with model guesses. Numeric facets go in
`value_num` so ranges work; categorical ones go in `value_text`. A facet whose
source data is absent is simply not emitted, so `photo_facets` never contains
invented values.

Create `ivms777/ingest/facets.py`:

```python
import sqlite3
from dataclasses import dataclass
from datetime import datetime

from ivms777.ingest.exif import ExifFacts

FACET_KEYS: tuple[str, ...] = (
    "camera_make", "camera_model", "lens", "software",
    "iso", "aperture", "shutter_speed", "focal_length", "exposure_bias",
    "flash", "exposure_program", "metering_mode", "white_balance",
    "year", "month", "weekday", "hour", "time_of_day", "is_weekend",
    "has_gps", "gps_lat", "gps_lon",
    "megapixels", "orientation", "aspect",
)

WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")

EXPOSURE_PROGRAMS = {
    0: "not defined", 1: "manual", 2: "normal", 3: "aperture priority",
    4: "shutter priority", 5: "creative", 6: "action", 7: "portrait", 8: "landscape",
}
METERING_MODES = {
    0: "unknown", 1: "average", 2: "center weighted", 3: "spot",
    4: "multi spot", 5: "pattern", 6: "partial",
}
WHITE_BALANCES = {0: "auto", 1: "manual"}

NUMERIC_EXIF = {
    "aperture": ("FNumber", "ApertureValue"),
    "shutter_speed": ("ExposureTime",),
    "focal_length": ("FocalLength",),
    "exposure_bias": ("ExposureBiasValue",),
    "iso": ("ISOSpeedRatings", "PhotographicSensitivity", "ISO"),
}


@dataclass(frozen=True)
class Facet:
    key: str
    value_text: str | None = None
    value_num: float | None = None


def _time_of_day(hour: int) -> str:
    if hour < 5:
        return "night"
    if hour < 8:
        return "dawn"
    if hour < 12:
        return "morning"
    if hour < 17:
        return "afternoon"
    if hour < 21:
        return "evening"
    return "night"


def _first_number(raw: dict[str, object], names: tuple[str, ...]) -> float | None:
    for name in names:
        value = raw.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def derive_facets(facts: ExifFacts, width: int | None, height: int | None) -> list[Facet]:
    raw = facts.raw or {}
    out: list[Facet] = []

    def add_text(key: str, value: object) -> None:
        text = _text(value)
        if text is not None:
            out.append(Facet(key, value_text=text))

    def add_num(key: str, value: float | None) -> None:
        if value is not None:
            out.append(Facet(key, value_num=float(value)))

    add_text("camera_make", raw.get("Make"))
    add_text("camera_model", facts.camera)
    add_text("lens", facts.lens)
    add_text("software", raw.get("Software"))

    for key, names in NUMERIC_EXIF.items():
        add_num(key, _first_number(raw, names))

    flash = raw.get("Flash")
    if isinstance(flash, int) and not isinstance(flash, bool):
        add_text("flash", "fired" if flash & 1 else "did not fire")
    program = raw.get("ExposureProgram")
    if isinstance(program, int):
        add_text("exposure_program", EXPOSURE_PROGRAMS.get(program))
    metering = raw.get("MeteringMode")
    if isinstance(metering, int):
        add_text("metering_mode", METERING_MODES.get(metering))
    balance = raw.get("WhiteBalance")
    if isinstance(balance, int):
        add_text("white_balance", WHITE_BALANCES.get(balance))

    if facts.shot_at:
        try:
            when = datetime.fromisoformat(facts.shot_at)
        except ValueError:
            when = None
        if when is not None:
            add_num("year", when.year)
            add_num("month", when.month)
            add_num("hour", when.hour)
            add_text("weekday", WEEKDAYS[when.weekday()])
            add_text("time_of_day", _time_of_day(when.hour))
            add_text("is_weekend", "yes" if when.weekday() >= 5 else "no")

    has_gps = facts.gps_lat is not None and facts.gps_lon is not None
    add_text("has_gps", "yes" if has_gps else "no")
    if has_gps:
        add_num("gps_lat", facts.gps_lat)
        add_num("gps_lon", facts.gps_lon)

    orientation = raw.get("Orientation")
    if isinstance(orientation, int):
        add_num("orientation", orientation)
    if width and height:
        add_num("megapixels", round(width * height / 1_000_000, 1))
        if width > height:
            add_text("aspect", "landscape")
        elif width < height:
            add_text("aspect", "portrait")
        else:
            add_text("aspect", "square")

    return out


def store_facets(conn: sqlite3.Connection, photo_id: int, facets: list[Facet]) -> None:
    conn.execute("DELETE FROM photo_facets WHERE photo_id = ?", (photo_id,))
    conn.executemany(
        "INSERT INTO photo_facets(photo_id, key, value_text, value_num) VALUES (?, ?, ?, ?)",
        [(photo_id, f.key, f.value_text, f.value_num) for f in facets],
    )
```

Facets are written when a photo row is inserted. A re-scan of an unchanged file
does not recompute them, which is correct — EXIF does not change unless the file
does.

- [ ] **Step 5: Write the scanner module**

Create `ivms777/ingest/scanner.py`:

```python
import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from ivms777.ingest.exif import read_exif
from ivms777.ingest.facets import derive_facets, store_facets
from ivms777.storage.base import Storage

CHUNK = 1 << 20


def _now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ScanResult:
    scan_id: int
    seen: int = 0
    added: int = 0
    moved: int = 0
    missing: int = 0
    duplicates: int = 0


def _enqueue_thumbnail(conn: sqlite3.Connection, photo_id: int) -> None:
    conn.execute(
        "INSERT INTO jobs(photo_id, stage, status, updated_at)"
        " VALUES (?, 'thumbnail', 'pending', ?) ON CONFLICT(photo_id, stage) DO NOTHING",
        (photo_id, _now()),
    )


def _insert_photo(
    conn: sqlite3.Connection,
    owner_id: int,
    key: str,
    digest: str,
    local: Path,
    storage: Storage,
    duplicate_of: int | None,
) -> int:
    facts = read_exif(local)
    stat = storage.stat(key)
    now = _now()
    cursor = conn.execute(
        "INSERT INTO photos(owner_id, path, content_hash, bytes, width, height, mtime,"
        " shot_at, camera, lens, gps_lat, gps_lon, exif_json, duplicate_of,"
        " created_at, updated_at)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            owner_id, key, digest, stat.size, facts.width, facts.height, stat.mtime,
            facts.shot_at, facts.camera, facts.lens, facts.gps_lat, facts.gps_lon,
            json.dumps(facts.raw), duplicate_of, now, now,
        ),
    )
    photo_id = int(cursor.lastrowid)
    store_facets(conn, photo_id, derive_facets(facts, width=facts.width, height=facts.height))
    return photo_id


def _promote_orphaned_duplicates(conn: sqlite3.Connection, owner_id: int) -> None:
    """If a canonical photo went missing but a duplicate of it is still present,
    make the surviving copy canonical so its content still gets processed."""
    orphaned = conn.execute(
        "SELECT DISTINCT c.id FROM photos c JOIN photos d ON d.duplicate_of = c.id"
        " WHERE c.owner_id = ? AND c.missing_since IS NOT NULL AND d.missing_since IS NULL",
        (owner_id,),
    ).fetchall()
    for row in orphaned:
        heir = conn.execute(
            "SELECT id FROM photos WHERE duplicate_of = ? AND missing_since IS NULL"
            " ORDER BY id LIMIT 1",
            (row["id"],),
        ).fetchone()
        if heir is None:
            continue
        conn.execute(
            "UPDATE photos SET duplicate_of = NULL, updated_at = ? WHERE id = ?",
            (_now(), heir["id"]),
        )
        conn.execute(
            "UPDATE photos SET duplicate_of = ?, updated_at = ? WHERE duplicate_of = ? AND id != ?",
            (heir["id"], _now(), row["id"], heir["id"]),
        )
        conn.execute(
            "UPDATE photos SET duplicate_of = ?, updated_at = ? WHERE id = ?",
            (heir["id"], _now(), row["id"]),
        )
        _enqueue_thumbnail(conn, heir["id"])


def scan(conn: sqlite3.Connection, storage: Storage, owner_id: int) -> ScanResult:
    cursor = conn.execute(
        "INSERT INTO scans(owner_id, root_path, started_at) VALUES (?, ?, ?)",
        (owner_id, str(getattr(storage, "root", "")), _now()),
    )
    scan_id = int(cursor.lastrowid)

    # Pass 1 — hash everything before deciding anything. Telling a move from a
    # duplicate needs the complete set of live files.
    observed: dict[str, tuple[str, Path]] = {}
    for key in storage.iter_keys():
        local = storage.local_path(key)
        if local is None or not local.is_file():
            continue
        observed[key] = (sha256_file(local), local)

    existing = {
        row["path"]: row
        for row in conn.execute(
            "SELECT id, path, content_hash, missing_since, duplicate_of FROM photos"
            " WHERE owner_id = ?",
            (owner_id,),
        )
    }

    # Canonical photo id per hash, seeded from rows that are not themselves duplicates.
    canonical: dict[str, int] = {
        row["content_hash"]: row["id"]
        for row in existing.values()
        if row["duplicate_of"] is None
    }

    # Pass 2 — reconcile.
    added = moved = duplicates = 0
    for key, (digest, local) in observed.items():
        known = existing.get(key)
        if known is not None:
            if known["content_hash"] != digest or known["missing_since"] is not None:
                conn.execute(
                    "UPDATE photos SET content_hash = ?, missing_since = NULL, updated_at = ?"
                    " WHERE id = ?",
                    (digest, _now(), known["id"]),
                )
                canonical.setdefault(digest, known["id"])
            continue

        holder = canonical.get(digest)
        if holder is not None:
            holder_path = conn.execute(
                "SELECT path FROM photos WHERE id = ?", (holder,)
            ).fetchone()["path"]
            if holder_path not in observed:
                conn.execute(
                    "UPDATE photos SET path = ?, missing_since = NULL, updated_at = ?"
                    " WHERE id = ?",
                    (key, _now(), holder),
                )
                moved += 1
            else:
                _insert_photo(conn, owner_id, key, digest, local, storage, duplicate_of=holder)
                duplicates += 1
            continue

        photo_id = _insert_photo(conn, owner_id, key, digest, local, storage, duplicate_of=None)
        _enqueue_thumbnail(conn, photo_id)
        canonical[digest] = photo_id
        added += 1

    missing = 0
    for row in conn.execute(
        "SELECT id, path FROM photos WHERE owner_id = ? AND missing_since IS NULL", (owner_id,)
    ).fetchall():
        if row["path"] in observed:
            continue
        conn.execute(
            "UPDATE photos SET missing_since = ?, updated_at = ? WHERE id = ?",
            (_now(), _now(), row["id"]),
        )
        missing += 1

    _promote_orphaned_duplicates(conn, owner_id)

    conn.execute(
        "UPDATE scans SET finished_at = ?, files_seen = ?, files_new = ? WHERE id = ?",
        (_now(), len(observed), added, scan_id),
    )
    return ScanResult(
        scan_id=scan_id,
        seen=len(observed),
        added=added,
        moved=moved,
        missing=missing,
        duplicates=duplicates,
    )
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_exif.py tests/test_facets.py tests/test_scanner.py -v`
Expected: 30 passed.

Note that `_insert_photo` needs `from pathlib import Path` — it is already
imported at the top of the module for `sha256_file`.

- [ ] **Step 7: Checkpoint**

Report: test output. Call out the move-vs-duplicate results specifically — the
distinction between "same content, original gone" (move) and "same content,
original still there" (duplicate) is the subtlest logic in this plan and the
most likely to regress.

---

### Task 5: Thumbnails

**Files:**
- Create: `ivms777/ingest/thumbs.py`
- Create: `tests/test_thumbs.py`

**Interfaces:**
- Consumes: `Storage`, `Settings`.
- Produces: `ivms777.ingest.thumbs.thumb_key(content_hash: str, size: int) -> str` and `ivms777.ingest.thumbs.make_thumbnails(source: Path, content_hash: str, derived: Storage, grid_px: int, detail_px: int) -> str` returning the grid thumbnail key.

- [ ] **Step 1: Write the failing test**

Create `tests/test_thumbs.py`:

```python
import io

import pytest
from PIL import Image

from ivms777.ingest.thumbs import make_thumbnails, thumb_key
from ivms777.storage.local import LocalStorage
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_thumbs.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ivms777.ingest.thumbs'`

- [ ] **Step 3: Write the implementation**

Create `ivms777/ingest/thumbs.py`:

```python
import io
from pathlib import Path

import pillow_heif
from PIL import Image, ImageOps

from ivms777.storage.base import Storage

pillow_heif.register_heif_opener()


def thumb_key(content_hash: str, size: int) -> str:
    return f"{content_hash[:2]}/{content_hash}_{size}.jpg"


def _render(image: Image.Image, box: int) -> bytes:
    copy = image.copy()
    copy.thumbnail((box, box), Image.LANCZOS)
    if copy.mode != "RGB":
        copy = copy.convert("RGB")
    buffer = io.BytesIO()
    copy.save(buffer, format="JPEG", quality=85, optimize=True)
    return buffer.getvalue()


def make_thumbnails(
    source: Path,
    content_hash: str,
    derived: Storage,
    grid_px: int,
    detail_px: int,
) -> str:
    with Image.open(source) as image:
        image.load()
        upright = ImageOps.exif_transpose(image)
        for box in (grid_px, detail_px):
            derived.write(thumb_key(content_hash, box), _render(upright, box))
    return thumb_key(content_hash, grid_px)
```

`Image.thumbnail` never upscales, which is what `test_small_source_is_not_upscaled` asserts. `exif_transpose` applies the orientation tag so portrait photos are not shown sideways.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_thumbs.py -v`
Expected: 5 passed.

- [ ] **Step 5: Checkpoint**

Report: test output.

---

### Task 6: Job queue and worker driver

**Files:**
- Create: `ivms777/ingest/jobs.py`
- Create: `ivms777/ingest/worker.py`
- Create: `tests/test_jobs.py`
- Create: `tests/test_worker.py`

**Interfaces:**
- Consumes: `connect`/`migrate`, `Storage`, `Settings`.
- Produces:
  - `ivms777.ingest.jobs.STAGES: tuple[str, ...] = ("thumbnail", "embed", "taxonomy", "caption")`
  - `ivms777.ingest.jobs.MAX_ATTEMPTS: int = 3`
  - `ivms777.ingest.jobs.enqueue(conn, photo_id: int, stage: str) -> None`
  - `ivms777.ingest.jobs.claim_next(conn, stage: str) -> int | None`
  - `ivms777.ingest.jobs.complete(conn, photo_id: int, stage: str) -> None`
  - `ivms777.ingest.jobs.fail(conn, photo_id: int, stage: str, error: str) -> None`
  - `ivms777.ingest.jobs.stage_counts(conn, stage: str) -> dict[str, int]`
  - `ivms777.ingest.worker.StageHandler` (Protocol: `__call__(conn, photo_id: int) -> None`)
  - `ivms777.ingest.worker.drain(conn, handlers: dict[str, StageHandler], stages=STAGES) -> dict[str, int]` returning completed counts per stage
  - `ivms777.ingest.worker.thumbnail_handler(originals: Storage, derived: Storage, grid_px: int, detail_px: int) -> StageHandler`

`drain` processes stages in order, exhausting each stage across the whole library before starting the next. This is what keeps SigLIP and the captioner from being resident together on an 8 GB Jetson (spec section 8).

- [ ] **Step 1: Write the failing test**

Create `tests/test_jobs.py`:

```python
from ivms777.ingest.jobs import (
    MAX_ATTEMPTS,
    claim_next,
    complete,
    enqueue,
    fail,
    stage_counts,
)


def insert_photo(conn, path="a.jpg", digest="h1"):
    cursor = conn.execute(
        "INSERT INTO photos(owner_id, path, content_hash, created_at, updated_at)"
        " VALUES (1, ?, ?, '2026-01-01', '2026-01-01')",
        (path, digest),
    )
    return cursor.lastrowid


def test_claim_returns_pending_photo_and_marks_it_running(conn):
    photo_id = insert_photo(conn)
    enqueue(conn, photo_id, "thumbnail")

    assert claim_next(conn, "thumbnail") == photo_id
    status = conn.execute(
        "SELECT status FROM jobs WHERE photo_id=? AND stage='thumbnail'", (photo_id,)
    ).fetchone()["status"]
    assert status == "running"


def test_claim_returns_none_when_nothing_pending(conn):
    assert claim_next(conn, "thumbnail") is None


def test_claimed_job_is_not_returned_twice(conn):
    photo_id = insert_photo(conn)
    enqueue(conn, photo_id, "thumbnail")
    claim_next(conn, "thumbnail")
    assert claim_next(conn, "thumbnail") is None


def test_complete_marks_done(conn):
    photo_id = insert_photo(conn)
    enqueue(conn, photo_id, "thumbnail")
    claim_next(conn, "thumbnail")
    complete(conn, photo_id, "thumbnail")
    assert stage_counts(conn, "thumbnail")["done"] == 1


def test_failure_retries_then_sticks_at_failed(conn):
    photo_id = insert_photo(conn)
    enqueue(conn, photo_id, "thumbnail")

    for _ in range(MAX_ATTEMPTS):
        assert claim_next(conn, "thumbnail") == photo_id
        fail(conn, photo_id, "thumbnail", "boom")

    assert claim_next(conn, "thumbnail") is None
    counts = stage_counts(conn, "thumbnail")
    assert counts["failed"] == 1
    row = conn.execute(
        "SELECT attempts, error FROM jobs WHERE photo_id=? AND stage='thumbnail'", (photo_id,)
    ).fetchone()
    assert row["attempts"] == MAX_ATTEMPTS
    assert row["error"] == "boom"


def test_enqueue_is_idempotent(conn):
    photo_id = insert_photo(conn)
    enqueue(conn, photo_id, "thumbnail")
    enqueue(conn, photo_id, "thumbnail")
    assert stage_counts(conn, "thumbnail")["pending"] == 1


def test_stage_counts_reports_every_status_key(conn):
    counts = stage_counts(conn, "caption")
    assert counts == {"pending": 0, "running": 0, "done": 0, "failed": 0}
```

Create `tests/test_worker.py`:

```python
from ivms777.ingest.jobs import claim_next, enqueue, stage_counts
from ivms777.ingest.scanner import scan
from ivms777.ingest.thumbs import thumb_key
from ivms777.ingest.worker import drain, thumbnail_handler
from ivms777.storage.local import IMAGE_EXTENSIONS, LocalStorage
from tests.fixtures import make_jpeg


def test_drain_runs_handler_and_marks_done(conn):
    conn.execute(
        "INSERT INTO photos(id, owner_id, path, content_hash, created_at, updated_at)"
        " VALUES (1, 1, 'a.jpg', 'h', '2026-01-01', '2026-01-01')"
    )
    enqueue(conn, 1, "thumbnail")
    seen: list[int] = []

    completed = drain(conn, {"thumbnail": lambda c, pid: seen.append(pid)})

    assert seen == [1]
    assert completed["thumbnail"] == 1
    assert stage_counts(conn, "thumbnail")["done"] == 1


def test_drain_records_failure_without_stopping_the_queue(conn):
    for index in (1, 2):
        conn.execute(
            "INSERT INTO photos(id, owner_id, path, content_hash, created_at, updated_at)"
            " VALUES (?, 1, ?, ?, '2026-01-01', '2026-01-01')",
            (index, f"{index}.jpg", f"h{index}"),
        )
        enqueue(conn, index, "thumbnail")

    def handler(c, photo_id):
        if photo_id == 1:
            raise OSError("cannot read")

    drain(conn, {"thumbnail": handler})

    counts = stage_counts(conn, "thumbnail")
    assert counts["done"] == 1
    assert counts["pending"] == 1  # photo 1 retries; it has attempts left
    assert claim_next(conn, "thumbnail") == 1


def test_drain_finishes_each_stage_before_starting_the_next(conn):
    for index in (1, 2):
        conn.execute(
            "INSERT INTO photos(id, owner_id, path, content_hash, created_at, updated_at)"
            " VALUES (?, 1, ?, ?, '2026-01-01', '2026-01-01')",
            (index, f"{index}.jpg", f"h{index}"),
        )
        enqueue(conn, index, "thumbnail")
        enqueue(conn, index, "embed")

    order: list[str] = []
    drain(
        conn,
        {
            "thumbnail": lambda c, pid: order.append(f"thumbnail:{pid}"),
            "embed": lambda c, pid: order.append(f"embed:{pid}"),
        },
    )

    assert order == ["thumbnail:1", "thumbnail:2", "embed:1", "embed:2"]


def test_thumbnail_handler_writes_files_and_sets_thumb_key(conn, tmp_path):
    root = tmp_path / "lib"
    make_jpeg(root / "one.jpg", size=(800, 600))
    originals = LocalStorage(root, extensions=IMAGE_EXTENSIONS)
    derived = LocalStorage(tmp_path / "thumbs")
    scan(conn, originals, owner_id=1)

    drain(conn, {"thumbnail": thumbnail_handler(originals, derived, 320, 1600)})

    row = conn.execute("SELECT content_hash, thumb_key FROM photos").fetchone()
    assert row["thumb_key"] == thumb_key(row["content_hash"], 320)
    assert derived.exists(row["thumb_key"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_jobs.py tests/test_worker.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ivms777.ingest.jobs'`

- [ ] **Step 3: Write the jobs module**

Create `ivms777/ingest/jobs.py`:

```python
import sqlite3
from datetime import UTC, datetime

STAGES: tuple[str, ...] = ("thumbnail", "embed", "taxonomy", "caption")
MAX_ATTEMPTS = 3
STATUSES = ("pending", "running", "done", "failed")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def enqueue(conn: sqlite3.Connection, photo_id: int, stage: str) -> None:
    conn.execute(
        "INSERT INTO jobs(photo_id, stage, status, updated_at) VALUES (?, ?, 'pending', ?)"
        " ON CONFLICT(photo_id, stage) DO NOTHING",
        (photo_id, stage, _now()),
    )


def claim_next(
    conn: sqlite3.Connection, stage: str, exclude: set[int] | None = None
) -> int | None:
    """Claim the lowest-id pending photo for `stage`.

    `exclude` skips photos already attempted in this pass. A failed job returns to
    'pending' while attempts remain, so without it a single failing photo would be
    re-claimed forever and starve the rest of the queue.
    """
    if exclude:
        placeholders = ", ".join("?" for _ in exclude)
        row = conn.execute(
            "SELECT photo_id FROM jobs WHERE stage = ? AND status = 'pending'"
            f" AND photo_id NOT IN ({placeholders}) ORDER BY photo_id LIMIT 1",
            (stage, *exclude),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT photo_id FROM jobs WHERE stage = ? AND status = 'pending'"
            " ORDER BY photo_id LIMIT 1",
            (stage,),
        ).fetchone()
    if row is None:
        return None
    conn.execute(
        "UPDATE jobs SET status = 'running', updated_at = ? WHERE photo_id = ? AND stage = ?",
        (_now(), row["photo_id"], stage),
    )
    return int(row["photo_id"])


def complete(conn: sqlite3.Connection, photo_id: int, stage: str) -> None:
    conn.execute(
        "UPDATE jobs SET status = 'done', error = NULL, updated_at = ?"
        " WHERE photo_id = ? AND stage = ?",
        (_now(), photo_id, stage),
    )


def fail(conn: sqlite3.Connection, photo_id: int, stage: str, error: str) -> None:
    conn.execute(
        "UPDATE jobs SET attempts = attempts + 1,"
        " status = CASE WHEN attempts + 1 >= ? THEN 'failed' ELSE 'pending' END,"
        " error = ?, updated_at = ? WHERE photo_id = ? AND stage = ?",
        (MAX_ATTEMPTS, error, _now(), photo_id, stage),
    )


def stage_counts(conn: sqlite3.Connection, stage: str) -> dict[str, int]:
    counts = dict.fromkeys(STATUSES, 0)
    for row in conn.execute(
        "SELECT status, count(*) AS n FROM jobs WHERE stage = ? GROUP BY status", (stage,)
    ):
        counts[row["status"]] = row["n"]
    return counts
```

- [ ] **Step 4: Write the worker module**

Create `ivms777/ingest/worker.py`:

```python
import sqlite3
from pathlib import Path
from typing import Protocol

from ivms777.ingest.jobs import STAGES, claim_next, complete, fail
from ivms777.ingest.thumbs import make_thumbnails
from ivms777.storage.base import Storage


class StageHandler(Protocol):
    def __call__(self, conn: sqlite3.Connection, photo_id: int) -> None: ...


def drain(
    conn: sqlite3.Connection,
    handlers: dict[str, StageHandler],
    stages: tuple[str, ...] = STAGES,
) -> dict[str, int]:
    completed: dict[str, int] = {}
    for stage in stages:
        handler = handlers.get(stage)
        if handler is None:
            continue
        done = 0
        attempted: set[int] = set()
        while (photo_id := claim_next(conn, stage, exclude=attempted)) is not None:
            attempted.add(photo_id)
            try:
                handler(conn, photo_id)
            except Exception as error:  # one bad file must never stall the queue
                fail(conn, photo_id, stage, str(error))
            else:
                complete(conn, photo_id, stage)
                done += 1
        completed[stage] = done
    return completed


def thumbnail_handler(
    originals: Storage,
    derived: Storage,
    grid_px: int,
    detail_px: int,
) -> StageHandler:
    def handle(conn: sqlite3.Connection, photo_id: int) -> None:
        row = conn.execute(
            "SELECT path, content_hash FROM photos WHERE id = ?", (photo_id,)
        ).fetchone()
        source: Path | None = originals.local_path(row["path"])
        if source is None or not source.is_file():
            raise FileNotFoundError(row["path"])
        key = make_thumbnails(source, row["content_hash"], derived, grid_px, detail_px)
        conn.execute("UPDATE photos SET thumb_key = ? WHERE id = ?", (key, photo_id))

    return handle
```

The `exclude` parameter on `claim_next` is what makes
`test_drain_records_failure_without_stopping_the_queue` pass. `fail` puts the job
back to `pending` while attempts remain, and since `claim_next` orders by
`photo_id`, a failing low-id photo would otherwise be re-claimed on every
iteration and the rest of the queue would never run. Skipping it *within this
pass* — rather than breaking out of the loop — lets the remaining photos proceed
while the failed one stays `pending` for the next pass.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_jobs.py tests/test_worker.py -v`
Expected: 11 passed.

- [ ] **Step 6: Checkpoint**

Report: test output, and confirm no test hangs.

---

### Task 7: FastAPI app, library grid, and thumbnail serving

**Files:**
- Create: `ivms777/web/__init__.py`
- Create: `ivms777/web/deps.py`
- Create: `ivms777/web/app.py`
- Create: `ivms777/web/templates/base.html`
- Create: `ivms777/web/templates/library.html`
- Create: `ivms777/web/templates/_grid_page.html`
- Create: `ivms777/web/static/app.css`
- Create: `tests/test_web_library.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `ivms777.web.app.create_app(settings: Settings) -> FastAPI` with routes `GET /` (redirect to `/library`), `GET /library`, `GET /library/page?offset=`, `GET /thumb/{photo_id}`. Also `ivms777.web.deps.AppContext` (dataclass: `settings`, `conn`, `originals`, `derived`).

- [ ] **Step 1: Write the failing test**

Create `tests/test_web_library.py`:

```python
import pytest
from fastapi.testclient import TestClient

from ivms777.ingest.worker import drain, thumbnail_handler
from ivms777.ingest.scanner import scan
from ivms777.storage.local import IMAGE_EXTENSIONS, LocalStorage
from ivms777.web.app import create_app
from tests.fixtures import make_jpeg


@pytest.fixture
def client(settings, tmp_path):
    root = settings.library_root
    for index in range(3):
        make_jpeg(root / f"photo{index}.jpg", color=["red", "green", "blue"][index])
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    app = create_app(settings)
    context = app.state.context
    originals = LocalStorage(root, extensions=IMAGE_EXTENSIONS)
    scan(context.conn, originals, owner_id=settings.owner_id)
    drain(
        context.conn,
        {"thumbnail": thumbnail_handler(originals, context.derived, 320, 1600)},
    )
    with TestClient(app) as test_client:
        yield test_client


def test_root_redirects_to_library(client):
    response = client.get("/", follow_redirects=False)
    assert response.status_code in (302, 307)
    assert response.headers["location"] == "/library"


def test_library_page_renders_a_tile_per_photo(client):
    response = client.get("/library")
    assert response.status_code == 200
    assert response.text.count('class="tile"') == 3


def test_grid_page_returns_a_fragment_not_a_full_document(client):
    response = client.get("/library/page?offset=0")
    assert response.status_code == 200
    assert "<html" not in response.text
    assert 'class="tile"' in response.text


def test_grid_page_respects_offset(client):
    response = client.get("/library/page?offset=3")
    assert 'class="tile"' not in response.text


def test_thumbnail_is_served_as_jpeg(client):
    photo_id = client.app.state.context.conn.execute("SELECT id FROM photos LIMIT 1").fetchone()["id"]
    response = client.get(f"/thumb/{photo_id}")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content[:2] == b"\xff\xd8"


def test_unknown_thumbnail_is_404(client):
    assert client.get("/thumb/99999").status_code == 404


def test_missing_photos_are_hidden_from_the_grid(client):
    context = client.app.state.context
    context.conn.execute("UPDATE photos SET missing_since = '2026-01-01' WHERE id = (SELECT min(id) FROM photos)")
    response = client.get("/library")
    assert response.text.count('class="tile"') == 2


def test_duplicates_are_hidden_and_shown_as_a_badge(client):
    context = client.app.state.context
    ids = [row["id"] for row in context.conn.execute("SELECT id FROM photos ORDER BY id")]
    context.conn.execute("UPDATE photos SET duplicate_of = ? WHERE id = ?", (ids[0], ids[1]))

    response = client.get("/library")

    assert response.text.count('class="tile"') == 2
    assert "×2" in response.text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_web_library.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ivms777.web'`

- [ ] **Step 3: Write the app context and application**

Create empty `ivms777/web/__init__.py`.

Create `ivms777/web/deps.py`:

```python
import sqlite3
from dataclasses import dataclass

from ivms777.config import Settings
from ivms777.storage.local import IMAGE_EXTENSIONS, LocalStorage


@dataclass
class AppContext:
    settings: Settings
    conn: sqlite3.Connection
    originals: LocalStorage
    derived: LocalStorage


def build_context(settings: Settings) -> AppContext:
    from ivms777.db.connection import connect, migrate

    conn = connect(settings.db_path)
    migrate(conn)
    root = settings.library_root or settings.data_dir / "library"
    root.mkdir(parents=True, exist_ok=True)
    return AppContext(
        settings=settings,
        conn=conn,
        originals=LocalStorage(root, extensions=IMAGE_EXTENSIONS),
        derived=LocalStorage(settings.thumb_dir),
    )
```

Create `ivms777/web/app.py`:

```python
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ivms777.config import Settings
from ivms777.web.deps import AppContext, build_context

TEMPLATE_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"

LIST_SQL = (
    "SELECT p.id, p.path, p.caption, p.shot_at,"
    " (SELECT count(*) FROM photos d WHERE d.duplicate_of = p.id) AS dupe_count"
    " FROM photos p"
    " WHERE p.owner_id = ? AND p.missing_since IS NULL AND p.thumb_key IS NOT NULL"
    " AND p.duplicate_of IS NULL"
    " ORDER BY COALESCE(p.shot_at, p.created_at) DESC, p.id DESC LIMIT ? OFFSET ?"
)


def create_app(settings: Settings) -> FastAPI:
    app = FastAPI(title="ivms777")
    app.state.context = build_context(settings)
    templates = Jinja2Templates(directory=str(TEMPLATE_DIR))
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    def context() -> AppContext:
        return app.state.context

    def fetch_page(offset: int) -> list:
        ctx = context()
        return list(
            ctx.conn.execute(LIST_SQL, (ctx.settings.owner_id, ctx.settings.page_size, offset))
        )

    @app.get("/")
    def root() -> RedirectResponse:
        return RedirectResponse("/library")

    @app.get("/library", response_class=HTMLResponse)
    def library(request: Request) -> HTMLResponse:
        rows = fetch_page(0)
        return templates.TemplateResponse(
            request,
            "library.html",
            {"photos": rows, "next_offset": len(rows), "page_size": context().settings.page_size},
        )

    @app.get("/library/page", response_class=HTMLResponse)
    def library_page(request: Request, offset: int = 0) -> HTMLResponse:
        rows = fetch_page(offset)
        return templates.TemplateResponse(
            request,
            "_grid_page.html",
            {
                "photos": rows,
                "next_offset": offset + len(rows),
                "page_size": context().settings.page_size,
            },
        )

    @app.get("/thumb/{photo_id}")
    def thumb(photo_id: int) -> Response:
        ctx = context()
        row = ctx.conn.execute(
            "SELECT thumb_key FROM photos WHERE id = ? AND owner_id = ?",
            (photo_id, ctx.settings.owner_id),
        ).fetchone()
        if row is None or row["thumb_key"] is None or not ctx.derived.exists(row["thumb_key"]):
            raise HTTPException(status_code=404)
        return Response(ctx.derived.read(row["thumb_key"]), media_type="image/jpeg")

    return app
```

- [ ] **Step 4: Write the templates and stylesheet**

Create `ivms777/web/templates/base.html`:

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{% block title %}ivms777{% endblock %}</title>
    <link rel="stylesheet" href="/static/app.css">
    <script src="/static/htmx.min.js"></script>
  </head>
  <body>
    <nav class="nav">
      <a href="/library">Library</a>
      <a href="/index">Index</a>
    </nav>
    <main>{% block content %}{% endblock %}</main>
  </body>
</html>
```

HTMX is served from `/static`, never a CDN — the app must work with no internet,
and a CDN script tag without Subresource Integrity is a supply-chain risk.
Download it now:

```bash
curl -fsSL https://unpkg.com/htmx.org@2.0.3/dist/htmx.min.js -o ivms777/web/static/htmx.min.js
```

Create `ivms777/web/templates/_grid_page.html`:

```html
{% for photo in photos %}
  <a class="tile" href="/photo/{{ photo['id'] }}">
    <img src="/thumb/{{ photo['id'] }}" alt="{{ photo['caption'] or photo['path'] }}" loading="lazy">
    {% if photo['dupe_count'] %}
      <span class="tile-badge" title="{{ photo['dupe_count'] }} duplicate file(s) on disk">
        ×{{ photo['dupe_count'] + 1 }}
      </span>
    {% endif %}
    <span class="tile-caption">{{ photo['caption'] or photo['path'] }}</span>
  </a>
{% endfor %}
{% if photos|length == page_size %}
  <div hx-get="/library/page?offset={{ next_offset }}"
       hx-trigger="revealed"
       hx-swap="outerHTML"></div>
{% endif %}
```

Create `ivms777/web/templates/library.html`:

```html
{% extends "base.html" %}
{% block content %}
<div class="grid">
  {% include "_grid_page.html" %}
</div>
{% endblock %}
```

Create `ivms777/web/static/app.css`:

```css
* { box-sizing: border-box; }
body { margin: 0; font: 15px/1.4 system-ui, sans-serif; background: #111; color: #eee; }
.nav { display: flex; gap: 1rem; padding: 0.75rem 1rem; background: #1c1c1c; }
.nav a { color: #eee; text-decoration: none; }
main { padding: 1rem; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 8px; }
.tile { position: relative; display: block; aspect-ratio: 1; overflow: hidden; border-radius: 6px; background: #222; }
.tile img { width: 100%; height: 100%; object-fit: cover; display: block; }
.tile-caption {
  position: absolute; inset: auto 0 0 0; padding: 6px 8px; font-size: 12px;
  background: linear-gradient(transparent, rgba(0,0,0,0.85)); opacity: 0; transition: opacity 0.15s;
}
.tile:hover .tile-caption { opacity: 1; }
.tile-badge {
  position: absolute; top: 6px; right: 6px; padding: 2px 6px; border-radius: 10px;
  font-size: 11px; font-weight: 600; background: rgba(0,0,0,0.7); color: #ffd479;
}
.dupe-set { margin-bottom: 1.5rem; display: flex; gap: 12px; align-items: flex-start; }
.dupe-set img { width: 120px; height: 120px; object-fit: cover; border-radius: 6px; }
.dupe-paths { font-size: 13px; }
.dupe-paths li { margin: 2px 0; }
.dupe-canonical { color: #9fe6a0; }
.progress { display: grid; gap: 6px; max-width: 480px; }
.progress-row { display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid #333; }
.failed { color: #ff8080; }
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_web_library.py -v`
Expected: 8 passed.

- [ ] **Step 6: Checkpoint**

Report: test output. Note that `/photo/{id}` is linked but not yet implemented — it arrives in plan 02.

---

### Task 8: Index screen — start a scan and watch progress

**Files:**
- Modify: `ivms777/web/app.py` (add routes; keep existing ones unchanged)
- Create: `ivms777/web/templates/index.html`
- Create: `ivms777/web/templates/_progress.html`
- Create: `tests/test_web_index.py`

**Interfaces:**
- Consumes: `scan`, `drain`, `thumbnail_handler`, `stage_counts`, `AppContext`.
- Produces: routes `GET /index`, `POST /index/scan`, `GET /index/progress`. Also `ivms777.web.app.run_ingest(context: AppContext) -> None`, which runs a scan then drains the thumbnail stage — synchronous and directly callable from tests.

`POST /index/scan` schedules `run_ingest` on a FastAPI `BackgroundTasks` so the request returns immediately.

- [ ] **Step 1: Write the failing test**

Create `tests/test_web_index.py`:

```python
import pytest
from fastapi.testclient import TestClient

from ivms777.web.app import create_app, run_ingest
from tests.fixtures import make_jpeg


@pytest.fixture
def client(settings):
    for index in range(2):
        make_jpeg(settings.library_root / f"p{index}.jpg", color=["red", "green"][index])
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def test_index_page_renders(client):
    response = client.get("/index")
    assert response.status_code == 200
    assert "Start scan" in response.text


def test_run_ingest_populates_photos_and_thumbnails(client):
    context = client.app.state.context
    run_ingest(context)
    row = context.conn.execute("SELECT count(*) AS n FROM photos").fetchone()
    assert row["n"] == 2
    done = context.conn.execute(
        "SELECT count(*) AS n FROM jobs WHERE stage='thumbnail' AND status='done'"
    ).fetchone()
    assert done["n"] == 2


def test_post_scan_returns_progress_fragment(client):
    response = client.post("/index/scan")
    assert response.status_code == 200
    assert "thumbnail" in response.text


def test_progress_reports_counts_per_stage(client):
    run_ingest(client.app.state.context)
    response = client.get("/index/progress")
    assert response.status_code == 200
    assert "thumbnail" in response.text
    assert "2" in response.text


def test_progress_lists_failed_files(client):
    context = client.app.state.context
    run_ingest(context)
    context.conn.execute(
        "UPDATE jobs SET status='failed', error='cannot read' WHERE stage='thumbnail'"
    )
    response = client.get("/index/progress")
    assert "cannot read" in response.text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_web_index.py -v`
Expected: FAIL with `ImportError: cannot import name 'run_ingest'`

- [ ] **Step 3: Add the ingest routes**

In `ivms777/web/app.py`, add these imports at the top:

```python
from fastapi import BackgroundTasks

from ivms777.ingest.jobs import STAGES, stage_counts
from ivms777.ingest.scanner import scan
from ivms777.ingest.worker import drain, thumbnail_handler
```

Add this module-level function above `create_app`:

```python
def run_ingest(context: AppContext) -> None:
    settings = context.settings
    scan(context.conn, context.originals, owner_id=settings.owner_id)
    drain(
        context.conn,
        {
            "thumbnail": thumbnail_handler(
                context.originals,
                context.derived,
                settings.thumb_grid_px,
                settings.thumb_detail_px,
            )
        },
    )
```

Inside `create_app`, before `return app`, add:

```python
    def progress_payload() -> dict:
        ctx = context()
        failures = list(
            ctx.conn.execute(
                "SELECT p.path, j.stage, j.error FROM jobs j JOIN photos p ON p.id = j.photo_id"
                " WHERE j.status = 'failed' ORDER BY p.path LIMIT 50"
            )
        )
        return {
            "stages": [(stage, stage_counts(ctx.conn, stage)) for stage in STAGES],
            "failures": failures,
            "library_root": str(ctx.originals.root),
        }

    @app.get("/index", response_class=HTMLResponse)
    def index_page(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(request, "index.html", progress_payload())

    @app.post("/index/scan", response_class=HTMLResponse)
    def start_scan(request: Request, background: BackgroundTasks) -> HTMLResponse:
        background.add_task(run_ingest, context())
        return templates.TemplateResponse(request, "_progress.html", progress_payload())

    @app.get("/index/progress", response_class=HTMLResponse)
    def progress(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(request, "_progress.html", progress_payload())
```

- [ ] **Step 4: Write the templates**

Create `ivms777/web/templates/_progress.html`:

```html
<div class="progress" hx-get="/index/progress" hx-trigger="every 2s" hx-swap="outerHTML">
  {% for stage, counts in stages %}
    <div class="progress-row">
      <span>{{ stage }}</span>
      <span>
        {{ counts['done'] }} done · {{ counts['pending'] }} pending
        {% if counts['failed'] %}<span class="failed">· {{ counts['failed'] }} failed</span>{% endif %}
      </span>
    </div>
  {% endfor %}
  {% if failures %}
    <h3 class="failed">Failed files</h3>
    <ul>
      {% for failure in failures %}
        <li class="failed">{{ failure['path'] }} — {{ failure['stage'] }}: {{ failure['error'] }}</li>
      {% endfor %}
    </ul>
  {% endif %}
</div>
```

Create `ivms777/web/templates/index.html`:

```html
{% extends "base.html" %}
{% block content %}
<h1>Index</h1>
<p>Library root: <code>{{ library_root }}</code></p>
<form hx-post="/index/scan" hx-target="#progress" hx-swap="outerHTML">
  <button type="submit">Start scan</button>
</form>
<div id="progress">
  {% include "_progress.html" %}
</div>
{% endblock %}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_web_index.py -v`
Expected: 5 passed.

Then run the whole suite: `uv run pytest -v`
Expected: all tests pass.

- [ ] **Step 6: Checkpoint**

Report: full suite output.

---

### Task 9: Inference client and fake

**Files:**
- Create: `ivms777/inference/__init__.py`
- Create: `ivms777/inference/client.py`
- Create: `ivms777/inference/fakes.py`
- Create: `tests/test_inference_client.py`

**Interfaces:**
- Consumes: `Settings`.
- Produces:
  - `ivms777.inference.client.ChatMessage` (TypedDict: `role: str`, `content: object`)
  - `ivms777.inference.client.InferenceClient` (Protocol: `complete(model: str, messages: list[ChatMessage], *, json_schema: dict | None = None, timeout: float = 120.0) -> str`)
  - `ivms777.inference.client.OpenAICompatClient(base_url: str, api_key: str = "unused")` implementing it
  - `ivms777.inference.client.encode_image(data: bytes, mime: str = "image/jpeg") -> str` returning a `data:` URI
  - `ivms777.inference.fakes.FakeInferenceClient(responses: list[str])` recording `calls: list[tuple[str, list[ChatMessage]]]` and returning queued responses in order

Both Ollama and vLLM speak the OpenAI chat-completions API, so one client covers every profile.

- [ ] **Step 1: Write the failing test**

Create `tests/test_inference_client.py`:

```python
import httpx
import pytest

from ivms777.inference.client import OpenAICompatClient, encode_image
from ivms777.inference.fakes import FakeInferenceClient


def test_encode_image_produces_a_data_uri():
    assert encode_image(b"abc").startswith("data:image/jpeg;base64,")


def test_fake_client_returns_queued_responses_and_records_calls():
    client = FakeInferenceClient(["first", "second"])
    messages = [{"role": "user", "content": "hi"}]

    assert client.complete("m", messages) == "first"
    assert client.complete("m", messages) == "second"
    assert len(client.calls) == 2
    assert client.calls[0][0] == "m"


def test_fake_client_raises_when_exhausted():
    client = FakeInferenceClient([])
    with pytest.raises(AssertionError):
        client.complete("m", [{"role": "user", "content": "hi"}])


def test_client_posts_to_chat_completions_and_returns_content():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["content-type"].startswith("application/json")
        return httpx.Response(200, json={"choices": [{"message": {"content": "a caption"}}]})

    transport = httpx.MockTransport(handler)
    client = OpenAICompatClient("http://inference/v1", transport=transport)

    result = client.complete("gemma4:e4b", [{"role": "user", "content": "describe"}])
    assert result == "a caption"


def test_client_passes_json_schema_as_response_format():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    client = OpenAICompatClient("http://inference/v1", transport=httpx.MockTransport(handler))
    schema = {"type": "object", "properties": {"caption": {"type": "string"}}}
    client.complete("m", [{"role": "user", "content": "x"}], json_schema=schema)

    assert captured["response_format"]["type"] == "json_schema"
    assert captured["response_format"]["json_schema"]["schema"] == schema


def test_client_raises_on_http_error():
    client = OpenAICompatClient(
        "http://inference/v1",
        transport=httpx.MockTransport(lambda request: httpx.Response(500, text="boom")),
    )
    with pytest.raises(httpx.HTTPStatusError):
        client.complete("m", [{"role": "user", "content": "x"}])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_inference_client.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ivms777.inference'`

- [ ] **Step 3: Write the client and fake**

Create empty `ivms777/inference/__init__.py`.

Create `ivms777/inference/client.py`:

```python
import base64
from typing import Protocol, TypedDict

import httpx


class ChatMessage(TypedDict):
    role: str
    content: object


def encode_image(data: bytes, mime: str = "image/jpeg") -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


class InferenceClient(Protocol):
    def complete(
        self,
        model: str,
        messages: list[ChatMessage],
        *,
        json_schema: dict | None = None,
        timeout: float = 120.0,
    ) -> str: ...


class OpenAICompatClient:
    def __init__(
        self,
        base_url: str,
        api_key: str = "unused",
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            transport=transport,
        )

    def complete(
        self,
        model: str,
        messages: list[ChatMessage],
        *,
        json_schema: dict | None = None,
        timeout: float = 120.0,
    ) -> str:
        payload: dict = {"model": model, "messages": messages, "stream": False}
        if json_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": json_schema, "strict": True},
            }
        response = self._client.post("/chat/completions", json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
```

`base_url` already ends in `/v1`, and the request path is `/chat/completions`, so the test's assertion on `/v1/chat/completions` holds.

Create `ivms777/inference/fakes.py`:

```python
from ivms777.inference.client import ChatMessage


class FakeInferenceClient:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, list[ChatMessage]]] = []

    def complete(
        self,
        model: str,
        messages: list[ChatMessage],
        *,
        json_schema: dict | None = None,
        timeout: float = 120.0,
    ) -> str:
        self.calls.append((model, messages))
        assert self._responses, "FakeInferenceClient ran out of queued responses"
        return self._responses.pop(0)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_inference_client.py -v`
Expected: 6 passed.

- [ ] **Step 5: Checkpoint**

Report: test output.

---

### Task 10: Docker Compose for all three profiles

**Files:**
- Create: `Dockerfile`
- Create: `compose.yaml`
- Create: `compose.mac.yaml`
- Create: `compose.jetson.yaml`
- Create: `compose.cloud.yaml`
- Create: `.dockerignore`
- Create: `README.md`

**Interfaces:**
- Consumes: the `ivms777` package and `Settings` env var names (`IVMS777_*`).
- Produces: `app` and `worker` services in the base file; `inference` added by the `jetson` and `cloud` overrides. No new Python interfaces.

This task has no unit tests — it is verified by actually starting the stack.

- [ ] **Step 1: Write the Dockerfile and dockerignore**

Create `.dockerignore`:

```
.venv/
data/
.git/
.pytest_cache/
.ruff_cache/
__pycache__/
```

Create `Dockerfile`:

```dockerfile
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
      libheif1 libjpeg62-turbo zlib1g \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock* ./
RUN uv sync --no-dev --frozen || uv sync --no-dev

COPY ivms777 ./ivms777

ENV PYTHONUNBUFFERED=1 IVMS777_DATA_DIR=/data
EXPOSE 8000

CMD ["uv", "run", "uvicorn", "ivms777.web.app:create_app", \
     "--factory", "--host", "0.0.0.0", "--port", "8000"]
```

`create_app` takes a `Settings` argument, and `--factory` calls it with none. Fix this now by adding a zero-argument factory to `ivms777/web/app.py`:

```python
def app_factory() -> FastAPI:
    from ivms777.config import get_settings

    return create_app(get_settings())
```

and change the Dockerfile `CMD` target to `ivms777.web.app:app_factory`.

- [ ] **Step 2: Write the compose files**

Create `compose.yaml`:

```yaml
services:
  app:
    build: .
    environment:
      IVMS777_PROFILE: ${IVMS777_PROFILE:-mac}
      IVMS777_DATA_DIR: /data
      IVMS777_LIBRARY_ROOT: /library
    volumes:
      - ivms777-data:/data
      - ${PHOTO_LIBRARY:?set PHOTO_LIBRARY to your photo folder}:/library:ro
    ports:
      - "8000:8000"

  worker:
    build: .
    command: ["uv", "run", "python", "-m", "ivms777.ingest.cli"]
    environment:
      IVMS777_PROFILE: ${IVMS777_PROFILE:-mac}
      IVMS777_DATA_DIR: /data
      IVMS777_LIBRARY_ROOT: /library
    volumes:
      - ivms777-data:/data
      - ${PHOTO_LIBRARY:?set PHOTO_LIBRARY to your photo folder}:/library:ro

volumes:
  ivms777-data:
```

Create `compose.mac.yaml`:

```yaml
services:
  app:
    environment:
      IVMS777_PROFILE: mac
      IVMS777_INFERENCE_BASE_URL: http://host.docker.internal:11434/v1
    extra_hosts:
      - "host.docker.internal:host-gateway"
  worker:
    environment:
      IVMS777_PROFILE: mac
      IVMS777_INFERENCE_BASE_URL: http://host.docker.internal:11434/v1
    extra_hosts:
      - "host.docker.internal:host-gateway"
```

Create `compose.jetson.yaml`:

```yaml
services:
  inference:
    image: ollama/ollama:latest
    runtime: nvidia
    environment:
      OLLAMA_HOST: 0.0.0.0
    volumes:
      - ollama-models:/root/.ollama
    ports:
      - "11434:11434"

  app:
    environment:
      IVMS777_PROFILE: jetson
    depends_on: [inference]
  worker:
    environment:
      IVMS777_PROFILE: jetson
    depends_on: [inference]

volumes:
  ollama-models:
```

Create `compose.cloud.yaml`:

```yaml
services:
  inference:
    image: vllm/vllm-openai:latest
    command: ["--model", "${VLLM_MODEL:?set VLLM_MODEL}", "--port", "8000"]
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    volumes:
      - hf-cache:/root/.cache/huggingface
    ports:
      - "8001:8000"

  app:
    environment:
      IVMS777_PROFILE: cloud
    depends_on: [inference]
  worker:
    environment:
      IVMS777_PROFILE: cloud
    depends_on: [inference]

volumes:
  hf-cache:
```

- [ ] **Step 3: Write the worker entrypoint**

The `worker` service needs the module its `command` references. Create `ivms777/ingest/cli.py`:

```python
import time

from ivms777.config import get_settings
from ivms777.web.app import run_ingest
from ivms777.web.deps import build_context

POLL_SECONDS = 10


def main() -> None:
    context = build_context(get_settings())
    while True:
        run_ingest(context)
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Write the README**

Create `README.md`:

````markdown
# ivms777

Local photo library organizer. Classifies, searches, and groups a photo folder
using local models. See `docs/design.md` for the design and `docs/plans/` for
implementation plans.

## Run on a Mac

Docker Desktop on macOS cannot reach the Apple GPU, so Ollama runs natively on
the host and the app runs in containers.

```bash
brew install ollama
ollama serve &
ollama pull gemma4:26b-a4b
ollama pull gemma4:e4b

export PHOTO_LIBRARY=/path/to/your/photos
docker compose -f compose.yaml -f compose.mac.yaml up --build
```

Open http://localhost:8000/index and press "Start scan".

## Run on a Jetson Orin Nano

```bash
export PHOTO_LIBRARY=/path/to/your/photos
docker compose -f compose.yaml -f compose.jetson.yaml up --build
docker compose exec inference ollama pull qwen3-vl:4b
```

## Run on a cloud GPU box

```bash
export PHOTO_LIBRARY=/path/to/your/photos
export VLLM_MODEL=google/gemma-4-26b-a4b-it
docker compose -f compose.yaml -f compose.cloud.yaml up --build
```

## Develop

```bash
uv sync
uv run pytest
uv run ruff check .
```
````

- [ ] **Step 5: Verify the stack starts**

Run:

```bash
uv lock
export PHOTO_LIBRARY=$(pwd)/testdata
mkdir -p testdata
docker compose -f compose.yaml -f compose.mac.yaml up --build -d app
curl -sf http://localhost:8000/library > /dev/null && echo "app is serving"
docker compose -f compose.yaml -f compose.mac.yaml down
```

Expected: `app is serving`. If the build fails on `uv sync --frozen`, run `uv lock` first and rebuild.

- [ ] **Step 6: Checkpoint**

Report: whether the container served `/library`, and any build warnings.

---

### Task 11: Caption model bake-off script

**Files:**
- Create: `scripts/bakeoff.py`
- Create: `tests/test_bakeoff.py`

**Interfaces:**
- Consumes: `OpenAICompatClient`, `FakeInferenceClient`, `encode_image`, `LocalStorage`.
- Produces: `scripts.bakeoff.BakeoffRow` (dataclass: `model: str`, `photo: str`, `seconds: float`, `caption: str`), `scripts.bakeoff.caption_once(client, model: str, image_bytes: bytes, clock) -> tuple[str, float]`, and `scripts.bakeoff.run_bakeoff(client, models: list[str], images: list[tuple[str, bytes]], clock) -> list[BakeoffRow]`.

`clock` is a zero-argument callable returning a float, injected so tests are deterministic. The spec makes this script the gate that picks the default caption model per profile (spec section 4).

- [ ] **Step 1: Write the failing test**

Create `tests/test_bakeoff.py`:

```python
from scripts.bakeoff import format_table, run_bakeoff
from ivms777.inference.fakes import FakeInferenceClient


def fake_clock(values):
    ticks = iter(values)
    return lambda: next(ticks)


def test_runs_every_model_against_every_image():
    client = FakeInferenceClient(["cap-a1", "cap-a2", "cap-b1", "cap-b2"])
    clock = fake_clock([0.0, 1.0, 1.0, 3.0, 3.0, 3.5, 3.5, 4.5])

    rows = run_bakeoff(
        client,
        models=["model-a", "model-b"],
        images=[("one.jpg", b"x"), ("two.jpg", b"y")],
        clock=clock,
    )

    assert len(rows) == 4
    assert [row.model for row in rows] == ["model-a", "model-a", "model-b", "model-b"]
    assert rows[0].caption == "cap-a1"
    assert rows[0].seconds == 1.0
    assert rows[1].seconds == 2.0


def test_image_is_sent_as_a_data_uri():
    client = FakeInferenceClient(["cap"])
    run_bakeoff(client, ["m"], [("one.jpg", b"x")], clock=fake_clock([0.0, 1.0]))

    _model, messages = client.calls[0]
    parts = messages[0]["content"]
    assert any(part["type"] == "image_url" for part in parts)
    image_part = next(part for part in parts if part["type"] == "image_url")
    assert image_part["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_format_table_reports_mean_seconds_per_model():
    client = FakeInferenceClient(["a", "b"])
    rows = run_bakeoff(client, ["m"], [("1.jpg", b"x"), ("2.jpg", b"y")], clock=fake_clock([0.0, 2.0, 2.0, 6.0]))

    table = format_table(rows)
    assert "m" in table
    assert "3.00" in table  # mean of 2.0 and 4.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_bakeoff.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts'`

- [ ] **Step 3: Write the script**

Create `scripts/__init__.py` (empty) and `scripts/bakeoff.py`:

```python
import argparse
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ivms777.config import get_settings
from ivms777.inference.client import InferenceClient, OpenAICompatClient, encode_image
from ivms777.storage.local import IMAGE_EXTENSIONS, LocalStorage

PROMPT = (
    "Describe this photo in one sentence, then list its mood and setting. "
    "Be concrete and factual. Do not speculate about people's identities."
)


@dataclass(frozen=True)
class BakeoffRow:
    model: str
    photo: str
    seconds: float
    caption: str


def caption_once(
    client: InferenceClient,
    model: str,
    image_bytes: bytes,
    clock: Callable[[], float],
) -> tuple[str, float]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": encode_image(image_bytes)}},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    started = clock()
    caption = client.complete(model, messages)
    return caption, clock() - started


def run_bakeoff(
    client: InferenceClient,
    models: list[str],
    images: list[tuple[str, bytes]],
    clock: Callable[[], float] = time.monotonic,
) -> list[BakeoffRow]:
    rows: list[BakeoffRow] = []
    for model in models:
        for name, data in images:
            caption, seconds = caption_once(client, model, data, clock)
            rows.append(BakeoffRow(model=model, photo=name, seconds=seconds, caption=caption))
    return rows


def format_table(rows: list[BakeoffRow]) -> str:
    models = sorted({row.model for row in rows})
    lines = ["model                     photos   mean s/photo", "-" * 48]
    for model in models:
        subset = [row for row in rows if row.model == model]
        mean = sum(row.seconds for row in subset) / len(subset)
        lines.append(f"{model:<25} {len(subset):>6}   {mean:>12.2f}")
    lines.append("")
    for model in models:
        lines.append(f"--- {model} ---")
        for row in [r for r in rows if r.model == model][:5]:
            lines.append(f"  {row.photo}: {row.caption}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare caption models on real photos.")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    settings = get_settings()
    storage = LocalStorage(args.library, extensions=IMAGE_EXTENSIONS)
    keys = list(storage.iter_keys())
    random.Random(args.seed).shuffle(keys)
    images = [(key, storage.read(key)) for key in keys[: args.count]]
    if not images:
        raise SystemExit(f"no images found under {args.library}")

    client = OpenAICompatClient(settings.inference_base_url)
    print(format_table(run_bakeoff(client, args.models, images)))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_bakeoff.py -v`
Expected: 3 passed.

- [ ] **Step 5: Run the full suite and the linter**

Run:

```bash
uv run pytest -v
uv run ruff check .
```

Expected: all tests pass, no lint errors.

- [ ] **Step 6: Checkpoint**

Report: full suite output. The bake-off is now runnable for real against your library:

```bash
uv run python -m scripts.bakeoff --models gemma4:26b-a4b gemma4:12b --library /path/to/photos --count 50
```

Record the result — it sets `IVMS777_CAPTION_MODEL` for the `mac` profile, and the equivalent run on Jetson (`qwen3-vl:4b` vs `gemma4:e4b`) sets it there.

---

### Task 12: Duplicates page

**Files:**
- Modify: `ivms777/web/app.py` (add the `/duplicates` route and its query)
- Modify: `ivms777/web/templates/base.html` (add a nav link)
- Create: `ivms777/web/templates/duplicates.html`
- Create: `tests/test_web_duplicates.py`

**Interfaces:**
- Consumes: `AppContext`, the `photos.duplicate_of` column.
- Produces: route `GET /duplicates`, and `ivms777.web.app.duplicate_sets(context: AppContext) -> list[dict]` where each dict is `{"photo_id": int, "canonical_path": str, "paths": list[str], "wasted_bytes": int}`.

`wasted_bytes` is the disk space the redundant copies occupy — `bytes` times the
number of duplicates — so the page says something actionable rather than just
listing files.

- [ ] **Step 1: Write the failing test**

Create `tests/test_web_duplicates.py`:

```python
import pytest
from fastapi.testclient import TestClient

from ivms777.ingest.scanner import scan
from ivms777.ingest.worker import drain, thumbnail_handler
from ivms777.storage.local import IMAGE_EXTENSIONS, LocalStorage
from ivms777.web.app import create_app, duplicate_sets
from tests.fixtures import make_jpeg


@pytest.fixture
def client(settings):
    root = settings.library_root
    make_jpeg(root / "holiday.jpg", color="teal")
    make_jpeg(root / "backup" / "holiday-copy.jpg", color="teal")
    make_jpeg(root / "solo.jpg", color="orange")
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    app = create_app(settings)
    context = app.state.context
    originals = LocalStorage(root, extensions=IMAGE_EXTENSIONS)
    scan(context.conn, originals, owner_id=settings.owner_id)
    drain(context.conn, {"thumbnail": thumbnail_handler(originals, context.derived, 320, 1600)})
    with TestClient(app) as test_client:
        yield test_client


def test_duplicate_sets_groups_paths_under_the_canonical(client):
    sets = duplicate_sets(client.app.state.context)

    assert len(sets) == 1
    entry = sets[0]
    assert entry["canonical_path"] == "holiday.jpg"
    assert entry["paths"] == ["backup/holiday-copy.jpg"]
    assert entry["wasted_bytes"] > 0


def test_photos_without_duplicates_are_not_listed(client):
    sets = duplicate_sets(client.app.state.context)
    assert all("solo.jpg" not in entry["canonical_path"] for entry in sets)


def test_duplicates_page_lists_every_path(client):
    response = client.get("/duplicates")

    assert response.status_code == 200
    assert "holiday.jpg" in response.text
    assert "backup/holiday-copy.jpg" in response.text


def test_duplicates_page_says_so_when_there_are_none(client):
    client.app.state.context.conn.execute("UPDATE photos SET duplicate_of = NULL")
    response = client.get("/duplicates")
    assert "No duplicates" in response.text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_web_duplicates.py -v`
Expected: FAIL with `ImportError: cannot import name 'duplicate_sets'`

- [ ] **Step 3: Add the query and route**

In `ivms777/web/app.py`, add this module-level function next to `run_ingest`:

```python
DUPLICATE_SQL = (
    "SELECT c.id AS photo_id, c.path AS canonical_path, c.bytes AS size,"
    " group_concat(d.path, char(10)) AS dupe_paths, count(d.id) AS dupe_count"
    " FROM photos c JOIN photos d ON d.duplicate_of = c.id"
    " WHERE c.owner_id = ? GROUP BY c.id ORDER BY count(d.id) * c.bytes DESC"
)


def duplicate_sets(context: AppContext) -> list[dict]:
    rows = context.conn.execute(DUPLICATE_SQL, (context.settings.owner_id,)).fetchall()
    return [
        {
            "photo_id": row["photo_id"],
            "canonical_path": row["canonical_path"],
            "paths": row["dupe_paths"].split("\n") if row["dupe_paths"] else [],
            "wasted_bytes": (row["size"] or 0) * row["dupe_count"],
        }
        for row in rows
    ]
```

Inside `create_app`, before `return app`, add:

```python
    @app.get("/duplicates", response_class=HTMLResponse)
    def duplicates(request: Request) -> HTMLResponse:
        sets = duplicate_sets(context())
        return templates.TemplateResponse(
            request,
            "duplicates.html",
            {"sets": sets, "wasted_total": sum(entry["wasted_bytes"] for entry in sets)},
        )
```

- [ ] **Step 4: Add the template and nav link**

Create `ivms777/web/templates/duplicates.html`:

```html
{% extends "base.html" %}
{% block content %}
<h1>Duplicates</h1>
{% if sets %}
  <p>{{ sets|length }} set(s), {{ '%.1f'|format(wasted_total / 1048576) }} MB of redundant copies.</p>
  {% for entry in sets %}
    <div class="dupe-set">
      <img src="/thumb/{{ entry['photo_id'] }}" alt="{{ entry['canonical_path'] }}" loading="lazy">
      <div class="dupe-paths">
        <ul>
          <li class="dupe-canonical">{{ entry['canonical_path'] }} (kept, processed)</li>
          {% for path in entry['paths'] %}
            <li>{{ path }}</li>
          {% endfor %}
        </ul>
      </div>
    </div>
  {% endfor %}
{% else %}
  <p>No duplicates found.</p>
{% endif %}
{% endblock %}
```

In `ivms777/web/templates/base.html`, replace the nav block with:

```html
    <nav class="nav">
      <a href="/library">Library</a>
      <a href="/duplicates">Duplicates</a>
      <a href="/index">Index</a>
    </nav>
```

The page is read-only — it reports duplicates, it does not delete files. Deleting
originals is destructive and out of scope for v1.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_web_duplicates.py -v`
Expected: 4 passed.

Then the full suite: `uv run pytest -v` and `uv run ruff check .`
Expected: all pass, no lint errors.

- [ ] **Step 6: Checkpoint**

Report: full suite output and the reported wasted-bytes figure from the test run.

---

### Task 13: EXIF facet filtering and sorting

**Files:**
- Create: `ivms777/search/__init__.py`
- Create: `ivms777/search/facets.py`
- Modify: `ivms777/web/app.py` (library routes accept filters and sort)
- Modify: `ivms777/web/templates/library.html` (add the sidebar)
- Create: `ivms777/web/templates/_facet_sidebar.html`
- Create: `tests/test_search_facets.py`
- Create: `tests/test_web_facet_filters.py`

**Interfaces:**
- Consumes: `photo_facets`, `AppContext`, `FACET_KEYS`.
- Produces:
  - `ivms777.search.facets.FacetFilter` (dataclass: `key: str`, `values: list[str] | None = None`, `gte: float | None = None`, `lte: float | None = None`)
  - `ivms777.search.facets.parse_filters(params: dict[str, str]) -> list[FacetFilter]` — reads `f_<key>=a,b` for categorical and `n_<key>=min:max` for numeric
  - `ivms777.search.facets.build_where(filters: list[FacetFilter]) -> tuple[str, list]` returning an SQL fragment of `EXISTS` clauses plus bound parameters
  - `ivms777.search.facets.facet_counts(conn, owner_id: int, key: str, limit: int = 12) -> list[tuple[str, int]]`
  - `ivms777.search.facets.SORTABLE: dict[str, str]` mapping a sort name to its facet key

Categorical filters OR within a key and AND across keys, matching the tag facet
behaviour in the spec. Numeric filters are inclusive ranges.

- [ ] **Step 1: Write the failing test**

Create `tests/test_search_facets.py`:

```python
import pytest

from ivms777.search.facets import (
    FacetFilter,
    build_where,
    facet_counts,
    parse_filters,
)


def add_photo(conn, photo_id, facets):
    conn.execute(
        "INSERT INTO photos(id, owner_id, path, content_hash, created_at, updated_at)"
        " VALUES (?, 1, ?, ?, '2026-01-01', '2026-01-01')",
        (photo_id, f"{photo_id}.jpg", f"h{photo_id}"),
    )
    for key, text, num in facets:
        conn.execute(
            "INSERT INTO photo_facets(photo_id, key, value_text, value_num) VALUES (?,?,?,?)",
            (photo_id, key, text, num),
        )


@pytest.fixture
def library(conn):
    add_photo(conn, 1, [("camera_model", "X-T5", None), ("iso", None, 200.0)])
    add_photo(conn, 2, [("camera_model", "X-T5", None), ("iso", None, 6400.0)])
    add_photo(conn, 3, [("camera_model", "iPhone", None), ("iso", None, 400.0)])
    return conn


def matching_ids(conn, filters):
    where, params = build_where(filters)
    sql = f"SELECT id FROM photos p WHERE 1=1 {where} ORDER BY id"
    return [row["id"] for row in conn.execute(sql, params)]


def test_no_filters_matches_everything(library):
    assert matching_ids(library, []) == [1, 2, 3]


def test_categorical_filter_matches_one_value(library):
    assert matching_ids(library, [FacetFilter("camera_model", values=["iPhone"])]) == [3]


def test_categorical_values_are_ored(library):
    filters = [FacetFilter("camera_model", values=["iPhone", "X-T5"])]
    assert matching_ids(library, filters) == [1, 2, 3]


def test_numeric_range_is_inclusive(library):
    assert matching_ids(library, [FacetFilter("iso", gte=200, lte=400)]) == [1, 3]


def test_open_ended_range_works(library):
    assert matching_ids(library, [FacetFilter("iso", gte=1000)]) == [2]


def test_filters_across_keys_are_anded(library):
    filters = [FacetFilter("camera_model", values=["X-T5"]), FacetFilter("iso", lte=1000)]
    assert matching_ids(library, filters) == [1]


def test_parse_filters_reads_categorical_and_numeric_params():
    filters = parse_filters({"f_camera_model": "X-T5,iPhone", "n_iso": "200:800"})
    by_key = {f.key: f for f in filters}

    assert by_key["camera_model"].values == ["X-T5", "iPhone"]
    assert (by_key["iso"].gte, by_key["iso"].lte) == (200.0, 800.0)


def test_parse_filters_accepts_open_ended_ranges():
    by_key = {f.key: f for f in parse_filters({"n_iso": "1000:"})}
    assert (by_key["iso"].gte, by_key["iso"].lte) == (1000.0, None)


def test_parse_filters_ignores_unknown_and_malformed_params():
    assert parse_filters({"f_not_a_facet": "x", "n_iso": "junk", "page": "2"}) == []


def test_facet_counts_are_ordered_by_frequency(library):
    counts = facet_counts(library, owner_id=1, key="camera_model")
    assert counts == [("X-T5", 2), ("iPhone", 1)]
```

Create `tests/test_web_facet_filters.py`:

```python
import pytest
from fastapi.testclient import TestClient

from ivms777.ingest.scanner import scan
from ivms777.ingest.worker import drain, thumbnail_handler
from ivms777.storage.local import IMAGE_EXTENSIONS, LocalStorage
from ivms777.web.app import create_app
from tests.fixtures import make_jpeg_with_exif


@pytest.fixture
def client(settings):
    root = settings.library_root
    make_jpeg_with_exif(root / "morning.jpg", when="2025:07:12 09:00:00", model="X-T5")
    make_jpeg_with_exif(root / "evening.jpg", when="2024:02:03 20:00:00", model="iPhone")
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    app = create_app(settings)
    context = app.state.context
    originals = LocalStorage(root, extensions=IMAGE_EXTENSIONS)
    scan(context.conn, originals, owner_id=settings.owner_id)
    drain(context.conn, {"thumbnail": thumbnail_handler(originals, context.derived, 320, 1600)})
    with TestClient(app) as test_client:
        yield test_client


def test_unfiltered_library_shows_both(client):
    assert client.get("/library").text.count('class="tile"') == 2


def test_categorical_facet_filter_narrows_the_grid(client):
    response = client.get("/library?f_camera_model=iPhone")
    assert response.text.count('class="tile"') == 1
    assert "evening.jpg" in response.text


def test_time_of_day_filter_works(client):
    assert client.get("/library?f_time_of_day=morning").text.count('class="tile"') == 1


def test_numeric_year_range_filter_works(client):
    assert client.get("/library?n_year=2025:").text.count('class="tile"') == 1


def test_sidebar_lists_facet_values_with_counts(client):
    body = client.get("/library").text
    assert "X-T5" in body
    assert "iPhone" in body
    assert "Camera" in body


def test_sort_by_facet_changes_order(client):
    ascending = client.get("/library?sort=year_asc").text
    descending = client.get("/library?sort=year_desc").text
    assert ascending.index("evening.jpg") < ascending.index("morning.jpg")
    assert descending.index("morning.jpg") < descending.index("evening.jpg")


def test_filters_survive_into_the_infinite_scroll_link(client):
    body = client.get("/library?f_camera_model=X-T5").text
    assert "f_camera_model=X-T5" in body or 'class="tile"' in body
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_search_facets.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ivms777.search'`

- [ ] **Step 3: Write the facet query module**

Create empty `ivms777/search/__init__.py`.

Create `ivms777/search/facets.py`:

```python
import sqlite3
from dataclasses import dataclass

from ivms777.ingest.facets import FACET_KEYS

SORTABLE: dict[str, str] = {
    "year_desc": "year", "year_asc": "year",
    "iso_desc": "iso", "iso_asc": "iso",
    "aperture_desc": "aperture", "aperture_asc": "aperture",
    "focal_length_desc": "focal_length", "focal_length_asc": "focal_length",
    "megapixels_desc": "megapixels", "megapixels_asc": "megapixels",
}

SIDEBAR_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Camera", ("camera_make", "camera_model", "lens")),
    ("Time", ("time_of_day", "weekday", "is_weekend")),
    ("Image", ("aspect", "has_gps", "flash")),
)


@dataclass(frozen=True)
class FacetFilter:
    key: str
    values: list[str] | None = None
    gte: float | None = None
    lte: float | None = None


def _number(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def parse_filters(params: dict[str, str]) -> list[FacetFilter]:
    filters: list[FacetFilter] = []
    for name, raw in params.items():
        if name.startswith("f_"):
            key = name[2:]
            values = [part for part in raw.split(",") if part]
            if key in FACET_KEYS and values:
                filters.append(FacetFilter(key, values=values))
        elif name.startswith("n_"):
            key = name[2:]
            if key not in FACET_KEYS or ":" not in raw:
                continue
            low, _, high = raw.partition(":")
            gte, lte = _number(low) if low else None, _number(high) if high else None
            if gte is not None or lte is not None:
                filters.append(FacetFilter(key, gte=gte, lte=lte))
    return filters


def build_where(filters: list[FacetFilter]) -> tuple[str, list]:
    fragments: list[str] = []
    params: list = []
    for item in filters:
        conditions = ["f.photo_id = p.id", "f.key = ?"]
        params.append(item.key)
        if item.values:
            placeholders = ", ".join("?" for _ in item.values)
            conditions.append(f"f.value_text IN ({placeholders})")
            params.extend(item.values)
        if item.gte is not None:
            conditions.append("f.value_num >= ?")
            params.append(item.gte)
        if item.lte is not None:
            conditions.append("f.value_num <= ?")
            params.append(item.lte)
        fragments.append(
            " AND EXISTS (SELECT 1 FROM photo_facets f WHERE " + " AND ".join(conditions) + ")"
        )
    return "".join(fragments), params


def facet_counts(
    conn: sqlite3.Connection, owner_id: int, key: str, limit: int = 12
) -> list[tuple[str, int]]:
    rows = conn.execute(
        "SELECT f.value_text AS value, count(*) AS n FROM photo_facets f"
        " JOIN photos p ON p.id = f.photo_id"
        " WHERE f.key = ? AND p.owner_id = ? AND p.duplicate_of IS NULL"
        " AND p.missing_since IS NULL AND f.value_text IS NOT NULL"
        " GROUP BY f.value_text ORDER BY n DESC, f.value_text LIMIT ?",
        (key, owner_id, limit),
    ).fetchall()
    return [(row["value"], row["n"]) for row in rows]
```

- [ ] **Step 4: Wire filters and sorting into the library routes**

In `ivms777/web/app.py`, add the imports:

```python
from urllib.parse import urlencode

from ivms777.search.facets import (
    SIDEBAR_GROUPS,
    SORTABLE,
    build_where,
    facet_counts,
    parse_filters,
)
```

Replace the `LIST_SQL` constant and the `fetch_page` helper with:

```python
BASE_SQL = (
    "SELECT p.id, p.path, p.caption, p.shot_at,"
    " (SELECT count(*) FROM photos d WHERE d.duplicate_of = p.id) AS dupe_count"
    " FROM photos p"
    " WHERE p.owner_id = ? AND p.missing_since IS NULL AND p.thumb_key IS NOT NULL"
    " AND p.duplicate_of IS NULL"
)

DEFAULT_ORDER = " ORDER BY COALESCE(p.shot_at, p.created_at) DESC, p.id DESC"


def _order_clause(sort: str | None) -> tuple[str, list]:
    facet_key = SORTABLE.get(sort or "")
    if facet_key is None:
        return DEFAULT_ORDER, []
    direction = "ASC" if (sort or "").endswith("_asc") else "DESC"
    clause = (
        " ORDER BY (SELECT f.value_num FROM photo_facets f"
        " WHERE f.photo_id = p.id AND f.key = ?) "
        f"{direction}, p.id DESC"
    )
    return clause, [facet_key]
```

Then, inside `create_app`, replace `fetch_page` with:

```python
    def fetch_page(offset: int, params: dict[str, str]) -> list:
        ctx = context()
        where, where_params = build_where(parse_filters(params))
        order, order_params = _order_clause(params.get("sort"))
        sql = BASE_SQL + where + order + " LIMIT ? OFFSET ?"
        bound = [ctx.settings.owner_id, *where_params, *order_params,
                 ctx.settings.page_size, offset]
        return list(ctx.conn.execute(sql, bound))
```

`BASE_SQL` binds `owner_id` first, `build_where` fragments bind next, then the
sort key, then limit and offset — the list above must stay in that order.

Replace the `library` and `library_page` routes with:

```python
    def _query_string(params: dict[str, str]) -> str:
        keep = {k: v for k, v in params.items() if k.startswith(("f_", "n_")) or k == "sort"}
        return urlencode(keep)

    def _sidebar() -> list[dict]:
        ctx = context()
        groups = []
        for title, keys in SIDEBAR_GROUPS:
            entries = [
                {"key": key, "values": facet_counts(ctx.conn, ctx.settings.owner_id, key)}
                for key in keys
            ]
            groups.append({"title": title, "entries": [e for e in entries if e["values"]]})
        return [group for group in groups if group["entries"]]

    @app.get("/library", response_class=HTMLResponse)
    def library(request: Request) -> HTMLResponse:
        params = dict(request.query_params)
        rows = fetch_page(0, params)
        return templates.TemplateResponse(
            request,
            "library.html",
            {
                "photos": rows,
                "next_offset": len(rows),
                "page_size": context().settings.page_size,
                "query": _query_string(params),
                "sidebar": _sidebar(),
                "active": params,
            },
        )

    @app.get("/library/page", response_class=HTMLResponse)
    def library_page(request: Request, offset: int = 0) -> HTMLResponse:
        params = dict(request.query_params)
        rows = fetch_page(offset, params)
        return templates.TemplateResponse(
            request,
            "_grid_page.html",
            {
                "photos": rows,
                "next_offset": offset + len(rows),
                "page_size": context().settings.page_size,
                "query": _query_string(params),
            },
        )
```

In `ivms777/web/templates/_grid_page.html`, change the infinite-scroll trigger
so filters survive paging:

```html
  <div hx-get="/library/page?offset={{ next_offset }}{% if query %}&{{ query }}{% endif %}"
       hx-trigger="revealed"
       hx-swap="outerHTML"></div>
```

- [ ] **Step 5: Add the sidebar template**

Create `ivms777/web/templates/_facet_sidebar.html`:

```html
<aside class="sidebar">
  <form method="get" action="/library">
    <label class="sort-row">
      Sort
      <select name="sort" onchange="this.form.submit()">
        <option value="">Date taken</option>
        <option value="year_asc" {% if active.get('sort') == 'year_asc' %}selected{% endif %}>Year, oldest first</option>
        <option value="year_desc" {% if active.get('sort') == 'year_desc' %}selected{% endif %}>Year, newest first</option>
        <option value="iso_desc" {% if active.get('sort') == 'iso_desc' %}selected{% endif %}>ISO, highest first</option>
        <option value="aperture_asc" {% if active.get('sort') == 'aperture_asc' %}selected{% endif %}>Aperture, widest first</option>
        <option value="focal_length_desc" {% if active.get('sort') == 'focal_length_desc' %}selected{% endif %}>Focal length, longest first</option>
        <option value="megapixels_desc" {% if active.get('sort') == 'megapixels_desc' %}selected{% endif %}>Megapixels, largest first</option>
      </select>
    </label>

    {% for group in sidebar %}
      <h3>{{ group['title'] }}</h3>
      {% for entry in group['entries'] %}
        <div class="facet-key">{{ entry['key'].replace('_', ' ') }}</div>
        {% for value, count in entry['values'] %}
          <label class="facet-row">
            <input type="checkbox" name="f_{{ entry['key'] }}" value="{{ value }}"
                   {% if value in (active.get('f_' ~ entry['key'], '')).split(',') %}checked{% endif %}>
            <span>{{ value }}</span><span class="facet-count">{{ count }}</span>
          </label>
        {% endfor %}
      {% endfor %}
    {% endfor %}

    <h3>Ranges</h3>
    <label class="facet-row">ISO <input type="text" name="n_iso" placeholder="200:6400" value="{{ active.get('n_iso', '') }}"></label>
    <label class="facet-row">Year <input type="text" name="n_year" placeholder="2020:2025" value="{{ active.get('n_year', '') }}"></label>

    <button type="submit">Apply</button>
    <a href="/library">Clear</a>
  </form>
</aside>
```

Checkboxes sharing one `name` submit as repeated params; FastAPI's
`dict(request.query_params)` keeps the last value, which would silently drop
multi-select. Make the parse read every value — in `library` and `library_page`,
build `params` as:

```python
        params = {
            key: ",".join(request.query_params.getlist(key))
            for key in request.query_params.keys()
        }
```

Use that in place of `dict(request.query_params)` in both routes.

Replace `ivms777/web/templates/library.html` with:

```html
{% extends "base.html" %}
{% block content %}
<div class="library-layout">
  {% include "_facet_sidebar.html" %}
  <div class="grid">
    {% include "_grid_page.html" %}
  </div>
</div>
{% endblock %}
```

Append to `ivms777/web/static/app.css`:

```css
.library-layout { display: grid; grid-template-columns: 240px 1fr; gap: 16px; align-items: start; }
.sidebar { font-size: 13px; background: #1a1a1a; padding: 12px; border-radius: 8px; }
.sidebar h3 { margin: 14px 0 6px; font-size: 12px; text-transform: uppercase; color: #999; }
.facet-key { margin-top: 8px; color: #bbb; font-size: 11px; text-transform: uppercase; }
.facet-row { display: flex; align-items: center; gap: 6px; padding: 2px 0; }
.facet-row span:first-of-type { flex: 1; }
.facet-count { color: #888; }
.sort-row select { width: 100%; margin-top: 4px; }
.sidebar input[type="text"] { width: 90px; }
.sidebar button { margin-top: 12px; width: 100%; }
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_search_facets.py tests/test_web_facet_filters.py -v`
Expected: 17 passed.

Then the full suite: `uv run pytest -v` and `uv run ruff check .`
Expected: all pass, no lint errors.

`tests/test_web_library.py` asserts an exact tile count on `/library`; the
sidebar adds no `class="tile"` occurrences, so those assertions still hold.

- [ ] **Step 7: Checkpoint**

Report: full suite output, plus a note on which facet keys actually appeared in
the sidebar when run against real photos — cameras and phones vary a lot in what
they write, and that shapes which facets are worth surfacing by default.

---

## What plan 01 delivers

Point the app at a folder, press "Start scan", and watch a progress screen fill in while a thumbnail grid populates in date order. Moves are detected by content hash, deleted files are marked missing rather than dropped, and one unreadable file never stalls the queue.

Exact duplicates — the same image bytes under any name, in any subfolder — are detected by sha256, collapsed to one canonical photo in the grid with an `×N` badge, and listed with all their paths and wasted disk space on `/duplicates`. Only the canonical copy is ever queued for work, so the expensive stages in later plans run once per unique image no matter how many copies exist.

EXIF is fully captured and turned into 25 queryable facets — camera, lens, ISO, aperture, shutter, focal length, flash, metering, year, weekday, time of day, GPS presence, megapixels, aspect. The sidebar filters by any of them with live counts, ranges work on the numeric ones, and any numeric facet is a sort key. All of it is exact, with no model involved, which is why it works before a single embedding exists.

The job table, storage interface, inference client, and compose profiles are all in place for the model stages.

**Not yet working:** `/photo/{id}` is linked from tiles but returns 404 — it arrives in plan 02 with embeddings and search. No captions, no tags, no similarity, no groups, no chat.

## Following plans

| Plan | Spec phase | Delivers |
|---|---|---|
| 02 | 2 | SigLIP embeddings, taxonomy scoring, semantic + tag-facet + keyword + fusion search layered onto the EXIF facet filters from Task 13, similar photos, `/photo` detail with the full EXIF panel |
| 03 | 3 | Caption stage against the inference service, captions in the UI |
| 04 | 4 | Query planner, parsed-filter chips, vocabulary mining |
| 05 | 5 | Event, cluster, and duplicate groups, `/groups` |
| 06 | 6 | Ask-your-library chat with streaming and citations |
