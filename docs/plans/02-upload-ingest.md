# Photo Library Organizer — Plan 02: Upload Ingest

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace folder-scan ingest with browser upload. Photos arrive over HTTP, are identified by content hash, stored content-addressed, and every local path they came from is recorded — so the library works with no host mount and stage 2 has the data it needs.

**Architecture:** The browser hashes files in a Web Worker, probes the server for which hashes are new, and sends only those. `app` verifies each hash, stores the original under a content-addressed key, reads EXIF, and inserts one `photos` row per distinct image plus one `photo_sources` row per local path. Duplicates need no detection pass: a second path on a known hash is just another source row. Everything after receipt stays on the existing job queue.

**Tech Stack:** Python 3.12, uv, FastAPI, Jinja2, HTMX, vanilla JS Web Worker + WebCrypto, SQLite + sqlite-vec, Pillow + pillow-heif, pytest.

**Spec:** `docs/design.md` — sections 3.2b, 5, 6, 6.1, 8, 13.

**Supersedes:** `docs/plans/01-foundation.md` tasks 3, 4, 7, 8 and 12. Plan 01 built a host-mounted folder scanner and a folder-picker UI; both are deleted here. Everything else plan 01 built — EXIF capture, facets, thumbnails, the job queue, the library grid, facet filters and sorting — survives unchanged.

**Covers:** Spec phase 1, restated for upload. Stage 2 (`ivms777-sync`, layouts, the manifest endpoint) is spec phase 7 and gets plan 07.

## Global Constraints

- Python 3.12. Dependencies managed by `uv` with a committed `uv.lock`.
- Package name is `ivms777`. Source lives in `ivms777/`, tests in `tests/`.
- **Never run `git commit`, `git add`, or any staging command.** The user commits. Every task ends with a checkpoint where you report what changed and stop.
- Every user-scoped query filters on `owner_id`. `owner_id` is the constant `settings.owner_id` — there is no auth, no login, no user table.
- All I/O paths come from `Settings`. No hardcoded paths outside `ivms777/config.py`.
- Deploy profiles are `mac`, `jetson`, `cloud`. Profile changes config only, never code. Ingest is identical on all three.
- Tests must not download model weights or make network calls. Use the fakes.
- SQLite connections open with `journal_mode=WAL` and `busy_timeout=5000`.
- The full suite must pass at the end of every task: `uv run pytest -q`.
- **This plan changes the schema incompatibly and there is no data migration.** The database is developer data only. Task 2 adds a version guard that tells you to delete the file.

---

### Task 1: Remove folder-scan ingest

Delete the scanner, the folder browser, and the `/index` screen. Nothing replaces them yet — this task only removes, so the suite stays green and the next task starts from a clean surface.

**Files:**
- Delete: `ivms777/ingest/scanner.py`
- Delete: `ivms777/browse.py`
- Delete: `tests/test_scanner.py`
- Delete: `tests/test_browse.py`
- Delete: `tests/test_web_index.py`
- Delete: `ivms777/web/templates/_browser.html`
- Delete: `ivms777/web/templates/_index_body.html`
- Delete: `ivms777/web/templates/index.html`
- Modify: `ivms777/config.py`
- Modify: `ivms777/web/deps.py`
- Modify: `ivms777/web/app.py`
- Modify: `ivms777/ingest/cli.py`
- Modify: `ivms777/web/templates/base.html`
- Modify: `tests/conftest.py`
- Modify: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `AppContext` without `host_root`, `library_subpath`, `library_root`, or `originals()`. `Settings` without `host_root` or `library_root`. `ivms777.web.app.create_app(settings) -> FastAPI` serving `/library`, `/library/page`, `/thumb/{id}`, `/duplicates` only.

- [ ] **Step 1: Delete the scan-side modules and their tests**

```bash
rm ivms777/ingest/scanner.py ivms777/browse.py
rm tests/test_scanner.py tests/test_browse.py tests/test_web_index.py
rm ivms777/web/templates/_browser.html
rm ivms777/web/templates/_index_body.html
rm ivms777/web/templates/index.html
```

- [ ] **Step 2: Run the suite to see exactly what broke**

Run: `uv run pytest -q`
Expected: collection errors in `tests/test_config.py` and any web test, because `ivms777/web/app.py` still imports `ivms777.browse` and `ivms777.ingest.scanner`.

- [ ] **Step 3: Drop the host-mount settings**

In `ivms777/config.py`, remove the `host_root` and `library_root` fields and the comment above them. The `Settings` class keeps everything else. After the edit the field block reads:

```python
    profile: Profile = "mac"
    data_dir: Path = Path("/data")

    caption_model: str | None = None
    planner_model: str | None = None
    embed_device: Literal["cpu", "cuda", "mps"] | None = None
    inference_base_url: str | None = None

    owner_id: int = 1
    thumb_grid_px: int = 320
    thumb_detail_px: int = 1600
    page_size: int = Field(default=100, ge=1, le=500)
```

- [ ] **Step 4: Reduce `AppContext` to what remains**

Replace the whole of `ivms777/web/deps.py` with:

```python
import sqlite3
from dataclasses import dataclass

from ivms777.config import Settings
from ivms777.storage.local import LocalStorage


@dataclass
class AppContext:
    settings: Settings
    conn: sqlite3.Connection
    derived: LocalStorage


def build_context(settings: Settings) -> AppContext:
    from ivms777.db.connection import connect, migrate

    conn = connect(settings.db_path)
    migrate(conn)
    return AppContext(
        settings=settings,
        conn=conn,
        derived=LocalStorage(settings.thumb_dir),
    )
```

- [ ] **Step 5: Strip the `/index` routes out of the app**

In `ivms777/web/app.py`:

- Delete the `from ivms777.browse import (...)` block, the `from ivms777.ingest.scanner import scan` line, and the `from fastapi import BackgroundTasks, ... Form` names that are now unused. The imports become:

```python
from pathlib import Path
from urllib.parse import urlencode

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ivms777.config import Settings
from ivms777.ingest.jobs import STAGES, stage_counts
from ivms777.ingest.worker import drain, thumbnail_handler
from ivms777.search.facets import (
    SIDEBAR_GROUPS,
    SORTABLE,
    build_where,
    facet_counts,
    parse_filters,
)
from ivms777.web.deps import AppContext, build_context
```

- Delete the `run_ingest` function.
- Delete the `browse_payload`, `index_page`, `browse`, `select_folder`, and `start_scan` route functions.
- In `progress_payload`, drop the `library_subpath` key. It becomes:

```python
    def progress_payload() -> dict:
        ctx = context()
        failures = list(
            ctx.conn.execute(
                "SELECT p.path, j.stage, j.error FROM jobs j JOIN photos p ON p.id = j.photo_id"
                " WHERE j.status = 'failed' ORDER BY p.path LIMIT 50"
            )
        )
        return {"stages": [(stage, stage_counts(ctx.conn, stage)) for stage in STAGES],
                "failures": failures}
```

- Keep the `/index/progress` route but move it to `/upload/progress`, so the template partial keeps working and task 5 has the endpoint it needs:

```python
    @app.get("/upload/progress", response_class=HTMLResponse)
    def progress(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(request, "_progress.html", progress_payload())
```

- `drain` and `thumbnail_handler` are imported but unused until task 3 re-wires them. Leave the import in place and add `# noqa: F401` if the linter objects.

- [ ] **Step 6: Point the nav and the progress poller at the upload screen**

In `ivms777/web/templates/base.html`, change the `/index` nav link to `/upload` and its label to `Upload`.

In `ivms777/web/templates/_progress.html`, the wrapper on line 1 polls the old URL. Give it an id and the new endpoint, and add a `refresh` trigger so `upload.js` can force a redraw when a transfer finishes:

```html
<div class="progress" id="progress" hx-get="/upload/progress"
     hx-trigger="every 2s, refresh" hx-swap="outerHTML">
```

The rest of the template is unchanged — it already renders `failure['path']`, `failure['stage']`, and `failure['error']`, which is exactly what `progress_payload` still supplies.

- [ ] **Step 7: Stub the worker CLI**

`ivms777/ingest/cli.py` calls the deleted `run_ingest`. It has no ingest to drive until task 3, and no storage of originals until task 2, so reduce it to the drain loop it will keep — with the handler wiring left for task 2 step 8, which is where `context.originals` appears:

```python
import time

from ivms777.config import get_settings
from ivms777.ingest.worker import drain
from ivms777.web.deps import build_context

POLL_SECONDS = 10


def main() -> None:
    context = build_context(get_settings())
    while True:
        drain(context.conn, {})
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
```

- [ ] **Step 8: Update the fixtures and config test**

In `tests/conftest.py`, the `settings` fixture no longer builds a host tree:

```python
from pathlib import Path

import pytest

from ivms777.config import Settings
from ivms777.db.connection import connect, migrate


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path)


@pytest.fixture
def conn(settings: Settings):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    c = connect(settings.db_path)
    migrate(c)
    yield c
    c.close()
```

In `tests/test_config.py`, delete every assertion that mentions `host_root` or `library_root`.

- [ ] **Step 9: Run the suite**

Run: `uv run pytest -q`
Expected: PASS. The scanner, browser, and index tests are gone; everything else is untouched.

- [ ] **Step 10: Checkpoint**

Report: modules deleted, routes removed, test count before and after. Stop for review. Do not commit.

---

### Task 2: Schema v2 — content-addressed photos, sources, uploads

**Files:**
- Modify: `ivms777/db/schema.sql`
- Modify: `ivms777/db/connection.py`
- Create: `ivms777/storage/keys.py`
- Create: `tests/factories.py`
- Modify: `ivms777/config.py`
- Modify: `ivms777/web/deps.py`
- Modify: `ivms777/storage/base.py`
- Modify: `ivms777/storage/local.py`
- Modify: `ivms777/ingest/worker.py`
- Modify: `ivms777/web/app.py`
- Modify: `ivms777/search/facets.py`
- Modify: `tests/test_db.py`
- Modify: `tests/test_jobs.py`
- Modify: `tests/test_worker.py`
- Modify: `tests/test_facets.py`
- Modify: `tests/test_search_facets.py`
- Modify: `tests/test_web_library.py`
- Modify: `tests/test_web_duplicates.py`
- Modify: `tests/test_web_facet_filters.py`
- Modify: `tests/test_storage.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `ivms777.storage.keys.content_key(content_hash: str, suffix: str) -> str`
  - `ivms777.db.connection.SCHEMA_VERSION: int` and `ivms777.db.connection.SchemaTooOldError`
  - `Settings.originals_dir -> Path`
  - `AppContext.originals: LocalStorage`
  - `LocalStorage.free_bytes() -> int`
  - `tests.factories.add_photo(conn, *, photo_id=None, owner_id=1, content_hash, suffix=".jpg", sources=(), **columns) -> int`

- [ ] **Step 1: Write the failing schema test**

Replace the table-shape tests in `tests/test_db.py`. Delete `test_owner_path_uniqueness_is_enforced` and `test_duplicate_of_points_at_another_photo` entirely, and replace `test_expected_tables_exist` with:

```python
def test_expected_tables_exist(conn):
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view')").fetchall()
    names = {r[0] for r in rows}
    for expected in [
        "photos", "photo_sources", "uploads", "tags", "photo_tags",
        "photo_facets", "jobs", "groups", "group_photos",
    ]:
        assert expected in names
    assert "scans" not in names
    assert "app_settings" not in names


def test_photos_are_unique_per_owner_and_hash(conn):
    args = ("aa" * 32, "aa/aa/" + "aa" * 32 + ".jpg", "2026-01-01T00:00:00", "2026-01-01T00:00:00")
    conn.execute(
        "INSERT INTO photos(owner_id, content_hash, storage_key, created_at, updated_at)"
        " VALUES (1, ?, ?, ?, ?)",
        args,
    )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO photos(owner_id, content_hash, storage_key, created_at, updated_at)"
            " VALUES (1, ?, ?, ?, ?)",
            args,
        )


def test_a_photo_can_have_many_sources_but_each_path_once(conn):
    photo_id = add_photo(conn, content_hash="bb" * 32)
    upload_id = conn.execute(
        "INSERT INTO uploads(owner_id, root_label, started_at) VALUES (1, 'Pictures', ?)",
        ("2026-01-01T00:00:00",),
    ).lastrowid
    for rel_path in ("a/one.jpg", "b/two.jpg"):
        conn.execute(
            "INSERT INTO photo_sources(photo_id, upload_id, rel_path, filename)"
            " VALUES (?, ?, ?, ?)",
            (photo_id, upload_id, rel_path, rel_path.split("/")[-1]),
        )
    assert conn.execute(
        "SELECT count(*) FROM photo_sources WHERE photo_id = ?", (photo_id,)
    ).fetchone()[0] == 2
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO photo_sources(photo_id, upload_id, rel_path, filename)"
            " VALUES (?, ?, 'a/one.jpg', 'one.jpg')",
            (photo_id, upload_id),
        )


def test_sources_vanish_with_their_photo(conn):
    photo_id = add_photo(conn, content_hash="cc" * 32, sources=("x/one.jpg",))
    conn.execute("DELETE FROM photos WHERE id = ?", (photo_id,))
    assert conn.execute("SELECT count(*) FROM photo_sources").fetchone()[0] == 0


def test_an_old_database_is_refused_rather_than_silently_wrong(settings):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    stale = connect(settings.db_path)
    stale.execute("CREATE TABLE photos (id INTEGER PRIMARY KEY, path TEXT)")
    with pytest.raises(SchemaTooOldError):
        migrate(stale)
    stale.close()
```

Add the imports `from ivms777.db.connection import SchemaTooOldError, connect, migrate` and `from tests.factories import add_photo` at the top of the file.

- [ ] **Step 2: Write the test factory**

Create `tests/factories.py`. Every test that needs a photo row goes through it, so the next schema change touches one file:

```python
"""Row builders for tests. Keep every INSERT INTO photos in here."""

import sqlite3
from collections.abc import Sequence

from ivms777.storage.keys import content_key

NOW = "2026-01-01T00:00:00"


def add_upload(
    conn: sqlite3.Connection, *, owner_id: int = 1, root_label: str = "Pictures"
) -> int:
    cursor = conn.execute(
        "INSERT INTO uploads(owner_id, root_label, started_at) VALUES (?, ?, ?)",
        (owner_id, root_label, NOW),
    )
    return int(cursor.lastrowid)


def add_photo(
    conn: sqlite3.Connection,
    *,
    photo_id: int | None = None,
    owner_id: int = 1,
    content_hash: str,
    suffix: str = ".jpg",
    sources: Sequence[str] = (),
    upload_id: int | None = None,
    **columns: object,
) -> int:
    """Insert one photo and its source paths. `columns` sets any other photo column."""
    fields = {
        "owner_id": owner_id,
        "content_hash": content_hash,
        "storage_key": content_key(content_hash, suffix),
        "created_at": NOW,
        "updated_at": NOW,
        **columns,
    }
    if photo_id is not None:
        fields["id"] = photo_id
    names = ", ".join(fields)
    placeholders = ", ".join("?" for _ in fields)
    cursor = conn.execute(
        f"INSERT INTO photos({names}) VALUES ({placeholders})", tuple(fields.values())
    )
    new_id = int(cursor.lastrowid)
    if sources:
        if upload_id is None:
            upload_id = add_upload(conn, owner_id=owner_id)
        for rel_path in sources:
            conn.execute(
                "INSERT INTO photo_sources(photo_id, upload_id, rel_path, filename)"
                " VALUES (?, ?, ?, ?)",
                (new_id, upload_id, rel_path, rel_path.rsplit("/", 1)[-1]),
            )
    return new_id
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `uv run pytest tests/test_db.py -q`
Expected: FAIL — `no such module ivms777.storage.keys`, then `no such table: photo_sources`.

- [ ] **Step 4: Write the content-addressed key helper**

Create `ivms777/storage/keys.py`:

```python
from pathlib import PurePosixPath

# Two levels of fan-out keep any one directory under a few thousand entries at
# 100k+ photos, which matters on filesystems that degrade with wide directories.
FANOUT = 2


def content_key(content_hash: str, suffix: str) -> str:
    """Storage key for an original, derived only from its bytes.

    The suffix is cosmetic — it makes the store browsable and lets a webserver
    guess a content type. Identity is the hash alone.
    """
    clean = suffix.lower()
    if not clean.startswith("."):
        clean = f".{clean}" if clean else ""
    return f"{content_hash[:FANOUT]}/{content_hash[FANOUT:FANOUT * 2]}/{content_hash}{clean}"


def suffix_of(rel_path: str) -> str:
    return PurePosixPath(rel_path).suffix.lower()
```

- [ ] **Step 5: Write the new schema**

Replace `ivms777/db/schema.sql` in full:

```sql
CREATE TABLE IF NOT EXISTS uploads (
  id            INTEGER PRIMARY KEY,
  owner_id      INTEGER NOT NULL,
  root_label    TEXT NOT NULL,
  started_at    TEXT NOT NULL,
  finished_at   TEXT,
  files_offered INTEGER NOT NULL DEFAULT 0,
  files_sent    INTEGER NOT NULL DEFAULT 0,
  files_failed  INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS photos (
  id              INTEGER PRIMARY KEY,
  owner_id        INTEGER NOT NULL,
  content_hash    TEXT NOT NULL,
  storage_key     TEXT NOT NULL,
  phash           TEXT,
  bytes           INTEGER,
  width           INTEGER,
  height          INTEGER,
  shot_at         TEXT,
  camera          TEXT,
  lens            TEXT,
  gps_lat         REAL,
  gps_lon         REAL,
  thumb_key       TEXT,
  caption         TEXT,
  caption_model   TEXT,
  embedding_model TEXT,
  exif_json       TEXT,
  created_at      TEXT NOT NULL,
  updated_at      TEXT NOT NULL,
  UNIQUE(owner_id, content_hash)
);
CREATE INDEX IF NOT EXISTS photos_owner_shot ON photos(owner_id, shot_at);

CREATE TABLE IF NOT EXISTS photo_sources (
  id        INTEGER PRIMARY KEY,
  photo_id  INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
  upload_id INTEGER NOT NULL REFERENCES uploads(id) ON DELETE CASCADE,
  rel_path  TEXT NOT NULL,
  filename  TEXT NOT NULL,
  mtime     REAL,
  UNIQUE(photo_id, rel_path)
);
CREATE INDEX IF NOT EXISTS photo_sources_photo ON photo_sources(photo_id);

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

CREATE VIRTUAL TABLE IF NOT EXISTS photo_fts USING fts5(caption, tags_text);
```

`scans` and `app_settings` are gone: nothing scans, and the only setting they held was the library folder.

- [ ] **Step 6: Add the version guard**

In `ivms777/db/connection.py`, add below `SCHEMA_PATH`:

```python
SCHEMA_VERSION = 2


class SchemaTooOldError(RuntimeError):
    """The database predates the upload schema and cannot be migrated in place."""
```

and replace `migrate`:

```python
def migrate(conn: sqlite3.Connection) -> None:
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version == SCHEMA_VERSION:
        return
    if version == 0 and _has_photos_table(conn):
        raise SchemaTooOldError(
            "This database was built by the folder-scan ingest and has no upgrade "
            "path — photos are now keyed by content hash. Delete the database file "
            "and re-upload."
        )
    conn.executescript(SCHEMA_PATH.read_text())
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")


def _has_photos_table(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'photos'"
    ).fetchone()
    return row is not None
```

A fresh file reports `user_version` 0 with no tables, so it is created and stamped. A v2 file returns early. A plan-01 file raises.

- [ ] **Step 7: Run the schema tests**

Run: `uv run pytest tests/test_db.py -q`
Expected: PASS.

- [ ] **Step 8: Add the originals store**

In `ivms777/config.py`, add next to `thumb_dir`:

```python
    @property
    def originals_dir(self) -> Path:
        return self.data_dir / "originals"
```

In `ivms777/web/deps.py`, add the field and build it:

```python
@dataclass
class AppContext:
    settings: Settings
    conn: sqlite3.Connection
    derived: LocalStorage
    originals: LocalStorage
```

```python
    return AppContext(
        settings=settings,
        conn=conn,
        derived=LocalStorage(settings.thumb_dir),
        originals=LocalStorage(settings.originals_dir),
    )
```

In `ivms777/storage/base.py`, add `free_bytes` to the protocol:

```python
class Storage(Protocol):
    def iter_keys(self) -> Iterator[str]: ...
    def read(self, key: str) -> bytes: ...
    def write(self, key: str, data: bytes) -> None: ...
    def exists(self, key: str) -> bool: ...
    def stat(self, key: str) -> StorageStat: ...
    def local_path(self, key: str) -> Path | None: ...
    def free_bytes(self) -> int: ...
```

In `ivms777/storage/local.py`, add the import `import shutil` and the method:

```python
    def free_bytes(self) -> int:
        """Space left where originals land. Checked before accepting an upload."""
        self.root.mkdir(parents=True, exist_ok=True)
        return shutil.disk_usage(self.root).free
```

Add to `tests/test_storage.py`:

```python
def test_free_bytes_reports_something_plausible(tmp_path):
    storage = LocalStorage(tmp_path / "originals")
    assert storage.free_bytes() > 0
```

Now give `ivms777/ingest/cli.py` its handler, replacing the empty `drain(context.conn, {})` from task 1:

```python
import time

from ivms777.config import get_settings
from ivms777.ingest.worker import drain, thumbnail_handler
from ivms777.web.deps import build_context

POLL_SECONDS = 10


def main() -> None:
    context = build_context(get_settings())
    settings = context.settings
    while True:
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
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
```

- [ ] **Step 9: Move every consumer off `path` and `duplicate_of`**

In `ivms777/ingest/worker.py`, `thumbnail_handler` reads the new column:

```python
    def handle(conn: sqlite3.Connection, photo_id: int) -> None:
        row = conn.execute(
            "SELECT storage_key, content_hash FROM photos WHERE id = ?", (photo_id,)
        ).fetchone()
        source: Path | None = originals.local_path(row["storage_key"])
        if source is None or not source.is_file():
            raise FileNotFoundError(row["storage_key"])
        key = make_thumbnails(source, row["content_hash"], derived, grid_px, detail_px)
        conn.execute("UPDATE photos SET thumb_key = ? WHERE id = ?", (key, photo_id))
```

In `ivms777/web/app.py`, the grid query counts extra sources instead of duplicate rows, and there is no `missing_since` or `duplicate_of` to filter on:

```python
BASE_SQL = (
    "SELECT p.id, p.caption, p.shot_at,"
    " (SELECT count(*) - 1 FROM photo_sources s WHERE s.photo_id = p.id) AS dupe_count"
    " FROM photos p"
    " WHERE p.owner_id = ? AND p.thumb_key IS NOT NULL"
)
```

`count(*) - 1` is the number of *redundant* copies, which is what the `×N` badge means. A photo with one source scores 0 and shows no badge.

The duplicates query groups sources instead of joining photos to themselves:

```python
DUPLICATE_SQL = (
    "SELECT p.id AS photo_id, p.bytes AS size, count(s.id) AS source_count,"
    " group_concat(s.rel_path, char(10)) AS paths"
    " FROM photos p JOIN photo_sources s ON s.photo_id = p.id"
    " WHERE p.owner_id = ? GROUP BY p.id HAVING count(s.id) > 1"
    " ORDER BY (count(s.id) - 1) * p.bytes DESC"
)


def duplicate_sets(context: AppContext) -> list[dict]:
    rows = context.conn.execute(DUPLICATE_SQL, (context.settings.owner_id,)).fetchall()
    return [
        {
            "photo_id": row["photo_id"],
            "paths": row["paths"].split("\n") if row["paths"] else [],
            "wasted_bytes": (row["size"] or 0) * (row["source_count"] - 1),
        }
        for row in rows
    ]
```

`progress_payload` selects a path that still exists:

```python
            ctx.conn.execute(
                "SELECT j.stage, j.error,"
                " (SELECT s.rel_path FROM photo_sources s WHERE s.photo_id = p.id"
                "  ORDER BY s.id LIMIT 1) AS path"
                " FROM jobs j JOIN photos p ON p.id = j.photo_id"
                " WHERE j.status = 'failed' ORDER BY path LIMIT 50"
            )
```

In `ivms777/search/facets.py`, `facet_counts` loses the two dead predicates:

```python
        "SELECT f.value_text AS value, count(*) AS n FROM photo_facets f"
        " JOIN photos p ON p.id = f.photo_id"
        " WHERE f.key = ? AND p.owner_id = ? AND f.value_text IS NOT NULL"
        " GROUP BY f.value_text ORDER BY n DESC, f.value_text LIMIT ?",
```

In `ivms777/web/templates/duplicates.html`, there is no canonical row any more — every path is just a path. Replace lines 8-15 with:

```html
      <img src="/thumb/{{ entry['photo_id'] }}" alt="{{ entry['paths'][0] }}" loading="lazy">
      <div class="dupe-paths">
        <p>{{ entry['paths']|length }} copies on disk</p>
        <ul>
          {% for path in entry['paths'] %}
            <li>{{ path }}</li>
          {% endfor %}
        </ul>
      </div>
```

The `wasted_total` line above it still works — `duplicate_sets` keeps returning `wasted_bytes` per entry.

- [ ] **Step 10: Move the remaining tests onto the factory**

In `tests/test_jobs.py`, `tests/test_worker.py`, `tests/test_facets.py`, `tests/test_search_facets.py`, `tests/test_web_library.py`, `tests/test_web_duplicates.py`, and `tests/test_web_facet_filters.py`, delete every local `INSERT INTO photos` helper and import `add_photo` from `tests.factories` instead. Two shapes cover every call site:

```python
from tests.factories import add_photo

photo_id = add_photo(conn, content_hash="h1")
photo_id = add_photo(conn, photo_id=3, content_hash="h3", thumb_key="ab/h3_320.jpg")
```

For `tests/test_web_duplicates.py`, a duplicate is now one photo with several sources:

```python
def test_duplicates_page_lists_every_path_and_the_wasted_space(client, conn):
    add_photo(
        conn,
        content_hash="dd" * 32,
        bytes=1000,
        sources=("Pictures/a.jpg", "Desktop/a copy.jpg", "Backup/a.jpg"),
    )
    body = client.get("/duplicates").text
    assert "Pictures/a.jpg" in body
    assert "Desktop/a copy.jpg" in body
    assert "3 copies on disk" in body


def test_a_photo_with_one_source_is_not_a_duplicate(client, conn):
    add_photo(conn, content_hash="ee" * 32, bytes=1000, sources=("Pictures/only.jpg",))
    assert "No duplicates found" in client.get("/duplicates").text
```

`duplicates.html` renders the wasted total as megabytes with one decimal (`'%.1f'|format(wasted_total / 1048576)`), so assert on the path text and the copy count rather than on a byte string.

In `tests/test_worker.py`, the fake originals store must hold the file under its content key:

```python
from ivms777.storage.keys import content_key

def test_thumbnail_handler_writes_a_thumb_key(conn, tmp_path):
    originals = LocalStorage(tmp_path / "originals")
    derived = LocalStorage(tmp_path / "thumbs")
    digest = "ee" * 32
    key = content_key(digest, ".jpg")
    make_jpeg(originals.local_path(key))
    photo_id = add_photo(conn, content_hash=digest)
    enqueue(conn, photo_id, "thumbnail")
    drain(conn, {"thumbnail": thumbnail_handler(originals, derived, 320, 1600)})
    row = conn.execute("SELECT thumb_key FROM photos WHERE id = ?", (photo_id,)).fetchone()
    assert row["thumb_key"] == thumb_key(digest, 320)
```

`make_jpeg` already creates parent directories, so writing straight to `local_path(key)` is safe.

- [ ] **Step 11: Run the suite**

Run: `uv run pytest -q`
Expected: PASS. If a stale `ivms777.db` exists in your data dir, delete it — the guard from step 6 will tell you so by name.

- [ ] **Step 12: Checkpoint**

Report: the new tables, the columns dropped, and every test file moved onto `tests/factories`. Stop for review. Do not commit.

---

### Task 3: The receive stage

One function turns uploaded bytes into a photo. It is the only writer that creates `photos` rows, and it is idempotent on content hash.

**Files:**
- Create: `ivms777/ingest/receive.py`
- Create: `tests/test_receive.py`
- Modify: `tests/fixtures.py`

**Interfaces:**
- Consumes: `content_key`, `suffix_of` from `ivms777.storage.keys`; `read_exif` from `ivms777.ingest.exif`; `derive_facets`, `store_facets` from `ivms777.ingest.facets`; `enqueue` from `ivms777.ingest.jobs`.
- Produces:
  - `ivms777.ingest.receive.ReceiveResult(photo_id: int, content_hash: str, created: bool, source_added: bool)`
  - `ivms777.ingest.receive.HashMismatchError`, `ivms777.ingest.receive.UnreadableImageError`
  - `ivms777.ingest.receive.receive(conn, originals, *, owner_id, upload_id, rel_path, declared_hash, data, mtime=None) -> ReceiveResult`
  - `ivms777.ingest.receive.link_existing(conn, *, owner_id, upload_id, rel_path, content_hash, mtime=None) -> int | None`
  - `ivms777.ingest.receive.known_hashes(conn, owner_id, hashes) -> set[str]`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_receive.py`:

```python
import hashlib

import pytest

from ivms777.ingest.jobs import stage_counts
from ivms777.ingest.receive import (
    HashMismatchError,
    UnreadableImageError,
    known_hashes,
    link_existing,
    receive,
)
from ivms777.storage.keys import content_key
from ivms777.storage.local import LocalStorage
from tests.factories import add_upload
from tests.fixtures import jpeg_bytes, jpeg_bytes_with_exif


@pytest.fixture
def originals(tmp_path):
    return LocalStorage(tmp_path / "originals")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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
```

- [ ] **Step 2: Add byte-producing fixtures**

Append to `tests/fixtures.py`:

```python
import io


def jpeg_bytes(size: tuple[int, int] = (64, 48), color: str = "red") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="JPEG")
    return buffer.getvalue()


def jpeg_bytes_with_exif(
    when: str = "2025:07:12 14:30:00", model: str = "TestCam"
) -> bytes:
    image = Image.new("RGB", (64, 48), "blue")
    exif = image.getexif()
    exif[0x0110] = model          # Model
    exif[0x9003] = when           # DateTimeOriginal
    exif[0x010F] = "TestMake"     # Make
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", exif=exif)
    return buffer.getvalue()
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_receive.py -q`
Expected: FAIL — `ModuleNotFoundError: ivms777.ingest.receive`.

- [ ] **Step 4: Write the receive module**

Create `ivms777/ingest/receive.py`:

```python
import hashlib
import io
import json
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import PurePosixPath

from PIL import Image, UnidentifiedImageError

from ivms777.ingest.exif import read_exif
from ivms777.ingest.facets import derive_facets, store_facets
from ivms777.ingest.jobs import enqueue
from ivms777.storage.base import Storage
from ivms777.storage.keys import content_key, suffix_of


class HashMismatchError(ValueError):
    """The bytes received do not hash to what the client declared."""


class UnreadableImageError(ValueError):
    """The bytes are not an image this build can open."""


@dataclass(frozen=True)
class ReceiveResult:
    photo_id: int
    content_hash: str
    created: bool
    source_added: bool


def _now() -> str:
    return datetime.now(UTC).isoformat()


def known_hashes(conn: sqlite3.Connection, owner_id: int, hashes: Iterable[str]) -> set[str]:
    """Which of these hashes this owner already has. Drives the upload probe."""
    wanted = list(dict.fromkeys(hashes))
    found: set[str] = set()
    # SQLite's parameter limit is 999 by default, so ask in chunks.
    for start in range(0, len(wanted), 500):
        chunk = wanted[start : start + 500]
        placeholders = ", ".join("?" for _ in chunk)
        rows = conn.execute(
            f"SELECT content_hash FROM photos WHERE owner_id = ?"
            f" AND content_hash IN ({placeholders})",
            (owner_id, *chunk),
        )
        found.update(row["content_hash"] for row in rows)
    return found


def _add_source(
    conn: sqlite3.Connection,
    photo_id: int,
    upload_id: int,
    rel_path: str,
    mtime: float | None,
) -> bool:
    cursor = conn.execute(
        "INSERT INTO photo_sources(photo_id, upload_id, rel_path, filename, mtime)"
        " VALUES (?, ?, ?, ?, ?) ON CONFLICT(photo_id, rel_path) DO NOTHING",
        (photo_id, upload_id, rel_path, PurePosixPath(rel_path).name, mtime),
    )
    return cursor.rowcount == 1


def link_existing(
    conn: sqlite3.Connection,
    *,
    owner_id: int,
    upload_id: int,
    rel_path: str,
    content_hash: str,
    mtime: float | None = None,
) -> int | None:
    """Record another local path for content already held. Returns None if unknown.

    This is what keeps duplicate detection honest when the probe tells the client
    not to send bytes it already has — the path still has to be recorded.
    """
    row = conn.execute(
        "SELECT id FROM photos WHERE owner_id = ? AND content_hash = ?",
        (owner_id, content_hash),
    ).fetchone()
    if row is None:
        return None
    _add_source(conn, int(row["id"]), upload_id, rel_path, mtime)
    return int(row["id"])


def receive(
    conn: sqlite3.Connection,
    originals: Storage,
    *,
    owner_id: int,
    upload_id: int,
    rel_path: str,
    declared_hash: str,
    data: bytes,
    mtime: float | None = None,
) -> ReceiveResult:
    """Turn uploaded bytes into a photo, or another source for one already held."""
    digest = hashlib.sha256(data).hexdigest()
    if digest != declared_hash.lower():
        raise HashMismatchError(f"declared {declared_hash!r}, received {digest!r}")

    existing = conn.execute(
        "SELECT id FROM photos WHERE owner_id = ? AND content_hash = ?", (owner_id, digest)
    ).fetchone()
    if existing is not None:
        photo_id = int(existing["id"])
        return ReceiveResult(
            photo_id=photo_id,
            content_hash=digest,
            created=False,
            source_added=_add_source(conn, photo_id, upload_id, rel_path, mtime),
        )

    # Open before storing, so a non-image never leaves a file or a row behind.
    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise UnreadableImageError(f"{rel_path}: {error}") from error

    key = content_key(digest, suffix_of(rel_path))
    originals.write(key, data)

    local = originals.local_path(key)
    facts = read_exif(local) if local is not None else None
    now = _now()
    cursor = conn.execute(
        "INSERT INTO photos(owner_id, content_hash, storage_key, bytes, width, height,"
        " shot_at, camera, lens, gps_lat, gps_lon, exif_json, created_at, updated_at)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            owner_id, digest, key, len(data),
            facts.width if facts else None,
            facts.height if facts else None,
            facts.shot_at if facts else None,
            facts.camera if facts else None,
            facts.lens if facts else None,
            facts.gps_lat if facts else None,
            facts.gps_lon if facts else None,
            json.dumps(facts.raw) if facts else None,
            now, now,
        ),
    )
    photo_id = int(cursor.lastrowid)
    if facts is not None:
        store_facets(
            conn, photo_id, derive_facets(facts, width=facts.width, height=facts.height)
        )
    added = _add_source(conn, photo_id, upload_id, rel_path, mtime)
    enqueue(conn, photo_id, "thumbnail")
    return ReceiveResult(
        photo_id=photo_id, content_hash=digest, created=True, source_added=added
    )
```

`Image.verify()` closes the file, which is why the EXIF read opens the stored copy separately rather than reusing the probe.

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_receive.py -q`
Expected: PASS.

- [ ] **Step 6: Run the whole suite**

Run: `uv run pytest -q`
Expected: PASS.

- [ ] **Step 7: Checkpoint**

Report: the receive contract, and which of its guarantees each test pins down. Stop for review. Do not commit.

---

### Task 4: The upload API

**Files:**
- Create: `ivms777/web/upload_api.py`
- Create: `tests/test_upload_api.py`
- Modify: `ivms777/web/app.py`

**Interfaces:**
- Consumes: `receive`, `link_existing`, `known_hashes`, `HashMismatchError`, `UnreadableImageError` from `ivms777.ingest.receive`; `AppContext` from `ivms777.web.deps`.
- Produces: `ivms777.web.upload_api.register(app, context_getter, drain_now)` mounting:
  - `POST /api/upload/start` → `{"upload_id": int}`
  - `POST /api/upload/probe` → `{"needed": [hash, ...]}`
  - `POST /api/upload/file` (multipart) → `{"status": "stored"|"linked", "photo_id": int}`
  - `POST /api/upload/finish` → `{"offered": int, "sent": int, "failed": int}`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_upload_api.py`:

```python
import hashlib

import pytest
from fastapi.testclient import TestClient

from ivms777.web.app import create_app
from tests.fixtures import jpeg_bytes


@pytest.fixture
def client(settings):
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def conn(client):
    return client.app.state.context.conn


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def start(client, root_label="Pictures") -> int:
    response = client.post("/api/upload/start", json={"root_label": root_label})
    assert response.status_code == 200
    return response.json()["upload_id"]


def send(client, upload_id, rel_path, data):
    return client.post(
        "/api/upload/file",
        data={"upload_id": upload_id, "rel_path": rel_path, "content_hash": sha(data)},
        files={"file": (rel_path.rsplit("/", 1)[-1], data, "image/jpeg")},
    )


def test_probe_asks_for_everything_when_the_library_is_empty(client):
    upload_id = start(client)
    data = jpeg_bytes()
    response = client.post(
        "/api/upload/probe",
        json={"upload_id": upload_id, "files": [{"hash": sha(data), "rel_path": "a.jpg"}]},
    )
    assert response.json()["needed"] == [sha(data)]


def test_probe_skips_known_bytes_but_still_records_the_new_path(client, conn):
    upload_id = start(client)
    data = jpeg_bytes()
    assert send(client, upload_id, "Pictures/a.jpg", data).status_code == 200

    response = client.post(
        "/api/upload/probe",
        json={
            "upload_id": upload_id,
            "files": [{"hash": sha(data), "rel_path": "Backup/a.jpg"}],
        },
    )
    assert response.json()["needed"] == []
    paths = {
        row["rel_path"] for row in conn.execute("SELECT rel_path FROM photo_sources")
    }
    assert paths == {"Pictures/a.jpg", "Backup/a.jpg"}


def test_uploading_a_file_stores_it_and_reports_the_photo(client, conn):
    upload_id = start(client)
    data = jpeg_bytes()
    response = send(client, upload_id, "Pictures/a.jpg", data)
    assert response.status_code == 200
    assert response.json()["status"] == "stored"
    assert conn.execute("SELECT count(*) FROM photos").fetchone()[0] == 1


def test_sending_bytes_already_held_links_instead_of_storing(client):
    upload_id = start(client)
    data = jpeg_bytes()
    send(client, upload_id, "Pictures/a.jpg", data)
    response = send(client, upload_id, "Desktop/a.jpg", data)
    assert response.json()["status"] == "linked"


def test_a_corrupted_transfer_is_rejected_with_422(client, conn):
    upload_id = start(client)
    data = jpeg_bytes()
    response = client.post(
        "/api/upload/file",
        data={"upload_id": upload_id, "rel_path": "a.jpg", "content_hash": "00" * 32},
        files={"file": ("a.jpg", data, "image/jpeg")},
    )
    assert response.status_code == 422
    assert conn.execute("SELECT count(*) FROM photos").fetchone()[0] == 0
    assert conn.execute(
        "SELECT files_failed FROM uploads WHERE id = ?", (upload_id,)
    ).fetchone()[0] == 1


def test_a_non_image_is_rejected_with_415(client):
    upload_id = start(client)
    data = b"not an image at all"
    response = client.post(
        "/api/upload/file",
        data={"upload_id": upload_id, "rel_path": "notes.jpg", "content_hash": sha(data)},
        files={"file": ("notes.jpg", data, "image/jpeg")},
    )
    assert response.status_code == 415


def test_counters_add_up_across_an_upload(client, conn):
    upload_id = start(client)
    first, second = jpeg_bytes(color="red"), jpeg_bytes(color="green")
    client.post(
        "/api/upload/probe",
        json={
            "upload_id": upload_id,
            "files": [
                {"hash": sha(first), "rel_path": "a.jpg"},
                {"hash": sha(second), "rel_path": "b.jpg"},
            ],
        },
    )
    send(client, upload_id, "a.jpg", first)
    send(client, upload_id, "b.jpg", second)
    summary = client.post("/api/upload/finish", json={"upload_id": upload_id}).json()
    assert summary == {"offered": 2, "sent": 2, "failed": 0}
    assert conn.execute(
        "SELECT finished_at FROM uploads WHERE id = ?", (upload_id,)
    ).fetchone()[0] is not None


def test_an_unknown_upload_id_is_rejected(client):
    data = jpeg_bytes()
    response = send(client, 999, "a.jpg", data)
    assert response.status_code == 404


def test_uploaded_photos_appear_in_the_library_with_a_thumbnail(client, conn):
    upload_id = start(client)
    send(client, upload_id, "Pictures/a.jpg", jpeg_bytes())
    client.post("/api/upload/finish", json={"upload_id": upload_id})
    assert conn.execute(
        "SELECT thumb_key FROM photos LIMIT 1"
    ).fetchone()["thumb_key"] is not None
    assert client.get("/library").status_code == 200
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_upload_api.py -q`
Expected: FAIL — 404 on every upload route.

- [ ] **Step 3: Write the upload API module**

Create `ivms777/web/upload_api.py`:

```python
from collections.abc import Callable
from datetime import UTC, datetime

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from ivms777.ingest.receive import (
    HashMismatchError,
    UnreadableImageError,
    known_hashes,
    link_existing,
    receive,
)
from ivms777.web.deps import AppContext

# Refuse an upload that would leave the disk this close to full. Failing at the
# door with a clear message beats failing halfway through a 5,000-photo library.
FREE_SPACE_FLOOR = 512 * 1024 * 1024


class StartRequest(BaseModel):
    root_label: str = Field(default="photos", max_length=200)


class ProbeFile(BaseModel):
    hash: str = Field(min_length=64, max_length=64)
    rel_path: str = Field(min_length=1, max_length=1024)


class ProbeRequest(BaseModel):
    upload_id: int
    files: list[ProbeFile]


class FinishRequest(BaseModel):
    upload_id: int


def _now() -> str:
    return datetime.now(UTC).isoformat()


def register(
    app: FastAPI,
    context: Callable[[], AppContext],
    drain_now: Callable[[], None],
) -> None:
    def _upload_row(ctx: AppContext, upload_id: int):
        row = ctx.conn.execute(
            "SELECT id FROM uploads WHERE id = ? AND owner_id = ?",
            (upload_id, ctx.settings.owner_id),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="unknown upload")
        return row

    @app.post("/api/upload/start")
    def start(payload: StartRequest) -> dict:
        ctx = context()
        free = ctx.originals.free_bytes()
        if free < FREE_SPACE_FLOOR:
            raise HTTPException(
                status_code=507,
                detail=f"only {free // (1024 * 1024)} MB free on the server",
            )
        cursor = ctx.conn.execute(
            "INSERT INTO uploads(owner_id, root_label, started_at) VALUES (?, ?, ?)",
            (ctx.settings.owner_id, payload.root_label, _now()),
        )
        return {"upload_id": int(cursor.lastrowid)}

    @app.post("/api/upload/probe")
    def probe(payload: ProbeRequest) -> dict:
        ctx = context()
        _upload_row(ctx, payload.upload_id)
        owner_id = ctx.settings.owner_id
        held = known_hashes(ctx.conn, owner_id, [item.hash for item in payload.files])

        needed: list[str] = []
        for item in payload.files:
            if item.hash in held:
                # Bytes we already have, at a path we may not. Record the path now;
                # the client will not send this file at all.
                link_existing(
                    ctx.conn,
                    owner_id=owner_id,
                    upload_id=payload.upload_id,
                    rel_path=item.rel_path,
                    content_hash=item.hash,
                )
            elif item.hash not in needed:
                needed.append(item.hash)

        ctx.conn.execute(
            "UPDATE uploads SET files_offered = files_offered + ? WHERE id = ?",
            (len(payload.files), payload.upload_id),
        )
        return {"needed": needed}

    @app.post("/api/upload/file")
    def upload_file(
        upload_id: int = Form(...),
        rel_path: str = Form(...),
        content_hash: str = Form(...),
        file: UploadFile = File(...),
    ) -> dict:
        ctx = context()
        _upload_row(ctx, upload_id)
        data = file.file.read()
        try:
            result = receive(
                ctx.conn,
                ctx.originals,
                owner_id=ctx.settings.owner_id,
                upload_id=upload_id,
                rel_path=rel_path,
                declared_hash=content_hash,
                data=data,
            )
        except HashMismatchError as error:
            ctx.conn.execute(
                "UPDATE uploads SET files_failed = files_failed + 1 WHERE id = ?",
                (upload_id,),
            )
            raise HTTPException(status_code=422, detail=str(error)) from error
        except UnreadableImageError as error:
            ctx.conn.execute(
                "UPDATE uploads SET files_failed = files_failed + 1 WHERE id = ?",
                (upload_id,),
            )
            raise HTTPException(status_code=415, detail=str(error)) from error

        ctx.conn.execute(
            "UPDATE uploads SET files_sent = files_sent + 1 WHERE id = ?", (upload_id,)
        )
        return {
            "status": "stored" if result.created else "linked",
            "photo_id": result.photo_id,
        }

    @app.post("/api/upload/finish")
    def finish(payload: FinishRequest) -> dict:
        ctx = context()
        _upload_row(ctx, payload.upload_id)
        ctx.conn.execute(
            "UPDATE uploads SET finished_at = ? WHERE id = ?", (_now(), payload.upload_id)
        )
        drain_now()
        row = ctx.conn.execute(
            "SELECT files_offered, files_sent, files_failed FROM uploads WHERE id = ?",
            (payload.upload_id,),
        ).fetchone()
        return {
            "offered": row["files_offered"],
            "sent": row["files_sent"],
            "failed": row["files_failed"],
        }
```

- [ ] **Step 4: Mount it and give it a way to build thumbnails**

In `ivms777/web/app.py`, inside `create_app` after `context` is defined, add:

```python
    def drain_now() -> None:
        """Build thumbnails for whatever has arrived.

        The `worker` container drains continuously in deployment. Doing it here
        too means a single-container run — and every test — still produces a
        usable grid without waiting on a poll.
        """
        ctx = context()
        drain(
            ctx.conn,
            {
                "thumbnail": thumbnail_handler(
                    ctx.originals,
                    ctx.derived,
                    ctx.settings.thumb_grid_px,
                    ctx.settings.thumb_detail_px,
                )
            },
        )

    register_upload_api(app, context, drain_now)
```

and import it at the top:

```python
from ivms777.web.upload_api import register as register_upload_api
```

This is what the `drain`/`thumbnail_handler` imports left over from task 1 are for; drop any `# noqa: F401` you added.

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_upload_api.py -q`
Expected: PASS.

- [ ] **Step 6: Run the whole suite**

Run: `uv run pytest -q`
Expected: PASS.

- [ ] **Step 7: Checkpoint**

Report: the four endpoints, their status codes, and how the probe keeps duplicate paths recorded without transferring bytes. Stop for review. Do not commit.

---

### Task 5: The upload screen

**Files:**
- Create: `ivms777/web/static/hash-worker.js`
- Create: `ivms777/web/static/upload.js`
- Create: `ivms777/web/templates/upload.html`
- Modify: `ivms777/web/app.py`
- Modify: `ivms777/web/templates/_progress.html`
- Create: `tests/test_web_upload.py`

**Interfaces:**
- Consumes: the four `/api/upload/*` endpoints from task 4.
- Produces: `GET /upload` rendering the picker and the progress panel.

- [ ] **Step 1: Write the failing test**

Create `tests/test_web_upload.py`:

```python
import pytest
from fastapi.testclient import TestClient

from ivms777.web.app import create_app
from tests.factories import add_photo


@pytest.fixture
def client(settings):
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client


def test_upload_page_offers_a_directory_picker(client):
    body = client.get("/upload").text
    assert "webkitdirectory" in body
    assert "/static/upload.js" in body


def test_upload_page_shows_stage_progress(client):
    assert "thumbnail" in client.get("/upload").text


def test_progress_fragment_lists_failed_files_by_path(client):
    conn = client.app.state.context.conn
    photo_id = add_photo(conn, content_hash="ab" * 32, sources=("Pictures/bad.jpg",))
    conn.execute(
        "INSERT INTO jobs(photo_id, stage, status, error, updated_at)"
        " VALUES (?, 'thumbnail', 'failed', 'cannot open', '2026-01-01T00:00:00')",
        (photo_id,),
    )
    body = client.get("/upload/progress").text
    assert "Pictures/bad.jpg" in body
    assert "cannot open" in body


def test_static_assets_are_served(client):
    for asset in ("/static/upload.js", "/static/hash-worker.js"):
        assert client.get(asset).status_code == 200
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_web_upload.py -q`
Expected: FAIL — 404 on `/upload`.

- [ ] **Step 3: Write the hashing worker**

Create `ivms777/web/static/hash-worker.js`:

```js
// Hashes one file at a time and posts the digest back. Runs off the main thread
// so a 5,000-file selection never freezes the tab.
//
// WebCrypto has no incremental digest, so each file is read whole. Files are
// handled strictly one at a time, which bounds memory to the largest single
// photo rather than the size of the selection.

function toHex(buffer) {
  return Array.from(new Uint8Array(buffer))
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('');
}

self.onmessage = async (event) => {
  const { files } = event.data;
  for (let index = 0; index < files.length; index += 1) {
    const entry = files[index];
    try {
      const buffer = await entry.file.arrayBuffer();
      const digest = await crypto.subtle.digest('SHA-256', buffer);
      self.postMessage({ type: 'hashed', index, hash: toHex(digest) });
    } catch (error) {
      self.postMessage({ type: 'error', index, message: String(error) });
    }
  }
  self.postMessage({ type: 'done' });
};
```

- [ ] **Step 4: Write the upload driver**

Create `ivms777/web/static/upload.js`:

```js
// Drives the three steps in spec section 3.2b: hash locally, ask what is new,
// send only that.

const CONCURRENCY = 4;
const IMAGE_PATTERN = /\.(jpe?g|png|heic|heif|webp|tiff?)$/i;

const picker = document.getElementById('picker');
const startButton = document.getElementById('start');
const status = document.getElementById('status');
const bar = document.getElementById('bar');
const failures = document.getElementById('upload-failures');

let selected = [];

function say(text) {
  status.textContent = text;
}

function setProgress(done, total) {
  bar.max = total;
  bar.value = done;
}

function noteFailure(relPath, message) {
  const item = document.createElement('li');
  item.textContent = `${relPath} — ${message}`;
  failures.appendChild(item);
}

picker.addEventListener('change', () => {
  selected = Array.from(picker.files)
    .filter((file) => IMAGE_PATTERN.test(file.name))
    .map((file) => ({ file, relPath: file.webkitRelativePath || file.name }));
  say(`${selected.length} images selected`);
  startButton.disabled = selected.length === 0;
});

function hashAll(entries) {
  return new Promise((resolve) => {
    const worker = new Worker('/static/hash-worker.js');
    const hashes = new Array(entries.length).fill(null);
    let done = 0;
    worker.onmessage = (event) => {
      const message = event.data;
      if (message.type === 'hashed') {
        hashes[message.index] = message.hash;
      } else if (message.type === 'error') {
        noteFailure(entries[message.index].relPath, message.message);
      } else if (message.type === 'done') {
        worker.terminate();
        resolve(hashes);
        return;
      }
      done += 1;
      setProgress(done, entries.length);
      say(`hashing ${done} / ${entries.length}`);
    };
    worker.postMessage({ files: entries });
  });
}

async function postJSON(url, body) {
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new Error(`${url} returned ${response.status}`);
  return response.json();
}

async function sendOne(uploadId, entry, hash) {
  const form = new FormData();
  form.append('upload_id', uploadId);
  form.append('rel_path', entry.relPath);
  form.append('content_hash', hash);
  form.append('file', entry.file, entry.file.name);
  const response = await fetch('/api/upload/file', { method: 'POST', body: form });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({ detail: response.status }));
    noteFailure(entry.relPath, detail.detail);
  }
}

async function run() {
  startButton.disabled = true;
  failures.replaceChildren();

  const rootLabel = selected[0]?.relPath.split('/')[0] || 'photos';
  const hashes = await hashAll(selected);

  const { upload_id: uploadId } = await postJSON('/api/upload/start', {
    root_label: rootLabel,
  });

  const pairs = selected
    .map((entry, index) => ({ entry, hash: hashes[index] }))
    .filter((pair) => pair.hash !== null);

  const { needed } = await postJSON('/api/upload/probe', {
    upload_id: uploadId,
    files: pairs.map((pair) => ({ hash: pair.hash, rel_path: pair.entry.relPath })),
  });

  const needSet = new Set(needed);
  // One file per hash: the rest are copies whose paths the probe already recorded.
  const seen = new Set();
  const queue = pairs.filter((pair) => {
    if (!needSet.has(pair.hash) || seen.has(pair.hash)) return false;
    seen.add(pair.hash);
    return true;
  });

  const total = queue.length;
  say(`${total} of ${pairs.length} need uploading`);
  let done = 0;
  setProgress(0, total);

  async function drain() {
    while (queue.length > 0) {
      const pair = queue.shift();
      await sendOne(uploadId, pair.entry, pair.hash);
      done += 1;
      setProgress(done, total);
      say(`uploading ${done} / ${total}`);
    }
  }
  await Promise.all(Array.from({ length: CONCURRENCY }, drain));

  const summary = await postJSON('/api/upload/finish', { upload_id: uploadId });
  say(`done — ${summary.sent} uploaded, ${summary.offered} seen, ${summary.failed} failed`);
  startButton.disabled = false;
  htmx.trigger('#progress', 'refresh');
}

startButton.addEventListener('click', () => {
  run().catch((error) => {
    say(`upload failed: ${error.message}`);
    startButton.disabled = false;
  });
});
```

- [ ] **Step 5: Write the template**

Create `ivms777/web/templates/upload.html`. Read `ivms777/web/templates/library.html` first and match its `{% extends %}` and block names exactly:

```html
{% extends "base.html" %}
{% block content %}
<h1>Upload photos</h1>

<p class="hint">
  Pick the folder holding your photos. Nothing is sent until every file has been
  hashed locally and the server has said which ones it does not already have.
</p>

<input type="file" id="picker" webkitdirectory directory multiple accept="image/*">
<button id="start" disabled>Start upload</button>

<p id="status">no folder selected</p>
<progress id="bar" value="0" max="1"></progress>

<ul id="upload-failures" class="failures"></ul>

{% include "_progress.html" %}

<script src="/static/upload.js" defer></script>
{% endblock %}
```

`_progress.html` carries its own `hx-get`, `id="progress"`, and `hx-swap="outerHTML"` from task 1 step 6, so including it directly is enough — do not wrap it in a second polling div or both will fire.

- [ ] **Step 6: Serve the page**

In `ivms777/web/app.py`, add next to the other routes:

```python
    @app.get("/upload", response_class=HTMLResponse)
    def upload_page(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(request, "upload.html", progress_payload())
```

- [ ] **Step 7: Run the tests**

Run: `uv run pytest tests/test_web_upload.py -q`
Expected: PASS.

- [ ] **Step 8: Run the whole suite**

Run: `uv run pytest -q`
Expected: PASS.

- [ ] **Step 9: Verify by hand, because the JavaScript has no test**

The worker and driver are not unit-tested — there is no JS test runner in this project and adding one is not justified by ~200 lines of glue. The API beneath them is covered by `tests/test_upload_api.py`. Verify the browser half by hand, once:

```bash
uv run uvicorn ivms777.web.app:app_factory --factory --port 8000
```

Open `http://localhost:8000/upload` in Chrome and check, in order:
1. Picking a folder of ~20 photos reports the count and enables the button.
2. The bar fills during hashing without the tab going unresponsive.
3. The upload phase reports fewer files than selected when the folder holds copies.
4. `/library` shows thumbnails for every distinct image.
5. `/duplicates` lists the copies with every path.
6. Picking the same folder a second time uploads nothing and reports 0 sent.

Record what you saw in the checkpoint.

- [ ] **Step 10: Update the docs**

In `README.md`, replace any instruction to bind-mount a photo folder or press "Start scan" with: open `/upload`, pick a folder, wait. Remove `IVMS777_HOST_ROOT` from the documented environment variables. Do the same in `compose.yaml`, `compose.mac.yaml`, `compose.jetson.yaml`, `compose.cloud.yaml`, and `compose.dev.yaml`: delete the `/host` bind mount and the `IVMS777_HOST_ROOT` variable, and add a named volume for `/data/originals` if the data volume does not already cover it.

- [ ] **Step 11: Checkpoint**

Report: the manual verification results, and the compose files changed. Stop for review. Do not commit.

---

## What plan 02 delivers

Open `/upload`, pick a folder, and watch it hash locally, transfer only what the server has never seen, and fill in a thumbnail grid. No host mount, no folder picker on the server, no path the server walks — the same deployment works whether the photos are on the same machine or a continent away.

A photo is its bytes. Uploading the same image from five folders creates one row, stores one copy, queues one thumbnail, and records five paths — so `/duplicates` can tell you exactly which copies on your disk are redundant and how much space they cost, without a reconciliation pass.

Everything plan 01 built on top of ingest still works unchanged: 25 EXIF facets with live counts, range filters, numeric sorting, the resumable job queue, and the failed-file list.

**Not yet working:** stage 2. There is no manifest endpoint, no layouts, and no `ivms777-sync` — nothing writes to your disk yet. `/photo/{id}` still 404s.

## Following plans

| Plan | Spec phase | Delivers |
|---|---|---|
| 03 | 2 | SigLIP embeddings, taxonomy scoring, semantic + tag-facet + keyword + fusion search, similar photos, `/photo` detail |
| 04 | 3 | Caption stage against the inference service, captions in the UI |
| 05 | 4 | Query planner, parsed-filter chips, vocabulary mining |
| 06 | 5 | Event, cluster, and duplicate groups, `/groups` |
| 07 | 6 | Ask-your-library chat with streaming and citations |
| 08 | 7 | Stage 2 — layouts, `/api/manifest`, `/export`, and the `ivms777-sync` CLI with plan, apply, undo, and verify |
