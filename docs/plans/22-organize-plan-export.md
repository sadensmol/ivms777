# Stage 2 — Organize → plan document → `ivms777-sync` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the organizer selected in Organize into a downloadable plan document, and ship the `ivms777-sync` CLI that applies it — creating one folder per album inside the user's own source folder and moving each file into it.

**Architecture:** `organize/` (server-side) turns the `Album` list an existing `albums/` organizer already produces into a JSON plan document; `/organize/plan` previews it and `/api/plan` downloads it. `ivms777_sync/` is a standalone CLI that reads that document, matches files on disk **by content hash**, and applies the moves through a journaled staging directory. The app never touches the user's disk; the CLI is the only writer.

**Tech Stack:** Python 3.12, FastAPI + Jinja + HTMX, SQLite, `pytest`. The CLI uses the standard library plus `httpx` (for `verify --url`) and nothing else.

**Spec:** `docs/design.md` §12 (with §3.2b, §5, §8, §11, §13) and `docs/sync-cli.md` — the exact plan/CLI contract. Read both before starting.

## Global Constraints

- **The app never reads, writes, moves, or deletes anything on the user's filesystem.** Only `ivms777_sync`, running on the user's machine, does — design §3.2b, §3.2c. No task here may add a filesystem walk to `app` or `worker`.
- **`ivms777_sync` imports nothing from the `ivms777` packages** (`db`, `albums`, `storage`, `config`, …) and imports no torch/transformers/Pillow — design §14. Standard library plus `httpx` only. Task 5 adds a test that enforces this.
- **Filenames never change.** A plan moves a file between folders; only a name collision inside one folder appends `_<hash8>`.
- **Nothing is ever deleted.** Redundant copies move to `_duplicates/`; unknown files are counted and left alone.
- **Every operation is journaled before it runs** and reversible by `undo`.
- `PLAN_VERSION = 1`. The plan document schema is exactly the one in `docs/sync-cli.md` — if a task changes it, `docs/sync-cli.md` changes in the same task (project CLAUDE.md).
- Tests live in `tests/`; run with `uv run pytest`. Lint with `uv run ruff check .` (line length 100).
- **Never run `git commit`** — the user commits. Each task ends with a green suite and a stop for review.

---

## File Structure

| File | Responsibility |
|---|---|
| `organize/names.py` | Album title → one safe, deterministic path segment; de-duplicate collisions. |
| `organize/plan.py` | `Album` list + DB facts → the plan document dict. The only place ownership and `_unsorted/` are decided. |
| `albums/memory_store.py` (modify) | Carry a memory's `kind` (`event`/`theme`) through `params`. |
| `albums/memories.py` (modify) | Expose that kind as `Album.meta["memory_kind"]`. |
| `web/app.py` (modify) | `GET /api/plan` (download) and `GET /organize/plan` (preview). |
| `web/templates/organize_plan.html` | The preview + confirm screen. |
| `web/templates/organize.html` (modify) | The **Organize files…** button. |
| `ivms777_sync/plan.py` | Read + validate a plan document. Rejects traversal targets. |
| `ivms777_sync/scan.py` | Walk a root and hash every file. |
| `ivms777_sync/match.py` | Plan + scan → the move list, missing/unknown counts, conflicts. |
| `ivms777_sync/journal.py` | Append-only JSONL journal: write before acting, read for `undo`. |
| `ivms777_sync/apply.py` | Execute moves through a staging dir; write `description.txt` / `photos.txt`; `undo`. |
| `ivms777_sync/cli.py` | `apply` / `undo` / `verify`, the summary, the confirmation prompt. |
| `ivms777_sync/client.py` | `verify --url`: fetch a fresh plan over HTTP. |

---

### Task 1: Folder names

**Files:**
- Create: `organize/__init__.py`, `organize/names.py`
- Test: `tests/test_organize_names.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `folder_name(title: str, *, fallback: str) -> str` and `unique_folder_names(pairs: list[tuple[str, str]]) -> list[str]` (each pair is `(title, fallback)`, returns one folder segment per pair, in order).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_organize_names.py
from organize.names import folder_name, unique_folder_names


def test_reserved_and_separator_characters_become_dashes():
    assert folder_name('Trip: Kyiv/Rome <2024>', fallback="k") == "Trip- Kyiv-Rome -2024-"


def test_whitespace_is_collapsed_and_trimmed():
    assert folder_name("  June   2024 \n", fallback="k") == "June 2024"


def test_trailing_dots_are_stripped_windows_cannot_hold_them():
    assert folder_name("June 2024...", fallback="k") == "June 2024"


def test_a_title_that_sanitizes_to_nothing_falls_back_to_the_album_key():
    assert folder_name("///", fallback="memory-3") == "memory-3"


def test_long_titles_are_truncated_to_120_characters():
    assert len(folder_name("x" * 400, fallback="k")) == 120


def test_colliding_folder_names_get_a_deterministic_suffix():
    names = unique_folder_names(
        [("Waterfalls", "a"), ("Waterfalls", "b"), ("waterfalls", "c")]
    )
    assert names == ["Waterfalls", "Waterfalls_2", "waterfalls_3"]
```

- [ ] **Step 2: Run it and watch it fail**

Run: `uv run pytest tests/test_organize_names.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'organize'`

- [ ] **Step 3: Implement**

```python
# organize/names.py
"""Album title -> one safe path segment (docs/sync-cli.md, "Folder names").

The CLI creates these folders inside the user's own photo folder, so the rules
are the strictest common denominator: no separators, nothing Windows rejects,
nothing that ends in a dot or a space. Deterministic, so a re-run of the same
plan produces the same tree.
"""

MAX_SEGMENT = 120
_ILLEGAL = set('/\\<>:"|?*')


def folder_name(title: str, *, fallback: str) -> str:
    cleaned = "".join(
        "-" if (ch in _ILLEGAL or ord(ch) < 32) else ch for ch in title
    )
    cleaned = " ".join(cleaned.split())
    cleaned = cleaned.strip(" .")[:MAX_SEGMENT].strip(" .")
    return cleaned or fallback


def unique_folder_names(pairs: list[tuple[str, str]]) -> list[str]:
    """One folder per (title, fallback), de-duplicated in order.

    The comparison is case-insensitive because macOS and Windows filesystems are:
    two albums differing only in case would otherwise collide on disk.
    """
    taken: set[str] = set()
    names: list[str] = []
    for title, fallback in pairs:
        base = folder_name(title, fallback=fallback)
        name, n = base, 1
        while name.casefold() in taken:
            n += 1
            name = f"{base}_{n}"
        taken.add(name.casefold())
        names.append(name)
    return names
```

Create `organize/__init__.py` as an empty file.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_organize_names.py -v` — expected: 6 passed.

- [ ] **Step 5: Add the package to the wheel**

In `pyproject.toml`, add `"organize"` to `[tool.hatch.build.targets.wheel] only-include` (keep the list alphabetical: `… "models", "modelsvc", "organize", "search", …`).

- [ ] **Step 6: Lint and stop for review**

Run: `uv run ruff check . && uv run pytest -q`
Then stop. The user commits.

---

### Task 2: A memory knows whether it is an event or a theme

Design §11 gives a photo one **event** and several **themes**; §12 makes the event own the file on disk. The composer only builds events today, so `kind` defaults to `"event"` and nothing changes visibly — but the plan builder in Task 3 needs the field to exist.

**Files:**
- Modify: `albums/memory_store.py` (`Memory`, `replace_memories`, `read_memories`)
- Modify: `albums/memories.py` (`MemoriesOrganizer.organize`)
- Test: `tests/test_memory_store.py` (add), `tests/test_albums.py` (add)

**Interfaces:**
- Consumes: nothing.
- Produces: `Memory(title, description, photo_ids, signature, kind="event")`; a memory `Album` carries `meta["memory_kind"]` of `"event"` or `"theme"`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_memory_store.py  (append)
from albums.memory_store import Memory, read_memories, replace_memories


def test_memory_kind_defaults_to_event_and_round_trips(conn):
    replace_memories(conn, 1, [
        Memory("A day at Borjomi", "…", [1, 2], "sig"),
        Memory("Waterfalls", "…", [1], "sig", kind="theme"),
    ])
    kinds = {m.title: m.kind for m in read_memories(conn, 1)}
    assert kinds == {"A day at Borjomi": "event", "Waterfalls": "theme"}
```

```python
# tests/test_albums.py  (append)
from albums.memories import MemoriesOrganizer
from albums.memory_store import Memory, replace_memories
from tests.factories import add_photo


def test_memory_albums_carry_their_kind(conn):
    add_photo(conn, photo_id=1, content_hash="h1", thumb_key="1.jpg")
    replace_memories(conn, 1, [Memory("Waterfalls", "…", [1], "sig", kind="theme")])
    album = MemoriesOrganizer().organize(conn, 1)[0]
    assert album.meta["memory_kind"] == "theme"
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest tests/test_memory_store.py tests/test_albums.py -v`
Expected: FAIL — `Memory.__init__() got an unexpected keyword argument 'kind'`

- [ ] **Step 3: Implement**

In `albums/memory_store.py`:

```python
@dataclass(frozen=True)
class Memory:
    title: str
    description: str
    photo_ids: list[int]   # cover first
    signature: str
    # event = a time/place-bounded happening, theme = a thread across time (§11).
    # Stage 2 gives the file to the event and makes a theme a description-only
    # folder (§12), so the kind has to survive the round-trip through `params`.
    kind: str = "event"
```

In `replace_memories`, change the `params` payload:

```python
                (owner_id, memory.title, memory.description,
                 json.dumps({"signature": memory.signature, "kind": memory.kind})),
```

In `read_memories`, read it back:

```python
        params = json.loads(row["params"] or "{}")
        memories.append(Memory(row["name"], row["description"], photo_ids,
                               params.get("signature", ""),
                               params.get("kind", "event")))
```

(Delete the now-unused `signature = json.loads(...)` line it replaces.)

In `albums/memories.py`, widen the album meta:

```python
                    meta={"kind": "memory", "memory_kind": memory.kind},
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_memory_store.py tests/test_albums.py tests/test_memories_build.py tests/test_web_memories.py -v` — expected: all pass.

- [ ] **Step 5: Lint, full suite, stop for review**

Run: `uv run ruff check . && uv run pytest -q`

---

### Task 3: The plan builder

The heart of the feature. Everything about *what goes where* is decided here and nowhere else.

**Files:**
- Create: `organize/plan.py`
- Test: `tests/test_organize_plan.py`

**Interfaces:**
- Consumes: `organize.names.unique_folder_names`; `albums.base.Album`; an `Organizer` (`albums/registry.py`).
- Produces:
  `build_plan(conn: sqlite3.Connection, owner_id: int, organizer, grain: str | None = None, *, generated_at: str) -> dict` — the plan document.
  `PLAN_VERSION: int = 1`.

The rules, exactly:

1. Albums come in organizer order, except memories whose `meta["memory_kind"] == "theme"` sort **last** — so an event claims a photo before a theme can (§12).
2. A photo's **owner** is the first album in that order that lists it. Its `target` is `<folder>/<basename of its first source path>`.
3. A later album listing an already-owned photo records the owner's target in its `references` list — that becomes `photos.txt`.
4. Sources are ordered shallowest-first, then lexicographically; the first is the copy that is kept.
5. A photo no album claims goes to `_unsorted/`.
6. A photo with no source path is skipped entirely (nothing on disk to move).
7. `complete` is false while any job row for this owner is `pending` or `running` (§8).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_organize_plan.py
import pytest

from albums.by_date import ByDateOrganizer
from albums.memories import MemoriesOrganizer
from albums.memory_store import Memory, replace_memories
from organize.plan import PLAN_VERSION, build_plan
from tests.factories import add_photo

WHEN = "2026-08-19T09:12:44Z"


def plan_for(conn, organizer, grain=None):
    return build_plan(conn, 1, organizer, grain, generated_at=WHEN)


def targets(plan):
    return {f["hash"]: f["target"] for f in plan["files"]}


@pytest.fixture
def dated(conn):
    add_photo(conn, photo_id=1, content_hash="aaaa1111", thumb_key="1.jpg",
              shot_at="2024-06-14T18:30:12", bytes=100,
              sources=["iphone dump/IMG_4471.jpg"])
    add_photo(conn, photo_id=2, content_hash="bbbb2222", thumb_key="2.jpg",
              shot_at="2024-07-02T10:00:00", bytes=200,
              sources=["IMG_9000.jpg"])
    return conn


def test_a_date_month_plan_puts_each_photo_in_its_month_folder(dated):
    plan = plan_for(dated, ByDateOrganizer(), "month")
    assert plan["plan_version"] == PLAN_VERSION
    assert plan["organizer"] == "date"
    assert plan["grain"] == "month"
    assert plan["generated_at"] == WHEN
    assert targets(plan) == {
        "aaaa1111": "June 2024/IMG_4471.jpg",
        "bbbb2222": "July 2024/IMG_9000.jpg",
    }


def test_filenames_are_never_changed(dated):
    plan = plan_for(dated, ByDateOrganizer(), "month")
    assert all(f["target"].endswith(f["sources"][0].rsplit("/", 1)[-1])
               for f in plan["files"])


def test_every_album_carries_its_description_for_description_txt(dated):
    plan = plan_for(dated, ByDateOrganizer(), "month")
    june = next(a for a in plan["albums"] if a["folder"] == "June 2024")
    assert june["title"] == "June 2024"
    assert june["description"].startswith("1 photo")
    assert june["photo_count"] == 1
    assert june["kind"] == "album"


def test_sources_are_shallowest_then_alphabetical_and_the_first_one_is_kept(conn):
    add_photo(conn, photo_id=1, content_hash="aaaa1111", thumb_key="1.jpg",
              shot_at="2024-06-14T18:30:12", bytes=100,
              sources=["z/deep/IMG.jpg", "b.jpg", "a/IMG.jpg"])
    plan = plan_for(conn, ByDateOrganizer(), "month")
    entry = plan["files"][0]
    assert entry["sources"] == ["b.jpg", "a/IMG.jpg", "z/deep/IMG.jpg"]
    assert entry["target"] == "June 2024/b.jpg"


def test_two_photos_with_the_same_filename_get_a_hash_suffix(conn):
    add_photo(conn, photo_id=1, content_hash="aaaa1111", thumb_key="1.jpg",
              shot_at="2024-06-01T09:00:00", bytes=1, sources=["a/IMG.jpg"])
    add_photo(conn, photo_id=2, content_hash="bbbb2222", thumb_key="2.jpg",
              shot_at="2024-06-02T09:00:00", bytes=1, sources=["b/IMG.jpg"])
    assert set(targets(plan_for(conn, ByDateOrganizer(), "month")).values()) == {
        "June 2024/IMG.jpg", "June 2024/IMG_bbbb2222.jpg",
    }


def test_a_photo_no_album_claims_goes_to_unsorted(conn):
    add_photo(conn, photo_id=1, content_hash="aaaa1111", thumb_key="1.jpg",
              bytes=1, sources=["scan012.jpg"])          # no shot_at
    plan = plan_for(conn, ByDateOrganizer(), "month")
    assert targets(plan) == {"aaaa1111": "_unsorted/scan012.jpg"}
    assert [a["folder"] for a in plan["albums"]] == ["_unsorted"]


def test_a_photo_with_no_source_path_is_left_out_entirely(conn):
    add_photo(conn, photo_id=1, content_hash="aaaa1111", thumb_key="1.jpg",
              shot_at="2024-06-14T18:30:12", bytes=1)     # no sources
    assert plan_for(conn, ByDateOrganizer(), "month")["files"] == []


def test_the_event_owns_the_file_and_the_theme_only_references_it(conn):
    add_photo(conn, photo_id=1, content_hash="aaaa1111", thumb_key="1.jpg",
              bytes=1, sources=["IMG_4471.jpg"])
    replace_memories(conn, 1, [
        Memory("Waterfalls", "A thread across time.", [1], "sig", kind="theme"),
        Memory("A day at Borjomi", "One long day.", [1], "sig"),
    ])
    plan = plan_for(conn, MemoriesOrganizer())
    assert targets(plan) == {"aaaa1111": "A day at Borjomi/IMG_4471.jpg"}
    theme = next(a for a in plan["albums"] if a["folder"] == "Waterfalls")
    assert theme["kind"] == "theme"
    assert theme["references"] == ["A day at Borjomi/IMG_4471.jpg"]


def test_the_plan_is_incomplete_while_a_job_is_pending(conn):
    add_photo(conn, photo_id=1, content_hash="aaaa1111", thumb_key="1.jpg",
              shot_at="2024-06-14T18:30:12", bytes=1, sources=["IMG.jpg"])
    assert plan_for(conn, ByDateOrganizer(), "month")["complete"] is True
    conn.execute(
        "INSERT INTO jobs(photo_id, stage, status, updated_at)"
        " VALUES (1, 'caption', 'pending', '2026-01-01T00:00:00')"
    )
    assert plan_for(conn, ByDateOrganizer(), "month")["complete"] is False


def test_photo_count_matches_the_files_listed(dated):
    plan = plan_for(dated, ByDateOrganizer(), "month")
    assert plan["photo_count"] == len(plan["files"]) == 2
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest tests/test_organize_plan.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'organize.plan'`

- [ ] **Step 3: Implement**

```python
# organize/plan.py
"""Albums -> the stage-2 plan document (design §12, docs/sync-cli.md).

The organizer IS the layout: whatever `/organize` is showing is exactly the tree
the CLI will build inside the user's own folder. Nothing here touches a
filesystem — the plan is data, and `ivms777_sync` is the only writer (§3.2b).
"""

import sqlite3
from typing import Any

from albums.base import Album
from organize.names import unique_folder_names

PLAN_VERSION = 1
UNSORTED = "_unsorted"


def build_plan(
    conn: sqlite3.Connection,
    owner_id: int,
    organizer: Any,
    grain: str | None = None,
    *,
    generated_at: str,
) -> dict:
    albums = organizer.organize(conn, owner_id, grain)
    ordered = [album for _, album in sorted(
        enumerate(albums), key=lambda pair: (_is_theme(pair[1]), pair[0])
    )]
    folders = unique_folder_names([(a.title, a.key) for a in ordered])

    sources = _sources(conn, owner_id)
    facts = _facts(conn, owner_id)

    owner_target: dict[int, str] = {}
    taken: dict[str, set[str]] = {}
    files: list[dict] = []
    entries: list[dict] = []

    for album, folder in zip(ordered, folders):
        references: list[str] = []
        for photo_id in album.photo_ids:
            target = _claim(photo_id, folder, owner_target, taken, sources, facts, files)
            if target is None:
                references.append(owner_target[photo_id]) if photo_id in owner_target else None
        entries.append(_entry(folder, album.title, album.description, album.size,
                              "theme" if _is_theme(album) else "album", references))

    unclaimed = [pid for pid in sorted(facts) if pid not in owner_target]
    leftovers = [pid for pid in unclaimed
                 if _claim(pid, UNSORTED, owner_target, taken, sources, facts, files)]
    if leftovers:
        entries.append(_entry(UNSORTED, "Unsorted",
                              "Photos this organization could not place.",
                              len(leftovers), "album", []))

    files.sort(key=lambda f: f["target"])
    return {
        "plan_version": PLAN_VERSION,
        "generated_at": generated_at,
        "organizer": organizer.name,
        "grain": grain,
        "title": organizer.label + (f" — {grain}" if grain else ""),
        "complete": _complete(conn, owner_id),
        "photo_count": len(files),
        "albums": entries,
        "files": files,
    }


def _entry(folder, title, description, photo_count, kind, references) -> dict:
    entry = {"folder": folder, "title": title, "description": description,
             "photo_count": photo_count, "kind": kind}
    if references:
        entry["references"] = references
    return entry


def _claim(photo_id, folder, owner_target, taken, sources, facts, files) -> str | None:
    """Give this photo's file to `folder`, or None when it is already owned or
    has nothing on disk to move."""
    if photo_id in owner_target:
        return None
    fact = facts.get(photo_id)
    paths = sources.get(photo_id)
    if fact is None or not paths:
        return None
    target = _unique_target(folder, paths[0], fact["hash"], taken)
    owner_target[photo_id] = target
    files.append({"hash": fact["hash"], "target": target,
                  "sources": paths, "bytes": fact["bytes"]})
    return target


def _unique_target(folder: str, rel_path: str, content_hash: str,
                   taken: dict[str, set[str]]) -> str:
    name = rel_path.rsplit("/", 1)[-1]
    used = taken.setdefault(folder, set())
    if name.casefold() in used:
        stem, dot, ext = name.rpartition(".")
        name = f"{stem}_{content_hash[:8]}{dot}{ext}" if dot else f"{name}_{content_hash[:8]}"
    used.add(name.casefold())
    return f"{folder}/{name}"


def _is_theme(album: Album) -> bool:
    return album.meta.get("memory_kind") == "theme"


def _sources(conn: sqlite3.Connection, owner_id: int) -> dict[int, list[str]]:
    """photo id -> every path it arrived from, shallowest then alphabetical. The
    first entry is the copy the CLI keeps; the rest become `_duplicates/`."""
    out: dict[int, list[str]] = {}
    rows = conn.execute(
        "SELECT s.photo_id AS pid, s.rel_path AS path FROM photo_sources s"
        " JOIN photos p ON p.id = s.photo_id WHERE p.owner_id = ?",
        (owner_id,),
    )
    for row in rows:
        out.setdefault(row["pid"], []).append(row["path"])
    for paths in out.values():
        paths.sort(key=lambda path: (path.count("/"), path))
    return out


def _facts(conn: sqlite3.Connection, owner_id: int) -> dict[int, dict]:
    rows = conn.execute(
        "SELECT id, content_hash, bytes FROM photos WHERE owner_id = ?", (owner_id,)
    )
    return {r["id"]: {"hash": r["content_hash"], "bytes": r["bytes"] or 0} for r in rows}


def _complete(conn: sqlite3.Connection, owner_id: int) -> bool:
    """The plan gate (§8): stage 2 must not run against a half-processed library."""
    row = conn.execute(
        "SELECT count(*) AS n FROM jobs j JOIN photos p ON p.id = j.photo_id"
        " WHERE p.owner_id = ? AND j.status IN ('pending', 'running')",
        (owner_id,),
    ).fetchone()
    return row["n"] == 0
```

Note on the `references` line in the album loop — write it plainly rather than as an expression statement:

```python
        for photo_id in album.photo_ids:
            target = _claim(photo_id, folder, owner_target, taken, sources, facts, files)
            if target is None and photo_id in owner_target:
                references.append(owner_target[photo_id])
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_organize_plan.py -v` — expected: 10 passed.

- [ ] **Step 5: Lint, full suite, stop for review**

Run: `uv run ruff check . && uv run pytest -q`

---

### Task 4: `/organize/plan` preview and `/api/plan` download

**Files:**
- Modify: `web/app.py` (add two routes near the existing `/organize` route, ~line 604)
- Create: `web/templates/organize_plan.html`
- Modify: `web/templates/organize.html` (the button)
- Test: `tests/test_web_organize_plan.py`

**Interfaces:**
- Consumes: `organize.plan.build_plan`, `albums.registry.get_organizer`.
- Produces: `GET /api/plan?by=&grain=` → the plan JSON as an attachment named `ivms777-plan.json`; `GET /organize/plan?by=&grain=` → the HTML preview.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_web_organize_plan.py
import pytest
from fastapi.testclient import TestClient

from tests.factories import add_photo
from web.app import create_app


@pytest.fixture
def client(settings):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    app = create_app(settings)
    conn = app.state.context.conn
    add_photo(conn, photo_id=1, content_hash="aaaa1111", thumb_key="1.jpg",
              shot_at="2024-06-14T18:30:12", bytes=10, sources=["IMG_4471.jpg"])
    with TestClient(app) as test_client:
        yield test_client


def test_organize_offers_the_button_carrying_the_current_principle(client):
    body = client.get("/organize?by=date&grain=year").text
    assert "/organize/plan?by=date&amp;grain=year" in body
    assert "Organize files" in body


def test_the_preview_lists_the_folders_and_asks_before_downloading(client):
    body = client.get("/organize/plan?by=date&grain=month").text
    assert "June 2024" in body
    assert "reorganize" in body.lower()
    assert "/api/plan?by=date&amp;grain=month" in body


def test_the_plan_downloads_as_a_named_attachment(client):
    response = client.get("/api/plan?by=date&grain=month")
    assert response.status_code == 200
    assert 'filename="ivms777-plan.json"' in response.headers["content-disposition"]
    plan = response.json()
    assert plan["files"][0]["target"] == "June 2024/IMG_4471.jpg"


def test_the_preview_warns_while_the_library_is_still_processing(client):
    conn = client.app.state.context.conn
    conn.execute(
        "INSERT INTO jobs(photo_id, stage, status, updated_at)"
        " VALUES (1, 'caption', 'pending', '2026-01-01T00:00:00')"
    )
    assert "still processing" in client.get("/organize/plan?by=date").text.lower()
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest tests/test_web_organize_plan.py -v`
Expected: FAIL — 404 on `/organize/plan`.

- [ ] **Step 3: Add the routes**

In `web/app.py`, import the builder near the other album imports:

```python
from organize.plan import build_plan
```

and add, directly after the `memories_progress` route:

```python
    def _plan(by: str | None, grain: str | None) -> dict:
        ctx = context()
        organizer = get_organizer(by)
        return build_plan(ctx.conn, ctx.settings.owner_id, organizer, grain,
                          generated_at=_utc_now())

    @app.get("/organize/plan", response_class=HTMLResponse)
    def organize_plan(
        request: Request, by: str | None = None, grain: str | None = None
    ) -> HTMLResponse:
        # The confirm screen for stage 2 (§12): it shows the tree the current
        # principle would create and asks before handing over the plan document.
        # Nothing here touches the user's disk — only ivms777-sync ever does.
        plan = _plan(by, grain)
        query = f"by={plan['organizer']}" + (f"&grain={grain}" if grain else "")
        return templates.TemplateResponse(
            request,
            "organize_plan.html",
            {"plan": plan, "query": query, "back": f"/organize?{query}"},
        )

    @app.get("/api/plan")
    def api_plan(by: str | None = None, grain: str | None = None) -> JSONResponse:
        return JSONResponse(
            _plan(by, grain),
            headers={"Content-Disposition": 'attachment; filename="ivms777-plan.json"'},
        )
```

`web/app.py` has no datetime helper yet, so add one beside the other module-level helpers, with `from datetime import UTC, datetime` at the top of the file:

```python
def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
```

`JSONResponse` is already imported from `fastapi.responses` (`web/app.py:14`).

- [ ] **Step 4: Add the preview template**

```html
{# web/templates/organize_plan.html #}
{% extends "base.html" %}
{% block content %}
<div class="organize-head">
  <h1>Organize your folders</h1>
  <p class="muted">{{ plan.title }} · {{ plan.albums|length }} folders ·
    {{ plan.photo_count }} photos</p>
</div>

{% if not plan.complete %}
  <p class="warn">Your library is still processing. A plan made now can miss
    photos — wait for Upload to finish, or pass <code>--allow-incomplete</code>.</p>
{% endif %}

<p>Reorganize your folders this way? Your source folder gets one folder per album
  below, each file moves into the folder its album chose, and filenames stay
  exactly as they are. Nothing is deleted and nothing happens until you run
  <code>ivms777-sync</code> yourself.</p>

<ul class="plan-tree">
  {% for album in plan.albums %}
    <li>
      <code>{{ album.folder }}/</code>
      {% if album.kind == "theme" %}
        <span class="muted">description only — {{ album.references|length }}
          photos listed in <code>photos.txt</code></span>
      {% else %}
        <span class="muted">{{ album.photo_count }} photos</span>
      {% endif %}
    </li>
  {% endfor %}
</ul>

<div class="plan-actions">
  <a class="button" href="/api/plan?{{ query }}">Download the plan</a>
  <a href="{{ back }}" onclick="history.back(); return false;">Cancel</a>
</div>

<p class="muted">Then, in the folder you want reorganized:</p>
<pre>ivms777-sync apply ivms777-plan.json</pre>
{% endblock %}
```

Cancel uses `history.back()` so `/organize` returns with its state and scroll intact (design §13.1, "never lose the user's place"), with the computed URL as the deep-link fallback.

- [ ] **Step 5: Add the button to `organize.html`**

Inside `<div class="organize-head">`, after the `</form>`:

```html
  {% if albums %}
    <a class="button organize-export"
       href="/organize/plan?by={{ active }}{% if active_grain %}&grain={{ active_grain }}{% endif %}"
    >Organize files…</a>
  {% endif %}
```

- [ ] **Step 6: Run the tests**

Run: `uv run pytest tests/test_web_organize_plan.py tests/test_web_organize.py -v` — expected: all pass.

- [ ] **Step 7: Lint, full suite, stop for review**

Run: `uv run ruff check . && uv run pytest -q`

---

### Task 5: The CLI package — reading and validating a plan

**Files:**
- Create: `ivms777_sync/__init__.py`, `ivms777_sync/plan.py`, `ivms777_sync/cli.py`
- Modify: `pyproject.toml` (wheel include + console script)
- Test: `tests/test_sync_plan.py`, `tests/test_sync_isolation.py`

**Interfaces:**
- Consumes: nothing from this repo.
- Produces:
  `PLAN_VERSION: int`, `PlanError(Exception)`,
  `PlanFile(hash: str, target: str, sources: list[str], bytes: int)`,
  `PlanAlbum(folder: str, title: str, description: str, kind: str, references: list[str])`,
  `Plan(organizer: str, title: str, generated_at: str, complete: bool, albums: list[PlanAlbum], files: list[PlanFile])`,
  `load_plan(path: Path) -> Plan`,
  `main(argv: list[str] | None = None) -> int` in `cli.py`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_sync_plan.py
import json

import pytest

from ivms777_sync.plan import PlanError, load_plan

GOOD = {
    "plan_version": 1,
    "generated_at": "2026-08-19T09:12:44Z",
    "organizer": "date",
    "grain": "month",
    "title": "By date — month",
    "complete": True,
    "photo_count": 1,
    "albums": [{"folder": "June 2024", "title": "June 2024",
                "description": "1 photo.", "photo_count": 1, "kind": "album"}],
    "files": [{"hash": "aaaa1111", "target": "June 2024/IMG.jpg",
               "sources": ["IMG.jpg"], "bytes": 10}],
}


def write(tmp_path, payload):
    path = tmp_path / "ivms777-plan.json"
    path.write_text(json.dumps(payload))
    return path


def test_a_good_plan_loads(tmp_path):
    plan = load_plan(write(tmp_path, GOOD))
    assert plan.organizer == "date"
    assert plan.files[0].target == "June 2024/IMG.jpg"
    assert plan.albums[0].references == []


def test_a_future_plan_version_is_refused(tmp_path):
    with pytest.raises(PlanError, match="version"):
        load_plan(write(tmp_path, {**GOOD, "plan_version": 99}))


def test_a_target_that_escapes_the_root_is_refused(tmp_path):
    bad = {**GOOD, "files": [{**GOOD["files"][0], "target": "../../etc/passwd"}]}
    with pytest.raises(PlanError, match="outside"):
        load_plan(write(tmp_path, bad))


def test_an_absolute_target_is_refused(tmp_path):
    bad = {**GOOD, "files": [{**GOOD["files"][0], "target": "/etc/passwd"}]}
    with pytest.raises(PlanError, match="outside"):
        load_plan(write(tmp_path, bad))


def test_a_folder_that_escapes_the_root_is_refused(tmp_path):
    bad = {**GOOD, "albums": [{**GOOD["albums"][0], "folder": "../evil"}]}
    with pytest.raises(PlanError, match="outside"):
        load_plan(write(tmp_path, bad))


def test_malformed_json_is_a_plan_error_not_a_traceback(tmp_path):
    path = tmp_path / "ivms777-plan.json"
    path.write_text("{not json")
    with pytest.raises(PlanError):
        load_plan(path)
```

```python
# tests/test_sync_isolation.py
"""ivms777_sync must install on a user's machine with no ivms777 anywhere near it
(design §14). Nothing under it may import this repo's packages or a heavy library."""

import ast
import pathlib

FORBIDDEN = {
    "albums", "captioning", "chat", "config", "db", "embedding", "inference",
    "ingest", "models", "modelsvc", "organize", "search", "storage", "web",
    "torch", "transformers", "PIL", "fastapi", "numpy",
}


def imported_roots(path: pathlib.Path) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_the_sync_cli_imports_nothing_from_the_app():
    root = pathlib.Path(__file__).resolve().parent.parent / "ivms777_sync"
    for module in sorted(root.glob("*.py")):
        assert not (imported_roots(module) & FORBIDDEN), module.name
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest tests/test_sync_plan.py tests/test_sync_isolation.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ivms777_sync'`

- [ ] **Step 3: Implement the plan reader**

```python
# ivms777_sync/plan.py
"""Read and validate a downloaded plan document (docs/sync-cli.md).

This runs on the user's machine against a file they moved there by hand, so it
trusts nothing: a target that escapes the root, an absolute path, or an unknown
schema version stops the run before anything is touched.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

PLAN_VERSION = 1


class PlanError(Exception):
    """The plan document cannot be trusted. Nothing has been changed."""


@dataclass(frozen=True)
class PlanFile:
    hash: str
    target: str
    sources: list[str]
    bytes: int


@dataclass(frozen=True)
class PlanAlbum:
    folder: str
    title: str
    description: str
    kind: str = "album"
    references: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Plan:
    organizer: str
    title: str
    generated_at: str
    complete: bool
    albums: list[PlanAlbum]
    files: list[PlanFile]


def load_plan(path: Path) -> Plan:
    try:
        raw = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise PlanError(f"cannot read {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise PlanError("the plan is not a JSON object")
    if raw.get("plan_version") != PLAN_VERSION:
        raise PlanError(
            f"plan version {raw.get('plan_version')} is not supported "
            f"(this tool speaks {PLAN_VERSION}) — update ivms777-sync"
        )
    albums = [
        PlanAlbum(
            folder=_relative(album["folder"]),
            title=album.get("title", album["folder"]),
            description=album.get("description", ""),
            kind=album.get("kind", "album"),
            references=[_relative(r) for r in album.get("references", [])],
        )
        for album in raw.get("albums", [])
    ]
    files = [
        PlanFile(
            hash=str(entry["hash"]),
            target=_relative(entry["target"]),
            sources=list(entry.get("sources", [])),
            bytes=int(entry.get("bytes", 0)),
        )
        for entry in raw.get("files", [])
    ]
    return Plan(
        organizer=raw.get("organizer", "?"),
        title=raw.get("title", raw.get("organizer", "?")),
        generated_at=raw.get("generated_at", ""),
        complete=bool(raw.get("complete", False)),
        albums=albums,
        files=files,
    )


def _relative(value: str) -> str:
    """Every path in a plan is relative to the root, and stays under it."""
    pure = PurePosixPath(str(value))
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise PlanError(f"path outside the folder being organized: {value!r}")
    return str(pure)
```

- [ ] **Step 4: Add a minimal `cli.py` so the entry point exists**

```python
# ivms777_sync/cli.py
"""ivms777-sync — the only component that writes to your disk (design §3.2b)."""

import argparse
import sys

from ivms777_sync.plan import PlanError, load_plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ivms777-sync")
    sub = parser.add_subparsers(dest="command", required=True)
    apply_cmd = sub.add_parser("apply", help="reorganize this folder from a plan")
    apply_cmd.add_argument("plan")
    apply_cmd.add_argument("--root", default=".")
    apply_cmd.add_argument("--allow-incomplete", action="store_true")
    apply_cmd.add_argument("--yes", action="store_true")

    args = parser.parse_args(argv)
    try:
        plan = load_plan(args.plan)
    except PlanError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"plan: {plan.title}, generated {plan.generated_at}, "
          f"{len(plan.files)} photos")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Create `ivms777_sync/__init__.py` as an empty file.

- [ ] **Step 5: Wire up packaging**

In `pyproject.toml`, add the console script and ship the package:

```toml
[project.scripts]
ivms777-sync = "ivms777_sync.cli:main"
```

and add `"ivms777_sync"` to `[tool.hatch.build.targets.wheel] only-include`.

- [ ] **Step 6: Run the tests**

Run: `uv run pytest tests/test_sync_plan.py tests/test_sync_isolation.py -v` — expected: 7 passed.

- [ ] **Step 7: Lint, full suite, stop for review**

Run: `uv run ruff check . && uv run pytest -q`

---

### Task 6: Scan the folder and match by hash

**Files:**
- Create: `ivms777_sync/scan.py`, `ivms777_sync/match.py`
- Test: `tests/test_sync_match.py`

**Interfaces:**
- Consumes: `ivms777_sync.plan.Plan`, `PlanFile`.
- Produces:
  `file_hash(path: Path) -> str`; `scan(root: Path) -> dict[str, list[Path]]` (hash → paths relative to root, sorted);
  `Move(src: str, dest: str, kind: str)` where `kind` is `"move"` | `"duplicate"`;
  `Outcome(moves: list[Move], in_place: int, missing: int, unknown: int, conflicts: list[str])`;
  `match(plan: Plan, found: dict[str, list[Path]]) -> Outcome`.

Rules: the first source path in the plan that exists on disk keeps the file; every *other* copy of the same content goes to `_duplicates/<its original path>`. A file already at its target counts as `in_place`. A target occupied by a file the plan does not know is a conflict, and that photo is skipped.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_sync_match.py
from pathlib import Path

from ivms777_sync.match import match
from ivms777_sync.plan import Plan, PlanAlbum, PlanFile
from ivms777_sync.scan import file_hash, scan


def make_plan(*files: PlanFile) -> Plan:
    return Plan(organizer="date", title="By date — month", generated_at="t",
                complete=True,
                albums=[PlanAlbum("June 2024", "June 2024", "1 photo.")],
                files=list(files))


def test_scan_hashes_every_file_and_ignores_the_journal_dir(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"one")
    (tmp_path / ".ivms777-sync").mkdir()
    (tmp_path / ".ivms777-sync" / "j.jsonl").write_bytes(b"one")
    found = scan(tmp_path)
    assert list(found) == [file_hash(tmp_path / "a.jpg")]
    assert found[file_hash(tmp_path / "a.jpg")] == [Path("a.jpg")]


def test_a_matched_file_is_moved_to_its_target(tmp_path):
    (tmp_path / "IMG.jpg").write_bytes(b"one")
    digest = file_hash(tmp_path / "IMG.jpg")
    plan = make_plan(PlanFile(digest, "June 2024/IMG.jpg", ["IMG.jpg"], 3))
    outcome = match(plan, scan(tmp_path))
    assert [(m.src, m.dest, m.kind) for m in outcome.moves] == [
        ("IMG.jpg", "June 2024/IMG.jpg", "move")
    ]


def test_a_file_already_at_its_target_is_left_alone(tmp_path):
    (tmp_path / "June 2024").mkdir()
    (tmp_path / "June 2024" / "IMG.jpg").write_bytes(b"one")
    digest = file_hash(tmp_path / "June 2024" / "IMG.jpg")
    plan = make_plan(PlanFile(digest, "June 2024/IMG.jpg", ["IMG.jpg"], 3))
    outcome = match(plan, scan(tmp_path))
    assert outcome.moves == []
    assert outcome.in_place == 1


def test_extra_copies_of_the_same_photo_go_to_duplicates(tmp_path):
    (tmp_path / "IMG.jpg").write_bytes(b"one")
    (tmp_path / "copies").mkdir()
    (tmp_path / "copies" / "IMG copy.jpg").write_bytes(b"one")
    digest = file_hash(tmp_path / "IMG.jpg")
    plan = make_plan(PlanFile(digest, "June 2024/IMG.jpg", ["IMG.jpg"], 3))
    kinds = {m.dest: m.kind for m in match(plan, scan(tmp_path)).moves}
    assert kinds == {
        "June 2024/IMG.jpg": "move",
        "_duplicates/copies/IMG copy.jpg": "duplicate",
    }


def test_a_photo_in_the_plan_but_not_on_disk_is_counted_not_moved(tmp_path):
    plan = make_plan(PlanFile("nothere", "June 2024/IMG.jpg", ["IMG.jpg"], 3))
    outcome = match(plan, scan(tmp_path))
    assert (outcome.moves, outcome.missing) == ([], 1)


def test_a_file_the_plan_does_not_know_is_counted_and_untouched(tmp_path):
    (tmp_path / "notes.txt").write_bytes(b"hello")
    outcome = match(make_plan(), scan(tmp_path))
    assert (outcome.moves, outcome.unknown) == ([], 1)


def test_a_target_occupied_by_a_stranger_is_a_conflict(tmp_path):
    (tmp_path / "IMG.jpg").write_bytes(b"one")
    (tmp_path / "June 2024").mkdir()
    (tmp_path / "June 2024" / "IMG.jpg").write_bytes(b"stranger")
    digest = file_hash(tmp_path / "IMG.jpg")
    plan = make_plan(PlanFile(digest, "June 2024/IMG.jpg", ["IMG.jpg"], 3))
    outcome = match(plan, scan(tmp_path))
    assert outcome.moves == []
    assert outcome.conflicts == ["June 2024/IMG.jpg"]
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest tests/test_sync_match.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ivms777_sync.scan'`

- [ ] **Step 3: Implement the scanner**

```python
# ivms777_sync/scan.py
"""Walk a folder and hash every file in it.

Matching is by content hash, never by path — a folder reorganized since upload
still matches perfectly (docs/sync-cli.md).
"""

import hashlib
from pathlib import Path

JOURNAL_DIR = ".ivms777-sync"
CHUNK = 1024 * 1024
SKIP_NAMES = {JOURNAL_DIR, ".git", ".DS_Store", "ivms777-plan.json"}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def scan(root: Path) -> dict[str, list[Path]]:
    """content hash -> every path holding it, relative to `root`, sorted."""
    found: dict[str, list[Path]] = {}
    for path in sorted(Path(root).rglob("*")):
        rel = path.relative_to(root)
        if any(part in SKIP_NAMES for part in rel.parts) or not path.is_file():
            continue
        if path.is_symlink():
            continue
        found.setdefault(file_hash(path), []).append(rel)
    return found
```

- [ ] **Step 4: Implement the matcher**

```python
# ivms777_sync/match.py
"""Plan + what is on disk -> the exact list of moves.

Nothing here touches the filesystem. `apply` executes this list and nothing else.
"""

from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from ivms777_sync.plan import Plan

DUPLICATES = "_duplicates"


@dataclass(frozen=True)
class Move:
    src: str
    dest: str
    kind: str          # "move" | "duplicate"


@dataclass
class Outcome:
    moves: list[Move] = field(default_factory=list)
    in_place: int = 0
    missing: int = 0
    unknown: int = 0
    conflicts: list[str] = field(default_factory=list)


def match(plan: Plan, found: dict[str, list[Path]]) -> Outcome:
    outcome = Outcome()
    claimed: set[str] = set()
    targets = {entry.target for entry in plan.files}

    for entry in plan.files:
        here = [str(PurePosixPath(path.as_posix())) for path in found.get(entry.hash, [])]
        if not here:
            outcome.missing += 1
            continue
        for path in here:
            claimed.add(path)
        keep = _preferred(here, entry.sources)
        occupant = [path for path in here if path == entry.target]
        if not occupant and entry.target in _all_paths(found) and entry.target not in claimed:
            # Someone else's file is sitting on the target. Skip this photo rather
            # than overwrite anything — nothing is ever destroyed.
            outcome.conflicts.append(entry.target)
            continue
        for path in here:
            if path == entry.target:
                outcome.in_place += 1
            elif path == keep:
                outcome.moves.append(Move(path, entry.target, "move"))
            else:
                outcome.moves.append(Move(path, f"{DUPLICATES}/{path}", "duplicate"))

    outcome.unknown = sum(
        1 for paths in found.values() for path in paths
        if str(PurePosixPath(path.as_posix())) not in claimed
    )
    # A target that is also a known source is not a conflict: that file moves too.
    outcome.conflicts = [c for c in outcome.conflicts if c not in targets or c not in claimed]
    return outcome


def _all_paths(found: dict[str, list[Path]]) -> set[str]:
    return {str(PurePosixPath(path.as_posix())) for paths in found.values() for path in paths}


def _preferred(here: list[str], sources: list[str]) -> str:
    """The copy to keep: the plan's first source that is actually present,
    otherwise the shallowest path on disk. Deterministic either way."""
    for source in sources:
        if source in here:
            return source
    return sorted(here, key=lambda path: (path.count("/"), path))[0]
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_sync_match.py -v` — expected: 7 passed.
If `test_a_target_occupied_by_a_stranger_is_a_conflict` fails, the conflict filter is the place to look: a stranger's path is never in `claimed`, a plan file's own path always is.

- [ ] **Step 6: Lint, full suite, stop for review**

Run: `uv run ruff check . && uv run pytest -q`

---

### Task 7: Journalled apply

**Files:**
- Create: `ivms777_sync/journal.py`, `ivms777_sync/apply.py`
- Test: `tests/test_sync_apply.py`

**Interfaces:**
- Consumes: `Move`, `Outcome`, `Plan`.
- Produces:
  `Journal(path: Path)` with `record(op: dict) -> None`, `entries() -> list[dict]`, and classmethod `open_new(root: Path, *, stamp: str) -> Journal`;
  `apply_plan(root: Path, plan: Plan, outcome: Outcome, *, stamp: str) -> Path` (returns the journal path).

Every move routes through `.ivms777-sync/staging/<n>` — a two-step rename inside the same folder. That is what makes ordering and cycles impossible: nothing is ever renamed onto a path another file still occupies.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_sync_apply.py
import json

from ivms777_sync.apply import apply_plan
from ivms777_sync.match import match
from ivms777_sync.plan import Plan, PlanAlbum, PlanFile
from ivms777_sync.scan import file_hash, scan

STAMP = "20260819T091244Z"


def tree(root):
    return sorted(str(p.relative_to(root)) for p in root.rglob("*")
                  if p.is_file() and ".ivms777-sync" not in p.parts)


def run(root, plan):
    return apply_plan(root, plan, match(plan, scan(root)), stamp=STAMP)


def test_files_move_into_their_album_folders(tmp_path):
    (tmp_path / "IMG.jpg").write_bytes(b"one")
    plan = Plan("date", "By date — month", "t", True,
                [PlanAlbum("June 2024", "June 2024", "1 photo.")],
                [PlanFile(file_hash(tmp_path / "IMG.jpg"),
                          "June 2024/IMG.jpg", ["IMG.jpg"], 3)])
    run(tmp_path, plan)
    assert "June 2024/IMG.jpg" in tree(tmp_path)
    assert (tmp_path / "June 2024" / "IMG.jpg").read_bytes() == b"one"


def test_two_files_swapping_places_do_not_overwrite_each_other(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"AAA")
    (tmp_path / "b.jpg").write_bytes(b"BBB")
    plan = Plan("date", "t", "t", True, [PlanAlbum("June 2024", "June", "")], [
        PlanFile(file_hash(tmp_path / "a.jpg"), "b.jpg", ["a.jpg"], 3),
        PlanFile(file_hash(tmp_path / "b.jpg"), "a.jpg", ["b.jpg"], 3),
    ])
    run(tmp_path, plan)
    assert (tmp_path / "a.jpg").read_bytes() == b"BBB"
    assert (tmp_path / "b.jpg").read_bytes() == b"AAA"


def test_every_album_folder_gets_a_description_file(tmp_path):
    (tmp_path / "IMG.jpg").write_bytes(b"one")
    plan = Plan("date", "t", "t", True,
                [PlanAlbum("June 2024", "June 2024", "1 photo, mostly Canon.")],
                [PlanFile(file_hash(tmp_path / "IMG.jpg"),
                          "June 2024/IMG.jpg", ["IMG.jpg"], 3)])
    run(tmp_path, plan)
    text = (tmp_path / "June 2024" / "description.txt").read_text()
    assert text.splitlines()[0] == "June 2024"
    assert "1 photo, mostly Canon." in text


def test_a_theme_folder_holds_only_a_description_and_a_photo_list(tmp_path):
    (tmp_path / "IMG.jpg").write_bytes(b"one")
    digest = file_hash(tmp_path / "IMG.jpg")
    plan = Plan("memories", "Memories", "t", True, [
        PlanAlbum("A day at Borjomi", "A day at Borjomi", "One long day."),
        PlanAlbum("Waterfalls", "Waterfalls", "A thread.", "theme",
                  ["A day at Borjomi/IMG.jpg"]),
    ], [PlanFile(digest, "A day at Borjomi/IMG.jpg", ["IMG.jpg"], 3)])
    run(tmp_path, plan)
    assert sorted(p.name for p in (tmp_path / "Waterfalls").iterdir()) == [
        "description.txt", "photos.txt"]
    assert (tmp_path / "Waterfalls" / "photos.txt").read_text().strip() == \
        "A day at Borjomi/IMG.jpg"


def test_a_redundant_copy_is_moved_aside_never_deleted(tmp_path):
    (tmp_path / "IMG.jpg").write_bytes(b"one")
    (tmp_path / "copies").mkdir()
    (tmp_path / "copies" / "IMG copy.jpg").write_bytes(b"one")
    plan = Plan("date", "t", "t", True, [PlanAlbum("June 2024", "June", "")],
                [PlanFile(file_hash(tmp_path / "IMG.jpg"),
                          "June 2024/IMG.jpg", ["IMG.jpg"], 3)])
    run(tmp_path, plan)
    assert (tmp_path / "_duplicates" / "copies" / "IMG copy.jpg").read_bytes() == b"one"


def test_every_operation_is_journalled_before_it_runs(tmp_path):
    (tmp_path / "IMG.jpg").write_bytes(b"one")
    plan = Plan("date", "t", "t", True, [PlanAlbum("June 2024", "June", "")],
                [PlanFile(file_hash(tmp_path / "IMG.jpg"),
                          "June 2024/IMG.jpg", ["IMG.jpg"], 3)])
    journal = run(tmp_path, plan)
    records = [json.loads(line) for line in journal.read_text().splitlines()]
    assert records[0]["op"] == "move"
    assert records[0]["src"] == "IMG.jpg"
    assert records[0]["dest"] == "June 2024/IMG.jpg"
    assert records[-1]["status"] == "done"


def test_a_file_that_changed_since_the_scan_is_skipped(tmp_path):
    (tmp_path / "IMG.jpg").write_bytes(b"one")
    plan = Plan("date", "t", "t", True, [PlanAlbum("June 2024", "June", "")],
                [PlanFile(file_hash(tmp_path / "IMG.jpg"),
                          "June 2024/IMG.jpg", ["IMG.jpg"], 3)])
    outcome = match(plan, scan(tmp_path))
    (tmp_path / "IMG.jpg").write_bytes(b"CHANGED")     # edited after the scan
    apply_plan(tmp_path, plan, outcome, stamp=STAMP)
    assert (tmp_path / "IMG.jpg").read_bytes() == b"CHANGED"
    assert not (tmp_path / "June 2024" / "IMG.jpg").exists()
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest tests/test_sync_apply.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ivms777_sync.apply'`

- [ ] **Step 3: Implement the journal**

```python
# ivms777_sync/journal.py
"""Write down what is about to happen, then do it.

One JSONL record per operation, appended and flushed BEFORE the rename. A crash
mid-run therefore leaves a journal `undo` can replay (docs/sync-cli.md, Safety).
"""

import json
from pathlib import Path

JOURNAL_DIR = ".ivms777-sync"


class Journal:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    @classmethod
    def open_new(cls, root: Path, *, stamp: str) -> "Journal":
        directory = Path(root) / JOURNAL_DIR
        directory.mkdir(parents=True, exist_ok=True)
        return cls(directory / f"journal-{stamp}.jsonl")

    def record(self, op: dict) -> None:
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(op) + "\n")
            handle.flush()

    def entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line]
```

- [ ] **Step 4: Implement apply**

```python
# ivms777_sync/apply.py
"""Execute a matched plan. The only code in the project that writes to the
user's disk (design §3.2b).

Every move is a two-step rename through `.ivms777-sync/staging/` — out of the
way first, into place second. That is why ordering never matters and a cycle
(two photos swapping paths) is impossible to get wrong.
"""

import os
from pathlib import Path

from ivms777_sync.journal import JOURNAL_DIR, Journal
from ivms777_sync.match import Outcome
from ivms777_sync.plan import Plan
from ivms777_sync.scan import file_hash

STAGING = "staging"


def apply_plan(root: Path, plan: Plan, outcome: Outcome, *, stamp: str) -> Path:
    root = Path(root)
    journal = Journal.open_new(root, stamp=stamp)
    staging = root / JOURNAL_DIR / STAGING
    staging.mkdir(parents=True, exist_ok=True)

    staged: list[tuple[Path, str, str]] = []
    for index, move in enumerate(outcome.moves):
        source = root / move.src
        if not source.is_file():
            continue
        journal.record({"op": move.kind, "src": move.src, "dest": move.dest,
                        "status": "staging"})
        held = staging / f"{index:06d}"
        _rename(source, held)
        staged.append((held, move.dest, move.kind))

    for held, dest, kind in staged:
        target = root / dest
        target.parent.mkdir(parents=True, exist_ok=True)
        _rename(held, target)
        journal.record({"op": kind, "src": None, "dest": dest, "status": "done"})

    _write_texts(root, plan, journal)
    return journal.path


def _write_texts(root: Path, plan: Plan, journal: Journal) -> None:
    for album in plan.albums:
        folder = root / album.folder
        folder.mkdir(parents=True, exist_ok=True)
        body = f"{album.title}\n\n{album.description}\n"
        journal.record({"op": "write", "dest": f"{album.folder}/description.txt",
                        "status": "done"})
        (folder / "description.txt").write_text(body, encoding="utf-8")
        if album.references:
            journal.record({"op": "write", "dest": f"{album.folder}/photos.txt",
                            "status": "done"})
            (folder / "photos.txt").write_text(
                "\n".join(album.references) + "\n", encoding="utf-8"
            )


def _rename(source: Path, target: Path) -> None:
    """Same filesystem: an atomic rename. Across filesystems: copy, fsync, verify
    the hash, and only then unlink — the original goes after the copy is proven."""
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.rename(source, target)
        return
    except OSError:
        pass
    with open(source, "rb") as src, open(target, "wb") as dst:
        while chunk := src.read(1024 * 1024):
            dst.write(chunk)
        dst.flush()
        os.fsync(dst.fileno())
    if file_hash(target) != file_hash(source):
        target.unlink(missing_ok=True)
        raise OSError(f"copy of {source} did not verify; nothing was removed")
    source.unlink()
```

The "changed since the scan" test passes because `match` produced its move from the *old* hash; guard it explicitly by re-hashing before staging:

```python
        if move.kind in ("move", "duplicate") and _stale(root, move, plan):
            continue
```

with

```python
def _stale(root: Path, move, plan: Plan) -> bool:
    """Re-hash immediately before touching a file: anything edited since the scan
    is skipped rather than filed under a stale identity."""
    expected = {entry.hash for entry in plan.files if entry.target == move.dest
                or move.dest == f"_duplicates/{move.src}"}
    return bool(expected) and file_hash(root / move.src) not in expected
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_sync_apply.py -v` — expected: 7 passed.

- [ ] **Step 6: Lint, full suite, stop for review**

Run: `uv run ruff check . && uv run pytest -q`

---

### Task 8: `undo`

**Files:**
- Modify: `ivms777_sync/apply.py` (add `undo`)
- Test: `tests/test_sync_undo.py`

**Interfaces:**
- Consumes: `Journal`.
- Produces: `undo(root: Path, journal_path: Path) -> int` — the number of operations reversed.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sync_undo.py
from ivms777_sync.apply import apply_plan, undo
from ivms777_sync.match import match
from ivms777_sync.plan import Plan, PlanAlbum, PlanFile
from ivms777_sync.scan import file_hash, scan

STAMP = "20260819T091244Z"


def snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes()
            for p in sorted(root.rglob("*")) if p.is_file()}


def test_undo_returns_the_folder_to_exactly_how_it_started(tmp_path):
    (tmp_path / "iphone dump").mkdir()
    (tmp_path / "iphone dump" / "IMG.jpg").write_bytes(b"one")
    (tmp_path / "copies").mkdir()
    (tmp_path / "copies" / "IMG copy.jpg").write_bytes(b"one")
    before = snapshot(tmp_path)

    digest = file_hash(tmp_path / "iphone dump" / "IMG.jpg")
    plan = Plan("date", "t", "t", True,
                [PlanAlbum("June 2024", "June 2024", "1 photo.")],
                [PlanFile(digest, "June 2024/IMG.jpg",
                          ["iphone dump/IMG.jpg"], 3)])
    journal = apply_plan(tmp_path, plan, match(plan, scan(tmp_path)), stamp=STAMP)
    assert snapshot(tmp_path) != before

    undo(tmp_path, journal)
    assert snapshot(tmp_path) == before
```

- [ ] **Step 2: Run it and watch it fail**

Run: `uv run pytest tests/test_sync_undo.py -v`
Expected: FAIL — `ImportError: cannot import name 'undo'`

- [ ] **Step 3: Implement**

For `undo` to reverse a move it needs the original path, so record it on the *done* entry too. In `apply_plan`, change the staging loop to keep the source and the completion record to carry it:

```python
        staged.append((held, move.dest, move.kind, move.src))
```

```python
    for held, dest, kind, src in staged:
        target = root / dest
        target.parent.mkdir(parents=True, exist_ok=True)
        _rename(held, target)
        journal.record({"op": kind, "src": src, "dest": dest, "status": "done"})
```

Then add:

```python
def undo(root: Path, journal_path: Path) -> int:
    """Replay the journal backwards. Files go back where they were; folders and
    text files this run created are removed if they are empty again."""
    root = Path(root)
    entries = Journal(journal_path).entries()
    reversed_count = 0
    for entry in reversed(entries):
        if entry.get("status") != "done":
            continue
        if entry["op"] == "write":
            (root / entry["dest"]).unlink(missing_ok=True)
            reversed_count += 1
            continue
        source, dest = entry.get("src"), root / entry["dest"]
        if source is None or not dest.is_file():
            continue
        _rename(dest, root / source)
        reversed_count += 1
    _prune_empty(root)
    return reversed_count


def _prune_empty(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if path.is_dir() and JOURNAL_DIR not in path.parts and not any(path.iterdir()):
            path.rmdir()
```

The snapshot in the test includes the journal file itself, so have `snapshot` skip `.ivms777-sync` — adjust the test helper:

```python
def snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes()
            for p in sorted(root.rglob("*"))
            if p.is_file() and ".ivms777-sync" not in p.parts}
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_sync_undo.py tests/test_sync_apply.py -v` — expected: all pass.

- [ ] **Step 5: Lint, full suite, stop for review**

Run: `uv run ruff check . && uv run pytest -q`

---

### Task 9: The command line — `apply`, `undo`, `verify`

**Files:**
- Modify: `ivms777_sync/cli.py`
- Create: `ivms777_sync/client.py`
- Test: `tests/test_sync_cli.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `main(argv) -> int`; `summary(plan, outcome) -> str`; `fetch_plan(url: str, by: str | None, grain: str | None) -> dict` in `client.py`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_sync_cli.py
import json

from ivms777_sync.cli import main
from ivms777_sync.scan import file_hash

STAMP_ARGS = ["--stamp", "20260819T091244Z"]


def write_plan(tmp_path, *, complete=True):
    (tmp_path / "IMG.jpg").write_bytes(b"one")
    plan = {
        "plan_version": 1, "generated_at": "2026-08-19T09:12:44Z",
        "organizer": "date", "grain": "month", "title": "By date — month",
        "complete": complete, "photo_count": 1,
        "albums": [{"folder": "June 2024", "title": "June 2024",
                    "description": "1 photo.", "photo_count": 1, "kind": "album"}],
        "files": [{"hash": file_hash(tmp_path / "IMG.jpg"),
                   "target": "June 2024/IMG.jpg", "sources": ["IMG.jpg"],
                   "bytes": 3}],
    }
    path = tmp_path / "ivms777-plan.json"
    path.write_text(json.dumps(plan))
    return path


def test_apply_with_yes_moves_the_files(tmp_path):
    path = write_plan(tmp_path)
    assert main(["apply", str(path), "--root", str(tmp_path), "--yes", *STAMP_ARGS]) == 0
    assert (tmp_path / "June 2024" / "IMG.jpg").read_bytes() == b"one"


def test_apply_prints_the_summary_before_asking(tmp_path, capsys):
    path = write_plan(tmp_path)
    main(["apply", str(path), "--root", str(tmp_path), "--yes", *STAMP_ARGS])
    out = capsys.readouterr().out
    assert "1 matched in this folder" in out
    assert "1 to move" in out


def test_answering_no_changes_nothing(tmp_path, monkeypatch):
    path = write_plan(tmp_path)
    monkeypatch.setattr("builtins.input", lambda _: "n")
    assert main(["apply", str(path), "--root", str(tmp_path), *STAMP_ARGS]) == 1
    assert (tmp_path / "IMG.jpg").exists()
    assert not (tmp_path / "June 2024").exists()


def test_an_incomplete_plan_is_refused_without_the_flag(tmp_path, capsys):
    path = write_plan(tmp_path, complete=False)
    code = main(["apply", str(path), "--root", str(tmp_path), "--yes", *STAMP_ARGS])
    assert code == 3
    assert "--allow-incomplete" in capsys.readouterr().err
    assert (tmp_path / "IMG.jpg").exists()


def test_verify_reports_without_changing_anything(tmp_path, capsys):
    path = write_plan(tmp_path)
    assert main(["verify", str(path), "--root", str(tmp_path)]) == 0
    assert "1 to move" in capsys.readouterr().out
    assert (tmp_path / "IMG.jpg").exists()


def test_undo_puts_everything_back(tmp_path):
    path = write_plan(tmp_path)
    main(["apply", str(path), "--root", str(tmp_path), "--yes", *STAMP_ARGS])
    journal = tmp_path / ".ivms777-sync" / "journal-20260819T091244Z.jsonl"
    assert main(["undo", str(journal), "--root", str(tmp_path)]) == 0
    assert (tmp_path / "IMG.jpg").read_bytes() == b"one"
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest tests/test_sync_cli.py -v`
Expected: FAIL — `unrecognized arguments: --stamp` / no `verify` subcommand.

- [ ] **Step 3: Implement the CLI**

```python
# ivms777_sync/cli.py
"""ivms777-sync — the only component that writes to your disk (design §3.2b).

    ivms777-sync apply  ivms777-plan.json [--root .] [--allow-incomplete] [--yes]
    ivms777-sync undo   .ivms777-sync/journal-<stamp>.jsonl [--root .]
    ivms777-sync verify ivms777-plan.json [--root .]
"""

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

from ivms777_sync.apply import apply_plan, undo
from ivms777_sync.match import Outcome, match
from ivms777_sync.plan import Plan, PlanError, load_plan
from ivms777_sync.scan import scan


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "undo":
        count = undo(Path(args.root), Path(args.journal))
        print(f"reversed {count} operations")
        return 0

    try:
        plan = load_plan(Path(args.plan))
    except PlanError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    root = Path(args.root)
    outcome = match(plan, scan(root))
    print(summary(plan, outcome, root))

    if args.command == "verify":
        return 0
    if not plan.complete and not args.allow_incomplete:
        print("error: this library is still being processed. Wait for it to "
              "finish, or re-run with --allow-incomplete.", file=sys.stderr)
        return 3
    if not args.yes and input("  reorganize this folder? [y/N] ").strip().lower() != "y":
        print("nothing was changed.")
        return 1
    journal = apply_plan(root, plan, outcome, stamp=args.stamp or _stamp())
    print(f"done. to reverse it: ivms777-sync undo {journal}")
    return 0


def summary(plan: Plan, outcome: Outcome, root: Path) -> str:
    moves = [m for m in outcome.moves if m.kind == "move"]
    dupes = [m for m in outcome.moves if m.kind == "duplicate"]
    lines = [
        f"  plan: {plan.title}, generated {plan.generated_at}, "
        f"{len(plan.files)} photos",
        "",
        f"  {len(plan.files) - outcome.missing} matched in this folder",
        f"  {outcome.missing} in the plan but not found here        (left alone)",
        f"  {outcome.unknown} files here that the plan doesn't know (left alone)",
        "",
        f"  {len(moves)} to move",
        f"  {outcome.in_place} already in place",
        f"  {len(dupes)} redundant copies -> _duplicates/",
        f"  {len(plan.albums)} folders to create, each with description.txt",
        f"  {len(outcome.conflicts)} conflicts",
    ]
    if not plan.complete:
        lines.insert(1, "  WARNING: the library was still processing when this "
                        "plan was made")
    lines.append("")
    lines.append(f"  nothing has been changed yet. root: {root}")
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ivms777-sync")
    sub = parser.add_subparsers(dest="command", required=True)

    apply_cmd = sub.add_parser("apply", help="reorganize this folder from a plan")
    apply_cmd.add_argument("plan")
    apply_cmd.add_argument("--root", default=".")
    apply_cmd.add_argument("--allow-incomplete", action="store_true")
    apply_cmd.add_argument("--yes", action="store_true")
    # Tests pin the journal name; a real run stamps it with the clock.
    apply_cmd.add_argument("--stamp", default=None, help=argparse.SUPPRESS)

    verify_cmd = sub.add_parser("verify", help="report what apply would do")
    verify_cmd.add_argument("plan")
    verify_cmd.add_argument("--root", default=".")

    undo_cmd = sub.add_parser("undo", help="reverse an earlier apply")
    undo_cmd.add_argument("journal")
    undo_cmd.add_argument("--root", default=".")
    return parser


def _stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Add the HTTP client for a fresh plan**

```python
# ivms777_sync/client.py
"""Fetch a fresh plan from the app — used by `verify --url` to answer "has the
library moved on since I applied this?". `apply` never needs the network."""

import httpx


def fetch_plan(url: str, by: str | None = None, grain: str | None = None) -> dict:
    params = {k: v for k, v in (("by", by), ("grain", grain)) if v}
    response = httpx.get(f"{url.rstrip('/')}/api/plan", params=params, timeout=60)
    response.raise_for_status()
    return response.json()
```

Wire it into `verify`: add `verify_cmd.add_argument("--url", default=None)`, and when `args.command == "verify"` and `args.url`, write the fetched plan to `args.plan` before loading it, printing `refreshed the plan from <url>`.

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_sync_cli.py -v` — expected: 6 passed.

- [ ] **Step 6: Lint, full suite, stop for review**

Run: `uv run ruff check . && uv run pytest -q`

---

### Task 10: End-to-end — the plan the app produced is the plan the CLI applies

The one test that proves the two stages actually fit. It builds a plan through `organize.plan`, writes it to a temp folder holding the same bytes, and applies it with the CLI.

**Files:**
- Test: `tests/test_stage2_end_to_end.py`

**Interfaces:**
- Consumes: `organize.plan.build_plan`, `ivms777_sync.cli.main`.
- Produces: nothing.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_stage2_end_to_end.py
"""Stage 1 makes the plan, stage 2 executes it. This is the seam."""

import hashlib
import json

from albums.by_date import ByDateOrganizer
from ivms777_sync.cli import main
from organize.plan import build_plan
from tests.factories import add_photo


def test_a_plan_built_by_the_app_reorganizes_a_real_folder(conn, tmp_path):
    payloads = {"iphone dump/IMG_4471.jpg": b"first", "scan012.jpg": b"second"}
    for rel, data in payloads.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    add_photo(conn, photo_id=1, content_hash=hashlib.sha256(b"first").hexdigest(),
              thumb_key="1.jpg", shot_at="2024-06-14T18:30:12", bytes=5,
              sources=["iphone dump/IMG_4471.jpg"])
    add_photo(conn, photo_id=2, content_hash=hashlib.sha256(b"second").hexdigest(),
              thumb_key="2.jpg", bytes=6, sources=["scan012.jpg"])   # undated

    plan = build_plan(conn, 1, ByDateOrganizer(), "month",
                      generated_at="2026-08-19T09:12:44Z")
    plan_path = tmp_path / "ivms777-plan.json"
    plan_path.write_text(json.dumps(plan))

    assert main(["apply", str(plan_path), "--root", str(tmp_path), "--yes",
                 "--stamp", "20260819T091244Z"]) == 0

    assert (tmp_path / "June 2024" / "IMG_4471.jpg").read_bytes() == b"first"
    assert (tmp_path / "_unsorted" / "scan012.jpg").read_bytes() == b"second"
    assert (tmp_path / "June 2024" / "description.txt").read_text().startswith(
        "June 2024")
    assert not (tmp_path / "iphone dump" / "IMG_4471.jpg").exists()
```

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/test_stage2_end_to_end.py -v`
Expected: it either passes (both halves already agree) or fails on a concrete mismatch between the plan builder and the CLI. Fix whichever side is wrong — the contract in `docs/sync-cli.md` decides.

- [ ] **Step 3: Verify the docs still match reality**

Read `docs/sync-cli.md` end to end against the code you just wrote. In particular check: the JSON example's field names, the album `kind`/`references` fields, the `_unsorted/` rule, the staging-directory description, and the CLI's summary text. Fix the doc if it drifted — project CLAUDE.md: the doc is wrong until it is fixed.

- [ ] **Step 4: Lint, full suite, stop for review**

Run: `uv run ruff check . && uv run pytest -q`

---

## Out of scope

- A browser executor via the File System Access API. It is secure-context only, so it does not work on `http://<jetson>:8000` at all, and Firefox/Safari have no directory picker (design §3.2b). Recorded in `docs/roadmap.md` as future work.
- A watch mode that applies a plan as soon as it lands in the folder. Also future work.
- Any change to how memories are composed. Task 2 only carries a `kind` that already exists conceptually in §11; the theme pass itself is not built here.
