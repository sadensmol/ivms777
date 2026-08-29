"""Albums -> the stage-2 plan document (design §12, docs/sync-cli.md).

The organizer IS the layout: whatever `/organize` is showing is exactly the tree
the CLI will build inside the user's own folder. Nothing here touches a
filesystem — the plan is data, and `ivms777_sync` is the only writer (§3.2b).
"""

import sqlite3

from albums.base import Album, Organizer
from organize.names import unique_folder_names

PLAN_VERSION = 1
UNSORTED = "_unsorted"


def build_plan(
    conn: sqlite3.Connection,
    owner_id: int,
    organizer: Organizer,
    grain: str | None = None,
    *,
    generated_at: str,
) -> dict:
    # Events claim a photo before a theme can (§12): themes sort last, everything
    # else keeps the organizer's own order (sorted() is stable, so this is the
    # only ordering rule needed).
    albums = organizer.organize(conn, owner_id, grain)
    ordered = sorted(albums, key=_is_theme)
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
            if target is None and photo_id in owner_target:
                references.append(owner_target[photo_id])
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
    has nothing on disk to move (§12 rules 2 and 6)."""
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
    # Filenames never change (§12) — only a same-name collision within one folder
    # appends the hash, deterministically, so a re-run does not shuffle names.
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
    first entry is the copy the CLI keeps; the rest are redundant copies (§12)."""
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
