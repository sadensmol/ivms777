# Stage 2 — the plan document and the `ivms777-sync` CLI

The plan format, the folder rules, the CLI commands, and the safety rules for the
local sync tool. Design §12 carries the two-stage design; this file is the exact
contract. Section references point into `docs/design.md`.

Stage 1 learns what the photos are. Stage 2 acts on it, on the machine that holds
them. The whole contract between the two is one JSON document you download from
Organize.

## The organizer is the layout

There is no layout picker and no `Layout` protocol. The principle selected in
Organize (§11) — `date` at `day`/`month`/`year`, `camera`, `place`, `memories` — is
the folder structure. One album becomes one folder; a photo's album decides its
folder; a new organizer is an export for free.

Two rules make it safe to run in place, inside a folder that already holds the
photos:

- **Filenames never change.** The plan moves a file between folders and nothing
  else. Only a collision — two different files landing on the same path — appends
  `_<hash8>` to the later one, deterministically, so a re-run does not shuffle names.
- **Only album folders are created.** Anything the plan does not name is left where
  it is.

## The plan document

`GET /api/plan?by=memories` (or `by=date&grain=month`, …) returns the whole plan for
the owner's library. The browser downloads it as `ivms777-plan.json`; you save it
into the folder you want reorganized.

```json
{
  "plan_version": 1,
  "generated_at": "2026-08-19T09:12:44Z",
  "organizer": "date",
  "grain": "month",
  "title": "By date — month",
  "complete": true,
  "photo_count": 4812,
  "albums": [
    {
      "folder": "2024-06 June",
      "title": "June 2024",
      "description": "140 photos, mostly Canon EOS R6.",
      "photo_count": 140,
      "kind": "album"
    }
  ],
  "files": [
    {
      "hash": "9f2c1a…",
      "target": "2024-06 June/IMG_4471.jpg",
      "sources": ["Pictures/iphone dump/IMG_4471.jpg",
                  "Desktop/to sort/IMG_4471 copy.jpg"],
      "bytes": 3841122
    }
  ]
}
```

- `complete` is false while any job row is still pending or running (§8).
- `sources` carries every path this content arrived from; the first entry is the
  copy the CLI keeps, chosen as the shallowest path and then the lexicographically
  smallest, so the result is stable across runs.
- `target` is relative to whatever root the CLI is pointed at. **One plan covers the
  whole library**; run it in each source folder you want reorganized and hash
  matching takes care of the rest.
- The plan is derived state. Regenerating it with a different organizer produces a
  different `target` for every file and nothing else changes.

### Folder names

An album's title is sanitized into one path segment: path separators and the
reserved Windows characters `<>:"|?*` become `-`, control characters become a
space, runs of whitespace collapse to one, leading and trailing spaces and dots are
stripped, the result is truncated to 120 characters and stripped again, and a result
that is empty — or nothing but dashes and spaces — falls back to the album key. Two
albums that sanitize to the same name get `_2`, `_3`, … in the plan's album order,
compared case-insensitively because macOS and Windows filesystems are. The naming is
deterministic.

### `description.txt`

Every album folder gets one, written by the CLI from the plan — never read back, and
overwritten on re-apply:

```
June 2024

140 photos, mostly Canon EOS R6.
```

For a memory the body is the agent's full description, so the story survives outside
the app.

### Memories: events own files, themes describe them

A photo belongs to one **event** and any number of **themes** (§11) — a file belongs
to one folder. The plan resolves the clash and the app keeps its overlap:

- An **event** memory is a normal album: its photos are moved into it.
- A **theme** memory (`"kind": "theme"`) creates a folder containing only
  `description.txt` and `photos.txt`. The album's `"references"` field carries one
  `target` path per photo — where the file actually lives — and `photos.txt` is
  exactly that list, one per line. No symlinks, no copies, nothing that breaks on a
  filesystem that dislikes them.
- A photo in no event goes to `_unsorted/`.

`_unsorted/` is also where every other organizer puts what it cannot place: a photo
with no capture date, no camera, or no GPS.

## The CLI

```
ivms777-sync apply  ivms777-plan.json [--root .] [--allow-incomplete] [--yes]
ivms777-sync undo   .ivms777-sync/journal-20260819T091244Z.jsonl
ivms777-sync verify ivms777-plan.json [--root .]
```

**`apply`** walks `--root` (the current directory by default), hashes every file it
finds, and matches by hash — never by path or filename, so a folder reorganized
since upload still matches perfectly. It needs no network: the plan is
self-contained. It prints the summary, waits for confirmation, then moves:

```
  plan: by date — month, generated 2026-08-19 09:12 UTC, 4,812 photos

  4,796 matched in this folder
     16 in the plan but not found here        (left alone)
    241 files here that the plan doesn't know (left alone)

  3,104 to move          e.g. iphone dump/IMG_4471.jpg
                           -> 2024-06 June/IMG_4471.jpg
  1,692 already in place
    387 redundant copies -> _duplicates/       (reclaims 4.1 GB)
     34 folders to create, each with description.txt
      0 conflicts

  nothing has been changed yet. reorganize this folder? [y/N]
```

It re-hashes each file immediately before touching it and skips any that changed
since the walk.

**`undo`** replays the journal backwards, returning every file to where it was and
removing folders it created that are empty again.

**`verify`** hashes the root and reports how it differs from the plan, changing
nothing. It is `apply` with no confirmation and no writes; given `--url` it re-fetches
the plan first, so it also answers "has the library moved on since I applied this?".

## Safety

The tool moves other people's photographs, so its defaults are paranoid.

- **Nothing is deleted, ever.** Redundant copies move to `_duplicates/` under their
  original relative path. Reclaiming the space is a folder the user deletes when
  they are satisfied.
- **Nothing outside the plan is touched.** Files the plan does not know are counted
  and reported, never moved.
- **Every operation is journaled before it runs.** `.ivms777-sync/journal-<ts>.jsonl`
  gets one record per operation with its status updated after. A crash mid-run
  leaves a journal that `undo` can replay.
- **Moves prefer `os.rename`.** Within one filesystem a move is atomic. Across
  filesystems it is copy, fsync, verify the hash, then unlink — the original goes
  only after the copy is proven good.
- **Every move routes through a staging directory.** A file goes from its old path
  into `.ivms777-sync/staging/` first and into its target second — two renames on
  one filesystem. Ordering therefore cannot matter and a cycle (two photos swapping
  paths) cannot be got wrong. A target already occupied by a file the plan does not
  know is reported as a conflict and that photo is skipped, never overwritten.
- **An incomplete plan is refused.** `complete: false` means the library is still
  being processed (§8); `apply` stops unless `--allow-incomplete` is passed.
- **The source folder is never read by the app.** Only this CLI, on your machine,
  ever touches it (§3.2b).
