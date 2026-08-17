# Stage 2 — the `ivms777-sync` CLI

The manifest format, the layout contract, the CLI commands, and the safety rules
for the local sync tool. Design §12 carries the two-stage design; this file is the
exact contract. Section references point into `docs/design.md`.

Stage 1 learns what the photos are. Stage 2 acts on it, on the machine that holds
them. The whole contract between the two is one JSON document.

## The manifest

`GET /api/manifest?layout=date` returns, for every photo the owner has, its content
hash, the path the chosen layout says it belongs at, and every local path it was
uploaded from.

```json
{
  "manifest_version": 1,
  "generated_at": "2026-08-13T09:12:44Z",
  "layout": "date",
  "complete": true,
  "photo_count": 4812,
  "files": [
    {
      "hash": "9f2c1a…",
      "target": "2024/2024-06 June/2024-06-14_183012_IMG_4471.jpg",
      "sources": ["Pictures/iphone dump/IMG_4471.jpg",
                  "Desktop/to sort/IMG_4471 copy.jpg"],
      "bytes": 3841122
    }
  ]
}
```

`complete` is false while any job row is still pending or running (§8). `sources`
carries every path this content arrived from; the first entry is the copy the plan
will keep, chosen as the shallowest path and then the lexicographically smallest, so
the result is stable across runs.

The manifest is derived state. Regenerating it with a different layout produces a
different `target` for every file and nothing else changes.

## Layouts

A layout is a pure function from a photo's facts to a relative path. It sees EXIF
facets, tags, captions, and group membership, and it may use none of them.

```python
class Layout(Protocol):
    name: str
    def target(self, photo: PhotoView) -> PurePosixPath: ...
```

Three ship in v1. `date` is the default.

**`date`** — a year/month tree, filenames prefixed with capture time. Depends only
on EXIF, so it is completely stable: re-running it after new captions or a retrained
model produces byte-identical output.

```
2024/2024-06 June/2024-06-14_183012_IMG_4471.jpg
2025/2025-01 January/2025-01-03_101533_DSC_0088.jpg
_undated/9f2c1a3e_scan012.jpg
```

**`date-tags`** — the same tree holds every real file, plus an `_albums/` directory
of symlinks grouped by the strongest tags. One copy of the bytes, many ways in.
Where symlinks are unavailable the tool reports it and writes the date tree alone
rather than duplicating files.

```
2024/2024-06 June/2024-06-14_183012_IMG_4471.jpg
_albums/beach/2024-06-14_183012_IMG_4471.jpg -> ../../2024/2024-06 June/…
```

**`flat`** — one directory, every file named by capture time. For people who search
rather than browse and want no tree at all.

Photos with no usable capture date go to `_undated/`, named by a short hash prefix
so the name is stable. When two photos would land on the same path, the later one
gains an `_<hash8>` suffix; the choice is deterministic, so a re-run does not shuffle
names.

Layouts live server-side in `ivms777/organize/`. Adding one is a new module and a
new option on `/export` — the CLI needs no change, because it only executes paths it
is handed.

## The CLI

```
ivms777-sync plan   --url https://photos.example --root ~/Pictures \
                      --layout date -o plan.json
ivms777-sync apply  plan.json
ivms777-sync undo   .ivms777-sync/journal-20260813T091244Z.jsonl
ivms777-sync verify --url https://photos.example --root ~/Pictures
```

**`plan`** fetches the manifest, walks `--root`, hashes every file it finds, and
matches by hash — never by path or filename, so a library reorganized since upload
still matches perfectly. It writes `plan.json` and prints a summary:

```
  4,812 photos in manifest
  4,796 matched on disk
     16 in manifest but not found locally      (left alone)
    241 files on disk not in manifest          (left alone)

  3,104 to move          e.g. Pictures/iphone dump/IMG_4471.jpg
                           -> 2024/2024-06 June/2024-06-14_183012_IMG_4471.jpg
  1,692 already in place
    387 redundant copies -> _duplicates/       (reclaims 4.1 GB)
      0 conflicts

  nothing has been changed. run: ivms777-sync apply plan.json
```

**`apply`** executes that plan and nothing else. It re-hashes each file immediately
before touching it and skips any that changed since planning.

**`undo`** replays the journal backwards, returning every file to where it was.

**`verify`** hashes the root and reports how it differs from the manifest, changing
nothing. It is `plan` without the output file.

## Safety

The tool moves other people's photographs, so its defaults are paranoid.

- **Nothing is deleted, ever.** Redundant copies move to `_duplicates/` under their
  original relative path. Reclaiming the space is a folder the user deletes when
  they are satisfied.
- **Nothing outside the manifest is touched.** Files the manifest does not know are
  counted and reported, never moved.
- **Every operation is journaled before it runs.** `.ivms777-sync/journal-<ts>.jsonl`
  gets one record per operation with its status updated after. A crash mid-run
  leaves a journal that `undo` can replay.
- **Moves prefer `os.rename`.** Within one filesystem a move is atomic. Across
  filesystems it is copy, fsync, verify the hash, then unlink — the original goes
  only after the copy is proven good.
- **Conflicts stop the plan, not the apply.** If a target path is occupied by a file
  that belongs elsewhere, `plan` orders the moves so the occupant leaves first,
  routing through a temporary name when the moves form a cycle. A conflict it cannot
  order is reported and that file is skipped.
- **Plans expire.** A plan records the manifest's `generated_at` and the root it was
  built against. `apply` refuses a plan built for a different root, and warns when
  the manifest has since changed.
