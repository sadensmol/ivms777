# Testing

The testing strategy for the app. Design lives in `docs/design.md`; this file is
how that design is verified.

- Inference and embedding sit behind protocols with deterministic fakes. The
  fake embedder returns a hash-derived unit vector, so similarity is
  reproducible. The whole pipeline, search, grouping, and chat context assembly
  test in milliseconds with no model weights and no network.
- Fixture images are generated with PIL at test time, including EXIF, so the
  repository carries no binary test data.
- Integration tests run the full pipeline over ~20 generated images against a
  temporary SQLite file with `sqlite-vec` loaded.
- Repository tests assert that every user-scoped query filters on `owner_id`,
  including a test that a second owner's photos never appear in the first
  owner's results.
- One optional, explicitly-marked test loads the real SigLIP model and asserts
  that a picture of a beach ranks above a picture of a keyboard for the query
  "beach". Skipped by default.
- Route tests use FastAPI's `TestClient` and assert on rendered HTML fragments.
- Upload is tested end to end through `TestClient`: probe returns only unknown
  hashes, a body whose bytes do not match the declared hash is rejected, the
  same file sent twice creates one `photos` row and two `photo_sources` rows.
- Layouts are pure functions and test as such — a `PhotoView` in, a path out —
  including collision suffixes, undated photos, and characters illegal on
  Windows.
- `ivms777_sync` tests build a real directory tree in `tmp_path` and run
  `plan` and `apply` against a manifest fixture, then assert the tree matches
  the expected layout exactly. Every such test also runs `undo` and asserts the
  tree is byte-identical to how it started.
- Failure injection covers the paths that can lose data: a crash between
  journal write and rename, a target that already exists, a file modified
  between plan and apply, and a cross-filesystem move whose copy is truncated.
