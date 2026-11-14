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
- Plan building is a pure function and tests as such — albums in, a plan document
  out — per organizer, including folder-name sanitizing, duplicate album titles,
  collision suffixes, `_unsorted/`, and the memory rules (event owns the file,
  theme folders carry only `description.txt` + `photos.txt`).
- The memory film tests entirely offline. Its **signals** and **director** are pure
  functions — rows in, an `Edit` out — so every craft rule (motion axis from
  orientation, burst detection, hero holds, chapter dips, the grade, the pacing curve)
  is asserted directly with no model. The **story pass** is tested against a stub
  client, including the case where the model returns junk and the film must still
  play. `FakeBackend.speak`/`music` return
  **real, parseable WAV bytes**, so `duration_seconds` measures a true length and the
  "stretch the slideshow to outlast the narration" rule is tested for real. A slot
  set to `off` is covered too: the prepare job must still produce a playable feed
  when the voice, the music, or both answer 503. No `audio/` module imports torch or
  a model library at module level, so tests import `audio/` under a plain `uv sync`.
  They run its pure parts against fakes: a fake model, a fake `torch` or
  `huggingface_hub` module, `pickle` in place of `torch.save`. No test imports
  `audio/kokoro_voice.py` or `audio/musicgen.py`; those two children run only on a
  machine with the `models` extra.
- `ivms777_sync` tests build a real directory tree in `tmp_path` and run `apply`
  against a plan fixture, then assert the tree matches the plan exactly. Every such
  test also runs `undo` and asserts the tree is byte-identical to how it started.
- Failure injection covers the paths that can lose data: a crash between journal
  write and rename, a target that already exists, a file modified between the walk
  and the move, and a cross-filesystem move whose copy is truncated.
