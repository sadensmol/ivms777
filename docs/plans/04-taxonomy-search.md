# Photo Library Organizer — Plan 04: Taxonomy tags and full search

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give every photo a set of model-derived tags across ten dimensions, put those tags in a `/library` sidebar with counts and filtering, and complete search — keyword (FTS5) plus reciprocal-rank fusion with the semantic ranking from plan 03. After this plan you can filter "beach + golden hour", search "birthday cake" and match the caption text as well as the meaning, and see on each photo which tags a model assigned and how confident it was.

**Architecture:** A new `taxonomy` job stage drains after `embed` (§8): for each photo it scores every `vocab.yaml` label with SigLIP zero-shot (text-embed the label prompt once per run, dot it against the photo's stored image vector), adds cheap pixel-statistic tags for `palette`/`quality`, thresholds per dimension, and writes rows to `photo_tags`. The same stage rebuilds the photo's `photo_fts` row from its caption and tag labels. Keyword search is FTS5 BM25 over `photo_fts`; fusion merges semantic and keyword rankings by RRF. Tag filtering and the sidebar mirror the existing EXIF-facet machinery.

**Tech Stack:** Python 3.12, SQLite + FTS5, `sqlite-vec`, NumPy, PyYAML, Pillow, FastAPI, Jinja2, HTMX. Reuses the `Embedder` protocol and `FakeEmbedder` (plan 03), the job queue (plan 02), and `search/semantic.py`.

**Spec:** `docs/design.md` — §7 (taxonomy, ten dimensions, `vocab.yaml`), §6 (`tags`, `photo_tags`, `photo_fts`), §8 (taxonomy stage), §9 (retrieval: tag facets, keyword, fusion), §13 (`/library` sidebar, `/photo` tag panel).

**Builds on:** plan 03. Every photo already carries a 1152-dim SigLIP vector in `photo_vec`; taxonomy reads those vectors and needs no re-embedding. `search_photos` and `knn` are reused verbatim.

**Covers:** the taxonomy + keyword/fusion slice of spec phase 2. **Deferred:** caption vocabulary *mining* (§7.1 — needs captions, plan 06); the caption stage and per-photo AI title/description (plan 05); the query planner and chat (plans 06, 08). The `vlm` tag source stays empty until captions land.

## Global Constraints

- Python 3.12. Dependencies via `uv` with a committed `uv.lock`.
- **Never run `git commit`/`git add`.** The user commits. Every task ends at a checkpoint.
- Every user-scoped query filters on `owner_id` (constant `settings.owner_id`).
- Tests must not load a real model or hit the network. Taxonomy is tested with `FakeEmbedder`; pixel stats with generated PIL images.
- The full fast suite passes at the end of every task: `uv run pytest -q`, and `uv run ruff check .` is clean.
- Tags carry a `score` (0..1) and a `source` (`siglip` | `pixel` | `vlm` | `exif` | `user`), per §6. This plan writes `siglip` and `pixel`.
- The `taxonomy` stage name already exists in `ingest/jobs.STAGES`; this plan supplies its handler.

---

### Task 1: `vocab.yaml`, the vocabulary loader, and tag seeding

Define the ten dimensions and their starting labels in editable YAML (§7), load them, and seed the `tags` table so tag ids are stable.

**Files:**
- Create: `vocab.yaml`
- Create: `ingest/vocab.py`
- Create: `tests/test_vocab.py`

**Interfaces:**
- Produces:
  - `ingest.vocab.Vocab` — loaded vocabulary: `dimensions: dict[str, list[str]]` (dimension → labels) plus `threshold(dimension) -> float`.
  - `ingest.vocab.load_vocab(path: Path) -> Vocab`.
  - `ingest.vocab.seed_tags(conn, vocab) -> None` — inserts any missing `(dimension, label)` rows (idempotent), so `tags.id` is assigned once and never churns.
  - `ingest.vocab.tag_id_map(conn) -> dict[tuple[str, str], int]` — `(dimension, label) -> tags.id`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_vocab.py`:

```python
from pathlib import Path

from ingest.vocab import load_vocab, seed_tags, tag_id_map

VOCAB = Path("vocab.yaml")


def test_vocab_has_the_ten_dimensions():
    vocab = load_vocab(VOCAB)
    assert set(vocab.dimensions) == {
        "subject", "setting", "vibe", "emotion", "light",
        "season_weather", "composition", "palette", "occasion", "quality",
    }
    assert "beach" in vocab.dimensions["setting"]


def test_seed_tags_is_idempotent(conn):
    vocab = load_vocab(VOCAB)
    seed_tags(conn, vocab)
    first = conn.execute("SELECT count(*) AS n FROM tags").fetchone()["n"]
    seed_tags(conn, vocab)  # second run adds nothing
    assert conn.execute("SELECT count(*) AS n FROM tags").fetchone()["n"] == first
    assert first == sum(len(v) for v in vocab.dimensions.values())


def test_tag_id_map_keys_by_dimension_and_label(conn):
    vocab = load_vocab(VOCAB)
    seed_tags(conn, vocab)
    ids = tag_id_map(conn)
    assert ("setting", "beach") in ids
    assert isinstance(ids[("setting", "beach")], int)


def test_thresholds_default_when_unspecified():
    vocab = load_vocab(VOCAB)
    assert 0.0 < vocab.threshold("subject") <= 1.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_vocab.py -q`
Expected: FAIL — `ModuleNotFoundError: ingest.vocab` (and no `vocab.yaml`).

- [ ] **Step 3: Write `vocab.yaml`**

Create `vocab.yaml` with the ten dimensions and the §7 starter labels. Include an optional `thresholds:` block (per-dimension) and a `default_threshold`:

```yaml
default_threshold: 0.18
thresholds:
  palette: 0.0      # palette/quality come from pixel stats; SigLIP only refines
  quality: 0.0
dimensions:
  subject: [portrait, group of people, pet, food, architecture, nature, vehicle, document, artwork]
  setting: [indoor, outdoor, beach, mountain, forest, city street, restaurant, home, office, water, snow]
  vibe: [cozy, energetic, serene, moody, festive, nostalgic, dramatic, minimal, chaotic, romantic]
  emotion: [joyful, sad, tense, affectionate, playful, contemplative, neutral]
  light: [golden hour, blue hour, night, harsh midday, overcast, backlit, neon, candlelit]
  season_weather: [summer, autumn, winter, spring, rain, snow, fog, clear sky]
  composition: [close-up, wide shot, aerial, shallow depth of field, symmetry, silhouette, leading lines]
  palette: [warm, cool, pastel, vivid, monochrome, dark, bright, high contrast]
  occasion: [birthday, wedding, travel, hike, concert, holiday, everyday, work]
  quality: [sharp, blurry, noisy, overexposed, underexposed]
```

(The threshold values are placeholders; §7 says they are tuned against a hand-labeled dev set. The bake-off/tuning is out of scope here — sensible defaults ship, tuning is a later note.)

- [ ] **Step 4: Add PyYAML**

```bash
uv add pyyaml
```

- [ ] **Step 5: Write the loader**

Create `ingest/vocab.py`:

```python
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Vocab:
    dimensions: dict[str, list[str]]
    _thresholds: dict[str, float]
    _default: float

    def threshold(self, dimension: str) -> float:
        return self._thresholds.get(dimension, self._default)


def load_vocab(path: Path) -> Vocab:
    data = yaml.safe_load(path.read_text())
    return Vocab(
        dimensions=data["dimensions"],
        _thresholds=data.get("thresholds", {}),
        _default=float(data.get("default_threshold", 0.18)),
    )


def seed_tags(conn: sqlite3.Connection, vocab: Vocab) -> None:
    for dimension, labels in vocab.dimensions.items():
        conn.executemany(
            "INSERT INTO tags(dimension, label) VALUES (?, ?)"
            " ON CONFLICT(dimension, label) DO NOTHING",
            [(dimension, label) for label in labels],
        )


def tag_id_map(conn: sqlite3.Connection) -> dict[tuple[str, str], int]:
    return {
        (row["dimension"], row["label"]): row["id"]
        for row in conn.execute("SELECT id, dimension, label FROM tags")
    }
```

- [ ] **Step 6: Run to verify it passes**

Run: `uv run pytest tests/test_vocab.py -q`
Expected: PASS.

- [ ] **Step 7: Run the whole suite**

Run: `uv run pytest -q`
Expected: PASS.

- [ ] **Step 8: Checkpoint** — report the ten dimensions, the seed idempotency, and the threshold defaults. Stop. Do not commit.

---

### Task 2: Pixel-statistic tags for `palette` and `quality`

`palette` and `quality` come from cheap pixel statistics first, SigLIP refines them (§7). Compute them as a pure function so it tests with generated images and needs no model.

**Files:**
- Create: `ingest/pixels.py`
- Create: `tests/test_pixels.py`

**Interfaces:**
- Produces:
  - `ingest.pixels.pixel_tags(image: Image.Image) -> list[tuple[str, str, float]]` — `(dimension, label, score)` for `palette` and `quality`, from HSV means, brightness, a Laplacian-variance sharpness proxy, and histogram clipping. Only labels that clear a small internal margin are returned.

- [ ] **Step 1: Write the failing test**

Create `tests/test_pixels.py`. Assert direction, not exact scores — a saturated warm image reads warm/vivid, a grey image reads monochrome, a blank flat image reads blurry, a blown-out image reads overexposed:

```python
from PIL import Image

from ingest.pixels import pixel_tags


def _labels(image):
    return {(dim, label) for dim, label, _ in pixel_tags(image)}


def test_a_saturated_orange_reads_warm_and_vivid():
    labels = _labels(Image.new("RGB", (64, 64), (230, 120, 20)))
    assert ("palette", "warm") in labels
    assert ("palette", "vivid") in labels


def test_a_grey_image_reads_monochrome():
    assert ("palette", "monochrome") in _labels(Image.new("RGB", (64, 64), (128, 128, 128)))


def test_a_flat_image_reads_blurry():
    # No edges at all -> zero Laplacian variance -> blurry, never sharp.
    labels = _labels(Image.new("RGB", (64, 64), (100, 100, 100)))
    assert ("quality", "blurry") in labels
    assert ("quality", "sharp") not in labels


def test_a_blown_out_image_reads_overexposed():
    assert ("quality", "overexposed") in _labels(Image.new("RGB", (64, 64), (255, 255, 255)))


def test_a_black_image_reads_underexposed():
    assert ("quality", "underexposed") in _labels(Image.new("RGB", (64, 64), (2, 2, 2)))
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_pixels.py -q`
Expected: FAIL — `ModuleNotFoundError: ingest.pixels`.

- [ ] **Step 3: Write the pixel scorer**

Create `ingest/pixels.py`. Work on a small downscale for speed. Use NumPy for the stats; keep the thresholds legible and commented.

```python
import numpy as np
from PIL import Image


def pixel_tags(image: Image.Image) -> list[tuple[str, str, float]]:
    small = image.convert("RGB").resize((128, 128))
    rgb = np.asarray(small, dtype=np.float32) / 255.0
    hsv = np.asarray(small.convert("HSV"), dtype=np.float32) / 255.0
    hue, sat, val = hsv[..., 0].mean(), hsv[..., 1].mean(), hsv[..., 2].mean()
    grey = rgb.mean(axis=2)
    # Laplacian-variance sharpness proxy: variance of a 4-neighbour edge response.
    lap = (
        -4 * grey
        + np.roll(grey, 1, 0) + np.roll(grey, -1, 0)
        + np.roll(grey, 1, 1) + np.roll(grey, -1, 1)
    )
    sharpness = float(lap.var())
    clipped_high = float((grey > 0.98).mean())
    clipped_low = float((grey < 0.02).mean())

    out: list[tuple[str, str, float]] = []

    def add(dimension: str, label: str, score: float) -> None:
        out.append((dimension, label, round(min(1.0, max(0.0, score)), 3)))

    # palette
    if sat < 0.12:
        add("palette", "monochrome", 1.0 - sat)
    else:
        add("palette", "warm" if (hue < 0.14 or hue > 0.92) else "cool", sat)
        if sat > 0.55:
            add("palette", "vivid", sat)
        elif sat < 0.30:
            add("palette", "pastel", 0.6)
    add("palette", "dark" if val < 0.30 else "bright", abs(val - 0.5) * 2)

    # quality
    if sharpness < 1e-4:
        add("quality", "blurry", 1.0)
    elif sharpness > 2e-3:
        add("quality", "sharp", min(1.0, sharpness * 100))
    if clipped_high > 0.25:
        add("quality", "overexposed", clipped_high)
    if clipped_low > 0.25:
        add("quality", "underexposed", clipped_low)
    return out
```

Tune the constants until the task-2 tests pass; they are the calibration this task owns.

- [ ] **Step 4: Run the pixel tests**

Run: `uv run pytest tests/test_pixels.py -q`
Expected: PASS.

- [ ] **Step 5: Run the whole suite**

Run: `uv run pytest -q`
Expected: PASS.

- [ ] **Step 6: Checkpoint** — report which labels each fixture image produced and the constants chosen. Stop. Do not commit.

---

### Task 3: The taxonomy stage — zero-shot scoring + FTS reindex

Wire it all into the job queue: score every label with SigLIP against the stored image vector, add the pixel tags, threshold per dimension, write `photo_tags`, and rebuild the `photo_fts` row.

**Files:**
- Create: `ingest/taxonomy.py`
- Create: `tests/test_taxonomy_stage.py`
- Modify: `ingest/receive.py` (enqueue `taxonomy` after `embed`)
- Modify: `web/app.py` and `ingest/cli.py` (drain `taxonomy`)

**Interfaces:**
- Consumes: `Embedder` (`embed_texts`), `read_vector` (`embedding/store.py`), `Vocab`/`seed_tags`/`tag_id_map` (task 1), `pixel_tags` (task 2), `LocalStorage` (for the thumbnail the pixel stats read).
- Produces:
  - `ingest.taxonomy.label_prompt(dimension, label) -> str` — the zero-shot prompt (exposed so tests can mirror it).
  - `ingest.taxonomy.reindex_fts(conn, photo_id) -> None` — rebuild the photo's `photo_fts` row from its caption + tag labels.
  - `ingest.taxonomy.taxonomy_handler(derived, embedder, vocab) -> StageHandler`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_taxonomy_stage.py`. The fake embedder makes a photo's vector match a label prompt exactly, so scoring is deterministic:

```python
from pathlib import Path

from embedding.fakes import FakeEmbedder
from embedding.store import write_vector
from ingest.jobs import enqueue, stage_counts
from ingest.taxonomy import label_prompt, taxonomy_handler
from ingest.vocab import load_vocab, seed_tags
from ingest.worker import drain
from storage.local import LocalStorage
from tests.factories import add_photo
from tests.fixtures import make_jpeg
from storage.keys import content_key

VOCAB = load_vocab(Path("vocab.yaml"))


def _photo_with_vector(conn, derived, pid, label_dim, label):
    key = content_key(f"{pid:02x}" * 32, ".jpg")
    make_jpeg(derived.local_path(f"{pid}_320.jpg"))  # thumbnail the pixel stats read
    add_photo(conn, photo_id=pid, content_hash=f"{pid:02x}" * 32, thumb_key=f"{pid}_320.jpg")
    # vector == the label's prompt embedding -> that label scores 1.0
    write_vector(conn, pid, FakeEmbedder().embed_texts([label_prompt(label_dim, label)])[0])


def test_taxonomy_assigns_the_matching_label(conn, tmp_path):
    derived = LocalStorage(tmp_path / "thumbs")
    seed_tags(conn, VOCAB)
    _photo_with_vector(conn, derived, 1, "setting", "beach")
    enqueue(conn, 1, "taxonomy")

    drain(conn, {"taxonomy": taxonomy_handler(derived, FakeEmbedder(), VOCAB)})

    rows = conn.execute(
        "SELECT t.dimension, t.label, pt.source FROM photo_tags pt"
        " JOIN tags t ON t.id = pt.tag_id WHERE pt.photo_id = 1"
    ).fetchall()
    pairs = {(r["dimension"], r["label"], r["source"]) for r in rows}
    assert ("setting", "beach", "siglip") in pairs
    assert stage_counts(conn, "taxonomy")["done"] == 1


def test_taxonomy_writes_pixel_tags_for_palette(conn, tmp_path):
    derived = LocalStorage(tmp_path / "thumbs")
    seed_tags(conn, VOCAB)
    _photo_with_vector(conn, derived, 1, "setting", "beach")
    enqueue(conn, 1, "taxonomy")
    drain(conn, {"taxonomy": taxonomy_handler(derived, FakeEmbedder(), VOCAB)})
    sources = {
        r["source"] for r in conn.execute(
            "SELECT DISTINCT pt.source FROM photo_tags pt WHERE pt.photo_id = 1"
        )
    }
    assert "pixel" in sources  # palette/quality came from pixel stats


def test_taxonomy_reindexes_fts_with_tag_labels(conn, tmp_path):
    derived = LocalStorage(tmp_path / "thumbs")
    seed_tags(conn, VOCAB)
    _photo_with_vector(conn, derived, 1, "setting", "beach")
    enqueue(conn, 1, "taxonomy")
    drain(conn, {"taxonomy": taxonomy_handler(derived, FakeEmbedder(), VOCAB)})
    hit = conn.execute(
        "SELECT rowid FROM photo_fts WHERE photo_fts MATCH 'beach'"
    ).fetchone()
    assert hit is not None and hit["rowid"] == 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_taxonomy_stage.py -q`
Expected: FAIL — `ModuleNotFoundError: ingest.taxonomy`.

- [ ] **Step 3: Write the taxonomy module**

Create `ingest/taxonomy.py`. Embed every label prompt once (cache on the handler closure), then per photo: dot the photo vector against each label vector, keep those above the dimension threshold, add pixel tags, upsert into `photo_tags`, and reindex FTS.

```python
import sqlite3

from PIL import Image

from embedding.base import Embedder
from embedding.store import read_vector
from embedding.vectors import l2_normalize
from ingest.pixels import pixel_tags
from ingest.vocab import Vocab, tag_id_map
from ingest.worker import StageHandler
from storage.base import Storage

# Per-dimension prompt templates for zero-shot scoring (§7). "{label}" is filled in.
_TEMPLATES = {
    "vibe": "a photo with a {label} mood",
    "emotion": "a {label} photo",
    "light": "a photo taken in {label} light",
    "palette": "a {label} colored photo",
    "quality": "a {label} photo",
}
_DEFAULT_TEMPLATE = "a photo of {label}"


def label_prompt(dimension: str, label: str) -> str:
    return _TEMPLATES.get(dimension, _DEFAULT_TEMPLATE).format(label=label)


def reindex_fts(conn: sqlite3.Connection, photo_id: int) -> None:
    caption = conn.execute("SELECT caption FROM photos WHERE id = ?", (photo_id,)).fetchone()["caption"]
    labels = [
        r["label"] for r in conn.execute(
            "SELECT t.label FROM photo_tags pt JOIN tags t ON t.id = pt.tag_id"
            " WHERE pt.photo_id = ?", (photo_id,)
        )
    ]
    conn.execute("DELETE FROM photo_fts WHERE rowid = ?", (photo_id,))
    conn.execute(
        "INSERT INTO photo_fts(rowid, caption, tags_text) VALUES (?, ?, ?)",
        (photo_id, caption or "", " ".join(labels)),
    )


def taxonomy_handler(derived: Storage, embedder: Embedder, vocab: Vocab) -> StageHandler:
    # Embed each label prompt once for the whole drain, not once per photo.
    entries: list[tuple[str, str, list[float]]] = []
    prompts = [(d, lbl) for d, labels in vocab.dimensions.items() for lbl in labels]
    vectors = embedder.embed_texts([label_prompt(d, lbl) for d, lbl in prompts])
    for (d, lbl), vec in zip(prompts, vectors):
        entries.append((d, lbl, l2_normalize(vec)))

    def handle(conn: sqlite3.Connection, photo_id: int) -> None:
        ids = tag_id_map(conn)
        image_vec = read_vector(conn, photo_id)
        scored: list[tuple[str, str, float, str]] = []
        if image_vec is not None:
            image_vec = l2_normalize(image_vec)
            for dimension, label, label_vec in entries:
                score = sum(a * b for a, b in zip(image_vec, label_vec))
                if score >= vocab.threshold(dimension):
                    scored.append((dimension, label, float(score), "siglip"))
        thumb = conn.execute("SELECT thumb_key FROM photos WHERE id = ?", (photo_id,)).fetchone()["thumb_key"]
        path = derived.local_path(thumb) if thumb else None
        if path is not None and path.is_file():
            with Image.open(path) as image:
                for dimension, label, score in pixel_tags(image):
                    scored.append((dimension, label, score, "pixel"))
        _write_tags(conn, photo_id, scored, ids)
        reindex_fts(conn, photo_id)

    return handle


def _write_tags(conn, photo_id, scored, ids):
    # Idempotent re-run: clear this photo's model/pixel tags, keep user/exif ones.
    conn.execute(
        "DELETE FROM photo_tags WHERE photo_id = ? AND source IN ('siglip', 'pixel')",
        (photo_id,),
    )
    conn.executemany(
        "INSERT INTO photo_tags(photo_id, tag_id, score, source) VALUES (?, ?, ?, ?)"
        " ON CONFLICT(photo_id, tag_id, source) DO UPDATE SET score = excluded.score",
        [
            (photo_id, ids[(dimension, label)], score, source)
            for dimension, label, score, source in scored
            if (dimension, label) in ids
        ],
    )
```

- [ ] **Step 4: Run the stage tests**

Run: `uv run pytest tests/test_taxonomy_stage.py -q`
Expected: PASS.

- [ ] **Step 5: Enqueue and drain the stage**

In `ingest/receive.py`, after the `embed` enqueue, add `enqueue(conn, photo_id, "taxonomy")`. Add a receive test asserting `stage_counts(conn, "taxonomy")["pending"] == 1`, mirroring the plan-03 embed test.

In `web/app.py`'s `drain_now` and `ingest/cli.py`, build one taxonomy handler (load `vocab.yaml`, seed tags once at startup) and add `"taxonomy": taxonomy_handler(ctx.derived, embedder, vocab)` to the handler dict. Seed tags at app startup so ids exist before the first drain:

```python
    vocab = load_vocab(Path("vocab.yaml"))
    seed_tags(ctx.conn, vocab)
```

The embedder is already built for `embed`; reuse it. Because stages drain in `STAGES` order (`thumbnail`, `embed`, `taxonomy`, `caption`), taxonomy runs only after every photo is embedded — exactly the §8 ordering that keeps the Jetson profile viable.

- [ ] **Step 6: Run the whole suite**

Run: `uv run pytest -q`
Expected: PASS. Upload/library tests now also tag via the fake; assert counts are unaffected (tags add no grid tiles).

- [ ] **Step 7: Checkpoint** — report the stage contract, the once-per-drain label embedding, the idempotent re-run, and the FTS reindex. Stop. Do not commit.

---

### Task 4: Keyword search and reciprocal-rank fusion

Add FTS5 BM25 keyword search and merge it with the semantic ranking by RRF (§9).

**Files:**
- Create: `search/keyword.py`
- Create: `search/fusion.py`
- Create: `tests/test_keyword_fusion.py`

**Interfaces:**
- Produces:
  - `search.keyword.keyword_search(conn, owner_id, query, k) -> list[int]` — photo ids best-BM25 first, owner-scoped, over `photo_fts`.
  - `search.fusion.reciprocal_rank_fusion(rankings: list[list[int]], k_const=60) -> list[int]` — merged ids by `sum 1/(k_const + rank)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_keyword_fusion.py`:

```python
from search.fusion import reciprocal_rank_fusion
from search.keyword import keyword_search
from tests.factories import add_photo


def _fts(conn, pid, caption, tags):
    add_photo(conn, photo_id=pid, content_hash=f"h{pid}", thumb_key=f"{pid}.jpg", caption=caption)
    conn.execute(
        "INSERT INTO photo_fts(rowid, caption, tags_text) VALUES (?, ?, ?)", (pid, caption, tags)
    )


def test_keyword_matches_caption_words(conn):
    _fts(conn, 1, "a birthday cake with candles", "food indoor")
    _fts(conn, 2, "a dog on a beach", "pet beach")
    assert keyword_search(conn, owner_id=1, query="birthday", k=10) == [1]


def test_keyword_matches_tag_text(conn):
    _fts(conn, 1, "no words here", "beach summer")
    assert 1 in keyword_search(conn, owner_id=1, query="beach", k=10)


def test_keyword_is_owner_scoped(conn):
    add_photo(conn, photo_id=1, owner_id=1, content_hash="a", thumb_key="a.jpg", caption="cake")
    add_photo(conn, photo_id=2, owner_id=2, content_hash="b", thumb_key="b.jpg", caption="cake")
    conn.execute("INSERT INTO photo_fts(rowid, caption, tags_text) VALUES (1, 'cake', '')")
    conn.execute("INSERT INTO photo_fts(rowid, caption, tags_text) VALUES (2, 'cake', '')")
    assert keyword_search(conn, owner_id=1, query="cake", k=10) == [1]


def test_fusion_prefers_items_ranked_well_by_both():
    fused = reciprocal_rank_fusion([[1, 2, 3], [2, 1, 4]])
    assert fused[0] in (1, 2)
    assert set(fused) == {1, 2, 3, 4}


def test_fusion_ignores_empty_rankings():
    assert reciprocal_rank_fusion([[], [5, 6]]) == [5, 6]
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_keyword_fusion.py -q`
Expected: FAIL — `ModuleNotFoundError: search.keyword`.

- [ ] **Step 3: Write keyword search**

Create `search/keyword.py`. FTS5 is not owner-aware, so join to `photos` for the owner filter; escape the query into a phrase so punctuation never breaks the MATCH.

```python
import sqlite3


def keyword_search(conn: sqlite3.Connection, owner_id: int, query: str, k: int) -> list[int]:
    text = query.strip()
    if not text:
        return []
    match = '"' + text.replace('"', '""') + '"'  # treat input as a phrase, punctuation-safe
    rows = conn.execute(
        "SELECT f.rowid AS photo_id FROM photo_fts f"
        " JOIN photos p ON p.id = f.rowid"
        " WHERE photo_fts MATCH ? AND p.owner_id = ?"
        " ORDER BY bm25(photo_fts) LIMIT ?",
        (match, owner_id, k),
    ).fetchall()
    return [row["photo_id"] for row in rows]
```

- [ ] **Step 4: Write fusion**

Create `search/fusion.py`:

```python
def reciprocal_rank_fusion(rankings: list[list[int]], k_const: int = 60) -> list[int]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k_const + rank)
    return sorted(scores, key=lambda item: scores[item], reverse=True)
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_keyword_fusion.py -q`
Expected: PASS.

- [ ] **Step 6: Run the whole suite**

Run: `uv run pytest -q`
Expected: PASS.

- [ ] **Step 7: Checkpoint** — report the keyword and fusion contracts and the phrase-escaping choice. Stop. Do not commit.

---

### Task 5: Tag filters, the sidebar, and fusion in `/library`

Surface tags in the left sidebar with counts and filtering (AND across dimensions, OR within — §9), and switch the search box from pure semantic to semantic+keyword fusion.

**Files:**
- Create: `search/tags.py`
- Create: `tests/test_tag_filters.py`
- Modify: `web/app.py`
- Modify: `web/templates/library.html`
- Modify: `tests/test_web_search.py` (fusion path)

**Interfaces:**
- Produces:
  - `search.tags.parse_tag_filters(params) -> dict[str, list[str]]` — `t_<dimension>=a,b` → `{dimension: [a, b]}`.
  - `search.tags.tag_where(filters, score_min) -> tuple[str, list]` — an `AND EXISTS(... photo_tags ...)` fragment per dimension (OR within a dimension, AND across), for splicing into `BASE_SQL`.
  - `search.tags.tag_sidebar(conn, owner_id, dimensions, score_min, limit) -> list[dict]` — per-dimension label counts for the sidebar.

- [ ] **Step 1: Write the failing test**

Create `tests/test_tag_filters.py`:

```python
from search.tags import parse_tag_filters, tag_sidebar, tag_where
from ingest.vocab import load_vocab, seed_tags
from tests.factories import add_photo
from pathlib import Path

VOCAB = load_vocab(Path("vocab.yaml"))


def _tagged(conn, pid, pairs):
    add_photo(conn, photo_id=pid, content_hash=f"h{pid}", thumb_key=f"{pid}.jpg")
    ids = {(r["dimension"], r["label"]): r["id"] for r in conn.execute("SELECT * FROM tags")}
    for dim, label in pairs:
        conn.execute(
            "INSERT INTO photo_tags(photo_id, tag_id, score, source) VALUES (?, ?, 0.9, 'siglip')",
            (pid, ids[(dim, label)]),
        )


def test_parse_reads_t_prefixed_params():
    assert parse_tag_filters({"t_setting": "beach,forest", "q": "x"}) == {"setting": ["beach", "forest"]}


def test_tag_where_filters_by_label(conn):
    seed_tags(conn, VOCAB)
    _tagged(conn, 1, [("setting", "beach")])
    _tagged(conn, 2, [("setting", "forest")])
    where, params = tag_where({"setting": ["beach"]}, score_min=0.2)
    rows = conn.execute(
        "SELECT p.id FROM photos p WHERE p.owner_id = ?" + where, (1, *params)
    ).fetchall()
    assert [r["id"] for r in rows] == [1]


def test_tag_where_ands_across_dimensions(conn):
    seed_tags(conn, VOCAB)
    _tagged(conn, 1, [("setting", "beach"), ("vibe", "serene")])
    _tagged(conn, 2, [("setting", "beach")])
    where, params = tag_where({"setting": ["beach"], "vibe": ["serene"]}, score_min=0.2)
    rows = conn.execute("SELECT p.id FROM photos p WHERE p.owner_id = ?" + where, (1, *params)).fetchall()
    assert [r["id"] for r in rows] == [1]


def test_sidebar_counts_labels(conn):
    seed_tags(conn, VOCAB)
    _tagged(conn, 1, [("setting", "beach")])
    _tagged(conn, 2, [("setting", "beach")])
    groups = tag_sidebar(conn, owner_id=1, dimensions=["setting"], score_min=0.2, limit=12)
    setting = next(g for g in groups if g["dimension"] == "setting")
    assert ("beach", 2) in [(v["label"], v["count"]) for v in setting["values"]]
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_tag_filters.py -q`
Expected: FAIL — `ModuleNotFoundError: search.tags`.

- [ ] **Step 3: Write the tag filter module**

Create `search/tags.py`, mirroring `search/facets.py`'s `build_where`/`facet_counts` shape so the two read alike:

```python
import sqlite3


def parse_tag_filters(params: dict[str, str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for name, raw in params.items():
        if name.startswith("t_"):
            labels = [part for part in raw.split(",") if part]
            if labels:
                out[name[2:]] = labels
    return out


def tag_where(filters: dict[str, list[str]], score_min: float) -> tuple[str, list]:
    fragments: list[str] = []
    params: list = []
    for dimension, labels in filters.items():
        placeholders = ", ".join("?" for _ in labels)
        fragments.append(
            " AND EXISTS (SELECT 1 FROM photo_tags pt JOIN tags t ON t.id = pt.tag_id"
            f" WHERE pt.photo_id = p.id AND t.dimension = ? AND t.label IN ({placeholders})"
            " AND pt.score >= ?)"
        )
        params += [dimension, *labels, score_min]
    return "".join(fragments), params


def tag_sidebar(conn, owner_id, dimensions, score_min, limit) -> list[dict]:
    groups = []
    for dimension in dimensions:
        rows = conn.execute(
            "SELECT t.label AS label, count(*) AS n FROM photo_tags pt"
            " JOIN tags t ON t.id = pt.tag_id JOIN photos p ON p.id = pt.photo_id"
            " WHERE t.dimension = ? AND p.owner_id = ? AND pt.score >= ?"
            " GROUP BY t.label ORDER BY n DESC, t.label LIMIT ?",
            (dimension, owner_id, score_min, limit),
        ).fetchall()
        if rows:
            groups.append({
                "dimension": dimension,
                "values": [{"label": r["label"], "count": r["n"]} for r in rows],
            })
    return groups
```

- [ ] **Step 4: Wire into `/library`**

In `web/app.py`:
- In `fetch_page`, splice `tag_where(parse_tag_filters(params), score_min)` into the non-search branch alongside `build_where`, and keep `t_*` in `_query_string`'s keep-set so infinite scroll and the search box preserve tag filters.
- Switch the search branch to fusion: fetch semantic ids (`search_photos`, k≈200) and keyword ids (`keyword_search`, k≈200), fuse with `reciprocal_rank_fusion`, then, if any facet/tag filters are active, keep only fused ids that also satisfy those filters (run the existing filtered `BASE_SQL` restricted to the fused id set, ordered by fused rank). This realises §9's "facets filter first, fusion ranks what survives".
- Add a tag section to `_sidebar()` output via `tag_sidebar(conn, owner_id, list(VOCAB.dimensions), score_min, 12)`, passed to the template under a separate heading from EXIF facets.

Use a `settings`-level `tag_score_min` (add it to `config.py`, default `0.2`) so the threshold is configurable.

- [ ] **Step 5: Update the library template**

In `web/templates/library.html`, render the model-tag groups above (or below) the EXIF facet groups, as checkboxes named `t_<dimension>` with the same submit-on-change behaviour the EXIF facets use. Show the count next to each label.

- [ ] **Step 6: Update the search route test**

In `tests/test_web_search.py`, add a keyword-only match: a photo whose caption contains a proper noun with no semantic vector still surfaces for that word via fusion. Assert the existing semantic test still ranks the vector match first.

- [ ] **Step 7: Run the tests**

Run: `uv run pytest tests/test_tag_filters.py tests/test_web_search.py -q`
Expected: PASS.

- [ ] **Step 8: Run the whole suite and lint**

Run: `uv run pytest -q && uv run ruff check .`
Expected: PASS, clean.

- [ ] **Step 9: Checkpoint** — report the tag filter grammar, the fusion-with-filters flow, and the sidebar. Stop. Do not commit.

---

### Task 6: Tags on the photo detail page

Render the "AI data" panel §13 describes: tags grouped by dimension with their score and a source badge.

**Files:**
- Modify: `web/app.py` (`photo_detail` passes grouped tags)
- Modify: `web/templates/photo.html`
- Modify: `web/static/app.css`
- Modify: `tests/test_web_photo.py`

**Interfaces:**
- Produces: the `/photo/{id}` panel showing each tag as `label · score · source`, grouped by dimension, highest score first.

- [ ] **Step 1: Write the failing test**

In `tests/test_web_photo.py`, add:

```python
def test_photo_page_shows_model_tags_grouped_by_dimension(client):
    conn = client.app.state.context.conn
    base = _first_id(client)
    conn.execute("INSERT INTO tags(dimension, label) VALUES ('setting', 'beach')")
    tag_id = conn.execute("SELECT id FROM tags WHERE label = 'beach'").fetchone()["id"]
    conn.execute(
        "INSERT INTO photo_tags(photo_id, tag_id, score, source) VALUES (?, ?, 0.87, 'siglip')",
        (base, tag_id),
    )
    body = client.get(f"/photo/{base}").text
    assert "beach" in body
    assert "setting" in body
    assert "siglip" in body  # the source badge
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_web_photo.py::test_photo_page_shows_model_tags_grouped_by_dimension -q`
Expected: FAIL — the tag is not rendered.

- [ ] **Step 3: Pass grouped tags from the route**

In `photo_detail`, query the photo's tags joined to `tags`, ordered by dimension then score desc, and group them into `{dimension: [{label, score, source}]}`. Pass as `tags` to the template.

- [ ] **Step 4: Render them**

Replace the `{# Tags ... plan 04 #}` placeholder in `web/templates/photo.html` with a block that lists each dimension and its tags, each showing the label, the score (e.g. `0.87`), and a small source badge. Add minimal badge styling to `app.css`.

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_web_photo.py -q`
Expected: PASS.

- [ ] **Step 6: Run the whole suite and lint**

Run: `uv run pytest -q && uv run ruff check .`
Expected: PASS, clean.

- [ ] **Step 7: Checkpoint** — report the tag panel layout. Stop. Do not commit.

---

### Task 7: Verify against the real model and update the docs

- [ ] **Step 1: Index and browse against real SigLIP**

Run the app with the real embedder (no `IVMS777_USE_FAKE_EMBEDDER`), let the `embed` then `taxonomy` stages drain, and confirm on `/library`:
1. The tag sidebar fills with plausible labels and counts.
2. Filtering "beach" (setting) and "golden hour" (light) narrows the grid sensibly.
3. Searching a proper noun that appears only in a filename/OCR-free caption still finds it via keyword; searching "dogs in snow" finds visually-matching photos via semantic — and fusion ranks a photo strong on both at the top.
4. A photo's detail page shows its tags with scores and `siglip`/`pixel` badges.

Record what you saw and roughly how long the taxonomy pass took.

- [ ] **Step 2: Update the docs**

- `README.md`: note the taxonomy stage runs after embedding and adds tags + the sidebar; keyword+fusion search is live.
- `docs/design.md` §16: mark phase 2 as fully delivered (plan 03 embeddings/search + plan 04 taxonomy/keyword/fusion). Confirm §7/§9/§13 still match the implementation; fix either the code or the doc if they diverge.

- [ ] **Step 3: Checkpoint** — report the manual results and stop. Do not commit.

---

## What plan 04 delivers

The library gets a brain and a filter rail. Every photo now wears model-assigned tags across ten dimensions — subject, setting, vibe, light, palette, and the rest — visible in a `/library` sidebar with live counts, and combinable ("beach" AND "golden hour"). Search stops being semantics-only: type a proper noun and FTS5 finds it in the caption or tags; type a scene and SigLIP finds the look; reciprocal-rank fusion blends the two so a photo strong on both wins. Open any photo and see exactly which tags a model gave it, how sure it was, and whether the signal came from SigLIP or from pixel statistics.

It all runs as one more resumable job stage that drains after embedding — so the Jetson never holds two models at once — and the whole thing tests offline against the fake embedder and generated images.

**Not yet working:** captions (plan 05) and the per-photo AI title/description that fills the rest of the `/photo` AI panel; caption-driven vocabulary mining (§7.1); the query planner and parsed-filter chips (plan 06); chat (plan 08). The `vlm` tag source stays empty until captions land.

## Following plans

| Plan | Spec phase | Delivers |
|---|---|---|
| 05 | 3 | Caption stage against the inference service; per-photo AI title + description; captions in search and the `/photo` AI panel |
| 06 | 4 | Query planner, parsed-filter chips, caption vocabulary mining with tag suggestions |
| 07 | 5 | Memories — agentic, persisted albums (see `docs/plans/07-memories.md`) |
| 08 | 6 | Ask-your-library chat with streaming and citations |
| 09 | 7 | Stage 2 — layouts, `/api/manifest`, `/export`, and the `ivms777-sync` CLI |
