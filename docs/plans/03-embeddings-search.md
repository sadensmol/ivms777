# Photo Library Organizer — Plan 03: Embeddings and Semantic Search

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the library meaning. Every photo gets a SigLIP 2 embedding; a search box finds photos by what they depict ("dogs in snow") with no matching caption; each photo has a detail page with a "similar photos" strip. All local, no external API.

**Architecture:** SigLIP 2 runs in-process behind an `Embedder` protocol with a deterministic fake, so the whole pipeline and search test offline in milliseconds. A new `embed` job stage writes each image vector into the existing `photo_vec` (`sqlite-vec`) table. A text query goes through the same encoder and does KNN over that table; "similar" does KNN against a photo's own vector.

**Tech Stack:** Python 3.12, PyTorch, Hugging Face `transformers`, `sqlite-vec`, NumPy, FastAPI, Jinja2, HTMX. Existing job queue and storage unchanged.

**Spec:** `docs/design.md` — sections 4 (models), 5 (sqlite-vec), 8 (ingest stages), 9 (retrieval), 12 UI (`/photo`).

**Builds on:** plan 02 (upload ingest). The `embed` stage drains after `thumbnail`, exactly as §8 describes stages draining library-wide in order.

**Covers:** the embedding and semantic-search slice of spec phase 2. Taxonomy scoring (zero-shot tags, `vocab.yaml`, the tag sidebar) and keyword/fusion search are plan 04.

## Global Constraints

- Python 3.12. Dependencies via `uv` with a committed `uv.lock`.
- **Never run `git commit`/`git add`.** The user commits. Every task ends at a checkpoint.
- Every user-scoped query filters on `owner_id` (constant `settings.owner_id`).
- Tests must not download model weights or hit the network. The real SigLIP test is marked `slow` and deselected by default.
- The full fast suite passes at the end of every task: `uv run pytest -q`.
- SigLIP 2 `so400m-patch14-384` produces **1152-dim** embeddings — matches the existing `photo_vec(embedding float[1152])`.
- Vectors are stored **L2-normalized**, so cosine similarity is a dot product and `sqlite-vec` L2 distance is monotonic in it.

---

### Task 1: Embedder protocol and deterministic fake

**Files:**
- Create: `ivms777/embedding/__init__.py`
- Create: `ivms777/embedding/base.py`
- Create: `ivms777/embedding/fakes.py`
- Create: `ivms777/embedding/vectors.py`
- Create: `tests/test_embedding_fake.py`

**Interfaces:**
- Produces:
  - `ivms777.embedding.base.EMBED_DIM = 1152`
  - `ivms777.embedding.base.Embedder` protocol: `embed_images(images: list[Image.Image]) -> list[list[float]]`, `embed_texts(texts: list[str]) -> list[list[float]]`
  - `ivms777.embedding.vectors.to_blob(vector: list[float]) -> bytes` and `from_blob(blob: bytes) -> list[float]` (float32, the `sqlite-vec` wire format)
  - `ivms777.embedding.vectors.l2_normalize(vector: list[float]) -> list[float]`
  - `ivms777.embedding.fakes.FakeEmbedder` — a hash-derived unit vector per input, so identical inputs match exactly and similarity is reproducible

- [ ] **Step 1: Write the failing test**

Create `tests/test_embedding_fake.py`:

```python
import math

from PIL import Image

from ivms777.embedding.base import EMBED_DIM
from ivms777.embedding.fakes import FakeEmbedder
from ivms777.embedding.vectors import from_blob, l2_normalize, to_blob


def test_blob_round_trips_as_float32():
    vector = l2_normalize([0.1 * i for i in range(EMBED_DIM)])
    restored = from_blob(to_blob(vector))
    assert len(restored) == EMBED_DIM
    assert all(abs(a - b) < 1e-6 for a, b in zip(vector, restored))


def test_l2_normalize_gives_a_unit_vector():
    vector = l2_normalize([3.0, 4.0] + [0.0] * (EMBED_DIM - 2))
    assert abs(math.sqrt(sum(x * x for x in vector)) - 1.0) < 1e-6


def test_fake_is_deterministic_and_unit_length():
    fake = FakeEmbedder()
    a = fake.embed_texts(["beach"])[0]
    b = fake.embed_texts(["beach"])[0]
    assert a == b
    assert abs(math.sqrt(sum(x * x for x in a)) - 1.0) < 1e-6
    assert len(a) == EMBED_DIM


def test_fake_separates_different_inputs():
    fake = FakeEmbedder()
    beach = fake.embed_texts(["beach"])[0]
    keyboard = fake.embed_texts(["keyboard"])[0]
    dot = sum(x * y for x, y in zip(beach, keyboard))
    assert dot < 0.99  # not identical


def test_fake_images_are_keyed_by_pixels():
    fake = FakeEmbedder()
    red = Image.new("RGB", (8, 8), "red")
    blue = Image.new("RGB", (8, 8), "blue")
    assert fake.embed_images([red])[0] == fake.embed_images([red])[0]
    assert fake.embed_images([red])[0] != fake.embed_images([blue])[0]
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_embedding_fake.py -q`
Expected: FAIL — `ModuleNotFoundError: ivms777.embedding`.

- [ ] **Step 3: Write the vector helpers**

Create `ivms777/embedding/__init__.py` (empty) and `ivms777/embedding/vectors.py`:

```python
import struct
from math import sqrt


def l2_normalize(vector: list[float]) -> list[float]:
    norm = sqrt(sum(x * x for x in vector))
    if norm == 0.0:
        return list(vector)
    return [x / norm for x in vector]


def to_blob(vector: list[float]) -> bytes:
    """Pack a vector as little-endian float32 — the sqlite-vec wire format."""
    return struct.pack(f"<{len(vector)}f", *vector)


def from_blob(blob: bytes) -> list[float]:
    return list(struct.unpack(f"<{len(blob) // 4}f", blob))
```

- [ ] **Step 4: Write the protocol**

Create `ivms777/embedding/base.py`:

```python
from typing import Protocol

from PIL import Image

EMBED_DIM = 1152


class Embedder(Protocol):
    """Maps images and text into one shared vector space.

    Both sides return L2-normalized vectors of length EMBED_DIM, so a dot product
    is cosine similarity and the same query text can rank images.
    """

    def embed_images(self, images: list[Image.Image]) -> list[list[float]]: ...
    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...
```

- [ ] **Step 5: Write the fake**

Create `ivms777/embedding/fakes.py`:

```python
import hashlib

from PIL import Image

from ivms777.embedding.base import EMBED_DIM
from ivms777.embedding.vectors import l2_normalize


def _vector_from_bytes(seed: bytes) -> list[float]:
    """A stable unit vector derived from a byte seed.

    Deterministic and reproducible, so tests never touch a model, yet identical
    inputs collide exactly and different inputs almost never do.
    """
    raw = bytearray()
    counter = 0
    while len(raw) < EMBED_DIM * 2:
        raw += hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
        counter += 1
    values = [
        int.from_bytes(raw[i : i + 2], "big") / 65535.0 - 0.5
        for i in range(0, EMBED_DIM * 2, 2)
    ]
    return l2_normalize(values)


class FakeEmbedder:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [_vector_from_bytes(text.encode("utf-8")) for text in texts]

    def embed_images(self, images: list[Image.Image]) -> list[list[float]]:
        return [_vector_from_bytes(self._image_seed(image)) for image in images]

    @staticmethod
    def _image_seed(image: Image.Image) -> bytes:
        small = image.convert("RGB").resize((16, 16))
        return small.tobytes()
```

- [ ] **Step 6: Run to verify it passes**

Run: `uv run pytest tests/test_embedding_fake.py -q`
Expected: PASS.

- [ ] **Step 7: Run the whole suite**

Run: `uv run pytest -q`
Expected: PASS.

- [ ] **Step 8: Checkpoint** — report the protocol and fake. Stop. Do not commit.

---

### Task 2: The embed stage

Wire embedding into the job queue so every uploaded photo gets a vector in `photo_vec`, draining after `thumbnail` exactly as the existing stages do.

**Files:**
- Create: `ivms777/embedding/store.py`
- Create: `ivms777/ingest/embed.py`
- Create: `tests/test_embed_stage.py`
- Modify: `ivms777/web/app.py`
- Modify: `ivms777/ingest/cli.py`

**Interfaces:**
- Consumes: `Embedder` (task 1); `to_blob`, `from_blob` (task 1); `LocalStorage`; `enqueue`, `drain`, `thumbnail_handler`.
- Produces:
  - `ivms777.embedding.store.write_vector(conn, photo_id, vector)` and `read_vector(conn, photo_id) -> list[float] | None`
  - `ivms777.embedding.store.knn(conn, owner_id, vector, k, exclude_id=None) -> list[tuple[int, float]]` — `(photo_id, distance)`, nearest first
  - `ivms777.ingest.embed.embed_handler(originals, embedder, model_name) -> StageHandler`

- [ ] **Step 1: Write the failing test**

Create `tests/test_embed_stage.py`:

```python
import pytest

from ivms777.embedding.fakes import FakeEmbedder
from ivms777.embedding.store import knn, read_vector, write_vector
from ivms777.embedding.vectors import l2_normalize
from ivms777.ingest.embed import embed_handler
from ivms777.ingest.jobs import enqueue, stage_counts
from ivms777.ingest.worker import drain
from ivms777.storage.keys import content_key
from ivms777.storage.local import LocalStorage
from tests.factories import add_photo
from tests.fixtures import make_jpeg


def test_write_and_read_round_trip(conn):
    add_photo(conn, photo_id=1, content_hash="h")
    vector = l2_normalize([0.5] * 1152)
    write_vector(conn, 1, vector)
    restored = read_vector(conn, 1)
    assert restored is not None
    assert all(abs(a - b) < 1e-6 for a, b in zip(vector, restored))


def test_knn_orders_by_distance_and_can_exclude_self(conn):
    for pid, seed in ((1, "a"), (2, "b"), (3, "c")):
        add_photo(conn, photo_id=pid, content_hash=seed)
        write_vector(conn, pid, FakeEmbedder().embed_texts([seed])[0])
    query = read_vector(conn, 1)
    hits = knn(conn, owner_id=1, vector=query, k=10)
    assert hits[0][0] == 1  # a photo is nearest to its own vector
    without_self = knn(conn, owner_id=1, vector=query, k=10, exclude_id=1)
    assert 1 not in [pid for pid, _ in without_self]


def test_knn_is_owner_scoped(conn):
    add_photo(conn, photo_id=1, owner_id=1, content_hash="a")
    add_photo(conn, photo_id=2, owner_id=2, content_hash="b")
    vec = FakeEmbedder().embed_texts(["a"])[0]
    write_vector(conn, 1, vec)
    write_vector(conn, 2, vec)
    hits = knn(conn, owner_id=1, vector=vec, k=10)
    assert [pid for pid, _ in hits] == [1]


def test_embed_handler_populates_the_vector_table(conn, tmp_path):
    originals = LocalStorage(tmp_path / "originals")
    key = content_key("ff" * 32, ".jpg")
    make_jpeg(originals.local_path(key))
    add_photo(conn, photo_id=1, content_hash="ff" * 32)
    enqueue(conn, 1, "embed")

    drain(conn, {"embed": embed_handler(originals, FakeEmbedder(), "fake-model")})

    assert read_vector(conn, 1) is not None
    assert stage_counts(conn, "embed")["done"] == 1
    assert conn.execute(
        "SELECT embedding_model FROM photos WHERE id = 1"
    ).fetchone()["embedding_model"] == "fake-model"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_embed_stage.py -q`
Expected: FAIL — `ModuleNotFoundError: ivms777.embedding.store`.

- [ ] **Step 3: Write the vector store**

Create `ivms777/embedding/store.py`:

```python
import sqlite3

from ivms777.embedding.vectors import from_blob, to_blob


def write_vector(conn: sqlite3.Connection, photo_id: int, vector: list[float]) -> None:
    # rowid == photos.id, so a delete-then-insert keeps re-embeds idempotent.
    conn.execute("DELETE FROM photo_vec WHERE rowid = ?", (photo_id,))
    conn.execute(
        "INSERT INTO photo_vec(rowid, embedding) VALUES (?, ?)", (photo_id, to_blob(vector))
    )


def read_vector(conn: sqlite3.Connection, photo_id: int) -> list[float] | None:
    row = conn.execute(
        "SELECT embedding FROM photo_vec WHERE rowid = ?", (photo_id,)
    ).fetchone()
    return from_blob(row["embedding"]) if row is not None else None


def knn(
    conn: sqlite3.Connection,
    owner_id: int,
    vector: list[float],
    k: int,
    exclude_id: int | None = None,
) -> list[tuple[int, float]]:
    """Nearest photo ids to `vector`, owner-scoped, nearest first.

    sqlite-vec's KNN is not itself owner-aware, so it over-fetches and the join to
    `photos` applies the owner filter. At this scale that is comfortably fast.
    """
    rows = conn.execute(
        "SELECT v.rowid AS photo_id, v.distance AS distance"
        " FROM photo_vec v JOIN photos p ON p.id = v.rowid"
        " WHERE v.embedding MATCH ? AND k = ? AND p.owner_id = ?",
        (to_blob(vector), k + (1 if exclude_id else 0) + 32, owner_id),
    ).fetchall()
    hits = [(row["photo_id"], row["distance"]) for row in rows if row["photo_id"] != exclude_id]
    return hits[:k]
```

- [ ] **Step 4: Write the embed stage handler**

Create `ivms777/ingest/embed.py`:

```python
import sqlite3
from pathlib import Path

from PIL import Image, ImageOps

from ivms777.embedding.base import Embedder
from ivms777.embedding.store import write_vector
from ivms777.embedding.vectors import l2_normalize
from ivms777.ingest.worker import StageHandler
from ivms777.storage.base import Storage


def embed_handler(originals: Storage, embedder: Embedder, model_name: str) -> StageHandler:
    def handle(conn: sqlite3.Connection, photo_id: int) -> None:
        row = conn.execute(
            "SELECT storage_key FROM photos WHERE id = ?", (photo_id,)
        ).fetchone()
        source: Path | None = originals.local_path(row["storage_key"])
        if source is None or not source.is_file():
            raise FileNotFoundError(row["storage_key"])
        with Image.open(source) as image:
            image.load()
            upright = ImageOps.exif_transpose(image).convert("RGB")
        vector = l2_normalize(embedder.embed_images([upright])[0])
        write_vector(conn, photo_id, vector)
        conn.execute(
            "UPDATE photos SET embedding_model = ? WHERE id = ?", (model_name, photo_id)
        )

    return handle
```

- [ ] **Step 5: Run the stage tests**

Run: `uv run pytest tests/test_embed_stage.py -q`
Expected: PASS.

- [ ] **Step 6: Enqueue `embed` on receipt and drain it in the app**

In `ivms777/ingest/receive.py`, after the thumbnail enqueue, also queue embedding:

```python
    enqueue(conn, photo_id, "thumbnail")
    enqueue(conn, photo_id, "embed")
```

Update the receive tests that assert on the thumbnail queue to also allow the embed row: in `tests/test_receive.py`, `test_receive_queues_a_thumbnail` stays as-is (it only checks `thumbnail`), and add:

```python
def test_receive_queues_an_embed(conn, originals):
    upload_id = add_upload(conn)
    data = jpeg_bytes()
    receive(
        conn, originals, owner_id=1, upload_id=upload_id,
        rel_path="a.jpg", declared_hash=sha(data), data=data,
    )
    assert stage_counts(conn, "embed")["pending"] == 1
```

In `ivms777/web/app.py`, `drain_now` must build the embedder once and drain both stages in order. Add near the top imports:

```python
from ivms777.embedding.fakes import FakeEmbedder
from ivms777.ingest.embed import embed_handler
```

and replace the body of `drain_now`:

```python
    def drain_now() -> None:
        ctx = context()
        embedder, model_name = ctx.settings.build_embedder()
        drain(
            ctx.conn,
            {
                "thumbnail": thumbnail_handler(
                    ctx.originals, ctx.derived,
                    ctx.settings.thumb_grid_px, ctx.settings.thumb_detail_px,
                ),
                "embed": embed_handler(ctx.originals, embedder, model_name),
            },
        )
```

`build_embedder` is added in step 7. Until then the app import of the real SigLIP would force a torch import at startup, so the settings method is what defers it.

- [ ] **Step 7: Add an embedder factory to settings, defaulting to the fake**

The real model arrives in task 3. For now `build_embedder` returns the fake unless a real one is registered, so nothing imports torch yet.

In `ivms777/config.py`, add:

```python
    embed_model_name: str = "siglip2-so400m-patch14-384"
    use_fake_embedder: bool = False

    def build_embedder(self):
        """Return (embedder, model_name).

        Defaults to the real SigLIP; tests and the fast path set
        IVMS777_USE_FAKE_EMBEDDER=1. Imports are local so torch loads only when
        the real model is actually built.
        """
        if self.use_fake_embedder:
            from ivms777.embedding.fakes import FakeEmbedder

            return FakeEmbedder(), "fake"
        from ivms777.embedding.siglip import SiglipEmbedder

        return SiglipEmbedder(self.embed_model_name, self.embed_device), self.embed_model_name
```

Because `ivms777/embedding/siglip.py` does not exist until task 3, **every test and the local dev run must set `IVMS777_USE_FAKE_EMBEDDER=1`**. Add it to the `settings` fixture in `tests/conftest.py`:

```python
@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path, use_fake_embedder=True)
```

and to `tests/test_config.py` where `Settings` is built directly without it, add `use_fake_embedder=True` is **not** needed there (those tests never call `build_embedder`). Leave them.

Remove the now-unused `FakeEmbedder`/`embed_handler` direct imports from `app.py` if `drain_now` builds via `settings.build_embedder()` — it does, so delete the two imports added in step 6 and keep only what the factory pulls in. Re-run ruff to confirm no unused imports.

- [ ] **Step 8: Make the worker CLI drain embeddings too**

Replace `ivms777/ingest/cli.py`'s `drain` call so it mirrors `drain_now`:

```python
import time

from ivms777.config import get_settings
from ivms777.ingest.embed import embed_handler
from ivms777.ingest.worker import drain, thumbnail_handler
from ivms777.web.deps import build_context

POLL_SECONDS = 10


def main() -> None:
    context = build_context(get_settings())
    settings = context.settings
    embedder, model_name = settings.build_embedder()
    handlers = {
        "thumbnail": thumbnail_handler(
            context.originals, context.derived,
            settings.thumb_grid_px, settings.thumb_detail_px,
        ),
        "embed": embed_handler(context.originals, embedder, model_name),
    }
    while True:
        drain(context.conn, handlers)
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
```

- [ ] **Step 9: Run the whole suite**

Run: `uv run pytest -q`
Expected: PASS. Every web/upload test now also embeds via the fake on `finish`; assert counts are unaffected because embedding adds no tiles.

- [ ] **Step 10: Checkpoint** — report the stage, the KNN contract, and the fake-by-default switch. Stop. Do not commit.

---

### Task 3: The real SigLIP embedder

The only task that touches PyTorch. It sits behind the protocol; the fake keeps every other test offline. Its own test is marked `slow` and skipped by default.

**Files:**
- Modify: `pyproject.toml`
- Create: `ivms777/embedding/siglip.py`
- Create: `tests/test_siglip_real.py`

**Interfaces:**
- Produces: `ivms777.embedding.siglip.SiglipEmbedder(model_name: str, device: str)` implementing `Embedder`, returning L2-normalized 1152-dim vectors.

- [ ] **Step 1: Add the dependencies**

```bash
uv add torch transformers numpy
```

On Apple Silicon this pulls the CPU/MPS PyTorch wheel. Commit the updated `uv.lock` later with the rest.

- [ ] **Step 2: Write the slow test**

Create `tests/test_siglip_real.py`. This is the §14 calibration test — a beach photo must rank above a keyboard photo for the query "a photo of a beach":

```python
import pytest
from PIL import Image

pytestmark = pytest.mark.slow


def _solid(color):
    return Image.new("RGB", (384, 384), color)


def test_beach_query_ranks_a_beach_above_a_keyboard():
    from ivms777.embedding.siglip import SiglipEmbedder

    embedder = SiglipEmbedder("siglip2-so400m-patch14-384", "cpu")
    # Two real photos rather than solids make this a meaningful check.
    beach = Image.open("tests/data/beach.jpg").convert("RGB")
    keyboard = Image.open("tests/data/keyboard.jpg").convert("RGB")
    images = embedder.embed_images([beach, keyboard])
    query = embedder.embed_texts(["a photo of a beach"])[0]

    def dot(a, b):
        return sum(x * y for x, y in zip(a, b))

    assert dot(query, images[0]) > dot(query, images[1])
```

Note: this test needs two real JPEGs at `tests/data/`. If they are not present, mark it `skip` with a clear reason rather than committing binaries — the fast suite never runs it. Document in the test how to supply them.

- [ ] **Step 3: Verify it is deselected by default**

Run: `uv run pytest -q`
Expected: PASS, with `test_siglip_real` deselected (the `slow` marker is filtered by the default config in `pyproject.toml`).

If `slow` is not deselected by default, add to `[tool.pytest.ini_options]`:

```toml
addopts = "-m 'not slow'"
```

- [ ] **Step 4: Write the SigLIP embedder**

Create `ivms777/embedding/siglip.py`:

```python
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from ivms777.embedding.base import EMBED_DIM

# Hugging Face id for SigLIP 2 so400m/14 at 384px. The short name in config maps
# here so config stays terse and the actual repo id lives in one place.
_HF_ID = "google/siglip2-so400m-patch14-384"


class SiglipEmbedder:
    def __init__(self, model_name: str, device: str) -> None:
        self.device = device
        self.model = AutoModel.from_pretrained(_HF_ID).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(_HF_ID)

    @torch.no_grad()
    def embed_images(self, images: list[Image.Image]) -> list[list[float]]:
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        features = self.model.get_image_features(**inputs)
        return self._normalized(features)

    @torch.no_grad()
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        inputs = self.processor(
            text=texts, return_tensors="pt", padding="max_length", truncation=True
        ).to(self.device)
        features = self.model.get_text_features(**inputs)
        return self._normalized(features)

    @staticmethod
    def _normalized(features: torch.Tensor) -> list[list[float]]:
        array = features.float().cpu().numpy()
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        unit = array / norms
        assert unit.shape[1] == EMBED_DIM, f"expected {EMBED_DIM}, got {unit.shape[1]}"
        return unit.tolist()
```

- [ ] **Step 5: Run the fast suite again**

Run: `uv run pytest -q`
Expected: PASS (real SigLIP still deselected; nothing else imports torch).

- [ ] **Step 6: Checkpoint** — report the deps added, the model id, and that the fast suite never loads torch. Stop. Do not commit.

---

### Task 4: Semantic search and similar photos

**Files:**
- Create: `ivms777/search/semantic.py`
- Create: `tests/test_semantic_search.py`
- Modify: `ivms777/web/app.py`

**Interfaces:**
- Consumes: `knn`, `read_vector` (task 2); `Embedder`; `AppContext`.
- Produces:
  - `ivms777.search.semantic.search_photos(conn, embedder, owner_id, query, k) -> list[int]` — photo ids, best first
  - `ivms777.search.semantic.similar_photos(conn, owner_id, photo_id, k) -> list[int]`

- [ ] **Step 1: Write the failing test**

Create `tests/test_semantic_search.py`:

```python
import pytest

from ivms777.embedding.fakes import FakeEmbedder
from ivms777.embedding.store import write_vector
from ivms777.search.semantic import search_photos, similar_photos
from tests.factories import add_photo


@pytest.fixture
def library(conn):
    # Give each photo the fake's embedding for a distinct word, so a text query
    # for that word retrieves it exactly. This tests the plumbing; real ranking
    # quality is the slow SigLIP test.
    fake = FakeEmbedder()
    for pid, word in ((1, "beach"), (2, "keyboard"), (3, "mountain")):
        add_photo(conn, photo_id=pid, content_hash=word, thumb_key=f"{word}.jpg")
        write_vector(conn, pid, fake.embed_texts([word])[0])
    return conn


def test_search_finds_the_photo_whose_vector_matches_the_query(library):
    ids = search_photos(library, FakeEmbedder(), owner_id=1, query="beach", k=3)
    assert ids[0] == 1


def test_search_returns_at_most_k(library):
    assert len(search_photos(library, FakeEmbedder(), owner_id=1, query="beach", k=2)) == 2


def test_similar_excludes_the_photo_itself(library):
    ids = similar_photos(library, owner_id=1, photo_id=1, k=5)
    assert 1 not in ids


def test_similar_of_an_unembedded_photo_is_empty(library):
    add_photo(library, photo_id=99, content_hash="novec")
    assert similar_photos(library, owner_id=1, photo_id=99, k=5) == []
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_semantic_search.py -q`
Expected: FAIL — `ModuleNotFoundError: ivms777.search.semantic`.

- [ ] **Step 3: Write the search module**

Create `ivms777/search/semantic.py`:

```python
import sqlite3

from ivms777.embedding.base import Embedder
from ivms777.embedding.store import knn, read_vector
from ivms777.embedding.vectors import l2_normalize


def search_photos(
    conn: sqlite3.Connection, embedder: Embedder, owner_id: int, query: str, k: int
) -> list[int]:
    """Photo ids best matching a natural-language query, best first."""
    if not query.strip():
        return []
    vector = l2_normalize(embedder.embed_texts([query])[0])
    return [photo_id for photo_id, _ in knn(conn, owner_id, vector, k)]


def similar_photos(
    conn: sqlite3.Connection, owner_id: int, photo_id: int, k: int
) -> list[int]:
    """Photo ids nearest to a given photo, excluding itself. Empty if unembedded."""
    vector = read_vector(conn, photo_id)
    if vector is None:
        return []
    return [pid for pid, _ in knn(conn, owner_id, vector, k, exclude_id=photo_id)]
```

- [ ] **Step 4: Run the search tests**

Run: `uv run pytest tests/test_semantic_search.py -q`
Expected: PASS.

- [ ] **Step 5: Wire search into `/library`**

In `ivms777/web/app.py`, the library route already reads query params. Add a `q` search box that, when present, replaces the facet-filtered SQL page with a semantic ranking. Keep facet filtering when `q` is empty.

In `fetch_page`, branch on `q`:

```python
    def fetch_page(offset: int, params: dict[str, str]) -> list:
        ctx = context()
        query = params.get("q", "").strip()
        if query:
            embedder, _ = ctx.settings.build_embedder()
            ids = search_photos(ctx.conn, embedder, ctx.settings.owner_id, query, k=200)
            page_ids = ids[offset : offset + ctx.settings.page_size]
            if not page_ids:
                return []
            placeholders = ", ".join("?" for _ in page_ids)
            order = " ".join(f"WHEN {pid} THEN {rank}" for rank, pid in enumerate(page_ids))
            rows = ctx.conn.execute(
                f"SELECT p.id, p.caption, p.shot_at,"
                f" (SELECT count(*) - 1 FROM photo_sources s WHERE s.photo_id = p.id) AS dupe_count"
                f" FROM photos p WHERE p.owner_id = ? AND p.id IN ({placeholders})"
                f" ORDER BY CASE p.id {order} END",
                (ctx.settings.owner_id, *page_ids),
            )
            return list(rows)
        where, where_params = build_where(parse_filters(params))
        order, order_params = _order_clause(params.get("sort"))
        sql = BASE_SQL + where + order + " LIMIT ? OFFSET ?"
        bound = [
            ctx.settings.owner_id, *where_params, *order_params,
            ctx.settings.page_size, offset,
        ]
        return list(ctx.conn.execute(sql, bound))
```

Import at the top: `from ivms777.search.semantic import search_photos`.

In `_query_string`, keep `q` alongside the facet keys so infinite-scroll preserves the search:

```python
    def _query_string(params: dict[str, str]) -> str:
        keep = {
            k: v for k, v in params.items()
            if k.startswith(("f_", "n_")) or k in ("sort", "q")
        }
        return urlencode(keep)
```

- [ ] **Step 6: Add the search box to the template**

In `ivms777/web/templates/library.html`, above the grid, add a GET form that submits `q`:

```html
<form class="search" method="get" action="/library">
  <input type="search" name="q" value="{{ active.get('q', '') }}"
         placeholder="Search by meaning — “dogs in snow”, “birthday cake”…">
  <button type="submit">Search</button>
  {% if active.get('q') %}<a class="clear-search" href="/library">clear</a>{% endif %}
</form>
```

`active` is already passed to the library template. Confirm the library route passes `active=params`; it does.

- [ ] **Step 7: Write a route test for search**

Add `tests/test_web_search.py`:

```python
import pytest
from fastapi.testclient import TestClient

from ivms777.embedding.fakes import FakeEmbedder
from ivms777.embedding.store import write_vector
from ivms777.web.app import create_app
from tests.factories import add_photo


@pytest.fixture
def client(settings):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    app = create_app(settings)
    fake = FakeEmbedder()
    for pid, word in ((1, "beach"), (2, "keyboard")):
        add_photo(app.state.context.conn, photo_id=pid, content_hash=word, thumb_key=f"{word}.jpg")
        write_vector(app.state.context.conn, pid, fake.embed_texts([word])[0])
    with TestClient(app) as test_client:
        yield test_client


def test_search_box_is_on_the_library_page(client):
    assert 'name="q"' in client.get("/library").text


def test_search_returns_the_matching_photo_first(client):
    body = client.get("/library?q=beach").text
    assert '/thumb/1' in body
    # the keyboard photo ranks lower; the first tile is the beach photo
    assert body.index('/thumb/1') < body.index('/thumb/2')
```

- [ ] **Step 8: Run the whole suite**

Run: `uv run pytest -q`
Expected: PASS.

- [ ] **Step 9: Checkpoint** — report the search contract and how `q` overrides facet filtering. Stop. Do not commit.

---

### Task 5: The full-screen photo view

The last 404 in the grid becomes a full-screen page: the image fills the viewport, and a panel carries everything known about the photo — EXIF, the AI data (embedding status now; tags in plan 04), every source path with the wasted-space total, and a "similar photos" strip. There is no separate duplicates screen, so this page is where duplicate paths are seen.

**Files:**
- Create: `ivms777/web/templates/photo.html`
- Modify: `ivms777/web/app.py`
- Create: `tests/test_web_photo.py`

**Interfaces:**
- Consumes: `similar_photos` (task 4); the detail thumbnail (`thumb_detail_px`) already written by the thumbnail stage.
- Produces:
  - `GET /photo/{photo_id}` → the detail page, 404 for an unknown or other-owner photo
  - `GET /thumb/{photo_id}?size=detail` → the 1600px image (grid stays the default 320px)

- [ ] **Step 1: Write the failing test**

Create `tests/test_web_photo.py`:

```python
import pytest
from fastapi.testclient import TestClient

from ivms777.embedding.fakes import FakeEmbedder
from ivms777.embedding.store import write_vector
from ivms777.ingest.receive import receive
from ivms777.ingest.worker import drain, thumbnail_handler
from ivms777.web.app import create_app
from tests.factories import add_photo, add_upload
from tests.fixtures import jpeg_bytes_with_exif, sha


@pytest.fixture
def client(settings):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    app = create_app(settings)
    ctx = app.state.context
    upload_id = add_upload(ctx.conn)
    data = jpeg_bytes_with_exif(model="X-T5")
    receive(
        ctx.conn, ctx.originals, owner_id=1, upload_id=upload_id,
        rel_path="Pictures/holiday/a.jpg", declared_hash=sha(data), data=data,
    )
    drain(ctx.conn, {"thumbnail": thumbnail_handler(ctx.originals, ctx.derived, 320, 1600)})
    with TestClient(app) as test_client:
        yield test_client


def _first_id(client):
    return client.app.state.context.conn.execute("SELECT id FROM photos LIMIT 1").fetchone()["id"]


def test_photo_page_shows_exif_and_source_path(client):
    body = client.get(f"/photo/{_first_id(client)}").text
    assert "X-T5" in body
    assert "Pictures/holiday/a.jpg" in body


def test_photo_page_shows_every_duplicate_path_and_wasted_space(client):
    ctx = client.app.state.context
    base = _first_id(client)
    upload_id = add_upload(ctx.conn)
    # a second source path for the same photo, plus a byte size to waste
    ctx.conn.execute("UPDATE photos SET bytes = 1000 WHERE id = ?", (base,))
    ctx.conn.execute(
        "INSERT INTO photo_sources(photo_id, upload_id, rel_path, filename)"
        " VALUES (?, ?, 'Backup/a.jpg', 'a.jpg')",
        (base, upload_id),
    )
    body = client.get(f"/photo/{base}").text
    assert "Pictures/holiday/a.jpg" in body
    assert "Backup/a.jpg" in body


def test_photo_page_serves_the_detail_image(client):
    photo_id = _first_id(client)
    assert client.get(f"/photo/{photo_id}").status_code == 200
    detail = client.get(f"/thumb/{photo_id}?size=detail")
    assert detail.status_code == 200
    assert detail.headers["content-type"] == "image/jpeg"


def test_unknown_photo_is_404(client):
    assert client.get("/photo/99999").status_code == 404


def test_similar_strip_lists_other_photos(client):
    ctx = client.app.state.context
    fake = FakeEmbedder()
    base = _first_id(client)
    write_vector(ctx.conn, base, fake.embed_texts(["a"])[0])
    other = add_photo(ctx.conn, content_hash="bb" * 32, thumb_key="bb.jpg")
    write_vector(ctx.conn, other, fake.embed_texts(["a"])[0])  # identical → nearest
    body = client.get(f"/photo/{base}").text
    assert f"/thumb/{other}" in body
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_web_photo.py -q`
Expected: FAIL — 404 on `/photo/{id}`.

- [ ] **Step 3: Let `/thumb` serve the detail size**

In `ivms777/web/app.py`, the thumbnail route currently serves the grid key. Add a `size` query param that picks the detail thumbnail. The thumbnail stage already writes both sizes via `thumb_key(hash, 1600)`; derive the detail key from the stored grid key by swapping the size suffix, or recompute from `content_hash`. Recompute is clearer:

```python
    @app.get("/thumb/{photo_id}")
    def thumb(photo_id: int, size: str = "grid") -> Response:
        ctx = context()
        row = ctx.conn.execute(
            "SELECT content_hash, thumb_key FROM photos WHERE id = ? AND owner_id = ?",
            (photo_id, ctx.settings.owner_id),
        ).fetchone()
        if row is None or row["thumb_key"] is None:
            raise HTTPException(status_code=404)
        px = ctx.settings.thumb_detail_px if size == "detail" else ctx.settings.thumb_grid_px
        from ivms777.ingest.thumbs import thumb_key as _thumb_key
        key = _thumb_key(row["content_hash"], px)
        if not ctx.derived.exists(key):
            raise HTTPException(status_code=404)
        return Response(ctx.derived.read(key), media_type="image/jpeg")
```

- [ ] **Step 4: Add the detail route**

In `ivms777/web/app.py`:

```python
    @app.get("/photo/{photo_id}", response_class=HTMLResponse)
    def photo_detail(request: Request, photo_id: int) -> HTMLResponse:
        ctx = context()
        photo = ctx.conn.execute(
            "SELECT * FROM photos WHERE id = ? AND owner_id = ?",
            (photo_id, ctx.settings.owner_id),
        ).fetchone()
        if photo is None:
            raise HTTPException(status_code=404)
        sources = list(ctx.conn.execute(
            "SELECT rel_path FROM photo_sources WHERE photo_id = ? ORDER BY rel_path",
            (photo_id,),
        ))
        facets = list(ctx.conn.execute(
            "SELECT key, value_text, value_num FROM photo_facets WHERE photo_id = ? ORDER BY key",
            (photo_id,),
        ))
        # Redundant copies past the first waste (n-1) x the file size.
        wasted = (photo["bytes"] or 0) * max(0, len(sources) - 1)
        similar = similar_photos(ctx.conn, ctx.settings.owner_id, photo_id, k=12)
        return templates.TemplateResponse(
            request,
            "photo.html",
            {
                "photo": photo, "sources": sources, "facets": facets,
                "similar": similar, "wasted_bytes": wasted,
                "embedded": photo["embedding_model"] is not None,
            },
        )
```

Import at the top: `from ivms777.search.semantic import similar_photos, search_photos` (extend the task-4 import).

- [ ] **Step 5: Write the detail template**

Create `ivms777/web/templates/photo.html`. It fills the viewport — a two-pane
layout with the image on the left and a scrolling data panel on the right:

```html
{% extends "base.html" %}
{% block content %}
<div class="photo-view">
  <div class="photo-stage">
    <a class="photo-close" href="/library" title="Back to library">✕</a>
    <img class="photo-full" src="/thumb/{{ photo['id'] }}?size=detail" alt="">
  </div>
  <aside class="photo-panel">
    {% if photo['caption'] %}<p class="photo-caption">{{ photo['caption'] }}</p>{% endif %}

    <h3>AI</h3>
    {% if embedded %}
      <p>Embedded for semantic search ({{ photo['embedding_model'] }}).</p>
    {% else %}
      <p>Not yet embedded — it will appear in search once the embed stage runs.</p>
    {% endif %}
    <!-- Tags grouped by dimension land here in plan 04. -->

    <h3>Files on disk{% if sources|length > 1 %} — {{ sources|length }} copies{% endif %}</h3>
    {% if wasted_bytes %}
      <p class="wasted">{{ '%.1f'|format(wasted_bytes / 1048576) }} MB of redundant copies</p>
    {% endif %}
    <ul>{% for s in sources %}<li>{{ s['rel_path'] }}</li>{% endfor %}</ul>

    <h3>EXIF</h3>
    <table class="exif">
      {% for f in facets %}
        <tr><td>{{ f['key'] }}</td>
            <td>{{ f['value_text'] if f['value_text'] is not none else f['value_num'] }}</td></tr>
      {% endfor %}
    </table>
  </aside>
</div>
{% if similar %}
<h3>Similar photos</h3>
<div class="similar-strip">
  {% for sid in similar %}
    <a href="/photo/{{ sid }}"><img src="/thumb/{{ sid }}" loading="lazy" alt=""></a>
  {% endfor %}
</div>
{% endif %}
{% endblock %}
```

Add the layout to `ivms777/web/static/app.css` so the stage actually fills the
screen — the image is contained, the panel scrolls independently:

```css
.photo-view { display: flex; gap: 1rem; height: calc(100vh - 4rem); }
.photo-stage { position: relative; flex: 1 1 70%; display: flex;
  align-items: center; justify-content: center; background: #000; }
.photo-full { max-width: 100%; max-height: 100%; object-fit: contain; }
.photo-close { position: absolute; top: 0.5rem; right: 0.75rem; font-size: 1.5rem;
  color: #fff; text-decoration: none; }
.photo-panel { flex: 0 0 30%; overflow-y: auto; padding-right: 0.5rem; }
.similar-strip { display: flex; gap: 0.5rem; overflow-x: auto; }
.similar-strip img { height: 96px; border-radius: 4px; }
```

- [ ] **Step 6: Run the detail tests**

Run: `uv run pytest tests/test_web_photo.py -q`
Expected: PASS.

- [ ] **Step 7: Run the whole suite and lint**

Run: `uv run pytest -q && uv run ruff check .`
Expected: PASS, clean.

- [ ] **Step 8: Verify by hand against the real model**

The fast suite never loads SigLIP. Verify the real thing once:

```bash
IVMS777_DATA_DIR=~/.ivms777 uv run uvicorn ivms777.web.app:app_factory --factory --port 8100
```

(No `IVMS777_USE_FAKE_EMBEDDER`, so the real SigLIP loads and downloads on first use.) Then:
1. Re-index is automatic — the `embed` stage drains the already-uploaded photos on the next `finish`, or run the worker CLI once. Confirm `SELECT count(*) FROM photo_vec` matches the photo count.
2. On `/library`, search "beach" / "people" / "food" and confirm the top results actually match.
3. Open a photo, confirm the EXIF panel, the source paths, and that the "similar" strip shows genuinely related shots.

Record what you saw in the checkpoint. Note the first search triggers the model download (~1.5 GB) and the first embed pass runs at ~1 s/photo on CPU.

- [ ] **Step 9: Update the docs**

In `README.md`, add a line under the run instructions: the first search downloads the SigLIP model (~1.5 GB) and the library embeds in the background at ~1 s/photo on a Mac. In `docs/design.md` §15 Phases, note phase 2's embedding/search slice is delivered by plan 03 and taxonomy/keyword/fusion by plan 04.

- [ ] **Step 10: Checkpoint** — report the manual results, the photo count vs vector count, and stop. Do not commit.

---

## What plan 03 delivers

Type "kids on the beach" into the library search and get the photos that look like that — even ones with no caption, no tag, and a filename like `IMG_4471.jpg`. Open any photo and see its EXIF, every folder it came from, and a strip of visually similar shots. All of it runs on your machine against a model you downloaded once; nothing leaves the box.

Under the hood every uploaded photo now carries a 1152-dim SigLIP vector in `sqlite-vec`, embedded as a background job stage that resumes on restart like every other stage. The fake embedder keeps the whole test suite offline and instant; the real model is exercised by one hand-run check and one `slow`-marked test.

**Not yet working:** taxonomy tags and the tag sidebar, keyword (FTS) and fusion search, the query planner, groups, and chat. Those are plans 04+.

## Following plans

| Plan | Spec phase | Delivers |
|---|---|---|
| 04 | 2 | Zero-shot taxonomy tags + `vocab.yaml`, tag sidebar, keyword (FTS5) + reciprocal-rank fusion search |
| 05 | 3 | Caption stage against the inference service, captions in the UI and in search |
| 06 | 4 | Query planner, parsed-filter chips, caption vocabulary mining |
| 07 | 5 | Event, cluster, and duplicate groups, `/groups` |
| 08 | 6 | Ask-your-library chat with streaming and citations |
| 09 | 7 | Stage 2 — layouts, `/api/manifest`, `/export`, and the `ivms777-sync` CLI |
