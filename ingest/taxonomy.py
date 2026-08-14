import sqlite3

from PIL import Image

from embedding.base import Embedder
from embedding.store import read_vector
from embedding.vectors import l2_normalize, siglip_probability
from ingest.jobs import enqueue
from ingest.pixels import pixel_tags
from ingest.vocab import Vocab, tag_id_map
from ingest.worker import StageHandler
from storage.base import Storage


def backfill_taxonomy(conn: sqlite3.Connection) -> int:
    """Queue a taxonomy job for every embedded photo that has none yet.

    Photos ingested before the taxonomy stage existed carry an embedding but no
    taxonomy job, so they would never be tagged. This makes the pipeline
    self-healing on the next drain. Idempotent — `enqueue` skips existing jobs.
    """
    rows = conn.execute(
        "SELECT p.id FROM photos p WHERE p.embedding_model IS NOT NULL"
        " AND NOT EXISTS (SELECT 1 FROM jobs j WHERE j.photo_id = p.id AND j.stage = 'taxonomy')"
    ).fetchall()
    for row in rows:
        enqueue(conn, row["id"], "taxonomy")
    return len(rows)

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
    """Rebuild the photo's photo_fts row from its caption and tag labels."""
    caption = conn.execute(
        "SELECT caption FROM photos WHERE id = ?", (photo_id,)
    ).fetchone()["caption"]
    labels = [
        r["label"] for r in conn.execute(
            "SELECT t.label FROM photo_tags pt JOIN tags t ON t.id = pt.tag_id"
            " WHERE pt.photo_id = ?",
            (photo_id,),
        )
    ]
    conn.execute("DELETE FROM photo_fts WHERE rowid = ?", (photo_id,))
    conn.execute(
        "INSERT INTO photo_fts(rowid, caption, tags_text) VALUES (?, ?, ?)",
        (photo_id, caption or "", " ".join(labels)),
    )


def taxonomy_handler(derived: Storage, embedder: Embedder, vocab: Vocab) -> StageHandler:
    # Embed each label prompt once for the whole drain, not once per photo.
    prompts = [(d, lbl) for d, labels in vocab.dimensions.items() for lbl in labels]
    vectors = embedder.embed_texts([label_prompt(d, lbl) for d, lbl in prompts])
    entries = [(d, lbl, l2_normalize(vec)) for (d, lbl), vec in zip(prompts, vectors)]

    def handle(conn: sqlite3.Connection, photo_id: int) -> None:
        ids = tag_id_map(conn)
        scored: list[tuple[str, str, float, str]] = []
        image_vec = read_vector(conn, photo_id)
        if image_vec is not None:
            image_vec = l2_normalize(image_vec)
            for dimension, label, label_vec in entries:
                cosine = sum(a * b for a, b in zip(image_vec, label_vec))
                prob = siglip_probability(cosine, embedder.logit_scale, embedder.logit_bias)
                if prob >= vocab.threshold(dimension):
                    scored.append((dimension, label, prob, "siglip"))
        thumb = conn.execute(
            "SELECT thumb_key FROM photos WHERE id = ?", (photo_id,)
        ).fetchone()["thumb_key"]
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
