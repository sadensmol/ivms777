import sqlite3
from math import exp

from PIL import Image

from embedding.base import Embedder
from embedding.store import read_vector
from embedding.vectors import l2_normalize
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
# These are the FALLBACK: a label listed under `prompts:` in vocab.yaml says itself
# in its own words instead. Measured on a hand-labelled sample, the template alone
# gets `subject` right 37% of the time — it produces prompts like "a photo of
# portrait" and "a photo of top-down", and SigLIP's text tower is trained on real
# captions, so ungrammatical phrasing costs real accuracy (§7).
_TEMPLATES = {
    "setting": "a photo taken in a {label}",
    "occasion": "a photo taken at a {label}",
    "season_weather": "a photo taken in {label} weather",
    "composition": "a photo composed as a {label}",
    "vibe": "a photo with a {label} mood",
    "emotion": "a photo of a {label} moment",
    "light": "a photo taken in {label} light",
    "palette": "a {label} colored photo",
    "quality": "a {label} photo",
}
_DEFAULT_TEMPLATE = "a photo of a {label}"


def label_prompts(vocab: Vocab, dimension: str, label: str) -> list[str]:
    """What to SAY to the text tower for one label — its own sentences from
    `vocab.prompts` if it has any, else the dimension's template (§7)."""
    own = vocab.prompts.get(dimension, {}).get(label)
    return list(own) if own else [
        _TEMPLATES.get(dimension, _DEFAULT_TEMPLATE).format(label=label)
    ]


def select_dimension_tags(
    scored: list[tuple[str, float]],
    *,
    max_per_dim: int,
    select_ratio: float,
) -> list[tuple[str, float]]:
    """Pick a dimension's tags by softmax over its own labels' logits (§7).

    SigLIP zero-shot is a per-dimension classifier: raw sigmoid probabilities are
    tiny and near-uniform, so an absolute floor keeps nothing. Softmax over the
    dimension's logits (``cosine·scale + bias``) turns them into a real
    distribution; the argmax always wins, and a runner-up is kept only when its
    probability is within ``select_ratio`` of the top's — capped at
    ``max_per_dim``. ``scored`` is ``[(label, logit)]``; returns
    ``[(label, softmax_probability)]``, highest first.
    """
    if not scored:
        return []
    top_logit = max(logit for _, logit in scored)
    exps = [(label, exp(logit - top_logit)) for label, logit in scored]
    total = sum(weight for _, weight in exps)
    ranked = sorted(
        ((label, weight / total) for label, weight in exps), key=lambda x: -x[1]
    )
    floor = select_ratio * ranked[0][1]
    return [(label, prob) for label, prob in ranked if prob >= floor][:max_per_dim]


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
    # Embed each label prompt once for the whole drain, not once per photo, and
    # group the normalized vectors by dimension so each is scored on its own.
    by_dim: dict[str, list[tuple[str, list[float]]]] = {}
    for dimension, labels in vocab.dimensions.items():
        # A label may be said several ways (§7). Embed every sentence, then average
        # a label's vectors into one — prompt ensembling, so a single unlucky
        # phrasing cannot decide the tag.
        said = [label_prompts(vocab, dimension, lbl) for lbl in labels]
        vectors = embedder.embed_texts([p for prompts in said for p in prompts])
        by_dim[dimension] = []
        at = 0
        for lbl, prompts in zip(labels, said):
            group = [l2_normalize(v) for v in vectors[at : at + len(prompts)]]
            at += len(prompts)
            mean = [sum(axis) / len(group) for axis in zip(*group)]
            by_dim[dimension].append((lbl, l2_normalize(mean)))

    def handle(conn: sqlite3.Connection, photo_id: int) -> None:
        ids = tag_id_map(conn)
        scored: list[tuple[str, str, float, str]] = []
        image_vec = read_vector(conn, photo_id)
        if image_vec is not None:
            image_vec = l2_normalize(image_vec)
            for dimension, entries in by_dim.items():
                logits = [
                    (
                        label,
                        sum(a * b for a, b in zip(image_vec, label_vec))
                        * embedder.logit_scale
                        + embedder.logit_bias,
                    )
                    for label, label_vec in entries
                ]
                for label, prob in select_dimension_tags(
                    logits, max_per_dim=vocab.max_per_dim, select_ratio=vocab.select_ratio
                ):
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
