"""Measure what the caption-meaning signal is actually worth in "similar photos" (§9.3).

Runs `similar_photos` over every seed in the library TWICE — once with the caption
signal on, once with `use_captions=False` — and reports what changes. No model is
loaded: both `caption_vec` and the image vector are already stored, so this is pure
SQL + numpy-free cosine, and the timing below is the real per-click cost of the
signal.

The number that decides it is **rescued**: results that exist ONLY because of the
caption signal — either the caption pulled them into the candidate union, or the
caption contribution was their only CONTENT signal, so without it the content gate
drops them. If `rescued` is ~0 the nomic text embedder is dead weight; if it is
material, the signal is doing the job it was added for.

Usage:  uv run python -m scripts.caption_ablation [--k 12] [--limit N]
"""

import argparse
import sqlite3
import time
from pathlib import Path

from config import get_settings
from db.connection import connect
from ingest.vocab import load_vocab
from search.semantic import similar_photos

VOCAB_PATH = Path(__file__).resolve().parent.parent / "vocab.yaml"


def _seeds(conn: sqlite3.Connection, owner_id: int, limit: int | None) -> list[int]:
    """Every photo that can be a seed: it has an image vector to compare against."""
    rows = conn.execute(
        "SELECT id FROM photos WHERE owner_id = ? AND embedding_model IS NOT NULL"
        " ORDER BY id",
        (owner_id,),
    ).fetchall()
    ids = [r["id"] for r in rows]
    return ids[:limit] if limit else ids


def _caption_reason(result: dict) -> bool:
    return any(r["text"] == "caption (meaning)" for r in result["reasons"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=12, help="results per seed (UI uses 12)")
    parser.add_argument("--limit", type=int, default=None, help="only the first N seeds")
    args = parser.parse_args()

    settings = get_settings()
    conn = connect(settings.db_path)
    vocab = load_vocab(VOCAB_PATH)
    weights = vocab.dimension_weights
    owner_id = settings.owner_id

    seeds = _seeds(conn, owner_id, args.limit)
    if not seeds:
        print("no seeds: no photo has an image embedding yet")
        return

    captioned = conn.execute(
        "SELECT COUNT(*) AS n FROM photos WHERE owner_id = ? AND caption_vec IS NOT NULL",
        (owner_id,),
    ).fetchone()["n"]
    total = conn.execute(
        "SELECT COUNT(*) AS n FROM photos WHERE owner_id = ?", (owner_id,)
    ).fetchone()["n"]
    print(f"library: {total} photos · {captioned} with caption_vec · {len(seeds)} seeds · k={args.k}\n")

    def run(use_captions: bool) -> tuple[dict[int, list[dict]], float]:
        started = time.perf_counter()
        out = {
            seed: similar_photos(
                conn, owner_id, seed, k=args.k,
                min_cosine=settings.similar_min_cosine,
                caption_min=settings.similar_caption_min,
                dimension_weights=weights,
                use_captions=use_captions,
            )
            for seed in seeds
        }
        return out, time.perf_counter() - started

    with_cap, t_on = run(True)
    without_cap, t_off = run(False)

    seeds_with_any = 0          # seeds where the caption signal changed the result set
    seeds_caption_reason = 0    # seeds where >=1 result cites caption in its top-3
    rescued_total = 0           # results that exist ONLY because of captions
    lost_total = 0              # results present without captions but pushed out with them
    rank_moved = 0              # results in both sets, at a different rank
    both_total = 0
    top1_changed = 0
    empty_on = empty_off = 0

    worst: list[tuple[int, int, list[str]]] = []  # (rescued, seed, captions of rescued)

    for seed in seeds:
        on, off = with_cap[seed], without_cap[seed]
        on_ids = [r["id"] for r in on]
        off_ids = [r["id"] for r in off]
        empty_on += not on_ids
        empty_off += not off_ids
        rescued = [pid for pid in on_ids if pid not in set(off_ids)]
        lost = [pid for pid in off_ids if pid not in set(on_ids)]
        rescued_total += len(rescued)
        lost_total += len(lost)
        if rescued or lost:
            seeds_with_any += 1
        if any(_caption_reason(r) for r in on):
            seeds_caption_reason += 1
        if on_ids[:1] != off_ids[:1]:
            top1_changed += 1
        off_rank = {pid: i for i, pid in enumerate(off_ids)}
        for i, pid in enumerate(on_ids):
            if pid in off_rank:
                both_total += 1
                rank_moved += off_rank[pid] != i
        if rescued:
            caps = [
                (conn.execute("SELECT caption FROM photos WHERE id = ?", (pid,))
                 .fetchone()["caption"] or "")[:60]
                for pid in rescued[:3]
            ]
            worst.append((len(rescued), seed, caps))

    n = len(seeds)
    pct = lambda x: f"{100 * x / n:5.1f}%"  # noqa: E731 — local formatting shorthand
    print("--- what the caption signal changes ------------------------------")
    print(f"seeds whose result SET changed          : {seeds_with_any:4d} / {n}  {pct(seeds_with_any)}")
    print(f"seeds whose #1 result changed           : {top1_changed:4d} / {n}  {pct(top1_changed)}")
    print(f"seeds citing 'caption (meaning)' in top3: {seeds_caption_reason:4d} / {n}  {pct(seeds_caption_reason)}")
    print(f"results RESCUED by captions (would vanish): {rescued_total:4d}"
          f"  ({rescued_total / n:.2f} per seed)")
    print(f"results DISPLACED by captions            : {lost_total:4d}")
    print(f"results in both sets but re-ranked       : {rank_moved:4d} / {both_total}")
    print(f"seeds with NO results  on={empty_on} off={empty_off}")
    print()
    print("--- cost ---------------------------------------------------------")
    print(f"{n} seeds · captions ON  {t_on:6.3f}s  ({1000 * t_on / n:6.2f} ms/seed)")
    print(f"{n} seeds · captions OFF {t_off:6.3f}s  ({1000 * t_off / n:6.2f} ms/seed)")
    print(f"caption signal costs {1000 * (t_on - t_off) / n:+.2f} ms/seed at this library size")
    print("  (plus ~800 MB resident for the nomic embedder that writes caption_vec)")

    if worst:
        worst.sort(reverse=True)
        print("\n--- top seeds the caption signal rescued results for --------------")
        for count, seed, caps in worst[:10]:
            seed_cap = (conn.execute("SELECT caption FROM photos WHERE id = ?", (seed,))
                        .fetchone()["caption"] or "")[:60]
            print(f"  seed {seed:5d} (+{count}): {seed_cap}")
            for cap in caps:
                print(f"        rescued: {cap}")


if __name__ == "__main__":
    main()
