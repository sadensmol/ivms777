"""Calibrate the chat rerank floor (search.rerank.RERANK_FLOOR, §10, §17).

Runs the same candidate build the chat retriever uses (plan -> retriever-core
candidates -> hard facet/date filter), then sweeps the relevance floor over a
hand-labelled dev set and reports precision / recall / F1 per floor. Pick the
F1-maximising floor and set RERANK_FLOOR to it.

Dev set: a JSON file mapping each query to the photo ids that truly answer it:

    {"a photo with a dog": [12, 87], "sunset over the sea": [3]}

Usage:  uv run python -m scripts.rerank_bakeoff dev_set.json
"""

import argparse
import json
from pathlib import Path

from config import get_settings
from db.connection import connect
from embedding.vectors import l2_normalize
from ingest.vocab import load_vocab
from search.planner import plan, spec_to_params
from search.rerank import rerank
from search.retriever import Query, _hard_filter, candidates

FLOORS = [round(0.2 + 0.05 * i, 2) for i in range(9)]  # 0.20 .. 0.60
VOCAB_PATH = Path(__file__).resolve().parent.parent / "vocab.yaml"


def _candidates(conn, embedder, client, settings, dimensions, question):
    """The retriever's core candidates + hard facet/date filter, minus rerank
    (mirrors chat.agent.retrieve — plan 12 task 5)."""
    spec = plan(client, settings.planner_model or "fake", question, dimensions)
    hard_filters = spec_to_params(spec, query=question, dimensions=dimensions)
    query = Query(text=spec.semantic or question, hard_filters=hard_filters,
                  soft_tags=spec.tags, k=200)
    ids = candidates(conn, embedder, settings.owner_id, query)
    return _hard_filter(conn, settings.owner_id, query.hard_filters, ids)


def main(dev_path: str) -> None:
    settings = get_settings()
    conn = connect(settings.db_path)
    embedder, _ = settings.build_embedder()
    client, _ = settings.build_inference_client()
    dimensions = list(load_vocab(VOCAB_PATH).dimensions)
    dev: dict[str, list[int]] = json.loads(Path(dev_path).read_text())

    best = (0.0, -1.0)  # (f1, floor)
    for floor in FLOORS:
        tp = fp = fn = 0
        for question, gold_list in dev.items():
            gold = set(gold_list)
            candidates = _candidates(conn, embedder, client, settings, dimensions, question)
            query_vec = l2_normalize(client.embed(settings.caption_embed_model, [question])[0])
            got = {pid for pid, _ in rerank(conn, query_vec, candidates, floor=floor)}
            tp += len(got & gold)
            fp += len(got - gold)
            fn += len(gold - got)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        print(f"floor={floor:.2f}  P={precision:.2f} R={recall:.2f} F1={f1:.2f}")
        if f1 > best[0]:
            best = (f1, floor)

    print(f"\nBest: floor={best[1]:.2f} (F1={best[0]:.2f}) -> set RERANK_FLOOR to this.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate the chat rerank floor.")
    parser.add_argument("dev_set", help="JSON: {query: [relevant photo ids]}")
    main(parser.parse_args().dev_set)
