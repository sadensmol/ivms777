def reciprocal_rank_fusion(rankings: list[list[int]], k_const: int = 60) -> list[int]:
    """Merge several rankings by reciprocal rank fusion: score = sum 1/(k_const + rank).

    Items ranked well by more than one input rise to the top; an empty input
    contributes nothing.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k_const + rank)
    return sorted(scores, key=lambda item: scores[item], reverse=True)
