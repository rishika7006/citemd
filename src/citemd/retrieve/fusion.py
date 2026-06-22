"""Reciprocal Rank Fusion (RRF).

RRF combines several ranked lists into one without needing comparable scores across
retrievers. Each item gets ``sum over lists of 1 / (k + rank)`` where rank is 1-based.
This is the fusion step that turns separate vector and BM25 results into a single hybrid
ranking. It is pure Python and dependency-free so it is trivially testable.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def rrf_fuse(
    ranked_lists: Sequence[Iterable[str]],
    *,
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[tuple[str, float]]:
    """Fuse ranked id lists into a single descending ``(id, score)`` ranking.

    Args:
        ranked_lists: Each inner iterable is a list of item ids in best-first order.
        k: RRF damping constant. Larger k flattens the contribution of top ranks.
        weights: Optional per-list weights; defaults to 1.0 for every list.

    Returns:
        ``(id, score)`` tuples sorted by score descending. Ties are broken by the id for
        deterministic output.
    """
    if k <= 0:
        raise ValueError("k must be positive")

    lists = [list(rl) for rl in ranked_lists]
    if weights is None:
        weights = [1.0] * len(lists)
    if len(weights) != len(lists):
        raise ValueError("weights must have one entry per ranked list")

    scores: dict[str, float] = {}
    for weight, ranked in zip(weights, lists):
        for rank, item_id in enumerate(ranked, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + weight * (1.0 / (k + rank))

    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
