"""Retrieval-quality metrics.

These quantify how well the retriever surfaces relevant evidence, independent of the
downstream generation. All functions take a ranked list of retrieved ids and a set of
relevant (gold) ids, and are pure Python so they need no extra dependencies.

  - hit_rate@k:   1.0 if any relevant id appears in the top k, else 0.0
  - mrr@k:        reciprocal rank of the first relevant id within the top k, else 0.0
  - ndcg@k:       normalized discounted cumulative gain with binary relevance

Aggregate helpers average a metric across many queries.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def _top_k(retrieved: Sequence[str], k: int) -> list[str]:
    if k <= 0:
        raise ValueError("k must be positive")
    return list(retrieved[:k])


def hit_rate_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """1.0 if at least one relevant id is in the top k, else 0.0."""
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    return 1.0 if any(rid in relevant_set for rid in _top_k(retrieved, k)) else 0.0


def reciprocal_rank_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Reciprocal rank of the first relevant id within the top k, else 0.0."""
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    for rank, rid in enumerate(_top_k(retrieved, k), start=1):
        if rid in relevant_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Normalized DCG at k with binary relevance.

    Returns 0.0 when there are no relevant ids. The ideal DCG assumes all relevant items
    could be packed into the top positions.
    """
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    top = _top_k(retrieved, k)
    dcg = sum(
        (1.0 / math.log2(rank + 1)) for rank, rid in enumerate(top, start=1) if rid in relevant_set
    )
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def aggregate(
    rows: Iterable[tuple[Sequence[str], Iterable[str]]],
    k: int,
) -> dict[str, float]:
    """Average hit_rate, MRR, and nDCG at k over many (retrieved, relevant) rows."""
    hits: list[float] = []
    mrrs: list[float] = []
    ndcgs: list[float] = []
    for retrieved, relevant in rows:
        relevant = list(relevant)
        hits.append(hit_rate_at_k(retrieved, relevant, k))
        mrrs.append(reciprocal_rank_at_k(retrieved, relevant, k))
        ndcgs.append(ndcg_at_k(retrieved, relevant, k))
    n = len(hits)
    if n == 0:
        return {f"hit_rate@{k}": 0.0, f"mrr@{k}": 0.0, f"ndcg@{k}": 0.0, "n": 0}
    return {
        f"hit_rate@{k}": sum(hits) / n,
        f"mrr@{k}": sum(mrrs) / n,
        f"ndcg@{k}": sum(ndcgs) / n,
        "n": n,
    }
