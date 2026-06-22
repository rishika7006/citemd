import math

from citemd.eval.metrics import (
    aggregate,
    hit_rate_at_k,
    ndcg_at_k,
    reciprocal_rank_at_k,
)


def test_hit_rate():
    assert hit_rate_at_k(["a", "b", "c"], ["c"], 3) == 1.0
    assert hit_rate_at_k(["a", "b", "c"], ["c"], 2) == 0.0
    assert hit_rate_at_k(["a", "b"], [], 2) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank_at_k(["a", "b", "c"], ["b"], 3) == 0.5
    assert reciprocal_rank_at_k(["a", "b", "c"], ["a"], 3) == 1.0
    assert reciprocal_rank_at_k(["a", "b", "c"], ["z"], 3) == 0.0


def test_ndcg_perfect_and_partial():
    # Relevant item at rank 1 -> perfect nDCG.
    assert ndcg_at_k(["a", "b", "c"], ["a"], 3) == 1.0
    # Relevant item at rank 2 with one relevant total: dcg = 1/log2(3), idcg = 1/log2(2)=1.
    expected = (1.0 / math.log2(3)) / 1.0
    assert math.isclose(ndcg_at_k(["a", "b", "c"], ["b"], 3), expected)


def test_ndcg_two_relevant():
    # Both relevant retrieved at ranks 1 and 3.
    val = ndcg_at_k(["a", "x", "b"], ["a", "b"], 3)
    dcg = 1.0 / math.log2(2) + 1.0 / math.log2(4)
    idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
    assert math.isclose(val, dcg / idcg)


def test_aggregate_averages():
    rows = [
        (["a", "b"], ["a"]),  # hit=1, rr=1, ndcg=1
        (["x", "y"], ["z"]),  # hit=0, rr=0, ndcg=0
    ]
    out = aggregate(rows, k=2)
    assert out["n"] == 2
    assert out["hit_rate@2"] == 0.5
    assert out["mrr@2"] == 0.5


def test_aggregate_empty():
    out = aggregate([], k=5)
    assert out["n"] == 0
    assert out["hit_rate@5"] == 0.0
