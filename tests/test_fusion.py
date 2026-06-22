import pytest

from citemd.retrieve.fusion import rrf_fuse


def test_rrf_rewards_agreement_across_lists():
    vector = ["a", "b", "c"]
    bm25 = ["b", "a", "d"]
    fused = rrf_fuse([vector, bm25], k=60)
    ids = [item_id for item_id, _ in fused]
    # 'a' and 'b' appear high in both lists, so they outrank single-list 'c' and 'd'.
    assert set(ids[:2]) == {"a", "b"}
    assert ids[-1] in {"c", "d"}


def test_rrf_scores_descending():
    fused = rrf_fuse([["x", "y", "z"]], k=60)
    scores = [s for _, s in fused]
    assert scores == sorted(scores, reverse=True)


def test_rrf_deterministic_tie_break_by_id():
    # Two ids at identical rank in identical single list -> tie broken by id ordering.
    fused = rrf_fuse([["b"], ["a"]], k=60)
    assert [i for i, _ in fused] == ["a", "b"]


def test_rrf_weights_shift_ranking():
    vector = ["a", "b"]
    bm25 = ["b", "a"]
    weighted = rrf_fuse([vector, bm25], k=60, weights=[5.0, 1.0])
    assert weighted[0][0] == "a"  # vector list dominates


def test_rrf_validates_inputs():
    with pytest.raises(ValueError):
        rrf_fuse([["a"]], k=0)
    with pytest.raises(ValueError):
        rrf_fuse([["a"], ["b"]], weights=[1.0])
