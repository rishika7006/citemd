from citemd.models import RetrievedChunk
from citemd.retrieve.rerank import rerank


def _chunks(n):
    return [
        RetrievedChunk(chunk_id=f"d#{i}", source_id="d", text=f"passage {i}", score=0.0)
        for i in range(n)
    ]


def test_rerank_reorders_by_scorer_and_truncates():
    chunks = _chunks(4)
    # Scorer prefers the last passage most, then the second.
    scores = {"passage 0": 0.1, "passage 1": 0.9, "passage 2": 0.2, "passage 3": 1.5}

    def scorer(query, passages):
        return [scores[p] for p in passages]

    out = rerank("q", chunks, k=2, scorer=scorer)
    assert [c.chunk_id for c in out] == ["d#3", "d#1"]
    assert all(c.retriever == "rerank" for c in out)
    assert out[0].score == 1.5


def test_rerank_empty_input():
    assert rerank("q", [], scorer=lambda q, p: []) == []
