from citemd.eval.retrieval import load_labels, run_retrieval_eval
from citemd.models import RetrievedChunk


def _fake_retriever(query, k):
    # Deterministic toy index: the gold chunk id encodes the query.
    table = {
        "q1": ["c1", "c2", "c3"],
        "q2": ["c9", "c8", "c2"],
    }
    return [
        RetrievedChunk(chunk_id=cid, source_id=cid.rstrip("0123456789"), text="...")
        for cid in table.get(query, [])[:k]
    ]


def test_run_retrieval_eval_by_chunk():
    labels = [
        {"query": "q1", "relevant_ids": ["c1"]},  # rank 1
        {"query": "q2", "relevant_ids": ["c2"]},  # rank 3
    ]
    out = run_retrieval_eval(labels, _fake_retriever, k=3)
    assert out["n"] == 2
    assert out["hit_rate@3"] == 1.0
    assert out["mrr@3"] == (1.0 + 1.0 / 3) / 2


def test_run_retrieval_eval_by_source():
    labels = [{"query": "q1", "relevant_ids": ["c"]}]
    out = run_retrieval_eval(labels, _fake_retriever, k=3, by_source=True)
    assert out["hit_rate@3"] == 1.0


def test_load_labels(tmp_path):
    p = tmp_path / "labels.jsonl"
    p.write_text(
        '{"query": "a", "relevant_ids": ["x"]}\n\n{"query": "b", "relevant_ids": ["y"]}\n',
        encoding="utf-8",
    )
    rows = load_labels(p)
    assert len(rows) == 2
    assert rows[0]["query"] == "a"
