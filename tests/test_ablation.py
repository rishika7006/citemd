from citemd.eval.ablation import default_arms, format_table, run_ablation
from citemd.eval.mirage import QuestionItem
from citemd.generate.client import FakeLLMClient
from citemd.models import RetrievedChunk


def _items(n=5):
    return [
        QuestionItem(
            dataset="t", qid=str(i), question=f"q{i}",
            options={"A": "x", "B": "y"}, answer="A" if i % 2 == 0 else "B",
        )
        for i in range(n)
    ]


def _retrieve_fn(query, k):
    return [
        RetrievedChunk(chunk_id=f"d#{i}", source_id=f"pmid:{i}", text=f"passage {i}", score=0.5)
        for i in range(1, k + 1)
    ]


def _scorer(query, passages):
    return [1.0 for _ in passages]


def _responder(messages):
    # Confidence varies with question id so the selective curve is non-trivial.
    user = messages[1]["content"]
    conf = 0.9 if "q0" in user or "q2" in user else 0.3
    return f'{{"answer": "A", "confidence": {conf}, "citations": [1]}}'


def test_run_ablation_produces_four_rows(tmp_path):
    retrieve_fns = {"vector": _retrieve_fn, "hybrid": _retrieve_fn}
    result = run_ablation(
        _items(), FakeLLMClient(_responder),
        out_dir=tmp_path, abstain_fraction=0.4,
        retrieve_fns=retrieve_fns, scorer=_scorer,
    )
    rows = result["rows"]
    assert [r["arm"] for r in rows][:3] == ["vector-only", "hybrid", "hybrid+rerank"]
    assert "abstention" in rows[3]["arm"]
    # Abstention is derived from the best retrieval arm and should not increase its error.
    best_error = min(r["error"] for r in rows[:3])
    assert rows[3]["coverage"] < 1.0
    assert rows[3]["error"] <= best_error + 1e-9


def test_default_arms_progression():
    arms = default_arms()
    assert [a.slug for a in arms] == ["vector", "hybrid", "hybrid_rerank"]
    assert arms[0].config.retriever == "vector" and not arms[0].config.use_rerank
    assert arms[2].config.retriever == "hybrid" and arms[2].config.use_rerank
    assert all(a.config.allow_abstain is False for a in arms)


def test_format_table_has_header_and_rows():
    rows = [{"arm": "x", "n": 3, "accuracy": 0.5, "error": 0.5, "coverage": 1.0}]
    table = format_table(rows)
    assert "arm" in table and "accuracy" in table
    assert "x" in table
