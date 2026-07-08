from citemd.eval.mirage import QuestionItem
from citemd.eval.qa import (
    retrieval_hit_rate,
    run_qa,
    score_records,
    selective_items,
)
from citemd.generate.client import FakeLLMClient
from citemd.models import RetrievedChunk
from citemd.pipeline import PipelineConfig


def _items():
    return [
        QuestionItem(dataset="t", qid="1", question="q1", options={"A": "x", "B": "y"}, answer="A"),
        QuestionItem(dataset="t", qid="2", question="q2", options={"A": "x", "B": "y"}, answer="B"),
        QuestionItem(dataset="t", qid="3", question="q3", options={"A": "x", "B": "y"}, answer="A"),
    ]


def _retrieve_fn(query, k):
    return [RetrievedChunk(chunk_id="d#1", source_id="pmid:1", text="ev", score=0.5)]


def _always_a(messages):
    return '{"answer": "A", "confidence": 0.8, "citations": [1]}'


def test_run_qa_scores_and_caches(tmp_path):
    out = tmp_path / "qa.jsonl"
    cfg = PipelineConfig(use_rerank=False, allow_abstain=False)
    recs = run_qa(
        _items(), FakeLLMClient(_always_a), config=cfg, out_path=out, retrieve_fn=_retrieve_fn
    )
    assert len(recs) == 3
    assert [r.correct for r in recs] == [True, False, True]
    m = score_records(recs)
    assert abs(m["accuracy"] - 2 / 3) < 1e-9
    assert m["coverage"] == 1.0
    assert out.read_text().count("\n") == 3


def test_run_qa_resumes_without_calling_llm(tmp_path):
    out = tmp_path / "qa.jsonl"
    cfg = PipelineConfig(use_rerank=False, allow_abstain=False)
    run_qa(_items(), FakeLLMClient(_always_a), config=cfg, out_path=out, retrieve_fn=_retrieve_fn)

    def boom(messages):
        raise AssertionError("LLM should not be called when resuming a finished run")

    resumed_llm = FakeLLMClient(boom)
    recs = run_qa(_items(), resumed_llm, config=cfg, out_path=out, retrieve_fn=_retrieve_fn)
    assert len(recs) == 3
    assert resumed_llm.calls == []  # all served from cache


def test_retrieval_hit_rate_from_gold_pmids():
    # q1 gold pmid "1" is retrieved (source_id pmid:1); q2 gold pmid "9" is not; q3 has no pmid.
    items = [
        QuestionItem(dataset="t", qid="1", question="q1", options={"A": "x", "B": "y"},
                     answer="A", pmids=["1"]),
        QuestionItem(dataset="t", qid="2", question="q2", options={"A": "x", "B": "y"},
                     answer="A", pmids=["9"]),
        QuestionItem(dataset="t", qid="3", question="q3", options={"A": "x", "B": "y"},
                     answer="A", pmids=[]),
    ]
    recs = run_qa(
        items, FakeLLMClient(_always_a),
        config=PipelineConfig(use_rerank=False, allow_abstain=False),
        retrieve_fn=_retrieve_fn,
    )
    by_qid = {r.qid: r for r in recs}
    assert by_qid["1"].gold_retrieved is True and by_qid["1"].gold_rank == 1
    assert by_qid["2"].gold_retrieved is False and by_qid["2"].gold_rank is None
    # q3 has no gold PMID and is excluded from hit-rate; of the 2 with gold, 1 hit.
    hr = retrieval_hit_rate(recs)
    assert hr.k == 1 and hr.n == 2


def test_selective_items_shape():
    recs = run_qa(
        _items(), FakeLLMClient(_always_a),
        config=PipelineConfig(use_rerank=False, allow_abstain=False),
        retrieve_fn=_retrieve_fn,
    )
    items = selective_items(recs)
    assert len(items) == 3
    assert all(isinstance(c, float) and isinstance(ok, bool) for c, ok in items)
