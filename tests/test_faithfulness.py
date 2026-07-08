from citemd.eval.faithfulness import (
    judge_support,
    run_faithfulness,
    score_faithfulness,
)
from citemd.eval.mirage import QuestionItem
from citemd.generate.client import FakeLLMClient
from citemd.models import RetrievedChunk
from citemd.pipeline import PipelineConfig


def _items():
    opts = {"A": "yes", "B": "no"}
    return [
        QuestionItem(dataset="t", qid="1", question="q1", options=opts, answer="A"),
        QuestionItem(dataset="t", qid="2", question="q2", options=opts, answer="A"),
    ]


def _retrieve_fn(query, k):
    return [RetrievedChunk(chunk_id="d#1", source_id="pmid:1", text="evidence text", score=0.5)]


def _gen(messages):
    # Always answer A citing passage 1.
    return '{"answer": "A", "confidence": 0.8, "citations": [1]}'


def test_judge_support_parses_yes_no():
    yes = FakeLLMClient("yes")
    no = FakeLLMClient("No, the passage does not support it.")
    assert judge_support("q", "yes", "p", yes) is True
    assert judge_support("q", "yes", "p", no) is False


def test_run_faithfulness_supported():
    recs = run_faithfulness(
        _items(), FakeLLMClient(_gen), FakeLLMClient("yes"),
        config=PipelineConfig(use_rerank=False, allow_abstain=False),
        retrieve_fn=_retrieve_fn,
    )
    assert len(recs) == 2
    assert all(r.n_cited == 1 and r.n_supported == 1 and r.supported for r in recs)
    s = score_faithfulness(recs)
    assert s["coverage"].value == 1.0
    assert s["answer_support_rate"].value == 1.0
    assert s["citation_support_rate"].value == 1.0


def test_run_faithfulness_unsupported():
    recs = run_faithfulness(
        _items(), FakeLLMClient(_gen), FakeLLMClient("no"),
        config=PipelineConfig(use_rerank=False, allow_abstain=False),
        retrieve_fn=_retrieve_fn,
    )
    assert all(r.n_cited == 1 and r.n_supported == 0 and not r.supported for r in recs)
    s = score_faithfulness(recs)
    assert s["coverage"].value == 1.0  # they cited, judge just disagreed
    assert s["answer_support_rate"].value == 0.0
    assert s["citation_support_rate"].value == 0.0


def test_run_faithfulness_resumes(tmp_path):
    out = tmp_path / "f.jsonl"
    run_faithfulness(
        _items(), FakeLLMClient(_gen), FakeLLMClient("yes"),
        config=PipelineConfig(use_rerank=False, allow_abstain=False),
        retrieve_fn=_retrieve_fn, out_path=out,
    )

    def boom(messages):
        raise AssertionError("should not be called when resuming")

    resumed = run_faithfulness(
        _items(), FakeLLMClient(boom), FakeLLMClient(boom),
        config=PipelineConfig(use_rerank=False, allow_abstain=False),
        retrieve_fn=_retrieve_fn, out_path=out,
    )
    assert len(resumed) == 2
