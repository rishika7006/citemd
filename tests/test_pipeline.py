from citemd.generate.client import FakeLLMClient
from citemd.models import RetrievedChunk
from citemd.pipeline import PipelineConfig, answer


def _retrieve_fn(n=4):
    def fn(query, k):
        return [
            RetrievedChunk(
                chunk_id=f"d#{i}", source_id=f"pmid:{i}", text=f"passage {i}", score=0.5
            )
            for i in range(1, n + 1)
        ][:k]

    return fn


def _json(answer="B", conf=0.9, cites=(1,)):
    cites_s = ", ".join(str(c) for c in cites)
    return f'{{"answer": "{answer}", "confidence": {conf}, "citations": [{cites_s}]}}'


def test_pipeline_answers_with_citations():
    llm = FakeLLMClient(_json("B", 0.9, (1, 2)))
    cfg = PipelineConfig(use_rerank=False, top_k=3, allow_abstain=False)
    res = answer("q?", llm, options={"A": "x", "B": "y"}, config=cfg, retrieve_fn=_retrieve_fn())
    assert res.option == "B"
    assert not res.abstained
    assert [c.marker for c in res.citations] == [1, 2]
    assert res.confidence == 0.9
    assert len(res.contexts) == 3


def test_pipeline_rerank_changes_context_order():
    # Scorer prefers passage 3 highest, so it should become context[0].
    def scorer(query, passages):
        return [float(p.split()[-1]) for p in passages]  # "passage N" -> N

    llm = FakeLLMClient(_json("A", 0.5, (1,)))
    cfg = PipelineConfig(use_rerank=True, top_k=2, allow_abstain=False, confidence_source="rerank")
    res = answer(
        "q?", llm, options={"A": "x"}, config=cfg, retrieve_fn=_retrieve_fn(4), scorer=scorer
    )
    assert res.contexts[0].chunk_id == "d#4"
    assert res.contexts[0].retriever == "rerank"
    # rerank confidence = sigmoid(top score=4.0) ~ 0.98
    assert res.confidence > 0.9


def test_pipeline_threshold_gate_abstains():
    llm = FakeLLMClient(_json("B", 0.1, (1,)))
    cfg = PipelineConfig(
        use_rerank=False, allow_abstain=False, abstain_threshold=0.5, confidence_source="model"
    )
    res = answer("q?", llm, options={"A": "x", "B": "y"}, config=cfg, retrieve_fn=_retrieve_fn())
    assert res.abstained
    assert res.abstain_reason == "low_confidence"
    assert res.option is None


def test_pipeline_model_abstention_respected():
    llm = FakeLLMClient('{"answer": "insufficient", "citations": []}')
    cfg = PipelineConfig(use_rerank=False, allow_abstain=True)
    res = answer("q?", llm, options={"A": "x", "B": "y"}, config=cfg, retrieve_fn=_retrieve_fn())
    assert res.abstained
    assert res.abstain_reason == "model_insufficient"


def test_pipeline_blend_confidence():
    # model conf 0.6, rerank not used so retrieval conf = clamp(top cosine 0.5) = 0.5
    llm = FakeLLMClient(_json("A", 0.6, (1,)))
    cfg = PipelineConfig(use_rerank=False, allow_abstain=False, confidence_source="blend")
    res = answer("q?", llm, options={"A": "x"}, config=cfg, retrieve_fn=_retrieve_fn())
    assert abs(res.confidence - 0.55) < 1e-9
