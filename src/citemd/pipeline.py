"""The CiteMD query pipeline: retrieve -> (rerank) -> generate cited answer -> abstain.

A single ``answer()`` function, parameterized by ``PipelineConfig``, drives every arm of the
ablation (vector-only -> hybrid -> +rerank -> +abstention). Retrieval, reranking, and the LLM
are all injectable, so the whole pipeline is unit-testable with fakes and never requires a
database, a GPU, or a paid API call during tests.

Two notions of abstention are supported and kept distinct:

  - prompt-level (``allow_abstain``): whether the model may reply "insufficient".
  - gate-level (``abstain_threshold``): abstain when the confidence signal falls below a
    threshold, regardless of what the model said.

For evaluation we typically run with ``allow_abstain=False`` (force a choice) to obtain a
prediction and a confidence for every question, then derive the abstention/coverage tradeoff
offline in :mod:`citemd.eval.selective`. That yields the whole risk-coverage curve from one
LLM run per retrieval setting instead of one run per threshold.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from citemd.config import get_settings
from citemd.generate.client import LLMClient
from citemd.generate.prompt import build_messages, parse_answer, resolve_citations
from citemd.models import CitedAnswer, RetrievedChunk
from citemd.retrieve.rerank import ScoreFn, rerank

# A retriever maps (query, k) -> ranked chunks.
RetrieveFn = Callable[[str, int], Sequence[RetrievedChunk]]

VALID_RETRIEVERS = ("vector", "bm25", "hybrid")
VALID_CONFIDENCE_SOURCES = ("model", "rerank", "blend")


@dataclass
class PipelineConfig:
    """Knobs that define one pipeline configuration (one ablation arm)."""

    retriever: str = "hybrid"
    use_rerank: bool = True
    candidate_k: int = 30
    top_k: int = 5
    allow_abstain: bool = True
    abstain_threshold: float | None = None
    confidence_source: str = "model"
    temperature: float = 0.0

    def __post_init__(self) -> None:
        if self.retriever not in VALID_RETRIEVERS:
            raise ValueError(f"retriever must be one of {VALID_RETRIEVERS}")
        if self.confidence_source not in VALID_CONFIDENCE_SOURCES:
            raise ValueError(f"confidence_source must be one of {VALID_CONFIDENCE_SOURCES}")


@dataclass
class _Deps:
    """Injectable dependencies; defaults wire up the real retrievers and reranker."""

    retrieve_fns: dict[str, RetrieveFn] = field(default_factory=dict)
    scorer: ScoreFn | None = None


def _default_retrieve_fn(name: str) -> RetrieveFn:
    if name == "vector":
        from citemd.retrieve.vector import vector_search

        return lambda q, k: vector_search(q, k=k)
    if name == "bm25":
        from citemd.retrieve.bm25 import bm25_search

        return lambda q, k: bm25_search(q, k=k)
    from citemd.retrieve.hybrid import hybrid_search

    return lambda q, k: hybrid_search(q, k=k)


def _sigmoid(x: float) -> float:
    import math

    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def _retrieval_confidence(contexts: Sequence[RetrievedChunk]) -> float:
    """Map the top context's score to [0, 1].

    Reranker scores are cross-encoder logits, so a sigmoid is the natural squashing. Vector
    scores are already cosine similarities in [-1, 1]; clamp them. This is a heuristic signal
    for selective prediction, not a calibrated probability.
    """
    if not contexts:
        return 0.0
    top = contexts[0]
    if top.retriever == "rerank":
        return _sigmoid(top.score)
    return max(0.0, min(1.0, top.score))


def answer(
    question: str,
    llm: LLMClient,
    *,
    options: dict[str, str] | None = None,
    config: PipelineConfig | None = None,
    retrieve_fn: RetrieveFn | None = None,
    scorer: ScoreFn | None = None,
) -> CitedAnswer:
    """Run the full pipeline for one question and return a CitedAnswer."""
    config = config or PipelineConfig()

    retrieve = retrieve_fn or _default_retrieve_fn(config.retriever)
    candidates = list(retrieve(question, config.candidate_k))

    if config.use_rerank and candidates:
        contexts = rerank(question, candidates, k=config.top_k, scorer=scorer)
    else:
        contexts = candidates[: config.top_k]

    messages = build_messages(
        question,
        contexts,
        options=options,
        allow_abstain=config.allow_abstain,
    )
    raw = llm.complete(messages, temperature=config.temperature)
    parsed = parse_answer(raw, option_keys=list(options.keys()) if options else None)

    model_conf = parsed.confidence
    retr_conf = _retrieval_confidence(contexts)
    if config.confidence_source == "model":
        confidence = model_conf
    elif config.confidence_source == "rerank":
        confidence = retr_conf
    else:  # blend
        confidence = 0.5 * (model_conf + retr_conf)

    abstained = parsed.abstained
    abstain_reason = "model_insufficient" if abstained else ""
    if (
        not abstained
        and config.abstain_threshold is not None
        and confidence < config.abstain_threshold
    ):
        abstained = True
        abstain_reason = "low_confidence"

    citations = resolve_citations(parsed.citation_markers, contexts)

    return CitedAnswer(
        question=question,
        text=parsed.text,
        option=None if abstained else parsed.option,
        abstained=abstained,
        abstain_reason=abstain_reason,
        confidence=confidence,
        citations=citations,
        contexts=list(contexts),
        model=getattr(llm, "model", ""),
        raw_response=raw,
    )


def make_default_llm() -> LLMClient:
    """Construct the default OpenAI-compatible client from settings (KVGate by default)."""
    from citemd.generate.client import OpenAICompatibleClient

    _ = get_settings()  # surface config errors early
    return OpenAICompatibleClient()
