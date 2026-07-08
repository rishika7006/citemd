"""MIRAGE question-answering harness.

Runs the RAG pipeline over MIRAGE multiple-choice questions and records, per question, the
committed option, whether it was correct, the confidence signal, and citation/grounding
provenance. Results are written to a JSONL artifact as they are produced and re-runs resume
from it, so a long or paid evaluation can be stopped and continued and is never repeated.

Scoring is exact option-key match against the gold answer, the standard MIRAGE/MedRAG setup.
The selective-prediction curve (abstention tradeoff) is computed separately in
:mod:`citemd.eval.selective` from the per-question confidences this harness records.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from citemd.eval.mirage import QuestionItem
from citemd.generate.client import LLMClient
from citemd.models import RetrievedChunk
from citemd.pipeline import PipelineConfig, RetrieveFn, answer
from citemd.retrieve.rerank import ScoreFn


class QARecord(BaseModel):
    """The evaluation outcome for a single question."""

    dataset: str
    qid: str
    gold: str
    predicted: Optional[str] = None
    abstained: bool = False
    abstain_reason: str = ""
    correct: bool = False
    confidence: float = 0.0
    top_context_score: float = 0.0
    n_citations: int = 0
    n_contexts: int = 0
    # Retrieval provenance: did the question's gold source abstract(s) reach the context, and
    # at what rank. Lets the eval report retrieval hit-rate, showing the source is found among
    # distractors rather than handed to the model.
    gold_pmids: list[str] = Field(default_factory=list)
    gold_retrieved: bool = False
    gold_rank: Optional[int] = None
    model: str = ""


def _top_context_score(contexts: Sequence[RetrievedChunk]) -> float:
    return float(contexts[0].score) if contexts else 0.0


def _gold_retrieval(
    pmids: Sequence[str], contexts: Sequence[RetrievedChunk]
) -> tuple[bool, Optional[int]]:
    """Whether any gold source (by PMID) reached the context, and the 1-based rank of the first.

    Chunk ``source_id`` is ``pmid:<PMID>``; question ``pmids`` are bare PMIDs. Returns
    ``(False, None)`` when the question has no PMIDs or none were retrieved.
    """
    gold = {f"pmid:{p}" for p in pmids}
    if not gold:
        return False, None
    for rank, c in enumerate(contexts, start=1):
        if c.source_id in gold:
            return True, rank
    return False, None


def _load_done(path: Path) -> dict[str, QARecord]:
    done: dict[str, QARecord] = {}
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = QARecord(**json.loads(line))
            done[f"{rec.dataset}:{rec.qid}"] = rec
    return done


def run_qa(
    items: Sequence[QuestionItem],
    llm: LLMClient,
    *,
    config: Optional[PipelineConfig] = None,
    out_path: Optional[str | Path] = None,
    resume: bool = True,
    retrieve_fn: Optional[RetrieveFn] = None,
    scorer: Optional[ScoreFn] = None,
    on_record: Optional[Callable[[QARecord], None]] = None,
) -> list[QARecord]:
    """Evaluate the pipeline over MIRAGE questions, caching results to ``out_path``.

    Args:
        config: pipeline configuration (one ablation arm). Defaults to hybrid + rerank with
            forced answers (``allow_abstain=False``), which yields a prediction and confidence
            for every question so the abstention curve can be derived offline.
        out_path: JSONL artifact to append results to (and resume from).
        on_record: optional callback per finished question, e.g. to drive a progress bar.
    """
    config = config or PipelineConfig(allow_abstain=False)
    path = Path(out_path) if out_path else None
    done = _load_done(path) if (path and resume) else {}

    records: list[QARecord] = []
    fh = None
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = path.open("a", encoding="utf-8")
    try:
        for item in items:
            key = f"{item.dataset}:{item.qid}"
            if key in done:
                records.append(done[key])
                continue
            ans = answer(
                item.question,
                llm,
                options=item.options,
                config=config,
                retrieve_fn=retrieve_fn,
                scorer=scorer,
            )
            predicted = ans.option
            correct = (not ans.abstained) and predicted is not None and predicted == item.answer
            gold_retrieved, gold_rank = _gold_retrieval(item.pmids, ans.contexts)
            rec = QARecord(
                dataset=item.dataset,
                qid=item.qid,
                gold=item.answer,
                predicted=predicted,
                abstained=ans.abstained,
                abstain_reason=ans.abstain_reason,
                correct=correct,
                confidence=ans.confidence,
                top_context_score=_top_context_score(ans.contexts),
                n_citations=len(ans.citations),
                n_contexts=len(ans.contexts),
                gold_pmids=list(item.pmids),
                gold_retrieved=gold_retrieved,
                gold_rank=gold_rank,
                model=ans.model,
            )
            records.append(rec)
            if fh is not None:
                fh.write(rec.model_dump_json() + "\n")
                fh.flush()
            if on_record is not None:
                on_record(rec)
    finally:
        if fh is not None:
            fh.close()
    return records


def score_records(records: Sequence[QARecord]) -> dict[str, float]:
    """Aggregate accuracy and coverage over QA records.

    - accuracy: correct / total (abstentions count as not correct).
    - coverage: answered / total.
    - accuracy_answered: correct / answered (accuracy when the system commits).
    """
    n = len(records)
    if n == 0:
        return {"n": 0, "accuracy": 0.0, "coverage": 0.0, "accuracy_answered": 0.0}
    answered = [r for r in records if not r.abstained]
    n_ans = len(answered)
    correct = sum(1 for r in records if r.correct)
    return {
        "n": n,
        "accuracy": correct / n,
        "coverage": n_ans / n,
        "accuracy_answered": (sum(1 for r in answered if r.correct) / n_ans) if n_ans else 0.0,
    }


def selective_items(records: Sequence[QARecord]) -> list[tuple[float, bool]]:
    """Extract ``(confidence, correct)`` pairs for the selective-prediction curve."""
    return [(r.confidence, r.correct) for r in records]


def retrieval_hit_rate(records: Sequence[QARecord]):
    """Fraction of questions (that have a gold PMID) whose gold source reached the context.

    Returns a Wilson-interval Proportion. Questions without a gold PMID are excluded, since
    hit-rate is undefined for them.
    """
    from citemd.eval.stats import wilson_interval

    with_gold = [r for r in records if r.gold_pmids]
    hits = sum(1 for r in with_gold if r.gold_retrieved)
    return wilson_interval(hits, len(with_gold))


def accuracy_ci(records: Sequence[QARecord]):
    """Accuracy as a Wilson-interval Proportion over all records (abstentions count wrong)."""
    from citemd.eval.stats import wilson_interval

    return wilson_interval(sum(1 for r in records if r.correct), len(records))
