"""Citation faithfulness: do the cited passages actually support the answer?

Counting citations is not enough for a trustworthiness claim. This module measures whether
each cited passage provides evidence for the chosen answer, using an LLM judge, and reports:

  - coverage: fraction of answered questions that cite at least one passage.
  - citation support rate: fraction of (answer, cited-passage) pairs the judge calls supporting.
  - answer support rate: fraction of answered-with-citation questions where at least one cited
    passage supports the answer.

The judge is a separate LLM call and is injectable, so the whole thing is unit-testable with a
fake and the judge model can be varied (a judge at least as strong as the generator is
preferable; using the same model to check its own citations is a known limitation and is
reported as such). Results are cached to JSONL and resume, like the QA harness.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from citemd.eval.mirage import QuestionItem
from citemd.generate.client import LLMClient
from citemd.pipeline import PipelineConfig, RetrieveFn, answer
from citemd.retrieve.rerank import ScoreFn

_JUDGE_SYSTEM = (
    "You check whether a passage provides evidence supporting a proposed answer to a medical "
    "question. Judge only what the passage states. Reply with a single word: yes or no."
)


class FaithfulnessRecord(BaseModel):
    """Citation-support outcome for one answered question."""

    dataset: str
    qid: str
    option: Optional[str] = None
    n_cited: int = 0
    n_supported: int = 0
    supported: bool = False  # at least one cited passage supports the answer
    judge_model: str = ""


def _judge_prompt(question: str, answer_text: str, passage: str) -> list[dict[str, str]]:
    user = (
        f"Question: {question.strip()}\n\n"
        f"Proposed answer: {answer_text.strip()}\n\n"
        f"Passage:\n{' '.join(passage.split())}\n\n"
        "Does the passage provide evidence supporting the proposed answer? Answer yes or no."
    )
    return [{"role": "system", "content": _JUDGE_SYSTEM}, {"role": "user", "content": user}]


def judge_support(
    question: str, answer_text: str, passage: str, judge: LLMClient, *, temperature: float = 0.0
) -> bool:
    """Ask the judge whether ``passage`` supports ``answer_text``; parse a yes/no reply."""
    reply = judge.complete(_judge_prompt(question, answer_text, passage), temperature=temperature)
    token = reply.strip().lower().lstrip("*").strip()
    return token.startswith("yes")


def _load_done(path: Optional[Path]) -> dict[str, FaithfulnessRecord]:
    done: dict[str, FaithfulnessRecord] = {}
    if path is None or not path.exists():
        return done
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = FaithfulnessRecord(**json.loads(line))
            done[f"{rec.dataset}:{rec.qid}"] = rec
    return done


def run_faithfulness(
    items: Sequence[QuestionItem],
    llm: LLMClient,
    judge: LLMClient,
    *,
    config: Optional[PipelineConfig] = None,
    out_path: Optional[str | Path] = None,
    resume: bool = True,
    retrieve_fn: Optional[RetrieveFn] = None,
    scorer: Optional[ScoreFn] = None,
    on_record: Optional[Callable[[FaithfulnessRecord], None]] = None,
) -> list[FaithfulnessRecord]:
    """Answer each question, then judge whether its cited passages support the chosen answer."""
    config = config or PipelineConfig(allow_abstain=False)
    path = Path(out_path) if out_path else None
    done = _load_done(path) if (path and resume) else {}

    records: list[FaithfulnessRecord] = []
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
                item.question, llm, options=item.options, config=config,
                retrieve_fn=retrieve_fn, scorer=scorer,
            )
            if item.options:
                answer_text = item.options.get(ans.option, ans.option or "")
            else:
                answer_text = ans.text
            n_supported = 0
            for c in ans.citations:
                passage = next((x.text for x in ans.contexts if x.chunk_id == c.chunk_id), "")
                if passage and judge_support(item.question, answer_text, passage, judge):
                    n_supported += 1
            rec = FaithfulnessRecord(
                dataset=item.dataset,
                qid=item.qid,
                option=ans.option,
                n_cited=len(ans.citations),
                n_supported=n_supported,
                supported=n_supported > 0,
                judge_model=getattr(judge, "model", ""),
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


def score_faithfulness(records: Sequence[FaithfulnessRecord]) -> dict:
    """Aggregate coverage, citation support rate, and answer support rate (with CIs)."""
    from citemd.eval.stats import wilson_interval

    n = len(records)
    cited = [r for r in records if r.n_cited > 0]
    total_cited = sum(r.n_cited for r in records)
    total_supported = sum(r.n_supported for r in records)
    coverage = wilson_interval(len(cited), n)
    answer_support = wilson_interval(sum(1 for r in cited if r.supported), len(cited))
    citation_support = wilson_interval(total_supported, total_cited)
    return {
        "n": n,
        "coverage": coverage,
        "answer_support_rate": answer_support,
        "citation_support_rate": citation_support,
    }
