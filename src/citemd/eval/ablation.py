"""Ablation: vector-only -> hybrid -> +rerank -> +abstention.

This is the centerpiece result. The first three arms differ only in retrieval, so each is a
separate pipeline run (different evidence reaches the model). The fourth arm, +abstention, is
*derived* from the +rerank arm by declining to answer the least-confident questions: it is an
operating point on that arm's risk-coverage curve, not another LLM run. Reporting it this way
is honest about what abstention does and adds no cost.

All arms force a committed answer (``allow_abstain=False``) so every question yields a
prediction and a confidence; the abstention arm then applies the cutoff offline.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from citemd.eval.mirage import QuestionItem
from citemd.eval.qa import QARecord, run_qa, score_records, selective_items
from citemd.eval.selective import summarize_abstention
from citemd.generate.client import LLMClient
from citemd.pipeline import PipelineConfig, RetrieveFn
from citemd.retrieve.rerank import ScoreFn


@dataclass
class Arm:
    """One ablation arm: a display name, a slug for its artifact, and a pipeline config."""

    name: str
    slug: str
    config: PipelineConfig


def default_arms() -> list[Arm]:
    """The three retrieval arms (the abstention arm is derived from the last)."""
    return [
        Arm("vector-only", "vector", PipelineConfig(
            retriever="vector", use_rerank=False, allow_abstain=False)),
        Arm("hybrid", "hybrid", PipelineConfig(
            retriever="hybrid", use_rerank=False, allow_abstain=False)),
        Arm("hybrid+rerank", "hybrid_rerank", PipelineConfig(
            retriever="hybrid", use_rerank=True, allow_abstain=False)),
    ]


def run_ablation(
    items: Sequence[QuestionItem],
    llm: LLMClient,
    *,
    out_dir: str | Path,
    arms: Optional[Sequence[Arm]] = None,
    abstain_fraction: float = 0.2,
    resume: bool = True,
    retrieve_fns: Optional[dict[str, RetrieveFn]] = None,
    scorer: Optional[ScoreFn] = None,
) -> dict:
    """Run every retrieval arm, then derive the +abstention row from the strongest arm.

    Returns a dict with ``rows`` (one per arm, ready to tabulate), the raw ``records`` per
    arm slug, and ``abstention`` (the summary used for the headline number).
    """
    arms = list(arms or default_arms())
    out_dir = Path(out_dir)
    retrieve_fns = retrieve_fns or {}

    records_by_arm: dict[str, list[QARecord]] = {}
    rows: list[dict] = []
    for arm in arms:
        records = run_qa(
            items,
            llm,
            config=arm.config,
            out_path=out_dir / f"qa_{arm.slug}.jsonl",
            resume=resume,
            retrieve_fn=retrieve_fns.get(arm.config.retriever),
            scorer=scorer,
        )
        records_by_arm[arm.slug] = records
        m = score_records(records)
        rows.append({
            "arm": arm.name,
            "n": m["n"],
            "accuracy": m["accuracy"],
            "error": 1.0 - m["accuracy"],
            "coverage": m["coverage"],
        })

    # +abstention: derive from the last arm (strongest retrieval), abstaining on the riskiest
    # fraction by confidence.
    last = arms[-1]
    items_cb = selective_items(records_by_arm[last.slug])
    abst = summarize_abstention(items_cb, abstain_fraction)
    rows.append({
        "arm": f"{last.name} +abstention@{int(round(abstain_fraction * 100))}%",
        "n": abst["n_kept"],
        "accuracy": 1.0 - abst["kept_error"],
        "error": abst["kept_error"],
        "coverage": abst["coverage"],
    })

    return {
        "rows": rows,
        "records": records_by_arm,
        "abstention": abst,
        "abstain_fraction": abstain_fraction,
        "derived_from": last.slug,
    }


def format_table(rows: Sequence[dict]) -> str:
    """Render ablation rows as a fixed-width text table."""
    header = f"{'arm':<34} {'n':>6} {'accuracy':>9} {'error':>7} {'coverage':>9}"
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r['arm']:<34} {r['n']:>6} {r['accuracy']:>9.3f} "
            f"{r['error']:>7.3f} {r['coverage']:>9.3f}"
        )
    return "\n".join(lines)
