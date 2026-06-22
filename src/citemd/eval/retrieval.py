"""Retrieval evaluation runner.

Runs a retriever over a labeled retrieval set and reports aggregate hit-rate, MRR, and
nDCG. A labeled set is a JSONL file of objects::

    {"query": "...", "relevant_ids": ["pmid:123#0", "pmid:456#2"]}

where ``relevant_ids`` are the gold chunk (or source) ids that should be retrieved. The
retriever is injected as a callable so this runner is unit-testable without a database, and
so the same harness can score vector-only, BM25-only, or hybrid retrieval for the ablation.

Note on scope: MIRAGE itself is a QA benchmark; per-question gold-passage labels are not
part of its distribution. End-to-end QA accuracy with retrieval augmentation is the
milestone-2 headline. This runner measures retrieval quality on any labeled set you supply
(including small hand-built or dataset-derived ones), which is what makes the milestone-1
retriever measurable.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path

from citemd.eval.metrics import aggregate
from citemd.models import RetrievedChunk

# A retriever takes a query and a cutoff and returns ranked chunks.
Retriever = Callable[[str, int], Sequence[RetrievedChunk]]


def load_labels(path: str | Path) -> list[dict]:
    """Load a labeled retrieval set from JSONL."""
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "query" not in obj or "relevant_ids" not in obj:
                raise ValueError("each label row needs 'query' and 'relevant_ids'")
            rows.append(obj)
    return rows


def _ids(chunks: Sequence[RetrievedChunk], *, by_source: bool) -> list[str]:
    return [c.source_id if by_source else c.chunk_id for c in chunks]


def run_retrieval_eval(
    labels: Sequence[dict],
    retriever: Retriever,
    *,
    k: int = 10,
    by_source: bool = False,
) -> dict[str, float]:
    """Score a retriever against a labeled set.

    Args:
        labels: rows of ``{"query", "relevant_ids"}``.
        retriever: callable ``(query, k) -> ranked chunks``.
        k: cutoff for metrics and the number of results requested.
        by_source: match on ``source_id`` instead of ``chunk_id`` (useful when gold labels
            are at document granularity).
    """
    rows = []
    for label in labels:
        retrieved = retriever(label["query"], k)
        rows.append((_ids(retrieved, by_source=by_source), list(label["relevant_ids"])))
    return aggregate(rows, k)
