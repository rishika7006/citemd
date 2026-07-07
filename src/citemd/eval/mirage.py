"""MIRAGE benchmark loading.

MIRAGE (arXiv:2402.13178, github.com/Teddy-XiongGZ/MIRAGE) is the recognized standard
benchmark for medical RAG. It bundles five QA datasets:

  - medqa     (MedQA-US, multiple choice)
  - medmcqa   (MedMCQA, multiple choice)
  - pubmedqa  (PubMedQA, yes/no/maybe)
  - bioasq    (BioASQ-Y/N, yes/no)
  - mmlu      (MMLU medical subsets, multiple choice)

The canonical distribution is a single ``benchmark.json`` whose top level maps each
dataset name to a dict of ``id -> {question, options, answer}``. This module fetches that
file and normalizes it into a flat, versioned local format (one JSONL file per dataset),
keeping the network fetch separate from a pure parser so the parser is unit-testable.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

import httpx
from pydantic import BaseModel, Field

MIRAGE_BENCHMARK_URL = (
    "https://raw.githubusercontent.com/Teddy-XiongGZ/MIRAGE/main/benchmark.json"
)

DATASETS = ("medqa", "medmcqa", "pubmedqa", "bioasq", "mmlu")


class QuestionItem(BaseModel):
    """A single normalized MIRAGE question."""

    dataset: str
    qid: str
    question: str
    options: dict[str, str] = Field(default_factory=dict)
    answer: str = Field(default="", description="Gold answer; option key or yes/no/maybe.")
    pmids: list[str] = Field(
        default_factory=list,
        description="Source PubMed IDs (PubMedQA/BioASQ); used to ingest gold evidence.",
    )
    metadata: dict = Field(default_factory=dict)


def parse_benchmark(data: dict) -> dict[str, list[QuestionItem]]:
    """Normalize a raw MIRAGE benchmark mapping into QuestionItems per dataset.

    Tolerant of minor field-name variation: accepts ``answer`` or ``answer_idx`` for the
    gold label and ``question``/``query`` for the prompt. Unknown top-level keys are kept
    so the loader does not silently drop a dataset MIRAGE may add later.
    """
    out: dict[str, list[QuestionItem]] = {}
    for dataset, entries in data.items():
        if not isinstance(entries, dict):
            continue
        items: list[QuestionItem] = []
        for qid, entry in entries.items():
            if not isinstance(entry, dict):
                continue
            question = entry.get("question") or entry.get("query") or ""
            options = entry.get("options") or {}
            if isinstance(options, list):
                # Some variants ship options as a list; key them A, B, C, ...
                options = {chr(65 + i): str(v) for i, v in enumerate(options)}
            else:
                options = {str(key): str(val) for key, val in options.items()}
            answer = entry.get("answer")
            if answer is None:
                answer = entry.get("answer_idx", "")
            raw_pmids = entry.get("PMID") or entry.get("pmid") or []
            if not isinstance(raw_pmids, list):
                raw_pmids = [raw_pmids]
            pmids = [str(p) for p in raw_pmids if p not in (None, "")]
            items.append(
                QuestionItem(
                    dataset=dataset,
                    qid=str(qid),
                    question=str(question),
                    options=options,
                    answer=str(answer),
                    pmids=pmids,
                )
            )
        out[dataset] = items
    return out


def _dataset_path(data_dir: str | Path, dataset: str) -> Path:
    return Path(data_dir) / "mirage" / f"{dataset}.jsonl"


def fetch_benchmark(
    data_dir: str | Path,
    *,
    url: str = MIRAGE_BENCHMARK_URL,
    timeout: float = 60.0,
) -> dict[str, int]:
    """Download MIRAGE, normalize it, and write one JSONL file per dataset.

    Returns a mapping of ``dataset -> count`` for what was written.
    """
    resp = httpx.get(url, timeout=timeout, follow_redirects=True)
    resp.raise_for_status()
    parsed = parse_benchmark(resp.json())

    out_dir = Path(data_dir) / "mirage"
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for dataset, items in parsed.items():
        path = _dataset_path(data_dir, dataset)
        with path.open("w", encoding="utf-8") as fh:
            for item in items:
                fh.write(item.model_dump_json() + "\n")
        counts[dataset] = len(items)
    return counts


def load_dataset(
    dataset: str,
    data_dir: str | Path,
    *,
    limit: Optional[int] = None,
    sample: Optional[int] = None,
    seed: int = 0,
) -> list[QuestionItem]:
    """Load a previously fetched MIRAGE dataset from local JSONL.

    ``sample`` draws a deterministic random subset of that size (seeded by ``seed``) and is
    the right way to take a representative slice: the MIRAGE datasets are class-ordered (e.g.
    every PubMedQA "yes" precedes every "no"), so ``limit`` (the first N rows) yields a
    single-class slice and must not be used for evaluation. ``limit`` is kept only for quick,
    order-preserving debugging.
    """
    path = _dataset_path(data_dir, dataset)
    if not path.exists():
        raise FileNotFoundError(
            f"MIRAGE dataset '{dataset}' not found at {path}. Run `citemd eval fetch` first."
        )
    items: list[QuestionItem] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            items.append(QuestionItem(**json.loads(line)))
            if sample is None and limit is not None and len(items) >= limit:
                break
    if sample is not None and sample < len(items):
        items = random.Random(seed).sample(items, sample)
    return items
