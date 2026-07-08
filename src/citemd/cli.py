"""CiteMD command-line interface.

Heavy dependencies (database, embeddings, parsers) are imported inside command bodies so
that ``citemd --help`` and the pure-logic commands work with only the core install.
"""

from __future__ import annotations

import glob as _glob
from typing import Optional

import typer

from citemd import __version__
from citemd.config import get_settings

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="CiteMD: clinical-evidence RAG. Research and evaluation only; not for clinical use.",
)
db_app = typer.Typer(no_args_is_help=True, help="Database setup.")
ingest_app = typer.Typer(no_args_is_help=True, help="Ingest documents into the hybrid index.")
eval_app = typer.Typer(no_args_is_help=True, help="MIRAGE eval sets and retrieval metrics.")
app.add_typer(db_app, name="db")
app.add_typer(ingest_app, name="ingest")
app.add_typer(eval_app, name="eval")


@app.command()
def version() -> None:
    """Print the installed CiteMD version."""
    typer.echo(__version__)


@app.command()
def config() -> None:
    """Show effective configuration (secrets redacted)."""
    s = get_settings()
    data = s.model_dump()
    if data.get("llm_api_key"):
        data["llm_api_key"] = "***redacted***"
    for key, val in data.items():
        typer.echo(f"{key} = {val}")


# --- db ----------------------------------------------------------------------------------
@db_app.command("init")
def db_init() -> None:
    """Create the pgvector extension, tables, and indexes."""
    from citemd.db.connection import init_schema

    init_schema()
    typer.echo("Schema initialized.")


# --- ingest --------------------------------------------------------------------------------
@ingest_app.command("pubmed")
def ingest_pubmed(
    query: str = typer.Option(..., "--query", "-q", help="PubMed search query."),
    max_results: int = typer.Option(200, "--max", "-n", help="Maximum abstracts to fetch."),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="NCBI Entrez API key."),
) -> None:
    """Fetch PubMed abstracts and load them into the index."""
    from citemd.ingest.loader import load_documents
    from citemd.ingest.pubmed import fetch_documents

    typer.echo(f"Searching PubMed for: {query!r}")
    docs = fetch_documents(query, max_results=max_results, api_key=api_key)
    typer.echo(f"Fetched {len(docs)} abstracts. Embedding and loading...")
    counts = load_documents(docs)
    typer.echo(f"Loaded {counts['documents']} documents, {counts['chunks']} chunks.")


@ingest_app.command("mirage-sources")
def ingest_mirage_sources(
    datasets: str = typer.Option(
        "pubmedqa,bioasq", "--datasets", help="Comma-separated MIRAGE datasets with PMIDs."
    ),
) -> None:
    """Ingest the source PubMed abstracts (by PMID) behind PubMedQA/BioASQ questions.

    This makes the retrieval corpus contain the gold evidence for those questions (alongside
    the broader distractor corpus), which is what makes the QA eval measure the RAG pipeline
    rather than corpus coverage. Public abstracts only.
    """
    from citemd.eval.mirage import load_dataset
    from citemd.ingest.loader import load_documents
    from citemd.ingest.pubmed import fetch_by_pmids

    data_dir = get_settings().data_dir
    pmids: list[str] = []
    for ds in [d.strip() for d in datasets.split(",") if d.strip()]:
        try:
            items = load_dataset(ds, data_dir)
        except FileNotFoundError:
            typer.echo(f"  {ds}: not fetched, skipping")
            continue
        ds_pmids = [p for it in items for p in it.pmids]
        pmids.extend(ds_pmids)
        typer.echo(f"  {ds}: {len(items)} questions -> {len(ds_pmids)} PMIDs")

    unique = list(dict.fromkeys(pmids))
    if not unique:
        typer.echo("No PMIDs found. Did you run `citemd eval fetch`?")
        raise typer.Exit(code=1)
    typer.echo(f"Fetching {len(unique)} unique source abstracts...")
    docs = fetch_by_pmids(unique)
    typer.echo(f"Fetched {len(docs)} abstracts. Embedding and loading...")
    counts = load_documents(docs)
    typer.echo(f"Loaded {counts['documents']} documents, {counts['chunks']} chunks.")


@ingest_app.command("files")
def ingest_files(
    paths: list[str] = typer.Argument(..., help="Files or globs (.pdf, .html, .md, .txt)."),
) -> None:
    """Parse and load local PDF/HTML/Markdown files."""
    from citemd.ingest.loader import load_documents
    from citemd.ingest.parse import parse_file

    expanded: list[str] = []
    for pattern in paths:
        matched = _glob.glob(pattern)
        expanded.extend(matched if matched else [pattern])

    docs = []
    for path in expanded:
        try:
            docs.append(parse_file(path))
        except (ValueError, OSError) as exc:
            typer.echo(f"  skip {path}: {exc}")
    if not docs:
        typer.echo("No parseable files found.")
        raise typer.Exit(code=1)
    typer.echo(f"Parsed {len(docs)} files. Embedding and loading...")
    counts = load_documents(docs)
    typer.echo(f"Loaded {counts['documents']} documents, {counts['chunks']} chunks.")


# --- query ---------------------------------------------------------------------------------
@app.command()
def query(
    text: str = typer.Argument(..., help="Question to retrieve evidence for."),
    k: int = typer.Option(5, "--k", help="Number of chunks to return."),
    retriever: str = typer.Option(
        "hybrid", "--retriever", help="hybrid | vector | bm25 (for ablation)."
    ),
) -> None:
    """Retrieve and rank evidence for a question (retrieval only; use `ask` to generate)."""
    results = _retrieve(retriever, text, k)
    if not results:
        typer.echo("No results. Is the index populated?")
        raise typer.Exit(code=1)
    for i, r in enumerate(results, start=1):
        loc = " / ".join(part for part in [r.title, r.section] if part)
        page = f" p.{r.page}" if r.page else ""
        typer.echo(f"\n[{i}] score={r.score:.4f}  {loc}{page}  ({r.source_id})")
        snippet = r.text.strip().replace("\n", " ")
        typer.echo(f"    {snippet[:280]}")


@app.command()
def ask(
    text: str = typer.Argument(..., help="Clinical question to answer with cited evidence."),
    retriever: str = typer.Option("hybrid", "--retriever", help="hybrid | vector | bm25."),
    rerank: bool = typer.Option(True, "--rerank/--no-rerank", help="Cross-encoder reranking."),
    k: int = typer.Option(5, "--k", help="Passages given to the model."),
    abstain_threshold: Optional[float] = typer.Option(
        None, "--abstain-threshold", help="Abstain when confidence < this (0..1)."
    ),
) -> None:
    """Answer a question with grounded citations via the LLM backend (KVGate by default).

    Generation calls the configured backend and may incur API cost. Set CITEMD_LLM_* to point
    at your endpoint/key. Research and evaluation only; not for clinical use.
    """
    from citemd.pipeline import PipelineConfig, answer, make_default_llm

    config = PipelineConfig(
        retriever=retriever,
        use_rerank=rerank,
        top_k=k,
        allow_abstain=True,
        abstain_threshold=abstain_threshold,
    )
    result = answer(text, make_default_llm(), config=config)
    if result.abstained:
        typer.echo(f"ABSTAINED ({result.abstain_reason}); confidence={result.confidence:.2f}")
        typer.echo("Insufficient evidence in the retrieved passages to answer reliably.")
        raise typer.Exit(code=0)
    typer.echo(f"\n{result.text or result.option}\n")
    typer.echo(f"confidence={result.confidence:.2f}  model={result.model}")
    if result.citations:
        typer.echo("Citations:")
        for c in result.citations:
            loc = " / ".join(part for part in [c.title, c.section] if part)
            page = f" p.{c.page}" if c.page else ""
            typer.echo(f"  [{c.marker}] {loc}{page} ({c.source_id})")


def _retrieve(retriever: str, text: str, k: int):
    if retriever == "vector":
        from citemd.retrieve.vector import vector_search

        return vector_search(text, k=k)
    if retriever == "bm25":
        from citemd.retrieve.bm25 import bm25_search

        return bm25_search(text, k=k)
    if retriever == "hybrid":
        from citemd.retrieve.hybrid import hybrid_search

        return hybrid_search(text, k=k)
    raise typer.BadParameter("retriever must be one of: hybrid, vector, bm25")


# --- eval ----------------------------------------------------------------------------------
@eval_app.command("fetch")
def eval_fetch() -> None:
    """Download the MIRAGE benchmark and write one JSONL per dataset."""
    from citemd.eval.mirage import fetch_benchmark

    counts = fetch_benchmark(get_settings().data_dir)
    for dataset, n in sorted(counts.items()):
        typer.echo(f"  {dataset}: {n} questions")
    typer.echo("MIRAGE fetched.")


@eval_app.command("stats")
def eval_stats() -> None:
    """Show how many MIRAGE questions are loaded locally per dataset."""
    from citemd.eval.mirage import DATASETS, load_dataset

    data_dir = get_settings().data_dir
    any_found = False
    for dataset in DATASETS:
        try:
            items = load_dataset(dataset, data_dir)
        except FileNotFoundError:
            typer.echo(f"  {dataset}: not fetched")
            continue
        any_found = True
        typer.echo(f"  {dataset}: {len(items)} questions")
    if not any_found:
        typer.echo("Nothing fetched yet. Run `citemd eval fetch`.")
        raise typer.Exit(code=1)


@eval_app.command("retrieval")
def eval_retrieval(
    labels: str = typer.Option(..., "--labels", help="Labeled JSONL: {query, relevant_ids}."),
    k: int = typer.Option(10, "--k", help="Cutoff for hit-rate/MRR/nDCG."),
    retriever: str = typer.Option("hybrid", "--retriever", help="hybrid | vector | bm25."),
    by_source: bool = typer.Option(
        False, "--by-source", help="Match gold labels at document (source_id) granularity."
    ),
) -> None:
    """Score a retriever against a labeled retrieval set."""
    from citemd.eval.retrieval import load_labels, run_retrieval_eval

    rows = load_labels(labels)

    def _retriever(qtext: str, cutoff: int):
        return _retrieve(retriever, qtext, cutoff)

    metrics = run_retrieval_eval(rows, _retriever, k=k, by_source=by_source)
    typer.echo(f"retriever={retriever}  n={metrics.get('n')}")
    for key, val in metrics.items():
        if key == "n":
            continue
        typer.echo(f"  {key}: {val:.4f}")


@eval_app.command("qa")
def eval_qa(
    dataset: str = typer.Option(..., "--dataset", help="medqa|medmcqa|pubmedqa|bioasq|mmlu."),
    sample: Optional[int] = typer.Option(
        None, "--sample", help="Evaluate a deterministic random N-question subset (representative)."
    ),
    seed: int = typer.Option(0, "--seed", help="Sampling seed (reproducible subsets)."),
    limit: Optional[int] = typer.Option(
        None, "--limit", help="First N in file order (debug only; MIRAGE is class-ordered)."
    ),
    retriever: str = typer.Option("hybrid", "--retriever", help="hybrid | vector | bm25."),
    rerank: bool = typer.Option(True, "--rerank/--no-rerank", help="Cross-encoder reranking."),
    out: Optional[str] = typer.Option(None, "--out", help="JSONL artifact (append + resume)."),
    abstain_fraction: float = typer.Option(
        0.2, "--abstain-fraction", help="Report metrics abstaining on this riskiest fraction."
    ),
) -> None:
    """Run the QA pipeline over a MIRAGE dataset and report accuracy + the abstention tradeoff.

    Calls the LLM backend once per question (may incur API cost). Results are cached to --out
    and re-runs resume, so a stopped run is never repeated.
    """
    from citemd.eval.mirage import load_dataset
    from citemd.eval.qa import (
        accuracy_ci,
        retrieval_hit_rate,
        run_qa,
        score_records,
        selective_items,
    )
    from citemd.eval.selective import summarize_abstention
    from citemd.eval.stats import calibration
    from citemd.pipeline import PipelineConfig, make_default_llm

    items = load_dataset(dataset, get_settings().data_dir, limit=limit, sample=sample, seed=seed)
    config = PipelineConfig(retriever=retriever, use_rerank=rerank, allow_abstain=False)
    out_path = out or f"{get_settings().artifacts_dir}/qa_{dataset}_{retriever}.jsonl"

    typer.echo(
        f"Evaluating {len(items)} {dataset} questions "
        f"(retriever={retriever}, rerank={rerank})..."
    )
    records = run_qa(items, make_default_llm(), config=config, out_path=out_path)
    m = score_records(records)
    acc = accuracy_ci(records)
    hit = retrieval_hit_rate(records)
    ece = calibration(selective_items(records)).ece
    typer.echo(f"\nn={m['n']}  accuracy={acc.as_pct()}")
    typer.echo(f"coverage={m['coverage']:.3f}  accuracy_answered={m['accuracy_answered']:.3f}")
    if hit.n:
        typer.echo(f"retrieval hit-rate (gold source in context)={hit.as_pct()}")
    typer.echo(f"calibration ECE={ece:.3f} (lower is better)")
    abst = summarize_abstention(selective_items(records), abstain_fraction)
    typer.echo(
        f"abstain riskiest {int(round(abstain_fraction * 100))}%: "
        f"error {abst['baseline_error']:.3f} -> {abst['kept_error']:.3f} "
        f"({abst['error_reduction_rel'] * 100:.1f}% relative reduction) "
        f"at coverage {abst['coverage']:.2f}"
    )


@eval_app.command("ablation")
def eval_ablation(
    dataset: str = typer.Option(..., "--dataset", help="MIRAGE dataset to run the ablation on."),
    sample: Optional[int] = typer.Option(
        None, "--sample", help="Deterministic random N-question subset (representative)."
    ),
    seed: int = typer.Option(0, "--seed", help="Sampling seed (reproducible subsets)."),
    limit: Optional[int] = typer.Option(
        None, "--limit", help="First N in file order (debug only; MIRAGE is class-ordered)."
    ),
    out_dir: Optional[str] = typer.Option(None, "--out-dir", help="Directory for QA artifacts."),
    abstain_fraction: float = typer.Option(0.2, "--abstain-fraction", help="Abstention cutoff."),
    charts: bool = typer.Option(False, "--charts", help="Also write evaluation charts."),
) -> None:
    """Run the full vector -> hybrid -> +rerank -> +abstention ablation and print the table."""
    import json as _json
    from pathlib import Path as _Path

    from citemd.eval.ablation import format_table, run_ablation
    from citemd.eval.mirage import load_dataset
    from citemd.eval.qa import accuracy_ci, retrieval_hit_rate, selective_items
    from citemd.eval.selective import risk_coverage_curve
    from citemd.eval.stats import calibration
    from citemd.pipeline import make_default_llm

    items = load_dataset(dataset, get_settings().data_dir, limit=limit, sample=sample, seed=seed)
    base = out_dir or f"{get_settings().artifacts_dir}/ablation_{dataset}"
    typer.echo(f"Running ablation on {len(items)} {dataset} questions...")
    result = run_ablation(
        items, make_default_llm(), out_dir=base, abstain_fraction=abstain_fraction
    )
    typer.echo("\n" + format_table(result["rows"]))

    # Per-arm accuracy CIs, retrieval hit-rate, and calibration.
    typer.echo("\nper-arm detail (95% CI):")
    for slug, recs in result["records"].items():
        acc = accuracy_ci(recs)
        hit = retrieval_hit_rate(recs)
        ece = calibration(selective_items(recs)).ece
        hit_s = f"  hit-rate {hit.as_pct()}" if hit.n else ""
        typer.echo(f"  {slug:<14} accuracy {acc.as_pct()}  ECE {ece:.3f}{hit_s}")

    # Machine-readable summary (consumed by the frontend / reproducible report).
    best = result["derived_from"]
    summary = {
        "dataset": dataset,
        "sample": sample,
        "seed": seed,
        "abstain_fraction": abstain_fraction,
        "model": get_settings().llm_model,
        "best_arm": best,
        "rows": result["rows"],
        "arms": {
            slug: {
                "accuracy": accuracy_ci(recs).__dict__,
                "retrieval_hit_rate": retrieval_hit_rate(recs).__dict__,
                "ece": calibration(selective_items(recs)).ece,
                "risk_coverage": [p.__dict__ for p in risk_coverage_curve(selective_items(recs))],
            }
            for slug, recs in result["records"].items()
        },
    }
    _Path(base).mkdir(parents=True, exist_ok=True)
    (_Path(base) / "summary.json").write_text(_json.dumps(summary, indent=2))
    typer.echo(f"\nSummary: {base}/summary.json")

    if charts:
        from citemd.eval.charts import plot_ablation, plot_calibration, plot_risk_coverage

        best_items = selective_items(result["records"][best])
        rc = plot_risk_coverage(best_items, f"{base}/risk_coverage.png")
        cal = plot_calibration(best_items, f"{base}/calibration.png")
        ab = plot_ablation(result["rows"], f"{base}/ablation.png")
        typer.echo(f"Charts: {rc}  {cal}  {ab}")


@eval_app.command("faithfulness")
def eval_faithfulness(
    dataset: str = typer.Option(..., "--dataset", help="MIRAGE dataset (needs source ingest)."),
    sample: Optional[int] = typer.Option(
        60, "--sample", help="Deterministic random N-question subset (representative)."
    ),
    seed: int = typer.Option(0, "--seed", help="Sampling seed."),
    judge_model: Optional[str] = typer.Option(
        None, "--judge-model", help="Judge model id (default: same as generation)."
    ),
    out: Optional[str] = typer.Option(None, "--out", help="JSONL artifact (append + resume)."),
) -> None:
    """Judge whether each answer's cited passages actually support it (citation faithfulness).

    Calls the LLM once to answer and once per cited passage to judge (may incur API cost). Using
    the same model to judge its own citations is a known bias; pass --judge-model to vary it.
    """
    from citemd.eval.faithfulness import run_faithfulness, score_faithfulness
    from citemd.eval.mirage import load_dataset
    from citemd.generate.client import OpenAICompatibleClient
    from citemd.pipeline import PipelineConfig, make_default_llm

    items = load_dataset(dataset, get_settings().data_dir, sample=sample, seed=seed)
    llm = make_default_llm()
    judge = OpenAICompatibleClient(model=judge_model) if judge_model else llm
    out_path = out or f"{get_settings().artifacts_dir}/faithfulness_{dataset}.jsonl"

    typer.echo(f"Judging citation faithfulness on {len(items)} {dataset} answers...")
    records = run_faithfulness(
        items, llm, judge, config=PipelineConfig(allow_abstain=False), out_path=out_path
    )
    s = score_faithfulness(records)
    typer.echo(f"\nn={s['n']}  judge={getattr(judge, 'model', '')}")
    typer.echo(f"citation coverage (answers citing >=1 passage): {s['coverage'].as_pct()}")
    typer.echo(f"answer support rate (>=1 cited supports): {s['answer_support_rate'].as_pct()}")
    typer.echo(f"citation support rate (per cited passage): {s['citation_support_rate'].as_pct()}")


if __name__ == "__main__":  # pragma: no cover
    app()
