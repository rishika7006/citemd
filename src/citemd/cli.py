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
    """Retrieve evidence for a question (no generation yet; that is milestone 2)."""
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


if __name__ == "__main__":  # pragma: no cover
    app()
