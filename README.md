# CiteMD

CiteMD is an open-source clinical-evidence question-answering system. It answers medical
questions with grounded, cited answers drawn from the medical literature, and it abstains
when the retrieved evidence is insufficient instead of guessing. The project's headline is
measured trustworthiness: a faithfulness, hallucination, and abstention evaluation harness
run on the standard MIRAGE benchmark, plus multimodal retrieval from clinical tables and
figures.

Generation is served through a configurable LLM client that points at
[KVGate](https://github.com/rishika7006/kvgate) by default and can call a hosted API
directly as a fallback.

> Research and evaluation only. CiteMD is NOT a clinical tool and must NOT be used for
> diagnosis, treatment, or any patient-care decision. It uses public, non-PHI data only.

## Status

Milestone 1 in progress: ingestion, hybrid index, and the MIRAGE evaluation set. The
generation pipeline, abstention gate, headline eval, UI, and observability arrive in later
milestones. See [the roadmap](#roadmap).

## What makes it different

This is not a novel research algorithm and does not claim to be. It is a rigorously
evaluated, reproducible artifact with a sharp angle on two things that medical RAG systems
usually handle poorly:

- **Calibrated abstention.** It declines to answer when evidence is thin and reports the
  answer-versus-abstain tradeoff honestly rather than hiding it.
- **Multimodal evidence.** It retrieves from tables and figures in clinical PDFs, not only
  prose.

Prior art is acknowledged, not ignored: MIRAGE and the MedRAG toolkit
([arXiv:2402.13178](https://arxiv.org/abs/2402.13178)) are the recognized standard for
medical RAG, and CiteMD evaluates on MIRAGE deliberately.

## Architecture

Two pipelines plus an evaluation harness.

- **Ingestion (offline):** documents (PDF, HTML, Markdown, PubMed abstracts) are parsed and
  chunked with source, page, and section metadata, embedded, and stored in a hybrid index
  (Postgres with pgvector for dense vectors and metadata, Postgres full-text search for
  BM25, plus access-control tags).
- **Query (online):** a question is retrieved against both the vector and BM25 indexes,
  fused with Reciprocal Rank Fusion, reranked with a cross-encoder, answered with inline
  citations via the LLM client, then groundedness-checked and abstained on if the evidence
  is insufficient.
- **Evaluation (offline):** MIRAGE QA sets are run through the pipeline to compute retrieval
  quality (hit rate, MRR, nDCG), faithfulness, citation accuracy, hallucination rate, and
  the abstention tradeoff curve, producing the ablation table and charts.

## Install

CiteMD uses a light core so the pure-logic components install and test anywhere with no GPU
or database. Heavier capabilities are optional extras.

```bash
pip install -e .              # core: CLI, config, RRF, metrics, MIRAGE loader
pip install -e ".[all]"       # adds Postgres, embeddings/reranker, ingestion parsers
pip install -e ".[dev]"       # test and lint tooling
```

| Extra     | Adds                                                  |
|-----------|------------------------------------------------------|
| `db`      | `psycopg`, `pgvector` (hybrid index)                 |
| `ml`      | `sentence-transformers`, `numpy` (embeddings/rerank) |
| `ingest`  | `pypdf`, `beautifulsoup4`, `lxml` (parsers)          |
| `all`     | `db` + `ml` + `ingest`                               |

## Quickstart

```bash
# 1. Start Postgres + pgvector
docker compose up -d
citemd db init                      # create schema and extensions

# 2. Ingest a corpus
citemd ingest pubmed --query "metformin cardiovascular outcomes" --max 200
citemd ingest files ./data/guidelines/*.pdf

# 3. Retrieve
citemd query "What is the first-line treatment for type 2 diabetes?" --k 5

# 4. Load the MIRAGE evaluation sets
citemd eval fetch
citemd eval retrieval --dataset pubmedqa --k 10
```

## Configuration

All settings are environment variables prefixed `CITEMD_` (or a `.env` file). The most
relevant:

| Variable                  | Default                                      | Meaning                              |
|---------------------------|----------------------------------------------|--------------------------------------|
| `CITEMD_DATABASE_URL`     | `postgresql://citemd:citemd@localhost:5432/citemd` | Postgres connection            |
| `CITEMD_EMBEDDING_MODEL`  | `BAAI/bge-small-en-v1.5`                      | Sentence-Transformers embedder (CPU) |
| `CITEMD_RERANK_MODEL`     | `BAAI/bge-reranker-base`                      | Cross-encoder reranker (CPU)         |
| `CITEMD_LLM_BASE_URL`     | KVGate endpoint                              | OpenAI-compatible generation backend |
| `CITEMD_LLM_MODEL`        | `claude-sonnet-4-6`                          | Default generation model             |

The LLM client is OpenAI-compatible and provider-agnostic. By default it routes through
KVGate to `claude-sonnet-4-6` (best groundedness for the headline eval);
`claude-haiku-4-5` is the cheap option for large MIRAGE sweeps.

## Development

```bash
pip install -e ".[dev]"
ruff check .
pytest
```

The pure-logic core (chunking, Reciprocal Rank Fusion, retrieval metrics, MIRAGE parsing)
is covered by unit tests that need neither a GPU nor a database, so CI stays fast.

## Roadmap

1. **Ingestion + index + eval set (this milestone):** parse, chunk, embed a PubMed corpus
   into pgvector and BM25; a working hybrid retriever; the MIRAGE eval sets loaded with
   retrieval-quality metrics.
2. **Pipeline + headline eval:** retrieve, rerank, cite, abstain; the faithfulness,
   hallucination, and abstention harness; the ablation table and charts.
3. **Polish + ship:** multimodal table/figure retrieval; Next.js cited-answer UI;
   Prometheus and Grafana; Docker and CI; a demo deployment.

## License

[Apache-2.0](LICENSE).
