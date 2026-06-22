"""Hybrid retrieval: run dense and sparse retrievers, then fuse with RRF.

This is the milestone-1 retriever. Cross-encoder reranking is layered on top in a later
milestone; the fused output here is already a usable hybrid ranking and the baseline the
ablation table is built against (vector-only -> hybrid -> +rerank -> +abstention).
"""

from __future__ import annotations

from citemd.config import get_settings
from citemd.ingest.embed import Embedder
from citemd.models import RetrievedChunk
from citemd.retrieve.bm25 import bm25_search
from citemd.retrieve.fusion import rrf_fuse
from citemd.retrieve.vector import vector_search


def hybrid_search(
    query: str,
    *,
    k: int | None = None,
    candidate_k: int | None = None,
    rrf_k: int | None = None,
    weights: tuple[float, float] | None = None,
    embedder: Embedder | None = None,
    access_tags: list[str] | None = None,
    database_url: str | None = None,
) -> list[RetrievedChunk]:
    """Retrieve with vector + BM25 and fuse with Reciprocal Rank Fusion.

    Args:
        k: Number of fused results to return (defaults to settings.retrieve_k).
        candidate_k: Candidates pulled from each retriever before fusion.
        weights: ``(vector_weight, bm25_weight)`` for RRF; defaults to equal weight.
    """
    settings = get_settings()
    k = k or settings.retrieve_k
    candidate_k = candidate_k or settings.candidate_k
    rrf_k = rrf_k or settings.rrf_k
    embedder = embedder or Embedder()

    dense = vector_search(
        query,
        k=candidate_k,
        embedder=embedder,
        access_tags=access_tags,
        database_url=database_url,
    )
    sparse = bm25_search(
        query,
        k=candidate_k,
        access_tags=access_tags,
        database_url=database_url,
    )

    by_id: dict[str, RetrievedChunk] = {}
    for chunk in (*dense, *sparse):
        by_id.setdefault(chunk.chunk_id, chunk)

    fused = rrf_fuse(
        [[c.chunk_id for c in dense], [c.chunk_id for c in sparse]],
        k=rrf_k,
        weights=list(weights) if weights else None,
    )

    results: list[RetrievedChunk] = []
    for chunk_id, score in fused[:k]:
        base = by_id[chunk_id]
        results.append(base.model_copy(update={"score": score, "retriever": "hybrid"}))
    return results
