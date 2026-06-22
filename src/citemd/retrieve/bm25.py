"""Sparse retrieval using Postgres full-text search as a BM25-style ranker.

Postgres ``ts_rank_cd`` over a GIN-indexed ``tsvector`` gives lexical ranking that is a
strong sparse complement to dense retrieval. The interface is deliberately the same shape
as :mod:`citemd.retrieve.vector` so the BM25 layer is swappable for a dedicated engine
(e.g. Elasticsearch) later without touching the fusion code.
"""

from __future__ import annotations

from citemd.config import get_settings
from citemd.db.connection import connect
from citemd.models import RetrievedChunk


def bm25_search(
    query: str,
    *,
    k: int | None = None,
    access_tags: list[str] | None = None,
    database_url: str | None = None,
) -> list[RetrievedChunk]:
    """Return the top-k chunks by lexical relevance to the query.

    Uses ``websearch_to_tsquery`` so natural queries (quotes, OR, minus) behave sensibly.
    """
    settings = get_settings()
    k = k or settings.candidate_k
    tags = access_tags or ["public"]

    rows = []
    with connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT chunk_id, source_id, content, title, section, page, uri,
                       ts_rank_cd(fts, websearch_to_tsquery('english', %s)) AS score
                FROM chunks
                WHERE fts @@ websearch_to_tsquery('english', %s)
                  AND access_tags && %s
                ORDER BY score DESC
                LIMIT %s
                """,
                (query, query, tags, k),
            )
            rows = cur.fetchall()

    return [
        RetrievedChunk(
            chunk_id=r[0],
            source_id=r[1],
            text=r[2],
            title=r[3] or "",
            section=r[4],
            page=r[5],
            uri=r[6] or "",
            score=float(r[7]),
            retriever="bm25",
        )
        for r in rows
    ]
