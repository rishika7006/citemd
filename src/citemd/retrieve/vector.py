"""Dense retrieval over pgvector using cosine distance."""

from __future__ import annotations

from citemd.config import get_settings
from citemd.db.connection import connect
from citemd.ingest.embed import Embedder
from citemd.models import RetrievedChunk


def vector_search(
    query: str,
    *,
    k: int | None = None,
    embedder: Embedder | None = None,
    access_tags: list[str] | None = None,
    database_url: str | None = None,
) -> list[RetrievedChunk]:
    """Return the top-k chunks by cosine similarity to the query embedding.

    The reported score is cosine similarity in ``[-1, 1]`` (``1 - cosine_distance``).
    Results are filtered to chunks whose access_tags overlap ``access_tags`` (default
    ``['public']``).
    """
    settings = get_settings()
    k = k or settings.candidate_k
    tags = access_tags or ["public"]
    embedder = embedder or Embedder()
    # Wrap in pgvector's Vector so psycopg binds the parameter as a `vector`, not as a
    # double precision[]; the latter has no match for the `<=>` distance operator.
    from pgvector import Vector

    vector = Vector(embedder.embed_query(query))

    rows = []
    with connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT chunk_id, source_id, content, title, section, page, uri,
                       1 - (embedding <=> %s) AS score
                FROM chunks
                WHERE embedding IS NOT NULL AND access_tags && %s
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (vector, tags, vector, k),
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
            retriever="vector",
        )
        for r in rows
    ]
