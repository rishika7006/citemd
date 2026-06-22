"""Load parsed-and-chunked documents into the Postgres hybrid index.

This is the write side of ingestion: documents and their chunks are upserted, chunk text
is embedded in batches on CPU, and both the dense vector and the generated full-text vector
land in the ``chunks`` table. Requires the ``db`` and ``ml`` extras.
"""

from __future__ import annotations

from collections.abc import Iterable

from citemd.config import get_settings
from citemd.db.connection import connect
from citemd.ingest.chunk import chunk_document
from citemd.ingest.embed import Embedder
from citemd.ingest.parse import page_for_offset
from citemd.models import Document


def _attach_pdf_pages(document: Document, chunks) -> None:
    """For PDF documents, infer each chunk's page from the page-marked source text."""
    if document.source_type != "pdf":
        return
    cursor = 0
    for chunk in chunks:
        idx = document.text.find(chunk.text[:80], cursor) if chunk.text else -1
        offset = idx if idx >= 0 else cursor
        chunk.page = page_for_offset(document.text, offset)
        if idx >= 0:
            cursor = idx


def load_documents(
    documents: Iterable[Document],
    *,
    database_url: str | None = None,
    embedder: Embedder | None = None,
    batch_size: int = 64,
) -> dict[str, int]:
    """Chunk, embed, and upsert documents. Returns counts of documents and chunks written."""
    settings = get_settings()
    embedder = embedder or Embedder()
    documents = list(documents)

    n_docs = 0
    n_chunks = 0
    with connect(database_url) as conn:
        for document in documents:
            chunks = chunk_document(
                document,
                max_chars=settings.chunk_max_chars,
                overlap=settings.chunk_overlap_chars,
            )
            _attach_pdf_pages(document, chunks)
            if not chunks:
                continue

            vectors: list[list[float]] = []
            for start in range(0, len(chunks), batch_size):
                batch = chunks[start : start + batch_size]
                vectors.extend(embedder.embed_passages([c.text for c in batch]))

            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO documents
                        (source_id, title, source_type, uri, access_tags, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                    ON CONFLICT (source_id) DO UPDATE SET
                        title = EXCLUDED.title,
                        source_type = EXCLUDED.source_type,
                        uri = EXCLUDED.uri,
                        access_tags = EXCLUDED.access_tags,
                        metadata = EXCLUDED.metadata
                    """,
                    (
                        document.source_id,
                        document.title,
                        document.source_type,
                        document.uri,
                        document.access_tags,
                        _json(document.metadata),
                    ),
                )
                # Replace any prior chunks for this document so re-ingest is idempotent.
                cur.execute("DELETE FROM chunks WHERE source_id = %s", (document.source_id,))
                from pgvector import Vector

                for chunk, vector in zip(chunks, vectors):
                    vector = Vector(vector)
                    cur.execute(
                        """
                        INSERT INTO chunks
                            (chunk_id, source_id, chunk_index, title, section, page,
                             source_type, uri, access_tags, content, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            chunk.chunk_id,
                            chunk.source_id,
                            chunk.chunk_index,
                            chunk.title,
                            chunk.section,
                            chunk.page,
                            chunk.source_type,
                            chunk.uri,
                            chunk.access_tags,
                            chunk.text,
                            vector,
                        ),
                    )
            conn.commit()
            n_docs += 1
            n_chunks += len(chunks)
    return {"documents": n_docs, "chunks": n_chunks}


def _json(value: dict) -> str:
    import json

    return json.dumps(value)
