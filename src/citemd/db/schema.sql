-- CiteMD hybrid index schema.
-- Postgres holds both the dense vectors (pgvector) and the sparse BM25 index
-- (full-text search tsvector), so a single store backs hybrid retrieval.
--
-- The embedding dimension below MUST match CITEMD_EMBEDDING_DIM (default 384 for
-- BAAI/bge-small-en-v1.5). If you change the embedding model, change the vector size
-- here and re-ingest.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    source_id    TEXT PRIMARY KEY,
    title        TEXT NOT NULL DEFAULT '',
    source_type  TEXT NOT NULL DEFAULT 'text',
    uri          TEXT NOT NULL DEFAULT '',
    access_tags  TEXT[] NOT NULL DEFAULT ARRAY['public'],
    metadata     JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id     TEXT PRIMARY KEY,
    source_id    TEXT NOT NULL REFERENCES documents(source_id) ON DELETE CASCADE,
    chunk_index  INTEGER NOT NULL,
    title        TEXT NOT NULL DEFAULT '',
    section      TEXT,
    page         INTEGER,
    source_type  TEXT NOT NULL DEFAULT 'text',
    uri          TEXT NOT NULL DEFAULT '',
    access_tags  TEXT[] NOT NULL DEFAULT ARRAY['public'],
    content      TEXT NOT NULL,
    embedding    vector(384),
    -- Generated full-text vector for BM25-style ranking via ts_rank_cd.
    fts          tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Sparse retrieval: GIN index over the generated tsvector.
CREATE INDEX IF NOT EXISTS chunks_fts_idx ON chunks USING GIN (fts);

-- Dense retrieval: HNSW over cosine distance. HNSW needs no training and, unlike IVFFlat,
-- has no empty-list failure mode on small corpora (IVFFlat with too many lists and a low
-- probe count can return zero rows). Good recall for the few-thousand-doc target here.
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- Access-control filtering.
CREATE INDEX IF NOT EXISTS chunks_access_tags_idx ON chunks USING GIN (access_tags);
