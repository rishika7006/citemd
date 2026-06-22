"""Central configuration for CiteMD.

All values are overridable via ``CITEMD_*`` environment variables or a ``.env`` file.
The defaults are chosen so the project is laptop-friendly: CPU embeddings and reranking,
a single Postgres instance, and generation routed through KVGate.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings, read once and cached."""

    model_config = SettingsConfigDict(
        env_prefix="CITEMD_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Storage -----------------------------------------------------------------
    database_url: str = Field(
        default="postgresql://citemd:citemd@localhost:5432/citemd",
        description="Postgres connection string (pgvector extension required).",
    )

    # --- Embeddings and reranking (CPU) ------------------------------------------
    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="Sentence-Transformers model used to embed chunks and queries.",
    )
    embedding_dim: int = Field(
        default=384,
        description="Dimensionality of embedding_model; must match the pgvector column.",
    )
    rerank_model: str = Field(
        default="BAAI/bge-reranker-base",
        description="Cross-encoder used to rerank fused candidates (later milestone).",
    )

    # --- Chunking ----------------------------------------------------------------
    chunk_max_chars: int = Field(default=1200, description="Target chunk size in characters.")
    chunk_overlap_chars: int = Field(default=150, description="Overlap between adjacent chunks.")

    # --- Retrieval ---------------------------------------------------------------
    retrieve_k: int = Field(default=10, description="Final number of chunks returned.")
    candidate_k: int = Field(
        default=30,
        description="Candidates pulled from each retriever before fusion.",
    )
    rrf_k: int = Field(default=60, description="Reciprocal Rank Fusion damping constant.")

    # --- Generation backend ------------------------------------------------------
    # OpenAI-compatible client. Points at KVGate by default; can target a hosted API
    # directly as a fallback by overriding llm_base_url and llm_api_key.
    llm_base_url: str = Field(
        default="https://kvgate.vercel.app/v1",
        description="OpenAI-compatible base URL for generation (KVGate by default).",
    )
    llm_api_key: str = Field(default="", description="API key for the generation backend.")
    llm_model: str = Field(
        default="claude-sonnet-4-6",
        description="Default generation model. Use claude-haiku-4-5 for large eval sweeps.",
    )

    # --- Paths -------------------------------------------------------------------
    data_dir: str = Field(default="data", description="Root for downloaded corpora and eval sets.")
    artifacts_dir: str = Field(default="artifacts", description="Root for eval outputs and charts.")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached settings instance."""
    return Settings()
