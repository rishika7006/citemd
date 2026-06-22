"""Core data structures shared across ingestion, retrieval, and evaluation."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class Document(BaseModel):
    """A source document before chunking."""

    source_id: str = Field(description="Stable id, e.g. a PubMed PMID or a file path hash.")
    title: str = ""
    text: str = ""
    source_type: str = Field(default="text", description="pubmed | pdf | html | markdown")
    uri: str = Field(default="", description="Origin URI or file path.")
    access_tags: list[str] = Field(
        default_factory=lambda: ["public"],
        description="Access-control tags; only 'public' data is permitted in CiteMD.",
    )
    metadata: dict = Field(default_factory=dict)


class Chunk(BaseModel):
    """A retrievable unit derived from a Document."""

    source_id: str
    chunk_index: int = Field(description="0-based position of the chunk within its document.")
    text: str
    title: str = ""
    section: Optional[str] = None
    page: Optional[int] = None
    source_type: str = "text"
    uri: str = ""
    access_tags: list[str] = Field(default_factory=lambda: ["public"])

    @property
    def chunk_id(self) -> str:
        """Deterministic id combining the document id and chunk position."""
        return f"{self.source_id}#{self.chunk_index}"


class RetrievedChunk(BaseModel):
    """A chunk returned by a retriever, with its score and provenance."""

    chunk_id: str
    source_id: str
    text: str
    title: str = ""
    section: Optional[str] = None
    page: Optional[int] = None
    uri: str = ""
    score: float = 0.0
    retriever: str = Field(default="hybrid", description="vector | bm25 | hybrid | rerank")
