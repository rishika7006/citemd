"""CPU embeddings via Sentence-Transformers.

BGE models prepend a short instruction to queries (but not to passages) for best retrieval
quality. This module handles that asymmetry. ``sentence-transformers`` is an optional
dependency (the ``ml`` extra); it is imported lazily so the core stays light.
"""

from __future__ import annotations

from functools import lru_cache

from citemd.config import get_settings

# Recommended retrieval instruction for BGE query encoding.
_BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def _require_st():
    try:
        from sentence_transformers import SentenceTransformer  # noqa: F401
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise ModuleNotFoundError(
            "Embeddings need the 'ml' extra. Install with: pip install 'citemd[ml]'"
        ) from exc
    return SentenceTransformer


@lru_cache(maxsize=2)
def _load_model(model_name: str):
    SentenceTransformer = _require_st()
    return SentenceTransformer(model_name, device="cpu")


class Embedder:
    """Wraps a Sentence-Transformers model with BGE-aware query encoding."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or get_settings().embedding_model
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = _load_model(self.model_name)
        return self._model

    def _is_bge(self) -> bool:
        return "bge" in self.model_name.lower()

    def embed_passages(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Embed document passages (no query instruction)."""
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [list(map(float, v)) for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query, applying the BGE instruction prefix when relevant."""
        payload = (_BGE_QUERY_INSTRUCTION + text) if self._is_bge() else text
        vector = self.model.encode(
            [payload],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        return list(map(float, vector))
