"""Cross-encoder reranking.

A bi-encoder (the BGE embedder) retrieves cheaply but scores query and passage
independently. A cross-encoder reads the query and passage together and is markedly more
accurate at ordering a small candidate set, at the cost of one forward pass per candidate.
The standard recipe, used here, is: retrieve a larger candidate pool with the hybrid
retriever, then rerank with ``BAAI/bge-reranker-base`` and keep the top k.

``sentence-transformers`` is the optional ``ml`` extra and is imported lazily. The scorer is
injectable so the pipeline and tests can substitute a deterministic fake without torch.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import lru_cache

from citemd.config import get_settings
from citemd.models import RetrievedChunk

# A scorer maps (query, [passages]) -> one relevance score per passage.
ScoreFn = Callable[[str, Sequence[str]], Sequence[float]]


def _require_cross_encoder():
    try:
        from sentence_transformers import CrossEncoder  # noqa: F401
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise ModuleNotFoundError(
            "Reranking needs the 'ml' extra. Install with: pip install 'citemd[ml]'"
        ) from exc
    return CrossEncoder


@lru_cache(maxsize=2)
def _load_cross_encoder(model_name: str):
    CrossEncoder = _require_cross_encoder()
    return CrossEncoder(model_name, device="cpu")


class CrossEncoderReranker:
    """Default reranker backed by a Sentence-Transformers CrossEncoder on CPU."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or get_settings().rerank_model
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = _load_cross_encoder(self.model_name)
        return self._model

    def __call__(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        scores = self.model.predict(
            [(query, p) for p in passages],
            show_progress_bar=False,
        )
        return [float(s) for s in scores]


def rerank(
    query: str,
    chunks: Sequence[RetrievedChunk],
    *,
    k: int | None = None,
    scorer: ScoreFn | None = None,
) -> list[RetrievedChunk]:
    """Rescore candidate chunks with a cross-encoder and return the top k.

    The returned chunks carry the cross-encoder score in ``score`` and ``retriever="rerank"``.
    Relative order of the input is otherwise irrelevant; only the rerank score decides the
    output order, which is what makes this a genuine reranking step rather than a re-weighting.
    """
    if not chunks:
        return []
    scorer = scorer or CrossEncoderReranker()
    scores = scorer(query, [c.text for c in chunks])
    ranked = sorted(
        zip(chunks, scores),
        key=lambda cs: cs[1],
        reverse=True,
    )
    if k is not None:
        ranked = ranked[:k]
    return [
        chunk.model_copy(update={"score": float(score), "retriever": "rerank"})
        for chunk, score in ranked
    ]
