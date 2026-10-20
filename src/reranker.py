"""Cross-encoder reranker for second-stage retrieval refinement.

Takes the top-N results from first-stage hybrid retrieval and rescores
each (query, chunk) pair using a cross-encoder that attends across both.
Returns the top-K most relevant chunks after reranking.
"""
from pathlib import Path
from typing import TypedDict
from sentence_transformers import CrossEncoder


class RetrievedChunk(TypedDict):
    """Shape of a chunk coming out of first-stage retrieval."""
    text: str
    metadata: dict
    first_stage_score: float


class Reranker:
    """Cross-encoder reranker. Loads the model lazily on first use."""

    # ms-marco-MiniLM-L-6-v2 is the standard baseline: ~22M params, CPU-fast,
    # trained on MS MARCO query-passage relevance. Good enough that most RAG
    # systems don't need to look further.
    _MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, model_name: str | None = None):
        self._model_name = model_name or self._MODEL_NAME
        self._model: CrossEncoder | None = None

    def _load(self) -> CrossEncoder:
        if self._model is None:
            # First call downloads the model (~90 MB), cached locally afterward
            self._model = CrossEncoder(self._model_name)
        return self._model

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """Rescore chunks by query-chunk relevance; return top_k.

        Args:
            query: The user's question.
            chunks: First-stage retrieval results (typically 20 chunks).
            top_k: How many chunks to keep after reranking.

        Returns:
            Up to top_k chunks, sorted by rerank score (highest first).
            Each chunk gains a 'rerank_score' field alongside its original fields.
        """
        if not chunks:
            return []

        model = self._load()

        # Access .text attribute (not dict key)
        pairs = [[query, chunk.text] for chunk in chunks]
        scores = model.predict(pairs)

        for chunk, score in zip(chunks, scores):
            chunk.rerank_score = float(score)   # attribute assignment
            chunk.score = float(score)          # overwrite .score so downstream sees rerank score

        reranked = sorted(chunks, key=lambda c: c.rerank_score, reverse=True)
        return reranked[:top_k]

# Module-level singleton — follows the get_kb() lazy-init pattern
_reranker: Reranker | None = None


def get_reranker() -> Reranker:
    """Lazy-init the reranker (same pattern as get_kb() in tools.py)."""
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker