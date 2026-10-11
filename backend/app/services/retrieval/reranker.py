"""
Cross-encoder reranking for precision refinement.

Why reranking after retrieval?

Retriever (bi-encoder): encodes query and document SEPARATELY
  → fast but approximate similarity

Reranker (cross-encoder): encodes query AND document TOGETHER
  → slow but much more accurate relevance scoring

Pipeline:
  Query → Retriever (fast, high recall) → 20 candidates
       → Reranker (slow, high precision) → Top 5

Analogy:
  Retriever = casting a wide net (catch many fish)
  Reranker = inspecting each fish (keep only the best)

Interview Note: "Bi-encoders embed query and documents
independently, so they can only compute approximate similarity.
Cross-encoders process the query-document pair together through
all transformer attention layers, enabling deep semantic
comparison. This is 100x slower but significantly more accurate
for relevance scoring."

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - 22M parameters
  - Trained on MS MARCO passage ranking
  - Fits easily in 8GB RAM
  - ~100ms per query-document pair
"""

# from sentence_transformers import CrossEncoder
from typing import Optional, Any
from app.core.logging import get_logger

logger = get_logger("reranker")


class RerankerService:
    """
    Cross-encoder reranker for refining retrieval results.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k: int = 5,
    ):
        self.model_name = model_name
        self.top_k = top_k
        self._model = None

    @property
    def model(self) -> Any:
        """
        100% True Lazy Loading:
        Neither PyTorch nor CrossEncoder is imported until this property is accessed!
        """
        if self._model is None:
            logger.info(
                "loading_reranker", model=self.model_name
            )
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        results: list[dict],
        top_k: int | None = None,
    ) -> list[dict]:
        """
        Rerank retrieval results using the cross-encoder.

        Args:
            query: The search query
            results: List of result dicts from hybrid retriever
                     (must have 'content' key)
            top_k: Number of top results to return

        Returns:
            Reranked and filtered results with added
            'rerank_score' field
        """
        if not results:
            return []

        top_k = top_k or self.top_k

        # Prepare query-document pairs
        pairs = [
            [query, result["content"]] for result in results
        ]

        # Score all pairs
        logger.info(
            "reranking",
            query=query[:50],
            candidates=len(pairs),
        )

        scores = self.model.predict(pairs)

        # Attach scores
        for result, score in zip(results, scores):
            result["rerank_score"] = float(score)

        # Sort by rerank score
        reranked = sorted(
            results, key=lambda x: x["rerank_score"], reverse=True
        )

        logger.info(
            "reranking_complete",
            top_score=round(reranked[0]["rerank_score"], 4) if reranked else 0,
            returned=min(top_k, len(reranked)),
        )

        return reranked[:top_k]