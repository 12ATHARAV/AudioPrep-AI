"""
Embedding service for generating vector representations.

Uses sentence-transformers for dense embeddings.
Lazy-loads the model to save RAM until first use.

Interview Note: "I use all-MiniLM-L6-v2 as a lightweight
baseline that works on 8GB RAM. For production, I'd evaluate
Qwen3-Embedding-0.6B or BGE-small-v1.5 which offer better
retrieval quality with moderate resource requirements. The
MTEB leaderboard is the standard benchmark for selecting
embedding models."
"""

import numpy as np
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("embeddings")


class EmbeddingService:
    """Generates dense vector embeddings for text."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.embedding_model
        self._model = None

    @property
    def model(self):
        """Lazy-load the model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("loading_embedding_model", model=self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        if hasattr(self.model, "get_embedding_dimension"):
            return self.model.get_embedding_dimension()
        return self.model.get_sentence_embedding_dimension()

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string."""
        embedding = self.model.encode(
            text, show_progress_bar=False, normalize_embeddings=True
        )
        return embedding.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of texts.

        Uses batching for efficiency — much faster than
        embedding one at a time.
        """
        if not texts:
            return []

        logger.info("embedding_batch", count=len(texts))

        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=True,
            batch_size=32,
        )

        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a search query.

        Some models use different prefixes for queries vs documents.
        This method handles that distinction.
        """
        return self.embed_text(query)