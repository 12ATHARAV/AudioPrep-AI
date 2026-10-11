"""
Hybrid Retrieval: Dense + BM25 with Reciprocal Rank Fusion.

Architecture:
  Query
   │
   ├──► Dense Vector Search (semantic)
   │
   └──► BM25 Search (lexical)
           │
           ▼
     Reciprocal Rank Fusion
           │
           ▼
     Merged + Deduplicated results

Interview Note: "Reciprocal Rank Fusion (RRF) combines ranked
lists without needing to normalize scores across different
retrieval methods. Each result gets a score of 1/(k + rank),
where k is typically 60. This naturally handles the fact that
BM25 scores and vector similarity scores are on completely
different scales."

Why RRF instead of score normalization?
"Score normalization requires knowing the score distribution,
which can vary dramatically between queries. RRF only uses
rank positions, making it robust and query-independent."
"""

from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

from app.models.chunk import Chunk
from app.services.retrieval.embeddings import EmbeddingService
from app.services.retrieval.bm25_index import BM25Index
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("hybrid_retriever")


class HybridRetriever:
    """
    Combines dense vector search with BM25 sparse retrieval
    using Reciprocal Rank Fusion.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        qdrant_client: QdrantClient | None = None,
        collection_name: str | None = None,
        rrf_k: int = 60,
    ):
        self.embedding_service = embedding_service or EmbeddingService()
        self.collection_name = collection_name or settings.qdrant_collection
        self.bm25 = BM25Index()
        self.rrf_k = rrf_k
        self._chunks: dict[str, Chunk] = {}

        # --- QDRANT CLIENT INITIALIZATION ---
        if qdrant_client is not None:
            self.qdrant = qdrant_client
        elif getattr(settings, "qdrant_storage_path", None) == ":memory:":
            self.qdrant = QdrantClient(":memory:")
        elif getattr(settings, "qdrant_storage_path", None):
            # 🚀 LOCAL DISK PERSISTENCE (NO DOCKER!)
            storage_path = Path(settings.qdrant_storage_path)
            storage_path.mkdir(parents=True, exist_ok=True)
            self.qdrant = QdrantClient(path=str(storage_path))
        else:
            # Fallback to server/Docker if path is not specified
            self.qdrant = QdrantClient(
                host=settings.qdrant_host, 
                port=settings.qdrant_port
            )

    def create_collection(self) -> None:
        """Create (or recreate) the Qdrant collection."""
        collections = self.qdrant.get_collections().collections
        existing = [c.name for c in collections]

        if self.collection_name in existing:
            logger.info("collection_exists", name=self.collection_name)
            return

        self.qdrant.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.embedding_service.dimension,
                distance=Distance.COSINE,
            ),
        )

        logger.info("collection_created", name=self.collection_name)

    def index_chunks(self, chunks: list[Chunk]) -> None:
        """
        Index chunks into both dense (Qdrant) and sparse (BM25) indices.

        Steps:
        1. Generate embeddings for all chunks
        2. Upsert into Qdrant with metadata
        3. Build BM25 index
        """
        if not chunks:
            logger.warning("no_chunks_to_index")
            return

        # Filter out parent chunks (only index children/leaf chunks)
        indexable = [c for c in chunks if not c.is_parent]

        # Store all chunks (including parents) for parent lookup
        for chunk in chunks:
            self._chunks[chunk.chunk_id] = chunk

        # Step 1: Generate embeddings
        texts = [c.embedding_text for c in indexable]
        embeddings = self.embedding_service.embed_batch(texts)

        # Step 2: Upsert into Qdrant using chunk_id (UUID) to prevent collisions
        points = []
        for chunk, embedding in zip(indexable, embeddings):
            points.append(
                PointStruct(
                    id=chunk.chunk_id,  # Uses the chunk's unique UUID
                    vector=embedding,
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "doc_id": chunk.doc_id,
                        "content": chunk.content,
                        "strategy": chunk.strategy,
                        "heading_hierarchy": chunk.heading_hierarchy,
                        "page_number": chunk.page_number,
                        "parent_id": chunk.parent_id,
                        "is_parent": chunk.is_parent,
                        "contextualized": chunk.contextualized_content or "",
                    },
                )
            )

        # Batch upsert
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.qdrant.upsert(
                collection_name=self.collection_name, points=batch
            )

        # Step 3: Build BM25 index
        self.bm25.index(indexable)

        logger.info(
            "indexing_complete",
            total_chunks=len(chunks),
            indexed=len(indexable),
            parents_stored=len(chunks) - len(indexable),
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
        dense_weight: float = 0.6,
        bm25_weight: float = 0.4,
        expand_parents: bool = True,
    ) -> list[dict]:
        """
        Hybrid search combining dense and BM25 retrieval.

        Args:
            query: Search query
            top_k: Number of results to return
            dense_weight: Weight for dense results in RRF
            bm25_weight: Weight for BM25 results in RRF
            expand_parents: If True, expand child chunks to parents

        Returns:
            List of result dicts with chunk, score, and source info
        """
        # Step 1: Dense search (using query_points in qdrant-client >= 1.11)
        query_embedding = self.embedding_service.embed_query(query)

        if hasattr(self.qdrant, "query_points"):
            dense_response = self.qdrant.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=top_k * 2,  # Get more candidates for fusion
            )
            dense_results = dense_response.points
        else:
            dense_results = self.qdrant.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k * 2,
            )

        # Step 2: BM25 search
        bm25_results = self.bm25.search(query, top_k=top_k * 2)

        # Step 3: Reciprocal Rank Fusion
        fused = self._reciprocal_rank_fusion(
            dense_results=[(r.payload["chunk_id"], r.score) for r in dense_results],
            bm25_results=[(c.chunk_id, s) for c, s in bm25_results],
            dense_weight=dense_weight,
            bm25_weight=bm25_weight,
        )

        # Step 4: Build result objects
        results = []
        for chunk_id, rrf_score in fused[:top_k]:
            chunk = self._chunks.get(chunk_id)
            if not chunk:
                continue

            result = {
                "chunk": chunk,
                "rrf_score": rrf_score,
                "content": chunk.content,
            }

            # Expand to parent if available
            if expand_parents and chunk.parent_id:
                parent = self._chunks.get(chunk.parent_id)
                if parent:
                    result["parent_content"] = parent.content
                    result["content"] = parent.content  # Use parent for LLM

            results.append(result)

        logger.info(
            "hybrid_search",
            query=query[:50],
            dense_hits=len(dense_results),
            bm25_hits=len(bm25_results),
            fused_results=len(results),
        )

        return results

    def _reciprocal_rank_fusion(
        self,
        dense_results: list[tuple[str, float]],
        bm25_results: list[tuple[str, float]],
        dense_weight: float = 0.6,
        bm25_weight: float = 0.4,
    ) -> list[tuple[str, float]]:
        """
        Combine results using Reciprocal Rank Fusion.

        RRF score = Σ weight / (k + rank)
        """
        scores: dict[str, float] = {}

        # Score from dense results
        for rank, (chunk_id, _) in enumerate(dense_results):
            rrf = dense_weight / (self.rrf_k + rank + 1)
            scores[chunk_id] = scores.get(chunk_id, 0) + rrf

        # Score from BM25 results
        for rank, (chunk_id, _) in enumerate(bm25_results):
            rrf = bm25_weight / (self.rrf_k + rank + 1)
            scores[chunk_id] = scores.get(chunk_id, 0) + rrf

        # Sort by combined score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return ranked