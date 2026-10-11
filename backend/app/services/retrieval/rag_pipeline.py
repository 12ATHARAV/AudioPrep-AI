"""
Complete RAG pipeline: Query → Retrieve → Rerank → Context.

This is the full retrieval pipeline that the dialogue
generator will use.

Pipeline:
  User Query
      │
      ▼
  Query Expansion (optional: generate sub-queries)
      │
      ├──► Dense Search (Qdrant)
      │
      └──► BM25 Search (sparse)
              │
              ▼
        Reciprocal Rank Fusion
              │
              ▼
         Cross-Encoder Reranking
              │
              ▼
         Top-K Context Chunks

Interview Note: "Each stage adds value:
  - Dense search provides semantic recall
  - BM25 provides lexical precision
  - RRF combines without score normalization
  - Reranking provides deep relevance assessment
  The pipeline progressively filters from high recall
  to high precision."
"""

from app.services.retrieval.hybrid_retriever import HybridRetriever
from app.services.retrieval.reranker import RerankerService
from app.core.logging import get_logger

logger = get_logger("rag_pipeline")


class RAGPipeline:
    """
    End-to-end RAG pipeline with hybrid retrieval and reranking.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: RerankerService | None = None,
    ):
        self.retriever = retriever
        self.reranker = reranker or RerankerService()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        retrieval_top_k: int = 20,
        use_reranker: bool = True,
    ) -> list[dict]:
        """
        Full RAG retrieval pipeline.

        1. Hybrid retrieval (dense + BM25 + RRF) → 20 candidates
        2. Cross-encoder reranking → top 5
        """
        # Step 1: Broad retrieval
        candidates = self.retriever.search(
            query=query,
            top_k=retrieval_top_k,
        )

        if not candidates:
            logger.warning("no_retrieval_results", query=query[:50])
            return []

        # Step 2: Reranking (optional)
        if use_reranker and len(candidates) > top_k:
            results = self.reranker.rerank(
                query=query,
                results=candidates,
                top_k=top_k,
            )
        else:
            results = candidates[:top_k]

        logger.info(
            "rag_pipeline_complete",
            query=query[:50],
            candidates=len(candidates),
            returned=len(results),
        )

        return results

    def retrieve_for_concepts(
        self,
        concepts: list[str],
        queries_per_concept: int = 2,
        top_k_per_concept: int = 3,
    ) -> dict[str, list[dict]]:
        """
        Retrieve evidence for multiple concepts.

        Used by the dialogue planner to gather context
        for each topic in the podcast.

        Returns:
            Dict mapping concept → list of relevant contexts
        """
        concept_contexts = {}

        for concept in concepts:
            # Generate multiple queries per concept
            queries = self._expand_concept_queries(
                concept, queries_per_concept
            )

            all_results = []
            seen_ids = set()

            for query in queries:
                results = self.retrieve(
                    query=query,
                    top_k=top_k_per_concept,
                )
                for r in results:
                    chunk_id = r["chunk"].chunk_id
                    if chunk_id not in seen_ids:
                        seen_ids.add(chunk_id)
                        all_results.append(r)

            concept_contexts[concept] = all_results

        return concept_contexts

    def _expand_concept_queries(
        self, concept: str, n_queries: int
    ) -> list[str]:
        """
        Generate multiple search queries for a concept.

        Different phrasings improve retrieval recall.
        """
        queries = [concept]

        # Add common query patterns
        patterns = [
            f"What is {concept}?",
            f"Explain {concept}",
            f"How does {concept} work?",
            f"{concept} definition and examples",
        ]

        queries.extend(patterns[: n_queries - 1])
        return queries[:n_queries]