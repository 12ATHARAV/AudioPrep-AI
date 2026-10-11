"""
Tests for hybrid retrieval system.

Run with: python -m pytest tests/test_retrieval.py -v

Note: These tests use Qdrant in-memory mode (no Docker required).
"""

import pytest
from qdrant_client import QdrantClient

from app.models.chunk import Chunk
from app.services.retrieval.embeddings import EmbeddingService
from app.services.retrieval.bm25_index import BM25Index
from app.services.retrieval.hybrid_retriever import HybridRetriever


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    """Create sample chunks for testing retrieval."""
    return [
        Chunk(
            doc_id="doc1",
            content="Docker containers share the host OS kernel, making them lightweight.",
            strategy="test",
            heading_hierarchy=["Docker", "Containers"],
        ),
        Chunk(
            doc_id="doc1",
            content="Virtual machines include a full guest operating system and hypervisor.",
            strategy="test",
            heading_hierarchy=["Docker", "VMs"],
        ),
        Chunk(
            doc_id="doc1",
            content="The JVM garbage collector automatically reclaims unused heap memory.",
            strategy="test",
            heading_hierarchy=["Java", "GC"],
        ),
        Chunk(
            doc_id="doc1",
            content="Bridge network is Docker's default network driver for container communication.",
            strategy="test",
            heading_hierarchy=["Docker", "Networking"],
        ),
        Chunk(
            doc_id="doc1",
            content="TCP three-way handshake uses SYN, SYN-ACK, and ACK packets.",
            strategy="test",
            heading_hierarchy=["Networking", "TCP"],
        ),
    ]


class TestBM25Index:
    def test_index_and_search(self, sample_chunks):
        bm25 = BM25Index()
        bm25.index(sample_chunks)

        results = bm25.search("Docker containers", top_k=3)
        assert len(results) > 0
        assert "docker" in results[0][0].content.lower() or \
               "container" in results[0][0].content.lower()

    def test_exact_term_matching(self, sample_chunks):
        bm25 = BM25Index()
        bm25.index(sample_chunks)

        results = bm25.search("JVM garbage collector", top_k=1)
        assert len(results) > 0
        assert "JVM" in results[0][0].content

    def test_empty_query(self, sample_chunks):
        bm25 = BM25Index()
        bm25.index(sample_chunks)
        results = bm25.search("", top_k=5)
        assert len(results) == 0


class TestHybridRetriever:
    def test_index_and_search(self, sample_chunks):
        """Test full hybrid retrieval pipeline."""
        # Use in-memory Qdrant
        client = QdrantClient(":memory:")
        embedding_service = EmbeddingService()

        retriever = HybridRetriever(
            embedding_service=embedding_service,
            qdrant_client=client,
            collection_name="test_collection",
        )

        retriever.create_collection()
        retriever.index_chunks(sample_chunks)

        results = retriever.search("Docker containers vs VMs", top_k=3)

        assert len(results) > 0
        # Top result should be about Docker/containers
        top_content = results[0]["content"].lower()
        assert "docker" in top_content or "container" in top_content

    def test_technical_term_retrieval(self, sample_chunks):
        """BM25 should help find exact technical terms."""
        client = QdrantClient(":memory:")
        embedding_service = EmbeddingService()

        retriever = HybridRetriever(
            embedding_service=embedding_service,
            qdrant_client=client,
            collection_name="test_terms",
        )

        retriever.create_collection()
        retriever.index_chunks(sample_chunks)

        results = retriever.search("TCP SYN ACK handshake", top_k=3)

        assert len(results) > 0
        # Should find the TCP chunk
        found_tcp = any(
            "TCP" in r["content"] or "SYN" in r["content"]
            for r in results
        )
        assert found_tcp