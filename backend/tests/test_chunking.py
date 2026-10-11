"""
Tests for all chunking strategies.

Run with: python -m pytest tests/test_chunking.py -v
"""

import pytest
from app.models.document import (
    ProcessedDocument,
    DocumentMetadata,
    ContentBlock,
    BlockType,
)
from app.models.chunk import Chunk
from app.services.chunking.recursive import RecursiveChunking
from app.services.chunking.structure_aware import StructureAwareChunking
from app.services.chunking.parent_child import ParentChildChunking
from app.services.chunking.contextual import ContextualEnricher
from app.services.chunking.manager import ChunkingManager


@pytest.fixture
def sample_document() -> ProcessedDocument:
    """Create a sample processed document for testing."""
    blocks = [
        ContentBlock(
            block_type=BlockType.HEADING,
            content="Docker",
            page_number=1,
            heading_hierarchy=["Docker"],
            metadata={"level": 1},
        ),
        ContentBlock(
            block_type=BlockType.HEADING,
            content="What is Docker?",
            page_number=1,
            heading_hierarchy=["Docker", "What is Docker?"],
            metadata={"level": 2},
        ),
        ContentBlock(
            block_type=BlockType.PARAGRAPH,
            content=(
                "Docker is a platform for developing, shipping, and running "
                "applications in containers. Containers are lightweight, "
                "standalone, executable packages that include everything "
                "needed to run an application: code, runtime, system tools, "
                "libraries, and settings."
            ),
            page_number=1,
            heading_hierarchy=["Docker", "What is Docker?"],
        ),
        ContentBlock(
            block_type=BlockType.HEADING,
            content="Docker Networking",
            page_number=2,
            heading_hierarchy=["Docker", "Docker Networking"],
            metadata={"level": 2},
        ),
        ContentBlock(
            block_type=BlockType.HEADING,
            content="Bridge Network",
            page_number=2,
            heading_hierarchy=[
                "Docker", "Docker Networking", "Bridge Network"
            ],
            metadata={"level": 3},
        ),
        ContentBlock(
            block_type=BlockType.PARAGRAPH,
            content=(
                "The bridge network is Docker's default network driver. "
                "When you start a container without specifying a network, "
                "it connects to the default bridge network. Containers on "
                "the same bridge can communicate using IP addresses."
            ),
            page_number=2,
            heading_hierarchy=[
                "Docker", "Docker Networking", "Bridge Network"
            ],
        ),
        ContentBlock(
            block_type=BlockType.HEADING,
            content="Host Network",
            page_number=2,
            heading_hierarchy=[
                "Docker", "Docker Networking", "Host Network"
            ],
            metadata={"level": 3},
        ),
        ContentBlock(
            block_type=BlockType.PARAGRAPH,
            content=(
                "Host networking removes the network isolation between "
                "the Docker host and the containers. The container shares "
                "the host's networking namespace. This is faster because "
                "there is no NAT overhead."
            ),
            page_number=2,
            heading_hierarchy=[
                "Docker", "Docker Networking", "Host Network"
            ],
        ),
    ]

    markdown = "\n\n".join(
        f"{'#' * b.metadata.get('level', 0)} {b.content}"
        if b.block_type == BlockType.HEADING
        else b.content
        for b in blocks
    )

    return ProcessedDocument(
        metadata=DocumentMetadata(
            filename="docker_notes.pdf",
            title="Docker Notes",
            total_pages=2,
        ),
        blocks=blocks,
        raw_markdown=markdown,
    )


class TestRecursiveChunking:
    def test_produces_chunks(self, sample_document):
        strategy = RecursiveChunking(chunk_size=200, chunk_overlap=50)
        chunks = strategy.chunk(sample_document)
        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_strategy_name(self):
        strategy = RecursiveChunking()
        assert strategy.name == "recursive"

    def test_respects_chunk_size(self, sample_document):
        strategy = RecursiveChunking(chunk_size=300, chunk_overlap=0)
        chunks = strategy.chunk(sample_document)
        for chunk in chunks:
            # Allow some tolerance for overlap
            assert len(chunk.content) <= 500


class TestStructureAwareChunking:
    def test_produces_chunks(self, sample_document):
        strategy = StructureAwareChunking()
        chunks = strategy.chunk(sample_document)
        assert len(chunks) > 0

    def test_preserves_hierarchy(self, sample_document):
        strategy = StructureAwareChunking()
        chunks = strategy.chunk(sample_document)

        # At least some chunks should have heading hierarchy
        chunks_with_hierarchy = [
            c for c in chunks if c.heading_hierarchy
        ]
        assert len(chunks_with_hierarchy) > 0

    def test_separate_sections(self, sample_document):
        """Bridge and Host should be in separate chunks."""
        strategy = StructureAwareChunking()
        chunks = strategy.chunk(sample_document)

        bridge_chunks = [
            c for c in chunks if "bridge" in c.content.lower()
        ]
        host_chunks = [
            c for c in chunks if "host" in c.content.lower()
            and "bridge" not in c.content.lower()
        ]

        # They should be in different chunks
        if bridge_chunks and host_chunks:
            assert bridge_chunks[0].chunk_id != host_chunks[0].chunk_id

    def test_strategy_name(self):
        strategy = StructureAwareChunking()
        assert strategy.name == "structure_aware"


class TestParentChildChunking:
    def test_produces_both_parents_and_children(self, sample_document):
        strategy = ParentChildChunking(
            parent_chunk_size=1000, child_chunk_size=150
        )
        chunks = strategy.chunk(sample_document)

        parents = [c for c in chunks if c.is_parent]
        children = [c for c in chunks if not c.is_parent]

        assert len(parents) > 0
        assert len(children) > 0

    def test_children_reference_parents(self, sample_document):
        strategy = ParentChildChunking(
            parent_chunk_size=1000, child_chunk_size=150
        )
        chunks = strategy.chunk(sample_document)

        children = [c for c in chunks if not c.is_parent]
        parent_ids = {c.chunk_id for c in chunks if c.is_parent}

        for child in children:
            assert child.parent_id in parent_ids

    def test_children_smaller_than_parents(self, sample_document):
        strategy = ParentChildChunking(
            parent_chunk_size=1000, child_chunk_size=200
        )
        chunks = strategy.chunk(sample_document)

        parents = [c for c in chunks if c.is_parent]
        children = [c for c in chunks if not c.is_parent]

        if parents and children:
            avg_parent = sum(len(p.content) for p in parents) / len(parents)
            avg_child = sum(len(c.content) for c in children) / len(children)
            assert avg_child < avg_parent


class TestContextualEnricher:
    def test_adds_context(self, sample_document):
        # First create some chunks
        strategy = StructureAwareChunking()
        chunks = strategy.chunk(sample_document)

        # Then enrich them
        enricher = ContextualEnricher()
        enriched = enricher.enrich(chunks, sample_document)

        for chunk in enriched:
            assert chunk.contextualized_content is not None
            assert "Document:" in chunk.contextualized_content
            # Original content should still be there
            assert chunk.content in chunk.contextualized_content

    def test_embedding_text_uses_contextualized(self, sample_document):
        strategy = StructureAwareChunking()
        chunks = strategy.chunk(sample_document)

        enricher = ContextualEnricher()
        enriched = enricher.enrich(chunks, sample_document)

        for chunk in enriched:
            assert chunk.embedding_text == chunk.contextualized_content


class TestChunkingManager:
    def test_default_strategy(self, sample_document):
        manager = ChunkingManager(apply_contextual=False)
        chunks = manager.chunk(sample_document, "structure_aware")
        assert len(chunks) > 0

    def test_invalid_strategy_raises(self, sample_document):
        manager = ChunkingManager()
        with pytest.raises(ValueError):
            manager.chunk(sample_document, "nonexistent")

    def test_all_strategies(self, sample_document):
        manager = ChunkingManager(apply_contextual=False)
        results = manager.chunk_all_strategies(sample_document)

        assert "recursive" in results
        assert "structure_aware" in results
        assert "parent_child" in results
        # Each should produce chunks
        for name, chunks in results.items():
            assert len(chunks) > 0, f"{name} produced no chunks"