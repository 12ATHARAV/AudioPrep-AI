"""
Chunking Manager — orchestrates all chunking strategies.

This is the entry point for the chunking service.
It can run one strategy or all strategies for comparison.
"""

from typing import Optional
from app.models.document import ProcessedDocument
from app.models.chunk import Chunk
from app.services.chunking.base import ChunkingStrategy
from app.services.chunking.recursive import RecursiveChunking
from app.services.chunking.structure_aware import StructureAwareChunking
from app.services.chunking.semantic import SemanticChunking
from app.services.chunking.parent_child import ParentChildChunking
from app.services.chunking.contextual import ContextualEnricher
from app.core.logging import get_logger

logger = get_logger("chunking_manager")


class ChunkingManager:
    """
    Manages all chunking strategies and provides a unified interface.
    """

    def __init__(self, apply_contextual: bool = True):
        self.apply_contextual = apply_contextual
        self.enricher = ContextualEnricher()

        # Register all strategies
        self.strategies: dict[str, ChunkingStrategy] = {
            "recursive": RecursiveChunking(),
            "structure_aware": StructureAwareChunking(),
            "semantic": SemanticChunking(),
            "parent_child": ParentChildChunking(),
        }

    def chunk(
        self,
        document: ProcessedDocument,
        strategy_name: str = "structure_aware",
    ) -> list[Chunk]:
        """
        Chunk a document using the specified strategy.

        Args:
            document: Parsed PDF document
            strategy_name: Which strategy to use

        Returns:
            List of chunks, optionally with contextual enrichment
        """
        if strategy_name not in self.strategies:
            raise ValueError(
                f"Unknown strategy: {strategy_name}. "
                f"Available: {list(self.strategies.keys())}"
            )

        strategy = self.strategies[strategy_name]
        logger.info(
            "chunking_document",
            strategy=strategy_name,
            doc=document.metadata.filename,
        )

        # Run the strategy
        chunks = strategy.chunk(document)

        # Apply contextual enrichment
        if self.apply_contextual:
            chunks = self.enricher.enrich(chunks, document)

        logger.info(
            "chunking_complete",
            strategy=strategy_name,
            total_chunks=len(chunks),
            avg_size=sum(c.char_count for c in chunks) // max(len(chunks), 1),
        )

        return chunks

    def chunk_all_strategies(
        self, document: ProcessedDocument
    ) -> dict[str, list[Chunk]]:
        """
        Run ALL strategies on the document.
        Used for evaluation and comparison.

        Returns:
            Dict mapping strategy name → list of chunks
        """
        results = {}

        for name in self.strategies:
            try:
                chunks = self.chunk(document, strategy_name=name)
                results[name] = chunks
                logger.info(
                    "strategy_result",
                    strategy=name,
                    chunks=len(chunks),
                )
            except Exception as e:
                logger.error(
                    "strategy_failed",
                    strategy=name,
                    error=str(e),
                )
                results[name] = []

        return results