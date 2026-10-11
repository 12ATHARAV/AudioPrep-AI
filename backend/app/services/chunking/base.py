"""
Abstract base class for all chunking strategies.

Interview Note: Using an abstract base class here is the
Strategy Pattern. Each chunking implementation follows the
same interface, so the rest of the system doesn't need to
know which strategy is being used.

This is exactly what an interviewer wants to hear when they
ask "How did you design your chunking system?"
"""

from abc import ABC, abstractmethod
from app.models.document import ProcessedDocument
from app.models.chunk import Chunk


class ChunkingStrategy(ABC):
    """Base class for all chunking strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this strategy."""
        ...

    @abstractmethod
    def chunk(self, document: ProcessedDocument) -> list[Chunk]:
        """
        Split a processed document into chunks.

        Args:
            document: The parsed PDF document

        Returns:
            List of Chunk objects ready for embedding
        """
        ...

    def _count_tokens_approx(self, text: str) -> int:
        """Approximate token count (1 token ≈ 4 chars)."""
        return len(text) // 4