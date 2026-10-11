"""
Chunk models used across all chunking strategies.

Interview Note: Having a unified chunk model means:
1. Any strategy can produce chunks
2. The retrieval layer doesn't care which strategy created a chunk
3. We can mix strategies and compare them fairly
"""

from pydantic import BaseModel, Field
from typing import Optional
import uuid


class Chunk(BaseModel):
    """
    A single chunk of text ready for embedding and indexing.

    Every chunk carries its provenance — where it came from,
    what strategy created it, and its structural context.
    """
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    doc_id: str
    content: str  # The actual text to embed
    strategy: str  # Which strategy created this chunk

    # Structural context
    heading_hierarchy: list[str] = Field(default_factory=list)
    page_number: int = 0
    section_title: Optional[str] = None

    # For parent-child strategy
    parent_id: Optional[str] = None
    is_parent: bool = False

    # For contextual strategy
    contextualized_content: Optional[str] = None
    # ^ This is the content WITH added context prefix

    # Metadata
    char_count: int = 0
    word_count: int = 0
    metadata: dict = Field(default_factory=dict)

    def model_post_init(self, __context):
        """Calculate char and word counts after creation."""
        text = self.contextualized_content or self.content
        self.char_count = len(text)
        self.word_count = len(text.split())

    @property
    def embedding_text(self) -> str:
        """The text that will actually be embedded.
        Uses contextualized content if available."""
        return self.contextualized_content or self.content