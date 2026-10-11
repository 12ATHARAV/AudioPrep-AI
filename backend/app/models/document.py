"""
Domain models for document processing.

These are our internal representations — not tied to any
specific PDF library. This means we can swap PyMuPDF for
Docling later without changing anything downstream.

Interview Note: This is the Dependency Inversion Principle.
Our chunking and retrieval services depend on these models,
not on PyMuPDF directly.
"""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum
from datetime import datetime, timezone
import uuid


class BlockType(str, Enum):
    """Types of content blocks in a document."""
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"
    CODE = "code"
    IMAGE_CAPTION = "image_caption"
    UNKNOWN = "unknown"


class ContentBlock(BaseModel):
    """
    A single block of content from a document.

    This is our atomic unit before chunking.
    Each block preserves its structural context.
    """
    block_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    block_type: BlockType
    content: str
    page_number: int
    heading_hierarchy: list[str] = Field(default_factory=list)
    # e.g., ["Docker", "Networking", "Bridge Network"]
    metadata: dict = Field(default_factory=dict)


class TableBlock(ContentBlock):
    """A table extracted from the document."""
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    markdown_repr: str = ""  # Table as markdown string


class DocumentMetadata(BaseModel):
    """Metadata about the entire document."""
    doc_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    title: Optional[str] = None
    author: Optional[str] = None
    total_pages: int = 0
    total_blocks: int = 0
    subjects_detected: list[str] = Field(default_factory=list)
    upload_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    file_size_bytes: int = 0


class ProcessedDocument(BaseModel):
    """Complete processed document with all blocks and metadata."""
    metadata: DocumentMetadata
    blocks: list[ContentBlock] = Field(default_factory=list)
    raw_markdown: str = ""  # Full markdown from pymupdf4llm
    table_blocks: list[TableBlock] = Field(default_factory=list)