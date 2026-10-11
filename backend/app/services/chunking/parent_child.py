"""
Strategy 4: Parent-Child (Hierarchical) Chunking.

This is where the project gets really interesting.

How it works:
1. Create SMALL child chunks → used for precise retrieval
2. Store LARGE parent chunks → used for generation context
3. When a child is retrieved, we send its parent to the LLM

Why?
- Small chunks = better retrieval precision
  (a 100-char chunk about "bridge network" matches the query
   "What is bridge networking?" very precisely)

- Large chunks = better generation context
  (the LLM needs the surrounding paragraphs to give a
   complete, coherent explanation)

Analogy:
- Think of a book index. The index entry is tiny and precise
  (child chunk). But when you look it up, you read the full
  page (parent chunk).

Interview Note: "Parent-child retrieval separates the retrieval
unit from the generation unit. We retrieve on small, precise
child chunks but expand to larger parent chunks for context.
This gives us both high retrieval precision AND rich generation
context — solving the fundamental chunk-size tradeoff in RAG."

This is THE ANSWER to: "How do you handle the chunk size tradeoff?"
"""

from app.services.chunking.base import ChunkingStrategy
from app.models.document import ProcessedDocument, BlockType
from app.models.chunk import Chunk
import uuid


class ParentChildChunking(ChunkingStrategy):
    """
    Creates small child chunks for retrieval and
    larger parent chunks for LLM context.
    """

    def __init__(
        self,
        parent_chunk_size: int = 1500,
        child_chunk_size: int = 300,
        child_overlap: int = 50,
    ):
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size
        self.child_overlap = child_overlap

    @property
    def name(self) -> str:
        return "parent_child"

    def chunk(self, document: ProcessedDocument) -> list[Chunk]:
        """
        Create parent chunks, then split each into children.

        Returns BOTH parents and children.
        Children have parent_id pointing to their parent.
        Parents have is_parent=True.
        """
        all_chunks = []

        # Step 1: Create parent chunks (large, section-based)
        parents = self._create_parents(document)

        for parent in parents:
            all_chunks.append(parent)

            # Step 2: Create child chunks from each parent
            children = self._split_into_children(parent)
            all_chunks.extend(children)

        return all_chunks

    def _create_parents(
        self, document: ProcessedDocument
    ) -> list[Chunk]:
        """
        Create large parent chunks from sections.
        Each section becomes one parent (if it fits).
        """
        parents = []
        current_text = ""
        current_hierarchy = []
        current_page = 1

        for block in document.blocks:
            if block.block_type == BlockType.HEADING:
                # Save current parent if it has content
                if current_text.strip():
                    parents.append(
                        Chunk(
                            doc_id=document.metadata.doc_id,
                            content=current_text.strip(),
                            strategy=self.name,
                            heading_hierarchy=current_hierarchy.copy(),
                            page_number=current_page,
                            is_parent=True,
                            metadata={"chunk_role": "parent"},
                        )
                    )
                current_text = ""
                current_hierarchy = list(block.heading_hierarchy)
                current_page = block.page_number
            else:
                current_text += block.content + "\n\n"

                # If parent is getting too large, split
                if len(current_text) > self.parent_chunk_size:
                    parents.append(
                        Chunk(
                            doc_id=document.metadata.doc_id,
                            content=current_text.strip(),
                            strategy=self.name,
                            heading_hierarchy=current_hierarchy.copy(),
                            page_number=current_page,
                            is_parent=True,
                            metadata={"chunk_role": "parent"},
                        )
                    )
                    current_text = ""

        # Last parent
        if current_text.strip():
            parents.append(
                Chunk(
                    doc_id=document.metadata.doc_id,
                    content=current_text.strip(),
                    strategy=self.name,
                    heading_hierarchy=current_hierarchy.copy(),
                    page_number=current_page,
                    is_parent=True,
                    metadata={"chunk_role": "parent"},
                )
            )

        return parents

    def _split_into_children(self, parent: Chunk) -> list[Chunk]:
        """
        Split a parent chunk into smaller child chunks.
        Each child points back to its parent.
        """
        children = []
        text = parent.content
        start = 0

        while start < len(text):
            end = min(start + self.child_chunk_size, len(text))

            # Try to find a good break point (sentence end) if not at end of text
            if end < len(text):
                # Search only in a window near 'end' to avoid snapping back to previous boundaries
                search_start = max(start + self.child_overlap + 1, end - 50)
                search_end = min(len(text), end + 50)
                for boundary in [". ", ".\n", "\n\n", "\n"]:
                    break_pos = text.rfind(boundary, search_start, search_end)
                    if break_pos != -1:
                        end = break_pos + len(boundary)
                        break

            child_text = text[start:end].strip()

            if child_text and len(child_text) > 30:  # Skip tiny fragments
                children.append(
                    Chunk(
                        doc_id=parent.doc_id,
                        content=child_text,
                        strategy=self.name,
                        heading_hierarchy=parent.heading_hierarchy.copy(),
                        page_number=parent.page_number,
                        parent_id=parent.chunk_id,
                        is_parent=False,
                        metadata={"chunk_role": "child"},
                    )
                )

            if end >= len(text):
                break

            # Ensure start strictly advances past previous start
            next_start = end - self.child_overlap
            if next_start <= start:
                start = end
            else:
                start = next_start

        return children