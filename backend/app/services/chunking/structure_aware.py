"""
Strategy 2: Structure-Aware Chunking.

This is our FIRST SERIOUS implementation.

How it works:
1. Respect document structure (headings, sections)
2. Never split across section boundaries
3. Keep heading context with every chunk
4. Group related blocks under the same heading

Example:
  # Docker
  ## Networking
  ### Bridge Network
  Bridge is the default...
  It creates an isolated...

  ### Host Network
  Host networking removes...

Structure-aware produces:
  Chunk 1: {
    hierarchy: ["Docker", "Networking", "Bridge Network"],
    content: "Bridge is the default... It creates an isolated..."
  }
  Chunk 2: {
    hierarchy: ["Docker", "Networking", "Host Network"],
    content: "Host networking removes..."
  }

Recursive baseline would produce:
  Chunk 1: "...Bridge is the default... It creates an isolated..."
  Chunk 2: "...ed... Host networking removes..."
  ^ Lost the section boundary! Chunk 2 has mixed concepts.

Interview Note: "Structure-aware chunking preserves semantic
boundaries. Each chunk maps to exactly one concept in the
document, which dramatically improves retrieval precision
because the retrieved context is coherent and complete."
"""

from app.services.chunking.base import ChunkingStrategy
from app.models.document import ProcessedDocument, BlockType
from app.models.chunk import Chunk


class StructureAwareChunking(ChunkingStrategy):
    """
    Chunks documents by respecting structural boundaries.
    Never splits across section headings.
    """

    def __init__(
        self,
        max_chunk_size: int = 1000,
        min_chunk_size: int = 100,
    ):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size

    @property
    def name(self) -> str:
        return "structure_aware"

    def chunk(self, document: ProcessedDocument) -> list[Chunk]:
        """
        Group content blocks by their heading hierarchy,
        then split large sections if needed.
        """
        if not document.blocks:
            return []

        # Step 1: Group blocks by section
        sections = self._group_by_section(document)

        # Step 2: Convert sections to chunks
        chunks = []
        for section in sections:
            section_chunks = self._section_to_chunks(
                section, document.metadata.doc_id
            )
            chunks.extend(section_chunks)

        # Step 3: Add table blocks as separate chunks
        for table in document.table_blocks:
            chunks.append(
                Chunk(
                    doc_id=document.metadata.doc_id,
                    content=table.markdown_repr or table.content,
                    strategy=self.name,
                    heading_hierarchy=table.heading_hierarchy,
                    page_number=table.page_number,
                    section_title="Table",
                    metadata={"is_table": True},
                )
            )

        return chunks

    def _group_by_section(
        self, document: ProcessedDocument
    ) -> list[dict]:
        """
        Group content blocks by their nearest heading.

        Each section = {
            "hierarchy": ["Docker", "Networking", "Bridge"],
            "heading": "Bridge Network",
            "blocks": [block1, block2, ...],
            "page": 5
        }
        """
        sections = []
        current_section = {
            "hierarchy": [],
            "heading": "",
            "blocks": [],
            "page": 1,
        }

        for block in document.blocks:
            if block.block_type == BlockType.HEADING:
                # Save current section if it has content
                if current_section["blocks"]:
                    sections.append(current_section)

                # Start new section
                current_section = {
                    "hierarchy": list(block.heading_hierarchy),
                    "heading": block.content,
                    "blocks": [],
                    "page": block.page_number,
                }
            else:
                current_section["blocks"].append(block)

        # Don't forget the last section
        if current_section["blocks"]:
            sections.append(current_section)

        return sections

    def _section_to_chunks(
        self, section: dict, doc_id: str
    ) -> list[Chunk]:
        """
        Convert a section to one or more chunks.

        If the section is small enough, it becomes one chunk.
        If it's too large, we split on paragraph boundaries
        WITHIN the section (preserving hierarchy).
        """
        # Concatenate all block contents in this section
        content = "\n\n".join(
            block.content for block in section["blocks"]
        )

        if not content.strip():
            return []

        # If section fits in one chunk, great
        if len(content) <= self.max_chunk_size:
            return [
                Chunk(
                    doc_id=doc_id,
                    content=content.strip(),
                    strategy=self.name,
                    heading_hierarchy=section["hierarchy"],
                    page_number=section["page"],
                    section_title=section["heading"],
                )
            ]

        # Section too large — split on paragraphs
        paragraphs = content.split("\n\n")
        chunks = []
        current_text = ""

        for para in paragraphs:
            if not para.strip():
                continue

            candidate = (
                current_text + "\n\n" + para if current_text else para
            )

            if len(candidate) <= self.max_chunk_size:
                current_text = candidate
            else:
                if current_text:
                    chunks.append(
                        Chunk(
                            doc_id=doc_id,
                            content=current_text.strip(),
                            strategy=self.name,
                            heading_hierarchy=section["hierarchy"],
                            page_number=section["page"],
                            section_title=section["heading"],
                        )
                    )
                current_text = para

        if current_text and len(current_text) >= self.min_chunk_size:
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    content=current_text.strip(),
                    strategy=self.name,
                    heading_hierarchy=section["hierarchy"],
                    page_number=section["page"],
                    section_title=section["heading"],
                )
            )

        return chunks