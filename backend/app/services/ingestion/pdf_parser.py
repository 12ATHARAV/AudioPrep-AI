"""
PDF parsing service using PyMuPDF and pymupdf4llm.

Architecture:
1. pymupdf4llm.to_markdown() → structured markdown (primary)
2. fitz page-level extraction → structural analysis (headings, blocks)
3. fitz find_tables() → table extraction
4. fitz OCR fallback → for scanned PDFs

Interview Note: We use TWO extraction passes:
- Pass 1 (pymupdf4llm): Gets us clean markdown with proper
  heading hierarchy. This is our primary content.
- Pass 2 (raw fitz): Gets us font-level metadata so we can
  detect headings by size/weight for structure-aware chunking.

Why two passes? pymupdf4llm is great for content but doesn't
expose font metadata. Raw fitz gives us the structural signals
we need for our chunking strategies.
"""

import pymupdf
import pymupdf4llm
import re
from pathlib import Path
from typing import Optional

from app.models.document import (
    ProcessedDocument,
    DocumentMetadata,
    ContentBlock,
    TableBlock,
    BlockType,
)
from app.core.logging import get_logger

logger = get_logger("pdf_parser")


class PDFParser:
    """
    Extracts structured content from PDF files.

    This is our primary parser. Falls back to raw text
    extraction if structured extraction fails.
    """

    def __init__(self):
        self._heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    async def parse(self, file_path: Path) -> ProcessedDocument:
        """
        Parse a PDF file into a ProcessedDocument.

        Returns a structured document with:
        - Markdown content
        - Content blocks with heading hierarchy
        - Extracted tables
        - Document metadata
        """
        logger.info("parsing_pdf", file_path=str(file_path))

        doc = pymupdf.open(str(file_path))

        try:
            # Step 1: Extract structured markdown
            markdown_content = self._extract_markdown(file_path)

            # Step 2: Extract metadata
            metadata = self._extract_metadata(doc, file_path)

            # Step 3: Parse markdown into content blocks
            blocks = self._markdown_to_blocks(markdown_content)

            # Step 4: Extract tables separately
            table_blocks = self._extract_tables(doc)

            # Step 5: If markdown extraction produced nothing, fallback
            if not blocks:
                logger.warning("markdown_empty_fallback", file=str(file_path))
                blocks = self._fallback_extraction(doc)

            metadata.total_blocks = len(blocks)

            processed = ProcessedDocument(
                metadata=metadata,
                blocks=blocks,
                raw_markdown=markdown_content,
                table_blocks=table_blocks,
            )

            logger.info(
                "pdf_parsed",
                file=file_path.name,
                pages=metadata.total_pages,
                blocks=len(blocks),
                tables=len(table_blocks),
            )

            return processed

        finally:
            doc.close()

    def _extract_markdown(self, file_path: Path) -> str:
        """
        Use pymupdf4llm for structured markdown extraction.

        This preserves:
        - Heading hierarchy (# ## ### etc.)
        - Bold/italic formatting
        - Code blocks
        - Lists
        - Table structures
        """
        try:
            md_text = pymupdf4llm.to_markdown(str(file_path))
            return md_text
        except Exception as e:
            logger.error("pymupdf4llm_failed", error=str(e))
            return ""

    def _extract_metadata(
        self, doc: pymupdf.Document, file_path: Path
    ) -> DocumentMetadata:
        """Extract document metadata."""
        meta = doc.metadata or {}

        return DocumentMetadata(
            filename=file_path.name,
            title=meta.get("title") or file_path.stem,
            author=meta.get("author"),
            total_pages=len(doc),
            file_size_bytes=file_path.stat().st_size,
        )

    def _markdown_to_blocks(self, markdown: str) -> list[ContentBlock]:
        """
        Parse markdown text into structured ContentBlocks.

        This is where we build the heading hierarchy that
        our structure-aware chunking strategy will use.

        Example:
          # Docker
          ## Networking
          ### Bridge Network
          Some content here...

        Produces a block with:
          heading_hierarchy = ["Docker", "Networking", "Bridge Network"]
          content = "Some content here..."
        """
        blocks = []
        current_hierarchy = {}  # level -> heading text
        current_content_lines = []
        current_page = 1

        lines = markdown.split("\n")

        for line in lines:
            # Track page breaks (pymupdf4llm inserts these)
            if line.strip() == "-----" or "<!-- Page" in line:
                current_page += 1
                continue

            # Check if this line is a heading
            heading_match = self._heading_pattern.match(line.strip())

            if heading_match:
                # Save any accumulated content as a block
                if current_content_lines:
                    content = "\n".join(current_content_lines).strip()
                    if content:
                        hierarchy = self._get_hierarchy(current_hierarchy)
                        blocks.append(
                            ContentBlock(
                                block_type=BlockType.PARAGRAPH,
                                content=content,
                                page_number=current_page,
                                heading_hierarchy=hierarchy,
                            )
                        )
                    current_content_lines = []

                # Update heading hierarchy
                level = len(heading_match.group(1))  # number of #
                heading_text = heading_match.group(2).strip()

                # Clear all lower-level headings
                current_hierarchy[level] = heading_text
                for l in list(current_hierarchy.keys()):
                    if l > level:
                        del current_hierarchy[l]

                # Also add the heading itself as a block
                blocks.append(
                    ContentBlock(
                        block_type=BlockType.HEADING,
                        content=heading_text,
                        page_number=current_page,
                        heading_hierarchy=self._get_hierarchy(
                            current_hierarchy
                        ),
                        metadata={"level": level},
                    )
                )
            else:
                # Detect block type
                stripped = line.strip()
                if stripped.startswith("```"):
                    # Collect code block
                    current_content_lines.append(line)
                elif stripped.startswith(("- ", "* ", "1.", "2.", "3.")):
                    current_content_lines.append(line)
                elif stripped:
                    current_content_lines.append(line)

        # Don't forget the last block
        if current_content_lines:
            content = "\n".join(current_content_lines).strip()
            if content:
                hierarchy = self._get_hierarchy(current_hierarchy)
                blocks.append(
                    ContentBlock(
                        block_type=BlockType.PARAGRAPH,
                        content=content,
                        page_number=current_page,
                        heading_hierarchy=hierarchy,
                    )
                )

        return blocks

    def _get_hierarchy(self, hierarchy_dict: dict) -> list[str]:
        """Convert hierarchy dict to ordered list."""
        return [
            hierarchy_dict[level]
            for level in sorted(hierarchy_dict.keys())
        ]

    def _extract_tables(self, doc: pymupdf.Document) -> list[TableBlock]:
        """
        Extract tables from the PDF using PyMuPDF's find_tables().

        Interview Note: Tables are notoriously difficult in PDFs
        because PDF is a visual format, not semantic. PyMuPDF uses
        line detection and cell boundary analysis — not ML — which
        makes it fast but can miss complex merged-cell tables.
        """
        table_blocks = []

        for page_num, page in enumerate(doc, start=1):
            try:
                tables = page.find_tables()
                for table in tables:
                    # Extract as list of lists
                    data = table.extract()
                    if not data or len(data) < 2:
                        continue

                    headers = [str(h) if h else "" for h in data[0]]
                    rows = [
                        [str(cell) if cell else "" for cell in row]
                        for row in data[1:]
                    ]

                    # Build markdown representation
                    md_lines = []
                    md_lines.append(
                        "| " + " | ".join(headers) + " |"
                    )
                    md_lines.append(
                        "| " + " | ".join(["---"] * len(headers)) + " |"
                    )
                    for row in rows:
                        md_lines.append(
                            "| " + " | ".join(row) + " |"
                        )

                    md_repr = "\n".join(md_lines)

                    table_blocks.append(
                        TableBlock(
                            block_type=BlockType.TABLE,
                            content=md_repr,
                            page_number=page_num,
                            headers=headers,
                            rows=rows,
                            markdown_repr=md_repr,
                        )
                    )
            except Exception as e:
                logger.warning(
                    "table_extraction_failed",
                    page=page_num,
                    error=str(e),
                )

        return table_blocks

    def _fallback_extraction(
        self, doc: pymupdf.Document
    ) -> list[ContentBlock]:
        """
        Fallback: extract raw text when markdown extraction fails.

        This handles scanned PDFs or heavily formatted documents
        where pymupdf4llm produces empty output.

        Interview Note: Always have a fallback. This is the
        'Graceful Degradation' principle. We lose structure
        but at least we get content.
        """
        blocks = []

        for page_num, page in enumerate(doc, start=1):
            # Try normal text extraction first
            text = page.get_text("text")

            if not text.strip():
                # Try OCR for scanned pages
                try:
                    tp = page.get_textpage_ocr(language="eng")
                    text = page.get_text("text", textpage=tp)
                    logger.info("ocr_applied", page=page_num)
                except Exception as e:
                    logger.warning(
                        "ocr_failed", page=page_num, error=str(e)
                    )
                    continue

            if text.strip():
                blocks.append(
                    ContentBlock(
                        block_type=BlockType.PARAGRAPH,
                        content=text.strip(),
                        page_number=page_num,
                        heading_hierarchy=[],
                        metadata={"extraction": "fallback"},
                    )
                )

        return blocks


# Factory function for easy importing
def create_pdf_parser() -> PDFParser:
    return PDFParser()