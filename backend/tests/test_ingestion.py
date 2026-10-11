"""
Tests for PDF ingestion service.

Run with: python -m pytest tests/test_ingestion.py -v
"""

import pytest
import pymupdf
from pathlib import Path
from app.services.ingestion.pdf_parser import PDFParser
from app.models.document import BlockType


@pytest.fixture
def sample_pdf(tmp_path) -> Path:
    """Create a sample PDF for testing."""
    pdf_path = tmp_path / "test_docker_notes.pdf"

    doc = pymupdf.open()

    # Page 1: Docker basics
    page = doc.new_page()
    text = """Docker Notes

            What is Docker?

            Docker is a platform for developing, shipping, and running
            applications in containers. Containers are lightweight,
            standalone packages that include everything needed to run
            an application.

            Docker vs Virtual Machines

            Docker containers share the host OS kernel, while VMs
            include a full guest OS. This makes containers much more
            lightweight and faster to start.

            | Feature | Docker | VM |
            | Startup | Seconds | Minutes |
            | Size | MBs | GBs |
            | OS | Shared kernel | Full OS |"""

    page.insert_text((72, 72), text, fontsize=11)

    # Page 2: Networking
    page2 = doc.new_page()
    text2 = """Docker Networking

            Bridge Network

            The default network driver. Containers on the same bridge
            network can communicate with each other.

            Host Network

            Removes network isolation. The container shares the host's
            networking namespace directly.

            Overlay Network

            Used in Docker Swarm for multi-host networking.
            Enables containers on different hosts to communicate."""

    page2.insert_text((72, 72), text2, fontsize=11)

    doc.save(str(pdf_path))
    doc.close()

    return pdf_path


@pytest.fixture
def parser():
    return PDFParser()


@pytest.mark.asyncio
async def test_parse_produces_document(parser, sample_pdf):
    """Parser should produce a ProcessedDocument."""
    result = await parser.parse(sample_pdf)

    assert result is not None
    assert result.metadata.filename == "test_docker_notes.pdf"
    assert result.metadata.total_pages == 2


@pytest.mark.asyncio
async def test_parse_extracts_content(parser, sample_pdf):
    """Parser should extract text content from PDF."""
    result = await parser.parse(sample_pdf)

    # Should have some blocks
    assert len(result.blocks) > 0

    # Content should contain our text
    all_text = " ".join(b.content for b in result.blocks)
    assert "Docker" in all_text or "container" in all_text.lower()


@pytest.mark.asyncio
async def test_metadata_extraction(parser, sample_pdf):
    """Metadata should include file size and page count."""
    result = await parser.parse(sample_pdf)

    assert result.metadata.file_size_bytes > 0
    assert result.metadata.total_pages == 2
    assert result.metadata.total_blocks > 0


@pytest.mark.asyncio
async def test_fallback_on_empty_markdown(parser, tmp_path):
    """Should fallback to raw text when markdown is empty."""
    # Create a minimal PDF
    pdf_path = tmp_path / "minimal.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello World", fontsize=12)
    doc.save(str(pdf_path))
    doc.close()

    result = await parser.parse(pdf_path)

    # Should still extract something
    assert len(result.blocks) > 0 or result.raw_markdown.strip()