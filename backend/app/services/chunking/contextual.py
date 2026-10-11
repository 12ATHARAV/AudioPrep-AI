"""
Strategy 5 (Enhancement): Contextual Chunking.

This isn't a standalone strategy — it's an ENHANCEMENT
that can be applied to any of the above strategies.

Problem:
  A chunk containing "Uses a bridge network by default."
  has low embedding quality because the embedding doesn't
  know WHAT uses a bridge network, or in what context.

Solution:
  Prepend structural context to each chunk before embedding:

  "Document: Docker Networking Guide
   Section: Network Drivers > Bridge Network
   Context: This chunk describes the default Docker network driver.

   Uses a bridge network by default."

The embedding now captures the full semantic meaning.

Interview Note: "Contextual chunking enriches each chunk with
its structural context before embedding. This is especially
valuable for short technical chunks whose meaning depends on
their position in the document. Without it, 'Uses a bridge
network by default' could be about anything — with it, the
embedding knows it's about Docker networking."

This technique was popularized by Anthropic's contextual
retrieval work and is now considered a best practice.
"""

from app.models.chunk import Chunk
from app.models.document import ProcessedDocument
from app.core.logging import get_logger

logger = get_logger("contextual_chunking")


class ContextualEnricher:
    """
    Enriches chunks with structural context.
    Can be applied to chunks from ANY strategy.
    """

    def enrich(
        self,
        chunks: list[Chunk],
        document: ProcessedDocument,
    ) -> list[Chunk]:
        """
        Add contextual prefix to each chunk's embedding text.

        The original content is preserved unchanged.
        The contextualized_content field gets the enriched version.
        """
        doc_title = (
            document.metadata.title or document.metadata.filename
        )

        enriched = []
        for chunk in chunks:
            context_prefix = self._build_context_prefix(
                chunk, doc_title
            )

            # Create enriched version
            enriched_chunk = chunk.model_copy()
            enriched_chunk.contextualized_content = (
                f"{context_prefix}\n\n{chunk.content}"
            )
            enriched.append(enriched_chunk)

        logger.info(
            "contextual_enrichment_complete",
            chunks_enriched=len(enriched),
        )

        return enriched

    def _build_context_prefix(
        self, chunk: Chunk, doc_title: str
    ) -> str:
        """
        Build the context prefix for a chunk.

        Example output:
        "Document: Docker Notes
         Section: Networking > Bridge Network
         Page: 5"
        """
        parts = [f"Document: {doc_title}"]

        if chunk.heading_hierarchy:
            section_path = " > ".join(chunk.heading_hierarchy)
            parts.append(f"Section: {section_path}")

        if chunk.section_title:
            parts.append(f"Topic: {chunk.section_title}")

        if chunk.page_number:
            parts.append(f"Page: {chunk.page_number}")

        return "\n".join(parts)