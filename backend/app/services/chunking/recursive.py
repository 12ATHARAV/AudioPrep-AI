"""
Strategy 1: Recursive Character Text Splitting.

This is our BASELINE. We compare all other strategies against this.

How it works:
1. Try to split on "\n\n" (paragraphs)
2. If chunks are still too big, split on "\n"
3. If still too big, split on ". "
4. Last resort: split on " "

Problems:
- Can split mid-sentence or mid-concept
- No awareness of document structure
- A chunk about "Bridge Network" might include
  the start of "Host Network" section

Interview Note: When asked "Why not just use this?", answer:
"Recursive splitting is structure-unaware. A chunk might
contain the tail of one concept and the head of another,
making retrieval imprecise. It's a good baseline for
comparison, but structure-aware or semantic strategies
produce higher-quality retrieval contexts."
"""

from app.services.chunking.base import ChunkingStrategy
from app.models.document import ProcessedDocument
from app.models.chunk import Chunk


class RecursiveChunking(ChunkingStrategy):
    """
    Recursive character-based text splitting.
    Splits on paragraph boundaries, then sentences, then words.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        separators: list[str] | None = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " "]

    @property
    def name(self) -> str:
        return "recursive"

    def chunk(self, document: ProcessedDocument) -> list[Chunk]:
        """Split the full markdown into fixed-size chunks."""
        text = document.raw_markdown
        if not text.strip():
            # Fallback: concatenate all block contents
            text = "\n\n".join(
                block.content for block in document.blocks
            )

        raw_chunks = self._recursive_split(text, self.separators)

        chunks = []
        for i, chunk_text in enumerate(raw_chunks):
            if not chunk_text.strip():
                continue

            chunks.append(
                Chunk(
                    doc_id=document.metadata.doc_id,
                    content=chunk_text.strip(),
                    strategy=self.name,
                    metadata={
                        "chunk_index": i,
                        "total_chunks": len(raw_chunks),
                    },
                )
            )

        return chunks

    def _recursive_split(
        self, text: str, separators: list[str]
    ) -> list[str]:
        """
        Recursively split text using a hierarchy of separators.

        This mimics LangChain's RecursiveCharacterTextSplitter
        but without the dependency.
        """
        if not text:
            return []

        # If text fits in one chunk, return it
        if len(text) <= self.chunk_size:
            return [text]

        # Try each separator in order
        final_chunks = []
        separator = separators[0] if separators else " "
        remaining_separators = separators[1:] if len(separators) > 1 else []

        splits = text.split(separator)

        current_chunk = ""
        for split in splits:
            candidate = (
                current_chunk + separator + split
                if current_chunk
                else split
            )

            if len(candidate) <= self.chunk_size:
                current_chunk = candidate
            else:
                if current_chunk:
                    # If current chunk is still too big, recurse
                    if len(current_chunk) > self.chunk_size and remaining_separators:
                        final_chunks.extend(
                            self._recursive_split(
                                current_chunk, remaining_separators
                            )
                        )
                    else:
                        final_chunks.append(current_chunk)

                current_chunk = split

        if current_chunk:
            if len(current_chunk) > self.chunk_size and remaining_separators:
                final_chunks.extend(
                    self._recursive_split(
                        current_chunk, remaining_separators
                    )
                )
            else:
                final_chunks.append(current_chunk)

        # Apply overlap
        if self.chunk_overlap > 0 and len(final_chunks) > 1:
            final_chunks = self._apply_overlap(final_chunks)

        return final_chunks

    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        """Add overlap between consecutive chunks."""
        overlapped = [chunks[0]]

        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            # Take the last `overlap` chars from previous chunk
            overlap_text = prev[-self.chunk_overlap:]
            # Prepend to current chunk
            overlapped.append(overlap_text + chunks[i])

        return overlapped