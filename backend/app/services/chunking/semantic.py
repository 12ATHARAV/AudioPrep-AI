"""
Strategy 3: Semantic Chunking.

How it works:
1. Split text into sentences
2. Compute embedding for each sentence
3. Calculate cosine similarity between consecutive sentences
4. When similarity drops below threshold → new chunk boundary

This finds NATURAL topic boundaries rather than arbitrary
character positions.

Example:
  "Docker uses namespaces for isolation."
  "Cgroups limit resource usage."             ← similar (Docker internals)
  "Docker Compose uses YAML files."           ← similar (Docker)
  "TCP uses a three-way handshake."           ← DIFFERENT! New chunk.
  "The SYN flag initiates the connection."    ← similar (networking)

Interview Note: "Semantic chunking uses embedding similarity
to detect topic shifts. When consecutive sentences are about
different concepts, their embeddings will be dissimilar,
creating a natural chunk boundary. This produces chunks that
are topically coherent, which improves retrieval precision."

Tradeoff: Slower than recursive (needs embeddings per sentence)
but produces higher-quality chunks for documents with multiple
interleaved topics.
"""

import numpy as np
import re

from app.services.chunking.base import ChunkingStrategy
from app.models.document import ProcessedDocument
from app.models.chunk import Chunk
from app.core.logging import get_logger

logger = get_logger("semantic_chunking")


class SemanticChunking(ChunkingStrategy):
    """
    Chunks documents based on semantic similarity between sentences.
    Creates boundaries where topic shifts occur.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.5,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1500,
    ):
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self._model = None

    @property
    def name(self) -> str:
        return "semantic"

    @property
    def model(self):
        """Lazy-load embedding model (saves RAM and startup time when not used)."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("loading_embedding_model", model=self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def chunk(self, document: ProcessedDocument) -> list[Chunk]:
        """Split document into semantically coherent chunks."""
        # Get full text
        text = document.raw_markdown
        if not text.strip():
            text = "\n\n".join(b.content for b in document.blocks)

        # Step 1: Split into sentences
        sentences = self._split_sentences(text)
        if len(sentences) < 2:
            return [
                Chunk(
                    doc_id=document.metadata.doc_id,
                    content=text.strip(),
                    strategy=self.name,
                )
            ]

        # Step 2: Embed all sentences
        logger.info("embedding_sentences", count=len(sentences))
        embeddings = self.model.encode(
            sentences, show_progress_bar=False, batch_size=32
        )

        # Step 3: Find semantic boundaries
        boundaries = self._find_boundaries(embeddings)

        # Step 4: Create chunks from boundaries
        chunks = self._create_chunks(
            sentences, boundaries, document.metadata.doc_id
        )

        logger.info(
            "semantic_chunking_complete",
            sentences=len(sentences),
            boundaries=len(boundaries),
            chunks=len(chunks),
        )

        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        """
        Split text into sentences.
        Uses regex-based splitting that handles abbreviations.
        """
        # Clean up the text
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Split on sentence boundaries
        # This pattern handles: periods, question marks, exclamation marks
        # But avoids splitting on abbreviations like "e.g.", "Dr.", etc.
        sentences = re.split(
            r"(?<=[.!?])\s+(?=[A-Z])|(?<=\n\n)",
            text,
        )

        # Filter empty and very short sentences
        return [
            s.strip()
            for s in sentences
            if s.strip() and len(s.strip()) > 10
        ]

    def _find_boundaries(
        self, embeddings: np.ndarray
    ) -> list[int]:
        """
        Find indices where semantic similarity drops.

        Calculate cosine similarity between consecutive sentences.
        When similarity drops below threshold → boundary.
        """
        similarities = []

        for i in range(len(embeddings) - 1):
            sim = self._cosine_similarity(
                embeddings[i], embeddings[i + 1]
            )
            similarities.append(sim)

        # Find boundaries where similarity drops
        boundaries = []
        for i, sim in enumerate(similarities):
            if sim < self.similarity_threshold:
                boundaries.append(i + 1)  # Boundary AFTER sentence i

        return boundaries

    def _cosine_similarity(
        self, vec1: np.ndarray, vec2: np.ndarray
    ) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def _create_chunks(
        self,
        sentences: list[str],
        boundaries: list[int],
        doc_id: str,
    ) -> list[Chunk]:
        """Create chunks from sentences and boundary indices."""
        chunks = []

        # Add start and end boundaries
        all_boundaries = [0] + boundaries + [len(sentences)]

        for i in range(len(all_boundaries) - 1):
            start = all_boundaries[i]
            end = all_boundaries[i + 1]

            chunk_sentences = sentences[start:end]
            content = " ".join(chunk_sentences)

            # Enforce max size by splitting further if needed
            if len(content) > self.max_chunk_size:
                # Simple split for oversized semantic chunks
                mid = len(chunk_sentences) // 2
                part1 = " ".join(chunk_sentences[:mid])
                part2 = " ".join(chunk_sentences[mid:])

                for part in [part1, part2]:
                    if part.strip() and len(part) >= self.min_chunk_size:
                        chunks.append(
                            Chunk(
                                doc_id=doc_id,
                                content=part.strip(),
                                strategy=self.name,
                                metadata={
                                    "sentence_range": f"{start}-{end}",
                                },
                            )
                        )
            elif len(content) >= self.min_chunk_size:
                chunks.append(
                    Chunk(
                        doc_id=doc_id,
                        content=content.strip(),
                        strategy=self.name,
                        metadata={
                            "sentence_range": f"{start}-{end}",
                        },
                    )
                )

        return chunks