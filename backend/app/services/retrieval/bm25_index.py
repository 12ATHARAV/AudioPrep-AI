"""
BM25 sparse retrieval index.

BM25 (Best Matching 25) is a probabilistic retrieval function
that scores documents based on term frequency and inverse
document frequency.

Why BM25 alongside vector search?

Vector search: "explain memory management" → understands
  that this relates to "garbage collection", "heap", "stack"

BM25: "JVM G1 GC" → finds documents containing exactly
  "JVM", "G1", "GC" — vector search might rank general
  memory management docs higher.

Interview Note: "BM25 provides exact lexical matching that
complements semantic search. Technical documents contain
precise terms (JVM, G1, ZGC, k8s) that vector embeddings
may not capture precisely because they focus on semantic
similarity. Hybrid retrieval combines both signals."
"""

import re
import math
from collections import Counter
from app.models.chunk import Chunk
from app.core.logging import get_logger

logger = get_logger("bm25")


class BM25Index:
    """
    BM25 sparse retrieval index.

    Implements BM25 scoring from scratch so we understand
    exactly how it works (great for interviews).
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Args:
            k1: Term frequency saturation parameter.
                Higher = more weight to term frequency.
            b: Document length normalization.
                0 = no normalization, 1 = full normalization.
        """
        self.k1 = k1
        self.b = b
        self.documents: list[Chunk] = []
        self.doc_term_freqs: list[Counter] = []
        self.doc_lengths: list[int] = []
        self.avg_doc_length: float = 0.0
        self.idf_cache: dict[str, float] = {}
        self.n_docs: int = 0

    def index(self, chunks: list[Chunk]) -> None:
        """
        Build the BM25 index from a list of chunks.

        Steps:
        1. Tokenize each chunk
        2. Calculate term frequencies
        3. Compute IDF for each term
        4. Store for scoring
        """
        self.documents = chunks
        self.n_docs = len(chunks)

        # Tokenize and count term frequencies
        self.doc_term_freqs = []
        self.doc_lengths = []

        for chunk in chunks:
            tokens = self._tokenize(chunk.embedding_text)
            self.doc_term_freqs.append(Counter(tokens))
            self.doc_lengths.append(len(tokens))

        self.avg_doc_length = (
            sum(self.doc_lengths) / max(self.n_docs, 1)
        )

        # Pre-compute IDF for all terms
        self._compute_idf()

        logger.info(
            "bm25_indexed",
            documents=self.n_docs,
            unique_terms=len(self.idf_cache),
            avg_doc_length=round(self.avg_doc_length, 1),
        )

    def search(
        self, query: str, top_k: int = 10
    ) -> list[tuple[Chunk, float]]:
        """
        Search the BM25 index.

        Returns list of (chunk, score) tuples sorted by score.
        """
        query_tokens = self._tokenize(query)

        scores = []
        for i in range(self.n_docs):
            score = self._score_document(query_tokens, i)
            if score > 0:
                scores.append((self.documents[i], score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:top_k]

    def _tokenize(self, text: str) -> list[str]:
        """
        Simple tokenization: lowercase, split on non-alphanumeric.

        Interview Note: In production, you'd want stemming
        or lemmatization. For our use case, simple tokenization
        works because technical terms (Docker, JVM, TCP) are
        already in their base form.
        """
        text = text.lower()
        tokens = re.findall(r"\b\w+\b", text)
        # Remove very short tokens but keep technical abbreviations
        return [t for t in tokens if len(t) > 1]

    def _compute_idf(self) -> None:
        """
        Compute Inverse Document Frequency for all terms.

        IDF = log((N - n + 0.5) / (n + 0.5))

        Where:
          N = total documents
          n = documents containing the term

        High IDF = rare term = more discriminative
        Low IDF = common term = less useful for retrieval
        """
        # Count document frequency for each term
        df = Counter()
        for doc_tf in self.doc_term_freqs:
            for term in doc_tf:
                df[term] += 1

        # Compute IDF
        for term, freq in df.items():
            self.idf_cache[term] = math.log(
                (self.n_docs - freq + 0.5) / (freq + 0.5) + 1
            )

    def _score_document(
        self, query_tokens: list[str], doc_idx: int
    ) -> float:
        """
        Score a single document against the query.

        BM25 score = Σ IDF(qi) * (f(qi, D) * (k1 + 1)) /
                     (f(qi, D) + k1 * (1 - b + b * |D|/avgdl))

        Where:
          qi = query term
          f(qi, D) = term frequency in document
          |D| = document length
          avgdl = average document length
        """
        score = 0.0
        doc_tf = self.doc_term_freqs[doc_idx]
        doc_len = self.doc_lengths[doc_idx]

        for token in query_tokens:
            if token not in self.idf_cache:
                continue

            tf = doc_tf.get(token, 0)
            if tf == 0:
                continue

            idf = self.idf_cache[token]

            # BM25 scoring formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (
                1 - self.b + self.b * doc_len / self.avg_doc_length
            )

            score += idf * (numerator / denominator)

        return score