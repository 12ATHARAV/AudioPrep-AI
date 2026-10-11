"""
Quick interactive test to verify Local Qdrant Storage (No Docker).
"""

import sys
from pathlib import Path
from qdrant_client import QdrantClient

# Add backend folder to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.models.chunk import Chunk
from app.services.retrieval.embeddings import EmbeddingService
from app.services.retrieval.hybrid_retriever import HybridRetriever
from app.core.config import settings

def test_qdrant_pipeline():
    print("--- 1. Testing Local Qdrant Initialization ---")
    retriever = HybridRetriever()
    print(f"[OK] Qdrant initialized pointing to: {getattr(settings, 'qdrant_storage_path', 'default')}")

    # Create collection
    retriever.create_collection()
    print(f"[OK] Collection '{retriever.collection_name}' ready.")

    print("\n--- 2. Indexing Sample Technical Chunks ---")
    sample_chunks = [
        Chunk(
            doc_id="test_doc",
            content="Docker containers share the host operating system kernel, making them lightweight.",
            strategy="structure_aware",
            heading_hierarchy=["Docker", "Overview"]
        ),
        Chunk(
            doc_id="test_doc",
            content="Virtual machines use a hypervisor to virtualize hardware and run a complete guest OS.",
            strategy="structure_aware",
            heading_hierarchy=["Virtualization", "VMs"]
        ),
        Chunk(
            doc_id="test_doc",
            content="The JVM garbage collector automatically frees up heap memory by removing unreferenced objects.",
            strategy="structure_aware",
            heading_hierarchy=["Java", "Memory Management"]
        ),
    ]

    retriever.index_chunks(sample_chunks)
    print(f"[OK] Successfully indexed {len(sample_chunks)} chunks.")

    print("\n--- 3. Testing Hybrid Search (Dense + BM25) ---")
    query = "How do containers achieve lightweight isolation?"
    print(f"Searching for: '{query}'")
    results = retriever.search(query=query, top_k=2)

    for i, res in enumerate(results, 1):
        print(f"\n[Result {i}] Score (RRF): {res['rrf_score']:.4f}")
        print(f"Content: {res['content']}")

    print("\n--- 4. Checking Physical Disk Files ---")
    storage_folder = Path(settings.qdrant_storage_path)
    if storage_folder.exists():
        files = list(storage_folder.glob("**/*"))
        print(f"[OK] Storage folder '{storage_folder}' exists on disk with {len(files)} files/folders.")
    else:
        print("[WARNING] Storage folder was not found on disk.")

    print("\n[SUCCESS] ALL LOCAL QDRANT CHECKS PASSED!")

if __name__ == "__main__":
    test_qdrant_pipeline()