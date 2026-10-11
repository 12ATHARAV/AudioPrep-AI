from evals.run_eval import create_eval_document
from app.services.chunking.manager import ChunkingManager
from evals.eval_dataset import DOCKER_EVAL_DATASET

doc = create_eval_document()

mgr_rec = ChunkingManager(apply_contextual=True)
chunks_rec = mgr_rec.chunk(doc, "recursive")

mgr_sa = ChunkingManager(apply_contextual=True)
chunks_sa = mgr_sa.chunk(doc, "structure_aware")

print(f"Recursive chunks count: {len(chunks_rec)}")
print(f"Structure-Aware chunks count: {len(chunks_sa)}")

print("\n--- Recursive Chunks Snippets ---")
for i, c in enumerate(chunks_rec):
    print(f"\n[Rec Chunk {i+1}]: len={len(c.content)}")
    print("Content:", c.content)
    print("Contextualized:", c.contextualized_content[:100] if c.contextualized_content else "None")

print("\n--- Structure-Aware Chunks Snippets ---")
for i, c in enumerate(chunks_sa):
    print(f"\n[SA Chunk {i+1}]: len={len(c.content)}")
    print("Content:", c.content)
    print("Contextualized:", c.contextualized_content[:100] if c.contextualized_content else "None")
