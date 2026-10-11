"""
RAG Evaluation & Chunking Comparison Verification Test.

Can be run via:
    pytest tests/test_rag_evaluation.py -s
or as a script:
    python -m evals.run_eval
"""

import pytest
from evals.run_eval import run_benchmark


@pytest.mark.asyncio
async def test_rag_evaluation_and_proof():
    """Runs the benchmark across all chunking strategies and prints the forensic proof."""
    await run_benchmark()
