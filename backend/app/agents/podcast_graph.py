"""
Complete LangGraph pipeline for podcast generation.

This is the ORCHESTRATOR. It wires together all the services
into a stateful graph that processes a PDF into a podcast.

Graph:
  START → INGEST → CHUNK → INDEX → PLAN_TOPICS →
  RETRIEVE → GENERATE_DIALOGUE → CRITIC →
    ├─ PASS → TTS → PROCESS_AUDIO → END
    └─ FAIL → REVISE → CRITIC (loop)

Interview Note: "LangGraph gives us explicit control over
the pipeline flow. Each node is independently testable.
The conditional edge after CRITIC implements our quality
gate — if the dialogue doesn't meet our standards, we
loop back for revision up to 3 times."
"""

from langgraph.graph import StateGraph, START, END
from pathlib import Path

from app.agents.state import PodcastState
from app.agents.topic_planner import topic_planner_node
from app.agents.quality_critic import (
    quality_critic_node,
    should_revise,
)
from app.services.dialogue.generator import dialogue_generator_node
from app.services.ingestion.pdf_parser import PDFParser
from app.services.chunking.manager import ChunkingManager
from app.services.retrieval.hybrid_retriever import HybridRetriever
from app.services.retrieval.embeddings import EmbeddingService
from app.services.retrieval.reranker import RerankerService
from app.services.retrieval.rag_pipeline import RAGPipeline
from app.services.tts.router import TTSRouter
from app.services.audio.processor import AudioProcessor
from app.core.config import settings
from app.core.logging import get_logger

from qdrant_client import QdrantClient

logger = get_logger("podcast_graph")


# === Node Functions ===

async def ingest_node(state: PodcastState) -> dict:
    """Parse the uploaded PDF."""
    parser = PDFParser()
    doc = await parser.parse(Path(state["doc_id"]))
    # doc_id here is used as file_path for simplicity

    raw_text = doc.raw_markdown or "\n".join(
        b.content for b in doc.blocks
    )

    return {
        "raw_text": raw_text,
        "chunks": [],  # Will be filled by chunk_node
        "current_stage": "ingested",
        "status": "ingested",
    }


async def chunk_node(state: PodcastState) -> dict:
    """Chunk the document using the selected strategy."""
    parser = PDFParser()
    doc = await parser.parse(Path(state["doc_id"]))

    preferences = state.get("user_preferences", {})
    strategy = preferences.get("chunking_strategy", "structure_aware")

    manager = ChunkingManager()
    chunks = manager.chunk(doc, strategy_name=strategy)

    return {
        "chunks": chunks,
        "current_stage": "chunked",
    }


async def index_and_retrieve_node(state: PodcastState) -> dict:
    """Index chunks and retrieve relevant contexts."""
    chunks = state["chunks"]
    plan = state.get("topic_plan")

    if not chunks:
        return {
            "retrieval_contexts": {},
            "current_stage": "retrieval_skipped",
        }

    # Set up retrieval (in-memory for development)
    client = QdrantClient(":memory:")
    embedding_service = EmbeddingService()

    retriever = HybridRetriever(
        embedding_service=embedding_service,
        qdrant_client=client,
        collection_name="podcast_chunks",
    )

    retriever.create_collection()
    retriever.index_chunks(chunks)

    # Retrieve for each concept in the plan
    concepts = []
    if plan:
        from app.agents.state import TopicPlan
        if isinstance(plan, dict):
            plan = TopicPlan(**plan)
        concepts = plan.key_concepts + plan.interview_focus

    if not concepts:
        concepts = ["main topic"]

    # Use RAG pipeline with reranking
    reranker = RerankerService()
    rag = RAGPipeline(retriever=retriever, reranker=reranker)

    contexts = rag.retrieve_for_concepts(
        concepts[:16],
        queries_per_concept=1,
        top_k_per_concept=2,
    )

    return {
        "retrieval_contexts": contexts,
        "current_stage": "retrieved",
    }


async def tts_node(state: PodcastState) -> dict:
    """Generate audio from dialogue."""
    dialogue = state.get("dialogue", [])
    if not dialogue:
        return {
            "audio_path": None,
            "current_stage": "tts_skipped",
            "error": "No dialogue to synthesize",
        }

    from app.agents.state import DialogueTurn
    # Ensure dialogue items are DialogueTurn objects
    turns = []
    for t in dialogue:
        if isinstance(t, DialogueTurn):
            turns.append(t)
        elif isinstance(t, dict):
            turns.append(DialogueTurn(**t))

    router = TTSRouter()
    doc_stem = Path(state["doc_id"]).stem
    output_path = settings.audio_dir / f"{doc_stem}_podcast.wav"

    try:
        path, provider = await router.generate_audio(turns, output_path)
        return {
            "audio_path": str(path),
            "current_stage": "tts_complete",
            "status": f"audio_generated_via_{provider}",
        }
    except Exception as e:
        return {
            "audio_path": None,
            "current_stage": "tts_failed",
            "error": str(e),
        }


async def process_audio_node(state: PodcastState) -> dict:
    """Post-process the audio."""
    audio_path = state.get("audio_path")
    if not audio_path:
        return {"current_stage": "audio_processing_skipped"}

    processor = AudioProcessor()
    output = Path(audio_path).with_suffix(".mp3")

    try:
        final_path = await processor.process(
            Path(audio_path), output, format="mp3"
        )
        return {
            "audio_path": str(final_path),
            "current_stage": "completed",
            "status": "completed",
        }
    except Exception as e:
        # If processing fails, the raw WAV is still usable
        logger.warning("audio_processing_failed", error=str(e))
        return {
            "current_stage": "completed_raw",
            "status": "completed_without_processing",
        }


async def revise_dialogue_node(state: PodcastState) -> dict:
    """Revise dialogue based on critic feedback."""
    critic = state.get("critic_result")
    issues = []
    if critic:
        if hasattr(critic, 'issues'):
            issues = critic.issues
        elif isinstance(critic, dict):
            issues = critic.get('issues', [])

    logger.info("revising_dialogue", issues=issues)

    # Re-generate with critic feedback
    # In a full implementation, we'd pass the issues to the
    # dialogue generator as additional constraints
    return await dialogue_generator_node(state)


# === Build the Graph ===

def build_podcast_graph():
    """
    Build the complete LangGraph for podcast generation.

    Returns a compiled graph ready to invoke.
    """
    graph = StateGraph(PodcastState)

    # Add nodes
    graph.add_node("ingest", ingest_node)
    graph.add_node("plan_topics", topic_planner_node)
    graph.add_node("chunk", chunk_node)
    graph.add_node("retrieve", index_and_retrieve_node)
    graph.add_node("generate_dialogue", dialogue_generator_node)
    graph.add_node("critic", quality_critic_node)
    graph.add_node("revise_dialogue", revise_dialogue_node)
    graph.add_node("tts", tts_node)
    graph.add_node("process_audio", process_audio_node)

    # Add edges
    graph.add_edge(START, "ingest")
    graph.add_edge("ingest", "chunk")
    graph.add_edge("chunk", "plan_topics")
    graph.add_edge("plan_topics", "retrieve")
    graph.add_edge("retrieve", "generate_dialogue")
    graph.add_edge("generate_dialogue", "critic")

    # Conditional: revise or proceed
    graph.add_conditional_edges(
        "critic",
        should_revise,
        {
            "revise_dialogue": "revise_dialogue",
            "proceed_to_tts": "tts",
        },
    )

    graph.add_edge("revise_dialogue", "critic")
    graph.add_edge("tts", "process_audio")
    graph.add_edge("process_audio", END)

    return graph.compile()