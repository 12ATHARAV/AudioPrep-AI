"""
LangGraph state definition for the podcast generation pipeline.

This is THE central data structure. Everything flows through it.

Interview Note: "The state is a TypedDict that carries all
information between graph nodes. Each node reads what it needs,
does its work, and writes its results back. This is the
Single Responsibility Principle applied to pipeline stages."
"""

from typing import TypedDict, Optional, Annotated
from pydantic import BaseModel, Field
import operator


class TopicPlanSegment(BaseModel):
    """An individual act/segment within a topic plan."""
    title: str = ""
    concepts: list[str] = Field(default_factory=list)
    description: str = ""


class TopicPlan(BaseModel):
    """Output of the Topic Planner node."""
    topic: str
    difficulty: str = "intermediate"
    learning_goals: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    key_concepts: list[str] = Field(default_factory=list)
    interview_focus: list[str] = Field(default_factory=list)
    estimated_duration_minutes: int = 120
    coverage_mode: str = "comprehensive"  # "summary" or "comprehensive"
    segments: list[TopicPlanSegment] = Field(default_factory=list)


class DialogueTurn(BaseModel):
    """A single turn in the podcast dialogue."""
    speaker: str  # "HOST" or "EXPERT"
    text: str
    style: str = ""  # e.g., "curious", "explaining", "surprised"
    evidence_chunk_ids: list[str] = Field(default_factory=list)


class QualityCriticResult(BaseModel):
    """Output of the Quality Critic node."""
    groundedness: float = 0.0
    coverage: float = 0.0
    technical_accuracy: float = 0.0
    dialogue_naturalness: float = 0.0
    interview_value: float = 0.0
    issues: list[str] = Field(default_factory=list)
    revision_required: bool = False
    overall_score: float = 0.0


class PodcastState(TypedDict):
    """
    Complete state for the podcast generation pipeline.

    Each node in the LangGraph reads from and writes to this state.
    """
    # Input
    doc_id: str
    raw_text: str
    user_preferences: dict  # difficulty, length, focus

    # Chunking & Retrieval
    chunks: list  # List of Chunk objects
    retrieval_contexts: dict  # concept -> list of contexts

    # Planning
    topic_plan: Optional[TopicPlan]

    # Dialogue
    dialogue: list[DialogueTurn]
    dialogue_script: str  # Formatted script for TTS

    # Quality
    critic_result: Optional[QualityCriticResult]
    revision_count: int

    # Audio
    audio_path: Optional[str]

    # Status
    status: str
    error: Optional[str]
    current_stage: str