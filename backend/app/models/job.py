"""
Job model for asynchronous podcast generation.

Interview Note: "We don't make the user wait 5 minutes for
a response. POST /podcasts returns a job_id immediately.
The user polls GET /podcasts/{job_id} for status updates.
This is the standard async job pattern used by Stripe,
Twilio, and every production API."

The job progresses through stages:
  QUEUED → INGESTING → CHUNKING → RETRIEVING →
  PLANNING → GENERATING → CRITIQUING → TTS → PROCESSING →
  COMPLETED
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from enum import Enum
import uuid


class JobStatus(str, Enum):
    QUEUED = "QUEUED"
    INGESTING = "INGESTING"
    CHUNKING = "CHUNKING"
    RETRIEVING = "RETRIEVING"
    PLANNING = "PLANNING"
    GENERATING_DIALOGUE = "GENERATING_DIALOGUE"
    CRITIQUING = "CRITIQUING"
    REVISING = "REVISING"
    GENERATING_AUDIO = "GENERATING_AUDIO"
    PROCESSING_AUDIO = "PROCESSING_AUDIO"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class JobRequest(BaseModel):
    """User's request to generate a podcast."""
    difficulty: str = "intermediate"  # beginner, intermediate, advanced
    podcast_length: int = 120  # minutes (default 2 hours)
    focus: str = "general"  # general, interview, deep-dive
    chunking_strategy: str = "structure_aware"
    tts_provider: str = "auto"  # auto, gemini, kokoro


class JobResponse(BaseModel):
    """Response when creating a new job."""
    job_id: str
    status: JobStatus
    message: str


class JobStatusResponse(BaseModel):
    """Response when checking job status."""
    job_id: str
    status: JobStatus
    progress: int = 0  # 0-100
    current_stage: str = ""
    created_at: datetime
    completed_at: Optional[datetime] = None
    audio_url: Optional[str] = None
    pdf_s3_url: Optional[str] = None
    media_s3_url: Optional[str] = None
    error: Optional[str] = None
    metadata: dict = Field(default_factory=dict)


class Job(BaseModel):
    """Internal job representation."""
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: JobStatus = JobStatus.QUEUED
    progress: int = 0
    current_stage: str = "queued"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    filename: str = ""
    file_path: str = ""
    pdf_s3_key: Optional[str] = None
    pdf_s3_url: Optional[str] = None
    audio_path: Optional[str] = None
    media_s3_key: Optional[str] = None
    media_s3_url: Optional[str] = None
    error: Optional[str] = None
    request: Optional[JobRequest] = None
    metadata: dict = Field(default_factory=dict)