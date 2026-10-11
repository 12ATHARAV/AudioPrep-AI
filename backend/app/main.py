"""
FastAPI application — the HTTP API layer.

Endpoints:
  POST   /api/podcasts          Upload PDF and start generation
  GET    /api/podcasts/{job_id}  Check job status
  GET    /api/podcasts/{job_id}/audio  Download audio
  GET    /api/health             Health check
"""

import shutil
import asyncio
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from app.models.job import (
    Job,
    JobRequest,
    JobResponse,
    JobStatusResponse,
    JobStatus,
)
from app.core.job_store import job_store
from app.core.config import settings
from app.core.database import init_db
from app.services.storage.s3_service import s3_service
from app.core.logging import setup_logging, get_logger
from app.agents.podcast_graph import build_podcast_graph

# Setup
setup_logging()
logger = get_logger("main")

app = FastAPI(
    title="AudioPrep AI",
    description="Turn technical notes into interactive AI podcasts",
    version="1.0.0",
)

@app.on_event("startup")
async def startup_event():
    await init_db()

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# === Background Job Runner ===

async def run_podcast_pipeline(job_id: str, file_path: str, request: JobRequest):
    """Run the full podcast generation pipeline in the background."""
    try:
        # Update status through stages
        job_store.update_status(job_id, JobStatus.INGESTING, 10, "Parsing PDF")

        graph = build_podcast_graph()

        # Initial state
        initial_state = {
            "doc_id": file_path,
            "raw_text": "",
            "user_preferences": {
                "difficulty": request.difficulty,
                "duration": request.podcast_length,
                "focus": request.focus,
                "chunking_strategy": request.chunking_strategy,
            },
            "chunks": [],
            "retrieval_contexts": {},
            "topic_plan": None,
            "dialogue": [],
            "dialogue_script": "",
            "critic_result": None,
            "revision_count": 0,
            "audio_path": None,
            "status": "starting",
            "error": None,
            "current_stage": "starting",
        }

        # Run the graph
        result = await graph.ainvoke(initial_state)

        if result.get("audio_path"):
            audio_path = Path(result["audio_path"])
            media_s3_key = f"media/{job_id}/{audio_path.name}"
            media_s3_url = None
            try:
                s3_service.upload_file(audio_path, media_s3_key, content_type="audio/mpeg")
                media_s3_url = s3_service.generate_presigned_url(media_s3_key)
            except Exception as e:
                logger.warning("s3_audio_upload_failed", error=str(e), job_id=job_id)

            job_store.update_status(
                job_id,
                JobStatus.COMPLETED,
                100,
                "Complete",
                audio_path=result["audio_path"],
                media_s3_key=media_s3_key,
                media_s3_url=media_s3_url,
            )
        else:
            job_store.update_status(
                job_id,
                JobStatus.FAILED,
                0,
                "No audio generated",
                error=result.get("error", "Unknown error"),
            )

    except Exception as e:
        logger.error("pipeline_failed", job_id=job_id, error=str(e))
        job_store.update_status(
            job_id, JobStatus.FAILED, 0, "Failed", error=str(e)
        )


# === API Endpoints ===

@app.post("/api/podcasts", response_model=JobResponse)
async def create_podcast(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    difficulty: str = "intermediate",
    podcast_length: int = 120,
    focus: str = "general",
    chunking_strategy: str = "structure_aware",
):
    """
    Upload a PDF and start podcast generation.

    Returns a job_id for tracking progress.
    """
    # Validate file
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")

    # Save uploaded file locally
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Create job
    request = JobRequest(
        difficulty=difficulty,
        podcast_length=podcast_length,
        focus=focus,
        chunking_strategy=chunking_strategy,
    )

    job = Job(
        filename=file.filename,
        request=request,
    )

    file_path = upload_dir / f"{job.job_id}_{file.filename}"
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    job.file_path = str(file_path)

    # Upload PDF to S3
    pdf_s3_key = f"pdfs/{job.job_id}/{file.filename}"
    pdf_s3_url = None
    try:
        s3_service.upload_file(file_path, pdf_s3_key, content_type="application/pdf")
        pdf_s3_url = s3_service.generate_presigned_url(pdf_s3_key)
    except Exception as e:
        logger.warning("s3_pdf_upload_failed", error=str(e), job_id=job.job_id)

    job.pdf_s3_key = pdf_s3_key
    job.pdf_s3_url = pdf_s3_url

    # Persist job to store & Postgres
    job_store.create(job)

    # Start background processing
    background_tasks.add_task(
        run_podcast_pipeline,
        job.job_id,
        str(file_path),
        request,
    )

    return JobResponse(
        job_id=job.job_id,
        status=JobStatus.QUEUED,
        message=f"Podcast generation started for {file.filename}",
    )


@app.get("/api/podcasts/{job_id}", response_model=JobStatusResponse)
async def get_podcast_status(job_id: str):
    """Check the status of a podcast generation job."""
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    # If media S3 URL expired or was generated earlier, regenerate or fallback to local route
    audio_url = job.media_s3_url or (f"/api/podcasts/{job_id}/audio" if job.audio_path else None)

    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        progress=job.progress,
        current_stage=job.current_stage,
        created_at=job.created_at,
        completed_at=job.completed_at,
        audio_url=audio_url,
        pdf_s3_url=job.pdf_s3_url,
        media_s3_url=job.media_s3_url,
        error=job.error,
    )


@app.get("/api/podcasts/{job_id}/audio")
async def download_audio(job_id: str):
    """Download the generated podcast audio."""
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    if job.status != JobStatus.COMPLETED or not job.audio_path:
        raise HTTPException(400, "Audio not yet available")

    audio_path = Path(job.audio_path)
    if not audio_path.exists():
        raise HTTPException(404, "Audio file not found")

    return FileResponse(
        str(audio_path),
        media_type="audio/mpeg",
        filename=f"{job.filename.replace('.pdf', '')}_podcast.mp3",
    )


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "tts_primary": settings.tts_primary,
    }