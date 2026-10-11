"""
PostgreSQL-backed Job Store with in-memory caching.
Persists job state, progress, and S3 keys/URLs to Postgres.
"""

import asyncio
from typing import Optional
from datetime import datetime
from sqlalchemy import update
from app.models.job import Job, JobStatus
from app.models.db_models import JobEntity
from app.core.database import AsyncSessionLocal
from app.core.logging import get_logger

logger = get_logger("job_store")


class JobStore:
    """Hybrid Job Store: In-memory cache + PostgreSQL persistence."""

    def __init__(self):
        self._jobs: dict[str, Job] = {}

    def _schedule_db_task(self, coro):
        """Schedule database task in the active asyncio loop without blocking."""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(coro)
        except RuntimeError:
            pass

    async def _save_job_to_db(self, job: Job):
        try:
            async with AsyncSessionLocal() as session:
                entity = JobEntity(
                    job_id=job.job_id,
                    status=job.status.value if hasattr(job.status, "value") else str(job.status),
                    progress=job.progress,
                    current_stage=job.current_stage,
                    filename=job.filename,
                    pdf_s3_key=job.pdf_s3_key,
                    pdf_s3_url=job.pdf_s3_url,
                    media_s3_key=job.media_s3_key,
                    media_s3_url=job.media_s3_url,
                    parameters=job.request.model_dump() if job.request else {},
                    created_at=job.created_at,
                    error=job.error,
                )
                session.add(entity)
                await session.commit()
                logger.info("job_persisted_to_db", job_id=job.job_id)
        except Exception as e:
            logger.warning("db_persist_warning", job_id=job.job_id, error=str(e))

    async def _update_job_in_db(self, job_id: str, update_dict: dict):
        try:
            async with AsyncSessionLocal() as session:
                query = update(JobEntity).where(JobEntity.job_id == job_id).values(**update_dict)
                await session.execute(query)
                await session.commit()
                logger.info("job_updated_in_db", job_id=job_id)
        except Exception as e:
            logger.warning("db_update_warning", job_id=job_id, error=str(e))

    def create(self, job: Job) -> Job:
        self._jobs[job.job_id] = job
        logger.info("job_created", job_id=job.job_id)
        self._schedule_db_task(self._save_job_to_db(job))
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def update_status(
        self,
        job_id: str,
        status: JobStatus,
        progress: int = 0,
        current_stage: str = "",
        error: str = None,
        audio_path: str = None,
        pdf_s3_key: str = None,
        pdf_s3_url: str = None,
        media_s3_key: str = None,
        media_s3_url: str = None,
    ) -> Optional[Job]:
        job = self._jobs.get(job_id)
        if not job:
            return None

        job.status = status
        job.progress = progress
        job.current_stage = current_stage

        db_updates = {
            "status": status.value if hasattr(status, "value") else str(status),
            "progress": progress,
            "current_stage": current_stage,
        }

        if error:
            job.error = error
            db_updates["error"] = error
        if audio_path:
            job.audio_path = audio_path
        if pdf_s3_key:
            job.pdf_s3_key = pdf_s3_key
            db_updates["pdf_s3_key"] = pdf_s3_key
        if pdf_s3_url:
            job.pdf_s3_url = pdf_s3_url
            db_updates["pdf_s3_url"] = pdf_s3_url
        if media_s3_key:
            job.media_s3_key = media_s3_key
            db_updates["media_s3_key"] = media_s3_key
        if media_s3_url:
            job.media_s3_url = media_s3_url
            db_updates["media_s3_url"] = media_s3_url
        if status == JobStatus.COMPLETED:
            now = datetime.utcnow()
            job.completed_at = now
            db_updates["completed_at"] = now

        logger.info(
            "job_updated",
            job_id=job_id,
            status=status,
            progress=progress,
        )

        self._schedule_db_task(self._update_job_in_db(job_id, db_updates))
        return job

    def list_all(self) -> list[Job]:
        return list(self._jobs.values())


# Singleton
job_store = JobStore()