from sqlalchemy import Column, String, Integer, DateTime, Text, JSON
from datetime import datetime
import uuid
from app.core.database import Base

class JobEntity(Base):
    __tablename__ = "jobs"

    job_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    status = Column(String(30), nullable=False, default="QUEUED", index=True)
    progress = Column(Integer, default=0)
    current_stage = Column(String(50), default="queued")
    
    # Original document info
    filename = Column(String(255), nullable=False)
    pdf_s3_key = Column(String(512), nullable=True)
    pdf_s3_url = Column(Text, nullable=True)
    
    # Output media info (Audio / Video)
    media_s3_key = Column(String(512), nullable=True)
    media_s3_url = Column(Text, nullable=True)
    media_type = Column(String(20), default="audio/mpeg") # or video/mp4
    
    # Job execution parameters & error handling
    parameters = Column(JSON, default=dict)
    error = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
