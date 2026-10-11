"""
Application configuration using Pydantic Settings.

Interview Note: Using pydantic-settings gives us:
1. Automatic .env file loading
2. Type validation on startup (fail fast)
3. Environment variable override (12-factor app)
4. Clear documentation of all config values
"""

from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional


class Settings(BaseSettings):
    """Central configuration for AudioPrep AI."""

    # --- LLM Providers ---
    gemini_api_key: str
    groq_api_key: Optional[str] = None
    nvidia_api_key: Optional[str] = None

    # --- Vector DB ---
    qdrant_storage_path: Optional[str] = "../data/qdrant_storage"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "audioprep_chunks"

    # --- Redis ---
    redis_url: str = "redis://localhost:6379/0"

    # --- Paths ---
    upload_dir: Path = Path("../data/uploads")
    processed_dir: Path = Path("../data/processed")
    audio_dir: Path = Path("../data/audio")

    # --- Upload Limits ---
    max_upload_size_mb: int = 50

    # --- Embedding Model ---
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # --- Chunking Defaults ---
    default_chunk_size: int = 500
    default_chunk_overlap: int = 100

    # --- LLM Models ---
    gemini_model: str = "gemini-3.8-flash"
    groq_model: str = "openai/gpt-oss-20b"
    nvidia_model: str = "deepseek-ai/deepseek-v4.1-flash"
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    llm_primary: str = "nvidia"

    # --- TTS ---
    tts_primary: str = "gemini"
    tts_fallback: str = "kokoro"
    gemini_tts_model: str = "gemini-3.8-flash-lite-tts"
    tts_host_voice: str = "Puck"       # Male, upbeat, engaging
    tts_expert_voice: str = "Kore"     # Female, professional, composed authoritative

    # --- Podcast Defaults ---
    default_podcast_length: int = 120  # minutes (2 hours default)
    max_podcast_length: int = 180  # minutes
    max_revision_loops: int = 1

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }

    # --- Database ---
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/audioprep_db"
    
    # --- AWS S3 ---
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_region: str = "ap-southeast-2"
    aws_s3_bucket: str = "audioprep-ai-storage"



# Singleton instance
settings = Settings()