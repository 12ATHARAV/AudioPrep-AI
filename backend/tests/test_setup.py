"""Verify the project setup is working."""

import pytest
from pathlib import Path
from app.core.config import Settings


def test_settings_load():
    """Settings should load from .env without errors."""
    # This will fail if required fields are missing
    settings = Settings(
        gemini_api_key="test-key-123",
    )
    assert settings.gemini_api_key == "test-key-123"
    assert settings.embedding_dimension == 384
    assert settings.max_revision_loops == 3


def test_directories_configurable():
    """Paths should be configurable."""
    custom_path = Path("/custom/uploads")
    settings = Settings(
        gemini_api_key="test",
        upload_dir=custom_path,
    )
    assert settings.upload_dir == custom_path


def test_default_models():
    """Default model selections should be set."""
    settings = Settings(gemini_api_key="test")
    assert "gemini" in settings.gemini_model
    assert "openai" in settings.groq_model
    assert settings.tts_primary == "gemini"
    assert settings.tts_fallback == "kokoro"