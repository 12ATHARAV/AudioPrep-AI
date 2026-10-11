import pytest
from app.core.config import settings, Settings
from app.core.llm_provider import llm_provider, LLMProvider

def test_settings_nvidia_configured():
    """Verify NVIDIA settings load properly."""
    assert settings.nvidia_api_key is not None
    assert settings.nvidia_api_key.startswith("nvapi-")
    assert "https://integrate.api.nvidia.com" in settings.nvidia_base_url
    assert settings.llm_primary == "nvidia"

@pytest.mark.asyncio
async def test_nvidia_generate():
    """Verify live generation using NVIDIA NIM."""
    response = await llm_provider.generate(
        prompt='Respond with JSON: {"status": "ok"}',
        temperature=0.1,
        max_tokens=100,
    )
    assert response.provider == "nvidia"
    assert "ok" in response.text or "status" in response.text
