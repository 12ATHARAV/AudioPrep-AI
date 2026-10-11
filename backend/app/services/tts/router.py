"""
TTS Router — automatically selects the best available TTS provider.

Architecture:
  TTS Router
      │
      ├──► Gemini TTS (primary — best quality, multi-speaker)
      │
      └──► Kokoro TTS (fallback — local, no API needed)

Interview Note: "The router implements the Circuit Breaker
pattern. If Gemini TTS fails (rate limit, network error),
we automatically fall back to Kokoro local TTS. The user
gets audio either way — the quality might differ slightly,
but the pipeline doesn't fail."
"""

from pathlib import Path
from app.agents.state import DialogueTurn
from app.services.tts.gemini_tts import GeminiTTSService
from app.services.tts.kokoro_tts import KokoroTTSService
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("tts_router")


class TTSRouter:
    """Routes TTS requests to available providers with fallback."""

    def __init__(self):
        self.gemini_tts = GeminiTTSService()
        self.kokoro_tts = KokoroTTSService()
        self.primary = settings.tts_primary

    async def generate_audio(
        self,
        dialogue: list[DialogueTurn],
        output_path: Path,
    ) -> tuple[Path, str]:
        """
        Generate audio using the best available TTS provider.

        Returns:
            Tuple of (audio_path, provider_name)
        """
        # Try primary first
        if self.primary == "gemini":
            try:
                if len(dialogue) > 25:
                    segment_dir = output_path.parent / f"{output_path.stem}_segments"
                    segments = await self.gemini_tts.generate_chunked_audio(
                        dialogue, segment_dir, chunk_size=20
                    )
                    from app.services.audio.processor import AudioProcessor
                    processor = AudioProcessor()
                    path = await processor.concatenate_segments(segments, output_path)
                    return path, "gemini"
                else:
                    path = await self.gemini_tts.generate_audio(
                        dialogue, output_path
                    )
                    return path, "gemini"
            except Exception as e:
                logger.warning(
                    "gemini_tts_failed_using_kokoro",
                    error=str(e),
                )

        # Fallback to Kokoro
        try:
            kokoro_path = output_path.with_suffix(".wav")
            path = await self.kokoro_tts.generate_audio(
                dialogue, kokoro_path
            )
            return path, "kokoro"
        except Exception as e:
            logger.error(
                "all_tts_providers_failed",
                error=str(e),
            )
            raise RuntimeError(
                f"All TTS providers failed. Last error: {e}"
            )