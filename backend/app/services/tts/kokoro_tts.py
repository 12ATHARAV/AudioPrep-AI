"""
Kokoro TTS — local open-source fallback for speech synthesis.

Kokoro-82M is an 82-million parameter TTS model under Apache 2.0.
It runs entirely locally with ~500MB-1.2GB RAM usage.

Why as fallback?
- Gemini TTS has free tier limits (5-15 RPM)
- If quota is exhausted, Kokoro generates locally
- No API key needed, no network dependency
- Quality is very good for its size

Available voices (54+ across 9 languages):
- af_heart: Flagship female (warm, expressive) — Grade A
- am_adam: Authoritative male (deep) — good for EXPERT
- af_bella: Warm, friendly female — good for HOST
- am_michael: Conversational male

Prerequisites:
  pip install "kokoro>=0.9.4" soundfile
  # Windows: winget install eSpeak-NG.eSpeak-NG
  # Linux: sudo apt install espeak-ng

Interview Note: "Kokoro is our offline TTS fallback. It uses
only 82M parameters and runs on CPU with <1.2GB RAM, making
it viable on my 8GB machine. The model uses StyleTTS2 with
ISTFTNet vocoder — no diffusion, which is why it's so fast.
It generates 4x faster than real-time on a modern CPU."
"""

import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Optional

from app.agents.state import DialogueTurn
from app.core.logging import get_logger

logger = get_logger("kokoro_tts")


class KokoroTTSService:
    """
    Local TTS fallback using Kokoro-82M.
    Generates separate audio for HOST and EXPERT,
    then concatenates them.
    """

    def __init__(
        self,
        host_voice: str = "am_adam",
        expert_voice: str = "af_heart",
        speed: float = 1.0,
    ):
        self.host_voice = host_voice
        self.expert_voice = expert_voice
        self.speed = speed
        self._pipeline = None
        self.sample_rate = 24000

    @property
    def pipeline(self):
        """Lazy-load Kokoro pipeline."""
        if self._pipeline is None:
            from kokoro import KPipeline
            logger.info("loading_kokoro_pipeline")
            self._pipeline = KPipeline(lang_code="a")  # American English
        return self._pipeline

    async def generate_audio(
        self,
        dialogue: list[DialogueTurn],
        output_path: Path,
        pause_between_turns: float = 0.5,
    ) -> Path:
        """
        Generate multi-speaker audio from dialogue turns.

        Each turn is synthesized with the appropriate voice,
        then concatenated with pauses between turns.
        """
        logger.info("kokoro_tts_start", turns=len(dialogue))

        all_audio = []
        pause_samples = int(pause_between_turns * self.sample_rate)
        pause = np.zeros(pause_samples, dtype=np.float32)

        for i, turn in enumerate(dialogue):
            speaker = turn.speaker if isinstance(turn, DialogueTurn) else turn.get("speaker", "HOST")
            text = turn.text if isinstance(turn, DialogueTurn) else turn.get("text", "")

            # Select voice based on speaker
            voice = (
                self.host_voice
                if speaker == "HOST"
                else self.expert_voice
            )

            # Clean text for TTS
            clean_text = self._clean_for_tts(text)

            if not clean_text.strip():
                continue

            # Generate audio for this turn
            try:
                turn_audio = self._synthesize(clean_text, voice)
                all_audio.append(turn_audio)
                all_audio.append(pause)

                if (i + 1) % 10 == 0:
                    logger.info(
                        "kokoro_progress",
                        turns_complete=i + 1,
                        total=len(dialogue),
                    )
            except Exception as e:
                logger.warning(
                    "kokoro_turn_failed",
                    turn=i,
                    error=str(e),
                )
                continue

        if not all_audio:
            raise RuntimeError("No audio generated")

        # Concatenate all audio
        combined = np.concatenate(all_audio)

        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), combined, self.sample_rate)

        duration_seconds = len(combined) / self.sample_rate
        logger.info(
            "kokoro_tts_complete",
            output=str(output_path),
            duration_sec=round(duration_seconds, 1),
            size_mb=round(output_path.stat().st_size / 1024 / 1024, 2),
        )

        return output_path

    def _synthesize(self, text: str, voice: str) -> np.ndarray:
        """Synthesize a single text with a specific voice."""
        generator = self.pipeline(
            text, voice=voice, speed=self.speed
        )

        audio_chunks = []
        for gs, ps, audio in generator:
            audio_chunks.append(audio)

        if not audio_chunks:
            return np.zeros(self.sample_rate, dtype=np.float32)

        return np.concatenate(audio_chunks)

    def _clean_for_tts(self, text: str) -> str:
        """
        Clean text for TTS processing.
        Remove markup tags that Kokoro doesn't understand.
        """
        import re

        # Remove tags like <laugh>, <sigh>, <short pause>
        text = re.sub(r"<[^>]+>", "", text)

        # Remove pipe notation |word|
        text = re.sub(r"\|([^|]+)\|", r"\1", text)

        # Clean up extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text