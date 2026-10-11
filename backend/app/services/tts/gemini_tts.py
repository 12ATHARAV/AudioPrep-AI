"""
Gemini Multi-Speaker TTS Service.

This uses Google's Gemini 2.5 Flash TTS (or 3.8 Flash lite TTS)
to generate natural two-speaker podcast audio in a SINGLE API call.

Key Features:
- Multi-speaker: HOST and EXPERT get different voices
- speech_metadata: Controls style/emotion per turn
- Inline vocal bursts: <laugh>, <sigh>, <short pause>
- SynthID watermark: Automatically embedded

Available Voices (pick contrasting ones for podcast):
- Puck: Upbeat, engaging (good for HOST)
- Charon: Deep, authoritative (good for EXPERT)
- Kore: Professional, composed
- Aoede: Smooth, relaxed
- Fenrir: Energetic, intense
- Zephyr: Bright, crisp

Interview Note: "Gemini's TTS model natively supports
two-speaker conversation with per-turn style control.
The multi_speaker_voice_config binds speaker identities
to voices, and speech_metadata on each part controls
prosody. This is fundamentally different from stitching
separate TTS calls together — the model understands
conversational flow."

Free Tier: ~5-15 RPM, ~250-1500 RPD
Paid: $0.50/1M input tokens, $9.00/1M audio output tokens
"""

import os
from pathlib import Path
from google import genai
from google.genai import types

from app.agents.state import DialogueTurn
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("gemini_tts")


class GeminiTTSService:
    """
    Generates multi-speaker podcast audio using Gemini TTS.
    """

    def __init__(
        self,
        host_voice: str | None = None,
        expert_voice: str | None = None,
        model: str | None = None,
    ):
        self.host_voice = host_voice or getattr(settings, "tts_host_voice", "Puck")
        self.expert_voice = expert_voice or getattr(settings, "tts_expert_voice", "Kore")
        self.model = model or settings.gemini_tts_model
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = genai.Client(
                api_key=settings.gemini_api_key
            )
        return self._client

    async def generate_audio(
        self,
        dialogue: list[DialogueTurn],
        output_path: Path,
    ) -> Path:
        """
        Generate multi-speaker audio from dialogue turns.

        Args:
            dialogue: List of DialogueTurn (speaker + text + style)
            output_path: Where to save the WAV file

        Returns:
            Path to the generated audio file
        """
        logger.info(
            "gemini_tts_start",
            turns=len(dialogue),
            model=self.model,
        )

        # Step 1: Configure speakers
        multi_speaker_config = types.MultiSpeakerVoiceConfig(
            speaker_voice_configs=[
                types.SpeakerVoiceConfig(
                    speaker="HOST",
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=self.host_voice
                        )
                    ),
                ),
                types.SpeakerVoiceConfig(
                    speaker="EXPERT",
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=self.expert_voice
                        )
                    ),
                ),
            ]
        )

        # Step 2: Build script parts with speech_metadata
        script_parts = []
        for turn in dialogue:
            speaker = turn.speaker if isinstance(turn, DialogueTurn) else turn.get("speaker", "HOST")
            text = turn.text if isinstance(turn, DialogueTurn) else turn.get("text", "")
            style = turn.style if isinstance(turn, DialogueTurn) else turn.get("style", "")

            # Map styles to TTS-friendly descriptions
            style_map = {
                "curious": "curious and engaged",
                "explaining": "clear and authoritative",
                "surprised": "genuinely surprised",
                "amused": "amused and warm",
                "thoughtful": "thoughtful and measured",
                "excited": "enthusiastic and energetic",
                "": "natural and conversational",
            }

            tts_style = style_map.get(style, style or "natural and conversational")

            script_parts.append(
                types.Part(
                    text=text,
                    speech_metadata=types.SpeechMetadata(
                        speaker=speaker,
                        style=tts_style,
                    ),
                )
            )

        contents = [
            types.Content(role="user", parts=script_parts)
        ]

        # Step 3: Generate audio
        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    multi_speaker_voice_config=multi_speaker_config
                ),
            ),
        )

        # Step 4: Save audio
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        for part in response.candidates[0].content.parts:
            if part.inline_data:
                audio_data = part.inline_data.data
                mime_type = part.inline_data.mime_type

                with open(output_path, "wb") as f:
                    f.write(audio_data)

                logger.info(
                    "gemini_tts_complete",
                    output=str(output_path),
                    size_mb=round(len(audio_data) / 1024 / 1024, 2),
                    mime=mime_type,
                )

                return output_path

        raise RuntimeError("No audio data in Gemini TTS response")

    async def generate_chunked_audio(
        self,
        dialogue: list[DialogueTurn],
        output_dir: Path,
        chunk_size: int = 20,
    ) -> list[Path]:
        """
        Generate audio in chunks for long dialogues.

        Gemini TTS has limits on how much text can be
        processed in one call. For long podcasts, we
        split into segments.

        Returns list of audio file paths.
        """
        chunks = []
        for i in range(0, len(dialogue), chunk_size):
            chunk = dialogue[i : i + chunk_size]
            chunk_path = output_dir / f"segment_{i // chunk_size:03d}.wav"

            try:
                await self.generate_audio(chunk, chunk_path)
                chunks.append(chunk_path)
            except Exception as e:
                logger.error(
                    "chunk_tts_failed",
                    segment=i // chunk_size,
                    error=str(e),
                )
                raise

        return chunks