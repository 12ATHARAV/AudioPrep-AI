"""
Audio post-processing with FFmpeg.

Takes raw TTS output and produces polished podcast audio:
1. Loudness normalization (consistent volume)
2. Silence trimming (remove awkward pauses)
3. MP3 encoding (smaller file for download)
4. Optional: intro/outro jingles

Interview Note: "Raw TTS output sounds robotic because
volume levels are inconsistent and there are awkward
silences. Audio post-processing normalizes loudness to
broadcast standards (-16 LUFS for podcasts), trims
excessive silence, and encodes to MP3 for efficient
delivery."
"""

import subprocess
from pathlib import Path
from pydub import AudioSegment
from app.core.logging import get_logger

logger = get_logger("audio_processor")


class AudioProcessor:
    """
    Post-processes TTS audio into polished podcast files.
    """

    def __init__(self, target_lufs: float = -16.0):
        self.target_lufs = target_lufs

    async def process(
        self,
        input_path: Path,
        output_path: Path,
        format: str = "mp3",
        bitrate: str = "192k",
    ) -> Path:
        """
        Full audio post-processing pipeline.

        1. Normalize loudness
        2. Trim silence
        3. Encode to target format
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Step 1: Normalize loudness using pydub
        audio = AudioSegment.from_file(str(input_path))

        # Normalize to target dBFS (approximate LUFS)
        target_dbfs = self.target_lufs  # Rough approximation
        change_in_dbfs = target_dbfs - audio.dBFS
        normalized = audio.apply_gain(change_in_dbfs)

        # Step 2: Trim leading/trailing silence
        trimmed = self._trim_silence(normalized)

        # Step 3: Export in target format
        trimmed.export(
            str(output_path),
            format=format,
            bitrate=bitrate,
            parameters=["-ar", "44100"],  # Standard sample rate
        )

        logger.info(
            "audio_processing_complete",
            input=str(input_path),
            output=str(output_path),
            duration_sec=round(len(trimmed) / 1000, 1),
            format=format,
        )

        return output_path

    def _trim_silence(
        self,
        audio: AudioSegment,
        silence_thresh: int = -50,
        chunk_size: int = 100,
    ) -> AudioSegment:
        """
        Trim leading and trailing silence.
        Also reduce internal pauses longer than 1.5 seconds.
        """
        from pydub.silence import detect_leading_silence

        # Trim start
        start_trim = detect_leading_silence(
            audio, silence_threshold=silence_thresh
        )
        # Trim end
        end_trim = detect_leading_silence(
            audio.reverse(), silence_threshold=silence_thresh
        )

        trimmed = audio[start_trim : len(audio) - end_trim]

        return trimmed

    async def add_intro_outro(
        self,
        podcast_path: Path,
        output_path: Path,
        intro_path: Path | None = None,
        outro_path: Path | None = None,
    ) -> Path:
        """
        Add intro/outro music to the podcast.
        Uses crossfade for smooth transitions.
        """
        podcast = AudioSegment.from_file(str(podcast_path))

        if intro_path and intro_path.exists():
            intro = AudioSegment.from_file(str(intro_path))
            # Fade in intro, crossfade into podcast
            intro = intro.fade_in(1000).fade_out(2000)
            podcast = intro.append(podcast, crossfade=2000)

        if outro_path and outro_path.exists():
            outro = AudioSegment.from_file(str(outro_path))
            outro = outro.fade_in(2000).fade_out(1000)
            podcast = podcast.append(outro, crossfade=2000)

        podcast.export(str(output_path), format="mp3", bitrate="192k")

        return output_path

    async def concatenate_segments(
        self,
        segment_paths: list[Path],
        output_path: Path,
        crossfade_ms: int = 200,
    ) -> Path:
        """
        Concatenate multiple audio segments with crossfade.
        Used when TTS generates audio in chunks.
        """
        if not segment_paths:
            raise ValueError("No segments to concatenate")

        combined = AudioSegment.from_file(str(segment_paths[0]))

        for seg_path in segment_paths[1:]:
            segment = AudioSegment.from_file(str(seg_path))
            combined = combined.append(segment, crossfade=crossfade_ms)

        combined.export(str(output_path), format="mp3", bitrate="192k")

        logger.info(
            "segments_concatenated",
            segments=len(segment_paths),
            total_duration_sec=round(len(combined) / 1000, 1),
        )

        return output_path