"""
Whisper transcription client using Groq API (OpenAI-compatible).

Provides word-level timestamps for subtitle generation and word-boundary snapping.
Uses Groq's hosted Whisper models via the OpenAI SDK.

Pricing (per hour of audio):
  - whisper-large-v3-turbo: $0.04  (default, recommended)
  - whisper-large-v3:       $0.111
  - distil-whisper:         $0.02  (cheapest, lower quality)
  - OpenAI Whisper:         $0.36  (9x more expensive)
"""

import json
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

logger = logging.getLogger(__name__)

MAX_AUDIO_SIZE_BYTES = 25 * 1024 * 1024  # 25MB Groq limit


@dataclass
class TranscriptionResult:
    """Result from Whisper transcription"""
    text: str
    words: list[dict]  # [{"word": str, "start": float, "end": float}]
    duration: float


class WhisperClient:
    """Groq-hosted Whisper transcription via OpenAI-compatible API"""

    def __init__(self, api_key: str | None = None):
        api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found. Set GROQ_API_KEY environment variable or pass api_key parameter."
            )

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )

    def transcribe_video(
        self,
        video_path: str,
        model: str = "whisper-large-v3-turbo",
    ) -> TranscriptionResult:
        """Transcribe video and return word-level timestamps.

        Args:
            video_path: Path to video file
            model: Whisper model to use (default: whisper-large-v3-turbo)

        Returns:
            TranscriptionResult with text, word timestamps, and duration
        """
        audio_path = self._extract_audio(video_path)
        try:
            with open(audio_path, "rb") as audio_file:
                response = self.client.audio.transcriptions.create(
                    model=model,
                    file=audio_file,
                    response_format="verbose_json",
                    timestamp_granularities=["word"],
                )

            words = [
                {"word": w.word, "start": w.start, "end": w.end}
                for w in (response.words or [])
            ]

            return TranscriptionResult(
                text=response.text,
                words=words,
                duration=response.duration or 0.0,
            )
        finally:
            os.unlink(audio_path)

    def _extract_audio(self, video_path: str) -> str:
        """Extract audio from video as MP3.

        Tries 128kbps first, falls back to 64kbps if file exceeds 25MB.
        """
        for bitrate in ("128k", "64k"):
            temp_path = tempfile.mktemp(suffix=".mp3")
            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-vn",
                "-acodec", "libmp3lame",
                "-b:a", bitrate,
                temp_path,
            ]

            result = subprocess.run(cmd, capture_output=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg audio extraction failed: {result.stderr.decode()[:500]}"
                )

            file_size = os.path.getsize(temp_path)
            if file_size <= MAX_AUDIO_SIZE_BYTES:
                logger.info(
                    "Audio extracted at %s: %.1fMB",
                    bitrate, file_size / (1024 * 1024),
                )
                return temp_path

            # Too large, clean up and try lower bitrate
            os.unlink(temp_path)
            logger.info(
                "Audio at %s is %.1fMB (>25MB), trying lower bitrate",
                bitrate, file_size / (1024 * 1024),
            )

        raise RuntimeError(
            f"Audio file exceeds 25MB even at 64kbps. "
            f"Video may be too long for direct transcription."
        )


def load_cached_transcription(cache_path: str) -> list[dict] | None:
    """Load cached word timestamps from JSON file.

    Returns None if file doesn't exist, is invalid, or not a list.
    """
    try:
        with open(cache_path) as f:
            data = json.load(f)
        if not isinstance(data, list):
            return None
        return data
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def save_transcription_cache(words: list[dict], cache_path: str) -> None:
    """Save word timestamps to JSON cache file."""
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(words, f, indent=2)
