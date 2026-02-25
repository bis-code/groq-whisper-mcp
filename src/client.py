"""
Whisper transcription client with configurable provider (OpenAI or Groq).

Provides word-level timestamps for subtitle generation and word-boundary snapping.
Supports OpenAI Whisper (default) and Groq-hosted Whisper via the OpenAI SDK.

Pricing (per hour of audio):
  OpenAI:
  - whisper-1:                $0.36
  Groq:
  - whisper-large-v3-turbo:   $0.04  (cheapest)
  - whisper-large-v3:         $0.111
  - distil-whisper-large-v3-en: $0.02
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

MAX_AUDIO_SIZE_BYTES = 25 * 1024 * 1024  # 25MB limit (both OpenAI and Groq)

PROVIDER_CONFIG = {
    "openai": {
        "env_key": "OPENAI_API_KEY",
        "base_url": None,  # Use OpenAI default
        "default_model": "whisper-1",
    },
    "groq": {
        "env_key": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "whisper-large-v3-turbo",
    },
}


@dataclass
class TranscriptionResult:
    """Result from Whisper transcription"""
    text: str
    words: list[dict]  # [{"word": str, "start": float, "end": float}]
    duration: float


def detect_provider() -> str:
    """Auto-detect provider from available environment variables.

    Priority: OPENAI_API_KEY > GROQ_API_KEY.
    Explicit WHISPER_PROVIDER env var overrides auto-detection.
    """
    explicit = os.environ.get("WHISPER_PROVIDER", "").lower()
    if explicit in PROVIDER_CONFIG:
        return explicit

    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("GROQ_API_KEY"):
        return "groq"

    return "openai"  # Default, will fail with clear error if no key


class WhisperClient:
    """Whisper transcription via OpenAI or Groq API"""

    def __init__(self, api_key: str | None = None, provider: str | None = None):
        self.provider = provider or detect_provider()

        if self.provider not in PROVIDER_CONFIG:
            raise ValueError(
                f"Unknown provider: {self.provider}. Use 'openai' or 'groq'."
            )

        config = PROVIDER_CONFIG[self.provider]
        api_key = api_key or os.environ.get(config["env_key"])

        if not api_key:
            raise ValueError(
                f"{config['env_key']} not found. "
                f"Set {config['env_key']} environment variable or pass api_key parameter."
            )

        client_kwargs = {"api_key": api_key}
        if config["base_url"]:
            client_kwargs["base_url"] = config["base_url"]

        self.client = OpenAI(**client_kwargs)
        self.default_model = config["default_model"]

    def transcribe_video(
        self,
        video_path: str,
        model: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe video and return word-level timestamps.

        Args:
            video_path: Path to video file
            model: Whisper model to use (default: provider-specific)

        Returns:
            TranscriptionResult with text, word timestamps, and duration
        """
        model = model or self.default_model
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
