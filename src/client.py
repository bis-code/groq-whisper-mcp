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
import random
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from openai import APIError, APITimeoutError, RateLimitError, OpenAI

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
        # large-v3 (not -turbo): turbo drops/garbles words in noisy audio. The
        # quality gain matters more than turbo's speed/cost for our pipeline.
        "default_model": "whisper-large-v3",
    },
}

# Default vocabulary-priming prompt. Whisper uses `prompt` as a style/vocab hint,
# which sharply reduces missing/mis-heard domain jargon. Keep it short (<~200 tokens)
# and channel-relevant; override per-call via transcribe_video(prompt=...).
DEFAULT_PROMPT = (
    "Developer screencast about Claude Code, MCP servers, plugins, hooks, slash "
    "commands, rules, agents, the Anthropic SDK, Premiere Pro, and the cloud toolkit."
)
DEFAULT_LANGUAGE = "en"


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

        client_kwargs = {"api_key": api_key, "timeout": 120.0}
        if config["base_url"]:
            client_kwargs["base_url"] = config["base_url"]

        self.client = OpenAI(**client_kwargs)
        self.default_model = config["default_model"]

    def transcribe_video(
        self,
        video_path: str,
        model: str | None = None,
        max_retries: int = 3,
        language: str | None = DEFAULT_LANGUAGE,
        prompt: str | None = DEFAULT_PROMPT,
    ) -> TranscriptionResult:
        """Transcribe video and return word-level timestamps.

        Args:
            video_path: Path to video file
            model: Whisper model to use (default: provider-specific)
            max_retries: Max retry attempts for transient API failures
            language: ISO-639-1 code (default 'en'); skips language detection and
                cuts mis-heard words. Pass None to let Whisper auto-detect.
            prompt: vocabulary/style priming hint (default: channel jargon). Pass
                None or "" to disable.

        Returns:
            TranscriptionResult with text, word timestamps, and duration
        """
        model = model or self.default_model
        audio_path = self._extract_audio(video_path)
        try:
            return self._transcribe_with_retry(audio_path, model, max_retries, language, prompt)
        finally:
            os.unlink(audio_path)

    def _transcribe_with_retry(
        self,
        audio_path: str,
        model: str,
        max_retries: int,
        language: str | None = DEFAULT_LANGUAGE,
        prompt: str | None = DEFAULT_PROMPT,
    ) -> TranscriptionResult:
        """Call Whisper API with exponential backoff on transient failures."""
        last_error = None
        # Only send optional params when set, so callers can opt out cleanly.
        extra = {}
        if language:
            extra["language"] = language
        if prompt:
            extra["prompt"] = prompt

        for attempt in range(max_retries + 1):
            try:
                with open(audio_path, "rb") as audio_file:
                    response = self.client.audio.transcriptions.create(
                        model=model,
                        file=audio_file,
                        response_format="verbose_json",
                        timestamp_granularities=["word"],
                        **extra,
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

            except (RateLimitError, APITimeoutError) as e:
                last_error = e
                if attempt < max_retries:
                    delay = _backoff_delay(attempt, e)
                    logger.warning(
                        "Retryable error (attempt %d/%d): %s. Retrying in %.1fs",
                        attempt + 1, max_retries, type(e).__name__, delay,
                    )
                    time.sleep(delay)

            except APIError as e:
                if e.status_code and e.status_code >= 500:
                    last_error = e
                    if attempt < max_retries:
                        delay = _backoff_delay(attempt, e)
                        logger.warning(
                            "Server error %d (attempt %d/%d). Retrying in %.1fs",
                            e.status_code, attempt + 1, max_retries, delay,
                        )
                        time.sleep(delay)
                    continue
                raise  # 4xx (except 429) are not retryable

        raise last_error  # type: ignore[misc]

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


def _backoff_delay(attempt: int, error: Exception | None = None) -> float:
    """Calculate exponential backoff delay with jitter.

    Respects Retry-After header if present on RateLimitError.
    """
    if isinstance(error, RateLimitError) and hasattr(error, 'response'):
        retry_after = None
        if error.response is not None:
            retry_after = error.response.headers.get("retry-after")
        if retry_after:
            try:
                return float(retry_after)
            except (ValueError, TypeError):
                pass

    base_delay = 2 ** attempt  # 1s, 2s, 4s
    jitter = random.uniform(0, base_delay * 0.5)
    return base_delay + jitter


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
