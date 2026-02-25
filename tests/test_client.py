"""Tests for Whisper transcription client (OpenAI/Groq via OpenAI SDK)"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest
from openai import RateLimitError, APITimeoutError, APIError

from client import WhisperClient, TranscriptionResult, load_cached_transcription, save_transcription_cache, detect_provider, _backoff_delay


# ============================================================
# Provider Detection
# ============================================================


class TestDetectProvider:
    """Tests for provider auto-detection from environment"""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-123"}, clear=True)
    def test_openai_key_selects_openai(self):
        assert detect_provider() == "openai"

    @patch.dict(os.environ, {"GROQ_API_KEY": "gsk-123"}, clear=True)
    def test_groq_key_selects_groq(self):
        assert detect_provider() == "groq"

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-123", "GROQ_API_KEY": "gsk-123"}, clear=True)
    def test_both_keys_prefers_openai(self):
        assert detect_provider() == "openai"

    @patch.dict(os.environ, {"WHISPER_PROVIDER": "groq", "OPENAI_API_KEY": "sk-123"}, clear=True)
    def test_explicit_provider_overrides_autodetect(self):
        assert detect_provider() == "groq"

    @patch.dict(os.environ, {"WHISPER_PROVIDER": "openai", "GROQ_API_KEY": "gsk-123"}, clear=True)
    def test_explicit_openai_with_groq_key(self):
        assert detect_provider() == "openai"

    @patch.dict(os.environ, {}, clear=True)
    def test_no_keys_defaults_to_openai(self):
        assert detect_provider() == "openai"

    @patch.dict(os.environ, {"WHISPER_PROVIDER": "invalid"}, clear=True)
    def test_invalid_provider_falls_through_to_autodetect(self):
        """Invalid explicit provider is ignored, falls to auto-detect"""
        assert detect_provider() == "openai"


# ============================================================
# WhisperClient Init
# ============================================================


class TestWhisperClientInit:
    """Tests for WhisperClient initialization"""

    @patch("client.OpenAI")
    def test_init_with_explicit_key_openai(self, mock_openai_cls):
        client = WhisperClient(api_key="test-key-123", provider="openai")
        mock_openai_cls.assert_called_once_with(api_key="test-key-123", timeout=120.0)
        assert client.provider == "openai"
        assert client.default_model == "whisper-1"

    @patch("client.OpenAI")
    def test_init_with_explicit_key_groq(self, mock_openai_cls):
        client = WhisperClient(api_key="test-key-123", provider="groq")
        mock_openai_cls.assert_called_once_with(
            api_key="test-key-123",
            timeout=120.0,
            base_url="https://api.groq.com/openai/v1",
        )
        assert client.provider == "groq"
        assert client.default_model == "whisper-large-v3-turbo"

    @patch.dict(os.environ, {"OPENAI_API_KEY": "env-key-456"}, clear=True)
    @patch("client.OpenAI")
    def test_init_from_env_openai(self, mock_openai_cls):
        client = WhisperClient()
        mock_openai_cls.assert_called_once_with(api_key="env-key-456", timeout=120.0)
        assert client.provider == "openai"

    @patch.dict(os.environ, {"GROQ_API_KEY": "env-key-789"}, clear=True)
    @patch("client.OpenAI")
    def test_init_from_env_groq(self, mock_openai_cls):
        client = WhisperClient()
        mock_openai_cls.assert_called_once_with(
            api_key="env-key-789",
            timeout=120.0,
            base_url="https://api.groq.com/openai/v1",
        )
        assert client.provider == "groq"

    @patch.dict(os.environ, {}, clear=True)
    @patch("client.OpenAI")
    def test_init_no_key_raises(self, mock_openai_cls):
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            WhisperClient()

    @patch.dict(os.environ, {"GROQ_API_KEY": "gsk-123", "WHISPER_PROVIDER": "openai"}, clear=True)
    @patch("client.OpenAI")
    def test_init_explicit_openai_but_only_groq_key_raises(self, mock_openai_cls):
        """Explicit openai provider with only GROQ_API_KEY should fail"""
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            WhisperClient()

    @patch("client.OpenAI")
    def test_init_unknown_provider_raises(self, mock_openai_cls):
        with pytest.raises(ValueError, match="Unknown provider"):
            WhisperClient(api_key="test", provider="deepgram")


# ============================================================
# Audio Extraction
# ============================================================


class TestAudioExtraction:
    """Tests for _extract_audio (video -> temp MP3)"""

    @patch("client.OpenAI")
    @patch("client.subprocess.run")
    def test_extract_128kbps_under_limit(self, mock_run, mock_openai_cls):
        """128kbps MP3 under 25MB passes through"""
        mock_run.return_value = MagicMock(returncode=0)
        client = WhisperClient(api_key="test", provider="openai")

        with patch("client.os.path.getsize", return_value=10 * 1024 * 1024):  # 10MB
            result = client._extract_audio("/fake/video.mp4")

        assert result.endswith(".mp3")
        assert mock_run.call_count == 1
        cmd = mock_run.call_args[0][0]
        assert "-b:a" in cmd
        idx = cmd.index("-b:a")
        assert cmd[idx + 1] == "128k"

    @patch("client.OpenAI")
    @patch("client.subprocess.run")
    def test_extract_fallback_64kbps(self, mock_run, mock_openai_cls):
        """Falls back to 64kbps when 128kbps exceeds 25MB"""
        mock_run.return_value = MagicMock(returncode=0)
        client = WhisperClient(api_key="test", provider="openai")

        call_count = 0

        def size_side_effect(path):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return 30 * 1024 * 1024  # 30MB (128kbps too large)
            return 15 * 1024 * 1024  # 15MB (64kbps OK)

        with patch("client.os.path.getsize", side_effect=size_side_effect), \
             patch("client.os.unlink"):
            result = client._extract_audio("/fake/video.mp4")

        assert result.endswith(".mp3")
        assert mock_run.call_count == 2
        second_cmd = mock_run.call_args_list[1][0][0]
        idx = second_cmd.index("-b:a")
        assert second_cmd[idx + 1] == "64k"

    @patch("client.OpenAI")
    @patch("client.subprocess.run")
    def test_extract_too_large_even_at_64k(self, mock_run, mock_openai_cls):
        """Raises when even 64kbps exceeds 25MB"""
        mock_run.return_value = MagicMock(returncode=0)
        client = WhisperClient(api_key="test", provider="openai")

        with patch("client.os.path.getsize", return_value=30 * 1024 * 1024), \
             patch("client.os.unlink"):
            with pytest.raises(RuntimeError, match="25MB"):
                client._extract_audio("/fake/video.mp4")

    @patch("client.OpenAI")
    @patch("client.subprocess.run")
    def test_extract_ffmpeg_failure(self, mock_run, mock_openai_cls):
        """Raises on ffmpeg failure"""
        mock_run.return_value = MagicMock(
            returncode=1,
            stderr=b"ffmpeg error: codec not found",
        )
        client = WhisperClient(api_key="test", provider="openai")

        with pytest.raises(RuntimeError, match="ffmpeg"):
            client._extract_audio("/fake/video.mp4")


# ============================================================
# Transcription
# ============================================================


class TestTranscription:
    """Tests for transcribe_video"""

    @patch("client.OpenAI")
    def test_transcribe_success(self, mock_openai_cls):
        """Successful transcription returns TranscriptionResult"""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_word = MagicMock()
        mock_word.word = "hello"
        mock_word.start = 0.0
        mock_word.end = 0.5

        mock_word2 = MagicMock()
        mock_word2.word = "world"
        mock_word2.start = 0.5
        mock_word2.end = 1.0

        mock_response = MagicMock()
        mock_response.text = "hello world"
        mock_response.words = [mock_word, mock_word2]
        mock_response.duration = 1.0
        mock_client.audio.transcriptions.create.return_value = mock_response

        client = WhisperClient(api_key="test", provider="openai")

        with patch.object(client, "_extract_audio", return_value="/tmp/audio.mp3"):
            with patch("builtins.open", mock_open(read_data=b"audio")):
                with patch("client.os.unlink"):
                    result = client.transcribe_video("/fake/video.mp4")

        assert isinstance(result, TranscriptionResult)
        assert result.text == "hello world"
        assert len(result.words) == 2
        assert result.words[0] == {"word": "hello", "start": 0.0, "end": 0.5}
        assert result.words[1] == {"word": "world", "start": 0.5, "end": 1.0}
        assert result.duration == 1.0

    @patch("client.OpenAI")
    def test_transcribe_uses_provider_default_model(self, mock_openai_cls):
        """Transcription uses provider-specific default model"""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.text = "test"
        mock_response.words = []
        mock_response.duration = 1.0
        mock_client.audio.transcriptions.create.return_value = mock_response

        client = WhisperClient(api_key="test", provider="openai")

        with patch.object(client, "_extract_audio", return_value="/tmp/audio.mp3"):
            with patch("builtins.open", mock_open(read_data=b"audio")):
                with patch("client.os.unlink"):
                    client.transcribe_video("/fake/video.mp4")

        call_kwargs = mock_client.audio.transcriptions.create.call_args
        assert call_kwargs.kwargs.get("model") == "whisper-1"

    @patch("client.OpenAI")
    def test_transcribe_groq_default_model(self, mock_openai_cls):
        """Groq provider uses whisper-large-v3-turbo by default"""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.text = "test"
        mock_response.words = []
        mock_response.duration = 1.0
        mock_client.audio.transcriptions.create.return_value = mock_response

        client = WhisperClient(api_key="test", provider="groq")

        with patch.object(client, "_extract_audio", return_value="/tmp/audio.mp3"):
            with patch("builtins.open", mock_open(read_data=b"audio")):
                with patch("client.os.unlink"):
                    client.transcribe_video("/fake/video.mp4")

        call_kwargs = mock_client.audio.transcriptions.create.call_args
        assert call_kwargs.kwargs.get("model") == "whisper-large-v3-turbo"

    @patch("client.OpenAI")
    def test_transcribe_custom_model(self, mock_openai_cls):
        """Can specify a different model"""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.text = "test"
        mock_response.words = []
        mock_response.duration = 1.0
        mock_client.audio.transcriptions.create.return_value = mock_response

        client = WhisperClient(api_key="test", provider="openai")

        with patch.object(client, "_extract_audio", return_value="/tmp/audio.mp3"):
            with patch("builtins.open", mock_open(read_data=b"audio")):
                with patch("client.os.unlink"):
                    client.transcribe_video("/fake/video.mp4", model="whisper-large-v3")

        call_kwargs = mock_client.audio.transcriptions.create.call_args
        assert call_kwargs.kwargs.get("model") == "whisper-large-v3"

    @patch("client.OpenAI")
    def test_transcribe_cleans_up_temp_audio(self, mock_openai_cls):
        """Temp audio file is deleted after transcription"""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.text = "test"
        mock_response.words = []
        mock_response.duration = 1.0
        mock_client.audio.transcriptions.create.return_value = mock_response

        client = WhisperClient(api_key="test", provider="openai")

        with patch.object(client, "_extract_audio", return_value="/tmp/audio_temp.mp3"):
            with patch("builtins.open", mock_open(read_data=b"audio")):
                with patch("client.os.unlink") as mock_unlink:
                    client.transcribe_video("/fake/video.mp4")

        mock_unlink.assert_called_once_with("/tmp/audio_temp.mp3")

    @patch("client.OpenAI")
    def test_transcribe_cleans_up_on_error(self, mock_openai_cls):
        """Temp audio file is deleted even when API call fails"""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.audio.transcriptions.create.side_effect = Exception("API error")

        client = WhisperClient(api_key="test", provider="openai")

        with patch.object(client, "_extract_audio", return_value="/tmp/audio_temp.mp3"):
            with patch("builtins.open", mock_open(read_data=b"audio")):
                with patch("client.os.unlink") as mock_unlink:
                    with pytest.raises(Exception, match="API error"):
                        client.transcribe_video("/fake/video.mp4")

        mock_unlink.assert_called_once_with("/tmp/audio_temp.mp3")


# ============================================================
# Retry Logic
# ============================================================


def _make_mock_response():
    """Helper to create a successful mock transcription response"""
    mock_word = MagicMock()
    mock_word.word = "hello"
    mock_word.start = 0.0
    mock_word.end = 0.5
    mock_response = MagicMock()
    mock_response.text = "hello"
    mock_response.words = [mock_word]
    mock_response.duration = 0.5
    return mock_response


def _make_rate_limit_error(retry_after=None):
    """Create a RateLimitError with optional Retry-After header"""
    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_response.headers = {}
    if retry_after is not None:
        mock_response.headers["retry-after"] = str(retry_after)
    mock_response.json.return_value = {"error": {"message": "Rate limit exceeded"}}
    return RateLimitError(
        message="Rate limit exceeded",
        response=mock_response,
        body={"error": {"message": "Rate limit exceeded"}},
    )


def _make_server_error(status_code=502):
    """Create an APIError with a 5xx status code"""
    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.headers = {}
    mock_response.json.return_value = {"error": {"message": "Bad gateway"}}
    return APIError(
        message="Bad gateway",
        request=MagicMock(),
        body={"error": {"message": "Bad gateway"}},
    )


class TestRetryLogic:
    """Tests for retry with exponential backoff"""

    @patch("client.time.sleep")
    @patch("client.OpenAI")
    def test_retry_on_rate_limit(self, mock_openai_cls, mock_sleep):
        """Retries on 429 RateLimitError and succeeds"""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_client.audio.transcriptions.create.side_effect = [
            _make_rate_limit_error(),
            _make_mock_response(),
        ]

        client = WhisperClient(api_key="test", provider="openai")

        with patch.object(client, "_extract_audio", return_value="/tmp/audio.mp3"):
            with patch("builtins.open", mock_open(read_data=b"audio")):
                with patch("client.os.unlink"):
                    result = client.transcribe_video("/fake/video.mp4", max_retries=3)

        assert result.text == "hello"
        assert mock_client.audio.transcriptions.create.call_count == 2
        assert mock_sleep.call_count == 1

    @patch("client.time.sleep")
    @patch("client.OpenAI")
    def test_retry_on_timeout(self, mock_openai_cls, mock_sleep):
        """Retries on APITimeoutError"""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_client.audio.transcriptions.create.side_effect = [
            APITimeoutError(request=MagicMock()),
            _make_mock_response(),
        ]

        client = WhisperClient(api_key="test", provider="openai")

        with patch.object(client, "_extract_audio", return_value="/tmp/audio.mp3"):
            with patch("builtins.open", mock_open(read_data=b"audio")):
                with patch("client.os.unlink"):
                    result = client.transcribe_video("/fake/video.mp4", max_retries=3)

        assert result.text == "hello"
        assert mock_client.audio.transcriptions.create.call_count == 2

    @patch("client.time.sleep")
    @patch("client.OpenAI")
    def test_retry_on_server_error(self, mock_openai_cls, mock_sleep):
        """Retries on 5xx server errors"""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        error = _make_server_error(502)
        error.status_code = 502

        mock_client.audio.transcriptions.create.side_effect = [
            error,
            _make_mock_response(),
        ]

        client = WhisperClient(api_key="test", provider="openai")

        with patch.object(client, "_extract_audio", return_value="/tmp/audio.mp3"):
            with patch("builtins.open", mock_open(read_data=b"audio")):
                with patch("client.os.unlink"):
                    result = client.transcribe_video("/fake/video.mp4", max_retries=3)

        assert result.text == "hello"

    @patch("client.time.sleep")
    @patch("client.OpenAI")
    def test_exhausted_retries_raises(self, mock_openai_cls, mock_sleep):
        """Raises after exhausting all retries"""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_client.audio.transcriptions.create.side_effect = _make_rate_limit_error()

        client = WhisperClient(api_key="test", provider="openai")

        with patch.object(client, "_extract_audio", return_value="/tmp/audio.mp3"):
            with patch("builtins.open", mock_open(read_data=b"audio")):
                with patch("client.os.unlink"):
                    with pytest.raises(RateLimitError):
                        client.transcribe_video("/fake/video.mp4", max_retries=2)

        # 1 initial + 2 retries = 3 total attempts
        assert mock_client.audio.transcriptions.create.call_count == 3
        assert mock_sleep.call_count == 2

    @patch("client.OpenAI")
    def test_non_retryable_error_raises_immediately(self, mock_openai_cls):
        """4xx errors (except 429) are not retried"""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        error = _make_server_error(400)
        error.status_code = 400

        mock_client.audio.transcriptions.create.side_effect = error

        client = WhisperClient(api_key="test", provider="openai")

        with patch.object(client, "_extract_audio", return_value="/tmp/audio.mp3"):
            with patch("builtins.open", mock_open(read_data=b"audio")):
                with patch("client.os.unlink"):
                    with pytest.raises(APIError):
                        client.transcribe_video("/fake/video.mp4", max_retries=3)

        # Only 1 attempt — no retries for 400
        assert mock_client.audio.transcriptions.create.call_count == 1


class TestBackoffDelay:
    """Tests for _backoff_delay calculation"""

    def test_exponential_growth(self):
        """Delay increases exponentially"""
        d0 = _backoff_delay(0)
        d1 = _backoff_delay(1)
        d2 = _backoff_delay(2)
        # Base: 1, 2, 4 (with jitter up to 50%)
        assert 1.0 <= d0 <= 1.5
        assert 2.0 <= d1 <= 3.0
        assert 4.0 <= d2 <= 6.0

    def test_retry_after_header_respected(self):
        """Uses Retry-After header value when present"""
        error = _make_rate_limit_error(retry_after=30)
        delay = _backoff_delay(0, error)
        assert delay == 30.0

    def test_retry_after_invalid_falls_back(self):
        """Falls back to exponential when Retry-After is invalid"""
        error = _make_rate_limit_error()
        error.response.headers["retry-after"] = "not-a-number"
        delay = _backoff_delay(0, error)
        assert 1.0 <= delay <= 1.5


# ============================================================
# Cache Helpers
# ============================================================


class TestCacheHelpers:
    """Tests for transcription cache load/save"""

    def test_load_cached_success(self, tmp_path):
        """Load valid cached transcription"""
        cache_file = tmp_path / "whisper_words.json"
        words = [
            {"word": "hello", "start": 0.0, "end": 0.5},
            {"word": "world", "start": 0.5, "end": 1.0},
        ]
        cache_file.write_text(json.dumps(words))

        result = load_cached_transcription(str(cache_file))
        assert result == words

    def test_load_cached_missing_file(self, tmp_path):
        """Return None for missing cache file"""
        result = load_cached_transcription(str(tmp_path / "nonexistent.json"))
        assert result is None

    def test_load_cached_invalid_json(self, tmp_path):
        """Return None for corrupted cache file"""
        cache_file = tmp_path / "whisper_words.json"
        cache_file.write_text("not valid json {{{")

        result = load_cached_transcription(str(cache_file))
        assert result is None

    def test_load_cached_not_a_list(self, tmp_path):
        """Return None if cache is not a list"""
        cache_file = tmp_path / "whisper_words.json"
        cache_file.write_text(json.dumps({"words": "wrong format"}))

        result = load_cached_transcription(str(cache_file))
        assert result is None

    def test_save_cache(self, tmp_path):
        """Save word timestamps to cache file"""
        cache_file = tmp_path / "whisper_words.json"
        words = [
            {"word": "hello", "start": 0.0, "end": 0.5},
            {"word": "world", "start": 0.5, "end": 1.0},
        ]

        save_transcription_cache(words, str(cache_file))

        loaded = json.loads(cache_file.read_text())
        assert loaded == words

    def test_save_cache_creates_parent_dirs(self, tmp_path):
        """Save creates parent directories if needed"""
        cache_file = tmp_path / "deep" / "nested" / "whisper_words.json"
        words = [{"word": "test", "start": 0.0, "end": 0.5}]

        save_transcription_cache(words, str(cache_file))

        assert cache_file.exists()
        loaded = json.loads(cache_file.read_text())
        assert loaded == words


# ============================================================
# TranscriptionResult
# ============================================================


class TestTranscriptionResult:
    """Tests for TranscriptionResult dataclass"""

    def test_dataclass_fields(self):
        result = TranscriptionResult(
            text="hello world",
            words=[
                {"word": "hello", "start": 0.0, "end": 0.5},
                {"word": "world", "start": 0.5, "end": 1.0},
            ],
            duration=1.0,
        )
        assert result.text == "hello world"
        assert len(result.words) == 2
        assert result.duration == 1.0
