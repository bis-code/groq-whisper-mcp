"""Tests for Whisper transcription client (Groq API via OpenAI SDK)"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest

from client import WhisperClient, TranscriptionResult, load_cached_transcription, save_transcription_cache


# ============================================================
# WhisperClient Init
# ============================================================


class TestWhisperClientInit:
    """Tests for WhisperClient initialization"""

    @patch("client.OpenAI")
    def test_init_with_explicit_key(self, mock_openai_cls):
        client = WhisperClient(api_key="test-key-123")
        mock_openai_cls.assert_called_once_with(
            api_key="test-key-123",
            base_url="https://api.groq.com/openai/v1",
        )

    @patch.dict(os.environ, {"GROQ_API_KEY": "env-key-456"})
    @patch("client.OpenAI")
    def test_init_from_env(self, mock_openai_cls):
        client = WhisperClient()
        mock_openai_cls.assert_called_once_with(
            api_key="env-key-456",
            base_url="https://api.groq.com/openai/v1",
        )

    @patch.dict(os.environ, {}, clear=True)
    @patch("client.OpenAI")
    def test_init_no_key_raises(self, mock_openai_cls):
        os.environ.pop("GROQ_API_KEY", None)
        with pytest.raises(ValueError, match="GROQ_API_KEY"):
            WhisperClient()


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
        client = WhisperClient(api_key="test")

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
        client = WhisperClient(api_key="test")

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
        client = WhisperClient(api_key="test")

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
        client = WhisperClient(api_key="test")

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

        client = WhisperClient(api_key="test")

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
    def test_transcribe_uses_correct_model(self, mock_openai_cls):
        """Transcription uses whisper-large-v3-turbo by default"""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.text = "test"
        mock_response.words = []
        mock_response.duration = 1.0
        mock_client.audio.transcriptions.create.return_value = mock_response

        client = WhisperClient(api_key="test")

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

        client = WhisperClient(api_key="test")

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

        client = WhisperClient(api_key="test")

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

        client = WhisperClient(api_key="test")

        with patch.object(client, "_extract_audio", return_value="/tmp/audio_temp.mp3"):
            with patch("builtins.open", mock_open(read_data=b"audio")):
                with patch("client.os.unlink") as mock_unlink:
                    with pytest.raises(Exception, match="API error"):
                        client.transcribe_video("/fake/video.mp4")

        mock_unlink.assert_called_once_with("/tmp/audio_temp.mp3")


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
