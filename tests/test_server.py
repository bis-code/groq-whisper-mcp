"""Tests for Whisper MCP server tool handlers"""

import json
import os
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from server import _get_cache_path, call_tool


# ============================================================
# Cache Path Resolution
# ============================================================


class TestGetCachePath:
    """Tests for _get_cache_path project structure detection"""

    def test_raw_folder(self):
        """raw/ -> project/edited/whisper_words.<stem>.json"""
        result = _get_cache_path("/videos/my-project/raw/clip.mp4")
        assert result == "/videos/my-project/edited/whisper_words.clip.json"

    def test_edited_folder(self):
        """edited/ -> project/edited/whisper_words.<stem>.json"""
        result = _get_cache_path("/videos/my-project/edited/clip.mp4")
        assert result == "/videos/my-project/edited/whisper_words.clip.json"

    def test_final_folder(self):
        """final/ -> project/edited/whisper_words.<stem>.json"""
        result = _get_cache_path("/videos/my-project/final/output.mp4")
        assert result == "/videos/my-project/edited/whisper_words.output.json"

    def test_standalone_video(self):
        """Non-project video -> sibling whisper_words.<stem>.json"""
        result = _get_cache_path("/downloads/random-video.mp4")
        assert result == "/downloads/whisper_words.random-video.json"

    def test_root_video(self):
        """Video at filesystem root"""
        result = _get_cache_path("/video.mp4")
        assert result == "/whisper_words.video.json"

    def test_multiple_clips_same_dir_get_distinct_caches(self):
        """Regression: two clips in one dir must NOT share a cache file.

        Previously the cache was keyed per-directory, so transcribing a 2nd
        clip returned the 1st clip's words. The cache path is now per-file.
        """
        a = _get_cache_path("/videos/vid/raw/take-01.mp4")
        b = _get_cache_path("/videos/vid/raw/take-02.mp4")
        assert a != b
        assert a == "/videos/vid/edited/whisper_words.take-01.json"
        assert b == "/videos/vid/edited/whisper_words.take-02.json"


# ============================================================
# transcribe_video tool
# ============================================================


class TestTranscribeVideoTool:
    """Tests for the transcribe_video MCP tool handler"""

    @pytest.mark.asyncio
    async def test_file_not_found(self):
        """Returns error for missing video file"""
        result = await call_tool("transcribe_video", {"video_path": "/nonexistent/video.mp4"})
        assert len(result) == 1
        assert "not found" in result[0].text

    @pytest.mark.asyncio
    async def test_cache_hit(self, tmp_path):
        """Returns cached transcription when available"""
        video = tmp_path / "video.mp4"
        video.write_bytes(b"fake")
        cache = tmp_path / "whisper_words.video.json"
        words = [{"word": "cached", "start": 0.0, "end": 0.5}]
        cache.write_text(json.dumps(words))

        result = await call_tool("transcribe_video", {"video_path": str(video)})
        data = json.loads(result[0].text)
        assert data["source"] == "cache"
        assert data["word_count"] == 1

    @pytest.mark.asyncio
    async def test_force_retranscribe_skips_cache(self, tmp_path):
        """force_retranscribe=True bypasses cache"""
        video = tmp_path / "video.mp4"
        video.write_bytes(b"fake")
        cache = tmp_path / "whisper_words.video.json"
        cache.write_text(json.dumps([{"word": "old", "start": 0.0, "end": 0.5}]))

        mock_result = MagicMock()
        mock_result.text = "new transcription"
        mock_result.words = [{"word": "new", "start": 0.0, "end": 0.3}]
        mock_result.duration = 1.0

        with patch("server.WhisperClient") as mock_cls:
            mock_instance = mock_cls.return_value
            mock_instance.transcribe_video.return_value = mock_result
            mock_instance.provider = "openai"
            mock_instance.default_model = "whisper-1"
            result = await call_tool("transcribe_video", {
                "video_path": str(video),
                "force_retranscribe": True,
            })

        data = json.loads(result[0].text)
        assert data["source"] == "whisper_api"
        assert data["text"] == "new transcription"
        assert data["provider"] == "openai"

    @pytest.mark.asyncio
    async def test_api_error(self, tmp_path):
        """Returns error message on API failure"""
        video = tmp_path / "video.mp4"
        video.write_bytes(b"fake")

        with patch("server.WhisperClient") as mock_cls:
            mock_cls.return_value.transcribe_video.side_effect = RuntimeError("API timeout")
            result = await call_tool("transcribe_video", {"video_path": str(video)})

        assert "Transcription failed" in result[0].text
        assert "API timeout" in result[0].text


# ============================================================
# estimate_transcription_cost tool
# ============================================================


class TestEstimateCostTool:
    """Tests for the estimate_transcription_cost MCP tool handler"""

    @pytest.mark.asyncio
    async def test_file_not_found(self):
        """Returns error for missing video file"""
        result = await call_tool("estimate_transcription_cost", {"video_path": "/nonexistent.mp4"})
        assert "not found" in result[0].text

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True)
    async def test_cost_calculation_openai(self, tmp_path):
        """Calculates correct cost for OpenAI whisper-1"""
        video = tmp_path / "video.mp4"
        video.write_bytes(b"fake")

        with patch("server.subprocess") as mock_sp:
            mock_sp.run.return_value = MagicMock(stdout="600.0\n", returncode=0)
            result = await call_tool("estimate_transcription_cost", {"video_path": str(video)})

        data = json.loads(result[0].text)
        assert data["duration_seconds"] == 600.0
        assert data["duration_minutes"] == 10.0
        assert data["provider"] == "openai"
        assert data["model"] == "whisper-1"
        assert data["rate_per_hour"] == 0.36
        # 10 min = 1/6 hr * $0.36/hr = $0.06
        assert data["estimated_cost_usd"] == round((600 / 3600) * 0.36, 4)
        # Comparison includes all models
        assert "whisper-1" in data["comparison"]
        assert "whisper-large-v3-turbo" in data["comparison"]

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"GROQ_API_KEY": "gsk-test"}, clear=True)
    async def test_cost_calculation_groq(self, tmp_path):
        """Calculates correct cost for Groq default model"""
        video = tmp_path / "video.mp4"
        video.write_bytes(b"fake")

        with patch("server.subprocess") as mock_sp:
            mock_sp.run.return_value = MagicMock(stdout="600.0\n", returncode=0)
            result = await call_tool("estimate_transcription_cost", {"video_path": str(video)})

        data = json.loads(result[0].text)
        assert data["provider"] == "groq"
        assert data["model"] == "whisper-large-v3-turbo"
        assert data["rate_per_hour"] == 0.04
        assert data["estimated_cost_usd"] == round((600 / 3600) * 0.04, 4)


# ============================================================
# Unknown tool
# ============================================================


class TestUnknownTool:
    """Tests for unknown tool name handling"""

    @pytest.mark.asyncio
    async def test_unknown_tool(self):
        result = await call_tool("nonexistent_tool", {})
        assert "Unknown tool" in result[0].text


# ============================================================
# YouTube Auto-Captions
# ============================================================


from server import _extract_youtube_id


class TestExtractYoutubeId:
    """Tests for _extract_youtube_id"""

    def test_bare_id(self):
        assert _extract_youtube_id("fHx7eifXtSA") == "fHx7eifXtSA"

    def test_short_url(self):
        assert _extract_youtube_id("https://youtu.be/fHx7eifXtSA") == "fHx7eifXtSA"

    def test_watch_url(self):
        assert _extract_youtube_id("https://www.youtube.com/watch?v=fHx7eifXtSA") == "fHx7eifXtSA"

    def test_watch_url_with_query(self):
        assert _extract_youtube_id("https://www.youtube.com/watch?v=fHx7eifXtSA&feature=youtu.be") == "fHx7eifXtSA"

    def test_shorts_url(self):
        assert _extract_youtube_id("https://www.youtube.com/shorts/fHx7eifXtSA") == "fHx7eifXtSA"

    def test_too_short(self):
        assert _extract_youtube_id("too-short") is None

    def test_empty(self):
        assert _extract_youtube_id("") is None


class TestFetchYoutubeTranscript:
    """Tests for fetch_youtube_transcript tool handler"""

    @pytest.mark.asyncio
    async def test_happy_path(self):
        fake_segments = [
            {"text": "hello world", "start": 0.0, "duration": 1.5},
            {"text": "second line", "start": 1.5, "duration": 2.0},
        ]
        fake_fetched = MagicMock()
        fake_fetched.to_raw_data.return_value = fake_segments

        fake_ytt = MagicMock()
        fake_ytt.fetch.return_value = fake_fetched

        with patch("server.YouTubeTranscriptApi", return_value=fake_ytt):
            result = await call_tool("fetch_youtube_transcript", {
                "video_id_or_url": "https://youtu.be/fHx7eifXtSA"
            })

        data = json.loads(result[0].text)
        assert data["video_id"] == "fHx7eifXtSA"
        assert data["source"] == "youtube_auto_captions"
        assert data["language"] == "en"
        assert data["segment_count"] == 2
        assert data["text"] == "hello world second line"
        assert data["segments"] == fake_segments

    @pytest.mark.asyncio
    async def test_bare_id(self):
        fake_fetched = MagicMock()
        fake_fetched.to_raw_data.return_value = [{"text": "x", "start": 0.0, "duration": 1.0}]
        fake_ytt = MagicMock()
        fake_ytt.fetch.return_value = fake_fetched

        with patch("server.YouTubeTranscriptApi", return_value=fake_ytt):
            result = await call_tool("fetch_youtube_transcript", {"video_id_or_url": "fHx7eifXtSA"})

        data = json.loads(result[0].text)
        assert data["video_id"] == "fHx7eifXtSA"

    @pytest.mark.asyncio
    async def test_invalid_id_returns_error(self):
        result = await call_tool("fetch_youtube_transcript", {"video_id_or_url": "bad"})
        assert "Error: could not extract" in result[0].text

    @pytest.mark.asyncio
    async def test_fetch_failure_surfaces_error(self):
        fake_ytt = MagicMock()
        fake_ytt.fetch.side_effect = RuntimeError("no captions available")

        with patch("server.YouTubeTranscriptApi", return_value=fake_ytt):
            result = await call_tool("fetch_youtube_transcript", {"video_id_or_url": "fHx7eifXtSA"})

        assert "transcript fetch failed" in result[0].text.lower()
        assert "RuntimeError" in result[0].text
        assert "no captions available" in result[0].text

    @pytest.mark.asyncio
    async def test_language_param_passed_through(self):
        fake_fetched = MagicMock()
        fake_fetched.to_raw_data.return_value = []
        fake_ytt = MagicMock()
        fake_ytt.fetch.return_value = fake_fetched

        with patch("server.YouTubeTranscriptApi", return_value=fake_ytt):
            await call_tool("fetch_youtube_transcript", {
                "video_id_or_url": "fHx7eifXtSA",
                "language": "ro",
            })

        # languages kwarg includes user pref then en fallback.
        call_args, call_kwargs = fake_ytt.fetch.call_args
        assert call_args == ("fHx7eifXtSA",)
        assert call_kwargs == {"languages": ["ro", "en"]}
