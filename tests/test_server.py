"""Tests for Groq Whisper MCP server tool handlers"""

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
        """raw/ -> project/edited/whisper_words.json"""
        result = _get_cache_path("/videos/my-project/raw/clip.mp4")
        assert result == "/videos/my-project/edited/whisper_words.json"

    def test_edited_folder(self):
        """edited/ -> project/edited/whisper_words.json"""
        result = _get_cache_path("/videos/my-project/edited/clip.mp4")
        assert result == "/videos/my-project/edited/whisper_words.json"

    def test_final_folder(self):
        """final/ -> project/edited/whisper_words.json"""
        result = _get_cache_path("/videos/my-project/final/output.mp4")
        assert result == "/videos/my-project/edited/whisper_words.json"

    def test_standalone_video(self):
        """Non-project video -> sibling whisper_words.json"""
        result = _get_cache_path("/downloads/random-video.mp4")
        assert result == "/downloads/whisper_words.json"

    def test_root_video(self):
        """Video at filesystem root"""
        result = _get_cache_path("/video.mp4")
        assert result == "/whisper_words.json"


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
        cache = tmp_path / "whisper_words.json"
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
        cache = tmp_path / "whisper_words.json"
        cache.write_text(json.dumps([{"word": "old", "start": 0.0, "end": 0.5}]))

        mock_result = MagicMock()
        mock_result.text = "new transcription"
        mock_result.words = [{"word": "new", "start": 0.0, "end": 0.3}]
        mock_result.duration = 1.0

        with patch("server.WhisperClient") as mock_cls:
            mock_cls.return_value.transcribe_video.return_value = mock_result
            result = await call_tool("transcribe_video", {
                "video_path": str(video),
                "force_retranscribe": True,
            })

        data = json.loads(result[0].text)
        assert data["source"] == "whisper_api"
        assert data["text"] == "new transcription"

    @pytest.mark.asyncio
    async def test_api_error(self, tmp_path):
        """Returns error message on API failure"""
        video = tmp_path / "video.mp4"
        video.write_bytes(b"fake")

        with patch("server.WhisperClient") as mock_cls:
            mock_cls.return_value.transcribe_video.side_effect = RuntimeError("Groq API timeout")
            result = await call_tool("transcribe_video", {"video_path": str(video)})

        assert "Transcription failed" in result[0].text
        assert "Groq API timeout" in result[0].text


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
    async def test_cost_calculation(self, tmp_path):
        """Calculates correct cost for known duration"""
        video = tmp_path / "video.mp4"
        video.write_bytes(b"fake")

        with patch("server.subprocess") as mock_sp:
            mock_sp.run.return_value = MagicMock(stdout="600.0\n", returncode=0)
            result = await call_tool("estimate_transcription_cost", {"video_path": str(video)})

        data = json.loads(result[0].text)
        assert data["duration_seconds"] == 600.0
        assert data["duration_minutes"] == 10.0
        # 10 min = 1/6 hr * $0.04/hr = $0.0067
        assert data["rate_per_hour"] == 0.04
        assert data["estimated_cost_usd"] == round((600 / 3600) * 0.04, 4)
        # OpenAI comparison
        assert "openai_whisper_cost" in data["comparison"]


# ============================================================
# Unknown tool
# ============================================================


class TestUnknownTool:
    """Tests for unknown tool name handling"""

    @pytest.mark.asyncio
    async def test_unknown_tool(self):
        result = await call_tool("nonexistent_tool", {})
        assert "Unknown tool" in result[0].text
