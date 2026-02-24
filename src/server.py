#!/usr/bin/env python3
"""
Groq Whisper MCP Server — transcription with word-level timestamps.

Uses Groq-hosted Whisper models via the OpenAI SDK.
Cost: $0.04/hr (whisper-large-v3-turbo) vs $0.36/hr (OpenAI) = 9x cheaper.

Configure in Claude Code .mcp.json:
{
    "whisper": {
        "command": "path/to/venv/bin/python",
        "args": ["-m", "src.server"],
        "cwd": "path/to/groq-whisper-mcp",
        "env": { "GROQ_API_KEY": "your-key-here" }
    }
}
"""

import json
import os
import subprocess
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from client import WhisperClient, load_cached_transcription, save_transcription_cache

server = Server("groq-whisper")


def _get_cache_path(video_path: str) -> str:
    """Resolve cache path for whisper word timestamps.

    Project structure (raw/ or edited/) -> project/edited/whisper_words.json
    Standalone video -> sibling whisper_words.json
    """
    video_parent = Path(video_path).parent

    if video_parent.name in ("raw", "edited", "final"):
        project_folder = video_parent.parent
        return str(project_folder / "edited" / "whisper_words.json")

    return str(video_parent / "whisper_words.json")


@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="transcribe_video",
            description="""Transcribe a video using Whisper via Groq API. Returns word-level timestamps.

Uses Groq's hosted Whisper models for fast, accurate transcription with per-word timing.
Cost: ~$0.04/hour (whisper-large-v3-turbo). Results cached per-project.

Returns: Full text, word-level timestamps [{word, start, end}], and duration.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "video_path": {
                        "type": "string",
                        "description": "Path to the video file to transcribe"
                    },
                    "model": {
                        "type": "string",
                        "description": "Whisper model to use (default: whisper-large-v3-turbo)",
                        "default": "whisper-large-v3-turbo"
                    },
                    "force_retranscribe": {
                        "type": "boolean",
                        "description": "Force re-transcription even if cache exists (default: false)",
                        "default": False
                    }
                },
                "required": ["video_path"]
            }
        ),
        Tool(
            name="estimate_transcription_cost",
            description="""Estimate the cost of transcribing a video file.

Returns estimated cost based on video duration and selected model.
Models: whisper-large-v3-turbo ($0.04/hr), whisper-large-v3 ($0.111/hr), distil-whisper ($0.02/hr).
Includes comparison with OpenAI Whisper pricing ($0.36/hr).""",
            inputSchema={
                "type": "object",
                "properties": {
                    "video_path": {
                        "type": "string",
                        "description": "Path to the video file"
                    },
                    "model": {
                        "type": "string",
                        "description": "Whisper model (default: whisper-large-v3-turbo)",
                        "default": "whisper-large-v3-turbo"
                    }
                },
                "required": ["video_path"]
            }
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "transcribe_video":
        video_path = os.path.abspath(arguments["video_path"])
        model = arguments.get("model", "whisper-large-v3-turbo")
        force = arguments.get("force_retranscribe", False)

        if not os.path.exists(video_path):
            return [TextContent(type="text", text=f"Error: Video not found: {video_path}")]

        cache_path = _get_cache_path(video_path)

        # Check cache unless forced
        if not force:
            cached = load_cached_transcription(cache_path)
            if cached is not None:
                return [TextContent(type="text", text=json.dumps({
                    "video_path": video_path,
                    "source": "cache",
                    "cache_path": cache_path,
                    "word_count": len(cached),
                    "words": cached,
                }, indent=2))]

        try:
            client = WhisperClient()
            result = client.transcribe_video(video_path, model=model)
            save_transcription_cache(result.words, cache_path)

            return [TextContent(type="text", text=json.dumps({
                "video_path": video_path,
                "source": "whisper_api",
                "model": model,
                "cache_path": cache_path,
                "text": result.text,
                "duration": result.duration,
                "word_count": len(result.words),
                "words": result.words,
            }, indent=2))]

        except Exception as e:
            return [TextContent(type="text", text=f"Transcription failed: {str(e)}")]

    elif name == "estimate_transcription_cost":
        video_path = os.path.abspath(arguments["video_path"])
        model = arguments.get("model", "whisper-large-v3-turbo")

        if not os.path.exists(video_path):
            return [TextContent(type="text", text=f"Error: Video not found: {video_path}")]

        # Get video duration via ffprobe
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", video_path],
                capture_output=True, text=True
            )
            duration_seconds = float(result.stdout.strip())
        except Exception:
            return [TextContent(type="text", text="Error: Could not determine video duration")]

        rates = {
            "whisper-large-v3-turbo": 0.04,
            "whisper-large-v3": 0.111,
            "distil-whisper-large-v3-en": 0.02,
        }
        rate = rates.get(model, 0.04)
        duration_hours = duration_seconds / 3600
        cost = duration_hours * rate
        openai_cost = (duration_seconds / 60) * 0.006

        return [TextContent(type="text", text=json.dumps({
            "video_path": video_path,
            "model": model,
            "duration_seconds": round(duration_seconds, 1),
            "duration_minutes": round(duration_seconds / 60, 1),
            "rate_per_hour": rate,
            "estimated_cost_usd": round(cost, 4),
            "comparison": {
                "openai_whisper_cost": round(openai_cost, 4),
                "groq_savings_vs_openai": f"{round((1 - cost / openai_cost) * 100)}%" if openai_cost > 0 else "N/A"
            }
        }, indent=2))]

    return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def _main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main():
    import asyncio
    asyncio.run(_main())


if __name__ == "__main__":
    main()
