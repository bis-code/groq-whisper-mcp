# groq-whisper-mcp

MCP server for audio/video transcription via Groq-hosted Whisper. Word-level timestamps, automatic caching, cost estimation.

**9x cheaper than OpenAI Whisper** — same model, different API.

| Provider | Model | Cost/hour |
|----------|-------|-----------|
| Groq | whisper-large-v3-turbo | $0.04 |
| Groq | whisper-large-v3 | $0.111 |
| Groq | distil-whisper | $0.02 |
| OpenAI | whisper-1 | $0.36 |

## Prerequisites

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/) (for audio extraction from video)
- [Groq API key](https://console.groq.com/keys)

## Installation

```bash
git clone https://github.com/bis-code/groq-whisper-mcp.git
cd groq-whisper-mcp
python3 -m venv venv
venv/bin/pip install mcp openai
```

## Claude Code Configuration

Add to your project's `.mcp.json`:

```json
{
  "mcpServers": {
    "whisper": {
      "command": "/path/to/groq-whisper-mcp/venv/bin/python",
      "args": ["-m", "server"],
      "cwd": "/path/to/groq-whisper-mcp/src",
      "env": {
        "GROQ_API_KEY": "your-groq-api-key"
      }
    }
  }
}
```

## Tools

### `transcribe_video`

Transcribe a video file with word-level timestamps.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `video_path` | string | yes | Path to video file |
| `model` | string | no | Whisper model (default: `whisper-large-v3-turbo`) |
| `force_retranscribe` | boolean | no | Bypass cache (default: `false`) |

Returns full text, word-level timestamps `[{word, start, end}]`, and duration. Results are cached per-project.

### `estimate_transcription_cost`

Estimate cost before transcribing.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `video_path` | string | yes | Path to video file |
| `model` | string | no | Whisper model (default: `whisper-large-v3-turbo`) |

Returns duration, estimated cost, and comparison with OpenAI pricing.

## How It Works

1. Extracts audio from video via ffmpeg (128kbps MP3, falls back to 64kbps if >25MB)
2. Sends to Groq's OpenAI-compatible API (same `openai` SDK, different `base_url`)
3. Returns word-level timestamps with ~0.1s precision
4. Caches results as `whisper_words.json` alongside the video

## Running Tests

```bash
PYTHONPATH=src venv/bin/pytest tests/ -v
```

## License

MIT
