#!/usr/bin/env bash
# Launch wrapper for groq-whisper-mcp. Sources .env (mode 0600) so
# the configured API key reaches the Python process without leaking the
# value into Claude Code's .mcp.json file.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$HERE/.env" ]; then
  set -a
  source "$HERE/.env"
  set +a
fi
cd "$HERE/src"
exec "$HERE/venv/bin/python" -m server
