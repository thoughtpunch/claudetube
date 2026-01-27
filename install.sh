#!/bin/bash
set -e

# claudetube installer (macOS / Linux)
# Installs Python package into a stable venv + Claude Code slash commands + MCP server

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_DIR="$HOME/.claudetube"
VENV_DIR="$INSTALL_DIR/venv"
CLAUDE_COMMANDS_DIR="$HOME/.claude/commands"

echo "Installing claudetube..."
echo ""

# --- Detect OS ---
OS="unknown"
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
elif [[ "$OSTYPE" == "linux"* ]]; then
    OS="linux"
fi
echo "Detected OS: $OS"

# --- Check required system dependencies ---
MISSING=0

if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 is not installed."
    if [[ "$OS" == "macos" ]]; then
        echo "  Install with: brew install python3"
    elif [[ "$OS" == "linux" ]]; then
        echo "  Install with: sudo apt install python3 python3-venv"
    fi
    MISSING=1
fi

if ! command -v ffmpeg &>/dev/null; then
    echo "ERROR: ffmpeg is not installed (required for frame extraction)."
    if [[ "$OS" == "macos" ]]; then
        echo "  Install with: brew install ffmpeg"
    elif [[ "$OS" == "linux" ]]; then
        echo "  Install with: sudo apt install ffmpeg"
    fi
    MISSING=1
fi

if [[ "$MISSING" -eq 1 ]]; then
    echo ""
    echo "Please install the missing dependencies above and re-run this script."
    exit 1
fi

echo ""

# 1. Create stable venv at ~/.claudetube/venv/
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating Python venv at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi

# 2. Install package with MCP support into the venv
echo "Installing Python package (with MCP support)..."
"$VENV_DIR/bin/pip" install --upgrade pip -q
"$VENV_DIR/bin/pip" install "$REPO_DIR[mcp]" -q

# 3. Install faster-whisper (not in pyproject.toml due to heavy C++ deps)
echo "Installing faster-whisper..."
"$VENV_DIR/bin/pip" install faster-whisper -q

# 4. Install Claude Code commands
echo "Installing Claude Code slash commands..."
mkdir -p "$CLAUDE_COMMANDS_DIR"

# Install main command
if [ -f "$REPO_DIR/commands/yt.md" ]; then
    cp "$REPO_DIR/commands/yt.md" "$CLAUDE_COMMANDS_DIR/yt.md"
    echo "  Installed /yt"
fi

# Install subcommands
if [ -d "$REPO_DIR/commands/yt" ]; then
    rm -rf "$CLAUDE_COMMANDS_DIR/yt"
    cp -r "$REPO_DIR/commands/yt" "$CLAUDE_COMMANDS_DIR/yt"
    for cmd in "$CLAUDE_COMMANDS_DIR/yt"/*.md; do
        name=$(basename "$cmd" .md)
        echo "  Installed /yt:$name"
    done
fi

echo ""
echo "=== Installation complete ==="
echo ""
echo "Installed to: $INSTALL_DIR"
echo "Python venv:  $VENV_DIR"
echo ""
echo "Slash commands (in ANY Claude Code session):"
echo "  /yt <url> [question]          - Analyze a YouTube video"
echo "  /yt:see <id> <timestamp>      - Extract frames at timestamp"
echo "  /yt:hq <id> <timestamp>       - Extract high-quality frames"
echo "  /yt:transcript <id>           - Show cached transcript"
echo "  /yt:list                      - List cached videos"
echo ""
echo "=== MCP Server Setup (optional) ==="
echo ""
echo "To add claudetube as an MCP server for Claude Code:"
echo ""
echo "  claude mcp add claudetube $VENV_DIR/bin/claudetube-mcp"
echo ""
echo "Or start it manually:"
echo ""
echo "  $VENV_DIR/bin/claudetube-mcp"
echo ""
