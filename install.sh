#!/bin/bash
set -e

# claudetube installer (macOS / Linux)
# Installs Python package into a stable venv + registers MCP server

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_DIR="$HOME/.claudetube"
VENV_DIR="$INSTALL_DIR/venv"

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

# --- Check recommended dependencies ---
if ! command -v deno &>/dev/null; then
    echo ""
    echo "WARNING: deno is not installed."
    echo "  Since yt-dlp 2026.01.29, deno is required for full YouTube support."
    echo "  Without it, only limited YouTube clients (android_vr) are available."
    echo ""
    if [[ "$OS" == "macos" ]]; then
        echo "  Install with: brew install deno"
    elif [[ "$OS" == "linux" ]]; then
        echo "  Install with: curl -fsSL https://deno.land/install.sh | sh"
    fi
    echo "  More info: https://deno.land"
    echo ""
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

echo ""
echo "=== Installation complete ==="
echo ""
echo "Installed to: $INSTALL_DIR"
echo "Python venv:  $VENV_DIR"
echo ""
echo "=== Register MCP Server ==="
echo ""
echo "Add claudetube to Claude Code:"
echo ""
echo "  claude mcp add claudetube $VENV_DIR/bin/claudetube-mcp"
echo ""
echo "Then restart Claude Code. All 40+ tools will be available."
echo ""
echo "Example usage (just talk to Claude):"
echo "  'Summarize this video: https://youtube.com/watch?v=...'"
echo "  'What happens at minute 5 in video abc123?'"
echo ""
