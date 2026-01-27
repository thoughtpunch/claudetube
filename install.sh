#!/bin/bash
set -e

# claudetube installer
# Installs Python package into a stable venv + Claude Code slash commands

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_DIR="$HOME/.claudetube"
VENV_DIR="$INSTALL_DIR/venv"
CLAUDE_COMMANDS_DIR="$HOME/.claude/commands"

echo "Installing claudetube..."

# 1. Create stable venv at ~/.claudetube/venv/
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating Python venv at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi

# 2. Install package into the venv
echo "Installing Python package..."
"$VENV_DIR/bin/pip" install --upgrade pip -q
"$VENV_DIR/bin/pip" install "$REPO_DIR" -q

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
echo "Done! Restart Claude Code to use the new commands."
echo ""
echo "Installed to: $INSTALL_DIR"
echo "Python venv:  $VENV_DIR"
echo ""
echo "Commands available (in ANY Claude Code session):"
echo "  /yt <url> [question]          - Analyze a YouTube video"
echo "  /yt:see <id> <timestamp>      - Extract frames at timestamp"
echo "  /yt:hq <id> <timestamp>       - Extract high-quality frames"
echo "  /yt:transcript <id>           - Show cached transcript"
echo "  /yt:list                      - List cached videos"
