# MCP Server Setup

> Use claudetube with Claude Desktop and Claude Code.

claudetube includes an MCP (Model Context Protocol) server that exposes video processing tools to any MCP-compatible client.

## Prerequisites

- Python 3.10+
- ffmpeg (for frame extraction)
- yt-dlp (installed automatically)

## Installation

### Quick Install (Recommended)

```bash
# Clone and install
git clone https://github.com/thoughtpunch/claudetube.git
cd claudetube
./install.sh   # macOS/Linux
# or
./install.ps1  # Windows (PowerShell)
```

The install script creates a venv at `~/.claudetube/venv/` and installs the package.

### Manual Install

```bash
python3 -m venv ~/.claudetube/venv
~/.claudetube/venv/bin/pip install ".[mcp]"
```

### Docker Install

See [Docker Setup](../docker.md) for container-based setup.

## Configuring Claude Code

Add claudetube as an MCP server:

```bash
claude mcp add claudetube ~/.claudetube/venv/bin/claudetube-mcp
```

Windows:
```powershell
claude mcp add claudetube $HOME\.claudetube\venv\Scripts\claudetube-mcp.exe
```

Restart Claude Code after adding. The tools will be available automatically.

## Configuring Claude Desktop

Edit your config file:
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "claudetube": {
      "command": "/Users/YOU/.claudetube/venv/bin/claudetube-mcp"
    }
  }
}
```

Replace `/Users/YOU` with your actual home directory path.

## Available MCP Tools

| Tool | Description |
|------|-------------|
| `process_video` | Download, transcribe, cache a video. Returns metadata + transcript. |
| `get_frames` | Extract frames at a specific time range (480px default). |
| `get_hq_frames` | Extract high-quality frames (1280px) for code/text. |
| `transcribe_video` | Transcribe/re-transcribe with specific Whisper model. |
| `get_transcript` | Get full transcript (no 50k char cap). |
| `list_cached_videos` | List all processed videos in cache. |

## Usage Examples

Once configured, just paste video URLs in your conversation:

> "Summarize this video: https://youtube.com/watch?v=..."

> "What code is shown at the 5 minute mark? https://youtube.com/watch?v=..."

> "Find where they explain authentication in this video: https://youtube.com/watch?v=..."

Claude will automatically use the appropriate tools.

## Testing the Server

Run manually to verify installation:

```bash
~/.claudetube/venv/bin/claudetube-mcp
```

The server communicates via stdio (JSON-RPC). Logs go to stderr.

## Troubleshooting

### Tools not appearing

1. Restart Claude Code/Desktop
2. Check the MCP server path is correct
3. Verify Python venv has claudetube installed

### "ffmpeg not found"

Install ffmpeg and ensure it's in your PATH.

### Slow first transcription

First transcription downloads the Whisper model (~500MB). Subsequent runs are fast.

---

**See also**:
- [Installation](installation.md) - Full installation guide
- [Quick Start](quickstart.md) - Using the Python API
