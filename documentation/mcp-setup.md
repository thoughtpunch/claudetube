# MCP Server Setup

claudetube includes an MCP (Model Context Protocol) server that exposes video processing tools to any MCP-compatible client — Claude Desktop, Claude Code, or third-party tools.

## Prerequisites

- Python 3.10+
- ffmpeg (for frame extraction)
- yt-dlp (installed automatically with the package)

## Installation

### Native Install (recommended)

```bash
# Clone and install
git clone https://github.com/thoughtpunch/claudetube.git
cd claudetube
./install.sh   # macOS/Linux
# or
./install.ps1  # Windows (PowerShell)
```

The install script creates a venv at `~/.claudetube/venv/` and installs the package with MCP support.

### Manual Install

```bash
python3 -m venv ~/.claudetube/venv
~/.claudetube/venv/bin/pip install ".[mcp]"
```

### Docker Install

See [docker.md](docker.md) for container-based setup.

## Configuring Claude Code

Add claudetube as an MCP server:

```bash
claude mcp add claudetube ~/.claudetube/venv/bin/claudetube-mcp
```

Or on Windows:

```powershell
claude mcp add claudetube $HOME\.claudetube\venv\Scripts\claudetube-mcp.exe
```

After adding, restart Claude Code. The 5 tools will be available automatically.

## Configuring Claude Desktop

Add to your Claude Desktop config file (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "claudetube": {
      "command": "/Users/YOU/.claudetube/venv/bin/claudetube-mcp"
    }
  }
}
```

Replace `/Users/YOU` with your home directory path.

## Available Tools

| Tool | Description |
|------|-------------|
| `process_video` | Download, transcribe, and cache a YouTube video. Returns metadata + transcript (capped at 50k chars inline). |
| `get_frames` | Extract frames at a specific time range from a cached video. Supports quality tiers (lowest/low/medium/high/highest). |
| `get_hq_frames` | Extract high-quality frames for reading text, code, or small UI elements. |
| `list_cached_videos` | List all videos that have been processed and cached. |
| `get_transcript` | Get the full transcript for a cached video (no character cap). Supports txt and srt formats. |

## Cache Location

All processed videos are cached at `~/.claude/video_cache/`. Each video gets a directory named by its YouTube video ID containing:

- `state.json` — metadata and processing state
- `audio.txt` — plain text transcript
- `audio.srt` — SRT subtitle file
- `thumbnail.jpg` — video thumbnail
- `drill_*/` — extracted frame directories

## Running Manually

To test the server directly:

```bash
~/.claudetube/venv/bin/claudetube-mcp
```

The server communicates via stdio (JSON-RPC over stdin/stdout). All log output goes to stderr.
