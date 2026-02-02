[← Documentation](../README.md)

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

See [Docker Setup](../guides/docker.md) for container-based setup.

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

## Available MCP Tools (30 tools)

### Core Processing

| Tool | Description |
|------|-------------|
| `process_video_tool` | Download, transcribe, cache a video. Returns metadata + transcript. |
| `get_frames` | Extract frames at a specific time range (480px default). |
| `get_hq_frames` | Extract high-quality frames (1280px) for code/text. |
| `transcribe_video` | Transcribe/re-transcribe with specific Whisper model or provider. |
| `get_transcript` | Get full transcript (no 50k char cap). |
| `list_cached_videos` | List all processed videos in cache. |

### Scenes & Analysis

| Tool | Description |
|------|-------------|
| `get_scenes` | Get scene structure with timestamps and transcripts. |
| `generate_visual_transcripts` | Generate visual descriptions for scenes using vision AI. |
| `extract_entities_tool` | Extract entities (objects, people, text, concepts) from scenes. |
| `analyze_deep_tool` | Deep analysis with OCR, entities, and code detection. |
| `analyze_focus_tool` | Exhaustive frame-by-frame analysis of a time range. |
| `get_analysis_status_tool` | Check what analysis is cached for each scene. |

### People & Search

| Tool | Description |
|------|-------------|
| `track_people_tool` | Track people across scenes with optional face recognition. |
| `find_moments_tool` | Find moments matching a natural language query. |
| `watch_video_tool` | Actively watch and reason about a video to answer questions. |

### Playlists

| Tool | Description |
|------|-------------|
| `get_playlist` | Extract metadata from a playlist URL. |
| `list_playlists` | List all cached playlists. |

### Audio Description (Accessibility)

| Tool | Description |
|------|-------------|
| `get_descriptions` | Get visual descriptions for accessibility. |
| `describe_moment` | Describe visual content at a specific timestamp. |
| `get_accessible_transcript` | Get merged transcript with [AD] audio descriptions. |
| `has_audio_description` | Check if audio description content is available. |

### Knowledge Graph

| Tool | Description |
|------|-------------|
| `find_related_videos_tool` | Find videos related to a topic across all cached videos. |
| `index_video_to_graph_tool` | Index a video's entities into the knowledge graph. |
| `get_video_connections_tool` | Get videos connected by shared entities/concepts. |
| `get_knowledge_graph_stats_tool` | Get knowledge graph statistics. |

### Enrichment (Progressive Learning)

| Tool | Description |
|------|-------------|
| `record_qa_tool` | Record a Q&A interaction for future reference. |
| `search_qa_history_tool` | Search previously answered questions. |
| `get_scene_context_tool` | Get all learned context for a scene. |
| `get_enrichment_stats_tool` | Get cache enrichment statistics. |

### Providers

| Tool | Description |
|------|-------------|
| `list_providers_tool` | List available AI providers and capabilities. |

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
