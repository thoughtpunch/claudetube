<p align="center">
  <img src="documentation/files/images/logo.png" alt="claudetube" width="500">
</p>

<h1 align="center">claudetube</h1>

<p align="center">
  <strong>Let AI watch and understand online videos.</strong>
</p>

<p align="center">
  <a href="https://github.com/thoughtpunch/claudetube/actions/workflows/ci.yml"><img src="https://github.com/thoughtpunch/claudetube/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/claudetube/"><img src="https://img.shields.io/pypi/v/claudetube.svg?cacheSeconds=3600" alt="PyPI"></a>
  <a href="https://pypi.org/project/claudetube/"><img src="https://img.shields.io/pypi/pyversions/claudetube.svg?cacheSeconds=3600" alt="Python"></a>
  <a href="https://github.com/thoughtpunch/claudetube/blob/main/LICENSE"><img src="https://img.shields.io/github/license/thoughtpunch/claudetube.svg" alt="License"></a>
  <a href="https://github.com/thoughtpunch/claudetube/stargazers"><img src="https://img.shields.io/github/stars/thoughtpunch/claudetube.svg?style=social" alt="Stars"></a>
</p>

---

> **MCP Server for Video Understanding** — works with any AI that supports [Model Context Protocol](https://modelcontextprotocol.io): Claude Code, Claude Desktop, Cursor, Zed, and more.

claudetube downloads online videos, transcribes them with [faster-whisper](https://github.com/SYSTRAN/faster-whisper), and lets AI "see" specific moments by extracting frames on-demand. It's an **MCP server** exposing 40+ tools that any MCP-compatible client can use.

**Supports 1,500+ video sites** via [yt-dlp](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md) including YouTube, Vimeo, Dailymotion, Twitch, TikTok, Twitter/X, Instagram, Reddit, and many more.

## Why This Exists

**Claude doesn't have native video input.** When you share a YouTube link with Claude, it sees nothing—just a URL string.

Google's Gemini can process video natively: pass a URL, ask a question, get an answer. One API call. Claude can't do this (yet), so claudetube exists to bridge that gap.

I (Dan) built claudetube because I was using Claude to help me make a game, and I kept finding YouTube tutorials that explained exactly what I needed. The problem? I couldn't just *show* Claude the video.

Every other YouTube MCP tool just dumps the transcript and calls it a day. But when a tutorial says "look at this code here" or "notice how the sprite moves", the transcript alone is useless. I needed Claude to actually *see* what I was seeing—to look at the code on screen, read the diagrams, understand the visual context.

**[Read more about the vision](documentation/vision/problem-space.md)**

## Honest Assessment: claudetube vs Native Video AI

| Aspect | Gemini (native) | claudetube |
|--------|-----------------|------------|
| **UX** | URL + question → answer | process_video → get_frames → synthesize |
| **Sites** | YouTube only (public) | 1,500+ sites via yt-dlp |
| **Caching** | Reprocesses each time | Instant on second query |
| **Cost** | 1fps × full duration | Extract only what you need |
| **Precision** | 1fps sampling | Exact timestamps, HQ for code |
| **Offline** | No | Yes (cached content) |

**Where claudetube is worse:** More complex. Requires multi-step orchestration. 40 tools to learn.

**Where claudetube wins:** Works on more sites, cheaper for repeated queries, finer control, works offline.

**The goal:** Close the UX gap with a streamlined single-call interface while preserving the power-user capabilities. See the [roadmap](documentation/vision/roadmap.md).

## Quick Start

### Prerequisites

- **Python 3.10+**
- **ffmpeg** (system package)
  ```bash
  # macOS
  brew install ffmpeg

  # Ubuntu/Debian
  sudo apt install ffmpeg
  ```
- **deno** (recommended for YouTube) -- Since yt-dlp 2026.01.29, deno is required for full YouTube support (JS challenge solving). Without it, only limited YouTube clients are available.
  ```bash
  # macOS
  brew install deno

  # Linux
  curl -fsSL https://deno.land/install.sh | sh
  ```

### Install

```bash
git clone https://github.com/thoughtpunch/claudetube
cd claudetube
./install.sh
```

Or via pip (once published):

```bash
pip install claudetube[mcp]
```

### Install as MCP Server (Claude Code)

Add claudetube directly to Claude Code as an MCP server:

```bash
# Install the package first
pip install claudetube[mcp]

# Register with Claude Code
claude mcp add --transport stdio claudetube -- claudetube-mcp
```

Or add to your `.mcp.json` / `~/.claude.json`:

```json
{
  "mcpServers": {
    "claudetube": {
      "type": "stdio",
      "command": "claudetube-mcp"
    }
  }
}
```

Then restart Claude Code. All 40+ MCP tools will be available automatically.

### Traditional Install

The installer does two things:
1. Creates a Python venv at `~/.claudetube/venv/`
2. Installs the `claudetube` package + dependencies (yt-dlp, faster-whisper)

Then register the MCP server:
```bash
claude mcp add claudetube ~/.claudetube/venv/bin/claudetube-mcp
```

Restart Claude Code after registering.

### Why not a pre-built binary?

claudetube depends on faster-whisper (C++ transcription engine) and ffmpeg (system media tool). These have platform-specific native code that can't be bundled into a single static binary. The install script handles all of this automatically.

## Usage with Claude Code

Just talk naturally:

```
"Summarize this video: https://youtube.com/watch?v=abc123"
"What happens at minute 5?"
"How did they implement the auth flow?"
"Show me the code at 3:42"
```

Claude will use the appropriate MCP tools automatically:
1. Download and transcribe the video (~60s first time, cached after)
2. Read the transcript
3. If needed, extract frames to "see" specific moments
4. Answer your question

### Key MCP Tools

| Tool | Purpose |
|------|---------|
| `ask_video` | **Simplest** - URL + question → answer (handles everything) |
| `process_video_tool` | Download and transcribe a video |
| `get_frames` | Extract frames at a timestamp |
| `get_hq_frames` | HQ frames for code/text/diagrams |
| `watch_video_tool` | Deep analysis with evidence gathering |
| `find_moments_tool` | Find moments matching a query |
| `get_scenes` | Get scene structure and boundaries |

All 40+ tools are auto-discovered by Claude when the MCP server is registered.

## Python API

```python
from claudetube import process_video, transcribe_video, get_frames_at, get_hq_frames_at

# Process a video (downloads, transcribes, caches)
result = process_video("https://youtube.com/watch?v=VIDEO_ID")
print(result.transcript_txt.read_text())

# Standalone Whisper transcription (cache-first, no full processing)
result = transcribe_video("VIDEO_ID", whisper_model="small")
print(result["source"])  # "cached" or "whisper"

# Extract frames at a specific timestamp
frames = get_frames_at("VIDEO_ID", start_time=120, duration=10)

# Extract HQ frames for reading code/text
hq_frames = get_hq_frames_at("VIDEO_ID", start_time=120, duration=5)
```

## How It Works

1. **Download** -- Fetches lowest quality video (144p) for speed
2. **Transcribe** -- Uses faster-whisper with batched inference
3. **Cache** -- Stores everything at `~/.claudetube/cache/{VIDEO_ID}/`
4. **Drill-in** -- Extract frames on-demand when visual context is needed

### Data Location

All claudetube data is stored under `~/.claudetube/` by default:

```
~/.claudetube/
├── config.yaml              # User configuration
├── db/
│   ├── claudetube.db        # Metadata database
│   └── claudetube-vectors.db # Vector embeddings
├── cache/
│   └── {video_id}/          # Per-video cache
│       ├── state.json       # Metadata (title, description, tags)
│       ├── audio.mp3        # Extracted audio
│       ├── audio.srt        # Timestamped transcript
│       ├── audio.txt        # Plain text transcript
│       ├── thumbnail.jpg    # Video thumbnail
│       ├── drill/           # Quick frames (480p)
│       ├── hq/              # High-quality frames (1280p)
│       ├── scenes/          # Scene segmentation data
│       └── entities/        # People tracking, knowledge graph
└── logs/                    # Application logs (future)
```

### Configuration

**Override the root directory:** Set `CLAUDETUBE_ROOT` environment variable

**Override just the cache directory:** Configuration priority (highest first):

1. **Environment variable**: `CLAUDETUBE_CACHE_DIR=/path/to/cache`
2. **Project config**: `.claudetube/config.yaml` in your project
3. **User config**: `~/.claudetube/config.yaml`
4. **Default**: `~/.claudetube/cache`

Example project config:

```yaml
# .claudetube/config.yaml
cache_dir: ./video_cache
```

See [Configuration Guide](documentation/guides/configuration.md) for details.

## Architecture

claudetube uses a **provider-based architecture** with a modular design. Video downloading is handled through `yt-dlp` (1,500+ sites), while AI capabilities (transcription, vision analysis, reasoning, embeddings) are served by a configurable provider system supporting 11 providers (OpenAI, Anthropic, Google, Deepgram, AssemblyAI, Ollama, Voyage, and more). The MCP server exposes 40 tools for video processing, scene analysis, entity extraction, knowledge graphs, and accessibility features. See [Architecture](documentation/architecture.md) for details.

## Development

```bash
git clone https://github.com/thoughtpunch/claudetube
cd claudetube
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

### Linting

```bash
ruff check src/ tests/
ruff format src/ tests/
mypy src/
```

## Documentation

Full documentation is available in the [documentation/](documentation/) folder:

- **[Getting Started](documentation/getting-started/)** - Installation, quick start, MCP setup
- **[Core Concepts](documentation/concepts/)** - Video understanding, transcripts, frames, scenes
- **[Architecture](documentation/architecture.md)** - Modules, data flow, tool wrappers
- **[Vision](documentation/vision/)** - The problem space, roadmap, what makes claudetube different

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Run tests and linting before committing
4. Open a pull request against `main`

## Legal

This project is **not affiliated with, endorsed by, or associated with YouTube, Google, or Alphabet Inc.** "YouTube" is a trademark of Google LLC. This software is an independent, open-source tool that interacts with publicly available video content through third-party libraries ([yt-dlp](https://github.com/yt-dlp/yt-dlp)). Users are solely responsible for ensuring their use of this software complies with all applicable terms of service and laws.

## License

[MIT](LICENSE) -- free to use, modify, and distribute.
