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

claudetube downloads online videos, transcribes them with [faster-whisper](https://github.com/SYSTRAN/faster-whisper), and lets AI "see" specific moments by extracting frames on-demand. Built for [Claude Code](https://docs.anthropic.com/en/docs/claude-code) but works as a standalone Python library with any AI tool.

**Supports 1,500+ video sites** via [yt-dlp](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md) including YouTube, Vimeo, Dailymotion, Twitch, TikTok, Twitter/X, Instagram, Reddit, and many more.

## Why This Exists

I (Dan) built claudetube because I was using Claude to help me make a game, and I kept finding YouTube tutorials that explained exactly what I needed. The problem? I couldn't just *show* Claude the video.

Every other YouTube MCP tool just dumps the transcript and calls it a day. But when a tutorial says "look at this code here" or "notice how the sprite moves", the transcript alone is useless. I needed Claude to actually *see* what I was seeing -- to look at the code on screen, read the diagrams, understand the visual context.

Unlike other tools, claudetube doesn't just fetch transcripts. It lets AI work with video content the same way modern LLMs can browse the web -- fetching what's needed, when it's needed, with full visual context. The transcript is just the starting point. The real power is on-demand frame extraction that lets Claude read code, analyze diagrams, and understand what the speaker is actually showing.

**[Read more about the vision](documentation/vision/problem-space.md)**

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

The installer does three things:
1. Creates a Python venv at `~/.claudetube/venv/`
2. Installs the `claudetube` package + dependencies (yt-dlp, faster-whisper)
3. Copies slash commands to `~/.claude/commands/` (global to all Claude Code sessions)

Restart Claude Code after installing.

### Works from any Claude Code session

The installer puts slash commands in `~/.claude/commands/`, which is the global commands directory. Every Claude Code instance on your machine will have `/yt` available -- no per-project setup needed.

### Why not a pre-built binary?

claudetube depends on faster-whisper (C++ transcription engine) and ffmpeg (system media tool). These have platform-specific native code that can't be bundled into a single static binary. The install script handles all of this automatically.

## Usage with Claude Code

```
/yt https://youtube.com/watch?v=abc123 how did they make the sprites?
/yt https://vimeo.com/123456789 summarize the key points
/yt https://twitter.com/user/status/123 what is this video about?
```

Claude will:
1. Download and transcribe the video (~60s first time, cached after)
2. Read the transcript
3. If needed, extract frames to "see" specific moments
4. Answer your question

### Commands

| Command | Purpose |
|---------|---------|
| `/yt <url> [question]` | Analyze a video |
| `/yt:see <id> <timestamp>` | Quick frames (general visuals) |
| `/yt:hq <id> <timestamp>` | HQ frames (code, text, diagrams) |
| `/yt:transcribe <id> [model]` | Transcribe with Whisper (or return cached) |
| `/yt:transcript <id>` | Read cached transcript |
| `/yt:scenes <id>` | Get scene structure and boundaries |
| `/yt:find <id> <query>` | Find moments matching a query |
| `/yt:watch <id> <question>` | Actively watch and reason about a video |
| `/yt:deep <id>` | Deep analysis (OCR, entities, code detection) |
| `/yt:focus <id> <start> <end>` | Exhaustive frame-by-frame analysis of a section |
| `/yt:list` | List all cached videos |

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
