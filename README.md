<p align="center">
  <img src="logo.png" alt="claudetube" width="500">
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

### Install

```bash
git clone https://github.com/thoughtpunch/claudetube
cd claudetube
./install.sh
```

Or via pip (once published):

```bash
pip install claudetube
```

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
| `/yt:list` | List all cached videos |

## Python API

```python
from claudetube import process_video, transcribe_video, get_frames_at

# Transcribe a video
result = process_video("https://youtube.com/watch?v=VIDEO_ID")
print(result.transcript_srt.read_text())

# Standalone Whisper transcription (cache-first, no full processing)
result = transcribe_video("VIDEO_ID", whisper_model="small")
print(result["source"])  # "cached" or "whisper"

# Extract frames at a specific timestamp
frames = get_frames_at("VIDEO_ID", start_time=120, duration=10)
```

## How It Works

1. **Download** -- Fetches lowest quality video (144p) for speed
2. **Transcribe** -- Uses faster-whisper with batched inference
3. **Cache** -- Stores everything at `~/.claude/video_cache/{VIDEO_ID}/`
4. **Drill-in** -- Extract frames on-demand when visual context is needed

### Cache Structure

```
~/.claude/video_cache/
└── dYP2V_nK8o0/
    ├── state.json     # Metadata (title, description, tags, etc.)
    ├── audio.mp3      # Extracted audio
    ├── audio.srt      # Timestamped transcript
    ├── audio.txt      # Plain text transcript
    ├── drill/         # Quick frames (480p)
    └── hq/            # High-quality frames (1280p)
```

## Architecture

claudetube uses a **provider-based architecture**. Video downloading is handled through `yt-dlp`, which currently supports YouTube and [1000+ other sites](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md). The transcription and frame extraction pipeline is provider-agnostic -- it works with any video source that yt-dlp supports, and the architecture is designed to accommodate additional providers in the future.

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
```

## Documentation

Full documentation is available in the [documentation/](documentation/) folder:

- **[Getting Started](documentation/getting-started/)** - Installation, quick start, MCP setup
- **[Core Concepts](documentation/concepts/)** - Video understanding, transcripts, frames, scenes
- **[Architecture](documentation/architecture/)** - Modules, data flow, tool wrappers
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
