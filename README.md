# claudetube

**Let Claude watch YouTube videos.**

claudetube downloads YouTube videos, transcribes them with faster-whisper, and lets Claude "see" specific moments on-demand. Built for [Claude Code](https://docs.anthropic.com/en/docs/claude-code) but works as a standalone Python library too.

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
git clone https://github.com/dmarx/claudetube
cd claudetube
./install.sh
```

This does three things:
1. Creates a Python venv at `~/.claudetube/venv/`
2. Installs the `claudetube` package + dependencies (yt-dlp, faster-whisper)
3. Copies slash commands to `~/.claude/commands/` (global to all Claude Code sessions)

Restart Claude Code after installing.

### Can I use this from any Claude Code session?

**Yes.** The installer puts slash commands in `~/.claude/commands/`, which is the global commands directory. Every Claude Code instance on your machine will have `/yt` available.

### Why not a pre-built binary?

claudetube depends on faster-whisper (C++ transcription engine) and ffmpeg (system media tool). These have platform-specific native code that can't be bundled into a single static binary. The install script handles all of this automatically.

## Usage with Claude Code

```
/yt https://youtube.com/watch?v=abc123 how did they make the sprites?
```

Claude will:
1. Download and transcribe the video (~60s first time, cached after)
2. Read the transcript
3. If needed, extract frames to "see" specific moments
4. Answer your question

### Other Commands

| Command | Purpose |
|---------|---------|
| `/yt <url> [question]` | Analyze a video |
| `/yt:see <id> <timestamp>` | Quick frames (general visuals) |
| `/yt:hq <id> <timestamp>` | HQ frames (code, text, diagrams) |
| `/yt:transcript <id>` | Read cached transcript |
| `/yt:list` | List all cached videos |

## Python API

```python
from claudetube import process_video, get_frames_at

# Transcribe a video
result = process_video("https://youtube.com/watch?v=VIDEO_ID")
print(result.transcript_srt.read_text())

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

## Development

```bash
git clone https://github.com/dmarx/claudetube
cd claudetube
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## License

MIT
