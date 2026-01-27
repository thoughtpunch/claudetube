# Architecture

claudetube serves two interfaces from a single codebase:

1. **Claude Code slash commands** (`/yt`, `/yt:see`, etc.) — Markdown command files that inline-execute Python
2. **MCP server** (`claudetube-mcp`) — stdio JSON-RPC server for any MCP client

Both interfaces share the same `core.py` processing engine.

## Codebase Structure

```
src/claudetube/
  __init__.py          # Package exports, version
  core.py              # Video processing engine (all business logic)
  cli.py               # CLI entry point (claudetube command)
  mcp_server.py        # MCP server entry point (claudetube-mcp command)

commands/
  yt.md                # /yt slash command
  yt/
    see.md             # /yt:see slash command
    hq.md              # /yt:hq slash command
    transcript.md      # /yt:transcript slash command
    list.md            # /yt:list slash command

tests/
  test_core.py         # Core logic tests
  test_mcp_server.py   # MCP tool tests
```

## Data Flow

### Slash Commands

```
User runs /yt <url>
  → Claude reads commands/yt.md
  → Command template includes inline Python
  → Python calls core.process_video()
  → core.py downloads video, transcribes, caches
  → Results returned to Claude as text output
```

### MCP Server

```
MCP Client sends JSON-RPC request (stdin)
  → mcp_server.py routes to tool function
  → Tool function calls core.py via asyncio.to_thread()
  → core.py downloads video, transcribes, caches
  → Tool function returns JSON response (stdout)
```

## core.py — Processing Engine

The core module handles all video processing:

- **`process_video()`** — Main pipeline: fetch metadata → try subtitles → fall back to whisper → cache results
- **`get_frames_at()`** — Download a video segment and extract frames at a time range (with quality tiers)
- **`get_hq_frames_at()`** — Same as above but at high quality (1080p, high JPEG quality)
- **`extract_video_id()`** — Parse YouTube URLs into video IDs
- **Caching** — All results cached under `~/.claude/video_cache/{video_id}/`

### Transcript Pipeline Priority

1. **YouTube subtitles** (uploaded) — fastest, highest quality
2. **YouTube auto-generated captions** — fast, decent quality
3. **faster-whisper transcription** — slower, requires audio download

### Logging

All logging uses Python's `logging` module (not `print()`). This is critical because:
- The MCP server uses stdout for JSON-RPC protocol messages
- Any `print()` to stdout would corrupt the protocol
- `logging` defaults to stderr, keeping stdout clean

The CLI entry point (`cli.py`) configures `logging.basicConfig()` to write to stderr with a simple message format.

## Cache Structure

```
~/.claude/video_cache/
  {video_id}/
    state.json          # Metadata + processing state
    audio.mp3           # Downloaded audio (if whisper was needed)
    audio.srt           # SRT transcript
    audio.txt           # Plain text transcript
    thumbnail.jpg       # Video thumbnail
    drill_{quality}/    # Frames from get_frames_at()
      drill_MM-SS.jpg
    hq/                 # Frames from get_hq_frames_at()
      hq_MM-SS.jpg
```

### state.json

```json
{
  "video_id": "dYP2V_nK8o0",
  "url": "https://youtube.com/watch?v=dYP2V_nK8o0",
  "title": "Video Title",
  "duration": 300,
  "duration_string": "5:00",
  "transcript_complete": true,
  "transcript_source": "uploaded"
}
```

## Quality Tiers

Frame extraction supports 5 quality tiers, each with different resolution and compression settings:

| Tier | Resolution | JPEG Quality | Concurrent Downloads |
|------|-----------|-------------|---------------------|
| lowest | 480px | 5 | 1 |
| low | 640px | 4 | 2 |
| medium | 854px | 3 | 4 |
| high | 1280px | 2 | 4 |
| highest | 1280px | 2 | 4 |

The `get_frames_at()` function downloads only the needed video segment (not the full video), extracts frames, then deletes the segment file.
