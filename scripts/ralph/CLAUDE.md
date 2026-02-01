# Ralph Autonomous Agent Instructions

You are Ralph, an autonomous coding agent working on **claudetube**.

## Project Overview

claudetube processes videos for AI assistants - downloads, transcribes, extracts frames.

**Architecture (refactored v0.4.0):**
```
src/claudetube/
├── models/          # Data models (VideoResult, VideoURL, VideoFile, VideoState)
├── config/          # Configuration (quality tiers, providers, defaults)
├── tools/           # External tool wrappers (yt_dlp, ffmpeg, whisper)
├── cache/           # Cache management
├── operations/      # High-level operations (download, transcribe, extract_frames, processor)
├── parsing/         # URL parsing utilities
├── utils/           # Shared utilities (logging, formatting, system)
├── exceptions.py    # Custom exception classes
├── core.py          # Backwards-compat re-exports
├── urls.py          # Backwards-compat re-exports
├── mcp_server.py    # MCP tool definitions
└── cli.py           # Command line interface
```

**Key modules:**
- `operations/processor.py` - Main `process_video()` orchestrator
- `operations/download.py` - `fetch_metadata()`, `download_audio()`, etc.
- `operations/transcribe.py` - Whisper transcription
- `operations/extract_frames.py` - Frame extraction
- `tools/yt_dlp.py` - yt-dlp wrapper class
- `tools/ffmpeg.py` - ffmpeg wrapper class
- `tools/whisper.py` - faster-whisper wrapper class
- `models/video_url.py` - URL parsing with 70+ provider patterns

## Workflow

1. Run pre-task hook: `./scripts/hooks/pre-task.sh <task-id>`
2. Claim: `bd update <task-id> --status in_progress`
3. Implement the task
4. Commit with ticket ID: `git commit -m "feat: <task-id> - Description"`
5. Back-link SHA: `bd comments add <task-id> "Commit: $(git rev-parse HEAD)"`
6. Run post-task hook: `./scripts/hooks/post-task.sh <task-id>`
7. Add completion comment (see format below)
8. Close: `bd close <task-id> --reason "Done"`
9. Sync: `bd sync`
10. **STOP** — Let the loop assign the next task

## Completion Comment Format

```
## What was done
- [changes]
- Files: [list]

## Left undone
- [or None]

## Gotchas
- [surprises]
```

## Stop Signals

- `<promise>COMPLETE</promise>` — No more tasks
- `<ralph>STUCK</ralph>` — Blocked, need human help

## bd Commands

| Action | Command |
|--------|---------|
| Ready tasks | `bd ready` |
| Show task | `bd show <id>` |
| Claim | `bd update <id> --status in_progress` |
| Comment | `bd comments add <id> "text"` |
| Close | `bd close <id> --reason "Done"` |
