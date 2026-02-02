[← Documentation](../README.md)

# Quick Start

> Process your first video in minutes.

## Basic Usage

### Process a YouTube Video

```python
from claudetube import process_video

# Process a video (downloads, transcribes, caches)
result = process_video("https://youtube.com/watch?v=dQw4w9WgXcQ")

if result.success:
    print(f"Title: {result.metadata['title']}")
    print(f"Duration: {result.metadata['duration']}s")
    print(f"Transcript: {result.transcript_txt.read_text()[:200]}")
else:
    print(f"Error: {result.error}")
```

### Read the Transcript

```python
# Read from file
transcript = result.transcript_txt.read_text()
print(transcript[:500])
```

### Extract Frames

```python
from claudetube import get_frames_at, get_hq_frames_at

# Quick frames (480px) for context
frames = get_frames_at(
    result.video_id,
    start_time=60,    # 1 minute in
    duration=5,       # 5 seconds
    interval=1,       # 1 frame per second
)

for frame in frames:
    print(f"Frame: {frame}")

# HQ frames (1280px) for reading code/text
hq_frames = get_hq_frames_at(
    result.video_id,
    start_time=60,
    duration=3,
    interval=1,
)
```

## Command Line

```bash
# Process a video
python -m claudetube "https://youtube.com/watch?v=dQw4w9WgXcQ"

# With Whisper model selection
python -m claudetube "https://youtube.com/watch?v=..." --whisper-model medium
```

## With Claude (MCP)

Once configured (see [MCP Setup](mcp-setup.md)), use natural language:

> "Watch this video and explain the main concepts: https://youtube.com/watch?v=..."

Claude will:
1. Process the video (download + transcribe)
2. Read the transcript
3. Extract frames if visual context is needed
4. Answer your question

## Supported Sites

claudetube supports **1500+ sites** via yt-dlp:
- YouTube, Vimeo, Dailymotion
- Twitch (VODs and clips)
- Twitter/X, TikTok, Instagram
- And many more...

```python
# All of these work
process_video("https://youtube.com/watch?v=...")
process_video("https://vimeo.com/123456789")
process_video("https://twitter.com/user/status/...")
process_video("https://twitch.tv/videos/...")
```

## Local Files

```python
# Process a local video file
result = process_video("/path/to/video.mp4")
```

## Cache Location

Processed videos are cached at:
```
~/.claude/video_cache/{video_id}/
├── state.json      # Metadata
├── audio.mp3       # Audio track
├── audio.srt       # Timestamped transcript
├── audio.txt       # Plain text transcript
├── thumbnail.jpg   # Thumbnail
├── drill/          # Quick frames
└── hq/             # HQ frames
```

Second requests are instant—no re-downloading.

---

**Next**: [MCP Setup](mcp-setup.md) - Use with Claude Desktop/Code
