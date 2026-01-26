# claudetube

**Let Claude watch YouTube videos.**

claudetube downloads YouTube videos, transcribes them with faster-whisper, and lets Claude "see" specific moments on-demand. Built for Claude Code but works with any AI.

## How It Works

1. **Download** - Fetches lowest quality video (144p) for speed
2. **Transcribe** - Uses faster-whisper with batched inference (~28s for 7min video)
3. **Cache** - Stores everything by video ID for instant re-access
4. **Drill-in** - Extract frames for specific timestamps when visual context is needed

## Quick Start

```bash
# Install
pip install claudetube

# Or from source
git clone https://github.com/yourusername/claudetube
cd claudetube
pip install -e .

# Process a video
claudetube "https://youtube.com/watch?v=VIDEO_ID"
```

## Usage with Claude Code

Use the `/watch_youtube` slash command:

```
/watch_youtube https://youtube.com/watch?v=abc123 how did they make the sprites?
```

Claude will:
1. Process the video (or use cache)
2. Read the transcript to understand content
3. If needed, extract frames to "see" specific moments
4. Answer your question

## Python API

```python
from claudetube import process_video, get_frames_at

# Process video (transcript only - fast)
result = process_video("https://youtube.com/watch?v=VIDEO_ID")
print(result.transcript_srt.read_text())

# Drill into specific timestamp for frames
frames = get_frames_at(
    "VIDEO_ID",
    start_time=120,  # 2:00
    duration=10,     # 10 seconds
)
# frames is a list of Path objects to JPG files
```

## Performance

For a 7-minute video:

| Step | Time |
|------|------|
| Metadata fetch | ~4s |
| Download (144p, 6MB) | ~20-25s |
| Audio extraction | ~5s |
| Transcription (faster-whisper, batched) | ~28s |
| **First run total** | **~60s** |
| **Cached (subsequent)** | **~0.4s** |

## Cache Structure

Videos are cached at `~/.claude/video_cache/{VIDEO_ID}/`:

```
~/.claude/video_cache/
└── dYP2V_nK8o0/
    ├── state.json     # Metadata (title, description, tags, etc.)
    ├── audio.mp3      # Extracted audio
    ├── audio.srt      # Timestamped transcript
    ├── audio.txt      # Plain text transcript
    └── drill/         # On-demand frames
```

## state.json

Rich metadata from YouTube:

```json
{
  "video_id": "dYP2V_nK8o0",
  "title": "How To Make An ISOMETRIC Game",
  "duration_string": "6:57",
  "uploader": "Tamara Makes Games",
  "description": "3 ways to make an isometric game...",
  "categories": ["Gaming"],
  "tags": ["unity", "isometric"],
  "transcript_complete": true
}
```

## Requirements

- Python 3.10+
- ffmpeg (system)
- yt-dlp (pip, included)
- faster-whisper (pip, included)

## License

MIT
