---
description: Analyze YouTube videos - transcripts, metadata, and visual frames
argument-hint: <url> [question]
allowed-tools: ["Bash", "Read"]
---

# YouTube Video Analysis

You can now analyze YouTube videos. This tool:
1. **Fetches** transcript and metadata (cached for future questions)
2. **Extracts frames** when you need to SEE something visually
3. **Answers questions** using both audio content and visual frames

## Input: $ARGUMENTS

## Step 1: Process the Video

Extract the video URL and fetch transcript + metadata:

```bash
~/.claudetube/venv/bin/python3 << 'PYTHON'
import sys, re, json
from claudetube.core import process_video

args = """$ARGUMENTS"""
url_match = re.search(r'(https?://[^\s]+)', args)
url = url_match.group(1) if url_match else args.split()[0]

result = process_video(url)
if result.success:
    print(f'VIDEO_ID: {result.video_id}')
    if result.metadata.get('playlist_id'):
        print(f'PLAYLIST_ID: {result.metadata["playlist_id"]}')
    print(f'TRANSCRIPT: {result.transcript_srt}')
    print(f'METADATA: {result.output_dir}/state.json')
    print(f'CACHE_DIR: {result.output_dir}')
else:
    print(f'ERROR: {result.error}')
    sys.exit(1)
PYTHON
```

## Step 2: Read the Files

Read BOTH the transcript (.srt) and metadata (state.json) files printed above.

## Step 3: Answer the Question

Answer the user's question using the transcript and metadata.

---

## IMPORTANT: You Can Now "Watch" This Video

The video is **cached**. You have two powerful capabilities:

### Quick Frames (general visuals, UI, gameplay)
```bash
~/.claudetube/venv/bin/python3 << 'PYTHON'
from pathlib import Path
from claudetube.core import get_frames_at

frames = get_frames_at(
    'VIDEO_ID_HERE',      # Use the VIDEO_ID from above
    start_time=SECONDS,   # Timestamp in seconds (e.g., "5:30" = 330)
    duration=10,          # How many seconds to capture
    interval=2,           # Seconds between frames
    output_base=Path.home() / '.claude' / 'video_cache'
)
for f in frames: print(f)
PYTHON
```

### HQ Frames (code, text, diagrams - when you need to READ something)
```bash
~/.claudetube/venv/bin/python3 << 'PYTHON'
from pathlib import Path
from claudetube.core import get_hq_frames_at

frames = get_hq_frames_at(
    'VIDEO_ID_HERE',      # Use the VIDEO_ID from above
    start_time=SECONDS,   # Timestamp in seconds
    duration=10,
    interval=5,
    output_base=Path.home() / '.claude' / 'video_cache'
)
for f in frames: print(f)
PYTHON
```

After extracting frames, **READ the image files** to see them.

### When to Extract Frames

**Proactively extract frames when the user:**
- Asks to "see", "show", "look at", or "watch" something
- Asks about code, UI, visuals, or anything shown on screen
- Wants to understand something visual the speaker is demonstrating

**Quality auto-escalation:** Quick frames start at the lowest quality for speed. If frames are blurry or unreadable, re-extract at the next tier up (`lowest` → `low` → `medium` → `high` → `highest`), one step at a time. Pass the `quality` parameter to `get_frames_at()`.

**Use HQ frames when:**
- Reading code or text on screen
- Examining diagrams, charts, or detailed visuals
- The transcript mentions "as you can see here" or similar

**Use Quick frames when:**
- Getting general visual context
- Seeing gameplay, UI layouts, or demonstrations
- Understanding what's happening visually

### Timestamp Conversion
- "5:30" = 330 seconds
- "1:23:45" = 5025 seconds
- Use transcript timestamps to find the right moment

### Playlist Awareness
If the URL contains `&list=`, the playlist_id is captured in state.json.
This enables future features like navigating to other videos in the playlist.
Check the metadata for `playlist_id` to know if this video is part of a series.
