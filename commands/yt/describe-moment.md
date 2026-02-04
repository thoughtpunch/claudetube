---
description: Describe visual content at a specific moment
argument-hint: <video_id> <timestamp_seconds>
allowed-tools: ["Bash", "Read"]
---

# Describe a Specific Moment

Extract high-quality frames at a specific timestamp for visual description. Useful for accessibility or detailed visual analysis.

## Input: $ARGUMENTS

Parse the arguments:
- **video_id**: Video ID (e.g., `dQw4w9WgXcQ`)
- **timestamp**: Time to describe (e.g., `5:30` or `330` seconds)

## Step 1: Check Cache First

Before invoking Python, verify the video is cached:
- Check if `~/.claudetube/cache/{VIDEO_ID}/state.json` exists
- If NOT cached, tell the user to run `/yt <url>` first and stop

## Step 2: Extract HQ Frames at Timestamp

```bash
~/.claudetube/venv/bin/python3 << 'PYTHON'
from pathlib import Path
from claudetube.core import get_hq_frames_at
from claudetube.config import get_cache_dir

args = """$ARGUMENTS"""
parts = args.split()
video_id = parts[0] if parts else ''
timestamp = parts[1] if len(parts) > 1 else '0'

output_base = get_cache_dir()
cache_dir = output_base / video_id

if not cache_dir.exists():
    print(f"ERROR: Video {video_id} not cached. Run /yt <url> first.")
    exit(1)

# Convert timestamp like "5:30" to seconds
if ':' in timestamp:
    tparts = timestamp.split(':')
    if len(tparts) == 2:
        seconds = int(tparts[0]) * 60 + int(tparts[1])
    else:
        seconds = int(tparts[0]) * 3600 + int(tparts[1]) * 60 + int(tparts[2])
else:
    seconds = int(timestamp)

# Extract frames (5 second window, 1 frame per second for detailed analysis)
frames = get_hq_frames_at(
    video_id,
    start_time=seconds,
    duration=5,
    interval=1,
    output_base=output_base
)

if not frames:
    print(f"ERROR: Could not extract frames at {timestamp}")
    exit(1)

# Format timestamp for display
mins = seconds // 60
secs = seconds % 60
print(f"TIMESTAMP: {mins}:{secs:02d} ({seconds}s)")
print(f"FRAME_COUNT: {len(frames)}")
print(f"\nExtracted {len(frames)} HQ frames for visual description:")
for f in frames:
    print(f"  FRAME: {f}")
PYTHON
```

## Step 3: View and Describe Frames

READ each image file printed above to see the visual content.

Describe what you see for accessibility purposes:
- **People**: Who is visible? What are they doing?
- **Setting**: Where does this take place?
- **Objects**: What notable objects are visible?
- **Text on screen**: Any text, code, diagrams, or UI elements?
- **Actions**: What actions or movements are occurring?

Format the description as a concise 1-3 sentence audio description suitable for vision-impaired viewers.
