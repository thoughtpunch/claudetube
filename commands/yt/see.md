---
description: Extract frames from a cached YouTube video
argument-hint: <video_id> <timestamp> [duration]
allowed-tools: ["Bash", "Read"]
---

# See Video Frames

Extract and view frames from a previously cached YouTube video.

## Input: $ARGUMENTS

Parse the arguments:
- **video_id**: YouTube video ID (e.g., `dQw4w9WgXcQ`)
- **timestamp**: Time to look at (e.g., `5:30` or `330` seconds)
- **duration**: Optional, seconds to capture (default: 10)

## Extract Frames

```bash
~/.claudetube/venv/bin/python3 << 'PYTHON'
from pathlib import Path
from claudetube.core import get_frames_at

args = """$ARGUMENTS"""
parts = args.split()
video_id = parts[0] if parts else ''
timestamp = parts[1] if len(parts) > 1 else '0'
duration = int(parts[2]) if len(parts) > 2 else 10

# Convert timestamp like "5:30" to seconds
if ':' in timestamp:
    tparts = timestamp.split(':')
    if len(tparts) == 2:
        seconds = int(tparts[0]) * 60 + int(tparts[1])
    else:
        seconds = int(tparts[0]) * 3600 + int(tparts[1]) * 60 + int(tparts[2])
else:
    seconds = int(timestamp)

frames = get_frames_at(
    video_id,
    start_time=seconds,
    duration=duration,
    interval=2,
    output_base=Path.home() / '.claude' / 'video_cache',
    quality='lowest',
)
print(f'QUALITY: lowest')
print(f'VIDEO_ID: {video_id}')
print(f'TIMESTAMP: {timestamp} ({seconds}s)')
print(f'DURATION: {duration}s')
print(f'Extracted {len(frames)} frames:')
for f in frames: print(f)
PYTHON
```

## View the Frames

READ each image file printed above to see the video content.

## Auto-Escalation

After viewing the frames, if the content is **blurry, unreadable, or too low quality** to answer the user's question, automatically re-extract at the next quality tier. Always go up **one tier at a time**:

`lowest` → `low` → `medium` → `high` → `highest`

Use this code to escalate (replace CURRENT_QUALITY and NEXT_QUALITY):

```bash
~/.claudetube/venv/bin/python3 << 'PYTHON'
from pathlib import Path
from claudetube.core import get_frames_at

frames = get_frames_at(
    'VIDEO_ID',
    start_time=SECONDS,
    duration=DURATION,
    interval=2,
    output_base=Path.home() / '.claude' / 'video_cache',
    quality='NEXT_QUALITY',
)
print(f'QUALITY: NEXT_QUALITY')
print(f'Extracted {len(frames)} frames:')
for f in frames: print(f)
PYTHON
```

Then READ the new frames. Repeat escalation if still not clear enough.

For the highest quality (code/text), use `/yt:hq` instead.
