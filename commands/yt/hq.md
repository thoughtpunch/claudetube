---
description: Extract HIGH QUALITY frames (for code/text) from a cached video
argument-hint: <video_id> <timestamp> [duration]
allowed-tools: ["Bash", "Read"]
---

# High Quality Video Frames

Extract HQ frames for reading code, text, or detailed visuals.

## Input: $ARGUMENTS

Parse the arguments:
- **video_id**: YouTube video ID (e.g., `dQw4w9WgXcQ`)
- **timestamp**: Time to look at (e.g., `5:30` or `330` seconds)
- **duration**: Optional, seconds to capture (default: 10)

## Extract HQ Frames

```bash
~/.claudetube/venv/bin/python3 << 'PYTHON'
from pathlib import Path
from claudetube.core import get_hq_frames_at

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

frames = get_hq_frames_at(
    video_id,
    start_time=seconds,
    duration=duration,
    interval=5,
    output_base=Path.home() / '.claude' / 'video_cache'
)
print(f'Extracted {len(frames)} HQ frames at {timestamp}:')
for f in frames: print(f)
PYTHON
```

## View the Frames

READ each image file printed above to see the high-quality content.

Use this for:
- Reading code on screen
- Examining text, diagrams, or charts
- Any detailed visual analysis
