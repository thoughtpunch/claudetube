---
description: Get current analysis status for a video
argument-hint: <video_id>
allowed-tools: ["Bash", "Read"]
---

# Get Video Analysis Status

Show what analysis has been completed for each scene of a cached video.

## Input: $ARGUMENTS

The argument is the **video_id** (e.g., `dQw4w9WgXcQ`).

## Step 1: Check Cache

```bash
VIDEO_ID="$ARGUMENTS"
VIDEO_ID=$(echo "$VIDEO_ID" | tr -d ' ')
CACHE_DIR="$HOME/.claudetube/cache/$VIDEO_ID"

if [ -d "$CACHE_DIR" ]; then
    echo "CACHE_DIR: $CACHE_DIR"
    echo "STATUS: cached"
else
    echo "ERROR: Video not cached. Run /yt <url> first."
    exit 1
fi
```

## Step 2: Get Analysis Status

```bash
~/.claudetube/venv/bin/python3 << 'PYTHON'
import json
from claudetube.config import get_cache_dir
from claudetube.operations.analysis_depth import get_analysis_status

video_id = "$ARGUMENTS".strip()
output_base = get_cache_dir()

result = get_analysis_status(video_id, output_base=output_base)
print(json.dumps(result, indent=2))
PYTHON
```

## Output Format

Present the status in a clear format:

```
## Analysis Status for [video_id]

**Max Completed Depth**: [quick|standard|deep]
**Total Scenes**: N

### Summary
- Scenes with transcript: X/N
- Scenes with visual: X/N
- Scenes with technical: X/N
- Scenes with entities: X/N

### Per-Scene Status (if requested)
| Scene | Transcript | Visual | Technical | Entities |
|-------|------------|--------|-----------|----------|
| 0     | Yes        | Yes    | No        | No       |
| 1     | Yes        | No     | No        | No       |
...
```

Keep the output concise. Only show per-scene breakdown if there are issues or if specifically requested.
