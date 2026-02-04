---
description: Detect changes between consecutive scenes
argument-hint: <video_id>
allowed-tools: ["Bash", "Read"]
---

# Detect Scene Changes

Detect what changed between consecutive scenes in a video: visual changes, topic shifts, and content type changes.

## Input: $ARGUMENTS

The argument is the **video_id** (e.g., `dQw4w9WgXcQ`).

## Step 1: Check Cache

```bash
VIDEO_ID="$ARGUMENTS"
VIDEO_ID=$(echo "$VIDEO_ID" | tr -d ' ')
CACHE_DIR="$HOME/.claudetube/cache/$VIDEO_ID"

if [ -d "$CACHE_DIR" ]; then
    echo "CACHE_DIR: $CACHE_DIR"
    if [ -f "$CACHE_DIR/structure/changes.json" ]; then
        echo "STATUS: cached"
    else
        echo "STATUS: needs_detection"
    fi
else
    echo "ERROR: Video not cached. Run /yt <url> first."
    exit 1
fi
```

## Step 2: Detect Scene Changes

Run the detection (uses cache if available):

```bash
~/.claudetube/venv/bin/python3 << 'PYTHON'
import json
from claudetube.config import get_cache_dir
from claudetube.operations.change_detection import detect_scene_changes

video_id = "$ARGUMENTS".strip()
output_base = get_cache_dir()

result = detect_scene_changes(video_id, output_base=output_base)
print(json.dumps(result, indent=2))
PYTHON
```

## Output Format

Present the scene changes in a clear format:

```
## Scene Change Analysis

**Total Changes**: N
**Major Transitions**: M
**Content Type Changes**: X
**Average Topic Shift**: 0.XXX

### Major Transitions

These are significant structural changes in the video:

| From Scene | To Scene | Topic Shift | Content Change |
|------------|----------|-------------|----------------|
| 2 | 3 | 0.85 | code -> slides |
| 7 | 8 | 0.72 | - |

### All Changes (summary)

- Scene 0 -> 1: topic_shift=0.23, visual: +2/-1
- Scene 1 -> 2: topic_shift=0.15, visual: +0/-0
...
```

Focus on major transitions. Only show detailed visual changes if specifically requested.
