---
description: Get cached scene change data
argument-hint: <video_id>
allowed-tools: ["Bash", "Read"]
---

# Get Cached Scene Changes

Retrieve the cached scene change data for a video. Does NOT generate new detection - use `/yt changes` for that.

## Input: $ARGUMENTS

The argument is the **video_id** (e.g., `dQw4w9WgXcQ`).

## Step 1: Check Cache

```bash
VIDEO_ID="$ARGUMENTS"
VIDEO_ID=$(echo "$VIDEO_ID" | tr -d ' ')
CACHE_DIR="$HOME/.claudetube/cache/$VIDEO_ID"

if [ -d "$CACHE_DIR" ]; then
    if [ -f "$CACHE_DIR/structure/changes.json" ]; then
        echo "CHANGES_FILE: $CACHE_DIR/structure/changes.json"
        echo "STATUS: cached"
    else
        echo "ERROR: No scene changes data. Run /yt changes $VIDEO_ID first."
        exit 1
    fi
else
    echo "ERROR: Video not cached. Run /yt <url> first."
    exit 1
fi
```

## Step 2: Get Scene Changes

```bash
~/.claudetube/venv/bin/python3 << 'PYTHON'
import json
from claudetube.config import get_cache_dir
from claudetube.operations.change_detection import get_scene_changes

video_id = "$ARGUMENTS".strip()
output_base = get_cache_dir()

result = get_scene_changes(video_id, output_base=output_base)
print(json.dumps(result, indent=2))
PYTHON
```

## Output Format

Present the scene changes in a clear format:

```
## Scene Changes

**Total Changes**: N
**Major Transitions**: M
**Content Type Changes**: X

### Summary

| From Scene | To Scene | Topic Shift | Content Change |
|------------|----------|-------------|----------------|
| 0 | 1 | 0.23 | - |
| 1 | 2 | 0.15 | - |
...
```

If no scene changes data exists, inform the user to run `/yt changes <video_id>` first.
