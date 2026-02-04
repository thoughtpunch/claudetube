---
description: Get cached narrative structure for a video
argument-hint: <video_id>
allowed-tools: ["Bash", "Read"]
---

# Get Cached Narrative Structure

Retrieve the cached narrative structure for a video. Does NOT generate new analysis - use `/yt narrative` for that.

## Input: $ARGUMENTS

The argument is the **video_id** (e.g., `dQw4w9WgXcQ`).

## Step 1: Check Cache

```bash
VIDEO_ID="$ARGUMENTS"
VIDEO_ID=$(echo "$VIDEO_ID" | tr -d ' ')
CACHE_DIR="$HOME/.claudetube/cache/$VIDEO_ID"

if [ -d "$CACHE_DIR" ]; then
    if [ -f "$CACHE_DIR/structure/narrative.json" ]; then
        echo "NARRATIVE_FILE: $CACHE_DIR/structure/narrative.json"
        echo "STATUS: cached"
    else
        echo "ERROR: No narrative structure data. Run /yt narrative $VIDEO_ID first."
        exit 1
    fi
else
    echo "ERROR: Video not cached. Run /yt <url> first."
    exit 1
fi
```

## Step 2: Get Narrative Structure

```bash
~/.claudetube/venv/bin/python3 << 'PYTHON'
import json
from claudetube.config import get_cache_dir
from claudetube.operations.narrative_structure import get_narrative_structure

video_id = "$ARGUMENTS".strip()
output_base = get_cache_dir()

result = get_narrative_structure(video_id, output_base=output_base)
print(json.dumps(result, indent=2))
PYTHON
```

## Output Format

Present the narrative structure in a clear format:

```
## Narrative Structure

**Video Type**: [coding_tutorial|lecture|demo|interview|tutorial|presentation]
**Sections**: N

### Sections

| # | Label | Start | End | Duration | Scenes |
|---|-------|-------|-----|----------|--------|
| 0 | introduction | 0:00 | 1:30 | 1:30 | 0-2 |
| 1 | main_content | 1:30 | 15:00 | 13:30 | 3-10 |
| 2 | conclusion | 15:00 | 16:00 | 1:00 | 11-12 |
```

If no narrative structure exists, inform the user to run `/yt narrative <video_id>` first.
