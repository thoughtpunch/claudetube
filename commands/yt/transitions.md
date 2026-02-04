---
description: Get major transitions between scenes
argument-hint: <video_id>
allowed-tools: ["Bash", "Read"]
---

# Get Major Scene Transitions

Get only the major transitions between scenes - significant structural changes in the video where topics shift dramatically or content type changes.

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

## Step 2: Get Major Transitions

```bash
~/.claudetube/venv/bin/python3 << 'PYTHON'
import json
from claudetube.config import get_cache_dir
from claudetube.operations.change_detection import get_major_transitions, get_scene_changes

video_id = "$ARGUMENTS".strip()
output_base = get_cache_dir()

# Get full changes data for context
changes_data = get_scene_changes(video_id, output_base=output_base)

# Get major transition scene IDs
major_transitions = get_major_transitions(video_id, output_base=output_base)

# Build detailed output
result = {
    "video_id": video_id,
    "major_transition_scene_ids": major_transitions,
    "count": len(major_transitions),
}

# Add details for each major transition
if "changes" in changes_data:
    details = []
    for change in changes_data["changes"]:
        if change["scene_b_id"] in major_transitions:
            details.append({
                "from_scene": change["scene_a_id"],
                "to_scene": change["scene_b_id"],
                "topic_shift": change["topic_shift_score"],
                "content_change": change["content_type_change"],
                "from_type": change.get("content_type_from"),
                "to_type": change.get("content_type_to"),
            })
    result["transitions"] = details

print(json.dumps(result, indent=2))
PYTHON
```

## Output Format

Present the major transitions in a clear format:

```
## Major Scene Transitions

**Major Transitions Found**: N

| From Scene | To Scene | Topic Shift | Content Type Change |
|------------|----------|-------------|---------------------|
| 2 | 3 | 0.85 | code -> slides |
| 7 | 8 | 0.72 | slides -> presenter |
| 12 | 13 | 0.68 | - |
```

These are the key structural boundaries in the video where significant changes occur. Use this for:
- Understanding video structure at a glance
- Identifying chapter boundaries
- Finding topic transitions

If no major transitions, the video likely has consistent content throughout.
