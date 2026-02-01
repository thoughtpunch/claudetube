---
description: Get scene structure of a cached video
argument-hint: <video_id>
allowed-tools: ["Bash", "Read"]
---

# Get Video Scene Structure

Show the scene/chapter structure of a previously cached video.

## Input: $ARGUMENTS

The argument is the **video_id** (e.g., `dQw4w9WgXcQ`).

## Step 1: Check Cache First

```bash
VIDEO_ID="$ARGUMENTS"
VIDEO_ID=$(echo "$VIDEO_ID" | tr -d ' ')
CACHE_DIR="$HOME/.claude/video_cache/$VIDEO_ID"

if [ -d "$CACHE_DIR" ]; then
    if [ -f "$CACHE_DIR/scenes/scenes.json" ]; then
        echo "SCENES_FILE: $CACHE_DIR/scenes/scenes.json"
        echo "STATUS: cached"
    else
        echo "CACHE_DIR: $CACHE_DIR"
        echo "STATUS: needs_segmentation"
    fi
else
    echo "ERROR: Video not cached. Run /yt <url> first."
    exit 1
fi
```

## Step 2: Get Scenes

If **STATUS is cached**, read the scenes.json file directly.

If **STATUS is needs_segmentation**, run the segmentation:

```bash
~/.claudetube/venv/bin/python3 << 'PYTHON'
import json
from pathlib import Path
from claudetube.config import get_cache_dir
from claudetube.cache.scenes import has_scenes, load_scenes_data
from claudetube.analysis.pause import parse_srt_file
from claudetube.operations.segmentation import segment_video_smart

video_id = "$ARGUMENTS".strip()
cache_dir = get_cache_dir() / video_id

# Load state for metadata
state_file = cache_dir / "state.json"
state = json.loads(state_file.read_text())

# Build video_info for chapter extraction
video_info = {
    "duration": state.get("duration"),
    "description": state.get("description", ""),
}

# Load transcript segments
transcript_segments = None
srt_path = cache_dir / "audio.srt"
if srt_path.exists():
    transcript_segments = parse_srt_file(srt_path)

# Run segmentation
scenes_data = segment_video_smart(
    video_id=video_id,
    video_path=None,
    transcript_segments=transcript_segments,
    video_info=video_info,
    cache_dir=cache_dir,
    srt_path=srt_path if srt_path.exists() else None,
)

print(f"SCENES_FILE: {cache_dir}/scenes/scenes.json")
print(f"METHOD: {scenes_data.method}")
print(f"SCENE_COUNT: {len(scenes_data.scenes)}")
PYTHON
```

Then READ the scenes.json file.

## Output Format

Present the scenes in a clear format:

```
## Video Scenes

**Method**: [transcript|chapters|hybrid]
**Total Scenes**: N

| # | Start | End | Duration | Title/Summary |
|---|-------|-----|----------|---------------|
| 0 | 0:00 | 1:30 | 1:30 | Introduction... |
| 1 | 1:30 | 5:45 | 4:15 | Main topic... |
...
```

Include transcript summaries if available. Keep summaries concise (1-2 sentences).
