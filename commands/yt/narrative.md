---
description: Detect narrative structure (intro, main, conclusion)
argument-hint: <video_id>
allowed-tools: ["Bash", "Read"]
---

# Detect Narrative Structure

Detect the narrative structure of a video by clustering scenes into sections (introduction, main content, conclusion) and classifying the overall video type.

## Input: $ARGUMENTS

The argument is the **video_id** (e.g., `dQw4w9WgXcQ`).

## Step 1: Check Cache

```bash
VIDEO_ID="$ARGUMENTS"
VIDEO_ID=$(echo "$VIDEO_ID" | tr -d ' ')
CACHE_DIR="$HOME/.claudetube/cache/$VIDEO_ID"

if [ -d "$CACHE_DIR" ]; then
    echo "CACHE_DIR: $CACHE_DIR"
    if [ -f "$CACHE_DIR/structure/narrative.json" ]; then
        echo "STATUS: cached"
    else
        echo "STATUS: needs_detection"
    fi
else
    echo "ERROR: Video not cached. Run /yt <url> first."
    exit 1
fi
```

## Step 2: Detect Narrative Structure

Run the detection (uses cache if available):

```bash
~/.claudetube/venv/bin/python3 << 'PYTHON'
import json
from claudetube.config import get_cache_dir
from claudetube.operations.narrative_structure import detect_narrative_structure

video_id = "$ARGUMENTS".strip()
output_base = get_cache_dir()

result = detect_narrative_structure(video_id, output_base=output_base)
print(json.dumps(result, indent=2))
PYTHON
```

## Output Format

Present the narrative structure in a clear format:

```
## Narrative Structure

**Video Type**: [coding_tutorial|lecture|demo|interview|tutorial|presentation]
**Sections**: N
**Clusters**: M

### Sections

| # | Label | Start | End | Duration | Scenes | Summary |
|---|-------|-------|-----|----------|--------|---------|
| 0 | introduction | 0:00 | 1:30 | 1:30 | 0-2 | Welcome and overview... |
| 1 | main_content | 1:30 | 15:00 | 13:30 | 3-10 | Main topic covered... |
| 2 | conclusion | 15:00 | 16:00 | 1:00 | 11-12 | Summary and call to action... |
```

Keep summaries concise (1-2 sentences). Highlight the video type and key section transitions.
