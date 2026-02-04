---
description: Generate visual descriptions for video scenes
argument-hint: <video_id> [scene_id]
allowed-tools: ["Bash", "Read"]
---

# Generate Visual Descriptions

Generate visual descriptions for video scenes using vision AI. This analyzes keyframes
to describe what's visually happening: people present, objects visible, actions, and
on-screen text.

Follows "Cheap First, Expensive Last":
1. CACHE - Return instantly if visual.json already exists
2. SKIP - Skip scenes where transcript provides sufficient context
3. COMPUTE - Only generate for scenes that need visual context

## Input: $ARGUMENTS

Arguments: `<video_id> [scene_id]`
- **video_id**: The video ID (e.g., `dQw4w9WgXcQ`)
- **scene_id** (optional): Specific scene index to analyze (0-based)

Example: `/yt:visual dQw4w9WgXcQ` - All scenes
Example: `/yt:visual dQw4w9WgXcQ 3` - Only scene 3

## Step 1: Validate Cache

```bash
VIDEO_ID=$(echo "$ARGUMENTS" | awk '{print $1}')
SCENE_ID=$(echo "$ARGUMENTS" | awk '{print $2}')
CACHE_DIR="$HOME/.claudetube/cache/$VIDEO_ID"

if [ ! -d "$CACHE_DIR" ]; then
    echo "ERROR: Video not cached. Run /yt <url> first."
    exit 1
fi

echo "VIDEO_ID: $VIDEO_ID"
echo "SCENE_ID: ${SCENE_ID:-all}"
echo "CACHE_DIR: $CACHE_DIR"
```

## Step 2: Generate Visual Descriptions

```bash
~/.claudetube/venv/bin/python3 << 'PYTHON'
import json
import sys
sys.path.insert(0, '/Users/danielbarrett/sites/claudetube/src')

from claudetube.operations.visual_transcript import generate_visual_transcript
from claudetube.config.loader import get_cache_dir

args = "$ARGUMENTS".split()
video_id = args[0].strip()
scene_id = int(args[1]) if len(args) > 1 else None

result = generate_visual_transcript(
    video_id=video_id,
    scene_id=scene_id,
    force=False,
    output_base=get_cache_dir(),
)

if result.get("error"):
    print(f"ERROR: {result['error']}")
    sys.exit(1)

print(f"Video: {video_id}")
print(f"Processing time: {result.get('processing_time', 0):.1f}s")
print(f"Total scenes: {result.get('total_scenes', 0)}")
print(f"Scenes generated: {result.get('scenes_generated', 0)}")
print(f"Scenes cached: {result.get('scenes_cached', 0)}")
print(f"Scenes skipped: {result.get('scenes_skipped', 0)}")
print()

descriptions = result.get("descriptions", [])
if not descriptions:
    print("No visual descriptions generated.")
else:
    for desc in descriptions:
        sid = desc.get("scene_id", "?")
        print(f"Scene {sid}:")
        print(f"  Description: {desc.get('description', 'N/A')}")
        if desc.get("people"):
            print(f"  People: {', '.join(desc['people'])}")
        if desc.get("objects"):
            print(f"  Objects: {', '.join(desc['objects'][:10])}")
        if desc.get("text_on_screen"):
            print(f"  Text: {', '.join(desc['text_on_screen'][:5])}")
        if desc.get("actions"):
            print(f"  Actions: {', '.join(desc['actions'])}")
        if desc.get("setting"):
            print(f"  Setting: {desc['setting']}")
        print()
PYTHON
```

## Output Format

Present results clearly:

```
Video: dQw4w9WgXcQ
Processing time: 12.3s
Total scenes: 5
Scenes generated: 3
Scenes cached: 1
Scenes skipped: 1

Scene 0:
  Description: A man in a suit stands at a podium...
  People: man in suit, woman in audience
  Objects: podium, microphone, slides
  Text: "Welcome to the Conference"
  Actions: presenting, gesturing
  Setting: conference room

Scene 1:
  Description: Close-up of code editor...
  Objects: code editor, terminal
  Text: "function main()", "import os"
  Setting: screen recording
```

## Follow-up Actions

After generating visual descriptions, users can:
- `/yt:entities <video_id>` - Extract structured entities from scenes
- `/yt:people <video_id>` - Track people across all scenes
- `/yt:deep <video_id>` - Full deep analysis including OCR and code detection
- `/yt:see <video_id> <timestamp>` - View frames at specific moments
