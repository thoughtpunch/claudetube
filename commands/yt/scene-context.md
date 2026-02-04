---
description: Get all learned context for a specific scene
argument-hint: <video_id> <scene_id>
allowed-tools: ["Bash", "Read"]
---

# Get Scene Context

Get all learned context for a specific scene in a video.

Returns observations, related Q&A, and relevance boost accumulated
through prior interactions. Use this when revisiting a scene to
leverage prior analysis.

## Input: $ARGUMENTS

Arguments: `<video_id> <scene_id>`
- **video_id**: The video ID (e.g., `dQw4w9WgXcQ`)
- **scene_id**: Scene index (0-based)

Example: `/yt:scene-context dQw4w9WgXcQ 3`

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
echo "SCENE_ID: $SCENE_ID"
echo "CACHE_DIR: $CACHE_DIR"
```

## Step 2: Get Scene Context

```bash
~/.claudetube/venv/bin/python3 << 'PYTHON'
import sys
sys.path.insert(0, '/Users/danielbarrett/sites/claudetube/src')

from pathlib import Path
from claudetube.cache.enrichment import get_scene_context

video_id = "$ARGUMENTS".split()[0].strip()
scene_id_str = "$ARGUMENTS".split()[1].strip() if len("$ARGUMENTS".split()) > 1 else ""

if not scene_id_str:
    print("ERROR: Need a scene_id (0-based index)")
    sys.exit(1)

try:
    scene_id = int(scene_id_str)
except ValueError:
    print(f"ERROR: Invalid scene_id: {scene_id_str}")
    sys.exit(1)

cache_dir = Path.home() / ".claudetube" / "cache" / video_id

if not cache_dir.exists():
    print(f"ERROR: Video {video_id} not cached")
    sys.exit(1)

context = get_scene_context(
    video_id=video_id,
    cache_dir=cache_dir,
    scene_id=scene_id,
)

print(f"Scene {scene_id} Context:")
print(f"  Relevance boost: {context['boost']:.2f}x")
print()

observations = context.get('observations', [])
if observations:
    print(f"Observations ({len(observations)}):")
    for obs in observations:
        obs_type = obs.get('type', 'unknown')
        content = obs.get('content', '')
        timestamp = obs.get('timestamp', '')
        print(f"  [{obs_type}] {content[:100]}{'...' if len(content) > 100 else ''}")
        if timestamp:
            print(f"    at {timestamp}")
    print()
else:
    print("No observations recorded for this scene")
    print()

related_qa = context.get('related_qa', [])
if related_qa:
    print(f"Related Q&A ({len(related_qa)}):")
    for qa in related_qa:
        q = qa.get('question', '')
        a = qa.get('answer', '')
        print(f"  Q: {q}")
        print(f"  A: {a[:150]}{'...' if len(a) > 150 else ''}")
        print()
else:
    print("No related Q&A for this scene")
PYTHON
```

## Output Format

Present the context clearly:

```
Scene 3 Context:
  Relevance boost: 1.25x

Observations (2):
  [frames_examined] Examined hq frames at 125.0s for 5.0s
    at 2024-01-15T10:30:00
  [frames_examined] Examined standard frames at 128.0s for 3.0s
    at 2024-01-15T10:32:00

Related Q&A (1):
  Q: What function is being debugged?
  A: The authenticate() function in auth.py with an off-by-one error
```

## Follow-up Actions

After viewing scene context:
- `/yt:see <video_id> <timestamp>` - View frames in this scene
- `/yt:hq <video_id> <timestamp>` - HQ frames for code/text
- `/yt:qa-record <video_id> <question> <answer>` - Add more Q&A
- `/yt:enrichment <video_id>` - View overall enrichment stats
