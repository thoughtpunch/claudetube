---
description: Track people across scenes in a video
argument-hint: <video_id>
allowed-tools: ["Bash", "Read"]
---

# Track People Across Video

Identify and track distinct people across all scenes in a video. This creates a
unified view of who appears when and what they're doing.

Follows "Cheap First, Expensive Last":
1. CACHE - Return entities/people.json instantly if exists
2. VISUAL - Use visual transcript data (already generated)
3. AI_VIDEO - Use VideoAnalyzer (Gemini) for cross-scene tracking
4. AI_FRAMES - Use VisionAnalyzer for frame-by-frame analysis
5. COMPUTE - Run face_recognition only if requested

## Input: $ARGUMENTS

Arguments: `<video_id>`
- **video_id**: The video ID (e.g., `dQw4w9WgXcQ`)

Example: `/yt:people dQw4w9WgXcQ`

## Step 1: Validate Cache

```bash
VIDEO_ID=$(echo "$ARGUMENTS" | awk '{print $1}')
CACHE_DIR="$HOME/.claudetube/cache/$VIDEO_ID"

if [ ! -d "$CACHE_DIR" ]; then
    echo "ERROR: Video not cached. Run /yt <url> first."
    exit 1
fi

echo "VIDEO_ID: $VIDEO_ID"
echo "CACHE_DIR: $CACHE_DIR"
```

## Step 2: Track People

```bash
~/.claudetube/venv/bin/python3 << 'PYTHON'
import json
import sys
sys.path.insert(0, '/Users/danielbarrett/sites/claudetube/src')

from claudetube.operations.person_tracking import track_people
from claudetube.config.loader import get_cache_dir

video_id = "$ARGUMENTS".split()[0].strip()

result = track_people(
    video_id=video_id,
    force=False,
    use_face_recognition=False,
    output_base=get_cache_dir(),
)

if result.get("error"):
    print(f"ERROR: {result['error']}")
    sys.exit(1)

print(f"Video: {video_id}")
print(f"Processing time: {result.get('processing_time', 0):.1f}s")
print(f"Method: {result.get('method', 'unknown')}")
print(f"Total people: {result.get('total_people', 0)}")
print()

people = result.get("people", [])
if not people:
    print("No people identified in this video.")
else:
    for person in people:
        pid = person.get("person_id", "?")
        desc = person.get("description", "unknown person")
        print(f"{pid}: {desc}")

        appearances = person.get("appearances", [])
        if appearances:
            for app in appearances:
                scene_id = app.get("scene_id", "?")
                timestamp = app.get("timestamp", 0)
                mins, secs = divmod(int(timestamp), 60)
                action = app.get("action", "present")
                confidence = app.get("confidence", 0)
                print(f"  Scene {scene_id} [{mins}:{secs:02d}] - {action} (confidence: {confidence:.0%})")
        print()
PYTHON
```

## Output Format

Present results clearly:

```
Video: dQw4w9WgXcQ
Processing time: 8.2s
Method: ai_video
Total people: 3

person_0: man in dark suit with glasses
  Scene 0 [0:00] - speaking to camera (confidence: 95%)
  Scene 1 [1:30] - presenting slides (confidence: 90%)
  Scene 3 [5:45] - answering questions (confidence: 88%)

person_1: woman in red blouse
  Scene 2 [3:20] - typing on laptop (confidence: 92%)
  Scene 4 [8:00] - demonstrating code (confidence: 85%)

person_2: man in blue shirt
  Scene 3 [5:45] - asking question (confidence: 75%)
```

## Follow-up Actions

After tracking people, users can:
- `/yt:visual <video_id>` - Get full visual descriptions per scene
- `/yt:entities <video_id>` - Extract all entities (not just people)
- `/yt:see <video_id> <timestamp>` - View frames where a person appears
- `/yt:find <video_id> "person in blue"` - Search for specific person mentions
