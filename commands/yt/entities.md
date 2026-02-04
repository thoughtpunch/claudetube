---
description: Extract entities (objects, people, text, concepts) from video
argument-hint: <video_id> [scene_id]
allowed-tools: ["Bash", "Read"]
---

# Extract Entities from Video

Extract structured entities from video scenes: objects, people, text, code, UI elements,
and semantic concepts. This provides machine-readable data for downstream analysis.

Follows "Cheap First, Expensive Last":
1. CACHE - Return cached entities.json instantly if available
2. SKIP - Skip scenes with minimal content
3. COMPUTE - Only call AI providers when needed

Entities-first architecture: entities are PRIMARY, visual.json is DERIVED from entities.

## Input: $ARGUMENTS

Arguments: `<video_id> [scene_id]`
- **video_id**: The video ID (e.g., `dQw4w9WgXcQ`)
- **scene_id** (optional): Specific scene index to analyze (0-based)

Example: `/yt:entities dQw4w9WgXcQ` - All scenes
Example: `/yt:entities dQw4w9WgXcQ 2` - Only scene 2

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

## Step 2: Extract Entities

```bash
~/.claudetube/venv/bin/python3 << 'PYTHON'
import json
import sys
sys.path.insert(0, '/Users/danielbarrett/sites/claudetube/src')

from claudetube.operations.entity_extraction import extract_entities_for_video
from claudetube.config.loader import get_cache_dir

args = "$ARGUMENTS".split()
video_id = args[0].strip()
scene_id = int(args[1]) if len(args) > 1 else None

result = extract_entities_for_video(
    video_id=video_id,
    scene_id=scene_id,
    force=False,
    generate_visual=True,
    output_base=get_cache_dir(),
)

if result.get("error"):
    print(f"ERROR: {result['error']}")
    sys.exit(1)

print(f"Video: {video_id}")
print(f"Processing time: {result.get('processing_time', 0):.1f}s")
print(f"Total scenes: {result.get('total_scenes', 0)}")
print(f"Scenes extracted: {result.get('scenes_extracted', 0)}")
print(f"Scenes cached: {result.get('scenes_cached', 0)}")
print(f"Scenes skipped: {result.get('scenes_skipped', 0)}")
print()

scene_results = result.get("scenes", [])
if not scene_results:
    print("No entities extracted.")
else:
    for scene in scene_results:
        sid = scene.get("scene_id", "?")
        print(f"Scene {sid}:")

        entities = scene.get("entities", [])
        by_category = {}
        for ent in entities:
            cat = ent.get("category", "other")
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(ent.get("name", "?"))

        for cat, names in by_category.items():
            print(f"  {cat}: {', '.join(names[:10])}")

        concepts = scene.get("concepts", [])
        if concepts:
            primary = [c.get("term", "?") for c in concepts if c.get("importance") == "primary"]
            secondary = [c.get("term", "?") for c in concepts if c.get("importance") == "secondary"]
            if primary:
                print(f"  Primary concepts: {', '.join(primary[:5])}")
            if secondary:
                print(f"  Secondary concepts: {', '.join(secondary[:5])}")

        print()
PYTHON
```

## Output Format

Present results clearly:

```
Video: dQw4w9WgXcQ
Processing time: 18.5s
Total scenes: 6
Scenes extracted: 5
Scenes cached: 0
Scenes skipped: 1

Scene 0:
  person: man in suit, woman presenter
  object: laptop, microphone, presentation screen
  text: "Welcome", "Agenda"
  Primary concepts: introduction, agenda overview

Scene 1:
  person: developer
  object: code editor, terminal window
  code: function definition, import statement
  ui_element: VS Code sidebar, file tree
  Primary concepts: Python, web development
  Secondary concepts: imports, module structure

Scene 2:
  object: diagram, flowchart
  text: "Architecture", "Database", "API"
  Primary concepts: system architecture
```

## Follow-up Actions

After extracting entities, users can:
- `/yt:visual <video_id>` - Get natural language descriptions
- `/yt:people <video_id>` - Track people across all scenes
- `/yt:find <video_id> <query>` - Search for specific entities or concepts
- `/yt:hq <video_id> <timestamp>` - Get HQ frames of code or text
