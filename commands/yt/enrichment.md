---
description: Get statistics about cache enrichment
argument-hint: <video_id>
allowed-tools: ["Bash", "Read"]
---

# Get Enrichment Statistics

Get statistics about cache enrichment for a video.

Shows how much the cache has been enriched through interactions:
observations recorded, Q&A cached, scenes examined, etc.

## Input: $ARGUMENTS

The argument is the **video_id** (e.g., `dQw4w9WgXcQ`).

## Step 1: Validate Cache

```bash
VIDEO_ID="$ARGUMENTS"
VIDEO_ID=$(echo "$VIDEO_ID" | tr -d ' ')
CACHE_DIR="$HOME/.claudetube/cache/$VIDEO_ID"

if [ ! -d "$CACHE_DIR" ]; then
    echo "ERROR: Video not cached. Run /yt <url> first."
    exit 1
fi

echo "VIDEO_ID: $VIDEO_ID"
echo "CACHE_DIR: $CACHE_DIR"
```

## Step 2: Get Enrichment Stats

```bash
~/.claudetube/venv/bin/python3 << 'PYTHON'
import sys
sys.path.insert(0, '/Users/danielbarrett/sites/claudetube/src')

from pathlib import Path
from claudetube.cache.enrichment import get_enrichment_stats

video_id = "$ARGUMENTS".strip()
cache_dir = Path.home() / ".claudetube" / "cache" / video_id

if not cache_dir.exists():
    print(f"ERROR: Video {video_id} not cached")
    sys.exit(1)

stats = get_enrichment_stats(cache_dir)

print(f"Enrichment Stats for {video_id}:")
print()
print(f"  Observations recorded: {stats['observation_count']}")
print(f"  Q&A cached: {stats['qa_count']}")
print(f"  Scenes examined: {stats['total_scenes_examined']}")
print(f"  Scenes with boost: {stats['boosted_scenes']}")
print()

if stats['has_enrichment']:
    print("Status: ENRICHED - this video has learned context")
else:
    print("Status: FRESH - no enrichment yet")
    print()
    print("Tip: Examine frames, ask questions, and record Q&A to enrich the cache")
PYTHON
```

## Output Format

Present the statistics clearly:

```
Enrichment Stats for dQw4w9WgXcQ:

  Observations recorded: 12
  Q&A cached: 5
  Scenes examined: 8
  Scenes with boost: 6

Status: ENRICHED - this video has learned context
```

Or for a fresh video:

```
Enrichment Stats for abc123xyz:

  Observations recorded: 0
  Q&A cached: 0
  Scenes examined: 0
  Scenes with boost: 0

Status: FRESH - no enrichment yet

Tip: Examine frames, ask questions, and record Q&A to enrich the cache
```

## Follow-up Actions

After viewing enrichment stats:
- `/yt:scene-context <video_id> <scene_id>` - View context for a specific scene
- `/yt:qa-search <video_id> <query>` - Search recorded Q&A
- `/yt:qa-record <video_id> <question> <answer>` - Record new Q&A
- `/yt:watch <video_id> <question>` - Active watching (builds enrichment)
