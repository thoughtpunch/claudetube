---
description: Search previously answered questions about a video
argument-hint: <video_id> <query>
allowed-tools: ["Bash", "Read"]
---

# Search Q&A History

Search for previously answered questions about a video.

This enables "second query faster than first" by returning cached
answers when the same or similar question is asked again.

## Input: $ARGUMENTS

Arguments: `<video_id> <query>`
- **video_id**: The video ID (e.g., `dQw4w9WgXcQ`)
- **query**: The question to search for

Example: `/yt:qa-search dQw4w9WgXcQ what language`

## Step 1: Validate Cache

```bash
VIDEO_ID=$(echo "$ARGUMENTS" | awk '{print $1}')
QUERY=$(echo "$ARGUMENTS" | cut -d' ' -f2-)
CACHE_DIR="$HOME/.claudetube/cache/$VIDEO_ID"

if [ ! -d "$CACHE_DIR" ]; then
    echo "ERROR: Video not cached. Run /yt <url> first."
    exit 1
fi

echo "VIDEO_ID: $VIDEO_ID"
echo "QUERY: $QUERY"
echo "CACHE_DIR: $CACHE_DIR"
```

## Step 2: Search Q&A

```bash
~/.claudetube/venv/bin/python3 << 'PYTHON'
import sys
sys.path.insert(0, '/Users/danielbarrett/sites/claudetube/src')

from pathlib import Path
from claudetube.cache.enrichment import search_cached_qa

video_id = "$ARGUMENTS".split()[0].strip()
query = " ".join("$ARGUMENTS".split()[1:]).strip()

cache_dir = Path.home() / ".claudetube" / "cache" / video_id

if not cache_dir.exists():
    print(f"ERROR: Video {video_id} not cached")
    sys.exit(1)

if not query:
    print("ERROR: Need a search query")
    sys.exit(1)

results = search_cached_qa(
    video_id=video_id,
    cache_dir=cache_dir,
    query=query,
)

if not results:
    print(f"No matching Q&A found for: {query}")
    print("\nTip: Record Q&A with /yt:qa-record to build history")
else:
    print(f"Found {len(results)} matching Q&A:")
    print()
    for i, qa in enumerate(results, 1):
        q = qa.get('question', '')
        a = qa.get('answer', '')
        scenes = qa.get('relevant_scenes', [])
        print(f"{i}. Q: {q}")
        print(f"   A: {a[:200]}{'...' if len(a) > 200 else ''}")
        if scenes:
            print(f"   Scenes: {scenes}")
        print()
PYTHON
```

## Output Format

Present results clearly:

```
Found 2 matching Q&A:

1. Q: What language is used?
   A: Python 3.10 with asyncio
   Scenes: [0, 2, 5]

2. Q: What programming language does the tutorial use?
   A: The tutorial uses Python 3.10
   Scenes: [0, 1]
```

## Follow-up Actions

After searching Q&A:
- `/yt:see <video_id> <timestamp>` - View frames at relevant timestamps
- `/yt:scene-context <video_id> <scene_id>` - View all context for a scene
- `/yt:qa-record <video_id> <question> <answer>` - Record new Q&A
