---
description: Index a video's entities into the knowledge graph
argument-hint: <video_id>
allowed-tools: ["Bash", "Read"]
---

# Index Video to Knowledge Graph

Index a video's extracted entities and concepts into the cross-video knowledge graph.

## Input: $ARGUMENTS

The argument is the **video_id** (e.g., `dQw4w9WgXcQ`).

## Step 1: Check Cache

```bash
VIDEO_ID="$ARGUMENTS"
VIDEO_ID=$(echo "$VIDEO_ID" | tr -d ' ')
CACHE_DIR="$HOME/.claudetube/cache/$VIDEO_ID"

if [ -d "$CACHE_DIR" ]; then
    echo "CACHE_DIR: $CACHE_DIR"
    echo "STATUS: cached"
else
    echo "ERROR: Video not cached. Run /yt <url> first."
    exit 1
fi
```

## Step 2: Index to Graph

```bash
~/.claudetube/venv/bin/python3 << 'PYTHON'
import json
from pathlib import Path
from claudetube.config import get_cache_dir
from claudetube.cache.knowledge_graph import index_video_to_graph

video_id = "$ARGUMENTS".strip()
cache_dir = get_cache_dir() / video_id

result = index_video_to_graph(
    video_id=video_id,
    cache_dir=cache_dir,
    force=False,
)

print(json.dumps(result, indent=2))
PYTHON
```

## Output Format

Present the indexing result:

```
## Knowledge Graph Index

**Video ID**: [video_id]
**Status**: [indexed|already_indexed|no_data|error]

If indexed:
- Entities added: N
- Concepts added: N

If no_data:
- Suggestion: Run extract_entities_tool first to extract entities from scenes.
```

If the video has no entities or concepts, suggest running entity extraction first.
