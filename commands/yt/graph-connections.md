---
description: Get videos connected by shared entities
argument-hint: <video_id>
allowed-tools: ["Bash", "Read"]
---

# Get Video Connections

Find other videos that share entities or concepts with a specific video.

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

## Step 2: Get Connections

```bash
~/.claudetube/venv/bin/python3 << 'PYTHON'
import json
from claudetube.cache.knowledge_graph import get_knowledge_graph

video_id = "$ARGUMENTS".strip()
graph = get_knowledge_graph()

# Check if video is indexed
video_node = graph.get_video(video_id)
if not video_node:
    print(f"Video {video_id} is not indexed in the knowledge graph.")
    print("")
    print("To index it, run: /yt:graph-index " + video_id)
    exit(0)

# Get connections
connections = graph.get_video_connections(video_id)

if not connections:
    print(f"No connected videos found for {video_id}")
    print("")
    print("This could mean:")
    print("- This video has unique entities/concepts")
    print("- Other videos haven't been indexed yet")
else:
    print(f"Found {len(connections)} video(s) connected to {video_id}:")
    print("")
    for conn_id in connections:
        conn_node = graph.get_video(conn_id)
        if conn_node and conn_node.title:
            print(f"- {conn_id}: {conn_node.title}")
        else:
            print(f"- {conn_id}")
PYTHON
```

## Output Format

Present the connections:

```
## Videos Connected to [video_id]

**Source Video**: [title]

Found N connected video(s):

| Video ID | Title | Shared |
|----------|-------|--------|
| abc123   | Tutorial... | python, django |
| def456   | Overview... | web development |
```

If no connections, explain why and suggest indexing more videos.
