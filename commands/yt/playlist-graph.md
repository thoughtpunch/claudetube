---
description: Build knowledge graph for a playlist
argument-hint: <playlist_id>
allowed-tools: ["Bash", "Read"]
---

# Build Playlist Knowledge Graph

Build a cross-video knowledge graph for a playlist, identifying shared topics, entities, and prerequisite chains.

## Input: $ARGUMENTS

The argument is a **playlist_id** (e.g., `PLxxx`).

## Step 1: Load Playlist and Build Graph

```bash
~/.claudetube/venv/bin/python3 << 'PYTHON'
import json
from claudetube.operations.playlist import load_playlist_metadata
from claudetube.operations.knowledge_graph import build_knowledge_graph, save_knowledge_graph

playlist_id = "$ARGUMENTS".strip()

# Load playlist metadata
playlist_data = load_playlist_metadata(playlist_id)
if not playlist_data:
    print(f"ERROR: Playlist '{playlist_id}' not found in cache.")
    print("Use /yt:playlist <url> to extract the playlist first.")
    exit(1)

# Build knowledge graph
try:
    knowledge_graph = build_knowledge_graph(playlist_data)
    graph_file = save_knowledge_graph(knowledge_graph)

    print(f"PLAYLIST_ID: {playlist_id}")
    print(f"TITLE: {playlist_data['title']}")
    print(f"TYPE: {playlist_data.get('inferred_type', 'unknown')}")
    print(f"GRAPH_FILE: {graph_file}")
    print()

    # Common topics
    topics = knowledge_graph.get('common_topics', [])
    if topics:
        print(f"COMMON_TOPICS ({len(topics)}):")
        for t in topics[:10]:
            print(f"  - {t['keyword']} ({t['score']:.2f})")

    # Shared entities
    entities = knowledge_graph.get('shared_entities', [])
    if entities:
        print()
        print(f"SHARED_ENTITIES ({len(entities)}):")
        for e in entities[:10]:
            print(f"  - {e['text']} ({e['type']}, in {e['video_count']} videos)")

    # Cached videos
    cached = knowledge_graph.get('cached_videos', [])
    total = len(knowledge_graph.get('videos', []))
    print()
    print(f"CACHED_VIDEOS: {len(cached)}/{total}")

except Exception as e:
    print(f"ERROR: {e}")
    exit(1)
PYTHON
```

## Step 2: Read Full Graph (Optional)

If you need to show prerequisite chains or video relationships, read the graph file:

```bash
GRAPH_FILE="$HOME/.claudetube/cache/playlists/$ARGUMENTS/knowledge_graph.json"
echo "GRAPH_FILE: $GRAPH_FILE"
```

Then READ the knowledge_graph.json file for the full structure.

## Output Format

```
## Knowledge Graph: [Playlist Title]

**Type**: course
**Videos**: 12 (8 cached)

### Common Topics
- python (3.45)
- machine learning (2.89)
- neural network (2.12)

### Shared Entities
- python (technology, in 8 videos)
- function (concept, in 6 videos)
- tensorflow (technology, in 4 videos)

### Prerequisites
| Video | Requires | Unlocks |
|-------|----------|---------|
| 0. Intro | - | 1, 2, 3... |
| 1. Basics | 0 | 2, 3... |
...
```

## Follow-up Actions

After building a knowledge graph:
- `/yt:playlist-context <video_id> <playlist_id>` - Get context for a specific video
- `/yt <video_url>` - Process uncached videos from the playlist
