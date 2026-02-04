---
description: Get statistics about the cross-video knowledge graph
argument-hint:
allowed-tools: ["Bash", "Read"]
---

# Knowledge Graph Statistics

Show statistics about the cross-video knowledge graph.

## Step 1: Get Stats

```bash
~/.claudetube/venv/bin/python3 << 'PYTHON'
import json
from claudetube.cache.knowledge_graph import get_knowledge_graph

graph = get_knowledge_graph()
stats = graph.get_stats()

print("## Knowledge Graph Statistics")
print("")
print(f"**Videos indexed**: {stats.get('video_count', 0)}")
print(f"**Entities tracked**: {stats.get('entity_count', 0)}")
print(f"**Concepts tracked**: {stats.get('concept_count', 0)}")
print("")
print(f"**Graph location**: {stats.get('graph_path', 'unknown')}")

if stats.get('video_count', 0) == 0:
    print("")
    print("The knowledge graph is empty.")
    print("")
    print("To populate it:")
    print("1. Process videos with /yt <url>")
    print("2. Extract entities with extract_entities_tool")
    print("3. Index with /yt:graph-index <video_id>")
PYTHON
```

## Output Format

The script above produces formatted output directly. Present it as-is.

If the graph is empty, the tips for populating it are included.
