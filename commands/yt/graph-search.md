---
description: Find videos related to a topic across all cached videos
argument-hint: <query>
allowed-tools: ["Bash", "Read"]
---

# Find Related Videos

Search the cross-video knowledge graph for videos related to a topic, concept, or entity.

## Input: $ARGUMENTS

The argument is a **search query** (e.g., `python`, `machine learning`, `Elon Musk`).

## Step 1: Search Knowledge Graph

```bash
~/.claudetube/venv/bin/python3 << 'PYTHON'
import json
from claudetube.cache.knowledge_graph import get_knowledge_graph

query = "$ARGUMENTS".strip()
if not query:
    print("ERROR: Please provide a search query.")
    print("Usage: /yt:graph-search <query>")
    exit(1)

graph = get_knowledge_graph()
matches = graph.find_related_videos(query)

if not matches:
    print(f"No videos found matching '{query}'")
    print("")
    print("Tips:")
    print("- Try a broader search term")
    print("- Use /yt:graph-stats to see what's indexed")
    print("- Index videos with /yt:graph-index <video_id>")
else:
    print(f"Found {len(matches)} video(s) matching '{query}':")
    print("")
    for m in matches:
        match_info = f"[{m.match_type}: {m.matched_term}]"
        if m.video_title:
            print(f"- {m.video_id}: {m.video_title} {match_info}")
        else:
            print(f"- {m.video_id} {match_info}")
PYTHON
```

## Output Format

Present the results clearly:

```
## Related Videos for "[query]"

Found N video(s):

| Video ID | Title | Match Type | Matched Term |
|----------|-------|------------|--------------|
| abc123   | Tutorial... | entity | python |
| def456   | Overview... | concept | programming |
```

Include tips if no results are found.
