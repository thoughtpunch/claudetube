---
description: Get video context within a playlist
argument-hint: <video_id> <playlist_id>
allowed-tools: ["Bash", "Read"]
---

# Get Video Context in Playlist

Get contextual information for a video within a playlist, including prerequisites, related videos, and shared topics.

## Input: $ARGUMENTS

Arguments: `<video_id> <playlist_id>`
- **video_id**: The video ID (e.g., `dQw4w9WgXcQ`)
- **playlist_id**: The playlist ID (e.g., `PLxxx`)

Example: `/yt:playlist-context dQw4w9WgXcQ PLxxx`

## Step 1: Get Video Context

```bash
~/.claudetube/venv/bin/python3 << 'PYTHON'
import json
from claudetube.operations.knowledge_graph import get_video_context

args = "$ARGUMENTS".split()
if len(args) < 2:
    print("ERROR: Usage: /yt:playlist-context <video_id> <playlist_id>")
    exit(1)

video_id = args[0].strip()
playlist_id = args[1].strip()

context = get_video_context(video_id, playlist_id)
if not context:
    print(f"ERROR: Video '{video_id}' not found in playlist '{playlist_id}'.")
    print("Possible issues:")
    print("  - Playlist not cached: run /yt:playlist <url>")
    print("  - Knowledge graph not built: run /yt:playlist-graph <playlist_id>")
    print("  - Video not in this playlist")
    exit(1)

video = context.get('video', {})
print(f"VIDEO_ID: {video_id}")
print(f"TITLE: {video.get('title', '(unknown)')}")
print(f"PLAYLIST: {context.get('playlist_title', '(unknown)')}")
print(f"TYPE: {context.get('playlist_type', 'unknown')}")
print(f"POSITION: {context.get('position', 0) + 1} of {context.get('total_videos', 0)}")
print()

# Navigation
if context.get('previous'):
    print(f"PREVIOUS: {context['previous']} - {context.get('previous_title', '')}")
if context.get('next'):
    print(f"NEXT: {context['next']} - {context.get('next_title', '')}")
print()

# Prerequisites
prereqs = context.get('prerequisite_titles', [])
if prereqs:
    print(f"PREREQUISITES ({len(prereqs)}):")
    for p in prereqs:
        print(f"  - {p['video_id']}: {p.get('title', '(unknown)')}")
else:
    print("PREREQUISITES: None (this is a starting point)")
print()

# Common topics
topics = context.get('common_topics', [])
if topics:
    print(f"RELATED_TOPICS ({len(topics)}):")
    for t in topics[:5]:
        print(f"  - {t['keyword']}")

# Shared entities
entities = context.get('shared_entities', [])
if entities:
    print()
    print(f"SHARED_ENTITIES ({len(entities)}):")
    for e in entities[:5]:
        print(f"  - {e['text']} ({e['type']})")
PYTHON
```

## Output Format

```
## Video Context

**Video**: [Title]
**Playlist**: [Playlist Title] (course)
**Position**: 3 of 12

### Navigation
- Previous: abc123 - Introduction to Python
- Next: xyz789 - Advanced Functions

### Prerequisites (2)
1. abc123: Introduction to Python
2. def456: Basic Syntax

### Related Topics
- python
- functions
- variables

### Shared Entities
- python (technology)
- function (concept)
```

## Follow-up Actions

After getting video context:
- `/yt <prerequisite_video_id>` - Process a prerequisite video
- `/yt:transcript <video_id>` - Read the video transcript
- `/yt:scenes <video_id>` - View video scenes
