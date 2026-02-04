---
description: List all cached playlists
argument-hint:
allowed-tools: ["Bash"]
---

# List Cached Playlists

Show all playlists that have been cached locally.

```bash
~/.claudetube/venv/bin/python3 << 'PYTHON'
from claudetube.operations.playlist import list_cached_playlists

playlists = list_cached_playlists()

if not playlists:
    print("No cached playlists found.")
    print()
    print("Use /yt:playlist <url> to extract a playlist.")
else:
    print(f"Found {len(playlists)} cached playlist(s):")
    print()
    for p in playlists:
        playlist_type = p.get('inferred_type', 'unknown')
        video_count = p.get('video_count', 0)
        title = p.get('title', '(untitled)')
        print(f"- {p['playlist_id']} ({playlist_type}, {video_count} videos): {title}")
PYTHON
```

## Output Format

```
Found N cached playlist(s):

- PLxxx (course, 12 videos): Python Tutorial for Beginners
- PLyyy (series, 8 videos): React Deep Dive
- PLzzz (collection, 25 videos): Best Tech Talks 2024
```

## Follow-up Actions

After listing playlists, users can:
- `/yt:playlist-graph <playlist_id>` - Build knowledge graph for a playlist
- `/yt:playlist-context <video_id> <playlist_id>` - Get video context within a playlist
