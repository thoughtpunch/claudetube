---
description: Get playlist metadata from a URL
argument-hint: <playlist_url>
allowed-tools: ["Bash", "Read"]
---

# Get Playlist Metadata

Extract metadata from a playlist URL (title, description, videos, type).

## Input: $ARGUMENTS

The argument is a **playlist URL** (e.g., `https://www.youtube.com/playlist?list=PLxxx`).

## Step 1: Extract and Save Metadata

```bash
~/.claudetube/venv/bin/python3 << 'PYTHON'
import json
from claudetube.operations.playlist import extract_playlist_metadata, save_playlist_metadata

playlist_url = "$ARGUMENTS".strip()

try:
    # Extract metadata from URL
    playlist_data = extract_playlist_metadata(playlist_url)

    # Save to cache
    playlist_file = save_playlist_metadata(playlist_data)

    print(f"PLAYLIST_ID: {playlist_data['playlist_id']}")
    print(f"TITLE: {playlist_data['title']}")
    print(f"CHANNEL: {playlist_data['channel']}")
    print(f"VIDEO_COUNT: {playlist_data['video_count']}")
    print(f"TYPE: {playlist_data['inferred_type']}")
    print(f"CACHE_FILE: {playlist_file}")
    print()
    print("VIDEOS:")
    for v in playlist_data['videos']:
        duration = f" ({v['duration']}s)" if v.get('duration') else ""
        print(f"  {v['position']:2d}. {v['video_id']}{duration}: {v['title']}")

except Exception as e:
    print(f"ERROR: {e}")
    exit(1)
PYTHON
```

## Output Format

Present the playlist info clearly:

```
## Playlist: [Title]

**ID**: playlist_id
**Channel**: channel_name
**Type**: course/series/conference/collection
**Videos**: N

| # | Video ID | Duration | Title |
|---|----------|----------|-------|
| 0 | abc123   | 10:30    | Introduction... |
| 1 | def456   | 15:45    | Main topic... |
...
```

## Follow-up Actions

After extracting a playlist, users can:
- `/yt:playlist-graph <playlist_id>` - Build knowledge graph for the playlist
- `/yt <video_url>` - Process a video from the playlist
- `/yt:playlists` - List all cached playlists
