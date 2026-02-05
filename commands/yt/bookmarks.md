---
description: List all bookmarks in a playlist
argument-hint: [playlist_id] [video_id]
allowed-tools: ["Bash", "Read", "mcp__claudetube__*"]
---

# List Bookmarks

Show all bookmarks saved in a playlist.

## Input: $ARGUMENTS

Arguments are optional:
- `playlist_id`: The playlist (defaults to current)
- `video_id`: Filter to a specific video

## Step 1: Get Bookmarks

Use the `playlist_bookmarks_tool` MCP tool to retrieve bookmarks.

Parameters:
- `playlist_id`: Playlist to get bookmarks from
- `video_id`: Optional filter for a specific video

## Output Format

```
## Bookmarks: Python Masterclass

### Video 5: Authentication Deep Dive
- **15:30** - "JWT token explanation"
- **22:45** - "OAuth flow diagram"

### Video 8: File I/O
- **8:15** - "Context managers tip"
- **18:00** - "Binary file handling"

### Video 11: Debugging
- **5:30** - "Breakpoint tricks"

Total: 5 bookmarks

Say "watch 5 at 15:30" to jump to a bookmark.
```

If no bookmarks:
```
## Bookmarks: Python Masterclass

No bookmarks yet.

Use /yt:bookmark "note" to add one.
```

## Notes

- Bookmarks are sorted by video position, then timestamp
- Shows timestamp in human-readable format
- Notes help identify the bookmark purpose
- Quick navigation to any bookmark
