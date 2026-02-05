---
description: Bookmark current position with an optional note
argument-hint: [note] [video_id] [timestamp]
allowed-tools: ["Bash", "Read", "mcp__claudetube__*"]
---

# Add Bookmark

Save a bookmark at the current position or a specific location.

## Input: $ARGUMENTS

Arguments are optional:
- `note`: Description of the bookmark (e.g., "important concept")
- `video_id`: Specific video ID (defaults to current video)
- `timestamp`: Time in seconds (defaults to current timestamp)

## Step 1: Add Bookmark

Use the `playlist_bookmark_tool` MCP tool to save the bookmark.

Parameters:
- `playlist_id`: Current playlist
- `video_id`: Video to bookmark (or None for current)
- `timestamp`: Time in seconds (or None for current position)
- `note`: User's note

## Output Format

```
## Bookmark Added

**Video**: Authentication Deep Dive (5 of 12)
**Time**: 15:30
**Note**: "JWT token explanation"

Total bookmarks in playlist: 7

Use /yt:bookmarks to see all bookmarks.
```

## Notes

- If no video_id provided, uses current video
- If no timestamp provided, uses current position (or 0:00)
- Notes help you remember why you bookmarked
- Bookmarks persist in progress.json
