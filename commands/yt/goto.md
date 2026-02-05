---
description: Jump to a specific video in a playlist by position or title
argument-hint: <position|title> [playlist_id]
allowed-tools: ["Bash", "Read", "mcp__claudetube__*"]
---

# Jump to Video in Playlist

Navigate to a specific video by position number or title search.

## Input: $ARGUMENTS

Arguments can be:
- A number (e.g., `5`) - jump to video at position 5
- A search term (e.g., `"authentication"`) - find video with matching title
- Optionally followed by playlist_id

## Step 1: Parse Input

Determine if the first argument is:
- A number → use as position
- A string → use as title search

## Step 2: Find Video

Use the `playlist_goto_tool` MCP tool:
- `position`: 1-indexed position number
- `title_search`: partial title match

If multiple matches found, show options and ask user to pick.

## Step 3: Show Video Info

Display the found video information including:
- Title
- Position in playlist
- Current progress

## Output Format

For successful match:
```
## Video Found

**Position**: 5 of 12
**Title**: Authentication Deep Dive
**Video ID**: xyz789

### Current Progress
[████████████░░░░░░░░░░░░░░░░░░] 33%
You are at: Video 4

Ready to watch? Say "watch it" to start.
```

For multiple matches:
```
## Multiple Matches for "auth"

1. **Authentication Basics** (position 3)
2. **Authentication Deep Dive** (position 5)
3. **OAuth Authentication** (position 9)

Which one? Say "goto 5" to select.
```

## Notes

- Position is 1-indexed (first video is 1, not 0)
- Title search is case-insensitive partial match
- Show progress context to help user orient
