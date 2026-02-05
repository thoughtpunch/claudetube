---
description: Watch the next video in the current playlist
argument-hint: [playlist_id]
allowed-tools: ["Bash", "Read", "mcp__claudetube__*"]
---

# Watch Next Video in Playlist

Navigate to and optionally watch the next video in sequence.

## Input: $ARGUMENTS

The argument is the **playlist_id**. If omitted, check for a recently used playlist.

## Step 1: Get Next Video

Use the `playlist_next_tool` MCP tool to find the next video:

- If there are prerequisite warnings for courses/series, inform the user
- Show the video title, position, and progress

## Step 2: Watch (if requested)

If the user wants to watch the video, use `watch_video_in_playlist` with the video ID.

## Output Format

```
## Next Video

**Position**: 4 of 12
**Title**: Building Your First API
**Video ID**: abc123

### Progress
[████████░░░░░░░░░░░░░░░░░░░░░░] 25%
Completed: 3 of 12 videos

⚠️ Prerequisite Warning (for courses)
Missing: "Introduction to REST" (video 2)
Consider watching prerequisites first.

Ready to watch? Say "watch it" or "skip to it".
```

## Notes

- For courses/series, always show prerequisite warnings
- Show progress bar and completion stats
- Offer to watch the video if user confirms
