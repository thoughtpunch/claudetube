---
description: Resume watching a playlist from the last position
argument-hint: [playlist_id]
allowed-tools: ["Bash", "Read", "mcp__claudetube__*"]
---

# Resume Playlist

Resume watching from where you left off.

## Input: $ARGUMENTS

The argument is the **playlist_id**. If omitted, check for a recently used playlist.

## Step 1: Get Resume Position

Use the `playlist_resume_tool` MCP tool to find:
- Current video
- Timestamp (if available)
- Progress summary

## Step 2: Show Resume Info

Display where the user left off:
- Video title and position
- Timestamp if available
- Overall progress

## Output Format

```
## Resume: Python Masterclass

### Last Position
**Video**: File I/O (8 of 15)
**Timestamp**: 12:34

### Progress
[████████████████░░░░░░░░░░░░░░] 53%
Completed: 8 of 15 videos

Say "watch it" to continue from 12:34.
```

If no progress:
```
## Resume: Python Masterclass

No watch history found. Starting from the beginning.

**First Video**: Introduction to Python

Say "watch it" to start.
```

## Notes

- Show timestamp in human-readable format (MM:SS)
- Offer to continue watching
- If no history, suggest starting from video 1
