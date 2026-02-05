---
description: Find chapters about a topic across all videos
argument-hint: <topic> [playlist_id]
allowed-tools: ["Bash", "Read", "mcp__claudetube__*"]
---

# Find Chapters by Topic

Find chapters (YouTube chapters) about a specific topic across all videos in a playlist.

## Input: $ARGUMENTS

- First argument: Topic to search (e.g., "functions", "testing", "deployment")
- Second argument (optional): playlist_id

## Step 1: Search Chapters

Use the `find_chapter_across_playlist_tool` MCP tool to find matching chapters.

Parameters:
- `playlist_id`: The playlist to search
- `topic`: The topic to find
- `top_k`: Number of results (default 10)

## Step 2: Display Matching Chapters

Show chapters organized by video with timestamps.

## Output Format

```
## Chapters about "functions"

Found 4 chapters across 2 videos:

### Video 3: Functions Basics
- **What are Functions** at 0:00 (relevance: 0.95)
- **Function Arguments** at 5:30 (relevance: 0.88)
- **Return Values** at 12:15 (relevance: 0.75)

### Video 4: Functions Advanced
- **Lambda Functions** at 0:00 (relevance: 0.90)

Say "goto 3" to jump to a video, or "watch 3 at 5:30" to start at a chapter.
```

## Notes

- Only works with videos that have YouTube chapters
- Searches chapter titles for topic matches
- Results sorted by relevance score
- Provides direct navigation to chapters
