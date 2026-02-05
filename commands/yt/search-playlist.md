---
description: Search across all videos in a playlist
argument-hint: <query> [playlist_id]
allowed-tools: ["Bash", "Read", "mcp__claudetube__*"]
---

# Search Playlist

Search for content across all videos in a playlist.

## Input: $ARGUMENTS

- First argument: Search query (e.g., "authentication", "error handling")
- Second argument (optional): playlist_id

## Step 1: Run Search

Use the `search_playlist_tool` MCP tool to search transcripts across all videos.

Parameters:
- `playlist_id`: The playlist to search
- `query`: The search term
- `top_k`: Number of results (default 10)

## Step 2: Display Results

Show search results with video context and timestamps.

## Output Format

```
## Search Results for "authentication"

Found 5 matches across 3 videos:

### 1. Authentication Deep Dive (Video 5)
**Match at 3:45**: "...implement authentication using JWT tokens..."
**Relevance**: 0.92

### 2. Security Best Practices (Video 11)
**Match at 8:20**: "...authentication is the first line of defense..."
**Relevance**: 0.85

### 3. API Development (Video 7)
**Match at 15:30**: "...basic authentication vs OAuth..."
**Relevance**: 0.78

Say "goto 5" to jump to a video, or "watch 5 at 3:45" to start at a specific moment.
```

## Notes

- Results are sorted by relevance
- Show preview snippets with the match highlighted
- Include timestamps for precise navigation
- Offer quick actions to jump to results
