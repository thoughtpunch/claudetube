---
description: Show progress and recommendations for a playlist
argument-hint: [playlist_id]
allowed-tools: ["Bash", "Read", "mcp__claudetube__*"]
---

# Show Playlist Progress

Display progress through a playlist with visual indicators and recommendations.

## Input: $ARGUMENTS

The argument is the **playlist_id**. If omitted, check for a recently used playlist.

## Step 1: Get Progress

Use the `playlist_progress_tool` MCP tool to get:
- Completion percentage
- Visual progress bar
- Video status list
- Learning recommendations

## Output Format

```
## Playlist Progress: Python Masterclass

### Overview
[████████████████░░░░░░░░░░░░░░] 53%
**Completed**: 8 of 15 videos

### Videos
✓ 1. Introduction to Python
✓ 2. Variables and Data Types
✓ 3. Control Flow
✓ 4. Functions Basics
✓ 5. Functions Advanced
✓ 6. Object-Oriented Programming
✓ 7. Error Handling
▶ 8. File I/O (current)
○ 9. Working with APIs
○ 10. Testing
○ 11. Debugging
○ 12. Best Practices
○ 13. Project: CLI Tool
○ 14. Project: Web Scraper
○ 15. Final Review

### Recommendations
1. **Next**: Working with APIs (sequential)
2. **Goal-driven**: Testing (if you want to learn testing)
3. **Related**: Debugging (shares topics with current)
```

## Legend

- ✓ = Watched
- ▶ = Current
- ○ = Not watched

## Notes

- Show all videos with their status
- Include 2-3 recommendations based on:
  - Sequential order
  - Learning goals
  - Shared topics
- For courses, highlight prerequisites if any are missing
