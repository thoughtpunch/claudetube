---
description: Read transcript from a cached YouTube video
argument-hint: <video_id>
allowed-tools: ["Read", "Bash"]
---

# Read Cached Transcript

Quickly read the transcript from an already-cached video.

## Input: $ARGUMENTS

The argument is the **video_id** (e.g., `dQw4w9WgXcQ`).

## Find and Read Files

```bash
VIDEO_ID="$ARGUMENTS"
VIDEO_ID=$(echo "$VIDEO_ID" | tr -d ' ')
CACHE_DIR="$HOME/.claude/video_cache/$VIDEO_ID"

if [ -d "$CACHE_DIR" ]; then
    echo "TRANSCRIPT: $CACHE_DIR/audio.srt"
    echo "METADATA: $CACHE_DIR/state.json"
else
    echo "ERROR: Video not cached. Run /yt <url> first."
    exit 1
fi
```

Read both files printed above:
1. The `.srt` transcript file
2. The `state.json` metadata file
