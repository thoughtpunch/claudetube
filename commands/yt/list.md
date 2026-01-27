---
description: List all cached YouTube videos
argument-hint:
allowed-tools: ["Bash"]
---

# List Cached Videos

Show all YouTube videos that have been cached locally.

```bash
CACHE_DIR="$HOME/.claude/video_cache"

if [ -d "$CACHE_DIR" ]; then
    echo "Cached videos in $CACHE_DIR:"
    echo ""
    for dir in "$CACHE_DIR"/*/; do
        if [ -d "$dir" ]; then
            video_id=$(basename "$dir")
            state_file="$dir/state.json"
            if [ -f "$state_file" ]; then
                title=$(grep -o '"title": *"[^"]*"' "$state_file" | head -1 | sed 's/"title": *"//' | sed 's/"$//')
                duration=$(grep -o '"duration_string": *"[^"]*"' "$state_file" | head -1 | sed 's/"duration_string": *"//' | sed 's/"$//')
                playlist=$(grep -o '"playlist_id": *"[^"]*"' "$state_file" | head -1 | sed 's/"playlist_id": *"//' | sed 's/"$//')
                if [ -n "$playlist" ]; then
                    echo "- $video_id ($duration) [playlist: $playlist]: $title"
                else
                    echo "- $video_id ($duration): $title"
                fi
            else
                echo "- $video_id (no metadata)"
            fi
        fi
    done
else
    echo "No cached videos found."
fi
```

Use the video_id with:
- `/yt:see <video_id> <timestamp>` - view frames
- `/yt:hq <video_id> <timestamp>` - HQ frames for code
- `/yt:transcript <video_id>` - read transcript
