---
description: Find moments in a video matching a query
argument-hint: <video_id> <query>
allowed-tools: ["Bash", "Read"]
---

# Find Moments in Video

Search for specific moments in a cached video using natural language.

## Input: $ARGUMENTS

Arguments: `<video_id> <query>`
- **video_id**: The video ID (e.g., `dQw4w9WgXcQ`)
- **query**: Natural language query (e.g., "when they fix the bug")

Example: `/yt:find dQw4w9WgXcQ when do they show the code`

## Step 1: Validate Cache

```bash
VIDEO_ID=$(echo "$ARGUMENTS" | awk '{print $1}')
QUERY=$(echo "$ARGUMENTS" | cut -d' ' -f2-)
CACHE_DIR="$HOME/.claude/video_cache/$VIDEO_ID"

if [ ! -d "$CACHE_DIR" ]; then
    echo "ERROR: Video not cached. Run /yt <url> first."
    exit 1
fi

if [ ! -f "$CACHE_DIR/scenes/scenes.json" ]; then
    echo "ERROR: Video has no scene data. Run /yt:scenes $VIDEO_ID first."
    exit 1
fi

echo "VIDEO_ID: $VIDEO_ID"
echo "QUERY: $QUERY"
echo "CACHE_DIR: $CACHE_DIR"
```

## Step 2: Search for Moments

```bash
~/.claudetube/venv/bin/python3 << 'PYTHON'
import json
import sys
sys.path.insert(0, '/Users/danielbarrett/sites/claudetube/src')

from claudetube.analysis.search import find_moments, format_timestamp

video_id = "$ARGUMENTS".split()[0].strip()
query = " ".join("$ARGUMENTS".split()[1:]).strip()

try:
    moments = find_moments(video_id, query, top_k=5)

    if not moments:
        print("No relevant moments found.")
    else:
        print(f"Found {len(moments)} relevant moment{'s' if len(moments) != 1 else ''}:")
        print()

        for m in moments:
            end_str = format_timestamp(m.end_time)
            print(f"{m.rank}. [{m.timestamp_str}-{end_str}] (relevance: {m.relevance:.0%})")
            print(f"   {m.preview}")
            print()

except FileNotFoundError as e:
    print(f"ERROR: {e}")
except ValueError as e:
    print(f"ERROR: {e}")
PYTHON
```

## Output Format

Present results clearly with timestamps:

```
Found 3 relevant moments:

1. [4:32-5:15] (relevance: 92%)
   "...so the issue was we weren't validating the token expiry..."

2. [12:08-12:45] (relevance: 85%)
   "...and that's how we patched the auth middleware..."
```

## Follow-up Actions

After finding moments, users can:
- `/yt:see <video_id> <timestamp>` - View frames at that moment
- `/yt:hq <video_id> <timestamp>` - HQ frames for code/text
- `/yt:transcript <video_id>` - Read full transcript
