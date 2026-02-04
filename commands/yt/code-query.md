---
description: Query code evolution for a file or function
argument-hint: <video_id> <query>
allowed-tools: ["Bash", "Read"]
---

# Query Code Evolution

Search code evolution data for a specific file, function, class, or code pattern.

## Input: $ARGUMENTS

Arguments: `<video_id> <query>`
- **video_id**: The video ID (e.g., `dQw4w9WgXcQ`)
- **query**: Function name, class name, or code pattern to search for

Example: `/yt:code-query dQw4w9WgXcQ validate_token`

## Step 1: Validate Input

```bash
VIDEO_ID=$(echo "$ARGUMENTS" | awk '{print $1}')
QUERY=$(echo "$ARGUMENTS" | cut -d' ' -f2-)
CACHE_DIR="$HOME/.claudetube/cache/$VIDEO_ID"

if [ -z "$QUERY" ] || [ "$QUERY" = "$VIDEO_ID" ]; then
    echo "ERROR: Missing query argument."
    echo "Usage: /yt:code-query <video_id> <query>"
    exit 1
fi

if [ ! -d "$CACHE_DIR" ]; then
    echo "ERROR: Video not cached. Run /yt <url> first."
    exit 1
fi

echo "VIDEO_ID: $VIDEO_ID"
echo "QUERY: $QUERY"
echo "CACHE_DIR: $CACHE_DIR"
```

## Step 2: Query Code Evolution

```bash
~/.claudetube/venv/bin/python3 << 'PYTHON'
import json
import sys

from claudetube.operations.code_evolution import query_code_evolution
from claudetube.config import get_cache_dir

args = "$ARGUMENTS".split(maxsplit=1)
video_id = args[0].strip()
query = args[1].strip() if len(args) > 1 else ""

if not query:
    print("ERROR: No query provided")
    sys.exit(1)

result = query_code_evolution(
    video_id=video_id,
    query=query,
    output_base=get_cache_dir(),
)

if result.get("error"):
    print(f"ERROR: {result['error']}")
    print()
    print("Run /yt:code-track <video_id> to generate code evolution data first.")
    sys.exit(1)

print(f"Video ID: {result.get('video_id')}")
print(f"Query: {result.get('query')}")
print(f"Matches: {result.get('match_count', 0)}")
print()

matches = result.get("matches", [])
if not matches:
    print("No matching code units found.")
    print()
    print("Try a different search term (function name, class name, etc.)")
else:
    for unit_data in matches:
        unit_id = unit_data.get("unit_id", "")
        name = unit_data.get("name", "")
        unit_type = unit_data.get("unit_type", "")
        first_seen = unit_data.get("first_seen", 0)
        last_seen = unit_data.get("last_seen", 0)
        change_count = unit_data.get("change_count", 0)
        snapshots = unit_data.get("snapshots", [])

        # Format timestamps
        mins_f, secs_f = divmod(int(first_seen), 60)
        mins_l, secs_l = divmod(int(last_seen), 60)

        print(f"## {name} ({unit_type})")
        print(f"ID: {unit_id}")
        print(f"First seen: {mins_f}:{secs_f:02d}")
        print(f"Last seen: {mins_l}:{secs_l:02d}")
        print(f"Total changes: {change_count}")
        print()

        if snapshots:
            print("Evolution timeline:")
            for snap in snapshots:
                ts = snap.get("timestamp", 0)
                mins, secs = divmod(int(ts), 60)
                change = snap.get("change_type", "unknown")
                diff = snap.get("diff_summary", "")
                scene = snap.get("scene_id", "?")
                lang = snap.get("language", "")

                detail = f" ({diff})" if diff else ""
                lang_info = f" [{lang}]" if lang else ""
                print(f"  [{mins}:{secs:02d}] Scene {scene}: {change}{detail}{lang_info}")
            print()

            # Show latest code content (truncated)
            latest = snapshots[-1]
            content = latest.get("content", "")
            if content:
                print("Latest code:")
                print("```")
                lines = content.split('\n')
                if len(lines) > 20:
                    for line in lines[:20]:
                        print(line)
                    print(f"... ({len(lines) - 20} more lines)")
                else:
                    print(content)
                print("```")
        print()
PYTHON
```

## Output Format

Present results clearly:

```
Video ID: abc123
Query: validate
Matches: 2

## validate_token (function)
ID: function:validate_token
First seen: 2:30
Last seen: 15:45
Total changes: 4

Evolution timeline:
  [2:30] Scene 1: shown [python]
  [5:00] Scene 2: modified (+5 lines)
  [8:15] Scene 4: added_lines (+3 lines)
  [12:00] Scene 6: modified (+2 lines, -1 lines)
  [15:45] Scene 8: unchanged

Latest code:
```
def validate_token(token: str) -> bool:
    if not token:
        return False
    # ... rest of code
```

## validate_input (function)
ID: function:validate_input
...
```

## Follow-up Actions

- `/yt:hq <video_id> <timestamp>` - View HQ frames at a specific timestamp
- `/yt:code-get <video_id>` - Get all cached evolution data
- `/yt:code-track <video_id>` - Re-run tracking (if needed)
