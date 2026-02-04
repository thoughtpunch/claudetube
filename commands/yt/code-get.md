---
description: Get cached code evolution data
argument-hint: <video_id>
allowed-tools: ["Bash", "Read"]
---

# Get Code Evolution Data

Retrieve cached code evolution data for a video. Does NOT generate new tracking -
use `/yt:code-track` for that.

## Input: $ARGUMENTS

The argument is the **video_id** (e.g., `dQw4w9WgXcQ`).

## Step 1: Validate Cache

```bash
VIDEO_ID=$(echo "$ARGUMENTS" | awk '{print $1}' | tr -d ' ')
CACHE_DIR="$HOME/.claudetube/cache/$VIDEO_ID"

if [ ! -d "$CACHE_DIR" ]; then
    echo "ERROR: Video not cached. Run /yt <url> first."
    exit 1
fi

echo "VIDEO_ID: $VIDEO_ID"
echo "CACHE_DIR: $CACHE_DIR"
```

## Step 2: Get Code Evolution

```bash
~/.claudetube/venv/bin/python3 << 'PYTHON'
import json
import sys

from claudetube.operations.code_evolution import get_code_evolution
from claudetube.config import get_cache_dir

video_id = "$ARGUMENTS".split()[0].strip()

result = get_code_evolution(
    video_id=video_id,
    output_base=get_cache_dir(),
)

if result.get("error"):
    print(f"ERROR: {result['error']}")
    print()
    print("Run /yt:code-track <video_id> to generate code evolution data.")
    sys.exit(1)

print(f"Video ID: {result.get('video_id')}")
print(f"Method: {result.get('method')}")
print(f"Total code units: {result.get('unit_count', 0)}")
print()

# Summary stats
summary = result.get("summary", {})
by_type = summary.get("by_type", {})
if by_type:
    print("By type:")
    for unit_type, count in by_type.items():
        print(f"  {unit_type}: {count}")
    print()

# Most modified
most_modified = summary.get("most_modified", [])
if most_modified:
    print("Most modified code units:")
    for item in most_modified:
        print(f"  {item['name']} ({item['unit_id']}): {item['change_count']} changes")
    print()

# List all tracked units
code_units = result.get("code_units", {})
if code_units:
    print("Tracked code units:")
    for unit_id, unit_data in code_units.items():
        snapshots = unit_data.get("snapshots", [])
        first_seen = unit_data.get("first_seen", 0)
        last_seen = unit_data.get("last_seen", 0)
        change_count = unit_data.get("change_count", 0)

        # Format timestamps
        mins_f, secs_f = divmod(int(first_seen), 60)
        mins_l, secs_l = divmod(int(last_seen), 60)

        print(f"  {unit_data['name']} ({unit_data['unit_type']})")
        print(f"    First: {mins_f}:{secs_f:02d} | Last: {mins_l}:{secs_l:02d} | Changes: {change_count}")
PYTHON
```

## Output Format

Present results clearly:

```
Video ID: abc123
Method: technical_json
Total code units: 5

By type:
  function: 3
  class: 2

Most modified code units:
  validate_token (function:validate_token): 4 changes
  UserAuth (class:UserAuth): 2 changes

Tracked code units:
  validate_token (function)
    First: 2:30 | Last: 15:45 | Changes: 4
  UserAuth (class)
    First: 3:00 | Last: 12:00 | Changes: 2
```

## Follow-up Actions

- `/yt:code-track <video_id>` - Generate code evolution data (if not cached)
- `/yt:code-query <video_id> <query>` - Search for specific code units
