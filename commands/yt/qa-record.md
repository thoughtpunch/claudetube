---
description: Record a Q&A for progressive learning
argument-hint: <video_id> <question> <answer>
allowed-tools: ["Bash", "Read"]
---

# Record Q&A Interaction

Record a question-answer interaction about a video for progressive learning.

This enables "second query faster than first" - subsequent questions can benefit
from cached answers and learned context.

## Input: $ARGUMENTS

Arguments: `<video_id> <question> <answer>`
- **video_id**: The video ID (e.g., `dQw4w9WgXcQ`)
- **question**: The question that was asked (in quotes if contains spaces)
- **answer**: The answer that was given (in quotes if contains spaces)

Example: `/yt:qa-record dQw4w9WgXcQ "What language is used?" "Python 3.10"`

## Step 1: Validate Cache

```bash
VIDEO_ID=$(echo "$ARGUMENTS" | awk '{print $1}')
CACHE_DIR="$HOME/.claudetube/cache/$VIDEO_ID"

if [ ! -d "$CACHE_DIR" ]; then
    echo "ERROR: Video not cached. Run /yt <url> first."
    exit 1
fi

echo "VIDEO_ID: $VIDEO_ID"
echo "CACHE_DIR: $CACHE_DIR"
```

## Step 2: Record Q&A

```bash
~/.claudetube/venv/bin/python3 << 'PYTHON'
import sys
import shlex
sys.path.insert(0, '/Users/danielbarrett/sites/claudetube/src')

from pathlib import Path
from claudetube.cache.enrichment import record_qa_interaction

# Parse arguments - handle quoted strings
args = "$ARGUMENTS"
try:
    parts = shlex.split(args)
except ValueError:
    parts = args.split()

if len(parts) < 3:
    print("ERROR: Need <video_id> <question> <answer>")
    print("Use quotes around multi-word arguments")
    sys.exit(1)

video_id = parts[0].strip()
question = parts[1].strip()
answer = " ".join(parts[2:]).strip()

cache_dir = Path.home() / ".claudetube" / "cache" / video_id

if not cache_dir.exists():
    print(f"ERROR: Video {video_id} not cached")
    sys.exit(1)

result = record_qa_interaction(
    video_id=video_id,
    cache_dir=cache_dir,
    question=question,
    answer=answer,
)

print(f"Recorded Q&A:")
print(f"  Question: {result['question']}")
print(f"  Answer: {answer[:100]}{'...' if len(answer) > 100 else ''}")
print(f"  Relevant scenes: {result['scenes']}")
print(f"  Total Q&A cached: {result['qa_count']}")
PYTHON
```

## Output Format

Present the result clearly:

```
Recorded Q&A:
  Question: What language is used?
  Answer: Python 3.10
  Relevant scenes: [0, 2, 5]
  Total Q&A cached: 3
```

## Follow-up Actions

After recording Q&A:
- `/yt:qa-search <video_id> <query>` - Search recorded Q&A
- `/yt:scene-context <video_id> <scene_id>` - View all context for a scene
- `/yt:enrichment <video_id>` - View enrichment statistics
