---
description: Process and analyze YouTube videos via transcript and frames
argument-hint: <url> [question]
allowed-tools: ["Bash", "Read"]
---

# claudetube

Analyze a YouTube video by reading its transcript and metadata, with the ability to "see" specific moments when visual context is needed.

## Step 1: Parse Input

Extract the YouTube URL (first argument) and any question from: $ARGUMENTS

Examples:
- `https://youtube.com/watch?v=abc123` → summarize the video
- `https://youtube.com/watch?v=abc123 how did they make the sprites?` → answer question
- `https://youtube.com/watch?v=abc123 explain the technique at 2:30` → focus on timestamp

## Step 2: Process Video

!python3 -c "
import sys
from pathlib import Path

# Try pip-installed package first, fall back to plugin directory
try:
    from claudetube.fast import process_video
except ImportError:
    for p in [
        Path.home() / '.claude/plugins/claudetube/src',
        Path.home() / 'sites/claudetube/src',
    ]:
        if p.exists():
            sys.path.insert(0, str(p))
            break
    from claudetube.fast import process_video

import json

url = '$ARGUMENTS'.split()[0]
result = process_video(url)

if result.success:
    print(json.dumps({
        'success': True,
        'video_id': result.video_id,
        'transcript_path': str(result.transcript_srt) if result.transcript_srt else None,
        'state_path': str(result.output_dir / 'state.json'),
        'output_dir': str(result.output_dir),
    }, indent=2))
else:
    print(json.dumps({
        'success': False,
        'error': result.error
    }, indent=2))
"

## Step 3: Read ALL Context

You MUST read both files to fully understand the video:

1. **Read state.json** - Contains rich metadata:
   - `title`, `description` - What the video is about
   - `uploader`, `channel` - Who made it
   - `duration`, `duration_string` - How long
   - `categories`, `tags` - Topic classification
   - `upload_date` - When published
   - `view_count`, `like_count` - Popularity metrics
   - `language` - Spoken language
   - `thumbnail` - Video thumbnail URL

2. **Read the transcript SRT file** - The actual spoken content with timestamps

Use ALL of this information when answering questions. The description and tags often contain context not mentioned in the transcript.

## Step 4: Answer the Question

Based on the metadata AND transcript:
- If answerable from the available information → answer directly
- If you need to SEE what was shown visually → proceed to Step 5

## Step 5: Drill into Frames (only when needed)

When visual context is required for a specific timestamp:

```bash
python3 -c "
import sys
from pathlib import Path

try:
    from claudetube.fast import get_frames_at
except ImportError:
    for p in [
        Path.home() / '.claude/plugins/claudetube/src',
        Path.home() / 'sites/claudetube/src',
    ]:
        if p.exists():
            sys.path.insert(0, str(p))
            break
    from claudetube.fast import get_frames_at

import json

frames = get_frames_at(
    'VIDEO_ID',           # Replace with actual video ID
    start_time=SECONDS,   # e.g., 150 for 2:30
    duration=10,
    interval=2,
    output_base=Path.home() / '.claude' / 'video_cache',
)
print(json.dumps({
    'success': True,
    'frames': [str(f) for f in frames]
}, indent=2))
"
```

Then READ those frame images to see what was displayed.

### When to drill into frames:
- "show me the UI" → need frames
- "what code did they write?" → need frames
- "what game was shown?" → need frames
- "explain the concept discussed" → transcript enough
- "what did they say about X?" → transcript enough

## Step 6: High-Quality Frames (when standard frames aren't clear enough)

If you need to read text, code, or see fine details that aren't clear in the standard drill-in frames, use the HIGH QUALITY extraction:

```bash
python3 -c "
import sys
from pathlib import Path

try:
    from claudetube.fast import get_hq_frames_at
except ImportError:
    for p in [
        Path.home() / '.claude/plugins/claudetube/src',
        Path.home() / 'sites/claudetube/src',
    ]:
        if p.exists():
            sys.path.insert(0, str(p))
            break
    from claudetube.fast import get_hq_frames_at

import json

frames = get_hq_frames_at(
    'VIDEO_ID',           # Replace with actual video ID
    start_time=SECONDS,   # e.g., 150 for 2:30
    duration=5,
    interval=1,
)
print(json.dumps({
    'success': True,
    'frames': [str(f) for f in frames],
    'note': 'These are HD frames - downloads best quality video'
}, indent=2))
"
```

**Note:** HQ extraction downloads the full-quality video which takes longer and uses more bandwidth. Only use when you need to read small text or see fine details.

## Response Format

1. Video context (title, creator, duration, category/tags if relevant)
2. Direct answer using transcript AND metadata
3. Relevant timestamps from transcript
4. If you viewed frames, describe what you saw
