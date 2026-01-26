---
description: Process and analyze YouTube videos via transcript and frames
argument-hint: <url> [question]
allowed-tools: ["Bash", "Read"]
---

# claudetube

Process a YouTube video and answer questions about it.

## Input: $ARGUMENTS

## Step 1: Extract the URL

The first argument is the YouTube URL. Extract just the video ID or clean URL (ignore playlist params).

## Step 2: Process the video

Run this command with the VIDEO_URL properly quoted:

```bash
python3 << 'PYTHON'
import sys
from pathlib import Path

for p in [Path.home() / 'sites/claudetube/src', Path.home() / '.claude/plugins/claudetube/src']:
    if p.exists():
        sys.path.insert(0, str(p))
        break

from claudetube.fast import process_video

# Get just the video URL (first argument, strip playlist params)
import re
args = """$ARGUMENTS"""
url_match = re.search(r'(https?://[^\s]+)', args)
if url_match:
    url = url_match.group(1).split('&list=')[0]  # Remove playlist param
else:
    url = args.split()[0]

result = process_video(url)

if result.success:
    print('=== CLAUDETUBE READY ===')
    print(f'TRANSCRIPT: {result.transcript_srt}')
    print(f'METADATA: {result.output_dir}/state.json')
    print('=== NOW READ BOTH FILES ABOVE ===')
else:
    print(f'ERROR: {result.error}')
    sys.exit(1)
PYTHON
```

## Step 3: Read the files

After the command completes successfully:
1. Read the TRANSCRIPT file (the .srt path printed above)
2. Read the METADATA file (state.json path printed above)

## Step 4: Answer the question

Use the transcript AND metadata (title, description, tags, uploader, etc.) to answer the user's question.

## Step 5: If you need to SEE something

Only if the question requires visual context (code on screen, UI elements, game footage):

```bash
python3 << 'PYTHON'
import sys
from pathlib import Path

for p in [Path.home() / 'sites/claudetube/src', Path.home() / '.claude/plugins/claudetube/src']:
    if p.exists():
        sys.path.insert(0, str(p))
        break

from claudetube.fast import get_frames_at
from pathlib import Path

frames = get_frames_at(
    'VIDEO_ID',  # Replace with actual video ID
    start_time=SECONDS,  # Replace with timestamp in seconds
    duration=10,
    interval=2,
    output_base=Path.home() / '.claude' / 'video_cache'
)
for f in frames:
    print(f)
PYTHON
```

Then READ those image files.

For HIGH QUALITY frames (reading code/text), use `get_hq_frames_at` instead.
