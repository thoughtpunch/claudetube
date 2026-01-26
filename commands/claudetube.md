---
description: Process and analyze YouTube videos via transcript and frames
argument-hint: <url> [question]
allowed-tools: ["Bash", "Read"]
---

# claudetube

Process a YouTube video and answer questions about it.

## Input: $ARGUMENTS

## Step 1: Process the video

Run this command and WAIT for it to complete (may take 30-60 seconds for new videos):

```bash
python3 -c "
import sys
from pathlib import Path

for p in [Path.home() / 'sites/claudetube/src', Path.home() / '.claude/plugins/claudetube/src']:
    if p.exists():
        sys.path.insert(0, str(p))
        break

from claudetube.fast import process_video
import json

url = '$ARGUMENTS'.split()[0]
result = process_video(url)

if result.success:
    print('=== CLAUDETUBE READY ===')
    print(f'TRANSCRIPT: {result.transcript_srt}')
    print(f'METADATA: {result.output_dir}/state.json')
    print('=== NOW READ BOTH FILES ABOVE ===')
else:
    print(f'ERROR: {result.error}')
    sys.exit(1)
"
```

## Step 2: Read the files

After the command completes successfully:
1. Read the TRANSCRIPT file (the .srt path printed above)
2. Read the METADATA file (state.json path printed above)

## Step 3: Answer the question

Use the transcript AND metadata (title, description, tags, uploader, etc.) to answer: $ARGUMENTS

## Step 4: If you need to SEE something

Only if the question requires visual context (code on screen, UI elements, game footage):

```bash
python3 -c "
import sys
from pathlib import Path

for p in [Path.home() / 'sites/claudetube/src', Path.home() / '.claude/plugins/claudetube/src']:
    if p.exists():
        sys.path.insert(0, str(p))
        break

from claudetube.fast import get_frames_at

frames = get_frames_at('VIDEO_ID', start_time=SECONDS, duration=10, interval=2)
for f in frames: print(f)
"
```

Then READ those image files.

For HIGH QUALITY frames (reading code/text), use `get_hq_frames_at` instead.
