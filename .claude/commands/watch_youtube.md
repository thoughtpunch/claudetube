# Watch YouTube Video

Analyze a YouTube video by reading its transcript, with the ability to "see" specific moments when visual context is needed.

## Arguments

$ARGUMENTS

## Instructions

### Step 1: Parse the input

Extract the YouTube URL and any question/task from the arguments. Examples:
- `https://youtube.com/watch?v=abc123` - just summarize
- `https://youtube.com/watch?v=abc123 how did they make the sprites?` - answer question
- `https://youtube.com/watch?v=abc123 explain the technique at 2:30` - focus on timestamp

### Step 2: Process the video (or use cache)

```bash
cd ~/sites/youtube_downloader && .venv/bin/python -c "
from claudetube import process_video
result = process_video('$ARGUMENTS'.split()[0])
if result.success:
    print('VIDEO_ID:', result.video_id)
    print('TITLE:', result.metadata.get('title'))
    print('DURATION:', result.metadata.get('duration_string'))
    print('TRANSCRIPT:', result.transcript_srt)
else:
    print('ERROR:', result.error)
"
```

### Step 3: Read the transcript

Read the SRT file to understand the video content. Timestamps tell you WHEN things were said.

### Step 4: Answer the question

Based on the transcript:
- If answerable from dialogue/narration alone → answer it
- If you need to SEE what was shown → proceed to Step 5

### Step 5: Drill into frames (only when needed)

When you need visual context for a specific timestamp:

```bash
cd ~/sites/youtube_downloader && .venv/bin/python -c "
from claudetube import get_frames_at
frames = get_frames_at(
    'VIDEO_ID_HERE',
    start_time=START_SECONDS,  # e.g., 150 for 2:30
    duration=10,
    interval=2,
)
for f in frames: print(f)
"
```

Then READ those frame images to see what was displayed.

### When to drill into frames:
- "show me the UI" → need frames
- "what code did they write?" → need frames
- "what game was shown?" → need frames
- "explain the concept discussed" → transcript enough
- "what did they say about X?" → transcript enough

## Response Format

1. Brief video context (title, creator)
2. Direct answer to the question
3. Relevant timestamps
4. If you viewed frames, describe what you saw
