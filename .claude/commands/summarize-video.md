# Summarize YouTube Video (Fast)

Quickly process a YouTube video for AI understanding: download lowest quality, extract key frames, transcribe with Whisper tiny model.

## Arguments

$ARGUMENTS

## Instructions

Run the fast video processor:

```bash
cd ~/sites/youtube_downloader && .venv/bin/python -c "
from video_summarizer.fast import process_fast, select_key_frames
result = process_fast('$ARGUMENTS', frame_interval=30, frame_width=480, whisper_model='tiny')
if result.success:
    print(f'VIDEO_ID: {result.video_id}')
    print(f'OUTPUT_DIR: {result.output_dir}')
    print(f'FRAMES: {len(result.frames)}')
    print(f'TRANSCRIPT: {result.transcript_srt}')
    key = select_key_frames(result.frames, 5)
    print('KEY_FRAMES:')
    for f in key:
        print(f'  {f}')
else:
    print(f'ERROR: {result.error}')
"
```

After the script completes:

1. **Read the transcript** (audio.srt) for timestamped content
2. **View 3-5 key frames** to understand visual content
3. **Cross-reference** frame timestamps with transcript

Provide a **concise summary** including:
- What the video teaches/explains
- Key visual examples shown (reference frame timestamps)
- Main takeaways

The output is cached by video ID - re-running the same video returns instantly.
