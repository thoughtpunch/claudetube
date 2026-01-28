---
description: Transcribe a video with Whisper (or return cached transcript)
argument-hint: <video_id_or_url> [model]
allowed-tools: ["Read", "Bash", "mcp__claudetube__transcribe_video"]
---

# Transcribe Video

Transcribe a video's audio using Whisper. Returns a cached transcript immediately if one exists, otherwise runs Whisper transcription.

## Input: $ARGUMENTS

The argument is a **video_id or URL**, optionally followed by a **Whisper model** (tiny/base/small/medium/large).

Examples:
- `/yt:transcribe dQw4w9WgXcQ`
- `/yt:transcribe dQw4w9WgXcQ small`
- `/yt:transcribe https://youtube.com/watch?v=dQw4w9WgXcQ medium`

## Steps

Parse the arguments: the first token is the video ID or URL, the second (optional) is the Whisper model (default: `small`).

### 1. Check cache first (fast path)

```bash
INPUT="$ARGUMENTS"
VIDEO_ID=$(echo "$INPUT" | awk '{print $1}' | tr -d ' ')
CACHE_DIR="$HOME/.claude/video_cache/$VIDEO_ID"

if [ -f "$CACHE_DIR/audio.srt" ] && [ -f "$CACHE_DIR/audio.txt" ]; then
    echo "CACHED_TRANSCRIPT_FOUND: $CACHE_DIR/audio.srt"
    echo "METADATA: $CACHE_DIR/state.json"
else
    echo "NO_CACHED_TRANSCRIPT"
fi
```

### 2. If cached transcript found

Read both files:
1. The `audio.srt` transcript file
2. The `state.json` metadata file

Report the transcript to the user. Done.

### 3. If no cached transcript

This video has no cached transcript. Use the `transcribe_video` MCP tool to run Whisper audio transcription.

Note to user: Audio transcription with Whisper is slower than subtitle-based transcripts. The tool will download audio if needed and run Whisper locally.

Call `transcribe_video` with the video ID/URL and the requested model.
