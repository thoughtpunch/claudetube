---
description: Get visual descriptions for accessibility (audio description)
argument-hint: <video_id> [format:vtt|txt]
allowed-tools: ["Bash", "Read"]
---

# Get Visual Descriptions for Accessibility

Generate or retrieve audio descriptions (AD) for a video to help vision-impaired users understand visual content.

## Input: $ARGUMENTS

Parse the arguments:
- **video_id**: Video ID (e.g., `dQw4w9WgXcQ`)
- **format**: Optional output format - `vtt` (WebVTT, default) or `txt` (plain text)

## Step 1: Check Cache First

Before invoking Python, verify the video is cached:
- Check if `~/.claudetube/cache/{VIDEO_ID}/state.json` exists
- If NOT cached, tell the user to run `/yt <url>` first and stop

## Step 2: Get or Compile Descriptions

```bash
~/.claudetube/venv/bin/python3 << 'PYTHON'
import json
from pathlib import Path
from claudetube.config import get_cache_dir
from claudetube.operations.audio_description import get_scene_descriptions, compile_scene_descriptions

args = """$ARGUMENTS"""
parts = args.split()
video_id = parts[0] if parts else ''
output_format = 'vtt'
if len(parts) > 1 and parts[1].lower() in ('vtt', 'txt'):
    output_format = parts[1].lower()

output_base = get_cache_dir()
cache_dir = output_base / video_id

if not cache_dir.exists():
    print(f"ERROR: Video {video_id} not cached. Run /yt <url> first.")
    exit(1)

# Try to get cached descriptions first
result = get_scene_descriptions(video_id, output_base=output_base)

if 'error' in result:
    # No cached AD - try to compile from scenes
    print("No cached AD found, compiling from scene data...")
    result = compile_scene_descriptions(video_id, output_base=output_base)

if 'error' in result:
    print(f"ERROR: {result['error']}")
    print("Run /yt:scenes first to segment the video, then re-run this command.")
    exit(1)

print(f"STATUS: {result.get('status', 'available')}")
print(f"CUE_COUNT: {result.get('cue_count', 'unknown')}")
print(f"VTT_PATH: {result.get('vtt_path', '')}")
print(f"TXT_PATH: {result.get('txt_path', '')}")

# Output the requested format
if output_format == 'vtt' and result.get('vtt_path'):
    print(f"\n--- VTT OUTPUT ---")
    print(f"FILE: {result['vtt_path']}")
elif output_format == 'txt' and result.get('txt_path'):
    print(f"\n--- TXT OUTPUT ---")
    print(f"FILE: {result['txt_path']}")
PYTHON
```

## Step 3: Read the Output

READ the file indicated in the output above to view the audio descriptions.

## Output Format

Audio descriptions are available in two formats:

**WebVTT (`.ad.vtt`)** - Standard subtitle format with timestamps:
```
WEBVTT
Kind: descriptions
Language: en

1
00:00:00.000 --> 00:01:30.000
Introduction. Speaker at desk with laptop.
```

**Plain Text (`.ad.txt`)** - Simple timestamped descriptions:
```
[00:00] Introduction. Speaker at desk with laptop.
[01:30] Screen shows code editor with Python file.
```

Use these descriptions to understand visual content that may not be conveyed in the audio/transcript.
