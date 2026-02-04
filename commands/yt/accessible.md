---
description: Get merged transcript with audio descriptions
argument-hint: <video_id>
allowed-tools: ["Bash", "Read"]
---

# Get Accessible Transcript

Get a merged transcript with audio descriptions interspersed. This combines spoken content with visual descriptions marked with `[AD]` tags for a complete accessible experience.

## Input: $ARGUMENTS

The argument is the **video_id** (e.g., `dQw4w9WgXcQ`).

## Step 1: Check Cache and Files

```bash
VIDEO_ID="$ARGUMENTS"
VIDEO_ID=$(echo "$VIDEO_ID" | tr -d ' ')
CACHE_DIR="$HOME/.claudetube/cache/$VIDEO_ID"

if [ ! -d "$CACHE_DIR" ]; then
    echo "ERROR: Video not cached. Run /yt <url> first."
    exit 1
fi

# Check for transcript
if [ -f "$CACHE_DIR/audio.txt" ]; then
    echo "TRANSCRIPT: $CACHE_DIR/audio.txt"
else
    echo "WARNING: No transcript found at $CACHE_DIR/audio.txt"
fi

# Check for audio descriptions
if [ -f "$CACHE_DIR/audio.ad.txt" ]; then
    echo "AD_FILE: $CACHE_DIR/audio.ad.txt"
    echo "STATUS: has_ad"
else
    echo "AD_FILE: none"
    echo "STATUS: no_ad"
fi

# Check for SRT with timestamps
if [ -f "$CACHE_DIR/audio.srt" ]; then
    echo "TRANSCRIPT_SRT: $CACHE_DIR/audio.srt"
fi

if [ -f "$CACHE_DIR/audio.ad.vtt" ]; then
    echo "AD_VTT: $CACHE_DIR/audio.ad.vtt"
fi
```

## Step 2: Read and Merge Content

Based on the STATUS above:

**If STATUS is `has_ad`:**
1. READ the `audio.txt` (transcript)
2. READ the `audio.ad.txt` (audio descriptions)
3. Merge them chronologically based on timestamps

**If STATUS is `no_ad`:**
- READ just the `audio.txt` transcript
- Inform the user they can run `/yt:describe <video_id>` to generate audio descriptions

## Output Format

Present the merged accessible transcript with `[AD]` markers:

```
[00:00] Welcome to today's video about Python programming.

[AD 00:15] A person sits at a desk with a laptop. Code editor visible on screen.

[00:20] Let's start by looking at the basic syntax...

[AD 00:45] Screen shows Python code: def main() with syntax highlighting.

[01:00] This function takes two parameters...
```

The `[AD]` prefix indicates visual descriptions that supplement the spoken transcript. This format helps vision-impaired users understand both what is being said and what is being shown.
