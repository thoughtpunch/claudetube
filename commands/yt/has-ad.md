---
description: Check if video has audio description available
argument-hint: <video_id>
allowed-tools: ["Bash", "Read"]
---

# Check for Audio Description

Quickly check if a video has audio description (AD) files available in the cache.

## Input: $ARGUMENTS

The argument is the **video_id** (e.g., `dQw4w9WgXcQ`).

## Check AD Files

```bash
VIDEO_ID="$ARGUMENTS"
VIDEO_ID=$(echo "$VIDEO_ID" | tr -d ' ')
CACHE_DIR="$HOME/.claudetube/cache/$VIDEO_ID"

if [ ! -d "$CACHE_DIR" ]; then
    echo "ERROR: Video not cached. Run /yt <url> first."
    exit 1
fi

echo "VIDEO_ID: $VIDEO_ID"
echo "CACHE_DIR: $CACHE_DIR"
echo ""

# Check for AD files
HAS_AD="false"
AD_FORMAT=""

if [ -f "$CACHE_DIR/audio.ad.vtt" ]; then
    HAS_AD="true"
    AD_FORMAT="vtt"
    VTT_SIZE=$(wc -c < "$CACHE_DIR/audio.ad.vtt" | tr -d ' ')
    echo "AD_VTT: $CACHE_DIR/audio.ad.vtt ($VTT_SIZE bytes)"
fi

if [ -f "$CACHE_DIR/audio.ad.txt" ]; then
    HAS_AD="true"
    if [ -z "$AD_FORMAT" ]; then
        AD_FORMAT="txt"
    else
        AD_FORMAT="vtt,txt"
    fi
    TXT_SIZE=$(wc -c < "$CACHE_DIR/audio.ad.txt" | tr -d ' ')
    LINES=$(wc -l < "$CACHE_DIR/audio.ad.txt" | tr -d ' ')
    echo "AD_TXT: $CACHE_DIR/audio.ad.txt ($TXT_SIZE bytes, $LINES lines)"
fi

echo ""
echo "HAS_AD: $HAS_AD"
echo "AD_FORMAT: $AD_FORMAT"

if [ "$HAS_AD" = "false" ]; then
    echo ""
    echo "No audio descriptions found."
    echo "Run /yt:describe $VIDEO_ID to generate them."
fi
```

## Result Interpretation

- **HAS_AD: true** - Audio descriptions are available. Use `/yt:describe <video_id>` to view them.
- **HAS_AD: false** - No audio descriptions yet. Run `/yt:describe <video_id>` to generate them (requires scenes to be processed first via `/yt:scenes`).

## AD File Formats

- **audio.ad.vtt** - WebVTT format with precise timestamps, suitable for video players
- **audio.ad.txt** - Plain text format with simple timestamps, easy to read
