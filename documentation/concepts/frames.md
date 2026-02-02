[← Documentation](../README.md)

# Frames

> Extracting visual snapshots from video.

## Why Frames Matter

Transcripts capture what's said. Frames capture what's shown:
- Code on screen
- Diagrams and charts
- UI demonstrations
- Text overlays
- Whiteboard drawings

## Extraction Modes

### Quick Frames (Default)

Low-resolution frames for general context:
- **Resolution**: 480px width
- **Quality**: JPEG Q5 (good compression)
- **Use case**: Understanding what's happening
- **Speed**: Fast extraction

```python
frames = get_frames(video_id, start_time=120, duration=5, interval=1)
# → 5 frames at 480px, ~50KB each
```

### HQ Frames

High-resolution frames for reading text:
- **Resolution**: 1280px width
- **Quality**: JPEG Q3 (better quality)
- **Use case**: Reading code, small text, details
- **Speed**: Slower (downloads higher quality video)

```python
frames = get_hq_frames(video_id, start_time=120, duration=5, interval=1)
# → 5 frames at 1280px, ~200KB each
```

## Quality Tiers

claudetube uses a quality ladder for progressive enhancement:

| Tier | Width | JPEG Q | Typical Size | Use Case |
|------|-------|--------|--------------|----------|
| lowest | 320px | Q10 | ~20KB | Presence detection |
| low | 480px | Q8 | ~40KB | General context |
| medium | 720px | Q5 | ~80KB | Standard viewing |
| high | 1080px | Q3 | ~150KB | Detail work |
| highest | 1280px | Q2 | ~200KB | Code/text reading |

If a quality tier fails (e.g., video not available in 1080p), it falls back to the next available tier.

## The Extraction Process

```
┌──────────────┐
│  Request     │
│  frames at   │
│  t=120-125   │
└──────┬───────┘
       ↓
┌──────────────┐
│  Download    │  ← yt-dlp downloads just the segment
│  segment     │    (not the whole video)
└──────┬───────┘
       ↓
┌──────────────┐
│  Extract     │  ← ffmpeg extracts frames at interval
│  frames      │
└──────┬───────┘
       ↓
┌──────────────┐
│  Cache &     │  → ~/.claude/video_cache/{id}/drill/
│  Return      │
└──────────────┘
```

## Caching

Frames are cached by timestamp range:

```
~/.claude/video_cache/{video_id}/
├── drill/                    # Quick frames (480px)
│   ├── drill_02-00.jpg       # Frame at 2:00
│   ├── drill_02-01.jpg       # Frame at 2:01
│   └── drill_02-02.jpg       # Frame at 2:02
└── hq/                       # HQ frames (1280px)
    └── hq_02-00.jpg          # HQ frame at 2:00
```

Subsequent requests for cached frames are instant.

## API Usage

### Python

```python
from claudetube import get_frames_at, get_hq_frames_at

# Quick frames
frames = get_frames_at(
    video_id,
    start_time=120,      # 2:00
    duration=10,         # 10 seconds
    interval=2,          # Every 2 seconds
)
# → [Path("drill_02-00.jpg"), Path("drill_02-02.jpg"), ...]

# HQ frames for code reading
hq_frames = get_hq_frames_at(
    video_id,
    start_time=120,
    duration=5,
    interval=1,
)
```

### MCP Tools

```
get_frames(video_id, start_time=120, duration=5, interval=1)
get_hq_frames(video_id, start_time=120, duration=5, interval=1)
```

## Best Practices

### 1. Start with Transcript

Don't extract frames blindly. Read the transcript first to identify interesting moments:

```
[02:15] "Let me show you the code for this..."
[05:30] "Here's what the error looks like..."
[08:45] "The architecture diagram shows..."
```

### 2. Use Quick Frames First

Quick frames (480px) are sufficient for most context. Only use HQ when you need to:
- Read code
- Read small text
- Analyze fine details

### 3. Request Reasonable Ranges

Don't extract entire videos. Target specific sections:
- Bad: `duration=3600` (entire hour-long video)
- Good: `duration=10` (10-second window)

### 4. Consider Interval

More frames ≠ better understanding:
- 1 frame/second: Good for action/demonstrations
- 1 frame/5 seconds: Good for static content (slides, code)
- 1 frame/30 seconds: Good for overview

---

**See also**:
- [Transcripts](transcripts.md) - Audio complement to visuals
- [Scenes](scenes.md) - Semantic segmentation
