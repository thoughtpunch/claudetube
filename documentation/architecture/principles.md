[← Documentation](../README.md)

# Architecture Principles

> The guiding philosophy behind claudetube's design.

## Core Principle: Cheap First, Expensive Last

claudetube is an **opinionated pipeline** optimized for **speed and minimal compute**. The goal is to return useful data to Claude as quickly as possible.

### The Fallback Hierarchy

Every operation follows this priority order:

```
1. CACHE      → Already processed? Return immediately.
2. YT-DLP     → Free metadata from source (chapters, subtitles, etc.)
3. LOCAL      → Fast local processing (ffprobe, existing files)
4. COMPUTE    → Expensive operations (Whisper, visual analysis)
```

**Never do work that's already been done. Never use expensive methods when cheap ones suffice.**

### Examples

#### Transcription
```
1. Check cache (transcript_complete in state.json)     → instant
2. Fetch YouTube subtitles (yt-dlp)                    → ~2-5s
3. Check embedded/sidecar subs (local files)           → ~1s
4. Run Whisper transcription                           → 30s-5min
```

#### Scene Detection
```
1. Check cache (scenes/scenes.json exists)             → instant
2. Use YouTube chapters (yt-dlp metadata)              → instant
3. Parse timestamps from description                   → instant
4. Analyze transcript (pauses, vocabulary shifts)      → ~1-2s
5. Run visual scene detection (PySceneDetect)          → 30s-2min
```

#### Frame Extraction
```
1. Check cache (frame file exists)                     → instant
2. Download video segment (yt-dlp)                     → ~5-15s
3. Extract frame (ffmpeg)                              → ~1s
```

## Design Rules

### 1. Cache Everything

Every expensive operation writes to cache:
- `state.json` - Processing state and metadata
- `audio.srt` / `audio.txt` - Transcripts
- `scenes/scenes.json` - Scene boundaries
- `drill/*.jpg` / `hq/*.jpg` - Extracted frames

Check cache FIRST, before any processing.

### 2. Fail Fast, Degrade Gracefully

If a cheap method fails, fall back to the next option. Never block on a single method:

```python
# Good: Fallback chain
transcript = (
    fetch_cached_transcript(video_id) or
    fetch_youtube_subtitles(url) or
    fetch_embedded_subtitles(path) or
    run_whisper(audio_path)
)

# Bad: Single method, no fallback
transcript = run_whisper(audio_path)  # Always expensive!
```

### 3. Lazy Evaluation

Don't compute things until needed:
- Frames are extracted **on demand**, not upfront
- HQ frames only when explicitly requested
- Scene analysis only when scenes are queried

### 4. Prefer yt-dlp Built-ins

yt-dlp provides free metadata. Use it before building custom solutions:

| Need | yt-dlp provides | Don't build |
|------|-----------------|-------------|
| Transcripts | `--write-subs`, `--write-auto-subs` | Custom ASR |
| Chapters | `video_info['chapters']` | Manual detection |
| Thumbnails | `--write-thumbnail` | Frame extraction |
| Metadata | `--dump-json` | Custom scraping |
| Audio | `-f ba` | Video decode + extract |

### 5. Optimize for the Common Case

Most videos have:
- YouTube subtitles (70%+)
- Chapters in metadata or description (30%+)
- Reasonable quality audio

Design for these cases. Handle edge cases, but don't optimize for them.

## Latency Budget

Target latencies for common operations:

| Operation | Target | Max Acceptable |
|-----------|--------|----------------|
| Cached transcript | <100ms | <500ms |
| YouTube subtitles | <5s | <15s |
| Whisper (tiny) | <30s | <2min |
| Whisper (small) | <1min | <5min |
| Frame extraction | <10s | <30s |
| HQ frame extraction | <20s | <1min |

If an operation exceeds these, investigate caching or cheaper alternatives.

## State Tracking

Every video has a `state.json` that tracks what's been processed:

```json
{
  "video_id": "abc123",
  "transcript_complete": true,
  "transcript_source": "youtube_subtitles",
  "whisper_model": null,
  "scenes_complete": true,
  "scenes_source": "youtube_chapters",
  "thumbnail_complete": true
}
```

Check these flags before processing. Update them after processing.

## Adding New Features

When adding new capabilities, follow this checklist:

1. **Can yt-dlp provide this?** Check first.
2. **Can we cache the result?** Always yes.
3. **What's the fallback chain?** Cheap → expensive.
4. **What's the latency target?** Define it.
5. **How do we track completion?** Add to state.json.

---

**See also**:
- [Architecture Overview](architecture.md) - Codebase structure, data flow, cache layout
