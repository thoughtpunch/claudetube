# CLAUDE.md - AI Assistant Guide for claudetube

## What is claudetube?

claudetube is a video processing tool that downloads, transcribes, and extracts frames from online videos. It's designed for AI assistants (like Claude) to "watch" and understand video content.

## Architecture Principle: Cheap First, Expensive Last

claudetube is optimized for **speed and minimal compute**. Every operation follows this hierarchy:

```
1. CACHE     → Already processed? Return immediately.
2. YT-DLP    → Free metadata from source (chapters, subtitles).
3. LOCAL     → Fast local processing (ffprobe, transcript analysis).
4. COMPUTE   → Expensive operations (Whisper, visual analysis) ONLY as last resort.
```

**Examples:**
- Transcription: YouTube subtitles (free) → Whisper (expensive)
- Scenes: YouTube chapters (free) → Transcript analysis (fast) → Visual detection (expensive)
- Frames: Check cache first → Extract on demand

**Never re-process what's already cached.**

## Supported Sites

**claudetube supports 1,500+ video sites** through yt-dlp, not just YouTube. This includes:

- **YouTube** - youtube.com, youtu.be
- **Vimeo** - vimeo.com
- **Dailymotion** - dailymotion.com
- **Twitch** - twitch.tv (VODs and clips)
- **Twitter/X** - twitter.com, x.com
- **TikTok** - tiktok.com
- **Instagram** - instagram.com (reels, posts)
- **Facebook** - facebook.com, fb.watch
- **Reddit** - reddit.com (video posts)
- **Bilibili** - bilibili.com
- **Rumble** - rumble.com
- **Odysee/LBRY** - odysee.com
- And **1,500+ more** - see [full list](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md)

## When to Use claudetube

Use claudetube when a user shares a video URL and wants you to:
- Summarize or explain video content
- Answer questions about what happens in the video
- Extract specific information from a video
- Analyze code, diagrams, or text shown in the video (use HQ frames)
- Get timestamps for specific moments

## MCP Tools Available

### `process_video_tool`
Downloads and transcribes a video. Returns metadata, transcript, and file paths.

```
url: Video URL from any supported site
whisper_model: tiny (default), base, small, medium, large
```

### `get_frames`
Extract frames at a specific timestamp (480p, fast).

```
video_id_or_url: Video ID or URL
start_time: Seconds from start
duration: How many seconds to capture (default: 5)
interval: Seconds between frames (default: 1)
```

### `get_hq_frames`
Extract HIGH QUALITY frames (1280p) for reading text, code, or diagrams.

```
video_id_or_url: Video ID or URL
start_time: Seconds from start
duration: How many seconds to capture (default: 5)
interval: Seconds between frames (default: 1)
```

### `transcribe_video`
Transcribe a video's audio using Whisper. Returns cached transcript instantly if available, otherwise runs Whisper. Use `force=True` to re-transcribe with a different model.

```
video_id_or_url: Video ID or URL
whisper_model: small (default), tiny, base, medium, large
force: Re-transcribe even if cached (default: false)
```

### `get_transcript`
Get full transcript for a cached video (no 50k char limit).

```
video_id: Video ID
format: "txt" (plain) or "srt" (with timestamps)
```

### `list_cached_videos`
List all videos that have been processed and cached.

## Workflow Tips

1. **Start with `process_video_tool`** - This fetches metadata and transcript
2. **Read the transcript** - Answer most questions from the transcript alone
3. **Extract frames when needed** - Only when visual context is required
4. **Use HQ frames for text/code** - When you need to read text, code, or diagrams

## Authentication Notes

Some sites require authentication:
- Private/unlisted videos
- Age-restricted content
- Subscription-only content

If you get "Failed to fetch metadata", the video may require login credentials.

## Cache Location

Videos are cached at `~/.claude/video_cache/{video_id}/`

Structure:
```
~/.claude/video_cache/
└── {video_id}/
    ├── state.json     # Metadata
    ├── audio.mp3      # Audio track
    ├── audio.srt      # Timestamped transcript
    ├── audio.txt      # Plain text transcript
    ├── thumbnail.jpg  # Video thumbnail
    ├── drill/         # Quick frames (480p)
    └── hq/            # High-quality frames (1280p)
```
