[← Documentation](../README.md)

# Beyond Transcripts: What Makes claudetube Different

> Most YouTube tools give you text. We give you video understanding.

## The Typical YouTube MCP

Most MCP tools for YouTube follow this pattern:

```
URL → yt-dlp → transcript → text blob → done
```

This works for podcasts, interviews, and talking-head videos. But it fails for:
- **Coding tutorials** - "This line here" means nothing without seeing the code
- **Design reviews** - Visual feedback is the entire point
- **Demonstrations** - Actions speak louder than words
- **Presentations** - Slides often contain more than the speaker says

## The claudetube Approach

We treat video as a **multi-layered document**, not a text file:

```
URL → yt-dlp → {
    metadata    → who, what, when, duration, chapters
    audio       → transcript via whisper (timestamped)
    visuals     → frames on demand (any timestamp)
    structure   → scenes, segments, boundaries
}
```

### 1. On-Demand Frame Extraction

Instead of dumping thousands of frames, claudetube extracts frames **when needed**:

```python
# Quick frames (480p) for context
get_frames(video_id, start_time=120, duration=5)

# HQ frames (1280p) for reading code/text
get_hq_frames(video_id, start_time=120, duration=5)
```

This means Claude can:
- Scan the transcript
- Identify interesting moments
- Request visual context for those specific moments
- Read code, diagrams, or text from the frames

### 2. Timestamped Transcripts

Not just text—**timed text**:

```srt
1
00:00:05,000 --> 00:00:08,500
Let me show you how the authentication flow works.

2
00:00:08,500 --> 00:00:12,000
First, the user clicks the login button here.
```

This enables:
- Jumping to specific moments
- Correlating speech with visuals
- Answering "when did they mention X?"

### 3. Multi-Source Transcripts

claudetube tries multiple transcript sources:
1. **Uploaded subtitles** - Human-written, highest quality
2. **Auto-generated captions** - YouTube's ASR
3. **Whisper transcription** - Local fallback for any video

This means it works on:
- YouTube (with or without captions)
- Vimeo, Twitch, TikTok, etc. (1500+ sites via yt-dlp)
- Local files (screen recordings, downloads)

### 4. Smart Caching

Everything is cached intelligently:

```
~/.claudetube/cache/{video_id}/
├── state.json          # Metadata, state
├── audio.mp3           # Audio track
├── audio.srt           # Timestamped transcript
├── audio.txt           # Plain text transcript
├── thumbnail.jpg       # Video thumbnail
├── drill/              # Quick frames (480p)
│   ├── drill_02-00.jpg
│   └── drill_02-01.jpg
└── hq/                 # HQ frames (1280p)
    └── hq_02-00.jpg
```

Second requests are instant. Frames are extracted on-demand, not upfront.

### 5. Progressive Enhancement

Claude can start with transcript, then drill down:

1. **First pass**: Read transcript, understand structure
2. **Second pass**: Request frames for unclear sections
3. **Third pass**: Get HQ frames for code/diagram reading
4. **Fourth pass**: Extract specific timestamps for answers

This mirrors how humans watch educational content—skim, then focus.

## The Practical Difference

**With a transcript-only tool:**
> "They mentioned something about middleware at some point, but I can't see what they actually wrote."

**With claudetube:**
> "At 3:42, they show the Express middleware configuration. Here's the code from the screen: `app.use(authMiddleware)`. They explain it handles JWT validation before routes."

## The Honest Trade-off

claudetube offers more capability than transcript-only tools, but at the cost of complexity:

| Approach | Pros | Cons |
|----------|------|------|
| **Transcript-only** | Simple, fast | Misses visual content |
| **Native video AI (Gemini)** | Single call, "just works" | YouTube-only, no caching |
| **claudetube** | Multi-site, cached, precise | Multi-step orchestration |

claudetube is a **toolkit**, not a feature. You get power and control, but you pay in ceremony. The user (or Claude) must orchestrate multiple tool calls to answer a video question.

We're working to close this gap with a streamlined single-call interface—see the [roadmap](roadmap.md).

---

**Next**: [Roadmap](roadmap.md) - Where we're going
