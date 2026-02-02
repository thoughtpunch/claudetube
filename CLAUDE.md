# CLAUDE.md - AI Assistant Guide for claudetube

## Your Role: Product Manager & Software Architect

Your default role in this project is **Product Manager and Software Architect**. When users describe features, bugs, or improvements:

1. **Understand the requirement** - Ask clarifying questions, research the codebase
2. **Design the solution** - Consider architecture, patterns, dependencies, trade-offs
3. **Create BEADS tickets** - Every defined body of work gets tracked in `.beads/`
4. **Implement or delegate** - Execute the work or hand off to future sessions

**ALWAYS create beads tickets** for:
- New features (use `bd create --type=feature`)
- Bug fixes (use `bd create --type=bug`)
- Tasks and refactoring (use `bd create --type=task`)
- Epics with subtasks (use `bd create --type=epic`, then create child tickets)

Use `bd ready` to find available work. Use `bd sync` to save progress.

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

## MCP Tools Available (30 tools)

### Core Processing

#### `process_video_tool`
Downloads and transcribes a video. Returns metadata, transcript, and file paths.
```
url: Video URL, video ID, or local file path
whisper_model: tiny (default), base, small, medium, large
copy: For local files - copy instead of symlink (default: false)
```

#### `get_frames`
Extract frames at a specific timestamp.
```
video_id_or_url: Video ID or URL
start_time: Seconds from start
duration: How many seconds to capture (default: 5)
interval: Seconds between frames (default: 1)
quality: Quality tier - lowest/low/medium/high/highest (default: lowest)
```

#### `get_hq_frames`
Extract HIGH QUALITY frames (1280px) for reading text, code, or diagrams.
```
video_id_or_url: Video ID or URL
start_time: Seconds from start
duration: How many seconds to capture (default: 5)
interval: Seconds between frames (default: 1)
width: Frame width in pixels (default: 1280)
```

#### `transcribe_video`
Transcribe a video's audio. Returns cached transcript instantly if available.
```
video_id_or_url: Video ID or URL
whisper_model: small (default), tiny, base, medium, large
force: Re-transcribe even if cached (default: false)
provider: Override transcription provider (default: configured preference)
```

#### `get_transcript`
Get full transcript for a cached video (no 50k char limit).
```
video_id: Video ID
format: "txt" (plain) or "srt" (with timestamps)
```

#### `list_cached_videos`
List all videos that have been processed and cached.

### Scenes & Analysis

#### `get_scenes`
Get scene structure of a processed video. Uses cached scenes or runs smart segmentation.
```
video_id: Video ID
force: Re-run segmentation even if cached (default: false)
enrich: Generate visual descriptions for each scene (default: false, expensive)
```

#### `generate_visual_transcripts`
Generate visual descriptions for video scenes using vision AI.
```
video_id: Video ID
scene_id: Specific scene ID, or None for all scenes
force: Re-generate even if cached (default: false)
provider: Override vision provider
```

#### `extract_entities_tool`
Extract entities (objects, people, text, concepts) from video scenes.
```
video_id: Video ID
scene_id: Specific scene ID, or None for all scenes
force: Re-extract even if cached (default: false)
generate_visual: Generate visual.json from entities (default: true)
provider: Override AI provider
```

#### `analyze_deep_tool`
Deep analysis with OCR, entity extraction, and code detection.
```
video_id: Video ID
force: Re-run analysis even if cached (default: false)
```

#### `analyze_focus_tool`
Exhaustive frame-by-frame analysis of a specific video section.
```
video_id: Video ID
start_time: Start time in seconds
end_time: End time in seconds
force: Re-run analysis even if cached (default: false)
```

#### `get_analysis_status_tool`
Get current analysis status for a video (what's cached for each scene).
```
video_id: Video ID
```

### People & Objects

#### `track_people_tool`
Track people across scenes in a video.
```
video_id: Video ID
force: Re-generate even if cached (default: false)
use_face_recognition: Use face_recognition library for accuracy (default: false)
provider: Override vision/video provider
```

### Search

#### `find_moments_tool`
Find moments in a video matching a natural language query.
```
video_id: Video ID
query: Natural language query (e.g., "when they discuss authentication")
top_k: Maximum results (default: 5)
semantic_weight: Weight for semantic vs text matching, 0.0-1.0 (default: 0.5)
```

### Watch

#### `watch_video_tool`
Actively watch and reason about a video to answer a question. Most thorough analysis mode.
```
video_id: Video ID
question: Natural language question about the video
max_iterations: Maximum scene examinations (default: 15)
```

### Playlists

#### `get_playlist`
Extract metadata from a playlist URL (title, videos, structure).
```
playlist_url: URL to a playlist
```

#### `list_playlists`
List all cached playlists with IDs, titles, and types.

### Audio Description (Accessibility)

#### `get_descriptions`
Get visual descriptions for accessibility. Follows Cheap First, Expensive Last.
```
video_id_or_url: Video ID or URL
format: "vtt" (WebVTT) or "txt" (plain text)
regenerate: Re-generate even if cached (default: false)
```

#### `describe_moment`
Describe visual content at a specific moment for accessibility.
```
video_id_or_url: Video ID or URL
timestamp: Time in seconds to describe
context: Optional context about what the viewer is interested in
```

#### `get_accessible_transcript`
Get merged transcript with audio descriptions interspersed ([AD] tags).
```
video_id_or_url: Video ID or URL
format: "txt" (plain text) or "srt" (subtitles)
```

#### `has_audio_description`
Check if a video has audio description content available.
```
video_id_or_url: Video ID or URL
```

### Knowledge Graph

#### `find_related_videos_tool`
Find videos related to a topic across all cached videos.
```
query: Search query (e.g., "python", "machine learning")
```

#### `index_video_to_graph_tool`
Index a video's entities and concepts into the cross-video knowledge graph.
```
video_id: Video ID
force: Re-index even if already present (default: false)
```

#### `get_video_connections_tool`
Get videos connected to a specific video by shared entities/concepts.
```
video_id: Video ID
```

#### `get_knowledge_graph_stats_tool`
Get statistics about the cross-video knowledge graph.

### Enrichment (Progressive Learning)

#### `record_qa_tool`
Record a Q&A interaction for progressive learning.
```
video_id: Video ID
question: The question that was asked
answer: The answer that was given
```

#### `search_qa_history_tool`
Search for previously answered questions about a video.
```
video_id: Video ID
query: The question to search for
```

#### `get_scene_context_tool`
Get all learned context for a specific scene.
```
video_id: Video ID
scene_id: Scene index (0-based)
```

#### `get_enrichment_stats_tool`
Get statistics about cache enrichment for a video.
```
video_id: Video ID
```

### Providers

#### `list_providers_tool`
List available AI providers and their capabilities.

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
    ├── hq/            # High-quality frames (1280p)
    ├── scenes/        # Scene segmentation
    │   ├── scenes.json
    │   └── scene_NNN/ # Per-scene data
    │       ├── keyframes/
    │       ├── visual.json
    │       ├── technical.json
    │       └── entities.json
    ├── entities/      # People tracking, knowledge graph
    └── enrichment/    # Q&A history, observations
```

## AI Provider System

claudetube uses configurable AI providers for transcription, vision analysis, reasoning, and embeddings. The provider system lives in `src/claudetube/providers/`.

### Zero-Config Defaults

Without any configuration, claudetube uses free/local providers:

| Capability     | Default Provider | Notes |
|---------------|-----------------|-------|
| Transcription | whisper-local   | Local Whisper model |
| Vision        | claude-code     | Uses Claude Code's built-in vision |
| Reasoning     | claude-code     | Uses Claude Code's built-in reasoning |
| Embedding     | voyage          | Requires API key |

### Provider Architecture

- **Protocols** (`providers/base.py`): `Transcriber`, `StreamingTranscriber`, `VisionAnalyzer`, `VideoAnalyzer`, `Reasoner`, `Embedder`
- **Router** (`providers/router.py`): Selects provider based on config, handles fallbacks
- **Registry** (`providers/registry.py`): Lazy-loads provider modules
- **Config** (`providers/config.py`): YAML-based configuration with env var interpolation

### Configuration

Providers are configured in `.claudetube/config.yaml` or `~/.config/claudetube/config.yaml`:

```yaml
providers:
  openai:
    api_key: ${OPENAI_API_KEY}
  preferences:
    transcription: whisper-local
    vision: claude-code
  fallbacks:
    vision: [anthropic, openai, claude-code]
```

See `documentation/guides/configuration.md` for the full reference and `examples/` for sample configs.
