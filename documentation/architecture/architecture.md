[← Documentation](../README.md)

# Architecture

claudetube serves two interfaces from a single codebase:

1. **Claude Code slash commands** (`/yt`, `/yt:see`, `/yt:scenes`, etc.) — Markdown command files that inline-execute Python
2. **MCP server** (`claudetube-mcp`) — stdio JSON-RPC server for any MCP client (30 tools)

Both interfaces share the same operations layer.

## Codebase Structure

```
src/claudetube/
├── __init__.py              # Package exports, version
├── cli.py                   # CLI entry point (claudetube command)
├── mcp_server.py            # MCP server entry point (30 tools)
├── core.py                  # Legacy processing engine
├── exceptions.py            # Custom exceptions
├── urls.py                  # URL utilities
│
├── models/                  # Data models
│   ├── video_result.py      # VideoResult dataclass
│   ├── video_url.py         # VideoURL model
│   ├── video_file.py        # VideoFile model
│   ├── local_file.py        # LocalFile model
│   ├── state.py             # VideoState model
│   └── chapter.py           # Chapter model
│
├── config/                  # Configuration management
│   ├── loader.py            # Config file discovery and loading
│   ├── defaults.py          # Default settings
│   ├── providers.py         # Provider configuration
│   └── quality.py           # Quality tier definitions
│
├── operations/              # High-level operations (business logic)
│   ├── processor.py         # process_video, process_local_video
│   ├── download.py          # Audio/video downloading
│   ├── transcribe.py        # Transcription orchestration
│   ├── extract_frames.py    # Frame extraction (drill + HQ)
│   ├── segmentation.py      # Smart scene segmentation
│   ├── visual_transcript.py # Visual description generation
│   ├── entity_extraction.py # Entity extraction from scenes
│   ├── person_tracking.py   # People tracking across scenes
│   ├── analysis_depth.py    # Deep/focus analysis modes
│   ├── audio_description.py # Accessibility audio descriptions
│   ├── watch.py             # Intelligent video watching
│   ├── playlist.py          # Playlist metadata extraction
│   ├── knowledge_graph.py   # Cross-video knowledge graph ops
│   ├── chapters.py          # YouTube chapter extraction
│   ├── subtitles.py         # Subtitle fetching
│   ├── factory.py           # Provider factory
│   └── ...                  # Additional operations
│
├── tools/                   # External tool wrappers
│   ├── yt_dlp.py            # yt-dlp wrapper (download, metadata)
│   ├── ffmpeg.py            # ffmpeg wrapper (frame extraction)
│   ├── ffprobe.py           # ffprobe wrapper (media info)
│   └── whisper.py           # faster-whisper wrapper
│
├── cache/                   # Cache management
│   ├── manager.py           # CacheManager (state, paths, lookups)
│   ├── scenes.py            # SceneBoundary, ScenesData, load/save
│   ├── entities.py          # Entity cache
│   ├── enrichment.py        # Q&A history, observations, relevance
│   ├── knowledge_graph.py   # Cross-video knowledge graph storage
│   ├── memory.py            # Session memory
│   └── storage.py           # Low-level storage utilities
│
├── analysis/                # Analysis algorithms
│   ├── search.py            # Moment search (text + semantic)
│   ├── attention.py         # Attention modeling for scene relevance
│   ├── comprehension.py     # Comprehension verification
│   ├── embeddings.py        # Embedding generation and indexing
│   ├── visual.py            # Visual analysis
│   ├── ocr.py               # OCR text extraction
│   ├── code.py              # Code detection and extraction
│   ├── linguistic.py        # Transcript linguistic analysis
│   ├── watcher.py           # Intelligent watching strategy
│   └── ...                  # Additional analysis modules
│
├── providers/               # AI provider system (11 providers)
│   ├── base.py              # Protocols: Transcriber, StreamingTranscriber,
│   │                        #   VisionAnalyzer, VideoAnalyzer, Reasoner, Embedder
│   ├── router.py            # Provider selection + fallback chains
│   ├── registry.py          # Lazy-load provider modules
│   ├── config.py            # YAML config with env var interpolation
│   ├── capabilities.py      # Provider capability declarations
│   ├── anthropic/           # Anthropic Claude provider
│   ├── openai/              # OpenAI provider (GPT-4o, Whisper API)
│   ├── google/              # Google Gemini provider (native video)
│   ├── deepgram/            # Deepgram transcription + streaming
│   ├── assemblyai/          # AssemblyAI transcription
│   ├── ollama/              # Local Ollama models
│   ├── voyage/              # Voyage AI embeddings
│   ├── claude_code/         # Host Claude Code instance
│   ├── litellm/             # LiteLLM unified interface
│   └── local_embedder.py    # Local sentence-transformers
│
├── parsing/                 # URL and input parsing
│   └── utils.py             # extract_video_id, parse_input
│
└── utils/                   # Shared utilities
    ├── formatting.py        # Output formatting
    ├── logging.py           # Logging configuration
    └── system.py            # System utilities

commands/                    # Claude Code slash commands
├── yt.md                    # /yt — Main video analysis
└── yt/
    ├── see.md               # /yt:see — Quick frames
    ├── hq.md                # /yt:hq — HQ frames
    ├── transcribe.md        # /yt:transcribe — Whisper transcription
    ├── transcript.md        # /yt:transcript — Read transcript
    ├── scenes.md            # /yt:scenes — Scene structure
    ├── find.md              # /yt:find — Search moments
    ├── watch.md             # /yt:watch — Intelligent watching
    └── list.md              # /yt:list — List cached videos
```

## Data Flow

### Slash Commands

```
User runs /yt <url>
  → Claude reads commands/yt.md
  → Command template includes inline Python
  → Python calls operations/processor.process_video()
  → operations layer uses tools/ wrappers (yt-dlp, ffmpeg, whisper)
  → Results cached via cache/manager.py
  → Results returned to Claude as text output
```

### MCP Server

```
MCP Client sends JSON-RPC request (stdin)
  → mcp_server.py routes to @mcp.tool() function
  → Tool function calls operations/ layer via asyncio.to_thread()
  → operations layer uses tools/ wrappers and providers/
  → Results cached via cache/
  → Tool function returns JSON response (stdout)
```

### Processing Pipeline

```
URL → operations/processor.py
  ├── tools/yt_dlp.py         → Fetch metadata, download audio
  ├── operations/subtitles.py  → Try YouTube subtitles first (free)
  ├── operations/transcribe.py → Fall back to Whisper (expensive)
  ├── tools/ffmpeg.py          → Extract frames on demand
  └── cache/manager.py         → Cache all results
```

## Cache Structure

```
~/.claudetube/
├── config.yaml                 # User configuration
├── db/
│   ├── claudetube.db           # Metadata database
│   └── claudetube-vectors.db   # Vector embeddings
├── cache/
│   └── {video_id}/
│       ├── state.json              # Metadata + processing state
│       ├── audio.mp3               # Downloaded audio
│       ├── audio.srt               # SRT transcript
│       ├── audio.txt               # Plain text transcript
│       ├── thumbnail.jpg           # Video thumbnail
│       ├── drill_{quality}/        # Frames from get_frames (480p default)
│       │   └── drill_MM-SS.jpg
│       ├── hq/                     # Frames from get_hq_frames (1280px)
│       │   └── hq_MM-SS.jpg
│       ├── scenes/                 # Scene segmentation data
│       │   ├── scenes.json         # Scene boundaries + metadata
│       │   └── scene_NNN/          # Per-scene analysis
│       │       ├── keyframes/      # Representative frames
│       │       ├── visual.json     # Visual description
│       │       ├── technical.json  # OCR, code detection
│       │       └── entities.json   # Extracted entities
│       ├── entities/               # Cross-scene data
│       │   └── people.json         # People tracking results
│       └── enrichment/             # Progressive learning data
│           ├── qa_history.json     # Cached Q&A interactions
│           └── observations.json   # Scene observations + boosts
├── playlists/                      # Cached playlist metadata
└── knowledge_graph.json            # Cross-video entity graph
```

### state.json

```json
{
  "video_id": "dYP2V_nK8o0",
  "url": "https://youtube.com/watch?v=dYP2V_nK8o0",
  "title": "Video Title",
  "duration": 300,
  "duration_string": "5:00",
  "transcript_complete": true,
  "transcript_source": "uploaded"
}
```

## Quality Tiers

Frame extraction supports 5 quality tiers, each with different resolution and compression settings:

| Tier | Resolution | JPEG Quality | Concurrent Downloads |
|------|-----------|-------------|---------------------|
| lowest | 480px | 5 | 1 |
| low | 640px | 4 | 2 |
| medium | 854px | 3 | 4 |
| high | 1280px | 2 | 4 |
| highest | 1280px | 2 | 4 |

The `get_frames` function downloads only the needed video segment (not the full video), extracts frames, then deletes the segment file.

## Logging

All logging uses Python's `logging` module (not `print()`). This is critical because:
- The MCP server uses stdout for JSON-RPC protocol messages
- Any `print()` to stdout would corrupt the protocol
- `logging` defaults to stderr, keeping stdout clean

The CLI entry point (`cli.py`) configures `logging.basicConfig()` to write to stderr with a simple message format.
