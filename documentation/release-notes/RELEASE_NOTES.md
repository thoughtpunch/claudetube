[← Documentation](../README.md)

# claudetube v1.0.0rc1 Release Notes

**Release Candidate 1** | February 2, 2026

claudetube v1.0.0rc1 is the first release candidate for the v1.0.0 milestone. It
represents a ground-up rebuild from the original v0.1.x script into a modular video
understanding platform. This RC is feature-complete against the planned 5-phase roadmap
and is being stabilized for final release.

---

## What is claudetube?

claudetube lets AI assistants **watch and understand videos**. Point it at any video
URL (YouTube, Vimeo, Twitch, TikTok, and 1,500+ sites) or a local file, and it
extracts transcripts, frames, scene structure, and semantic context — all through an
MCP server that plugs into Claude Desktop, Claude Code, or any MCP-compatible client.

---

## Highlights

### Pluggable AI Provider System

Swap between 8 AI backends without changing your workflow. Transcription, vision,
reasoning, and embedding capabilities are all abstracted behind protocols with
automatic fallback chains.

```yaml
# ~/.config/claudetube/config.yaml
providers:
  preferences:
    transcription: whisper-local   # Free, local
    vision: anthropic              # Claude for vision
  fallbacks:
    vision: [openai, claude-code]  # Automatic failover
  cost_preference: cost            # Prefer cheaper providers
```

**Zero-config defaults** work out of the box: local Whisper for transcription, Claude
Code for vision and reasoning. Add API keys to unlock cloud providers.

**Supported providers:** WhisperLocal, ClaudeCode, Anthropic, OpenAI, Google Gemini,
Ollama, Deepgram, AssemblyAI

### Intelligent Scene Detection

Videos are automatically segmented into semantic scenes using a multi-strategy approach
that prioritizes free/fast methods:

1. YouTube chapters (free, human-curated)
2. Transcript analysis (linguistic cues, pauses, vocabulary shifts)
3. Visual scene detection via PySceneDetect (when needed)

Each scene gets transcript alignment, visual descriptions, OCR extraction, and code
block detection — all cached for instant retrieval.

### Semantic Video Search

Find specific moments in videos with natural language queries:

```
/yt:find abc123 "when they fix the authentication bug"
```

Powered by ChromaDB vector indexing with multimodal scene embeddings that combine
visual, audio, and text signals.

### Active Video Watching

The `/yt:watch` command implements a human-like video comprehension strategy:

- Generates hypotheses about content
- Prioritizes scenes using multi-factor attention modeling
- Examines the most relevant sections first
- Self-verifies comprehension before answering

### Local File Support

Process screen recordings, downloaded videos, and local media files with the same
pipeline as URL-based videos:

```
process_local_file("/path/to/recording.mp4")
```

### Audio Descriptions (Accessibility)

Generate WCAG 2.1 Level AA compliant audio descriptions from video content. Detects
existing AD tracks, or generates descriptions from scene analysis using configured
vision providers.

### YouTube Authentication & SABR Support

YouTube's migration to SABR streaming and PO token enforcement broke many third-party
tools. claudetube now handles this transparently:

- Smart client fallback chain with retry on 403 errors
- Cookie management (`cookies_file` or `cookies_from_browser`)
- PO token support (manual or automated via bgutil plugin)
- deno recommended for yt-dlp JS challenge solving
- `youtube_auth_status_tool` MCP tool for diagnostics (reports auth level 0–4)
- Tiered setup guide from zero-config to fully automated

### 31 MCP Tools

The MCP server exposes 31 tools covering the full feature set: video processing,
transcription, frame extraction, scene analysis, entity tracking, semantic search,
active watching, audio descriptions, playlist analysis, knowledge graphs, YouTube
auth diagnostics, and provider management.

---

## What's New Since v0.1.1

| Area | v0.1.1 | v1.0.0rc1 |
|------|--------|-----------|
| Architecture | Single-file script | Modular package (operations, analysis, providers, cache) |
| AI Backends | Hardcoded Whisper | 8 pluggable providers with fallback routing |
| Scene Understanding | None | Multi-strategy detection + visual transcripts |
| Search | None | Vector-indexed semantic search |
| Video Comprehension | Transcript-only | 5-phase progressive understanding |
| Entity Tracking | None | People, objects, code evolution across scenes |
| Local Files | Not supported | Full pipeline support |
| Accessibility | None | Audio description generation |
| Playlists | Not supported | Metadata + cross-video knowledge graph |
| Configuration | None | Hierarchical YAML config + env vars |
| YouTube Auth | None | PO tokens, cookies, bgutil plugin, auth diagnostics |
| MCP Tools | 5 | 31 |
| Tests | Minimal | 2,000+ |
| Slash Commands | 2 | 9 |

---

## Installation

```bash
pip install claudetube==1.0.0rc1
```

With optional providers:
```bash
pip install "claudetube[openai,anthropic]==1.0.0rc1"   # Cloud providers
pip install "claudetube[all-providers]==1.0.0rc1"       # Everything
pip install "claudetube[embeddings-local]==1.0.0rc1"    # Local embeddings
```

### Requirements

- Python 3.10+
- ffmpeg (for audio/video processing)
- yt-dlp (bundled as dependency)
- deno (recommended for YouTube JS challenge solving)

---

## RC Status: What's Left for v1.0.0 Final

This is a release candidate. The following items are tracked for resolution before
the final v1.0.0 release:

### Wiring Gaps (Epic: claudetube-lgb)
Implemented features that need to be connected to their public interfaces:

- [ ] Export 4 operations modules from package `__init__.py`
- [ ] Add MCP tools for code evolution, narrative structure, and change detection
- [ ] Add MCP tools for knowledge graph operations
- [ ] Route AudioDescriptionGenerator through ProviderRouter
- [ ] Auto-detect video type for attention model weighting
- [x] ~~Wire embedding-based attention scoring~~ (done)
- [x] ~~Wire visual transcripts through ProviderRouter~~ (done)

### Test & Quality (Epic: claudetube-axf)
Quality gates for release confidence:

- [x] ~~Unit tests for download, transcribe, and URL processor modules~~ (done)
- [x] ~~pytest-cov coverage baseline~~ (done)
- [ ] Integration test framework for real API calls
- [ ] Fix fragile async/sync bridge in OCR
- [ ] Complete mypy type checking setup

### Enhancements (P3-P4)
Nice-to-have improvements tracked for v1.0.x:

- [ ] `/yt:deep` and `/yt:focus` slash commands
- [x] ~~Multilingual audio descriptions~~ (done)
- [ ] Streaming transcription support
- [x] ~~OCR vs Vision quality benchmarking~~ (done)
- [ ] Local vision model support for visual transcripts

### Recently Completed (v1.0.0rc2 - February 3-4, 2026)

The following were resolved after the initial RC build:

**New Features:**
- **`process_video_tool` force flag**: New `force: bool` parameter clears cache and re-processes from scratch
- **YouTube chapters extraction**: Chapters are now extracted from yt-dlp metadata and included in VideoState
- **Playlist metadata extraction**: Fixed `get_playlist` to correctly extract title, channel, and video count using `--dump-single-json`

**Bug Fixes:**
- **`watch_video_tool` answer synthesis**: Now extracts relevant sentences instead of first 200 chars, and ranks hypotheses by relevance (60%) + confidence (40%) - fixes issue where irrelevant content was returned
- **YouTube subtitles**: Fixed to use YouTube-provided subtitles even when yt-dlp returns non-zero exit code
- **`describe_moment` delegation**: Returns proper delegation response for claude-code provider with frame paths
- **Long Whisper SRT segments**: Segments over 7 seconds are now split using word timestamps for better readability
- **`index_video_to_graph_tool`**: Now warns when video has no entities/concepts to index instead of silently succeeding
- **Major transition filtering**: Improved to use relative threshold (top 25%) instead of absolute threshold

**CI/Code Quality:**
- Fixed ruff formatting issues across 20+ files
- Fixed lint errors (unused variables, nested ifs, import sorting)

---

### Previously Completed (v1.0.0rc1)

- **YouTube SABR/403 fix**: Upgraded yt-dlp, added smart client fallback chain
- **YouTube auth support**: PO tokens, cookies, bgutil plugin, auth diagnostics tool
- **deno prerequisite**: Recommended in install.sh, logged if missing at runtime
- **Rotating file logging**: MCP server logs to `~/Library/Logs/Claude/claudetube-mcp.log`
- **Python 3.10 compatibility**: Fixed mock.patch name collision and protocol test portability
- **Documentation reorganization**: All docs navigable as a wiki from `documentation/README.md`

---

## Breaking Changes from v0.1.x

- **Package structure**: `claudetube` is now a proper Python package under `src/`.
  Direct imports from the old single-file structure will not work.
- **Cache location**: Default unchanged (`~/.claude/video_cache/`) but now
  configurable via `CLAUDETUBE_CACHE_DIR` or config files.
- **MCP server**: Tool names and parameters have been expanded. Existing integrations
  using only `process_video_tool`, `get_frames`, `get_hq_frames` are compatible.

---

## Contributors

Built by Daniel Barrett with extensive AI-assisted development (Claude).

## License

MIT
