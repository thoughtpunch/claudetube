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

### 30 MCP Tools

The MCP server exposes 30 tools covering the full feature set: video processing,
transcription, frame extraction, scene analysis, entity tracking, semantic search,
active watching, audio descriptions, playlist analysis, knowledge graphs, and provider
management.

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
| MCP Tools | 5 | 30 |
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
- [ ] Wire embedding-based attention scoring

### Test & Quality (Epic: claudetube-axf)
Quality gates for release confidence:

- [ ] Unit tests for download, transcribe, and URL processor modules
- [ ] pytest-cov coverage baseline
- [ ] Integration test framework for real API calls
- [ ] Fix fragile async/sync bridge in OCR
- [ ] Complete mypy type checking setup

### Enhancements (P3-P4)
Nice-to-have improvements tracked for v1.0.x:

- [ ] `/yt:deep` and `/yt:focus` slash commands
- [ ] Multilingual audio descriptions
- [ ] Streaming transcription support
- [ ] OCR vs Vision quality benchmarking
- [ ] Local vision model support for visual transcripts

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
