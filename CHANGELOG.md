# Changelog

All notable changes to claudetube are documented in this file.

## [1.0.0rc1] - 2026-02-02

**260 commits since v0.1.1** — A complete evolution from a video download-and-transcribe
script into a modular video understanding platform.

### Features

#### Configurable AI Provider System
A pluggable provider architecture with 8 backend implementations, smart routing,
and fallback chains.

- **Provider Framework**: Base protocols (`Transcriber`, `VisionAnalyzer`, `VideoAnalyzer`,
  `Reasoner`, `Embedder`), lazy-loading registry, YAML configuration with env var
  interpolation, and `ProviderRouter` with cost-based routing and parallel fallback execution
- **Providers**: WhisperLocal, ClaudeCode, Anthropic, OpenAI, Google (Gemini),
  Ollama, Deepgram, AssemblyAI, plus LiteLLM for generic reasoning
- **OperationFactory**: Centralized operation instantiation from provider config
- **CLI**: `validate-config` command for checking provider configuration
- **MCP**: `list_providers` tool for runtime provider discovery

#### Scene Detection & Structural Understanding (Phase 1)
Multi-strategy scene boundary detection following "Cheap First, Expensive Last":

- **Boundary detection**: YouTube chapters (free) → linguistic transition cues →
  pause detection → vocabulary shift analysis → PySceneDetect visual detection (expensive)
- **Smart segmentation**: Automatically selects best strategy per video
- **Transcript alignment**: Maps transcript segments to detected scenes
- **Visual transcripts**: Dense captioning of scene content via vision providers
- **Technical content**: OCR extraction, code block detection with language identification
- **`/yt:scenes` command** and `get_scenes` MCP tool with `enrich` parameter

#### Semantic Search & Retrieval (Phase 2)
Natural language moment finding without scanning entire videos:

- **Multimodal embeddings**: Combined visual + audio + text scene embeddings
- **Vector index**: ChromaDB-backed scene index with local embedding support
- **Temporal grounding**: `find_moments` search with blended text + semantic scoring
- **`/yt:find` command** and `find_moments_tool` MCP tool

#### Temporal Reasoning (Phase 3)
Understanding change over time across scenes:

- **People tracking**: Track person appearances across scenes with optional face_recognition
- **Object/concept tracking**: Track entities and concepts through a video
- **Code evolution**: Track how code changes across tutorial scenes
- **Change detection**: Identify what changed between consecutive scenes
- **Narrative structure**: Detect intro/main/conclusion sections, classify video type

#### Progressive Learning (Phase 4)
Agent gets smarter with each interaction:

- **VideoMemory**: Persistent per-video memory with Q&A history
- **Multi-pass analysis**: QUICK → STANDARD → DEEP → EXHAUSTIVE analysis depths
- **Interaction-driven enrichment**: Cache grows richer as users ask questions
- **Cross-video knowledge graph**: Connect topics and entities across videos in playlists

#### Human-Like Video Comprehension (Phase 5)
Active watching strategy modeled on human expert behavior:

- **ActiveVideoWatcher**: Generates hypotheses, examines promising sections, stops at
  high-confidence answers
- **Attention priority modeling**: Multi-factor scoring (relevance, density, novelty,
  visual salience, audio emphasis, structural importance) with video-type-aware weights
- **Comprehension verification**: Self-checks understanding before answering
- **`/yt:watch` command** and `watch_video_tool` MCP tool

#### Local File Processing
Full pipeline support for local video files (screen recordings, downloads):

- Local file path detection, content-hash based video_id generation
- Metadata extraction via ffprobe, embedded subtitle detection
- Audio extraction, frame extraction, thumbnail generation
- `process_local_file` MCP tool

#### Audio Description (Accessibility)
WCAG 2.1 Level AA compliant audio description generation:

- Detect existing AD tracks via yt-dlp
- Generate WebVTT description files from scene analysis
- AI-powered description generation via provider system
- 4 MCP tools: `get_descriptions`, `describe_moment`, `get_accessible_transcript`,
  `has_audio_description`

#### Playlist Support
Cross-video analysis for playlists and series:

- Playlist metadata extraction with type classification (course, series, conference)
- Cross-video knowledge graph with topic extraction and prerequisite chains
- `get_playlist` and `list_playlists` MCP tools

#### Configurable Cache Directory
- `CLAUDETUBE_CACHE_DIR` environment variable
- Project-level `.claudetube/config.yaml`
- User-level `~/.config/claudetube/config.yaml`
- Unified config loader with priority resolution

### Improvements

- Concurrent fragment downloads for faster video fetching
- Auto-download video thumbnails for instant visual context
- Partial video download via `--download-sections`
- Portable format selection (replaced hardcoded format codes)
- Subtitle-first transcription with Whisper fallback
- Configurable video quality with auto-escalation
- YouTube mweb client fallback for SABR 403 errors
- HQ frame extraction at 1280p for reading text/code
- OCR enhanced with VisionAnalyzer fallback for low-confidence results
- Embedder-based semantic query blending in search
- Slash commands check cache before invoking Python

### Architecture

- **Modular library refactor**: Monolith → `src/claudetube/` package with operations,
  analysis, providers, and cache subpackages
- **"Cheap First, Expensive Last" principle**: Codified as core architecture —
  Cache → yt-dlp → Local → Compute hierarchy
- **Operations layer refactoring**: All operations accept providers via constructor
  injection through OperationFactory
- **Linting**: Migrated from black/isort/flake8 to ruff

### Bug Fixes

- Fix batched Whisper dropping segments
- Fix stdout corruption from `print()` calls corrupting MCP protocol
- Fix URL parsing with shell `&` interpretation
- Fix import ordering issues
- Fix PyYAML missing from dependencies
- Fix requirements.txt / pyproject.toml sync

### Infrastructure

- Docker support: Dockerfile and `.dockerignore` for containerized MCP server
- Cross-platform installer: `install.sh` (macOS/Linux) and `install.ps1` (Windows)
- CI/CD: GitHub Actions with ruff linting and mypy type checking
- PyPI publishing workflow

### Dependencies

Core: yt-dlp, faster-whisper, pysubs2, pyyaml, scikit-learn, scenedetect, numpy,
pydantic, Pillow

Optional groups: `[mcp]`, `[openai]`, `[anthropic]`, `[google]`, `[deepgram]`,
`[assemblyai]`, `[ollama]`, `[embeddings-local]`, `[all-providers]`

### Known Issues (RC)

The following are tracked for resolution before v1.0.0 final:

- 4 operations modules not exported from `operations/__init__.py`
  (narrative_structure, code_evolution, playlist, knowledge_graph)
- 3 implemented features lack MCP tools (code evolution, narrative structure,
  change detection)
- AudioDescriptionGenerator uses registry directly instead of ProviderRouter
- Video type not auto-detected for attention model weighting
- AD language hardcoded to English
- Fragile async/sync bridge in OCR vision integration
- Missing unit tests for download.py, transcribe.py, and URL-based processor flow
- `/yt:deep` and `/yt:focus` slash commands not yet created

See beads epic `claudetube-lgb` (Wire Up All Implemented Features) and
`claudetube-axf` (Test Coverage and Quality Gates) for full tracking.

---

## [0.1.1] - 2026-01-27

- Fix batched Whisper segment handling
- Add CI/CD pipeline

## [0.1.0] - 2026-01-26

- Initial release
- Process videos from YouTube and 1500+ sites via yt-dlp
- Transcribe audio with local Whisper
- Extract frames at 480p and 1280p
- MCP server with core tools
- Intelligent caching at `~/.claude/video_cache/`
