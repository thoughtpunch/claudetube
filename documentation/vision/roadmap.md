# Roadmap

> Where claudetube is going.

## Current State (v0.4.0)

**What works today:**
- [x] Process videos from 1500+ sites (YouTube, Vimeo, Twitch, etc.)
- [x] Transcription via subtitles or Whisper
- [x] On-demand frame extraction (480px and 1280px)
- [x] Intelligent caching
- [x] MCP server for Claude integration
- [x] Modular library architecture (refactored)

## Local File Support (In Progress)

Enable processing of local video files (screen recordings, downloads).

- [x] Detect local file paths vs URLs
- [x] Generate deterministic video_id for local files
- [x] Extract metadata via ffprobe
- [x] Extract audio for Whisper transcription
- [x] Frame extraction from local files
- [x] Copy/symlink local files to cache
- [ ] MCP tool for local file processing (in progress)
- [ ] Generate thumbnail from local video
- [ ] Check for embedded subtitles
- [ ] Tests for local file processing

**Goal**: Same workflow for local files as URL-sourced videos.

## Development Phases

### Phase 1: Structural Understanding (In Progress)

Give the agent a semantic map of the video.

- [x] YouTube chapters extraction
- [x] Cache structure for scenes
- [ ] Unified cheap boundary detection
- [ ] Transcript-based boundary detection
- [ ] Visual scene detection (PySceneDetect)
- [ ] Transcript-scene alignment
- [ ] Visual transcripts (dense captioning)
- [ ] /yt:scenes command
- [ ] Smart segmentation strategy
- [ ] OCR extraction for technical content
- [ ] Code block detection

**Goal**: Videos become structured documents, not linear streams.

### Phase 2: Semantic Search & Retrieval

Enable "find the part where..." queries.

- [ ] Implement temporal grounding search
- [ ] Add /yt:find command
- [ ] Create multimodal scene embeddings
- [ ] Build vector index (ChromaDB/FAISS)
- [ ] Support local embedding models

**Goal**: Jump directly to relevant sections.

### Phase 3: Temporal Reasoning

Understand change over time.

- [ ] Track entities (people, objects, code) across scenes
- [ ] Detect changes between consecutive scenes
- [ ] Detect narrative structure (intro, sections, conclusion)
- [ ] Track code evolution in programming tutorials
- [ ] Track objects and concepts across scenes

**Goal**: Answer "how did X evolve during this video?"

### Phase 4: Progressive Learning

Get smarter with each interaction.

- [ ] Implement VideoMemory class
- [ ] Multi-pass analysis depths (quick/standard/deep)
- [ ] Interaction-driven cache enrichment
- [ ] Cross-video knowledge graph

**Goal**: Memory across sessions, connected knowledge.

### Phase 5: Active Comprehension

Watch video like a human expert.

- [ ] Implement ActiveVideoWatcher class
- [ ] Attention priority modeling
- [ ] Comprehension verification
- [ ] /yt:watch command for active viewing

**Goal**: True video comprehension, not just retrieval.

## Infrastructure

### Configurable Cache Directory (Planned)

- [ ] Environment variable (CLAUDETUBE_CACHE_DIR)
- [ ] Project config (.claudetube/config.yaml)
- [ ] User config (~/.config/claudetube/)
- [ ] Update CacheManager
- [ ] Update MCP server
- [ ] Documentation

**Goal**: Flexible storage location for different workflows.

## Contributing

See the [detailed roadmap](../claudetube-roadmap.md) for technical specifications and implementation details.

Track progress in our [issue tracker](https://github.com/thoughtpunch/claudetube/issues).

---

**See also**:
- [The Problem Space](problem-space.md) - Why this matters
- [Beyond Transcripts](beyond-transcripts.md) - What makes us different
