[← Documentation](../README.md)

# Roadmap

> Where claudetube is going.

## The UX Problem

claudetube currently requires multi-step orchestration:

```
process_video(url)     # Wait for download/transcription
get_scenes(video_id)   # Optional: understand structure
get_frames(...)        # Optional: extract visuals
# ... synthesize answer
```

Native video AI (like Gemini) offers: `URL + question → answer`

**The gap is real.** claudetube is a toolkit, not a feature. We're working to close this gap.

## Priority: Streamlined Single-Call Q&A

**Goal:** Match native video AI UX while preserving claudetube's advantages.

```python
# Future API
ask_video(url, question) → answer
```

This should:
- [ ] Handle all orchestration internally (no explicit `process_video`)
- [ ] Use query-aware frame sampling (extract frames relevant to the question)
- [ ] Leverage caching (instant for previously processed videos)
- [ ] Fall back gracefully (transcript → visuals as needed)
- [ ] Maintain access to granular tools for power users

**Tracking:** [beads-wdyt](../.beads/issues/claudetube-wdyt.md)

---

## Current State (v0.4.0)

**What works today:**
- [x] Process videos from 1500+ sites (YouTube, Vimeo, Twitch, etc.)
- [x] Transcription via subtitles or Whisper
- [x] On-demand frame extraction (480px and 1280px)
- [x] Intelligent caching
- [x] MCP server for Claude integration (40+ tools)
- [x] Scene segmentation and visual transcripts
- [x] Entity extraction and people tracking
- [x] Playlist navigation with learning intelligence
- [x] Cross-video knowledge graph
- [x] Accessibility features (audio descriptions)

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

See the [detailed roadmap](../prds/claudetube-roadmap.md) for technical specifications and implementation details.

Track progress in our [issue tracker](https://github.com/thoughtpunch/claudetube/issues).

---

**See also**:
- [The Problem Space](problem-space.md) - Why this matters
- [Beyond Transcripts](beyond-transcripts.md) - What makes us different
