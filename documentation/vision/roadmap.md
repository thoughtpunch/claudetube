# Roadmap

> Where claudetube is going.

## Current State (v0.4.0)

**What works today:**
- Process videos from 1500+ sites (YouTube, Vimeo, Twitch, etc.)
- Transcription via subtitles or Whisper
- On-demand frame extraction (480px and 1280px)
- Intelligent caching
- MCP server for Claude integration
- Local file support (in progress)

## Development Phases

### Phase 1: Structural Understanding (In Progress)

Give the agent a semantic map of the video.

- [x] YouTube chapters extraction
- [x] Cache structure for scenes
- [ ] Transcript-based boundary detection
- [ ] Visual scene detection (PySceneDetect)
- [ ] Transcript-scene alignment
- [ ] Visual transcripts (dense captioning)
- [ ] /yt:scenes command

**Goal**: Videos become structured documents, not linear streams.

### Phase 2: Semantic Search & Retrieval

Enable "find the part where..." queries.

- [ ] Multimodal scene embeddings
- [ ] Vector index (ChromaDB/FAISS)
- [ ] /yt:find command
- [ ] Natural language moment search

**Goal**: Jump directly to relevant sections.

### Phase 3: Temporal Reasoning

Understand change over time.

- [ ] Entity tracking (people, objects, code)
- [ ] Change detection between scenes
- [ ] Narrative structure detection
- [ ] Code evolution tracking

**Goal**: Answer "how did X evolve during this video?"

### Phase 4: Progressive Learning

Get smarter with each interaction.

- [ ] Interaction-driven cache enrichment
- [ ] Multi-pass analysis (quick/standard/deep)
- [ ] Cross-video knowledge graph

**Goal**: Memory across sessions, connected knowledge.

### Phase 5: Active Comprehension

Watch video like a human expert.

- [ ] Active watching strategy
- [ ] Attention modeling
- [ ] Comprehension verification
- [ ] /yt:watch command

**Goal**: True video comprehension, not just retrieval.

## Contributing

See the [detailed roadmap](../claudetube-roadmap.md) for technical specifications and implementation details.

Track progress in our [issue tracker](https://github.com/thoughtpunch/claudetube/issues).

---

**See also**:
- [The Problem Space](problem-space.md) - Why this matters
- [Beyond Transcripts](beyond-transcripts.md) - What makes us different
