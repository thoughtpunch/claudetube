[← Documentation](../README.md)

# The Problem Space: LLMs and Video

> How do you get an AI to truly understand video content?

## The Challenge

Large Language Models can read text, analyze images, and browse the web. But video remains a blind spot. When you share a YouTube link with Claude, it sees... nothing. Just a URL string.

This isn't a limitation of the models themselves. Modern LLMs like Claude are multimodal—they can process images, understand visual content, and reason about what they see. The problem is **access**: there's no standard way to feed video content into an AI conversation.

## Why Transcripts Aren't Enough

The naive solution is "just get the transcript." Every YouTube MCP tool does this. But transcripts alone miss critical information:

### 1. Visual Context
- Code on screen that the speaker references but doesn't read aloud
- Diagrams, charts, and UI elements
- Text overlays, annotations, and captions
- The speaker's demonstrations and gestures

### 2. Temporal Structure
- Where does the introduction end and the main content begin?
- What are the logical sections?
- When does the speaker show vs. tell?

### 3. Multi-Modal Correlation
- "As you can see here..." (see what?)
- "This line right here..." (which line?)
- "Let me scroll down to..." (to what?)

A transcript is like reading stage directions without seeing the play.

## The Vision: True Video Understanding

claudetube aims to give LLMs the same capabilities a human has when watching video:

### Level 1: Basic Perception (Current)
- **Transcript**: What was said, with timestamps
- **Frames**: Visual snapshots at any moment
- **Metadata**: Duration, chapters, source info

### Level 2: Structural Understanding (In Progress)
- **Scenes**: Semantic segments with boundaries
- **Visual Transcripts**: Descriptions of what's on screen
- **Technical Extraction**: OCR for code, diagrams, text

### Level 3: Temporal Reasoning (Planned)
- **Entity Tracking**: Follow people, objects, concepts across time
- **Change Detection**: What evolved between scenes?
- **Narrative Structure**: Intro, sections, conclusion

### Level 4: Semantic Search (Planned)
- **Moment Finding**: "Find where they explain the auth flow"
- **Multimodal Embeddings**: Search by concept, not keyword
- **Cross-Video Knowledge**: Connect information across videos

### Level 5: Active Comprehension (Future)
- **Attention Modeling**: Focus on what matters for the question
- **Progressive Analysis**: Quick scan → deep dive as needed
- **Comprehension Verification**: Self-check understanding

## The Key Insight

Video isn't a single format—it's a **bundle of synchronized media streams**:
- Audio (speech, music, sound effects)
- Visual (frames, motion, composition)
- Text (captions, on-screen text, code)
- Metadata (chapters, timestamps, annotations)

True understanding requires extracting, aligning, and synthesizing all of these. That's what claudetube does.

---

**Next**: [Beyond Transcripts](beyond-transcripts.md) - What makes claudetube different
