# claudetube Documentation

> Teaching AI to watch videos, not just read transcripts.

## Documentation Index

### Getting Started
- [Installation](getting-started/installation.md) - Install claudetube and dependencies
- [Quick Start](getting-started/quickstart.md) - Process your first video
- [MCP Setup](getting-started/mcp-setup.md) - Configure for Claude Desktop/Code

### Core Concepts
- [Video Understanding](concepts/video-understanding.md) - How LLMs "watch" video
- [The Pipeline](concepts/pipeline.md) - From URL to understanding
- [Transcripts](concepts/transcripts.md) - Speech-to-text with timing
- [Frames](concepts/frames.md) - Visual snapshots for context
- [Scenes](concepts/scenes.md) - Semantic segmentation

### Architecture
- [Module Overview](architecture/modules.md) - Code organization
- [Data Flow](architecture/data-flow.md) - How data moves through the system
- [Cache Structure](architecture/cache.md) - What gets stored and why
- [Tool Wrappers](architecture/tools.md) - yt-dlp, ffmpeg, whisper

### API Reference
- [Core API](api/core.md) - Main functions
- [Models](api/models.md) - Data structures
- [Operations](api/operations.md) - High-level operations
- [Exceptions](api/exceptions.md) - Error handling

### Guides
- [Processing YouTube Videos](guides/youtube.md) - YouTube-specific features
- [Processing Local Files](guides/local-files.md) - Screen recordings, downloads
- [Extracting Frames](guides/frames.md) - Visual analysis
- [Searching Video Content](guides/search.md) - Finding moments

### Vision
- [The Problem Space](vision/problem-space.md) - Why video understanding matters
- [Beyond Transcripts](vision/beyond-transcripts.md) - What makes claudetube different
- [Roadmap](vision/roadmap.md) - Where we're going

---

## Quick Links

- [GitHub Repository](https://github.com/thoughtpunch/claudetube)
- [Issue Tracker](https://github.com/thoughtpunch/claudetube/issues)
- [MCP Server Protocol](https://modelcontextprotocol.io/)
