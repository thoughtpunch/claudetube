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
- [**Overview**](architecture/architecture.md) - Codebase structure, data flow, cache layout
- [**Principles**](architecture/principles.md) - **Cheap first, expensive last** (start here)

### Guides
- [**Configuration**](guides/configuration.md) - **Cache directory, providers, and settings**
- [**Tool Reference**](guides/tool-reference.md) - **MCP tools, CLI commands, slash commands**
- [MCP Setup](guides/mcp-setup.md) - MCP server installation and configuration
- [Docker](guides/docker.md) - Container-based setup
- [YouTube Authentication](guides/youtube-auth.md) - Handling auth, 403 errors, PO tokens

### Vision
- [The Problem Space](vision/problem-space.md) - Why video understanding matters
- [Beyond Transcripts](vision/beyond-transcripts.md) - What makes claudetube different
- [Roadmap](vision/roadmap.md) - Where we're going

### PRDs
- [Audio Description Tracks](prds/audio-description-tracks.md) - Accessibility feature design
- [Configurable AI Providers](prds/configurable-ai-providers.md) - Provider system design
- [Provider Epics](prds/configurable-ai-providers-epics.md) - Implementation epics
- [ClaudeTube Roadmap](prds/claudetube-roadmap.md) - Evolution roadmap with technical specs
- [Hierarchical Storage](prds/hierarchical-storage-sqlite-index.md) - SQLite index design
- [YouTube PO Token & SABR](prds/youtube-po-token-sabr-support.md) - YouTube auth support

### Release Notes
- [v1.0.0rc1](release-notes/RELEASE_NOTES.md) - Current release candidate

---

## Quick Links

- [GitHub Repository](https://github.com/thoughtpunch/claudetube)
- [Issue Tracker](https://github.com/thoughtpunch/claudetube/issues)
- [MCP Server Protocol](https://modelcontextprotocol.io/)
