# claudetube Submission Copy

Ready-to-paste content for all manual submissions.

---

## 1. Official MCP Registry

Just run:
```bash
mcp-publisher publish
```

---

## 2. hesreallyhim/awesome-claude-code

**URL**: https://github.com/hesreallyhim/awesome-claude-code/issues/new?template=recommend-resource.yml

| Field | Value |
|-------|-------|
| Display Name | `claudetube` |
| Category | `Agent Skills` |
| Sub-Category | `CLAUDE.md Files: Project Scaffolding & MCP` |
| Primary Link | `https://github.com/thoughtpunch/claudetube` |
| Author Name | `thoughtpunch` |
| Author Link | `https://github.com/thoughtpunch` |
| License | `MIT` |

**Description** (paste this):
```
MCP server that lets Claude watch YouTube videos via transcripts and on-demand frame extraction. Supports 1500+ video sites through yt-dlp. Features include scene detection, visual analysis, accessibility audio descriptions, and 40+ tools for video understanding.
```

**Validate Claims** (paste this):
```
Install claudetube via pip, configure as MCP server in Claude Code, then give Claude any YouTube URL. Claude will be able to transcribe, search, and answer questions about the video content.
```

**Specific Task** (paste this):
```
Ask Claude to summarize a YouTube video or answer a specific question about video content.
```

**Specific Prompt** (paste this):
```
Watch this video and tell me the main points: https://www.youtube.com/watch?v=dQw4w9WgXcQ
```

**Checkboxes**: Check all 5

---

## 3. mcpservers.org (wong2/awesome-mcp-servers)

**URL**: https://mcpservers.org/submit

| Field | Value |
|-------|-------|
| Server Name | `claudetube` |
| Category | `Media` or `Development` |
| Link | `https://github.com/thoughtpunch/claudetube` |

**Short Description** (paste this):
```
Let AI assistants watch YouTube videos via transcripts and on-demand frame extraction. Supports 1500+ sites via yt-dlp.
```

---

## 4. mcp.so

**URL**: https://mcp.so (click Submit button)

| Field | Value |
|-------|-------|
| Name | `claudetube` |
| GitHub URL | `https://github.com/thoughtpunch/claudetube` |

**Description** (paste this):
```
Video analysis MCP server that lets AI assistants watch and understand video content. Downloads and transcribes videos from YouTube and 1500+ other sites via yt-dlp. Features smart transcription (YouTube captions with Whisper fallback), on-demand frame extraction for visual analysis, scene detection, and 40+ MCP tools for comprehensive video understanding.
```

**Features** (paste this):
```
- Process videos from YouTube + 1500 other sites
- Smart transcription (YouTube captions → Whisper fallback)
- On-demand frame extraction for visual analysis
- Scene segmentation and search
- Entity extraction (people, objects, concepts)
- Accessibility features (audio descriptions)
- Playlist support with progress tracking
- 40+ MCP tools
```

---

## 5. r/ClaudeAI Post

**URL**: https://reddit.com/r/ClaudeAI/submit

**Title**:
```
I built claudetube - MCP server that lets Claude watch YouTube videos
```

**Body** (paste this):
```
Hey r/ClaudeAI!

I built claudetube, an MCP server that lets Claude "watch" and understand video content. Share a YouTube link and Claude can summarize it, answer questions, or extract specific information.

**How it works:**
- Downloads video metadata and transcripts automatically
- Uses YouTube captions when available (fast), falls back to Whisper (accurate)
- Can extract frames on-demand when you need visual context
- Supports 1500+ video sites via yt-dlp (not just YouTube)

**Features:**
- 40+ MCP tools for video understanding
- Scene detection and search ("find when they talk about X")
- Entity extraction (people, objects, code snippets)
- Accessibility features (audio descriptions)
- Playlist support with progress tracking

**Install:**
```
pip install claudetube
```

Then add to your Claude Code MCP config.

GitHub: https://github.com/thoughtpunch/claudetube
PyPI: https://pypi.org/project/claudetube/

Happy to answer any questions!
```

---

## 6. r/ClaudeCode Post

**URL**: https://reddit.com/r/ClaudeCode/submit

**Title**:
```
claudetube - MCP server for video analysis (YouTube + 1500 sites)
```

**Body** (paste this):
```
Just released claudetube v1.0 - an MCP server that gives Claude the ability to watch and analyze videos.

**Quick setup:**
```bash
pip install claudetube
```

Add to `~/.claude.json`:
```json
{
  "mcpServers": {
    "claudetube": {
      "command": "python",
      "args": ["-m", "claudetube.mcp"]
    }
  }
}
```

**What it does:**
- Share any video URL → Claude gets the transcript + can extract frames
- Smart transcription: YouTube captions (free/fast) → Whisper (fallback)
- 40+ MCP tools: process_video, get_frames, get_transcript, find_moments, watch_video, etc.
- Supports YouTube + 1500 other sites via yt-dlp

**Use cases:**
- "Summarize this conference talk"
- "What code does he write at 15:30?" (extracts HQ frames)
- "Find all mentions of authentication in this tutorial"
- "Watch this playlist and track my progress"

GitHub: https://github.com/thoughtpunch/claudetube

Feedback welcome!
```

---

## 7. Hacker News (Show HN)

**URL**: https://news.ycombinator.com/submit

**Title** (exactly this format):
```
Show HN: claudetube – MCP server to let AI assistants watch YouTube videos
```

**URL field**:
```
https://github.com/thoughtpunch/claudetube
```

**First comment** (post immediately after submitting):
```
Hi HN! I built claudetube because I wanted Claude to understand video content when helping with projects.

It's an MCP server that:
- Downloads and transcribes videos from 1500+ sites (via yt-dlp)
- Uses YouTube captions when available, falls back to Whisper
- Extracts frames on-demand for visual analysis
- Provides 40+ tools for Claude to query video content

Technical details:
- Python package, ~15 dependencies
- Smart caching so repeated queries are instant
- "Cheap First, Expensive Last" architecture - always tries free/fast options before compute-heavy ones
- Scene detection uses transcript analysis before expensive visual processing

Install: `pip install claudetube`

The MCP tools include things like `find_moments` (semantic search in video), `watch_video` (multi-step reasoning about content), and `get_hq_frames` (for reading code/text in videos).

Would love feedback on the API design and any use cases I haven't thought of!
```

---

## 8. Product Hunt

**URL**: https://producthunt.com

**Tagline** (60 chars max):
```
Let AI assistants watch and understand YouTube videos
```

**Description**:
```
claudetube is an MCP server that gives AI assistants like Claude the ability to watch and analyze video content.

Share a YouTube link (or any of 1500+ supported sites) and your AI can:
- Summarize the video content
- Answer specific questions about what happens
- Extract code, text, or diagrams shown on screen
- Find specific moments ("when do they discuss X?")
- Track progress through playlists and courses

Built with a "Cheap First, Expensive Last" philosophy - it uses free YouTube captions before running Whisper, and only extracts frames when visual context is actually needed.

Open source, MIT licensed, available on PyPI.
```

**First Comment/Maker Comment**:
```
Hey Product Hunt! 👋

I built claudetube because I was tired of manually transcribing videos or copy-pasting timestamps when asking Claude about video content.

The key insight is that most video understanding doesn't need actual video processing - transcripts cover 80% of questions. But when you DO need visual context (code on screen, diagrams, UI elements), claudetube can extract high-quality frames on demand.

Some things I'm proud of:
- Works with 1500+ video sites, not just YouTube
- Smart caching means the second question about a video is instant
- 40+ MCP tools covering everything from basic transcription to entity tracking

Would love to hear what video-related tasks you'd want an AI to help with!
```

**Topics/Tags**: `AI`, `Developer Tools`, `Open Source`, `Productivity`, `Video`

---

## Quick Reference - All Links

| Platform | Submit URL |
|----------|------------|
| MCP Registry | Run `mcp-publisher publish` |
| awesome-claude-code | [Open Form](https://github.com/hesreallyhim/awesome-claude-code/issues/new?template=recommend-resource.yml) |
| mcpservers.org | [Submit](https://mcpservers.org/submit) |
| mcp.so | [Submit](https://mcp.so) |
| r/ClaudeAI | [Submit](https://reddit.com/r/ClaudeAI/submit) |
| r/ClaudeCode | [Submit](https://reddit.com/r/ClaudeCode/submit) |
| Hacker News | [Submit](https://news.ycombinator.com/submit) |
| Product Hunt | [Submit](https://producthunt.com) |

---

*Generated 2026-02-05*
