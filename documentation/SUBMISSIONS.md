# claudetube Listing Submissions Tracker

Track all submissions to awesome lists, directories, and communities.

## Summary

| Status | Count |
|--------|-------|
| Pending | 21 |
| Submitted | 1 |
| Approved | 0 |
| Rejected | 0 |

---

## Tier 1: Official MCP Registry (Highest Priority)

### Official MCP Registry
- **URL**: https://registry.modelcontextprotocol.io
- **Status**: 🟡 Ready to publish (needs OAuth)
- **Submitted**: -
- **Prerequisites**:
  - [x] ~~Publish to npm first~~ **PyPI is supported!** claudetube v1.0.0 is on PyPI
  - [x] GitHub account for OAuth
  - [x] Install mcp-publisher CLI
  - [x] server.json validated successfully
- **Next Step**: Run `mcp-publisher login github` and complete OAuth, then `mcp-publisher publish`
- **Submission Plan**:
  ```bash
  # 1. Install mcp-publisher
  curl -L "https://github.com/modelcontextprotocol/registry/releases/download/v1.0.0/mcp-publisher_1.0.0_darwin_arm64.tar.gz" | tar xz mcp-publisher
  sudo mv mcp-publisher /usr/local/bin/

  # 2. Initialize server.json
  mcp-publisher init

  # 3. Edit server.json with claudetube metadata
  # Namespace: io.github.thoughtpunch/claudetube
  # Package: PyPI (claudetube)

  # 4. Authenticate
  mcp-publisher login github

  # 5. Validate before publishing
  mcp-publisher publish --dry-run

  # 6. Publish
  mcp-publisher publish
  ```
- **Docs**: https://github.com/modelcontextprotocol/registry/blob/main/docs/guides/publishing/publish-server.md

---

## Tier 2: MCP Awesome Lists

### wong2/awesome-mcp-servers
- **URL**: https://github.com/wong2/awesome-mcp-servers
- **Status**: Pending
- **Submitted**: -
- **Method**: Web form only (no PRs accepted)
- **Submission Plan**:
  1. Go to https://mcpservers.org/submit
  2. Fill form:
     - **Server Name**: claudetube
     - **Short Description**: Let AI assistants watch YouTube videos via transcripts and on-demand frame extraction. Supports 1500+ sites via yt-dlp.
     - **Link**: https://github.com/thoughtpunch/claudetube
     - **Category**: Development (or Other)
     - **Contact Email**: [your email]
  3. Optional: Pay $39 for faster review + official badge
- **Notes**: Free listing, review required

### punkpeye/awesome-mcp-servers
- **URL**: https://github.com/punkpeye/awesome-mcp-servers
- **Status**: Pending
- **Submitted**: -
- **Prerequisites**:
  - [ ] 20+ GitHub stars required
- **Submission Plan**:
  1. Fork repo
  2. Create branch: `git checkout -b add-claudetube`
  3. Edit README.md, add to appropriate category in **alphabetical order**:
     ```markdown
     - [claudetube](https://github.com/thoughtpunch/claudetube) - Let AI assistants watch YouTube videos via transcripts and on-demand frame extraction.
     ```
  4. Format: `[Title](link) - Description.` (capital letter start, period end)
  5. Commit: `git commit -m "Add claudetube"`
  6. PR title: `Add claudetube`
- **Rules**: No draft PRs, check for duplicates, individual PRs only

### appcypher/awesome-mcp-servers
- **URL**: https://github.com/appcypher/awesome-mcp-servers
- **Status**: ✅ Submitted
- **Submitted**: 2026-02-05
- **PR**: https://github.com/appcypher/awesome-mcp-servers/pull/264
- **Submission Plan**:
  1. Search for duplicates first
  2. Fork repo, create branch
  3. Add to bottom of appropriate category (alphabetical within category)
  4. Format: Include name, link, and succinct description
  5. Check spelling/grammar, remove trailing whitespace
  6. PR with useful title
- **Rules**: Individual PRs for each suggestion, alphabetical order

### ever-works/awesome-mcp-servers
- **URL**: https://github.com/ever-works/awesome-mcp-servers
- **Status**: Pending
- **Submitted**: -
- **Notes**: Links to mcpserver.works - check if same submission as wong2

### TensorBlock/awesome-mcp-servers
- **URL**: https://github.com/TensorBlock/awesome-mcp-servers
- **Status**: Pending
- **Submitted**: -
- **Submission Plan**: Standard PR process (check CONTRIBUTING.md)

---

## Tier 3: Claude Code Ecosystem

### hesreallyhim/awesome-claude-code
- **URL**: https://github.com/hesreallyhim/awesome-claude-code
- **Status**: 🟡 Ready to submit (web form required)
- **Submitted**: -
- **Method**: Issue-based recommendation system (NOT PRs)
- **Submit URL**: https://github.com/hesreallyhim/awesome-claude-code/issues/new?template=recommend-resource.yml
- **Form Fields**:
  - Display Name: claudetube
  - Category: Agent Skills (or Tooling)
  - Sub-Category: CLAUDE.md Files: Project Scaffolding & MCP
  - Primary Link: https://github.com/thoughtpunch/claudetube
  - Author: thoughtpunch
  - License: MIT
  - Description: MCP server that lets Claude watch YouTube videos via transcripts and on-demand frame extraction. Supports 1500+ sites via yt-dlp.
- **Notes**: 22.8k stars, high visibility

### ccplugins/awesome-claude-code-plugins
- **URL**: https://github.com/ccplugins/awesome-claude-code-plugins
- **Status**: ❌ Not applicable
- **Reason**: Repo is for bundled Claude Code plugins, not external MCP servers installed via pip

### travisvn/awesome-claude-skills
- **URL**: https://github.com/travisvn/awesome-claude-skills
- **Status**: ❌ Not applicable
- **Reason**: Repo is for Claude Skills (a different feature), not MCP servers

### jqueryscript/awesome-claude-code
- **URL**: https://github.com/jqueryscript/awesome-claude-code
- **Status**: Pending
- **Submitted**: -
- **Submission Plan**: Standard PR, add under tools/integrations

---

## Tier 4: AI & Whisper Lists

### sindresorhus/awesome-whisper
- **URL**: https://github.com/sindresorhus/awesome-whisper
- **Status**: Pending
- **Submitted**: -
- **Prerequisites**:
  - [ ] 20+ GitHub stars required
- **Submission Plan**:
  1. Fork repo
  2. Add to bottom of relevant category (likely "Apps" or "Tools")
  3. Format: `[Title](link) - Description.`
  4. Title: `Add claudetube`
  5. English only, no marketing language
  6. Individual PR, check for duplicates first
- **Angle**: Video transcription tool that uses Whisper as fallback

### mahseema/awesome-ai-tools
- **URL**: https://github.com/mahseema/awesome-ai-tools
- **Status**: Pending
- **Submitted**: -
- **Category**: Code with AI / Generative AI Video
- **Submission Plan**:
  1. Fork repo
  2. Add to appropriate category
  3. Submit PR
- **Alternative**: Submit via altern.ai (free)

### freejacklee/Awesome-AI-Video-Projects
- **URL**: https://github.com/freejacklee/Awesome-AI-Video-Projects
- **Status**: Pending
- **Submitted**: -
- **Submission Plan**: Standard PR process
- **Angle**: AI video analysis/understanding tool

---

## Tier 5: MCP Directories (Web)

### mcpservers.org
- **URL**: https://mcpservers.org/submit
- **Status**: Pending (same as wong2)
- **Notes**: Free listing, $39 for premium badge + faster review

### mcp.so
- **URL**: https://mcp.so
- **Status**: Pending
- **Notes**: Check if uses same submission as mcpservers.org

### glama.ai
- **URL**: https://glama.ai/mcp/servers
- **Status**: Pending
- **Notes**: Linked from punkpeye repo, check submission process

### mcp-awesome.com
- **URL**: https://mcp-awesome.com
- **Status**: Pending
- **Notes**: 1200+ servers directory, check submission process

---

## Tier 6: Reddit Communities

### r/ClaudeAI
- **URL**: https://reddit.com/r/ClaudeAI
- **Status**: Pending
- **Members**: 386k
- **Submission Plan**:
  1. Check sidebar for self-promotion rules
  2. Follow 80/20 rule (engage first, then promote)
  3. Post as "Show and Tell" or project announcement
  4. Title: "I built claudetube - let Claude watch YouTube videos via MCP"
  5. Include: what it does, why you built it, demo/examples
  6. Be available to answer questions
- **Best Time**: Weekday mornings (US time)
- **Content**:
  ```
  Title: I built claudetube - MCP server that lets Claude watch YouTube videos

  Body:
  Been working on this for a while - claudetube is an MCP server that lets
  Claude "watch" video content through transcripts and on-demand frame extraction.

  Features:
  - Works with YouTube + 1500 other sites (via yt-dlp)
  - Smart transcription (YouTube captions → Whisper fallback)
  - Extract frames when you need visual context
  - 40+ MCP tools for video understanding

  GitHub: https://github.com/thoughtpunch/claudetube

  Happy to answer any questions!
  ```

### r/ClaudeCode
- **URL**: https://reddit.com/r/ClaudeCode
- **Status**: Pending
- **Members**: 49k
- **Submission Plan**: Same as r/ClaudeAI, more technical focus
- **Angle**: Focus on MCP integration, developer workflow

### r/LocalLLaMA
- **URL**: https://reddit.com/r/LocalLLaMA
- **Status**: Pending
- **Submission Plan**: Similar approach
- **Angle**: Local Whisper transcription, works with local models

---

## Tier 7: Launch Platforms

### Hacker News (Show HN)
- **URL**: https://news.ycombinator.com/submit
- **Status**: Pending
- **Prerequisites**:
  - [ ] Working product (not landing page)
  - [ ] Easy to try without signup
- **Submission Plan**:
  1. Create post with title: `Show HN: claudetube – MCP server to let AI assistants watch YouTube videos`
  2. Put GitHub URL in URL field, leave text blank
  3. Immediately add comment with backstory:
     ```
     Hi HN! I built claudetube because I wanted Claude to be able to understand
     video content when helping me with projects.

     It's an MCP server that:
     - Downloads and transcribes videos from 1500+ sites
     - Extracts frames on-demand for visual analysis
     - Provides 40+ tools for Claude to query video content

     Technical details:
     - Uses yt-dlp for downloads
     - Whisper for transcription (with YouTube captions as fast fallback)
     - Smart caching so repeated queries are instant

     GitHub: https://github.com/thoughtpunch/claudetube
     PyPI: pip install claudetube

     Would love feedback on the API design and any use cases I haven't thought of!
     ```
  4. Use factual language, no marketing speak
  5. Be available to respond to comments
- **Rules**:
  - Don't use brand name as username
  - Product must be tryable
  - No signup walls

### Product Hunt
- **URL**: https://producthunt.com
- **Status**: Pending
- **Category**: AI Agents / Developer Tools
- **Submission Plan**:
  1. Create maker account
  2. Prepare assets: logo, screenshots, demo GIF
  3. Write tagline: "Let AI assistants watch YouTube videos"
  4. Schedule launch for Tuesday-Thursday
  5. Prepare for Q&A on launch day
- **Notes**: Consider finding a hunter with followers

### DevHunt
- **URL**: https://devhunt.org
- **Status**: Pending
- **Notes**: Developer-focused, open source friendly, free submissions

### BetaList
- **URL**: https://betalist.com
- **Status**: Pending
- **Members**: 500k+
- **Notes**: Good for early-stage tools

---

## Submission Templates

### Awesome List PR Description

```markdown
## Add claudetube MCP Server

**claudetube** - Let AI assistants watch YouTube videos via transcripts and on-demand frame extraction.

- GitHub: https://github.com/thoughtpunch/claudetube
- PyPI: https://pypi.org/project/claudetube/
- License: MIT

### Features
- Process videos from YouTube + 1500 other sites (via yt-dlp)
- Smart transcription (YouTube captions → Whisper fallback)
- On-demand frame extraction for visual analysis
- Scene segmentation and search
- 40+ MCP tools for video understanding
```

### Short Description (for forms)

> Let AI assistants watch YouTube videos - transcripts + on-demand frame extraction for visual analysis. Supports 1500+ sites via yt-dlp.

### One-liner

> MCP server that lets Claude watch and understand video content through transcripts and visual analysis.

### Technical One-liner (for HN)

> MCP server for AI video understanding: yt-dlp downloads, Whisper transcription, on-demand frame extraction

---

## Submission Order (Recommended)

1. **Week 1**: Official MCP Registry (requires npm publish first)
2. **Week 1**: mcpservers.org/submit (wong2 list)
3. **Week 1**: punkpeye/awesome-mcp-servers PR
4. **Week 2**: hesreallyhim/awesome-claude-code PR
5. **Week 2**: r/ClaudeAI post
6. **Week 2**: r/ClaudeCode post
7. **Week 3**: sindresorhus/awesome-whisper PR (if 20+ stars)
8. **Week 3**: Show HN post
9. **Week 4**: Product Hunt launch
10. **Ongoing**: Other awesome lists as PRs

---

*Last updated: 2026-02-04*
