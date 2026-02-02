[← Documentation](../README.md)

# Video Understanding for LLMs

> How claudetube enables AI to "watch" video content.

## The Core Challenge

LLMs process text and images, but video is neither. Video is:
- **Temporal**: Information unfolds over time
- **Multi-modal**: Audio, visual, and text streams
- **Dense**: Hours of content, megabytes of data
- **Contextual**: Meaning depends on what came before

claudetube bridges this gap by decomposing video into LLM-friendly components.

## The Decomposition Strategy

### Audio Stream → Transcript

Speech is converted to timestamped text:

```
[00:00:05] "Let me show you the authentication flow."
[00:00:08] "First, we'll look at the middleware."
[00:00:12] "This function validates the JWT token."
```

The LLM can read this like a script, understanding the narrative arc.

### Visual Stream → Frames

Key moments become static images:

```
Frame at 00:00:08 → [Image: Code editor showing middleware.js]
Frame at 00:00:15 → [Image: Browser showing login form]
Frame at 00:00:22 → [Image: Terminal with auth logs]
```

The LLM can "see" what the speaker references.

### Metadata → Context

Structured information provides scaffolding:

```json
{
  "title": "Building Auth in Express.js",
  "duration": 1847,
  "chapters": [
    {"title": "Introduction", "start": 0},
    {"title": "JWT Basics", "start": 120},
    {"title": "Middleware Setup", "start": 480}
  ]
}
```

## The Interaction Model

### 1. Question Arrives

User asks: "How does the auth middleware validate tokens?"

### 2. Transcript Scan

Claude searches the transcript for relevant sections:
- "middleware" at 00:00:08, 00:05:30, 00:12:45
- "validate" at 00:00:12, 00:06:15
- "token" at 00:00:12, 00:06:20, 00:07:00

### 3. Frame Request

Claude requests frames at promising timestamps:
```
get_frames(video_id, start_time=360, duration=10)
```

### 4. Visual Analysis

Claude examines the frames:
- Sees code on screen
- Reads function names, variable names
- Understands the visual context

### 5. Synthesized Answer

Claude combines transcript + visuals:
> "At 6:15, the video shows the `validateToken` function in `middleware.js`. It extracts the JWT from the Authorization header, verifies it with the secret key, and attaches the decoded payload to `req.user`. Here's the relevant code..."

## Quality Tiers

Not all visual analysis needs the same fidelity:

| Tier | Resolution | Use Case |
|------|------------|----------|
| Lowest | 320px | Presence detection (is someone on screen?) |
| Low | 480px | General context (what's happening?) |
| Medium | 720px | Standard analysis (read large text) |
| High | 1080px | Detail work (read code, small text) |
| Highest | 1280px+ | Precision (read terminal output, tiny UI) |

claudetube defaults to 480px for speed, with HQ extraction (1280px) available on demand.

## The Progressive Approach

Efficient video understanding follows a funnel:

```
┌─────────────────────────────────────────┐
│  1. Metadata scan (instant)              │
│     - Duration, chapters, title          │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│  2. Transcript read (fast)               │
│     - Full text, keyword search          │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│  3. Targeted frames (on-demand)          │
│     - Specific timestamps only           │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│  4. HQ frames (when needed)              │
│     - Code, diagrams, small text         │
└─────────────────────────────────────────┘
```

This minimizes compute while maximizing understanding.

---

**See also**:
- [The Pipeline](pipeline.md) - How data flows through the system
- [Transcripts](transcripts.md) - Speech-to-text details
- [Frames](frames.md) - Visual extraction
