# Scenes

> Semantic segmentation of video content.

## What Are Scenes?

A scene is a semantically coherent segment of video:
- A chapter in a tutorial
- A topic in a presentation
- A section of a code walkthrough
- A distinct visual sequence

Scenes transform a linear video into a structured document.

## Why Scenes Matter

Without scenes, you have:
- A flat transcript with no structure
- No way to know where topics begin/end
- Must read everything to find anything

With scenes, you have:
- Structured content with clear boundaries
- Topic-level navigation
- Targeted frame extraction
- Efficient question answering

## Scene Detection Methods

### 1. YouTube Chapters (Best)

If the video has chapters, use them:

```json
{
  "chapters": [
    {"title": "Introduction", "start_time": 0, "end_time": 45},
    {"title": "Project Setup", "start_time": 45, "end_time": 180},
    {"title": "Writing the Code", "start_time": 180, "end_time": 600}
  ]
}
```

These are author-defined and semantically meaningful.

### 2. Transcript Analysis (Fast)

Detect boundaries from transcript patterns:
- Long pauses (silence)
- Topic shifts (vocabulary change)
- Transition phrases ("Now let's look at...", "Moving on to...")

### 3. Visual Detection (Accurate)

Detect visual scene changes:
- Cut detection (frame similarity drop)
- Content change (slide transitions, camera switches)

Uses PySceneDetect or similar tools.

## Scene Data Structure

```python
@dataclass
class SceneBoundary:
    start_time: float      # 120.5
    end_time: float        # 245.0
    method: str            # "chapters" | "transcript" | "visual"
    confidence: float      # 0.0-1.0

@dataclass
class ScenesData:
    video_id: str
    boundaries: list[SceneBoundary]
    generated_at: str      # ISO timestamp
    source: str            # "chapters" | "auto"
```

## Cache Structure

```
~/.claude/video_cache/{video_id}/
├── scenes/
│   ├── scenes.json           # Scene boundaries
│   ├── scene_000/            # First scene
│   │   ├── keyframe.jpg      # Representative frame
│   │   ├── transcript.txt    # Scene transcript chunk
│   │   └── visual.json       # Visual description (optional)
│   ├── scene_001/
│   │   └── ...
│   └── scene_002/
│       └── ...
```

## API Usage

### Get Scenes

```python
from claudetube.cache import get_scenes

scenes = get_scenes(cache_dir, video_id)
if scenes:
    for i, boundary in enumerate(scenes.boundaries):
        print(f"Scene {i}: {boundary.start_time}s - {boundary.end_time}s")
```

### MCP (Planned)

```
/yt:scenes video_id
# Returns list of scenes with timestamps and titles
```

## Scene-Aware Workflows

### Question Answering

1. Get scenes for the video
2. Match question to relevant scene(s) via transcript search
3. Extract frames from those scenes only
4. Answer with scene context

### Summarization

1. Get scenes for the video
2. Summarize each scene independently
3. Combine scene summaries into video summary

### Navigation

User: "Skip to the part about authentication"
1. Search scenes for "authentication"
2. Return timestamp of matching scene
3. Extract frames from that scene

## Status

Scenes are **partially implemented**:
- [x] YouTube chapters extraction
- [x] Cache structure for scenes
- [ ] Transcript-based boundary detection
- [ ] Visual scene detection
- [ ] /yt:scenes MCP command

See the [Roadmap](../vision/roadmap.md) for progress.

---

**See also**:
- [Video Understanding](video-understanding.md) - The bigger picture
- [Frames](frames.md) - Visual extraction per scene
