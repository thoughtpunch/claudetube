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
    scene_id: int                             # 0-based index
    start_time: float                         # 120.5 (seconds)
    end_time: float                           # 245.0 (seconds)
    title: str | None = None                  # Scene title/description
    transcript: list[dict] = []               # Segments with timestamps
    transcript_text: str = ""                 # Joined transcript text

@dataclass
class ScenesData:
    video_id: str
    method: str                               # "transcript", "visual", or "hybrid"
    scenes: list[SceneBoundary] = []
```

## Cache Structure

```
~/.claude/video_cache/{video_id}/
├── scenes/
│   ├── scenes.json           # Scene boundaries + metadata
│   ├── scene_000/            # First scene
│   │   ├── keyframes/        # Representative frames
│   │   ├── visual.json       # Visual description
│   │   ├── technical.json    # OCR, code detection
│   │   └── entities.json     # Extracted entities
│   ├── scene_001/
│   │   └── ...
│   └── scene_002/
│       └── ...
```

## API Usage

### Get Scenes

```python
from claudetube.cache.scenes import load_scenes_data

scenes_data = load_scenes_data(cache_dir)
if scenes_data:
    for scene in scenes_data.scenes:
        print(f"Scene {scene.scene_id}: {scene.start_time}s - {scene.end_time}s")
        if scene.title:
            print(f"  Title: {scene.title}")
```

### MCP

```
get_scenes(video_id, enrich=False)
# Returns list of scenes with timestamps, transcripts, and visual descriptions
```

### Slash Command

```
/yt:scenes <video_id>
# Returns scene structure with timestamps and titles
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

Scenes are **implemented**:
- [x] YouTube chapters extraction
- [x] Cache structure for scenes
- [x] Transcript-based boundary detection (smart segmentation)
- [x] Visual scene change detection
- [x] `get_scenes` MCP tool
- [x] `/yt:scenes` slash command
- [x] Visual transcript generation per scene
- [x] Entity extraction per scene

See the [Roadmap](../vision/roadmap.md) for progress.

---

**See also**:
- [Video Understanding](video-understanding.md) - The bigger picture
- [Frames](frames.md) - Visual extraction per scene
