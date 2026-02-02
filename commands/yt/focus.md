---
description: Exhaustive frame-by-frame analysis of a specific video section
argument-hint: <video_id> <start_time> [end_time]
allowed-tools: ["Bash", "Read"]
---

# Focused Exhaustive Analysis

Perform exhaustive frame-by-frame analysis on a specific section of a video.
This is the most expensive analysis mode - use it for detailed investigation
of specific moments (e.g., code demos, bug introductions, key explanations).

Extracts frames at 1-second intervals and runs OCR + technical analysis on each.

## Input: $ARGUMENTS

Arguments: `<video_id> <start_time> [end_time]`
- **video_id**: The video ID (e.g., `dQw4w9WgXcQ`)
- **start_time**: Start time in seconds (e.g., `120` for 2:00)
- **end_time**: End time in seconds (default: start_time + 30)

Example: `/yt:focus dQw4w9WgXcQ 120 180`

## Step 1: Validate Cache and Parse Arguments

```bash
VIDEO_ID=$(echo "$ARGUMENTS" | awk '{print $1}')
START_TIME=$(echo "$ARGUMENTS" | awk '{print $2}')
END_TIME=$(echo "$ARGUMENTS" | awk '{print $3}')
CACHE_DIR="$HOME/.claude/video_cache/$VIDEO_ID"

if [ ! -d "$CACHE_DIR" ]; then
    echo "ERROR: Video not cached. Run /yt <url> first."
    exit 1
fi

if [ -z "$START_TIME" ]; then
    echo "ERROR: start_time is required. Usage: /yt:focus <video_id> <start_time> [end_time]"
    exit 1
fi

if [ ! -f "$CACHE_DIR/scenes/scenes.json" ]; then
    echo "ERROR: Video has no scene data. Run /yt:scenes $VIDEO_ID first."
    exit 1
fi

echo "VIDEO_ID: $VIDEO_ID"
echo "START_TIME: $START_TIME"
echo "END_TIME: ${END_TIME:-auto}"
echo "CACHE_DIR: $CACHE_DIR"
```

## Step 2: Run Exhaustive Analysis

```bash
~/.claudetube/venv/bin/python3 << 'PYTHON'
import json
import sys
sys.path.insert(0, '/Users/danielbarrett/sites/claudetube/src')

from claudetube.cache.scenes import load_scenes_data
from claudetube.operations.analysis_depth import AnalysisDepth, analyze_video
from claudetube.config.loader import get_cache_dir

args = "$ARGUMENTS".split()
video_id = args[0].strip()
start_time = float(args[1]) if len(args) > 1 else 0
end_time = float(args[2]) if len(args) > 2 else start_time + 30

cache_dir = get_cache_dir() / video_id

# Find scenes in time range
scenes_data = load_scenes_data(cache_dir)
if not scenes_data:
    print("ERROR: No scenes found. Run /yt:scenes first.")
    sys.exit(1)

# Find overlapping scenes
focus_ids = [
    s.scene_id
    for s in scenes_data.scenes
    if not (s.end_time < start_time or s.start_time > end_time)
]

if not focus_ids:
    print(f"ERROR: No scenes found between {start_time}s and {end_time}s")
    print(f"Video has {len(scenes_data.scenes)} scenes:")
    for s in scenes_data.scenes:
        mins_s, secs_s = divmod(int(s.start_time), 60)
        mins_e, secs_e = divmod(int(s.end_time), 60)
        print(f"  Scene {s.scene_id}: {mins_s}:{secs_s:02d}-{mins_e}:{secs_e:02d}")
    sys.exit(1)

mins_s, secs_s = divmod(int(start_time), 60)
mins_e, secs_e = divmod(int(end_time), 60)
print(f"Focus range: {mins_s}:{secs_s:02d} - {mins_e}:{secs_e:02d}")
print(f"Scenes in range: {focus_ids}")
print()

result = analyze_video(
    video_id,
    depth=AnalysisDepth.EXHAUSTIVE,
    focus_sections=focus_ids,
    force=False,
    output_base=get_cache_dir(),
)

data = result.to_dict()

if data.get("errors") and not data.get("scenes"):
    print(f"ERROR: {data['errors']}")
    sys.exit(1)

print(f"Depth: {data['depth']}")
print(f"Processing time: {data['processing_time']:.1f}s")
print(f"Focus sections: {data.get('focus_sections', [])}")
print()

# Show detailed per-scene results
for scene in data.get("scenes", []):
    sid = scene.get("scene_id", "?")
    if sid not in focus_ids:
        continue

    s_start = scene.get("start_time", 0)
    s_end = scene.get("end_time", 0)
    mins_s, secs_s = divmod(int(s_start), 60)
    mins_e, secs_e = divmod(int(s_end), 60)

    title = scene.get("title", "")[:60]
    print(f"=== Scene {sid} [{mins_s}:{secs_s:02d}-{mins_e}:{secs_e:02d}] {title} ===")

    # Visual description
    visual = scene.get("visual", {})
    if visual:
        desc = visual.get("description", "")[:200]
        if desc:
            print(f"  Visual: {desc}")

    # Technical content
    tech = scene.get("technical", {})
    if tech:
        ocr_texts = tech.get("ocr_text", [])
        if ocr_texts:
            print(f"  OCR text ({len(ocr_texts)} regions):")
            for text in ocr_texts[:5]:
                print(f"    - {text[:120]}")
            if len(ocr_texts) > 5:
                print(f"    ... and {len(ocr_texts) - 5} more")

        code_blocks = tech.get("code_blocks", [])
        if code_blocks:
            print(f"  Code blocks ({len(code_blocks)}):")
            for block in code_blocks[:3]:
                lang = block.get("language", "unknown")
                snippet = block.get("code", "")[:100]
                print(f"    [{lang}] {snippet}")

        content_types = tech.get("content_types", [])
        if content_types:
            print(f"  Content types: {', '.join(content_types)}")

    # Entities
    ent = scene.get("entities", {})
    if ent:
        parts = []
        if ent.get("technologies"):
            parts.append(f"tech: {', '.join(ent['technologies'][:5])}")
        if ent.get("people"):
            parts.append(f"people: {', '.join(ent['people'][:3])}")
        if ent.get("keywords"):
            parts.append(f"keywords: {', '.join(ent['keywords'][:5])}")
        if parts:
            print(f"  Entities: {' | '.join(parts)}")

    # Frame-by-frame analysis
    frames = scene.get("frame_analysis", [])
    if frames:
        print(f"  Frame analysis ({len(frames)} frames):")
        for frame in frames:
            ts = frame.get("timestamp", 0)
            mins_f, secs_f = divmod(int(ts), 60)
            ocr = frame.get("ocr_text", [])
            ct = frame.get("content_type", "")
            parts = []
            if ct:
                parts.append(ct)
            if ocr:
                parts.append(f"text:{len(ocr)}")
            detail = " | ".join(parts) if parts else "no text"
            print(f"    [{mins_f}:{secs_f:02d}] {detail}")

    print()

if data.get("errors"):
    print(f"Errors ({len(data['errors'])}):")
    for err in data["errors"]:
        print(f"  Scene {err.get('scene_id', '?')} ({err.get('stage', '?')}): {err.get('error', '')[:100]}")
PYTHON
```

## Output Format

Present results clearly:

```
Focus range: 2:00 - 3:00
Scenes in range: [3, 4]

Depth: exhaustive
Processing time: 120.5s
Focus sections: [3, 4]

=== Scene 3 [1:45-2:30] Implementing the fix ===
  Visual: Developer typing in VS Code with Python file open
  OCR text (3 regions):
    - def authenticate(token):
    - if token.expiry <= datetime.now():
    - return False
  Code blocks (1):
    [python] def authenticate(token): ...
  Content types: code
  Entities: tech: python, jwt | people: John
  Frame analysis (45 frames):
    [1:45] code | text:2
    [1:46] code | text:3
    ...
```

## Follow-up Actions

After focused analysis, users can:
- `/yt:hq <video_id> <timestamp>` - HQ frames for reading code/text
- `/yt:see <video_id> <timestamp>` - Quick visual frames
- `/yt:deep <video_id>` - Full video deep analysis
- `/yt:watch <video_id> <question>` - Ask questions with evidence
