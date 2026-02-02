---
description: Deep analysis of a video (OCR, entities, technical content)
argument-hint: <video_id> [question]
allowed-tools: ["Bash", "Read"]
---

# Deep Video Analysis

Run comprehensive deep analysis on a cached video, including OCR text extraction,
code block detection, and entity extraction (people, technologies, keywords).

This is more expensive than standard scene analysis. Results are cached for
subsequent queries.

## Input: $ARGUMENTS

Arguments: `<video_id> [question]`
- **video_id**: The video ID (e.g., `dQw4w9WgXcQ`)
- **question** (optional): A question to answer using the enriched analysis

Example: `/yt:deep dQw4w9WgXcQ What code is shown?`

## Step 1: Validate Cache

```bash
VIDEO_ID=$(echo "$ARGUMENTS" | awk '{print $1}')
QUESTION=$(echo "$ARGUMENTS" | cut -d' ' -f2-)
CACHE_DIR="$HOME/.claude/video_cache/$VIDEO_ID"

if [ ! -d "$CACHE_DIR" ]; then
    echo "ERROR: Video not cached. Run /yt <url> first."
    exit 1
fi

echo "VIDEO_ID: $VIDEO_ID"
echo "QUESTION: $QUESTION"
echo "CACHE_DIR: $CACHE_DIR"
```

## Step 2: Run Deep Analysis

```bash
~/.claudetube/venv/bin/python3 << 'PYTHON'
import json
import sys
sys.path.insert(0, '/Users/danielbarrett/sites/claudetube/src')

from claudetube.operations.analysis_depth import AnalysisDepth, analyze_video
from claudetube.config.loader import get_cache_dir

video_id = "$ARGUMENTS".split()[0].strip()

result = analyze_video(
    video_id,
    depth=AnalysisDepth.DEEP,
    force=False,
    output_base=get_cache_dir(),
)

data = result.to_dict()

if data.get("errors") and not data.get("scenes"):
    print(f"ERROR: {data['errors']}")
    sys.exit(1)

print(f"Depth: {data['depth']}")
print(f"Scenes analyzed: {data['scene_count']}")
print(f"Processing time: {data['processing_time']:.1f}s")
print(f"Method: {data['method']}")
print()

# Summarize what was found
tech_count = 0
entity_count = 0
ocr_text_count = 0
code_block_count = 0

for scene in data.get("scenes", []):
    tech = scene.get("technical", {})
    if tech:
        tech_count += 1
        ocr_text_count += len(tech.get("ocr_text", []))
        code_block_count += len(tech.get("code_blocks", []))

    ent = scene.get("entities", {})
    if ent:
        entity_count += 1

print(f"Scenes with technical content: {tech_count}")
print(f"OCR text regions: {ocr_text_count}")
print(f"Code blocks detected: {code_block_count}")
print(f"Scenes with entities: {entity_count}")
print()

# Show per-scene details
for scene in data.get("scenes", []):
    sid = scene.get("scene_id", "?")
    start = scene.get("start_time", 0)
    end = scene.get("end_time", 0)

    mins_s, secs_s = divmod(int(start), 60)
    mins_e, secs_e = divmod(int(end), 60)

    tech = scene.get("technical", {})
    ent = scene.get("entities", {})
    visual = scene.get("visual", {})

    parts = []
    if tech.get("ocr_text"):
        parts.append(f"OCR:{len(tech['ocr_text'])}")
    if tech.get("code_blocks"):
        parts.append(f"code:{len(tech['code_blocks'])}")
    if tech.get("content_types"):
        parts.append(f"types:{','.join(tech['content_types'])}")
    if ent.get("technologies"):
        parts.append(f"tech:{','.join(ent['technologies'][:5])}")
    if ent.get("people"):
        parts.append(f"people:{','.join(ent['people'][:3])}")
    if visual:
        parts.append("visual")

    detail = " | ".join(parts) if parts else "transcript only"
    title = scene.get("title", "")[:60]
    print(f"  Scene {sid} [{mins_s}:{secs_s:02d}-{mins_e}:{secs_e:02d}] {title}")
    print(f"    {detail}")

if data.get("errors"):
    print()
    print(f"Errors ({len(data['errors'])}):")
    for err in data["errors"]:
        print(f"  Scene {err.get('scene_id', '?')} ({err.get('stage', '?')}): {err.get('error', '')[:100]}")
PYTHON
```

## Output Format

Present results clearly:

```
Depth: deep
Scenes analyzed: 8
Processing time: 45.2s
Method: transcript

Scenes with technical content: 5
OCR text regions: 23
Code blocks detected: 3
Scenes with entities: 8

  Scene 0 [0:00-1:30] Introduction
    transcript only
  Scene 1 [1:30-5:45] Setting up the project
    OCR:5 | code:2 | types:code | tech:python,git
  Scene 2 [5:45-10:20] Implementing auth
    OCR:8 | code:1 | types:code,slides | tech:jwt,fastapi | people:John Doe
```

## After Deep Analysis

If a **question** was provided, use the enriched data (OCR text, code blocks,
entities, visual descriptions) to answer it thoroughly.

## Follow-up Actions

After deep analysis, users can:
- `/yt:focus <video_id> <start_time> <end_time>` - Exhaustive analysis of a specific section
- `/yt:see <video_id> <timestamp>` - View frames at specific moments
- `/yt:hq <video_id> <timestamp>` - HQ frames for code/text at key moments
- `/yt:watch <video_id> <question>` - Ask questions with evidence-backed reasoning
