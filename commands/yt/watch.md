---
description: Actively watch and reason about a video to answer a question
argument-hint: <video_id> <question>
allowed-tools: ["Bash", "Read"]
---

# Active Video Watching

Actively watch a video to answer a question with evidence-backed reasoning.

Uses an intelligent watching strategy:
1. Checks cached Q&A for previously answered questions
2. Identifies most relevant scenes via attention modeling
3. Examines them progressively (quick transcript, then deep visual)
4. Builds hypotheses and gathers evidence
5. Verifies comprehension before answering

This is the most thorough analysis mode.

## Input: $ARGUMENTS

Arguments: `<video_id> <question>`
- **video_id**: The video ID (e.g., `dQw4w9WgXcQ`)
- **question**: Natural language question about the video

Example: `/yt:watch dQw4w9WgXcQ What bug was fixed and how?`

## Step 1: Validate Cache

```bash
VIDEO_ID=$(echo "$ARGUMENTS" | awk '{print $1}')
QUESTION=$(echo "$ARGUMENTS" | cut -d' ' -f2-)
CACHE_DIR="$HOME/.claude/video_cache/$VIDEO_ID"

if [ ! -d "$CACHE_DIR" ]; then
    echo "ERROR: Video not cached. Run /yt <url> first."
    exit 1
fi

if [ ! -f "$CACHE_DIR/scenes/scenes.json" ]; then
    echo "ERROR: Video has no scene data. Run /yt:scenes $VIDEO_ID first."
    exit 1
fi

echo "VIDEO_ID: $VIDEO_ID"
echo "QUESTION: $QUESTION"
echo "CACHE_DIR: $CACHE_DIR"
```

## Step 2: Active Watch

```bash
~/.claudetube/venv/bin/python3 << 'PYTHON'
import json
import sys
sys.path.insert(0, '/Users/danielbarrett/sites/claudetube/src')

from claudetube.operations.watch import watch_video

video_id = "$ARGUMENTS".split()[0].strip()
question = " ".join("$ARGUMENTS".split()[1:]).strip()

result = watch_video(video_id, question)

if "error" in result:
    print(f"ERROR: {result['error']}")
    sys.exit(1)

# Display answer
print(f"Question: {result['question']}")
print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']:.0%}")
print(f"Scenes examined: {result['scenes_examined']}")
print(f"Comprehension verified: {result['comprehension_verified']}")
print()

# Display evidence
evidence = result.get('evidence', [])
if evidence:
    print("Evidence:")
    for e in evidence:
        ts = e.get('timestamp')
        obs = e.get('observation', '')
        if ts is not None:
            from claudetube.analysis.search import format_timestamp
            print(f"  [{format_timestamp(ts)}] {obs[:150]}")
        else:
            print(f"  {obs[:150]}")
    print()

# Display alternatives
alts = result.get('alternative_interpretations', [])
if alts:
    print("Alternative interpretations:")
    for alt in alts:
        print(f"  - {alt[:150]}")
    print()

# Display examination log
log = result.get('examination_log', [])
if log:
    print("Examination log:")
    for entry in log:
        depth = "DEEP" if entry['depth'] == 'examine_deep' else "quick"
        print(f"  #{entry['iteration']}: Scene {entry['scene_id']} "
              f"({depth}) at {entry['timestamp']} "
              f"-> {entry['findings_count']} findings")

# Source info
source = result.get('source', '')
if source == 'cached_qa':
    print("\n(Answer from cached Q&A - previously answered)")
PYTHON
```

## Output Format

Present results clearly:

```
Question: What bug was fixed and how?
Answer: An off-by-one error in the authentication loop was fixed by changing < to <=
Confidence: 85%
Scenes examined: 5
Comprehension verified: true

Evidence:
  [4:05] Code shows loop with < operator
  [5:12] Fix applied: changed to <=, test passes

Alternative interpretations:
  - Could also be a race condition fix mentioned earlier

Examination log:
  #0: Scene 3 (quick) at 3:45 -> 2 findings
  #1: Scene 5 (DEEP) at 4:05 -> 3 findings
```

## Follow-up Actions

After watching, users can:
- `/yt:see <video_id> <timestamp>` - View frames at evidence timestamps
- `/yt:hq <video_id> <timestamp>` - HQ frames for code/text at key moments
- `/yt:find <video_id> <query>` - Find additional relevant moments
- `/yt:transcript <video_id>` - Read full transcript for context
