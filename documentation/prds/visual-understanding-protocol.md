# PRD: Visual Understanding Protocol

## Overview

**Problem**: AI assistants processing video content often rely solely on transcripts, missing critical visual information. This leads to incomplete understanding and lower-quality outputs, especially for educational, technical, and visual-heavy content.

**Observation**: When asked to synthesize a 3Blue1Brown series on neural networks, Claude processed only transcripts despite:
- The source being famous for mathematical visualizations
- The transcript containing 100+ visual reference phrases
- The concepts being inherently geometric (embeddings, vector fields, attention patterns)

A human researcher would naturally switch to visual inspection when text descriptions feel insufficient. We need to make the AI's gradient flow toward visual understanding when context warrants it.

**Goal**: Automatically detect when visual analysis is needed and either perform it proactively or surface strong recommendations that guide Claude toward visual verification.

---

## Approach: LLM-Based Visual Criticality Assessment

Rather than complex heuristics, we ask an LLM to directly assess visual criticality. This is more nuanced, handles edge cases, and provides explainable reasoning.

### Assessment Prompt

```
Given this video:
- Title: {title}
- Channel: {channel}
- Description: {description}
- Transcript excerpt: {first 2000 chars}
- Task: {task or "general understanding"}

How critical are the visuals for understanding this content, given the task?

Consider:
- Is this educational/technical content where diagrams, animations, or demonstrations matter?
- Does the speaker reference things on screen ("as you can see", "look at this")?
- Are there likely code snippets, charts, mathematical visualizations, or physical demonstrations?
- Would someone listening without video miss significant information for THIS task?
- For coding tutorials: code syntax IS visual information
- For music videos: depends on task (lyrics analysis vs. experiencing the art)

Respond with JSON:
{
  "visual_criticality": <0-10>,
  "confidence": "low|medium|high",
  "reasoning": "<1-2 sentences explaining why>",
  "likely_visual_elements": ["<list of expected visual content>"],
  "recommended_action": "none|suggested|recommended|required"
}
```

### Task-Dependent Scoring

The same video may have different visual criticality depending on the task:

| Video Type | Task | Score |
|------------|------|-------|
| Music video | Analyze lyrics | 2 |
| Music video | Study cinematography | 10 |
| Music video | Experience as cultural artifact | 8 |
| Coding tutorial | Understand concepts | 9 |
| Coding tutorial | Get code snippets | 10 |
| Interview | Understand discussion | 2-3 |
| Interview | Analyze body language | 8 |

### Assessment Thresholds

| Score | Action |
|-------|--------|
| 0-3 | No visual analysis needed |
| 4-6 | Visual analysis suggested |
| 7-8 | Visual analysis recommended |
| 9-10 | Visual analysis required (auto-trigger) |

---

## Validation: Test Results

The LLM assessment approach was tested against 6 videos of varying types:

### Test Matrix

| Video | Type | Channel | Score | Action | Key Insight |
|-------|------|---------|-------|--------|-------------|
| Attention in transformers | Math education | 3Blue1Brown | 10 | required | Animations ARE the teaching method |
| LLMs explained briefly | Conceptual education | 3Blue1Brown | 8 | recommended | Educational + known visual channel |
| Nathan For You clip | Comedy | Comedy Central | 6 | suggested | Reactions enhance but transcript works |
| React in 100 Seconds | Coding tutorial | Fireship | 9 | required | Code syntax must be seen |
| GPT-3 Interview | Talking head | Eric Elliott | 3 | none | All content is in dialogue |
| Despacito | Music video | Luis Fonsi | 7* | suggested | *Task-dependent (lyrics=2, art=8) |

### Key Findings

1. **Math/science education scores highest (9-10)**: For channels like 3Blue1Brown, the animations ARE the explanation. Grant Sanderson's teaching method is inherently visual - you cannot understand attention mechanisms from transcript alone.

2. **Coding tutorials score very high (9)**: Code syntax IS visual information. Even without explicit "as you can see" phrases, you cannot learn React without seeing JSX syntax highlighting, component structure, and live demos.

3. **Talking-head interviews score lowest (2-3)**: When the host explicitly explains the visual is just an AI avatar, and all content is Q&A dialogue, visuals are decoration not information.

4. **Music videos are task-dependent**: Reveals need for task parameter in assessment:
   - "Analyze the lyrics" → score 2
   - "Experience the cultural artifact" → score 8
   - "Study music video cinematography" → score 10

5. **Comedy is nuanced (5-7)**: Nathan For You humor comes partly from awkward reactions and timing, but the transcript alone is still funny. Visuals enhance but aren't required.

6. **Channel recognition matters**: The LLM correctly identifies that Fireship is visual-dense even without explicit visual references in transcript. It knows the channel's format from training data.

### Edge Cases Handled Well

- **"Painting" ambiguity**: A video titled "Painting in JavaScript" would be correctly identified as coding (canvas API) vs art
- **Podcast about visualizations**: Discussion ABOUT visual topics but purely audio → low score
- **AI-generated avatar interviews**: Visuals are synthetic/decorative, content is text → low score

---

### Why LLM Assessment > Heuristics

| Heuristic Approach | LLM Assessment |
|-------------------|----------------|
| Counts keywords blindly | Understands context |
| Needs maintained keyword lists | Self-updating with model improvements |
| Can't handle edge cases | Handles nuance naturally |
| "painting" → art? code? | Knows from description/transcript context |
| Opaque scoring | Provides human-readable reasoning |
| False positives on keyword matches | Understands semantic intent |

### Implementation

```python
async def assess_visual_criticality(
    title: str,
    channel: str,
    description: str,
    transcript_excerpt: str,
    task: str = "general understanding",
    model: str = "haiku"  # Fast, cheap model for assessment
) -> VisualAssessment:
    """
    Ask an LLM to assess how critical visuals are for understanding.

    Uses a small/fast model to keep costs low.
    Results are cached with the video metadata.

    Args:
        task: The user's goal, e.g. "learn React basics", "extract code snippets",
              "analyze lyrics", "comprehensive educational summary"
    """
    prompt = VISUAL_ASSESSMENT_PROMPT.format(
        title=title,
        channel=channel,
        description=description,
        transcript_excerpt=transcript_excerpt[:2000],
        task=task
    )

    response = await llm.complete(prompt, model=model, response_format="json")

    return VisualAssessment(
        score=response["visual_criticality"],
        confidence=response["confidence"],
        reasoning=response["reasoning"],
        likely_elements=response["likely_visual_elements"],
        recommended_action=response["recommended_action"]
    )
```

### Cost Optimization

- **Model**: Use haiku/small model (~$0.0001 per assessment)
- **Caching**: Store assessment in video state.json, never recompute
- **Batching**: Can assess multiple videos in parallel
- **Lazy**: Only assess when video is processed, not on every query

---

## Fallback: Heuristic Signals

For offline/fast-path scenarios, or to supplement LLM assessment:

### Content Type Signals (High Confidence)

Content in these categories should be assumed to have **high visual information density**:

| Content Type | Confidence | Rationale |
|--------------|------------|-----------|
| Educational | Very High | Teaching relies on diagrams, animations, demonstrations |
| Technical tutorials | Very High | Code, architectures, system diagrams |
| Scientific/mathematical | Very High | Graphs, proofs, visualizations |
| How-it-works explainers | Very High | Animations, cutaways, process flows |
| Product demos | High | UI/UX, feature walkthroughs |
| Conference talks | High | Slides, diagrams, live demos |
| Documentaries | Medium-High | B-roll, archival footage, visualizations |

### Source Signals (Channel Heuristics)

Certain channels/creators are inherently visual-first:

**Always extract frames:**
- 3Blue1Brown
- Veritasium
- Numberphile
- Welch Labs
- Computerphile
- Two Minute Papers
- Kurzgesagt
- CGP Grey
- SmarterEveryDay
- Primer

**High likelihood (tech/coding):**
- Fireship
- Theo - t3.gg
- ThePrimeagen
- Traversy Media
- Tech With Tim

### Transcript Signals (Phrase Detection)

**Direct visual references:**
- "as you can see"
- "look at this"
- "notice how"
- "in this animation"
- "the diagram shows"
- "shown here"
- "visualize this as"
- "picture"
- "on screen"
- "watch what happens"

**Visual element references:**
- "diagram"
- "chart"
- "graph"
- "plot"
- "animation"
- "figure"
- "illustration"
- "screenshot"
- "demo"

**Spatial/geometric language:**
- "direction"
- "vector"
- "dimension"
- "space"
- "axis"
- "coordinate"
- "gradient"
- "slope"
- "curve"
- "surface"

**Color references:**
- "the blue/red/green..."
- "colored"
- "highlighted"
- "shaded"

**Comparative visual language:**
- "bigger/smaller"
- "left/right"
- "above/below"
- "before/after"
- "side by side"

### Task Signals (Intent Detection)

| Task Type | Visual Need |
|-----------|-------------|
| Educational synthesis | Always |
| Summarization for learning | Always |
| Technical documentation | High |
| Code extraction from video | Very High |
| Q&A about visual elements | Required |
| General summary | Medium |
| Transcript extraction only | Low |

---

## Visual Signal Score

Compute a composite score to determine visual analysis need:

```python
def compute_visual_signal_score(
    transcript: str,
    metadata: dict,
    task: str | None = None
) -> VisualSignalResult:
    """
    Compute visual signal score from 0.0 to 1.0.

    Returns:
        VisualSignalResult with score, confidence, reasons, and recommended timestamps.
    """
    score = 0.0
    reasons = []

    # Source signals (0-0.4)
    channel = metadata.get("channel", "").lower()
    if channel in ALWAYS_VISUAL_CHANNELS:
        score += 0.4
        reasons.append(f"Source '{channel}' is known for visual content")
    elif channel in HIGH_VISUAL_CHANNELS:
        score += 0.25
        reasons.append(f"Source '{channel}' typically has visual content")

    # Content type signals (0-0.3)
    title = metadata.get("title", "").lower()
    description = metadata.get("description", "").lower()

    if any(kw in title for kw in EDUCATIONAL_KEYWORDS):
        score += 0.3
        reasons.append("Title indicates educational content")
    elif any(kw in title for kw in TECHNICAL_KEYWORDS):
        score += 0.25
        reasons.append("Title indicates technical content")

    # Transcript signals (0-0.3)
    transcript_lower = transcript.lower()

    visual_phrase_count = sum(
        transcript_lower.count(phrase)
        for phrase in VISUAL_REFERENCE_PHRASES
    )

    if visual_phrase_count > 20:
        score += 0.3
        reasons.append(f"Transcript contains {visual_phrase_count} visual references")
    elif visual_phrase_count > 10:
        score += 0.2
        reasons.append(f"Transcript contains {visual_phrase_count} visual references")
    elif visual_phrase_count > 5:
        score += 0.1
        reasons.append(f"Transcript contains {visual_phrase_count} visual references")

    # Geometric/spatial language (0-0.2)
    spatial_count = sum(
        transcript_lower.count(term)
        for term in SPATIAL_TERMS
    )

    if spatial_count > 30:
        score += 0.2
        reasons.append(f"High density of spatial/geometric language ({spatial_count} terms)")
    elif spatial_count > 15:
        score += 0.1
        reasons.append(f"Moderate spatial/geometric language ({spatial_count} terms)")

    # Task signals (0-0.2)
    if task:
        task_lower = task.lower()
        if any(kw in task_lower for kw in ["educational", "learning", "teach", "explain"]):
            score += 0.2
            reasons.append("Task is educational in nature")
        elif any(kw in task_lower for kw in ["synthesize", "summarize", "comprehensive"]):
            score += 0.15
            reasons.append("Task requires comprehensive understanding")

    # Cap at 1.0
    score = min(score, 1.0)

    return VisualSignalResult(
        score=score,
        recommended=score > 0.5,
        strongly_recommended=score > 0.75,
        reasons=reasons,
        suggested_timestamps=find_high_signal_timestamps(transcript)
    )
```

### Thresholds

| Score | Recommendation |
|-------|----------------|
| 0.0 - 0.3 | Visual analysis optional |
| 0.3 - 0.5 | Visual analysis suggested |
| 0.5 - 0.75 | Visual analysis recommended |
| 0.75 - 1.0 | Visual analysis strongly recommended / automatic |

---

## Implementation Layers

### Layer 1: Enhanced Tool Response

Modify `process_video_tool` to include visual signal analysis in its response.

**Changes to `mcp_server.py`:**

```python
@mcp.tool()
async def process_video_tool(url: str, ...) -> dict:
    # ... existing processing ...

    # NEW: Compute visual signal score
    visual_signals = compute_visual_signal_score(
        transcript=transcript_text,
        metadata={
            "channel": metadata.get("channel"),
            "title": metadata.get("title"),
            "description": metadata.get("description"),
        }
    )

    return {
        # ... existing fields ...

        # NEW: Visual analysis guidance
        "visual_analysis": {
            "recommended": visual_signals.recommended,
            "strongly_recommended": visual_signals.strongly_recommended,
            "score": visual_signals.score,
            "reasons": visual_signals.reasons,
            "suggested_timestamps": visual_signals.suggested_timestamps,
        }
    }
```

**Benefits:**
- Information is always available
- No additional API calls needed
- Claude sees the recommendation naturally

### Layer 2: CLAUDE.md Protocol

Add explicit guidance to project CLAUDE.md:

```markdown
## Visual Understanding Protocol

When processing video content, Claude MUST assess visual understanding needs.

### Automatic Visual Analysis Required

Extract frames WITHOUT asking when ANY of these apply:
- Source is: 3Blue1Brown, Veritasium, Numberphile, Kurzgesagt, Welch Labs
- Content type is: educational, tutorial, technical explanation, how-it-works
- Task involves: learning synthesis, educational content creation, comprehensive summary
- `visual_analysis.strongly_recommended` is true in tool response

### Visual Analysis Recommended

Extract frames UNLESS time-constrained when:
- Transcript contains 10+ visual reference phrases
- Topic involves geometry, physics, mathematics, or spatial concepts
- `visual_analysis.recommended` is true in tool response

### Visual Reference Phrases (trigger words)

When you see these in transcripts, the visual IS the explanation:
- "as you can see", "look at this", "notice how"
- "diagram", "chart", "graph", "animation"
- Color references: "the blue vector", "green arrows"
- Spatial language: "left", "right", "above", "below"

### Self-Check Question

Before synthesizing content from video, ask yourself:
> "Would a human researcher look at the visuals here?"

If yes, use `get_frames` on high-signal timestamps.
```

### Layer 3: Post-Processing Hook

Create a Claude Code hook that fires after video processing:

**File: `.claude/hooks/visual-reminder.js`**

```javascript
// Hook: Post-tool reminder for visual analysis
module.exports = {
  event: "post_tool_result",
  match: (result) => result.tool === "mcp__claudetube__process_video_tool",

  handler: async (result) => {
    const visual = result.output?.visual_analysis;

    if (!visual) return null;

    if (visual.strongly_recommended) {
      return {
        type: "system_reminder",
        content: `
⚠️ VISUAL ANALYSIS STRONGLY RECOMMENDED

Score: ${visual.score.toFixed(2)}
Reasons:
${visual.reasons.map(r => `  • ${r}`).join('\n')}

Suggested timestamps: ${visual.suggested_timestamps.join(', ')}

This content likely has HIGH VISUAL INFORMATION DENSITY.
Use get_frames or get_scenes before synthesizing.
        `.trim()
      };
    }

    if (visual.recommended) {
      return {
        type: "system_reminder",
        content: `ℹ️ Visual analysis recommended (score: ${visual.score.toFixed(2)}). Consider using get_frames for timestamps: ${visual.suggested_timestamps.slice(0, 5).join(', ')}`
      };
    }

    return null;
  }
};
```

### Layer 4: Smart Skill (Optional)

Create a `/deep-watch` skill that bundles comprehensive analysis:

```markdown
# /deep-watch skill

Performs comprehensive video analysis with automatic visual understanding.

## Workflow:
1. Process video with process_video_tool
2. Analyze visual signal score
3. If score > 0.5: automatically get_scenes
4. For high-signal scenes: extract frames
5. Synthesize understanding from transcript + visuals
6. Return comprehensive analysis

## Usage:
/deep-watch https://youtube.com/watch?v=... "explain the key concepts"
```

---

## Data Structures

### VisualSignalResult

```python
@dataclass
class VisualSignalResult:
    score: float  # 0.0 to 1.0
    recommended: bool  # score > 0.5
    strongly_recommended: bool  # score > 0.75
    reasons: list[str]  # Human-readable explanations
    suggested_timestamps: list[float]  # Seconds into video
```

### High-Signal Timestamp Detection

```python
def find_high_signal_timestamps(transcript: str) -> list[float]:
    """
    Find timestamps where visual references cluster.

    Uses SRT timestamps to locate phrases like "as you can see"
    and returns timestamps with high density of visual references.
    """
    # Parse SRT to get timestamp -> text mapping
    # Find windows with high visual phrase density
    # Return start times of high-signal windows
    pass
```

---

## Configuration

### User Preferences (config.yaml)

```yaml
visual_analysis:
  # Automatic behavior
  auto_analyze_threshold: 0.75  # Auto-extract frames above this score

  # Always analyze these channels
  always_visual_channels:
    - "3blue1brown"
    - "veritasium"
    - "numberphile"

  # Default frame extraction settings
  default_frame_interval: 10  # seconds
  default_frame_quality: "medium"
  max_auto_frames: 20

  # Disable for faster processing
  enabled: true
```

---

## Success Metrics

### Quantitative

| Metric | Target |
|--------|--------|
| Visual analysis performed when recommended | > 90% |
| False positive rate (unnecessary frame extraction) | < 20% |
| User satisfaction with educational synthesis | > 4.5/5 |
| Frame extraction latency | < 5s for 10 frames |

### Qualitative

- Claude proactively extracts frames for 3Blue1Brown content
- Educational summaries include visual concept descriptions
- Users don't need to prompt for visual analysis on obvious cases

---

## Implementation Phases

### Phase 1: Signal Detection (MVP)
- [ ] Implement `compute_visual_signal_score()` function
- [ ] Add visual signal keywords and channel lists
- [ ] Unit tests for score computation

### Phase 2: Tool Enhancement
- [ ] Modify `process_video_tool` to include visual analysis in response
- [ ] Add `suggested_timestamps` computation
- [ ] Integration tests

### Phase 3: CLAUDE.md Protocol
- [ ] Add Visual Understanding Protocol section
- [ ] Document trigger words and thresholds
- [ ] Test with real video processing tasks

### Phase 4: Hook Integration
- [ ] Create post-processing hook for visual reminders
- [ ] Test hook firing and message formatting
- [ ] Tune reminder verbosity

### Phase 5: Smart Skill (Optional)
- [ ] Create `/deep-watch` skill
- [ ] Bundle automatic visual analysis workflow
- [ ] Documentation and examples

---

## Open Questions

1. **Cost management**: How do we balance comprehensive analysis with token/compute costs?
2. **User control**: Should auto-extraction be opt-in or opt-out?
3. **Caching**: Should we cache visual analysis scores for re-processed videos?
4. **Learning**: Can we learn from user feedback which videos needed visual analysis?

---

## Appendix: Keyword Lists

### ALWAYS_VISUAL_CHANNELS
```python
ALWAYS_VISUAL_CHANNELS = {
    "3blue1brown", "veritasium", "numberphile", "welch labs",
    "computerphile", "two minute papers", "kurzgesagt",
    "cgp grey", "smartereveryday", "primer", "reducible",
    "the coding train", "sebastian lague"
}
```

### VISUAL_REFERENCE_PHRASES
```python
VISUAL_REFERENCE_PHRASES = [
    "as you can see", "look at this", "notice how", "watch what happens",
    "in this animation", "the diagram shows", "shown here", "visualize",
    "picture this", "on screen", "here we have", "take a look",
    "you'll see", "observe how", "the graph shows", "in the figure",
    "highlighted here", "pointing to", "on the left", "on the right"
]
```

### EDUCATIONAL_KEYWORDS
```python
EDUCATIONAL_KEYWORDS = [
    "explained", "tutorial", "how to", "learn", "introduction to",
    "basics of", "understanding", "what is", "guide to", "course",
    "lesson", "lecture", "chapter", "deep dive"
]
```

### SPATIAL_TERMS
```python
SPATIAL_TERMS = [
    "vector", "dimension", "space", "direction", "axis", "coordinate",
    "gradient", "slope", "curve", "surface", "plane", "projection",
    "transform", "rotation", "translation", "scale", "magnitude",
    "perpendicular", "parallel", "orthogonal", "tangent"
]
```
