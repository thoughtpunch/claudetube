# PRD: Audio Description Support for claudetube

**Status:** Draft
**Author:** Claude
**Created:** 2026-02-01
**Last Updated:** 2026-02-01

---

## Accessibility Mission Statement

> **This feature has a secondary goal of improving video accessibility for blind and low-vision users.** We are actively seeking beta testers from the accessibility community to provide feedback on description quality, timing, and usefulness. If you or someone you know would benefit from this feature, please reach out via GitHub issues.

---

## Executive Summary

Add support for generating and storing audio descriptions (AD) alongside existing transcripts and frames. This enables claudetube to produce accessibility-compliant video analysis that serves vision-impaired users, meeting WCAG 2.1 Level AA requirements and aligning with emerging DOJ accessibility mandates.

---

## Background: What is Audio Description?

### Terminology

Audio description goes by several names depending on region and context:

| Term | Usage |
|------|-------|
| **Audio Description (AD)** | Most common international term |
| **Descriptive Video Service (DVS)** | Trademark of WGBH, common in US broadcast |
| **Video Description** | Used by FCC and US government |
| **Described Video** | Common in Canada |
| **Audio Narration** | Sometimes used in educational contexts |

All refer to the same concept: **narration added to media that describes meaningful visual information for people who cannot see it.**

### History

- **1960s-70s**: Early experiments with audio cassettes alongside TV programs; radio broadcaster describes movies over Philadelphia radio
- **1981**: Dr. Margaret Pfanstiehl pioneers live theater description in Washington, DC; founds the Metropolitan Washington Ear
- **1980s**: WGBH (Boston PBS station, already known for closed captioning) develops television AD via Secondary Audio Program (SAP), branded as "Descriptive Video Service" (DVS)
- **January 24, 1990**: WGBH officially launches DVS across 32 PBS stations with American Playhouse
- **2002**: Federal court rules FCC exceeded jurisdiction in mandating video description
- **2010**: 21st Century Communications and Video Accessibility Act (CVAA) reinstates FCC authority
- **July 1, 2012**: Rules take effect requiring 50 hours/quarter of described programming on top networks
- **April 2015**: Netflix commits to audio description, starting with Daredevil
- **April 2024**: DOJ finalizes Title II ADA rule requiring WCAG 2.1 AA compliance
- **2026-2027**: Compliance deadlines for public entities

### How Traditional AD is Produced

1. **Script writing**: A trained describer watches the video and writes descriptions that fit into natural pauses in dialogue
2. **Voice talent**: Professional narrator records the descriptions
3. **Audio engineering**: Descriptions are mixed with original soundtrack
4. **Delivery**: Either as separate SAP track (broadcast) or pre-mixed alternate audio (streaming)

Professional AD costs **$15-75+ per minute** due to this labor-intensive process.

---

## Problem Statement

claudetube currently extracts:
- **Audio** (transcripts via Whisper or YouTube subtitles)
- **Video** (frames at various quality levels)
- **Metadata** (chapters, timestamps, thumbnails)

However, there's no mechanism to describe **what is visually happening** in a video. A user who is blind or has low vision cannot understand:
- On-screen text, code, or diagrams
- Physical actions and movements
- Scene changes and locations
- Visual context that speakers assume viewers can see

This gap means claudetube's output is not fully accessible and cannot be used to generate WCAG-compliant video alternatives.

---

## What yt-dlp Already Provides (Don't Reinvent the Wheel)

Before building new features, we must leverage yt-dlp's existing capabilities:

### Existing Audio Description Track Support

yt-dlp can download existing AD tracks when available:

```bash
# List all available formats including audio tracks
yt-dlp -F "URL"

# Download all audio tracks (including AD if present)
yt-dlp -f "all[vcodec=none]" --audio-multistreams "URL"

# Target tracks with "description" in format notes
yt-dlp -f "ba[format_note*=description]" "URL"
```

**Key findings from [yt-dlp issue #6194](https://github.com/yt-dlp/yt-dlp/issues/6194):**
- AD tracks appear as separate format options (e.g., "hls-english-Descriptions")
- The `--audio-multistreams` flag enables downloading multiple audio tracks
- Format selection syntax `-f all[vcodec=none]` grabs all audio including AD
- No dedicated AD flag exists; use `-F` to identify track IDs

### Subtitle/Caption Support

yt-dlp provides comprehensive subtitle handling that claudetube already uses:

```bash
# List available subtitles
yt-dlp --list-subs "URL"

# Download subtitles in multiple formats
yt-dlp --write-subs --sub-format "vtt/srt/ttml" "URL"

# Download auto-generated captions
yt-dlp --write-auto-subs "URL"

# Convert subtitle formats
yt-dlp --convert-subs vtt "URL"
```

**Supported subtitle formats:** vtt, srt, ttml, ass, lrc, srv1, srv2, srv3, json3

### Metadata Fields Available

From `--dump-json` output:
- `subtitles`: Manual subtitle tracks by language
- `automatic_captions`: Auto-generated captions
- `chapters`: Video chapters with timestamps
- `description`: Video description text

### Implementation Principle

**claudetube should:**
1. **First**: Check if source video has existing AD track via yt-dlp
2. **Second**: Download and cache existing AD if available
3. **Third**: Only generate AI descriptions when no AD exists

This follows our "Cheap First, Expensive Last" architecture.

---

## Standards Landscape

### Web Standards

| Standard | Owner | Purpose | Status |
|----------|-------|---------|--------|
| **WebVTT `kind="descriptions"`** | W3C | Text-based descriptions in HTML5 video | Stable, widely supported |
| **TTML/IMSC** | W3C | Broadcast-grade timed text with styling | Production use (Netflix, broadcasters) |
| **DAPT** (Dubbing and Audio Description Profile of TTML2) | W3C | Specialized AD authoring and exchange | Candidate Recommendation (Dec 2025) |

### DAPT Technical Details

DAPT is the emerging W3C standard for AD exchange. Key features:

```xml
<tt xmlns="http://www.w3.org/ns/ttml"
    xmlns:tta="http://www.w3.org/ns/ttml#audio"
    xmlns:daptm="http://www.w3.org/ns/ttml/profile/dapt#metadata"
    ttp:contentProfiles="http://www.w3.org/ns/ttml/profile/dapt1.0/content"
    xml:lang="en"
    daptm:scriptType="asRecorded"
    daptm:scriptRepresents="visual.nonText">
```

- **Timing**: Clock-time (`00:00:05.100`) or offset-time (`5.1s`)
- **Audio integration**: External references, embedded base64, TTS synthesis
- **Mixing controls**: `tta:gain` and `tta:pan` for ducking main audio
- **Animated transitions**: `<animate tta:gain="1 0.3" dur="0.25s"/>` for fade effects

### Regulatory Requirements

| Regulation | Requirement | Deadline |
|------------|-------------|----------|
| **WCAG 2.1 Level A (1.2.3)** | Audio description OR full text alternative | Immediate |
| **WCAG 2.1 Level AA (1.2.5)** | Audio description required (transcript insufficient) | Immediate |
| **DOJ Title II Rule** | WCAG 2.1 AA for public entities (50k+ population) | April 24, 2026 |
| **DOJ Title II Rule** | WCAG 2.1 AA for smaller public entities | April 26, 2027 |
| **FCC/CVAA** | 87.5 hours/quarter for top broadcast networks | In effect |

### Industry Implementation

**Netflix:**
- Requires TTML format (TTAL) for AD scripts
- Audio delivered as BWAV ADM with bed separation (dialogue vs M&E)
- 5.1 + 2.0 channel configuration, -27 LKFS loudness target

**Amazon Prime Video:**
- Separate audio track with "[Audio Description]" label
- Pre-mixed with primary track
- [Style guide published April 2024](https://m.media-amazon.com/images/G/01/CooperWebsite/dvp/downloads/AD_Style_Guide_Prime_Video.pdf)

---

## AI-Generated Audio Description: Current State

### Commercial Services

| Service | Pricing | Key Features |
|---------|---------|--------------|
| **ViddyScribe** | $1/min (100+ min); Free: 50 min/mo | Google Gemini-based, VTT/audio export |
| **Verbit** | ~$33k/year enterprise | AI + human review hybrid |
| **Audible Sight** | Pay-as-you-go | 100+ voices, extended AD support |
| **3Play Media** | $7.50/min (educational) | AI-enabled with human QA option |
| **YuJa** | Enterprise bundle | Integrated with LMS platforms |

### Open Source Options

- **[Microsoft AI Audio Descriptions](https://github.com/microsoft/ai-audio-descriptions)**: Azure-based (GPT-4O + TTS), requires Azure subscription
- **YouDescribe**: Hybrid AI-human platform from Smith-Kettlewell Eye Research Institute
- **DANTE-AD**: Research project using LLaMA (CVPR 2025)

### Quality Research (RNIB Study, August 2025)

The Royal National Institute of Blind People commissioned the most comprehensive AI AD study:

- **Methodology**: 3 AI models vs professional AD across 160+ BBC video segments
- **Strengths**: Fluency and sentence structure exceeded expectations
- **Weaknesses**: Accuracy, cohesion, contextual awareness issues
- **Common problems**: Misidentified characters, vague object references, over-description
- **Conclusion**: "More work is needed before it can be reliably put into practice"

**Recommendation**: Hybrid AI + human review for production-quality AD

### Cost Comparison

| Approach | Cost | Quality |
|----------|------|---------|
| Professional human AD | $15-75/minute | Gold standard |
| AI + human review | $3-10/minute | Near-professional |
| AI-only | $0-1/minute | 70% quality parity |

---

## Web Player Support for Text-Based AD

### Able Player

[Able Player](https://ableplayer.github.io/ableplayer/) is the gold standard for accessible web video:

```html
<video id="video1" data-able-player>
  <source type="video/mp4" src="video.mp4"/>
  <track kind="captions" src="captions.vtt" srclang="en"/>
  <track kind="descriptions" src="descriptions.vtt" srclang="en"/>
</video>
```

- Uses Web Speech API for TTS synthesis
- User can customize voice, pitch, rate, volume
- Optional "pause video during description" mode
- Falls back to ARIA live regions for screen readers

### BBC adhere-lib

[adhere-lib](https://github.com/bbc/adhere-lib) implements TTML2/DAPT in the browser:

- Web Audio API for gain/pan mixing
- Web Speech API for TTS
- Supports animated gain transitions (ducking)
- Apache 2.0 license (open source)

### Browser Native Support

| Feature | Browser Support |
|---------|-----------------|
| `<track kind="descriptions">` | **0%** - No browser synthesizes AD natively |
| Web Speech API (SpeechSynthesis) | **96%** - Widely supported |
| Web Audio API | **97%** - Widely supported |

**Key insight**: Native AD synthesis requires JavaScript libraries (Able Player, adhere-lib) since browsers don't implement it.

---

## Architecture Integration: The Key Insight

### We Already Have Scene Descriptions

**Critical realization**: claudetube already generates visual descriptions as part of scene processing:

```
scenes/scene_NNN/
├── keyframes/           # Representative frames (ALREADY EXTRACTED)
├── visual.json          # Visual analysis (ALREADY GENERATED)
└── technical.json       # Code/diagram analysis (ALREADY GENERATED)
```

The `visual.json` and `technical.json` files **ARE descriptions**. We're not building a new AI pipeline from scratch—we're **compiling existing scene analysis into an accessibility format**.

### AD is a View, Not a New Pipeline

```
EXISTING DATA                          NEW OUTPUT
─────────────────                      ──────────────
scenes/scene_000/visual.json    ───┐
scenes/scene_001/visual.json    ───┼──►  audio.ad.vtt (WebVTT)
scenes/scene_002/visual.json    ───┤     audio.ad.txt (plain text)
scenes/scene_NNN/technical.json ───┘     audio.accessible.txt (merged)
```

This means:
1. **No new expensive VLM calls** for videos with existing scene analysis
2. **Incremental cost** only for the compilation step
3. **Consistent descriptions** that match other claudetube outputs

### How It Fits the "Cheap First, Expensive Last" Principle

```
AD GENERATION PIPELINE (mirrors existing patterns)

1. CACHE CHECK (instant)
   └── Return audio.ad.vtt if exists

2. YT-DLP AD TRACK CHECK (free - metadata only)
   └── yt-dlp -F URL | grep -i description
   └── If found: download, transcribe to .ad.txt, done

3. SCENE COMPILATION (cheap - local JSON processing)
   └── If scenes/scenes.json exists:
       ├── Read all visual.json + technical.json
       ├── Format as WebVTT with scene timestamps
       └── Save to audio.ad.vtt

4. SCENE PROCESSING (moderate - triggers existing pipeline)
   └── If scenes not processed:
       ├── Run segment_video_smart() (mostly cheap)
       ├── Extract keyframes
       ├── Generate visual.json per scene
       └── Then compile (step 3)

5. ON-DEMAND MOMENT DESCRIPTION (expensive - only when needed)
   └── describe_moment(timestamp) for specific frames
   └── Uses get_hq_frames() + VLM analysis
```

### State Model Extension

Add to `VideoState` in `models/state.py`:

```python
@dataclass
class VideoState:
    # ... existing fields ...

    # Audio Description
    ad_complete: bool = False
    ad_source: str | None = None      # "source_track", "scene_compilation", "generated"
    ad_track_available: bool | None = None  # Did source have AD track?
```

### Cache Structure Extension

```
~/.claude/video_cache/{video_id}/
├── state.json              # + ad_complete, ad_source, ad_track_available
├── audio.mp3
├── audio.srt
├── audio.txt
├── audio.ad.vtt            # NEW: WebVTT descriptions
├── audio.ad.txt            # NEW: Plain text descriptions
├── audio.ad.mp3            # NEW: Original AD track (if downloaded from source)
├── audio.accessible.txt    # NEW: Merged transcript + descriptions
├── thumbnail.jpg
├── scenes/
│   ├── scenes.json
│   └── scene_NNN/
│       ├── keyframes/
│       ├── visual.json     # EXISTING - source for AD
│       └── technical.json  # EXISTING - source for AD
└── drill_*/hq/
```

### Code Organization

```
src/claudetube/
├── operations/
│   ├── audio_description.py    # NEW: AD generation logic
│   │   ├── check_source_ad()       # Check yt-dlp for AD track
│   │   ├── download_ad_track()     # Download existing AD
│   │   ├── compile_scene_descriptions()  # Compile visual.json → .ad.vtt
│   │   ├── generate_ad()           # Full pipeline orchestration
│   │   └── format_accessible_transcript()  # Merge transcript + AD
│   └── ... existing modules ...
│
├── tools/
│   └── yt_dlp.py               # ADD: check_audio_description(url)
│
├── cache/
│   └── manager.py              # ADD: get_ad_paths(), has_ad()
│
├── models/
│   └── state.py                # ADD: ad_complete, ad_source fields
│
└── mcp_server.py               # ADD: get_descriptions, describe_moment tools
```

### AI Provider Integration

> **Note:** claudetube is implementing a configurable AI providers architecture (see [configurable-ai-providers.md](./configurable-ai-providers.md)). Audio description generation should use this abstraction rather than hard-coding specific providers.

#### Provider Touchpoints for AD

The audio description feature will use the following capabilities from the provider system:

| AD Operation | Required Capability | Provider Interface | Fallback |
|--------------|--------------------|--------------------|----------|
| **Transcribe AD audio track** | `TRANSCRIBE` | `TranscriptionProvider.transcribe()` | `whisper-local` |
| **Describe frame (on-demand)** | `VISION` | `VisionAnalyzer.analyze_images()` | `claude-code` |
| **Native video description** | `VIDEO` | `VideoAnalyzer.analyze_video()` | Frame-by-frame fallback |
| **Process transcript for cues** | `REASON` | `Reasoner.reason()` | `claude-code` |

#### Using the Provider Facade

```python
# operations/audio_description.py
from ..providers import ClaudeTubeProviders

class AudioDescriptionGenerator:
    def __init__(self, providers: ClaudeTubeProviders):
        self.providers = providers
        self.caps = providers.get_capabilities_matrix()

    async def describe_moment(self, video_id: str, timestamp: float) -> str:
        """Describe visual content at a specific moment."""
        frames = extract_hq_frames(video_id, timestamp, duration=1)

        # Use provider abstraction - routes to best available
        description = await self.providers.caption_frame(
            frames[0],
            prompt=AD_FRAME_PROMPT
        )
        return description

    async def generate_scene_descriptions(self, video_id: str, video_path: Path) -> list[dict]:
        """Generate descriptions for all scenes."""

        # Check if native video analysis is available (Gemini)
        if self.caps.can(Capability.VIDEO):
            # BEST: Single API call for entire video
            result = await self.providers.analyze_video_native(
                video_path,
                prompt=AD_VIDEO_PROMPT,
                schema=AudioDescriptionResult  # Structured output
            )
            return result.scenes

        # FALLBACK: Frame-by-frame via VisionAnalyzer
        elif self.caps.can(Capability.VISION):
            scenes = load_scenes_data(video_id)
            descriptions = []
            for scene in scenes:
                keyframe = get_scene_keyframe(video_id, scene.scene_id)
                desc = await self.providers.caption_frame(
                    keyframe,
                    prompt=AD_FRAME_PROMPT
                )
                descriptions.append({
                    "scene_id": scene.scene_id,
                    "start_time": scene.start_time,
                    "description": desc
                })
            return descriptions

        # No vision capability - can only compile existing visual.json
        return None

    async def transcribe_ad_track(self, ad_audio_path: Path) -> str:
        """Transcribe downloaded AD audio track."""
        # Uses provider abstraction - may route to OpenAI Whisper API or local
        result = await self.providers.transcribe(ad_audio_path)
        return result.text
```

#### Capability-Driven AD Quality Tiers

The provider system enables different AD quality levels based on available capabilities:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    AD QUALITY TIERS BY CAPABILITY                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  TIER 1: Gemini Native Video (Capability.VIDEO)                     │
│  ├── Single API call analyzes full video                            │
│  ├── Tracks objects/people across scenes                            │
│  ├── Best temporal coherence                                        │
│  └── Cost: ~$0.10/M tokens for 2hr video                            │
│                                                                     │
│  TIER 2: Multi-Frame Vision (Capability.VISION)                     │
│  ├── Keyframe per scene (Claude/GPT-4o/LLaVA)                       │
│  ├── Good scene-level descriptions                                  │
│  ├── May miss inter-scene continuity                                │
│  └── Cost: ~$0.01-0.05 per frame                                    │
│                                                                     │
│  TIER 3: Scene Compilation (No AI needed)                           │
│  ├── Compiles existing visual.json + technical.json                 │
│  ├── Reuses scene analysis already performed                        │
│  ├── Instant, no API calls                                          │
│  └── Cost: $0                                                       │
│                                                                     │
│  TIER 4: Transcript-Only (Capability.REASON)                        │
│  ├── Extract visual cues from transcript                            │
│  ├── "As you can see..." → mark for description                     │
│  ├── Limited without actual visual analysis                         │
│  └── Cost: Minimal                                                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### Structured Output for AD

Use provider's structured output capability to enforce consistent AD format:

```python
# providers/schemas.py - Add to existing schemas
from pydantic import BaseModel

class SceneDescription(BaseModel):
    """A single scene's audio description."""
    scene_id: int
    start_time: float
    end_time: float
    description: str
    visual_elements: list[str] = []
    on_screen_text: str | None = None
    content_type: str | None = None  # "presenter", "code", "diagram", etc.

class AudioDescriptionResult(BaseModel):
    """Complete AD extraction result - enforced by provider."""
    video_id: str
    total_duration: float
    scenes: list[SceneDescription]
    speakers: list[str] = []
    has_code: bool = False
    has_diagrams: bool = False
```

#### Zero-Config Behavior

When no providers are configured, AD generation falls back gracefully:

```python
# Default behavior (no API keys configured)
AD_FALLBACK_CHAIN = [
    "gemini",      # Native video (if GOOGLE_API_KEY)
    "claude-api",  # Vision (if ANTHROPIC_API_KEY)
    "gpt-4o",      # Vision (if OPENAI_API_KEY)
    "claude-code", # Host AI vision (always available in Claude Code)
    "compile",     # Just compile existing visual.json (always works)
]
```

### Why This Approach is Optimal

| Alternative | Problem |
|-------------|---------|
| **New parallel VLM pipeline** | Duplicates scene analysis, expensive, inconsistent |
| **Real-time generation on every request** | Slow, no caching benefit |
| **Store AD separately from scenes** | Data duplication, sync issues |
| **Generate AD during initial processing** | Too slow for first-time use |

**Our approach:**
- **Reuses existing infrastructure** (scene detection, visual analysis)
- **Lazy evaluation** (only compile when requested)
- **Caches results** (compile once, serve forever)
- **Respects hierarchy** (check source AD → compile scenes → generate new)

---

## Proposed Solution

### Core Feature: Visual Description Track

Add a new artifact type to claudetube's cache structure: **descriptions** (`.ad.vtt` and `.ad.txt`).

```
~/.claude/video_cache/{video_id}/
├── state.json
├── audio.mp3
├── audio.srt          # Speech transcript with timestamps
├── audio.txt          # Plain text transcript
├── audio.ad.vtt       # NEW: Visual descriptions (WebVTT)
├── audio.ad.txt       # NEW: Visual descriptions (plain text)
├── audio.ad.mp3       # NEW: Optional synthesized AD audio
├── thumbnail.jpg
├── drill/
└── hq/
```

### Description Generation Pipeline

Following claudetube's "Cheap First, Expensive Last" principle:

```
1. CACHE      → Return existing .ad.vtt if present
2. YT-DLP     → Check for existing AD audio track in source
3. METADATA   → Extract any description hints from video metadata
4. TRANSCRIPT → Identify visual references ("as you can see...")
5. FRAMES     → Analyze key frames for visual content
6. AI/VLM     → Generate descriptions using vision-language model (EXPENSIVE)
```

### WebVTT Description Format

```vtt
WEBVTT
Kind: descriptions
Language: en

00:00:05.000 --> 00:00:08.000
A terminal window displays Python code with syntax highlighting.

00:00:15.000 --> 00:00:18.000
The presenter points to a diagram showing data flow between services.

00:01:23.000 --> 00:01:26.000
On-screen text reads: "API Response: 200 OK"
```

### MCP Tools

```python
# Get or generate descriptions
get_descriptions(
    video_id_or_url: str,
    format: Literal["vtt", "txt"] = "vtt",
    regenerate: bool = False
) -> str

# Describe specific moment
describe_moment(
    video_id_or_url: str,
    timestamp: float,
    context: Optional[str] = None
) -> str

# Get combined accessible transcript
get_accessible_transcript(
    video_id_or_url: str,
    format: Literal["txt", "html", "json"] = "txt"
) -> str

# Check if source has existing AD track
has_audio_description(
    video_id_or_url: str
) -> bool
```

---

## Technical Approach

### Phase 1: yt-dlp AD Track Detection (MVP)

**Goal**: Don't generate what already exists.

**Files to modify:**
- `tools/yt_dlp.py` - Add `check_audio_description()` method

```python
# tools/yt_dlp.py
def check_audio_description(self, url: str) -> dict | None:
    """Check if video has existing audio description track.

    Returns format info dict if AD track found, None otherwise.
    """
    formats = self.get_formats(url)  # Uses -F internally

    # Check format_note for AD indicators
    ad_indicators = ['description', 'descriptive', 'ad', 'dvs']
    for fmt in formats:
        note = fmt.get('format_note', '').lower()
        if any(ind in note for ind in ad_indicators):
            return fmt

    return None

def download_audio_description(self, url: str, output_dir: Path) -> Path | None:
    """Download AD track if available."""
    ad_format = self.check_audio_description(url)
    if not ad_format:
        return None

    # Use format ID to download specific track
    format_id = ad_format['format_id']
    output = output_dir / "audio.ad.mp3"
    # yt-dlp -f {format_id} -x --audio-format mp3 ...
    return output
```

**MCP integration:**
```python
# mcp_server.py
@mcp.tool()
async def has_audio_description(video_id_or_url: str) -> str:
    """Check if video source has an existing audio description track."""
    # Returns: {"has_ad": bool, "format_info": {...} | null}
```

### Phase 2: Scene Compilation

**Goal**: Compile existing `visual.json` + `technical.json` into WebVTT.

**New file:** `operations/audio_description.py`

```python
# operations/audio_description.py
from pathlib import Path
from ..cache.manager import CacheManager
from ..cache.scenes import ScenesData

def compile_scene_descriptions(video_id: str) -> Path | None:
    """Compile scene visual.json files into WebVTT description track.

    Returns path to audio.ad.vtt or None if scenes not processed.
    """
    cache = CacheManager()
    scenes_data = cache.load_scenes_data(video_id)

    if not scenes_data or not scenes_data.scenes:
        return None

    vtt_lines = ["WEBVTT", "Kind: descriptions", "Language: en", ""]

    for scene in scenes_data.scenes:
        scene_dir = cache.get_scene_dir(video_id, scene.scene_id)
        visual_path = scene_dir / "visual.json"
        technical_path = scene_dir / "technical.json"

        description_parts = []

        # Extract description from visual.json
        if visual_path.exists():
            visual = json.loads(visual_path.read_text())
            if desc := visual.get('description'):
                description_parts.append(desc)
            if elements := visual.get('visual_elements'):
                description_parts.append(f"Visual elements: {', '.join(elements)}")

        # Add technical content (code, diagrams)
        if technical_path.exists():
            technical = json.loads(technical_path.read_text())
            if code := technical.get('code_visible'):
                lang = technical.get('language', 'code')
                description_parts.append(f"{lang} code displayed on screen")
            if text := technical.get('on_screen_text'):
                description_parts.append(f'On-screen text: "{text}"')

        if description_parts:
            # Format timestamp: HH:MM:SS.mmm
            start = format_vtt_time(scene.start_time)
            end = format_vtt_time(min(scene.start_time + 5, scene.end_time))

            vtt_lines.append(f"{start} --> {end}")
            vtt_lines.append(" ".join(description_parts))
            vtt_lines.append("")

    # Write VTT file
    ad_path = cache.get_cache_dir(video_id) / "audio.ad.vtt"
    ad_path.write_text("\n".join(vtt_lines))

    # Update state
    state = cache.get_state(video_id)
    state.ad_complete = True
    state.ad_source = "scene_compilation"
    cache.save_state(video_id, state)

    return ad_path
```

### Phase 3: MCP Tools (Provider-Aware)

**Files to modify:** `mcp_server.py`

> **Provider Integration:** These tools use the `ClaudeTubeProviders` facade from the configurable AI providers system. See [configurable-ai-providers.md](./configurable-ai-providers.md) for details.

```python
from ..providers import ClaudeTubeProviders, Capability

# Initialize provider facade (once, at module level)
providers = ClaudeTubeProviders()

@mcp.tool()
async def get_descriptions(
    video_id_or_url: str,
    format: str = "vtt",  # "vtt" or "txt"
    regenerate: bool = False
) -> str:
    """Get visual descriptions for a video.

    Pipeline (provider-aware):
    1. Return cached .ad.vtt if exists (unless regenerate=True)
    2. Check source for existing AD track via yt-dlp
    3. If AD track found, transcribe using TranscriptionProvider
    4. Compile from scene analysis if available
    5. Generate via VisionProvider/VideoProvider if configured
    6. Return error if no descriptions possible
    """
    video_id = parse_video_id(video_id_or_url)
    cache = CacheManager()
    caps = providers.get_capabilities_matrix()

    ad_path = cache.get_cache_dir(video_id) / f"audio.ad.{format}"

    # Step 1: Cache check
    if ad_path.exists() and not regenerate:
        return ad_path.read_text()

    # Step 2: Check source for AD track
    state = cache.get_state(video_id)
    if state and state.url:
        ad_track = await asyncio.to_thread(
            yt_dlp.check_audio_description, state.url
        )
        if ad_track:
            # Download AD track
            ad_audio = await asyncio.to_thread(
                yt_dlp.download_audio_description, state.url, cache.get_cache_dir(video_id)
            )
            if ad_audio:
                # PROVIDER TOUCHPOINT: Transcribe AD track
                # Uses TranscriptionProvider - may route to OpenAI Whisper API,
                # Deepgram, or local Whisper depending on config
                transcript = await providers.transcribe(ad_audio)
                # Save as .ad.txt and .ad.vtt
                # Update state.ad_source = "source_track"
                return format_result(transcript, format)

    # Step 3: Compile from existing scene analysis (FREE - no AI needed)
    scenes_data = cache.load_scenes_data(video_id)
    if scenes_data and has_visual_json(video_id, scenes_data):
        ad_path = await asyncio.to_thread(
            compile_scene_descriptions, video_id
        )
        if ad_path:
            return ad_path.read_text()

    # Step 4: Generate descriptions if provider available
    # PROVIDER TOUCHPOINT: Native video or frame-by-frame vision
    if caps.can(Capability.VIDEO):
        # Gemini native video - single call for full video
        video_path = get_video_path(video_id)
        result = await providers.analyze_video_native(
            video_path,
            prompt=AD_VIDEO_PROMPT,
            schema=AudioDescriptionResult
        )
        ad_path = save_descriptions(video_id, result)
        return ad_path.read_text()

    elif caps.can(Capability.VISION):
        # Frame-by-frame vision analysis
        scenes = await ensure_scenes_processed(video_id)
        for scene in scenes:
            keyframe = get_scene_keyframe(video_id, scene.scene_id)
            # PROVIDER TOUCHPOINT: VisionAnalyzer
            desc = await providers.caption_frame(keyframe, AD_FRAME_PROMPT)
            save_scene_description(video_id, scene.scene_id, desc)

        ad_path = compile_scene_descriptions(video_id)
        return ad_path.read_text()

    # Step 5: No descriptions possible
    return json.dumps({
        "error": "No descriptions available",
        "capabilities_available": caps.summary(),
        "suggestions": [
            "Process video with scene analysis first",
            "Configure a vision provider (Gemini, Claude API, GPT-4o)",
            "Use describe_moment() for specific timestamps"
        ]
    })

@mcp.tool()
async def describe_moment(
    video_id_or_url: str,
    timestamp: float,
    context: str | None = None
) -> str:
    """Describe visual content at a specific moment.

    PROVIDER TOUCHPOINT: Uses VisionAnalyzer capability.
    Routes to best available: Gemini > Claude API > GPT-4o > claude-code

    This is the EXPENSIVE path - use sparingly.
    """
    caps = providers.get_capabilities_matrix()

    if not caps.can(Capability.VISION):
        return json.dumps({
            "error": "No vision provider available",
            "suggestion": "Configure a vision provider or use claude-code"
        })
    # Extract HQ frame at timestamp
    frames = await get_hq_frames(video_id_or_url, timestamp, duration=1, interval=1)

    # Return frame path for Claude to analyze
    # (Claude's vision capability handles the actual description)
    return json.dumps({
        "timestamp": timestamp,
        "frame_path": str(frames[0]) if frames else None,
        "context": context,
        "instruction": "Describe the visual content of this frame for a vision-impaired user"
    })

@mcp.tool()
async def get_accessible_transcript(
    video_id_or_url: str,
    format: str = "txt"  # "txt", "html", "json"
) -> str:
    """Get combined transcript with visual descriptions interleaved.

    Merges audio.srt + audio.ad.vtt into unified accessible format.
    """
    video_id = parse_video_id(video_id_or_url)
    cache = CacheManager()

    # Get transcript
    srt_path, txt_path = cache.get_transcript_paths(video_id)
    transcript_segments = parse_srt(srt_path)

    # Get descriptions
    ad_path = cache.get_cache_dir(video_id) / "audio.ad.vtt"
    if not ad_path.exists():
        # Try to generate
        await get_descriptions(video_id_or_url)

    if ad_path.exists():
        description_segments = parse_vtt(ad_path)
    else:
        description_segments = []

    # Merge by timestamp
    merged = merge_by_timestamp(transcript_segments, description_segments)

    return format_accessible_output(merged, format)
```

### Phase 4: State & Cache Integration

**Files to modify:** `models/state.py`, `cache/manager.py`

```python
# models/state.py - Add to VideoState dataclass
@dataclass
class VideoState:
    # ... existing fields ...

    # Audio Description
    ad_complete: bool = False
    ad_source: str | None = None  # "source_track" | "scene_compilation" | "generated"
    ad_track_available: bool | None = None  # Cached result of yt-dlp check

# cache/manager.py - Add methods
class CacheManager:
    def get_ad_paths(self, video_id: str) -> tuple[Path, Path]:
        """Return (vtt_path, txt_path) for audio descriptions."""
        cache_dir = self.get_cache_dir(video_id)
        return (cache_dir / "audio.ad.vtt", cache_dir / "audio.ad.txt")

    def has_ad(self, video_id: str) -> bool:
        """Check if audio descriptions exist in cache."""
        vtt, txt = self.get_ad_paths(video_id)
        return vtt.exists() or txt.exists()

    def get_accessible_transcript_path(self, video_id: str) -> Path:
        """Path to merged accessible transcript."""
        return self.get_cache_dir(video_id) / "audio.accessible.txt"
```

### Implementation Order

| Phase | Effort | Dependencies | Deliverable |
|-------|--------|--------------|-------------|
| **1. yt-dlp AD check** | Small | None | `has_audio_description()` tool |
| **2. Scene compilation** | Medium | Existing scene analysis | `compile_scene_descriptions()` |
| **3. MCP tools** | Medium | Phases 1-2 | `get_descriptions()`, `describe_moment()` |
| **4. State integration** | Small | Phase 3 | Updated `VideoState`, cache methods |
| **5. Accessible transcript** | Small | Phase 3 | `get_accessible_transcript()` |

**Total estimated new code: ~300-400 lines** (excluding tests)

---

## Success Criteria

### Functional Requirements

- [ ] Detect existing AD tracks via yt-dlp before generating
- [ ] Download and cache existing AD when available
- [ ] Generate WebVTT description files from video frames
- [ ] Cache descriptions alongside existing artifacts
- [ ] Expose descriptions via MCP tools
- [ ] Support manual description editing/override
- [ ] Merge descriptions with transcript for combined output

### Quality Requirements

- [ ] Descriptions fit within natural pauses (< 5 seconds typical)
- [ ] Descriptions focus on visually-essential content
- [ ] Text/code shown on screen is transcribed verbatim
- [ ] Scene changes are marked
- [ ] Timing aligns with visual content (not lagging)

### Accessibility Requirements

- [ ] Output meets WCAG 2.1 Level AA (1.2.5)
- [ ] WebVTT format compatible with Able Player
- [ ] Plain text alternative available for screen readers
- [ ] Language tag included in VTT header

---

## Open Questions

1. **Description verbosity level:** Should we offer "brief" vs "detailed" modes?

2. **Code handling:** When code is on screen, should we describe it ("Python code visible") or transcribe it verbatim?

3. **Cost management:** VLM analysis is expensive. Should we rate-limit or require explicit opt-in?

4. **TTS synthesis:** Should claudetube generate AD audio files, or only text?

5. **Existing AD preservation:** How do we handle videos that already have professional AD?

6. **Extended AD:** Should we support pausing video for longer descriptions (WCAG AAA)?

---

## References

### Background
- [Audio Description - Wikipedia](https://en.wikipedia.org/wiki/Audio_description)
- [A Brief History of Audio Description in the U.S.](https://audiodescriptionsolutions.com/a-brief-history-of-audio-description-in-the-u-s/)
- [FCC Audio Description](https://www.fcc.gov/audio-description)

### W3C Standards
- [WebVTT: The Web Video Text Tracks Format](https://www.w3.org/TR/webvtt1/)
- [TTML Profiles for Internet Media Subtitles and Captions 1.2](https://www.w3.org/TR/ttml-imsc1.2/)
- [Dubbing and Audio Description Profiles of TTML2 (DAPT)](https://w3c.github.io/dapt/)
- [W3C WAI: Description of Visual Information](https://www.w3.org/WAI/media/av/description/)

### WCAG Requirements
- [WCAG 1.2.3: Audio Description or Media Alternative](https://www.w3.org/WAI/WCAG22/Understanding/audio-description-or-media-alternative-prerecorded.html)
- [WCAG 1.2.5: Audio Description (Prerecorded)](https://www.w3.org/WAI/WCAG22/Understanding/audio-description-prerecorded.html)

### Industry Specifications
- [Netflix Audio Description Style Guide v2.5](https://partnerhelp.netflixstudios.com/hc/en-us/articles/215510667-Audio-Description-Style-Guide-v2-5)
- [Netflix Audio Description Delivery Specification](https://partnerhelp.netflixstudios.com/hc/en-us/articles/360060568773-Audio-Description-Delivery-Specification)
- [Amazon Prime Video Audio Description Style Guide (2024)](https://m.media-amazon.com/images/G/01/CooperWebsite/dvp/downloads/AD_Style_Guide_Prime_Video.pdf)

### yt-dlp
- [yt-dlp GitHub Repository](https://github.com/yt-dlp/yt-dlp)
- [yt-dlp Issue #6194: Audio Description Track Download](https://github.com/yt-dlp/yt-dlp/issues/6194)

### Implementation Resources
- [Able Player](https://ableplayer.github.io/ableplayer/)
- [BBC adhere-lib](https://github.com/bbc/adhere-lib)
- [Microsoft AI Audio Descriptions](https://github.com/microsoft/ai-audio-descriptions)
- [How to Create Free Audio Description VTT Files](https://meryl.net/audio-description-vtt-files/)
- [Audio Description using Web Speech API](https://terrillthompson.com/1173)

### AI/Automation Research
- [RNIB: Exploring AI-Generated Audio Description (2025)](https://www.rnib.org.uk/professionals/research-and-data/reports-and-insight/exploring-ai-generated-audio-description-can-emerging-technologies-help-expand-access-to-broadcast-media/)
- [AI is now used for audio description (The Conversation)](https://theconversation.com/ai-is-now-used-for-audio-description-but-it-should-be-accurate-and-actually-useful-for-people-with-low-vision-256808)
- [Audio Description in 2024: Emerging Trends](https://www.captioningstar.com/blog/audio-description-in-2024-emerging-trends-and-their-impact-on-accessibility/)

---

## Appendix A: Description Style Guidelines

Based on Netflix and Amazon style guides:

### What to Describe
- Actions and movements essential to the story
- Scene changes and settings
- On-screen text (titles, signs, captions)
- Character appearances when first introduced
- Visual information referenced by speakers
- Code, diagrams, or technical content when instructional

### What NOT to Describe
- Information already conveyed through dialogue
- Obvious actions ("he walks to the door" when you hear footsteps)
- Subjective interpretations
- Background details irrelevant to understanding

### Timing Principles
- Descriptions should not overlap with critical dialogue
- Aim for natural pauses in speech
- Maximum ~5 seconds per description
- If extended description needed, consider pause/extended mode

### Language Style
- Present tense ("A man enters" not "A man entered")
- Third person, objective
- Concise but complete
- No editorializing ("beautiful sunset" vs "orange sunset")

---

## Appendix B: yt-dlp AD Track Detection

Example workflow for checking existing audio description:

```bash
# Step 1: List all formats
yt-dlp -F "https://example.com/video"

# Look for entries like:
# ID     EXT  RESOLUTION  NOTE
# 234-1  m4a  audio only  [en] Audio Description

# Step 2: Download video with AD track
yt-dlp -f "bestvideo+234-1" --audio-multistreams "URL"

# Or download all audio tracks
yt-dlp -f "all[vcodec=none]" --audio-multistreams "URL"

# Step 3: Target description tracks specifically
yt-dlp -f "ba[format_note*=description]" "URL"
```

**Format selection patterns for AD:**
- `ba[format_note*=description]` - Audio with "description" in notes
- `ba[format_note*=AD]` - Audio with "AD" in notes
- `all[vcodec=none]` - All audio tracks (then filter locally)

---

## Appendix C: WebVTT Description Example

Complete example for a coding tutorial:

```vtt
WEBVTT
Kind: descriptions
Language: en
NOTE Generated by claudetube on 2026-02-01
NOTE Video: "Building a REST API in 10 Minutes"

NOTE Chapter: Introduction (0:00 - 0:45)
00:00:00.500 --> 00:00:04.000
Tutorial title card: "REST API Tutorial" in white text on dark background.

00:00:08.000 --> 00:00:11.500
Presenter visible from shoulders up, casual setting, laptop in front.

NOTE Chapter: Project Setup (0:45 - 3:20)
00:00:47.000 --> 00:00:50.000
Screen capture: Terminal window with black background, cursor blinking.

00:00:55.000 --> 00:00:58.500
Terminal displays: mkdir api-project && cd api-project

00:01:15.000 --> 00:01:18.000
Command output shows package.json created successfully.

00:01:45.000 --> 00:01:48.500
VS Code opens. File explorer on left shows project folder structure.

NOTE Chapter: Writing Code (3:20 - 7:00)
00:03:22.000 --> 00:03:26.000
New file created: index.js. Editor shows empty file with line numbers.

00:03:45.000 --> 00:03:50.000
Code typed on screen:
const express = require('express');
const app = express();

00:04:30.000 --> 00:04:35.000
Code continues:
app.get('/api/users', (req, res) => {
  res.json([{ id: 1, name: 'John' }]);
});

NOTE Chapter: Testing (7:00 - 9:00)
00:07:05.000 --> 00:07:09.000
Terminal split view: server running on left, curl command on right.

00:07:25.000 --> 00:07:29.000
API response displayed: JSON array with user object.

NOTE Chapter: Conclusion (9:00 - 10:00)
00:09:02.000 --> 00:09:06.000
Return to presenter view. Thumbs up gesture.

00:09:30.000 --> 00:09:34.000
End card with subscribe button and related video thumbnails.
```

---

## Appendix D: DAPT Document Structure

Minimal DAPT document for audio description:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<tt xmlns="http://www.w3.org/ns/ttml"
    xmlns:ttp="http://www.w3.org/ns/ttml#parameter"
    xmlns:tta="http://www.w3.org/ns/ttml#audio"
    xmlns:daptm="http://www.w3.org/ns/ttml/profile/dapt#metadata"
    ttp:contentProfiles="http://www.w3.org/ns/ttml/profile/dapt1.0/content"
    xml:lang="en"
    daptm:scriptType="asRecorded"
    daptm:scriptRepresents="visual.nonText">

  <body>
    <div begin="00:00:05.000" end="00:00:12.000">
      <!-- Duck main audio during description -->
      <animate tta:gain="1 0.3" begin="0s" dur="0.25s" fill="freeze"/>

      <!-- Description content -->
      <p xml:lang="en">
        <audio src="desc_001.mp3" type="audio/mpeg"/>
      </p>

      <!-- Restore main audio -->
      <animate tta:gain="0.3 1" begin="6.75s" dur="0.25s"/>
    </div>
  </body>
</tt>
```

Key DAPT attributes:
- `daptm:scriptType`: `originalTranscript`, `translatedTranscript`, `preRecording`, `asRecorded`
- `daptm:scriptRepresents`: `audio.dialogue`, `visual.text`, `visual.nonText`
- `tta:gain`: Volume control (0.0-1.0)
- `tta:pan`: Stereo position (-1 left, 0 center, 1 right)
