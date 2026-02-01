# PRD: Configurable AI Providers

**Status:** Draft
**Author:** Claude
**Created:** 2026-02-01
**Last Updated:** 2026-02-01

---

## Executive Summary

Extract claudetube's AI capabilities (transcription, image analysis, text analysis) into configurable provider classes, allowing users to bring their own models and API keys. The system should default to using the Claude Code host AI (zero config for most users), while enabling power users to configure alternatives like OpenAI Whisper API for transcription, GPT-4 for vision, or Gemini for video analysis.

---

## Problem Statement

### Current State

claudetube currently hard-codes AI provider choices:

| Capability | Current Implementation | Location |
|------------|----------------------|----------|
| **Transcription** | `faster-whisper` (local) | `tools/whisper.py` |
| **Visual Analysis** | Claude 3 Haiku API | `operations/visual_transcript.py` |
| **Embeddings** | Voyage AI or local | `analysis/embeddings.py` |
| **OCR** | EasyOCR (local) | `analysis/ocr.py` |

### Problems

1. **No API transcription option**: Local Whisper is slow (10-30 min/hour of audio). Users with API keys can't use faster cloud services.

2. **Hard-coded Claude dependency**: Visual analysis requires `ANTHROPIC_API_KEY`. Users of OpenAI or Gemini can't use their existing keys.

3. **No video analysis**: Gemini 2.0+ can analyze video directly (up to 2 hours). This capability is unavailable.

4. **Missing features**: Cloud transcription services offer speaker diarization, sentiment analysis, and real-time streaming that local Whisper lacks.

5. **Cost optimization blocked**: Users can't choose cheaper providers for bulk operations or use local models for privacy-sensitive content.

---

## Goals

### Primary Goals

1. **Zero-config default**: Works out-of-the-box using Claude Code's host AI for vision/analysis and local Whisper for transcription.

2. **Provider abstraction**: Clean interfaces allowing any compatible provider to be swapped in.

3. **Mixed configurations**: Use different providers for different capabilities (e.g., Claude for vision, OpenAI Whisper for transcription).

4. **Graceful degradation**: Fall back to available providers when preferred ones fail or lack credentials.

### Non-Goals

- Building a full LiteLLM-style universal proxy
- Supporting every possible AI provider
- Real-time streaming transcription (future work)
- Fine-tuning or custom model training

---

## Provider Capability Matrix

### Transcription Providers

| Provider | Speed | Accuracy | Max Duration | Max Size | Cost | Features |
|----------|-------|----------|--------------|----------|------|----------|
| **Local Whisper** | Slow | Good | Unlimited | Unlimited | Free | Offline, private |
| **OpenAI Whisper API** | Fast | Best | 25 min* | 25 MB | $0.006/min | Timestamps, translation |
| **OpenAI gpt-4o-transcribe** | Fast | Best | 25 min | 25 MB | $0.006/min | Enhanced accuracy |
| **Deepgram Nova-3** | Fastest | Good | Unlimited | Unlimited | $0.0043/min | Real-time, diarization |
| **AssemblyAI** | Fast | Good | Unlimited | Unlimited | $0.0025/min | Sentiment, chapters, PII redaction |
| **Google Cloud STT** | Fast | Good | 480 min | 2 GB | $0.016/min | 125+ languages |

*OpenAI requires chunking for longer audio

### Vision Providers

| Provider | Image | Video | Max Images | Tokens/Image | Cost |
|----------|-------|-------|------------|--------------|------|
| **Claude (via host)** | Yes | No | Unlimited* | ~1,600 (1024px) | Included |
| **Claude API** | Yes | No | 20/request | ~1,600 (1024px) | $0.80/M input |
| **OpenAI GPT-4o** | Yes | No | 10/request | 85-6,240 | $2.50/M input |
| **Google Gemini 2.0** | Yes | **Yes** | Unlimited | ~300/sec video | $0.10/M input |
| **Local LLaVA** | Yes | No | 1 | N/A | Free |
| **Local Moondream** | Yes | No | 1 | N/A | Free |

*Via Claude Code, limited by conversation context

### Text Analysis Providers

| Provider | Summarization | Q&A | Structured Output | Cost |
|----------|---------------|-----|-------------------|------|
| **Claude (via host)** | Yes | Yes | Yes | Included |
| **Any LLM API** | Yes | Yes | Varies | Varies |
| **Local LLM (Ollama)** | Yes | Yes | Limited | Free |

---

## Core Insight: Capabilities-Driven Architecture

### The Problem with Current Design

The current data flow is **backwards**:

```
┌─────────────────────────────────────────────────────────────────┐
│ CURRENT: visual.json is primary, entities derived              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Frames ──► Claude Vision ──► visual.json ──► entities.py      │
│               (hard-coded)     (fixed schema)   (reads objects) │
│                                                                 │
│  Problems:                                                      │
│  • Can't use other vision providers                             │
│  • Can't use native video (Gemini)                              │
│  • visual.json schema limits what we can extract                │
│  • entities.py just reads what's already there                  │
└─────────────────────────────────────────────────────────────────┘
```

### The Solution: Entities-First, Capabilities-Driven

```
┌─────────────────────────────────────────────────────────────────┐
│ PROPOSED: Entities are primary, visual.json derived            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Capabilities ──► Best Provider ──► Entities ──► visual.json   │
│  Matrix            Selection        (primary)    (derived)     │
│                                                                 │
│  Benefits:                                                      │
│  • Provider choice based on what's available + capable         │
│  • Gemini video can extract everything in one call             │
│  • Richer entity types (speakers, brands, code constructs)     │
│  • visual.json generated for backwards compatibility           │
└─────────────────────────────────────────────────────────────────┘
```

### Capabilities Matrix

Every provider declares what it can do:

```python
# src/claudetube/providers/capabilities.py

class Capability(Enum):
    TRANSCRIBE = "transcribe"      # Audio → text
    VISION = "vision"              # Image → text
    VIDEO = "video"                # Native video → text (Gemini only)
    REASON = "reason"              # Text → text (LLM chat)
    EMBED = "embed"                # Content → vector

@dataclass
class ProviderCapabilities:
    name: str
    capabilities: set[Capability]

    # Capability-specific limits
    transcription: TranscriptionLimits | None = None
    vision: VisionLimits | None = None
    video: VideoLimits | None = None
    reasoning: ReasoningLimits | None = None

    # Cost estimation
    cost_per_1m_input_tokens: float | None = None
    cost_per_minute_audio: float | None = None
```

**Built-in Capabilities Matrix:**

| Provider | TRANSCRIBE | VISION | VIDEO | REASON | EMBED | JSON Schema | Key Limits |
|----------|:----------:|:------:|:-----:|:------:|:-----:|:-----------:|------------|
| `whisper-local` | ✓ | | | | | | Unlimited, slow |
| `openai-whisper` | ✓ | | | | | | 25MB, 25min |
| `deepgram` | ✓ | | | | | | Diarization, streaming |
| `claude-api` | | ✓ | | ✓ | | **✓** | 20 images/req |
| `gpt-4o` | | ✓ | | ✓ | | **✓** | 10 images/req |
| `gemini-2.0-flash` | | ✓ | **✓** | ✓ | | **✓** | **2hr video**, 2M context |
| `ollama-llava` | | ✓ | | | | | 1 image/req, local |
| `voyage-multimodal-3` | | | | | ✓ | | 1024d, multimodal |

**JSON Schema** = Supports structured output (guaranteed valid JSON in our schema)

### Entity Types and Required Capabilities

| Entity Type | Required Capability | Best Provider |
|-------------|--------------------|--------------|
| **Objects** (visual) | `VISION` or `VIDEO` | Gemini (tracks across video) |
| **People** (visual) | `VISION` or `VIDEO` | Gemini (tracks individuals) |
| **Text on screen** | `VISION` or `VIDEO` | Any vision model |
| **Concepts** (semantic) | `REASON` | LLM on transcript |
| **Speakers** | `TRANSCRIBE` + diarization | Deepgram, AssemblyAI |
| **Code constructs** | `VISION` + `REASON` | Claude/GPT-4o (understands code) |

### Capability-Driven Entity Extraction

```python
class EntityExtractor:
    """Extract entities using best available provider for each type."""

    def __init__(self, providers: ClaudeTubeProviders):
        self.providers = providers
        self.caps = providers.get_capabilities_matrix()

    def extract_all(self, video_path: Path, transcript: str) -> EntityResult:
        entities = EntityResult()

        # VISUAL ENTITIES (objects, people, text on screen)
        if self.caps.can(Capability.VIDEO):
            # BEST: Native video - single API call, tracks objects across time
            entities.visual = self.providers.analyze_video_native(
                video_path,
                prompt=VISUAL_ENTITY_PROMPT
            )
        elif self.caps.can(Capability.VISION):
            # GOOD: Frame-by-frame analysis
            for frame in self._key_frames(video_path):
                frame_entities = self.providers.caption_frame(frame, VISUAL_ENTITY_PROMPT)
                entities.visual.merge(frame_entities)
        # else: No visual entities available

        # SEMANTIC CONCEPTS (from transcript)
        if self.caps.can(Capability.REASON):
            # LLM understands "React hooks" is a concept
            entities.concepts = self.providers.reason([{
                "role": "user",
                "content": f"Extract key concepts:\n{transcript}"
            }])
        else:
            # Fallback: TF-IDF keywords
            entities.concepts = tfidf_extract(transcript)

        # SPEAKERS (if diarization available)
        if self.caps.can(Capability.TRANSCRIBE, feature="diarization"):
            entities.speakers = self.providers.transcribe_with_diarization(...)

        return entities

    def generate_visual_json(self, entities: EntityResult, scene_id: int) -> dict:
        """Generate visual.json from entities (backwards compatibility)."""
        return {
            "scene_id": scene_id,
            "description": entities.get_scene_description(scene_id),
            "objects": entities.get_objects_in_scene(scene_id),
            "people": entities.get_people_in_scene(scene_id),
            "text_on_screen": entities.get_text_in_scene(scene_id),
            "actions": entities.get_actions_in_scene(scene_id),
        }
```

### Smart Routing Based on Capabilities

```python
def select_best_provider(task: str, constraints: dict) -> str:
    """Select optimal provider based on task and constraints."""

    available = get_available_providers()  # Has API key configured

    if task == "visual_entities":
        # Prefer native video for long content
        if constraints.get("duration_sec", 0) > 600:  # > 10 min
            if "gemini" in available and caps["gemini"].can(VIDEO):
                return "gemini"  # Single API call for whole video

        # Prefer multi-image vision for shorter content
        for p in ["claude-api", "gpt-4o", "gemini"]:
            if p in available and caps[p].can(VISION):
                return p

        # Fallback to local
        if "ollama-llava" in available:
            return "ollama-llava"

        return None  # No visual extraction possible

    if task == "transcription":
        duration = constraints.get("duration_sec", 0)

        # Short audio: OpenAI is fastest and most accurate
        if duration < 1500 and "openai-whisper" in available:
            return "openai-whisper"

        # Long audio with diarization needed
        if constraints.get("need_diarization") and "deepgram" in available:
            return "deepgram"

        # Default: local (free, unlimited)
        return "whisper-local"
```

### Fallback Hierarchy: Claude Code is Always Available

**Key principle:** Claude Code (the host AI) is the **ultimate fallback** for vision and reasoning. It's always available when running inside Claude Code, requires no API key, and costs nothing extra.

```
┌─────────────────────────────────────────────────────────────────┐
│                    FALLBACK HIERARCHY                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  User's preferred provider (from config)                        │
│           │                                                     │
│           ▼ (if fails 4xx/5xx)                                  │
│  Next provider in fallback chain                                │
│           │                                                     │
│           ▼ (if all configured providers fail)                  │
│  ┌─────────────────────────────────────┐                        │
│  │  CLAUDE CODE (always available)     │ ◄── Ultimate fallback │
│  │  • No API key needed                │                        │
│  │  • No extra cost                    │                        │
│  │  • Works if you're running in CC    │                        │
│  └─────────────────────────────────────┘                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**When Claude Code vision gets used:**

1. **No config** - User has no API keys, Claude Code handles vision by default
2. **Explicit choice** - User sets `vision.provider: claude-code`
3. **Fallback** - Configured provider returns 4xx/5xx, fall back to Claude Code

```python
class VisionProviderChain:
    """Try providers in order, Claude Code is always the final fallback."""

    def __init__(self, config: ProvidersConfig):
        self.preferred = config.vision.provider
        self.fallback_chain = config.fallbacks.vision or []
        self.claude_code = ClaudeCodeVisionProvider()  # Always available

    async def analyze(self, images: list[Path], prompt: str) -> str:
        providers_to_try = [self.preferred] + self.fallback_chain

        for provider_name in providers_to_try:
            provider = get_provider(provider_name)
            if not provider.is_available():
                continue

            try:
                return await provider.analyze_images(images, prompt)
            except ProviderError as e:
                if e.status_code in (400, 401, 403, 429, 500, 502, 503):
                    logger.warning(f"{provider_name} failed ({e.status_code}), trying next")
                    continue
                raise

        # Ultimate fallback: Claude Code (if running inside Claude Code)
        if self.claude_code.is_available():
            logger.info("Falling back to Claude Code for vision")
            return await self.claude_code.analyze_images(images, prompt)

        raise NoProviderAvailable("No vision provider available")
```

**Claude Code detection:**
```python
class ClaudeCodeVisionProvider(VisionProvider):
    """Uses the host Claude instance - always available inside Claude Code."""

    def is_available(self) -> bool:
        # Available when running as MCP server inside Claude Code
        # Detection: check if we're in an MCP context
        return os.environ.get("MCP_SERVER") == "1" or self._detect_claude_code_context()

    async def analyze_images(self, images: list[Path], prompt: str) -> str:
        # Return paths for host AI to see in conversation
        # The host Claude will analyze them directly
        image_refs = "\n".join(f"[Image: {p}]" for p in images)
        return f"{image_refs}\n\n{prompt}"
```

### Config: Declare Available Providers

```yaml
# ~/.config/claudetube/config.yaml

providers:
  # API keys make providers "available"
  openai:
    api_key: ${OPENAI_API_KEY}

  anthropic:
    api_key: ${ANTHROPIC_API_KEY}

  google:
    api_key: ${GOOGLE_API_KEY}

  deepgram:
    api_key: ${DEEPGRAM_API_KEY}

  # Local providers are always available if installed
  local:
    whisper_model: small
    ollama_models: [llava:13b, moondream]

  # Preferences (optional - system will choose smartly if not specified)
  preferences:
    transcription: openai-whisper  # Prefer this when available
    vision: gemini                  # Or claude-code to use host AI
    video: gemini                   # Only Gemini supports native video
    reasoning: claude-api

  # Explicit fallback chains (optional - defaults shown)
  fallbacks:
    vision: [claude-api, gpt-4o, claude-code]  # claude-code is ALWAYS last resort
    reasoning: [claude-api, gpt-4o, claude-code]
    transcription: [openai-whisper, whisper-local]  # No claude-code (can't transcribe)
```

**Zero-config behavior:**
```yaml
# If user has NO config, defaults are:
providers:
  preferences:
    transcription: whisper-local   # Free, always works
    vision: claude-code            # Host AI, free
    reasoning: claude-code         # Host AI, free
    video: none                    # Requires explicit Gemini config
```

### Structured Output: Force Our Schema

Both OpenAI and Anthropic support **structured outputs** - we define the JSON schema, they guarantee valid output:

```python
# src/claudetube/providers/schemas.py

from pydantic import BaseModel

class VisualEntity(BaseModel):
    """A visual entity detected in frame/video."""
    name: str
    category: Literal["object", "person", "text", "ui_element", "code"]
    first_seen_sec: float
    last_seen_sec: float | None = None
    confidence: float = 1.0
    attributes: dict[str, str] = {}  # e.g., {"color": "blue", "brand": "Apple"}

class SemanticConcept(BaseModel):
    """A concept discussed in the content."""
    term: str
    definition: str
    importance: Literal["primary", "secondary", "mentioned"]
    first_mention_sec: float
    related_terms: list[str] = []

class EntityExtractionResult(BaseModel):
    """Complete entity extraction result - OUR schema, enforced by API."""
    objects: list[VisualEntity]
    people: list[VisualEntity]
    text_on_screen: list[VisualEntity]
    concepts: list[SemanticConcept]
    speakers: list[str] = []
    code_snippets: list[dict] = []  # language, content, timestamp
```

**OpenAI Structured Output:**
```python
response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": [image, prompt]}],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "entity_extraction",
            "schema": EntityExtractionResult.model_json_schema()
        }
    }
)
# Guaranteed valid JSON matching our schema
entities = EntityExtractionResult.model_validate_json(response.choices[0].message.content)
```

**Anthropic Structured Output:**
```python
response = anthropic.messages.create(
    model="claude-sonnet-4-20250514",
    messages=[{"role": "user", "content": [image, prompt]}],
    # Use tool_choice to force structured output
    tools=[{
        "name": "extract_entities",
        "description": "Extract entities from visual content",
        "input_schema": EntityExtractionResult.model_json_schema()
    }],
    tool_choice={"type": "tool", "name": "extract_entities"}
)
# Guaranteed valid JSON matching our schema
entities = EntityExtractionResult.model_validate(response.content[0].input)
```

**Gemini Structured Output:**
```python
import google.generativeai as genai

model = genai.GenerativeModel(
    "gemini-2.0-flash",
    generation_config=genai.GenerationConfig(
        response_mime_type="application/json",
        response_schema=EntityExtractionResult  # Pydantic model
    )
)
response = model.generate_content([video_file, prompt])
entities = EntityExtractionResult.model_validate_json(response.text)
```

**Benefits of Enforced Schema:**

| Without Structured Output | With Structured Output |
|--------------------------|----------------------|
| Parse free-form text, hope for JSON | Guaranteed valid JSON |
| "laptop" vs "Laptop" vs "a laptop" | Normalized by schema |
| Missing fields crash parsing | Schema enforces required fields |
| Different providers = different formats | Same schema across all providers |
| Retry on parse failure | Never fails to parse |

**Capability Flag:**
```python
@dataclass
class ReasoningLimits:
    max_context_tokens: int | None = None
    supports_structured_output: bool = True  # OpenAI, Anthropic, Gemini
    supports_vision: bool = True
```

### MCP Tool: Query Capabilities

```python
@mcp.tool()
def get_provider_capabilities() -> dict:
    """Show what AI capabilities are available."""
    return {
        "available_providers": {
            "whisper-local": {
                "capabilities": ["transcribe"],
                "status": "ready",
                "limits": {"max_duration": "unlimited"}
            },
            "openai-whisper": {
                "capabilities": ["transcribe"],
                "status": "ready",  # API key configured
                "limits": {"max_size_mb": 25, "max_duration_sec": 1500}
            },
            "gemini-2.0-flash": {
                "capabilities": ["vision", "video", "reason"],
                "status": "ready",
                "limits": {"video_max_duration_sec": 7200}
            },
            "deepgram": {
                "capabilities": ["transcribe"],
                "status": "not_configured",  # No API key
                "features": ["diarization", "streaming"]
            }
        },
        "best_for": {
            "transcription": "openai-whisper",
            "vision": "gemini-2.0-flash",
            "video_analysis": "gemini-2.0-flash",
            "reasoning": "claude-api"
        }
    }
```

---

## Architectural Validation

### Current Architecture Analysis

The codebase already has **nascent provider patterns** we can build on:

| Component | Current Pattern | Assessment |
|-----------|----------------|------------|
| **Embeddings** (`analysis/embeddings.py`) | `get_embedding_model()` dispatches to `_embed_scene_voyage()` or `_embed_scene_local()` | **Best template** - clean dispatch pattern |
| **Vision** (`operations/visual_transcript.py`) | `get_vision_model()` returns "claude"/"molmo"/"llava", but only Claude implemented | Intent exists, needs completion |
| **Whisper** (`tools/whisper.py`) | `WhisperTool(VideoTool)` - single implementation | Needs provider abstraction |

The embeddings module is our architectural template:

```python
# Current pattern in analysis/embeddings.py - THIS IS WHAT WE WANT TO GENERALIZE
def get_embedding_model() -> str:
    """Dispatch based on CLAUDETUBE_EMBEDDING_MODEL env var."""
    model = os.environ.get("CLAUDETUBE_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    if model not in ("voyage", "local"):
        logger.warning(f"Unknown model '{model}', falling back...")
    return model

def embed_scene(..., model: str | None = None) -> SceneEmbedding:
    if model is None:
        model = get_embedding_model()

    if model == "voyage":
        embedding = _embed_scene_voyage(scene_text, kf_paths)  # Direct SDK
    else:
        embedding = _embed_scene_local(scene_text, kf_paths)   # Local fallback

    return SceneEmbedding(...)
```

### Hybrid Approach: LiteLLM + Direct SDKs

After analysis, the recommended architecture uses **LiteLLM for generic reasoning** and **direct SDKs for specialized capabilities**:

```
┌─────────────────────────────────────────────────────────────────┐
│                    ClaudeTubeProviders (Facade)                 │
│  Single entry point that routes to the right specialized tool  │
└─────────────────────────────────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
        ▼                       ▼                       ▼
┌───────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   REASONING   │     │  TRANSCRIPTION  │     │  SPECIALIZED    │
│   (LiteLLM)   │     │  (Direct SDKs)  │     │  (Direct SDKs)  │
├───────────────┤     ├─────────────────┤     ├─────────────────┤
│ • Summarize   │     │ • faster-whisper│     │ • Gemini video  │
│ • Q&A         │     │ • OpenAI Whisper│     │ • Voyage embed  │
│ • Analysis    │     │ • Deepgram      │     │ • Claude vision │
│ • Any LLM OK  │     │ • AssemblyAI    │     │   (when needed) │
└───────────────┘     └─────────────────┘     └─────────────────┘
       │                      │                       │
       ▼                      ▼                       ▼
  Claude/GPT/        Timestamps, speaker      Multimodal embed,
  Gemini/local       diarization, formats     native video, OCR
```

### Why This Hybrid Approach?

| Task | Use LiteLLM? | Why |
|------|--------------|-----|
| "Summarize this video" | **Yes** | Any good LLM works; want fallback & cost tracking |
| "Answer questions about content" | **Yes** | Generic reasoning; model choice is preference |
| Transcribe audio | **No** | Need word-level timestamps, specific formats, diarization |
| Analyze 90-minute video | **No** | Only Gemini can do this; direct `google-genai` SDK |
| Generate scene embeddings | **No** | Voyage multimodal-3 is specialized; direct SDK |
| Visual frame captioning | **Maybe** | LiteLLM supports Claude/GPT-4o vision, but local needs Ollama |

### The ClaudeTubeProviders Facade

```python
# src/claudetube/providers/facade.py

class ClaudeTubeProviders:
    """Unified entry point for all AI capabilities.

    Routes to the optimal provider based on task type:
    - Reasoning: LiteLLM (model flexibility, fallbacks, cost tracking)
    - Transcription: Direct SDKs (timestamps, diarization, formats)
    - Video: Gemini direct (unique native video capability)
    - Embeddings: Voyage direct (specialized multimodal format)
    """

    def __init__(self, config: ProvidersConfig | None = None):
        self.config = config or load_providers_config()
        self._litellm_model = self.config.reasoning.model

    def reason(self, messages: list[dict], **kwargs) -> str:
        """Unified reasoning - uses LiteLLM for flexibility.

        Good for: summarization, Q&A, analysis, content understanding.
        """
        from litellm import completion
        response = completion(model=self._litellm_model, messages=messages, **kwargs)
        return response.choices[0].message.content

    def transcribe(self, audio_path: Path, **kwargs) -> TranscriptionResult:
        """Transcription - direct SDK for specialized features.

        Returns word-level timestamps, handles diarization.
        """
        provider = self._get_transcription_provider()
        return provider.transcribe(audio_path, **kwargs)

    def analyze_video_native(self, video_path: Path, prompt: str) -> str:
        """Native video analysis - Gemini direct SDK.

        Skips frame extraction entirely for long videos.
        Only available when Gemini is configured.
        """
        if not self.config.video.provider == "gemini":
            raise ProviderNotConfigured("Native video requires Gemini")

        import google.generativeai as genai
        video_file = genai.upload_file(video_path)
        model = genai.GenerativeModel(self.config.video.gemini.model)
        return model.generate_content([video_file, prompt]).text

    def embed_scene(self, image: Path | None, text: str) -> np.ndarray:
        """Multimodal embedding - Voyage direct for quality.

        Falls back to local (CLIP + sentence-transformers) if unavailable.
        """
        provider = self._get_embedding_provider()
        return provider.embed(image, text)

    def caption_frame(self, frame: Path, prompt: str) -> str:
        """Visual captioning - route based on config.

        Can use LiteLLM (Claude/GPT-4o) or local (Ollama).
        """
        if self.config.vision.provider == "local":
            return self._caption_local(frame, prompt)
        else:
            # LiteLLM handles Claude, GPT-4o, Gemini vision
            return self._caption_litellm(frame, prompt)
```

### Decision Matrix: When to Use What

| If the user wants to... | Provider Approach | Reasoning |
|------------------------|-------------------|-----------|
| Process a 90-minute lecture | Gemini direct | Native video, no frame extraction |
| Get fast transcription | OpenAI Whisper direct | 25MB chunks, word timestamps |
| Summarize video content | LiteLLM | Any LLM works, want fallbacks |
| Answer follow-up questions | LiteLLM | Can swap models, cost optimize |
| Search scenes semantically | Voyage direct | voyage-multimodal-3 embeddings |
| Compare model quality | LiteLLM | Same interface, different models |
| Run fully offline | Local providers | faster-whisper + Ollama LLaVA |

### Dependency Strategy

```python
# Core (always installed)
faster-whisper  # Local transcription

# Optional provider groups
[litellm]       # pip install claudetube[litellm]
litellm>=1.0

[openai]        # pip install claudetube[openai]
openai>=1.0

[google]        # pip install claudetube[google]
google-generativeai>=0.5

[deepgram]      # pip install claudetube[deepgram]
deepgram-sdk>=3.0

[local-vision]  # pip install claudetube[local-vision]
ollama>=0.3
```

This keeps the core lightweight while allowing users to install only what they need.

---

## Proposed Architecture

### Provider Interface Hierarchy

```
AIProvider (base)
├── TranscriptionProvider
│   ├── WhisperLocalProvider (default)
│   ├── OpenAIWhisperProvider
│   ├── DeepgramProvider
│   └── AssemblyAIProvider
├── VisionProvider
│   ├── ClaudeCodeVisionProvider (default)
│   ├── ClaudeAPIVisionProvider
│   ├── OpenAIVisionProvider
│   ├── GeminiVisionProvider
│   └── LocalVisionProvider (LLaVA/Moondream)
├── VideoProvider
│   └── GeminiVideoProvider
└── TextAnalysisProvider
    ├── ClaudeCodeTextProvider (default)
    └── GenericLLMProvider
```

### Abstract Base Classes

```python
# src/claudetube/providers/base.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

@dataclass
class ProviderCapabilities:
    """Describes what a provider can do."""
    max_file_size_mb: float | None = None  # None = unlimited
    max_duration_seconds: float | None = None
    supports_timestamps: bool = True
    supports_streaming: bool = False
    supports_diarization: bool = False
    supported_formats: list[str] | None = None
    cost_per_minute: float | None = None  # None = free/included

class TranscriptionProvider(ABC):
    """Base class for audio transcription providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier (e.g., 'whisper-local', 'openai-whisper')."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is configured and ready."""
        ...

    @abstractmethod
    def get_capabilities(self) -> ProviderCapabilities:
        """Return provider capabilities and limits."""
        ...

    @abstractmethod
    async def transcribe(
        self,
        audio_path: Path,
        language: str | None = None,
        **kwargs
    ) -> TranscriptionResult:
        """Transcribe audio file."""
        ...

class VisionProvider(ABC):
    """Base class for image analysis providers."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def is_available(self) -> bool: ...

    @abstractmethod
    async def analyze_image(
        self,
        image_path: Path,
        prompt: str,
        **kwargs
    ) -> str:
        """Analyze a single image with a prompt."""
        ...

    @abstractmethod
    async def analyze_images(
        self,
        image_paths: list[Path],
        prompt: str,
        **kwargs
    ) -> str:
        """Analyze multiple images with a prompt."""
        ...

class VideoProvider(ABC):
    """Base class for direct video analysis (e.g., Gemini)."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def is_available(self) -> bool: ...

    @abstractmethod
    async def analyze_video(
        self,
        video_path: Path,
        prompt: str,
        start_time: float | None = None,
        end_time: float | None = None,
        **kwargs
    ) -> str:
        """Analyze video content directly."""
        ...
```

### Result Types

```python
# src/claudetube/providers/types.py

from dataclasses import dataclass

@dataclass
class TranscriptionSegment:
    """A segment of transcribed audio."""
    start: float  # seconds
    end: float
    text: str
    confidence: float | None = None
    speaker: str | None = None  # for diarization

@dataclass
class TranscriptionResult:
    """Complete transcription result."""
    text: str  # full transcript
    segments: list[TranscriptionSegment]
    language: str | None = None
    duration: float | None = None
    provider: str = ""

    def to_srt(self) -> str:
        """Convert to SRT format."""
        ...

    def to_vtt(self) -> str:
        """Convert to WebVTT format."""
        ...
```

---

## Configuration Design

### Config File Structure

Extend `config.yaml` to support provider configuration:

```yaml
# ~/.config/claudetube/config.yaml

cache_dir: ~/video_cache

# AI Provider Configuration
providers:
  # Transcription provider selection
  transcription:
    # Provider to use: whisper-local, openai-whisper, deepgram, assemblyai
    provider: whisper-local

    # Provider-specific settings
    whisper-local:
      model: small  # tiny, base, small, medium, large
      device: auto  # auto, cpu, cuda
      compute_type: int8  # int8, float16, float32

    openai-whisper:
      api_key: ${OPENAI_API_KEY}  # env var reference
      model: whisper-1  # or gpt-4o-transcribe

    deepgram:
      api_key: ${DEEPGRAM_API_KEY}
      model: nova-2
      features:
        - diarization
        - punctuation

    assemblyai:
      api_key: ${ASSEMBLYAI_API_KEY}
      features:
        - speaker_labels
        - auto_chapters

  # Vision provider selection
  vision:
    # Provider: claude-code, claude-api, openai, gemini, local
    provider: claude-code  # default: use host AI

    claude-api:
      api_key: ${ANTHROPIC_API_KEY}
      model: claude-3-haiku-20240307

    openai:
      api_key: ${OPENAI_API_KEY}
      model: gpt-4o
      detail: auto  # auto, low, high

    gemini:
      api_key: ${GOOGLE_API_KEY}
      model: gemini-2.0-flash

    local:
      backend: ollama  # ollama, transformers
      model: llava:13b  # or moondream

  # Video analysis (only Gemini currently supports this)
  video:
    provider: none  # none, gemini

    gemini:
      api_key: ${GOOGLE_API_KEY}
      model: gemini-2.0-flash
      max_duration: 3600  # 1 hour

  # Text analysis provider
  text:
    provider: claude-code  # default: use host AI

    # Alternative: use a specific API
    openai:
      api_key: ${OPENAI_API_KEY}
      model: gpt-4o-mini

# Fallback chain (try providers in order until one works)
fallbacks:
  transcription:
    - openai-whisper  # try API first (faster)
    - whisper-local   # fall back to local

  vision:
    - claude-code     # try host AI first (free)
    - claude-api      # fall back to API
```

### Environment Variable Support

```yaml
# Environment variables can be referenced with ${VAR} syntax
providers:
  transcription:
    openai-whisper:
      api_key: ${OPENAI_API_KEY}
```

Resolution order for API keys:
1. Explicit value in config
2. Environment variable reference in config
3. Standard environment variable (e.g., `OPENAI_API_KEY`)
4. Provider-specific env var (e.g., `CLAUDETUBE_OPENAI_API_KEY`)

### Claude Code Host Integration

The "claude-code" provider is special - it uses the host AI that's running claudetube:

```python
class ClaudeCodeVisionProvider(VisionProvider):
    """Uses the host Claude instance for vision tasks.

    When claudetube runs inside Claude Code, the host AI can see
    images directly without needing a separate API call. This is:
    - Free (no additional API cost)
    - Fast (no network round-trip)
    - Context-aware (sees conversation history)

    Implementation: Returns image paths for the MCP tool response,
    letting the host AI analyze them in the conversation context.
    """

    @property
    def name(self) -> str:
        return "claude-code"

    def is_available(self) -> bool:
        # Always available when running in Claude Code
        return os.environ.get("CLAUDE_CODE") == "1"

    async def analyze_image(self, image_path: Path, prompt: str) -> str:
        # Return instruction for host AI instead of making API call
        return f"[Image at {image_path}]\n\nAnalyze this image: {prompt}"
```

---

## Implementation Plan

### Phase 1: Foundation - Base Classes and First Providers

**Goal**: Establish the provider architecture with two working providers.

**Tasks:**

1. **Create `providers/` package with base contracts**
   ```
   src/claudetube/providers/
   ├── __init__.py          # Public API
   ├── base.py              # Provider ABC, Capability enum, Protocols
   ├── capabilities.py      # ProviderInfo dataclass
   ├── types.py             # TranscriptionResult, etc.
   └── registry.py          # get_provider(), list_available()
   ```

2. **Implement `whisper_local/` provider**
   ```python
   # Wrap existing tools/whisper.py
   class WhisperLocalProvider(Provider, Transcriber):
       info = ProviderInfo(
           name="whisper-local",
           capabilities={Capability.TRANSCRIBE},
           # No limits - unlimited local processing
       )
   ```

3. **Implement `claude_code/` provider**
   ```python
   # Ultimate fallback - always available
   class ClaudeCodeProvider(Provider, VisionAnalyzer, Reasoner):
       info = ProviderInfo(
           name="claude-code",
           capabilities={Capability.VISION, Capability.REASON},
           supports_structured_output=True,
       )
   ```

4. **Create simple router**
   ```python
   # For now: just return available provider
   def get_transcriber() -> Transcriber
   def get_vision() -> VisionAnalyzer
   ```

**Deliverable**: Two working providers, existing functionality preserved, foundation for more.

### Phase 2: OpenAI Provider

**Goal**: Full-featured OpenAI provider with transcription + vision + reasoning.

```
src/claudetube/providers/openai/
├── __init__.py
├── client.py        # OpenAIProvider class
└── chunker.py       # Audio chunking for 25MB limit
```

**Capabilities:**
- `TRANSCRIBE`: Whisper API (whisper-1, gpt-4o-transcribe)
- `VISION`: GPT-4o image analysis
- `REASON`: Chat completions with structured output

**Key features:**
- Auto-chunking for audio > 25MB
- Structured output with JSON schema enforcement
- Word-level timestamps

**Deliverable**: `get_provider("openai")` returns fully functional provider.

### Phase 3: Anthropic Provider

**Goal**: Anthropic provider with vision + reasoning.

```
src/claudetube/providers/anthropic/
├── __init__.py
└── client.py        # AnthropicProvider class
```

**Capabilities:**
- `VISION`: Claude vision (multi-image)
- `REASON`: Messages API with tool-based structured output

**Key features:**
- Up to 20 images per request
- Structured output via tool_choice
- Extracts existing `_generate_visual_claude()` logic

**Deliverable**: `get_provider("anthropic")` - migrate existing Claude code.

### Phase 4: Google Provider (Gemini)

**Goal**: Google provider with vision + **native video** + reasoning.

```
src/claudetube/providers/google/
├── __init__.py
├── client.py        # GoogleProvider class
└── file_api.py      # Video upload handling
```

**Capabilities:**
- `VISION`: Gemini image analysis
- `VIDEO`: **Native video analysis** (unique!)
- `REASON`: Chat with structured output

**Key features:**
- Upload videos up to 2GB via File API
- Analyze 2-hour videos in single call
- ~300 tokens/second of video

**Deliverable**: `get_provider("google").analyze_video()` works.

### Phase 5: Transcription Specialists

**Goal**: Add Deepgram and AssemblyAI for advanced transcription features.

```
src/claudetube/providers/deepgram/
└── client.py        # DeepgramProvider - diarization, streaming

src/claudetube/providers/assemblyai/
└── client.py        # AssemblyAIProvider - chapters, sentiment
```

**Deepgram capabilities:**
- Fastest transcription (20 sec/hour of audio)
- Speaker diarization
- Real-time streaming

**AssemblyAI capabilities:**
- Auto-chapters
- Sentiment analysis
- PII redaction

**Deliverable**: Choice of transcription backends based on needs.

### Phase 6: Local Providers

**Goal**: Fully offline operation with Ollama.

```
src/claudetube/providers/ollama/
└── client.py        # OllamaProvider - local LLaVA/Moondream

src/claudetube/providers/voyage/
└── client.py        # VoyageProvider - embeddings (migrate from embeddings.py)
```

**Ollama capabilities:**
- `VISION`: LLaVA, Moondream (1 image at a time)
- `REASON`: Any Ollama model

**Deliverable**: Fully offline video analysis pipeline.

### Phase 7: Router and Config

**Goal**: Smart routing, fallback chains, YAML config.

```
src/claudetube/providers/
├── router.py        # ProviderRouter with fallbacks
└── config.py        # Load from YAML, env var interpolation
```

**Features:**
- Automatic provider selection based on capabilities
- Fallback on 4xx/5xx errors
- Claude Code as ultimate fallback
- YAML config with `${ENV_VAR}` support

**Deliverable**: Production-ready routing with graceful degradation.

### Phase 8: AI-Powered Entity Extraction with Structured Outputs

**Goal**: Replace TF-IDF with LLM extraction; use structured outputs for guaranteed schema compliance.

**Current limitation** (`cache/entities.py`):
```python
# Objects: Read from visual.json (fixed schema, already extracted)
objects = track_objects_from_scenes(scenes)  # Just reads existing data

# Concepts: TF-IDF keyword extraction (no semantic understanding)
concepts = track_concepts_from_scenes(scenes)  # "python", "def", "class"
```

**Enhanced approach: Structured outputs enforce OUR schema:**

1. **Define the schema once (Pydantic)**
   ```python
   # src/claudetube/providers/schemas.py

   class VisualEntity(BaseModel):
       name: str
       category: Literal["object", "person", "text", "code", "ui"]
       first_seen_sec: float
       last_seen_sec: float | None = None
       bounding_box: tuple[int, int, int, int] | None = None  # If available

   class SemanticConcept(BaseModel):
       term: str
       definition: str
       importance: Literal["primary", "secondary", "mentioned"]
       first_mention_sec: float

   class EntityExtractionResult(BaseModel):
       """OUR schema - enforced by all providers."""
       objects: list[VisualEntity]
       people: list[VisualEntity]
       text_on_screen: list[VisualEntity]
       concepts: list[SemanticConcept]
       code_snippets: list[dict] = []
   ```

2. **Extract with guaranteed JSON (OpenAI example)**
   ```python
   def extract_entities_openai(frames: list[Path], transcript: str) -> EntityExtractionResult:
       response = openai.chat.completions.create(
           model="gpt-4o",
           messages=[{
               "role": "user",
               "content": [
                   *[{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode(f)}"}} for f in frames],
                   {"type": "text", "text": f"Extract entities. Transcript for context:\n{transcript}"}
               ]
           }],
           response_format={
               "type": "json_schema",
               "json_schema": {
                   "name": "entity_extraction",
                   "schema": EntityExtractionResult.model_json_schema()
               }
           }
       )
       # GUARANTEED valid - no try/except needed
       return EntityExtractionResult.model_validate_json(response.choices[0].message.content)
   ```

3. **Extract with Gemini native video**
   ```python
   def extract_entities_gemini_video(video_path: Path) -> EntityExtractionResult:
       """Single API call for entire video - tracks entities across time."""
       model = genai.GenerativeModel(
           "gemini-2.0-flash",
           generation_config=genai.GenerationConfig(
               response_mime_type="application/json",
               response_schema=EntityExtractionResult  # Enforced!
           )
       )
       video_file = genai.upload_file(video_path)
       response = model.generate_content([
           video_file,
           "Extract all entities with timestamps. Track objects/people across the video."
       ])
       return EntityExtractionResult.model_validate_json(response.text)
   ```

4. **Extract with Anthropic (tool use for schema)**
   ```python
   def extract_entities_anthropic(frames: list[Path], transcript: str) -> EntityExtractionResult:
       response = anthropic.messages.create(
           model="claude-sonnet-4-20250514",
           messages=[{"role": "user", "content": [...frames, transcript]}],
           tools=[{
               "name": "store_entities",
               "description": "Store extracted entities",
               "input_schema": EntityExtractionResult.model_json_schema()
           }],
           tool_choice={"type": "tool", "name": "store_entities"}  # Force it
       )
       return EntityExtractionResult.model_validate(response.content[0].input)
   ```

**Why structured outputs matter:**

| Without | With Structured Output |
|---------|----------------------|
| `"a blue laptop"` | `{"name": "laptop", "attributes": {"color": "blue"}}` |
| Parse errors, retries | Guaranteed valid JSON |
| Provider-specific formats | Same schema everywhere |
| `"React"` vs `"react"` vs `"ReactJS"` | Normalized by schema |

**Generate visual.json from entities (backwards compat):**
```python
def generate_visual_json(entities: EntityExtractionResult, scene: SceneBoundary) -> dict:
    """Derive visual.json from extracted entities."""
    scene_objects = [e for e in entities.objects if e.first_seen_sec <= scene.end_time and (e.last_seen_sec or inf) >= scene.start_time]

    return {
        "scene_id": scene.scene_id,
        "description": generate_description(scene_objects, entities.concepts),
        "objects": [e.name for e in scene_objects if e.category == "object"],
        "people": [e.name for e in scene_objects if e.category == "person"],
        "text_on_screen": [e.name for e in scene_objects if e.category == "text"],
        "model_used": "structured_extraction",
    }
```

**Using the clean provider API:**
```python
# Entity extraction uses providers directly
from claudetube.providers import get_provider
from claudetube.providers.base import Capability

async def extract_entities(video_path: Path, transcript: str) -> EntityExtractionResult:
    # Try native video first (best quality, single call)
    google = get_provider("google")
    if google.is_available() and Capability.VIDEO in google.info.capabilities:
        return await google.analyze_video(
            video_path,
            prompt="Extract entities...",
            schema=EntityExtractionResult
        )

    # Fall back to vision (frame-by-frame)
    vision = get_provider("openai")  # or anthropic, ollama
    if vision.is_available() and Capability.VISION in vision.info.capabilities:
        frames = extract_key_frames(video_path)
        return await vision.analyze_images(
            frames,
            prompt="Extract entities...",
            schema=EntityExtractionResult
        )

    # Ultimate fallback: Claude Code
    claude_code = get_provider("claude-code")
    return await claude_code.analyze_images(frames, "Extract entities...")
```

**Deliverable**: Reliable entity extraction with consistent schema across all providers.

---

## New Capabilities Unlocked

With configurable AI providers, claudetube gains these capabilities:

### Currently Impossible → Now Possible

| Capability | Current State | With Providers |
|------------|--------------|----------------|
| **90-min lecture analysis** | Extract 5,400 frames, analyze each | Single Gemini API call (~$2.70) |
| **Speaker identification** | Not possible | Deepgram/AssemblyAI diarization |
| **Fast transcription** | 30+ min for 1hr video | 20 seconds (Deepgram) |
| **Object continuity** | "laptop" in scene 1, "laptop" in scene 5 = unrelated | Gemini tracks same object across video |
| **Semantic concepts** | TF-IDF keywords: "python", "def", "class" | LLM extraction: "Python decorators", "dependency injection" |
| **Code understanding** | OCR text extraction | Vision model understands the code's purpose |
| **Offline processing** | Requires Claude API key | Local Whisper + Ollama LLaVA |
| **Model comparison** | Hard-coded to Claude | A/B test Claude vs GPT-4o vs Gemini |

### Example: Tutorial Video Analysis

**Current workflow (frame-based):**
```
1. Download video (30 sec)
2. Transcribe with local Whisper (25 min for 1hr video)
3. Detect scenes (45 sec)
4. Extract 180 keyframes (2 min)
5. Analyze each frame with Claude API (180 API calls, ~$0.15)
6. Extract entities from visual.json (instant)
Total: ~30 minutes, $0.15
```

**With Gemini native video:**
```
1. Download video (30 sec)
2. Upload to Gemini File API (10 sec)
3. Single prompt: "Analyze this tutorial..." (30 sec, ~$1.80)
4. Rich structured output includes transcript, scenes, entities
Total: ~1 minute, $1.80
```

**With fast cloud transcription:**
```
1. Download video (30 sec)
2. Transcribe with Deepgram (20 sec for 1hr video)
3. ... rest of pipeline ...
Total: ~5 minutes instead of 30
```

### Entity Extraction: TF-IDF vs LLM vs Video

Given a coding tutorial transcript:
```
"Today we'll learn about React hooks. The useState hook lets you add state
to functional components. We'll also cover useEffect for side effects..."
```

| Method | Output |
|--------|--------|
| **TF-IDF (current)** | `["react", "hooks", "usestate", "components", "useeffect"]` |
| **LLM reasoning** | `[{"term": "React hooks", "definition": "Functions that let you use state and lifecycle features in functional components", "importance": "Core React concept since v16.8"}, ...]` |
| **Gemini video** | Above + "Code shown at 2:15 demonstrates useState with counter example" + "Presenter draws diagram at 4:30 showing hook execution order" |

---

## File Structure: One Module Per Provider

**Principle:** Each provider is its own module. Clean APIs, separation of concerns, easy to mock/test.

```
src/claudetube/providers/
├── __init__.py              # Public API: get_provider(), registry
├── base.py                  # ABC, types, shared schemas
├── capabilities.py          # Capability enum, limits dataclasses
├── registry.py              # Auto-discovery, get_provider(name)
├── router.py                # Smart routing, fallback chains
│
├── whisper_local/
│   ├── __init__.py          # from .client import WhisperLocalProvider
│   └── client.py            # Single class, ~100 lines
│
├── openai/
│   ├── __init__.py          # from .client import OpenAIProvider
│   ├── client.py            # Transcription + Vision + Reasoning
│   └── chunker.py           # Audio chunking for 25MB limit
│
├── anthropic/
│   ├── __init__.py
│   └── client.py            # Vision + Reasoning (structured output)
│
├── google/
│   ├── __init__.py
│   ├── client.py            # Vision + Video + Reasoning
│   └── file_api.py          # Video upload handling
│
├── deepgram/
│   ├── __init__.py
│   └── client.py            # Transcription (diarization, streaming)
│
├── assemblyai/
│   ├── __init__.py
│   └── client.py            # Transcription (chapters, sentiment)
│
├── voyage/
│   ├── __init__.py
│   └── client.py            # Embeddings (multimodal)
│
├── ollama/
│   ├── __init__.py
│   └── client.py            # Vision + Reasoning (local)
│
└── claude_code/
    ├── __init__.py
    └── client.py            # Vision + Reasoning (host AI, always available)
```

### Provider Contract: `base.py`

```python
# src/claudetube/providers/base.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Protocol

class Capability(Enum):
    TRANSCRIBE = auto()
    VISION = auto()
    VIDEO = auto()
    REASON = auto()
    EMBED = auto()

@dataclass(frozen=True)
class ProviderInfo:
    """Immutable provider metadata."""
    name: str
    capabilities: frozenset[Capability]
    supports_structured_output: bool = False
    # Capability-specific limits
    max_audio_size_mb: float | None = None
    max_audio_duration_sec: float | None = None
    max_images_per_request: int | None = None
    max_video_duration_sec: float | None = None
    supports_diarization: bool = False
    supports_streaming: bool = False

class Provider(ABC):
    """Base class for all providers. Each provider implements only what it supports."""

    @property
    @abstractmethod
    def info(self) -> ProviderInfo:
        """Return provider capabilities and limits."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is configured and ready (API key set, etc.)."""
        ...

# Capability-specific protocols (providers implement relevant ones)

class Transcriber(Protocol):
    async def transcribe(self, audio: Path, language: str | None = None) -> TranscriptionResult: ...

class VisionAnalyzer(Protocol):
    async def analyze_images(self, images: list[Path], prompt: str) -> str: ...

class VideoAnalyzer(Protocol):
    async def analyze_video(self, video: Path, prompt: str) -> str: ...

class Reasoner(Protocol):
    async def reason(self, messages: list[dict], schema: type | None = None) -> str | dict: ...

class Embedder(Protocol):
    async def embed(self, text: str, images: list[Path] | None = None) -> list[float]: ...
```

### Example Provider: `openai/client.py`

```python
# src/claudetube/providers/openai/client.py

from pathlib import Path
from claudetube.providers.base import (
    Provider, ProviderInfo, Capability,
    Transcriber, VisionAnalyzer, Reasoner,
    TranscriptionResult
)

class OpenAIProvider(Provider, Transcriber, VisionAnalyzer, Reasoner):
    """OpenAI provider - transcription, vision, and reasoning."""

    def __init__(self, api_key: str | None = None, model: str = "gpt-4o"):
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._model = model
        self._client: openai.OpenAI | None = None

    @property
    def info(self) -> ProviderInfo:
        return ProviderInfo(
            name="openai",
            capabilities=frozenset({Capability.TRANSCRIBE, Capability.VISION, Capability.REASON}),
            supports_structured_output=True,
            max_audio_size_mb=25,
            max_audio_duration_sec=1500,
            max_images_per_request=10,
        )

    def is_available(self) -> bool:
        return self._api_key is not None

    @property
    def client(self) -> openai.OpenAI:
        if self._client is None:
            import openai
            self._client = openai.OpenAI(api_key=self._api_key)
        return self._client

    async def transcribe(self, audio: Path, language: str | None = None) -> TranscriptionResult:
        """Transcribe audio using Whisper API."""
        from .chunker import chunk_if_needed

        chunks = chunk_if_needed(audio, max_mb=25)
        segments = []

        for chunk in chunks:
            response = self.client.audio.transcriptions.create(
                file=open(chunk.path, "rb"),
                model="whisper-1",
                response_format="verbose_json",
                language=language,
            )
            segments.extend(self._parse_segments(response, offset=chunk.offset))

        return TranscriptionResult(
            text=" ".join(s.text for s in segments),
            segments=segments,
            provider="openai",
        )

    async def analyze_images(self, images: list[Path], prompt: str) -> str:
        """Analyze images using GPT-4o vision."""
        content = [
            *[{"type": "image_url", "image_url": {"url": self._encode_image(p)}} for p in images],
            {"type": "text", "text": prompt}
        ]
        response = self.client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": content}],
        )
        return response.choices[0].message.content

    async def reason(self, messages: list[dict], schema: type | None = None) -> str | dict:
        """Chat completion with optional structured output."""
        kwargs = {"model": self._model, "messages": messages}

        if schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": schema.model_json_schema()}
            }
            response = self.client.chat.completions.create(**kwargs)
            return schema.model_validate_json(response.choices[0].message.content)

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content
```

### Example Provider: `claude_code/client.py`

```python
# src/claudetube/providers/claude_code/client.py

class ClaudeCodeProvider(Provider, VisionAnalyzer, Reasoner):
    """Host AI provider - always available inside Claude Code, no API key needed."""

    @property
    def info(self) -> ProviderInfo:
        return ProviderInfo(
            name="claude-code",
            capabilities=frozenset({Capability.VISION, Capability.REASON}),
            supports_structured_output=True,  # Host Claude supports it
        )

    def is_available(self) -> bool:
        # Always available when running as MCP server in Claude Code
        return os.environ.get("MCP_SERVER") == "1" or self._detect_mcp_context()

    async def analyze_images(self, images: list[Path], prompt: str) -> str:
        """Return image paths for host AI to analyze in conversation context."""
        # The host Claude will see these images and analyze them
        return "\n".join(f"[Image: {p}]" for p in images) + f"\n\n{prompt}"

    async def reason(self, messages: list[dict], schema: type | None = None) -> str | dict:
        """Reasoning handled by host AI in conversation context."""
        # Format for host AI to process
        formatted = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        if schema:
            formatted += f"\n\nRespond with JSON matching: {schema.model_json_schema()}"
        return formatted
```

### Usage: Independent, Injectable, Testable

```python
# Direct instantiation
from claudetube.providers.openai import OpenAIProvider
from claudetube.providers.deepgram import DeepgramProvider

openai = OpenAIProvider(api_key="sk-...")
deepgram = DeepgramProvider(api_key="...")

# Dependency injection
class VideoProcessor:
    def __init__(self, transcriber: Transcriber, vision: VisionAnalyzer):
        self.transcriber = transcriber
        self.vision = vision

    async def process(self, video: Path):
        transcript = await self.transcriber.transcribe(video.with_suffix(".mp3"))
        frames = self.extract_frames(video)
        description = await self.vision.analyze_images(frames, "Describe")
        return {"transcript": transcript, "description": description}

# Easy to mock for testing
class MockTranscriber:
    async def transcribe(self, audio: Path, language: str | None = None):
        return TranscriptionResult(text="mock transcript", segments=[], provider="mock")

processor = VideoProcessor(transcriber=MockTranscriber(), vision=openai)
```

### Registry: Get Provider by Name

```python
# src/claudetube/providers/registry.py

from importlib import import_module

PROVIDER_MODULES = {
    "openai": "claudetube.providers.openai",
    "anthropic": "claudetube.providers.anthropic",
    "google": "claudetube.providers.google",
    "deepgram": "claudetube.providers.deepgram",
    "assemblyai": "claudetube.providers.assemblyai",
    "voyage": "claudetube.providers.voyage",
    "ollama": "claudetube.providers.ollama",
    "claude-code": "claudetube.providers.claude_code",
    "whisper-local": "claudetube.providers.whisper_local",
}

_cache: dict[str, Provider] = {}

def get_provider(name: str, **kwargs) -> Provider:
    """Get provider instance by name. Cached for reuse."""
    if name not in _cache:
        module = import_module(PROVIDER_MODULES[name])
        provider_class = getattr(module, f"{name.title().replace('-', '')}Provider")
        _cache[name] = provider_class(**kwargs)
    return _cache[name]

def list_available() -> list[str]:
    """List providers that are configured and ready."""
    return [name for name in PROVIDER_MODULES if get_provider(name).is_available()]
```

---

## Operations Layer Refactoring

Each operation in `src/claudetube/operations/` will be refactored to accept providers via dependency injection.

### Current State: Hard-coded Dependencies

| Operation | Current Dependency | Problem |
|-----------|-------------------|---------|
| `transcribe.py` | `WhisperTool` directly | Can't use OpenAI/Deepgram |
| `visual_transcript.py` | `anthropic.Anthropic` directly | Can't use GPT-4o/Gemini/local |
| `knowledge_graph.py` | `TfidfVectorizer` (sklearn) | No semantic understanding |
| `person_tracking.py` | Reads `visual.json` + `face_recognition` | Limited to existing data |
| `code_evolution.py` | Reads `technical.json` (OCR) | No code understanding |
| `change_detection.py` | Reads `visual.json` + embeddings | Uses existing data (OK) |

### Target State: Provider Injection

```python
# BEFORE: Hard-coded dependency
# src/claudetube/operations/transcribe.py

from claudetube.tools.whisper import WhisperTool

def transcribe_video(video_id: str, whisper_model: str = "small") -> dict:
    tool = WhisperTool(model_size=whisper_model)  # Hard-coded!
    return tool.transcribe(audio_path)
```

```python
# AFTER: Injected provider
# src/claudetube/operations/transcribe.py

from claudetube.providers.base import Transcriber

class TranscribeOperation:
    """Transcribe video audio using any transcription provider."""

    def __init__(self, transcriber: Transcriber):
        self.transcriber = transcriber

    async def execute(self, video_id: str, audio_path: Path) -> dict:
        result = await self.transcriber.transcribe(audio_path)
        return {
            "success": True,
            "video_id": video_id,
            "transcript": result.text,
            "segments": [s.to_dict() for s in result.segments],
            "provider": result.provider,
        }

# Usage with dependency injection
from claudetube.providers import get_provider

transcriber = get_provider("openai")  # or "whisper-local", "deepgram"
op = TranscribeOperation(transcriber)
result = await op.execute(video_id, audio_path)
```

### Operation → Provider Mapping

| Operation | Provider Protocol | Capabilities Used |
|-----------|------------------|-------------------|
| `TranscribeOperation` | `Transcriber` | `TRANSCRIBE` |
| `VisualTranscriptOperation` | `VisionAnalyzer` | `VISION` |
| `KnowledgeGraphOperation` | `Reasoner` | `REASON` |
| `PersonTrackingOperation` | `VisionAnalyzer` | `VISION` |
| `CodeEvolutionOperation` | `VisionAnalyzer` + `Reasoner` | `VISION`, `REASON` |
| `EntityExtractionOperation` | `VisionAnalyzer` or `VideoAnalyzer` | `VISION` or `VIDEO` |

### Refactored Operations

#### `TranscribeOperation`

```python
# src/claudetube/operations/transcribe.py

from claudetube.providers.base import Transcriber, TranscriptionResult

class TranscribeOperation:
    def __init__(self, transcriber: Transcriber):
        self.transcriber = transcriber

    async def execute(
        self,
        audio_path: Path,
        language: str | None = None,
    ) -> TranscriptionResult:
        return await self.transcriber.transcribe(audio_path, language=language)
```

#### `VisualTranscriptOperation`

```python
# src/claudetube/operations/visual_transcript.py

from claudetube.providers.base import VisionAnalyzer
from claudetube.providers.schemas import VisualDescription

class VisualTranscriptOperation:
    def __init__(self, vision: VisionAnalyzer):
        self.vision = vision

    async def execute(
        self,
        keyframes: list[Path],
        transcript_context: str,
    ) -> VisualDescription:
        # Use structured output if provider supports it
        if self.vision.info.supports_structured_output:
            return await self.vision.analyze_images(
                keyframes,
                prompt=VISUAL_DESCRIPTION_PROMPT,
                schema=VisualDescription,
            )
        else:
            # Parse free-form response
            response = await self.vision.analyze_images(keyframes, prompt=VISUAL_DESCRIPTION_PROMPT)
            return self._parse_response(response)
```

#### `KnowledgeGraphOperation`

```python
# src/claudetube/operations/knowledge_graph.py

from claudetube.providers.base import Reasoner

class KnowledgeGraphOperation:
    def __init__(self, reasoner: Reasoner | None = None):
        self.reasoner = reasoner  # Optional - falls back to TF-IDF

    async def extract_topics(self, texts: list[str]) -> list[dict]:
        if self.reasoner and self.reasoner.is_available():
            # LLM-based semantic extraction
            return await self.reasoner.reason(
                messages=[{"role": "user", "content": f"Extract key topics from:\n{texts}"}],
                schema=TopicList,
            )
        else:
            # Fallback: TF-IDF
            return self._tfidf_extract(texts)
```

#### `PersonTrackingOperation`

```python
# src/claudetube/operations/person_tracking.py

from claudetube.providers.base import VisionAnalyzer, VideoAnalyzer

class PersonTrackingOperation:
    def __init__(
        self,
        vision: VisionAnalyzer | None = None,
        video: VideoAnalyzer | None = None,
    ):
        self.vision = vision
        self.video = video

    async def execute(self, video_path: Path, scenes: list[SceneBoundary]) -> PeopleTrackingData:
        # Best: Native video analysis (tracks across time)
        if self.video and self.video.is_available():
            return await self._track_with_video(video_path)

        # Good: Vision analysis per scene
        if self.vision and self.vision.is_available():
            return await self._track_with_vision(scenes)

        # Fallback: Read from existing visual.json
        return self._track_from_cache(scenes)

    async def _track_with_video(self, video_path: Path) -> PeopleTrackingData:
        """Use Gemini to track people across entire video."""
        result = await self.video.analyze_video(
            video_path,
            prompt="Track all people in this video. For each person, provide timestamps of appearances.",
            schema=PeopleTrackingData,
        )
        return result
```

#### `EntityExtractionOperation`

```python
# src/claudetube/operations/entities.py

from claudetube.providers.base import VisionAnalyzer, VideoAnalyzer, Reasoner
from claudetube.providers.schemas import EntityExtractionResult

class EntityExtractionOperation:
    """Extract entities using best available provider."""

    def __init__(
        self,
        vision: VisionAnalyzer | None = None,
        video: VideoAnalyzer | None = None,
        reasoner: Reasoner | None = None,
    ):
        self.vision = vision
        self.video = video
        self.reasoner = reasoner

    async def execute(
        self,
        video_path: Path,
        transcript: str,
        frames: list[Path] | None = None,
    ) -> EntityExtractionResult:
        result = EntityExtractionResult()

        # Visual entities (objects, people, text on screen)
        if self.video and self.video.is_available():
            # BEST: Native video - single call, tracks across time
            visual = await self.video.analyze_video(
                video_path,
                prompt=VISUAL_ENTITY_PROMPT,
                schema=VisualEntities,
            )
            result.merge_visual(visual)
        elif self.vision and frames:
            # GOOD: Frame-by-frame
            visual = await self.vision.analyze_images(
                frames,
                prompt=VISUAL_ENTITY_PROMPT,
                schema=VisualEntities,
            )
            result.merge_visual(visual)

        # Semantic concepts (from transcript)
        if self.reasoner and self.reasoner.is_available():
            concepts = await self.reasoner.reason(
                messages=[{"role": "user", "content": f"Extract concepts:\n{transcript}"}],
                schema=ConceptList,
            )
            result.merge_concepts(concepts)
        else:
            # Fallback: TF-IDF
            result.merge_concepts(tfidf_extract(transcript))

        return result
```

### Factory for Operations

```python
# src/claudetube/operations/factory.py

from claudetube.providers import get_provider
from claudetube.providers.router import ProviderRouter

class OperationFactory:
    """Create operations with appropriate providers based on config."""

    def __init__(self, router: ProviderRouter):
        self.router = router

    def create_transcribe(self) -> TranscribeOperation:
        return TranscribeOperation(self.router.get_transcriber())

    def create_visual_transcript(self) -> VisualTranscriptOperation:
        return VisualTranscriptOperation(self.router.get_vision())

    def create_entity_extraction(self) -> EntityExtractionOperation:
        return EntityExtractionOperation(
            vision=self.router.get_vision(),
            video=self.router.get_video(),  # May be None
            reasoner=self.router.get_reasoner(),
        )

    def create_person_tracking(self) -> PersonTrackingOperation:
        return PersonTrackingOperation(
            vision=self.router.get_vision(),
            video=self.router.get_video(),
        )

# Usage
router = ProviderRouter(config)
factory = OperationFactory(router)

transcribe_op = factory.create_transcribe()
result = await transcribe_op.execute(audio_path)
```

### Testing with Mock Providers

```python
# tests/test_transcribe_operation.py

import pytest
from unittest.mock import AsyncMock

class MockTranscriber:
    """Mock transcriber for testing."""

    @property
    def info(self):
        return ProviderInfo(name="mock", capabilities={Capability.TRANSCRIBE})

    def is_available(self) -> bool:
        return True

    async def transcribe(self, audio: Path, **kwargs) -> TranscriptionResult:
        return TranscriptionResult(
            text="This is a test transcript.",
            segments=[],
            provider="mock",
        )

@pytest.mark.asyncio
async def test_transcribe_operation():
    mock = MockTranscriber()
    op = TranscribeOperation(transcriber=mock)

    result = await op.execute(Path("/fake/audio.mp3"))

    assert result.text == "This is a test transcript."
    assert result.provider == "mock"
```

### Migration Path

1. **Phase 1**: Add provider parameters with defaults
   ```python
   def transcribe_video(
       video_id: str,
       transcriber: Transcriber | None = None,  # NEW: optional
   ):
       if transcriber is None:
           transcriber = get_provider("whisper-local")  # Default behavior
       ...
   ```

2. **Phase 2**: Refactor to class-based operations
   ```python
   class TranscribeOperation:
       def __init__(self, transcriber: Transcriber): ...
   ```

3. **Phase 3**: Update MCP tools to use factory
   ```python
   @mcp.tool()
   def transcribe_video_tool(url: str, provider: str | None = None):
       op = factory.create_transcribe()
       return await op.execute(...)
   ```

---

---

## Analysis Layer Refactoring

The `src/claudetube/analysis/` layer contains modules that could benefit from provider abstraction.

### Current State

| Module | Current Implementation | AI Type | Provider Opportunity |
|--------|----------------------|---------|---------------------|
| `embeddings.py` | Voyage AI / local CLIP | **Already provider-patterned** | Generalize to `Embedder` protocol |
| `vector_index.py` | ChromaDB + embeddings | Uses `embeddings.py` | N/A |
| `search.py` | Text regex + embedding fallback | Uses `embeddings.py` | Add `Reasoner` for query expansion |
| `ocr.py` | EasyOCR (local) | Local library | Add `VisionAnalyzer` option |
| `visual.py` | PySceneDetect | Local library | N/A (not AI-based) |
| `unified.py` | Combines cheap methods | TF-IDF/regex | Add `Reasoner` for semantic detection |
| `linguistic.py` | Regex patterns | Pattern matching | Add `Reasoner` for semantic transitions |
| `vocabulary.py` | TF-IDF (sklearn) | Local ML | Add `Reasoner` for semantic shifts |
| `code.py` | Regex patterns | Pattern matching | Add `VisionAnalyzer` + `Reasoner` |
| `pause.py` | Timing analysis | No AI | N/A |

### Embeddings: Already Provider-Patterned (Template)

The `embeddings.py` module is our **reference implementation** - it already does provider dispatch:

```python
# Current pattern in analysis/embeddings.py - THIS IS THE TEMPLATE

def get_embedding_model() -> str:
    """Dispatch based on env var."""
    return os.environ.get("CLAUDETUBE_EMBEDDING_MODEL", "voyage")

def embed_scene(..., model: str | None = None) -> SceneEmbedding:
    if model is None:
        model = get_embedding_model()

    if model == "voyage":
        embedding = _embed_scene_voyage(...)  # Voyage SDK
    else:
        embedding = _embed_scene_local(...)   # CLIP + sentence-transformers

    return SceneEmbedding(...)
```

**Migration**: Extract to `providers/voyage/client.py` and implement `Embedder` protocol.

### OCR: EasyOCR vs Vision API

Current OCR uses EasyOCR (local). Vision APIs often do better OCR:

```python
# BEFORE: Local EasyOCR only
# src/claudetube/analysis/ocr.py

def extract_text_from_frame(frame_path: Path) -> FrameOCRResult:
    reader = get_reader()  # EasyOCR
    results = reader.readtext(str(frame_path))
    ...
```

```python
# AFTER: Provider-based with fallback
# src/claudetube/analysis/ocr.py

class OCRExtractor:
    def __init__(self, vision: VisionAnalyzer | None = None):
        self.vision = vision
        self._easyocr_reader = None

    async def extract_text(self, frame_path: Path) -> FrameOCRResult:
        # Try vision API first (better for code, handwriting)
        if self.vision and self.vision.is_available():
            result = await self.vision.analyze_images(
                [frame_path],
                prompt="Extract ALL text visible in this image. Return exact text, preserving formatting.",
                schema=OCRResult,
            )
            return self._convert_vision_result(result)

        # Fallback: EasyOCR (local, always available)
        return self._extract_with_easyocr(frame_path)
```

### Search: Add Query Expansion

Current search does text matching then embedding fallback. Could add LLM query expansion:

```python
# AFTER: Provider-enhanced search
# src/claudetube/analysis/search.py

class SemanticSearch:
    def __init__(
        self,
        embedder: Embedder | None = None,
        reasoner: Reasoner | None = None,
    ):
        self.embedder = embedder
        self.reasoner = reasoner

    async def search(self, query: str, cache_dir: Path, top_k: int = 5) -> list[SearchMoment]:
        # 1. TEXT - Fast transcript search
        results = self._search_transcript_text(cache_dir, query, top_k)
        if results:
            return results

        # 2. EXPAND - Use LLM to expand query (if available)
        expanded_query = query
        if self.reasoner and self.reasoner.is_available():
            expanded = await self.reasoner.reason([{
                "role": "user",
                "content": f"Expand this search query with synonyms and related terms: '{query}'"
            }])
            expanded_query = f"{query} {expanded}"

        # 3. EMBEDDINGS - Vector similarity search
        if self.embedder:
            return await self._search_embeddings(cache_dir, expanded_query, top_k)

        return []
```

### Unified Boundary Detection: Add Semantic Option

Current boundary detection uses TF-IDF and regex. Could add LLM-based detection:

```python
# AFTER: Provider-enhanced boundary detection
# src/claudetube/analysis/unified.py

class BoundaryDetector:
    def __init__(self, reasoner: Reasoner | None = None):
        self.reasoner = reasoner

    async def detect_boundaries(
        self,
        transcript_segments: list[dict],
        video_info: dict | None = None,
    ) -> list[Boundary]:
        # 1. CHEAP - YouTube chapters (highest confidence)
        boundaries = self._extract_chapters(video_info)

        # 2. CHEAP - Linguistic patterns (regex)
        boundaries.extend(self._detect_linguistic_patterns(transcript_segments))

        # 3. SEMANTIC - LLM-based detection (if available and needed)
        if self.reasoner and len(boundaries) < MIN_CHEAP_BOUNDARIES:
            semantic = await self._detect_semantic_boundaries(transcript_segments)
            boundaries.extend(semantic)

        return self._merge_boundaries(boundaries)

    async def _detect_semantic_boundaries(self, segments: list[dict]) -> list[Boundary]:
        """Use LLM to find topic transitions."""
        transcript = " ".join(s["text"] for s in segments)

        result = await self.reasoner.reason([{
            "role": "user",
            "content": f"""Analyze this transcript and identify major topic transitions.
For each transition, provide:
- timestamp (approximate seconds)
- topic_before: what was being discussed
- topic_after: what the new topic is

Transcript:
{transcript[:10000]}"""  # Limit context
        }], schema=TopicTransitions)

        return [
            Boundary(
                timestamp=t.timestamp,
                type="semantic",
                trigger_text=f"{t.topic_before} → {t.topic_after}",
                confidence=0.75,
            )
            for t in result.transitions
        ]
```

### Analysis Layer Provider Mapping

| Module | Provider Protocol | When Used |
|--------|------------------|-----------|
| `embeddings.py` | `Embedder` | Always (migrate existing) |
| `ocr.py` | `VisionAnalyzer` | Optional enhancement |
| `search.py` | `Embedder` + `Reasoner` | Embeddings required, reasoning optional |
| `unified.py` | `Reasoner` | Optional for semantic detection |
| `code.py` | `VisionAnalyzer` + `Reasoner` | Optional for code understanding |

### Migration Priority

1. **High**: `embeddings.py` → Extract to `providers/voyage/` (already patterned)
2. **Medium**: `ocr.py` → Add vision provider option
3. **Medium**: `search.py` → Add query expansion
4. **Low**: `unified.py`, `code.py` → Semantic enhancement (optional)

---

### Router: Smart Capability-Based Selection

```python
# src/claudetube/providers/router.py

class ProviderRouter:
    """Routes requests to best available provider based on capability."""

    def __init__(self, config: ProvidersConfig):
        self.config = config
        self._claude_code = get_provider("claude-code")  # Always available fallback

    def get_transcriber(self) -> Transcriber:
        """Get best available transcription provider."""
        for name in self.config.transcription_preference:
            provider = get_provider(name)
            if provider.is_available() and Capability.TRANSCRIBE in provider.info.capabilities:
                return provider
        return get_provider("whisper-local")  # Always available

    def get_vision(self) -> VisionAnalyzer:
        """Get best available vision provider, Claude Code as fallback."""
        for name in self.config.vision_preference:
            provider = get_provider(name)
            if provider.is_available() and Capability.VISION in provider.info.capabilities:
                return provider
        return self._claude_code  # Always available

    def get_video(self) -> VideoAnalyzer | None:
        """Get video provider (only Gemini supports this)."""
        google = get_provider("google")
        if google.is_available() and Capability.VIDEO in google.info.capabilities:
            return google
        return None  # No fallback - unique capability

    async def with_fallback(self, capability: Capability, operation, *args, **kwargs):
        """Execute operation with automatic fallback on failure."""
        providers = self._get_providers_for(capability)

        for provider in providers:
            try:
                return await operation(provider, *args, **kwargs)
            except ProviderError as e:
                if e.status_code in (400, 401, 403, 429, 500, 502, 503):
                    logger.warning(f"{provider.info.name} failed, trying next")
                    continue
                raise

        # Ultimate fallback for vision/reasoning
        if capability in (Capability.VISION, Capability.REASON):
            return await operation(self._claude_code, *args, **kwargs)

        raise NoProviderAvailable(f"No provider available for {capability}")
```

---

## Configuration Examples

### Default (Zero Config)

```yaml
# No config needed - uses defaults:
# - Transcription: whisper-local (tiny model)
# - Vision: claude-code (host AI)
# - Text: claude-code (host AI)
```

### Power User: Fast Cloud Transcription

```yaml
providers:
  transcription:
    provider: openai-whisper
    openai-whisper:
      model: whisper-1
```

### Privacy-Focused: All Local

```yaml
providers:
  transcription:
    provider: whisper-local
    whisper-local:
      model: large
      device: cuda

  vision:
    provider: local
    local:
      backend: ollama
      model: llava:34b
```

### Cost-Optimized: Cheap Defaults, Quality Fallback

```yaml
providers:
  transcription:
    provider: whisper-local  # free

  vision:
    provider: claude-code    # included

fallbacks:
  transcription:
    - whisper-local
    - deepgram  # if local too slow
```

### Research: Gemini Video Analysis

```yaml
providers:
  video:
    provider: gemini
    gemini:
      api_key: ${GOOGLE_API_KEY}
      model: gemini-2.0-flash
```

---

## API Changes

### New MCP Tool Options

```python
# process_video_tool gains provider options
@mcp.tool()
def process_video_tool(
    url: str,
    transcription_provider: str | None = None,  # override config
    whisper_model: str = "tiny",  # for whisper-local
    force_transcribe: bool = False
) -> dict:
    ...

# New tool for provider info
@mcp.tool()
def list_providers() -> dict:
    """List available AI providers and their status."""
    return {
        "transcription": {
            "configured": "openai-whisper",
            "available": ["whisper-local", "openai-whisper", "deepgram"],
            "status": {
                "whisper-local": {"available": True, "model": "small"},
                "openai-whisper": {"available": True, "model": "whisper-1"},
                "deepgram": {"available": False, "reason": "API key not configured"}
            }
        },
        "vision": {...},
        "video": {...}
    }
```

### Python API

```python
from claudetube.providers import get_transcription_provider, get_vision_provider

# Get configured provider
provider = get_transcription_provider()
result = await provider.transcribe(audio_path)

# Get specific provider
openai = get_transcription_provider("openai-whisper")
if openai.is_available():
    result = await openai.transcribe(audio_path)

# Check capabilities before use
caps = provider.get_capabilities()
if audio_duration > caps.max_duration_seconds:
    # Use chunking or different provider
    ...
```

---

## Migration Path

### Backwards Compatibility

- Existing code continues to work (defaults unchanged)
- Environment variables still work (`ANTHROPIC_API_KEY`, etc.)
- No breaking changes to MCP tools

### Deprecation Plan

| Current | Replacement | Timeline |
|---------|-------------|----------|
| `CLAUDETUBE_VISION_MODEL` env var | `providers.vision.provider` config | Keep both, prefer config |
| `CLAUDETUBE_EMBEDDING_MODEL` env var | `providers.embedding.provider` config | Keep both, prefer config |
| Direct Whisper tool usage | `TranscriptionProvider` interface | Internal only, no user impact |

---

## Success Criteria

### Functional Requirements

- [ ] Default configuration works without any setup
- [ ] Can configure transcription provider via YAML
- [ ] Can configure vision provider via YAML
- [ ] API keys can be provided via env vars or config
- [ ] Fallback chains work when primary provider fails
- [ ] Provider capabilities are queryable

### Performance Requirements

- [ ] Provider selection adds < 10ms overhead
- [ ] Lazy loading of provider dependencies
- [ ] No unnecessary API calls for capability checks

### User Experience

- [ ] Clear error messages when provider misconfigured
- [ ] `list_providers` tool shows available options
- [ ] Documentation covers common configurations

---

## Resolved Design Decisions

### LiteLLM Integration (RESOLVED)

**Decision:** Use LiteLLM for the **reasoning layer only**, direct SDKs for specialized capabilities.

| Capability | Approach | Rationale |
|------------|----------|-----------|
| Reasoning/Analysis | LiteLLM | Model flexibility, fallbacks, unified logging |
| Transcription | Direct SDK | Word timestamps, diarization, format-specific features |
| Native Video | Gemini direct | Unique capability, no LiteLLM equivalent |
| Embeddings | Direct SDK | voyage-multimodal-3 is specialized |
| Vision | Hybrid | LiteLLM for cloud (Claude/GPT-4o), Ollama for local |

LiteLLM becomes an **optional dependency** - installed via `pip install claudetube[litellm]`.

### Gemini Video Integration (RESOLVED)

**Decision:** Gemini native video is a **parallel path**, not a replacement.

```
User requests video analysis
         │
         ▼
┌────────────────────────┐
│ Is video < 5 minutes?  │
│ AND config prefers     │
│ frame-based?           │
└────────────────────────┘
    │ Yes            │ No
    ▼                ▼
┌──────────────┐  ┌──────────────┐
│ Frame-based  │  │ Gemini native│
│ (existing)   │  │ (new)        │
│ Extract HQ   │  │ Upload whole │
│ frames →     │  │ video →      │
│ Claude/GPT   │  │ ~300 tok/sec │
└──────────────┘  └──────────────┘
```

Benefits:
- Long videos (lectures, tutorials): Gemini native is cheaper and faster
- Short/precise analysis: Frame-based gives more control
- User choice via config

### Caching Strategy (RESOLVED)

**Decision:** Cache transcription results **per-provider** with metadata.

```
~/.claude/video_cache/{video_id}/
├── transcripts/
│   ├── whisper-local-small.srt      # Local whisper, small model
│   ├── whisper-local-small.txt
│   ├── openai-whisper-1.srt         # OpenAI API
│   ├── openai-whisper-1.txt
│   └── metadata.json                # Which is "primary"
```

Rationale: Different providers produce different results. User may want to compare or switch without re-processing.

## Open Questions (Remaining)

1. **Cost tracking**: Should we track and report estimated costs? LiteLLM has built-in cost tracking we could expose.

2. **Rate limiting**: Should we implement rate limiting to prevent accidental cost spikes? Or rely on provider defaults?

3. **Claude Code detection**: How do we reliably detect when running inside Claude Code vs standalone? Options:
   - Check for `CLAUDE_CODE` env var (if it exists)
   - Check if MCP server is running in expected context
   - Make it explicit in config (`provider: claude-code`)

4. **Streaming transcription**: Should we support real-time streaming for Deepgram? Adds complexity but enables live use cases.

5. **Provider health checks**: Should we periodically verify API keys are valid? Or only check on first use?

---

## Research Sources

### Transcription APIs
- [OpenAI Whisper API Limits](https://community.openai.com/t/whisper-api-limits-transcriptions/167507)
- [OpenAI Audio API Reference](https://platform.openai.com/docs/api-reference/audio)
- [Deepgram vs Whisper Comparison](https://deepgram.com/learn/whisper-vs-deepgram)
- [AssemblyAI vs Whisper](https://www.edenai.co/post/whisper-vs-assemblyai-best-speech-to-text-api)

### Vision APIs
- [Claude Vision Documentation](https://platform.claude.com/docs/en/build-with-claude/vision)
- [OpenAI Vision Guide](https://platform.openai.com/docs/guides/images-vision)
- [Gemini Video Understanding](https://ai.google.dev/gemini-api/docs/video-understanding)

### Provider Abstraction
- [LiteLLM - Universal LLM Interface](https://github.com/BerriAI/litellm)
- [Pydantic AI Model Overview](https://ai.pydantic.dev/models/overview/)

### Local Models
- [Moondream - Tiny Vision Model](https://github.com/vikhyat/moondream)
- [LLaVA with Ollama](https://eranfeit.net/llava-image-recognition-in-python-with-ollama-and-vision-language-models/)

---

## Appendix A: Provider Capability Details

### OpenAI Whisper API

```
Endpoint: POST https://api.openai.com/v1/audio/transcriptions
Models: whisper-1, gpt-4o-transcribe, gpt-4o-mini-transcribe
Max file size: 25 MB
Max duration: ~25 minutes (gpt-4o-transcribe limited to 1500 seconds)
Formats: flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, webm
Cost: $0.006/minute
Features: timestamps, translation, language detection
```

### Deepgram Nova-3

```
Endpoint: POST https://api.deepgram.com/v1/listen
Max file size: Unlimited (streaming)
Max duration: Unlimited
Real-time latency: <300ms
Cost: $0.0043/minute
Features: diarization, punctuation, paragraphs, smart formatting
```

### Gemini Video

```
Endpoint: Gemini API with File API
Max file size: 2 GB
Max duration: 2 hours (2M context) / 1 hour (1M context)
Token cost: ~300 tokens/second of video
Cost: $0.10/M input tokens (~$0.03/minute of video)
Features: Direct video analysis, timestamp references, multi-turn Q&A
```

---

## Appendix B: Configuration Schema

```python
# JSON Schema for config validation

PROVIDERS_SCHEMA = {
    "type": "object",
    "properties": {
        "transcription": {
            "type": "object",
            "properties": {
                "provider": {
                    "type": "string",
                    "enum": ["whisper-local", "openai-whisper", "deepgram", "assemblyai"]
                },
                "whisper-local": {
                    "type": "object",
                    "properties": {
                        "model": {"type": "string", "enum": ["tiny", "base", "small", "medium", "large"]},
                        "device": {"type": "string", "enum": ["auto", "cpu", "cuda"]},
                        "compute_type": {"type": "string", "enum": ["int8", "float16", "float32"]}
                    }
                },
                "openai-whisper": {
                    "type": "object",
                    "properties": {
                        "api_key": {"type": "string"},
                        "model": {"type": "string"}
                    }
                }
                # ... other providers
            }
        },
        "vision": {
            # Similar structure
        },
        "video": {
            # Similar structure
        }
    }
}
```
