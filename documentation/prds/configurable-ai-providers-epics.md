[← Documentation](../README.md)

# Configurable AI Providers - Epic Breakdown

**Priority:** P0
**PRD:** [configurable-ai-providers.md](./configurable-ai-providers.md)
**Created:** 2026-02-01

---

## Epic Overview

```
EPIC 1: Provider Foundation
    │
    ▼
EPIC 2: Core Providers (OpenAI, Anthropic, Google, Claude Code)
    │
    ▼
EPIC 3: Operations Layer Refactoring
    │
    ▼
EPIC 4: Analysis Layer Migration
    │
    ▼
EPIC 5: Specialist Providers (Deepgram, AssemblyAI, Voyage, Ollama)
    │
    ▼
EPIC 6: Router, Config & Polish
```

---

# EPIC 1: Provider Foundation

**Summary:** Establish the provider architecture with base classes, protocols, types, and registry.

**Blocked by:** Nothing (starting point)
**Blocks:** All other epics

---

## EPIC-1-T1: Create Provider Base Module Structure

**Priority:** P0
**Estimate:** S (Small)
**Blocked by:** None
**Blocks:** EPIC-1-T2, EPIC-1-T3

### Requirements

1. Create `src/claudetube/providers/` package structure
2. Create empty `__init__.py` with public API placeholders
3. Establish module layout for future providers

### Technical Details

```
src/claudetube/providers/
├── __init__.py          # Public API: get_provider, list_available
├── base.py              # ABCs and protocols (EPIC-1-T2)
├── types.py             # Result types (EPIC-1-T3)
├── capabilities.py      # Capability enum and limits (EPIC-1-T4)
├── registry.py          # Provider discovery (EPIC-1-T5)
├── config.py            # Config loading (EPIC-1-T6)
└── router.py            # Smart routing (EPIC-6)
```

### Gotchas / Things to Keep in Mind

- Do NOT add any providers yet - just the structure
- `__init__.py` should have `__all__` defined even if empty
- Use `from __future__ import annotations` in all files for forward refs

### Success Criteria

- [ ] `from claudetube.providers import ...` doesn't error (even if empty)
- [ ] Package structure matches spec
- [ ] All files have proper module docstrings
- [ ] No circular imports

---

## EPIC-1-T2: Define Provider Protocols and Base Classes

**Priority:** P0
**Estimate:** M (Medium)
**Blocked by:** EPIC-1-T1
**Blocks:** EPIC-2 (all provider implementations)

### Requirements

1. Define `Provider` abstract base class
2. Define capability-specific protocols: `Transcriber`, `VisionAnalyzer`, `VideoAnalyzer`, `Reasoner`, `Embedder`
3. Each protocol must be minimal and focused on single capability
4. Protocols must support both sync and async (prefer async)

### Technical Details

```python
# src/claudetube/providers/base.py

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable
from pathlib import Path

from claudetube.providers.capabilities import ProviderInfo
from claudetube.providers.types import TranscriptionResult

class Provider(ABC):
    """Base class for all AI providers."""

    @property
    @abstractmethod
    def info(self) -> ProviderInfo:
        """Return provider capabilities and limits."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is configured and ready (API key set, etc.)."""
        ...


@runtime_checkable
class Transcriber(Protocol):
    """Protocol for audio transcription providers."""

    async def transcribe(
        self,
        audio: Path,
        language: str | None = None,
        **kwargs,
    ) -> TranscriptionResult:
        """Transcribe audio file to text with timestamps."""
        ...


@runtime_checkable
class VisionAnalyzer(Protocol):
    """Protocol for image analysis providers."""

    async def analyze_images(
        self,
        images: list[Path],
        prompt: str,
        schema: type | None = None,
        **kwargs,
    ) -> str | dict:
        """Analyze one or more images with a prompt."""
        ...


@runtime_checkable
class VideoAnalyzer(Protocol):
    """Protocol for native video analysis (Gemini only currently)."""

    async def analyze_video(
        self,
        video: Path,
        prompt: str,
        schema: type | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        **kwargs,
    ) -> str | dict:
        """Analyze video content directly without frame extraction."""
        ...


@runtime_checkable
class Reasoner(Protocol):
    """Protocol for text reasoning/chat completion."""

    async def reason(
        self,
        messages: list[dict],
        schema: type | None = None,
        **kwargs,
    ) -> str | dict:
        """Generate text response, optionally with structured output."""
        ...


@runtime_checkable
class Embedder(Protocol):
    """Protocol for embedding generation."""

    async def embed(
        self,
        text: str,
        images: list[Path] | None = None,
        **kwargs,
    ) -> list[float]:
        """Generate embedding vector for text and optionally images."""
        ...
```

### Gotchas / Things to Keep in Mind

- Use `@runtime_checkable` on protocols for `isinstance()` checks
- Keep protocols MINIMAL - don't add methods "just in case"
- `schema` parameter enables structured output - providers that don't support it should ignore
- Return type `str | dict` allows both free-form and structured responses
- Consider adding `info` property to protocols for capability introspection

### Success Criteria

- [ ] All 5 protocols defined and documented
- [ ] `Provider` ABC defined with `info` and `is_available`
- [ ] Type hints are complete and mypy passes
- [ ] Protocols can be used with `isinstance()` checks
- [ ] Unit test file created with protocol compliance tests

---

## EPIC-1-T3: Define Result Types and Schemas

**Priority:** P0
**Estimate:** M (Medium)
**Blocked by:** EPIC-1-T1
**Blocks:** EPIC-2 (all provider implementations)

### Requirements

1. Define `TranscriptionResult` with segments, timestamps, speaker info
2. Define `TranscriptionSegment` for individual segments
3. Define entity extraction schemas (Pydantic models for structured output)
4. All types must be JSON-serializable
5. Add format conversion methods (to_srt, to_vtt, to_dict)

### Technical Details

```python
# src/claudetube/providers/types.py

from dataclasses import dataclass, field, asdict
from typing import Literal
from pydantic import BaseModel

@dataclass
class TranscriptionSegment:
    """A single segment of transcribed audio."""
    start: float          # seconds
    end: float            # seconds
    text: str
    confidence: float | None = None
    speaker: str | None = None  # For diarization

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TranscriptionResult:
    """Complete transcription result."""
    text: str                           # Full transcript
    segments: list[TranscriptionSegment]
    language: str | None = None
    duration: float | None = None
    provider: str = ""

    def to_srt(self) -> str:
        """Convert to SRT subtitle format."""
        lines = []
        for i, seg in enumerate(self.segments, 1):
            start = self._format_srt_time(seg.start)
            end = self._format_srt_time(seg.end)
            lines.append(f"{i}\n{start} --> {end}\n{seg.text}\n")
        return "\n".join(lines)

    def to_vtt(self) -> str:
        """Convert to WebVTT format."""
        lines = ["WEBVTT\n"]
        for seg in self.segments:
            start = self._format_vtt_time(seg.start)
            end = self._format_vtt_time(seg.end)
            lines.append(f"{start} --> {end}\n{seg.text}\n")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "segments": [s.to_dict() for s in self.segments],
            "language": self.language,
            "duration": self.duration,
            "provider": self.provider,
        }

    @staticmethod
    def _format_srt_time(seconds: float) -> str:
        h, remainder = divmod(int(seconds), 3600)
        m, s = divmod(remainder, 60)
        ms = int((seconds - int(seconds)) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    @staticmethod
    def _format_vtt_time(seconds: float) -> str:
        h, remainder = divmod(int(seconds), 3600)
        m, s = divmod(remainder, 60)
        ms = int((seconds - int(seconds)) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


# Pydantic models for structured output (entity extraction)

class VisualEntity(BaseModel):
    """A visual entity detected in frame/video."""
    name: str
    category: Literal["object", "person", "text", "code", "ui_element"]
    first_seen_sec: float
    last_seen_sec: float | None = None
    confidence: float = 1.0
    attributes: dict[str, str] = {}


class SemanticConcept(BaseModel):
    """A concept discussed in the content."""
    term: str
    definition: str
    importance: Literal["primary", "secondary", "mentioned"]
    first_mention_sec: float
    related_terms: list[str] = []


class EntityExtractionResult(BaseModel):
    """Complete entity extraction result - schema for structured output."""
    objects: list[VisualEntity] = []
    people: list[VisualEntity] = []
    text_on_screen: list[VisualEntity] = []
    concepts: list[SemanticConcept] = []
    code_snippets: list[dict] = []


class VisualDescription(BaseModel):
    """Visual description of a scene - for visual_transcript."""
    description: str
    objects: list[str] = []
    people: list[str] = []
    text_on_screen: list[str] = []
    actions: list[str] = []
    setting: str | None = None
```

### Gotchas / Things to Keep in Mind

- Dataclasses for simple data containers (TranscriptionResult)
- Pydantic models for structured output schemas (entity extraction)
- SRT uses comma for milliseconds, VTT uses period
- Keep schemas flat where possible - deeply nested schemas can confuse LLMs
- `Literal` types help LLMs stick to valid values

### Success Criteria

- [ ] All types defined with full type hints
- [ ] `to_srt()` produces valid SRT format
- [ ] `to_vtt()` produces valid WebVTT format
- [ ] Pydantic models can be serialized to JSON Schema
- [ ] Unit tests for format conversion methods
- [ ] Round-trip test: create → to_dict → from_dict matches original

---

## EPIC-1-T4: Define Capability Enum and ProviderInfo

**Priority:** P0
**Estimate:** S (Small)
**Blocked by:** EPIC-1-T1
**Blocks:** EPIC-1-T5, EPIC-2

### Requirements

1. Define `Capability` enum with all capability types
2. Define `ProviderInfo` dataclass with capability limits
3. Include cost estimation fields
4. Make `ProviderInfo` immutable (frozen dataclass)

### Technical Details

```python
# src/claudetube/providers/capabilities.py

from dataclasses import dataclass
from enum import Enum, auto

class Capability(Enum):
    """AI capabilities that providers can offer."""
    TRANSCRIBE = auto()    # Audio → text
    VISION = auto()        # Image → text
    VIDEO = auto()         # Native video → text (Gemini)
    REASON = auto()        # Text → text (chat/completion)
    EMBED = auto()         # Content → vector


@dataclass(frozen=True)
class ProviderInfo:
    """Immutable provider metadata and capabilities."""
    name: str
    capabilities: frozenset[Capability]

    # Feature flags
    supports_structured_output: bool = False
    supports_streaming: bool = False

    # Transcription limits
    max_audio_size_mb: float | None = None
    max_audio_duration_sec: float | None = None
    supports_diarization: bool = False
    supports_translation: bool = False

    # Vision limits
    max_images_per_request: int | None = None
    max_image_size_mb: float | None = None

    # Video limits (Gemini)
    max_video_duration_sec: float | None = None
    max_video_size_mb: float | None = None
    video_tokens_per_second: float = 300.0

    # Context limits
    max_context_tokens: int | None = None

    # Cost estimation (per unit)
    cost_per_1m_input_tokens: float | None = None
    cost_per_1m_output_tokens: float | None = None
    cost_per_minute_audio: float | None = None

    def can(self, capability: Capability) -> bool:
        """Check if provider has a specific capability."""
        return capability in self.capabilities

    def can_all(self, *capabilities: Capability) -> bool:
        """Check if provider has ALL specified capabilities."""
        return all(c in self.capabilities for c in capabilities)

    def can_any(self, *capabilities: Capability) -> bool:
        """Check if provider has ANY of specified capabilities."""
        return any(c in self.capabilities for c in capabilities)


# Pre-defined provider info (can be overridden by config)
PROVIDER_INFO = {
    "whisper-local": ProviderInfo(
        name="whisper-local",
        capabilities=frozenset({Capability.TRANSCRIBE}),
        supports_diarization=False,
        cost_per_minute_audio=0,  # Free
    ),
    "openai": ProviderInfo(
        name="openai",
        capabilities=frozenset({Capability.TRANSCRIBE, Capability.VISION, Capability.REASON}),
        supports_structured_output=True,
        max_audio_size_mb=25,
        max_audio_duration_sec=1500,
        max_images_per_request=10,
        supports_translation=True,
        cost_per_minute_audio=0.006,
        cost_per_1m_input_tokens=2.50,
    ),
    "anthropic": ProviderInfo(
        name="anthropic",
        capabilities=frozenset({Capability.VISION, Capability.REASON}),
        supports_structured_output=True,
        max_images_per_request=20,
        max_context_tokens=200_000,
        cost_per_1m_input_tokens=3.00,
    ),
    "google": ProviderInfo(
        name="google",
        capabilities=frozenset({Capability.VISION, Capability.VIDEO, Capability.REASON}),
        supports_structured_output=True,
        max_video_duration_sec=7200,  # 2 hours
        max_video_size_mb=2000,
        max_context_tokens=2_000_000,
        cost_per_1m_input_tokens=0.10,
    ),
    "deepgram": ProviderInfo(
        name="deepgram",
        capabilities=frozenset({Capability.TRANSCRIBE}),
        supports_diarization=True,
        supports_streaming=True,
        cost_per_minute_audio=0.0043,
    ),
    "claude-code": ProviderInfo(
        name="claude-code",
        capabilities=frozenset({Capability.VISION, Capability.REASON}),
        supports_structured_output=True,
        cost_per_1m_input_tokens=0,  # Included with Claude Code
    ),
    "ollama": ProviderInfo(
        name="ollama",
        capabilities=frozenset({Capability.VISION, Capability.REASON}),
        max_images_per_request=1,
        cost_per_1m_input_tokens=0,  # Local
    ),
    "voyage": ProviderInfo(
        name="voyage",
        capabilities=frozenset({Capability.EMBED}),
        cost_per_1m_input_tokens=0.06,
    ),
}
```

### Gotchas / Things to Keep in Mind

- Use `frozenset` for capabilities (immutable)
- Use `frozen=True` on dataclass for immutability
- `None` means "unlimited" for limits
- Cost fields are optional - `None` means unknown
- Helper methods `can()`, `can_all()`, `can_any()` make checks readable

### Success Criteria

- [ ] All capabilities defined in enum
- [ ] ProviderInfo is immutable (frozen)
- [ ] Pre-defined info for all planned providers
- [ ] `can()` methods work correctly
- [ ] Unit tests for capability checks

---

## EPIC-1-T5: Create Provider Registry

**Priority:** P0
**Estimate:** M (Medium)
**Blocked by:** EPIC-1-T2, EPIC-1-T4
**Blocks:** EPIC-2, EPIC-3

### Requirements

1. Implement provider discovery via `get_provider(name)` function
2. Lazy loading - don't import provider modules until needed
3. Caching - reuse provider instances
4. `list_available()` returns providers that are configured and ready
5. Support for provider aliases (e.g., "gpt-4o" → "openai")

### Technical Details

```python
# src/claudetube/providers/registry.py

from __future__ import annotations

import logging
from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from claudetube.providers.base import Provider

logger = logging.getLogger(__name__)

# Provider module mapping
PROVIDER_MODULES = {
    "whisper-local": "claudetube.providers.whisper_local",
    "openai": "claudetube.providers.openai",
    "anthropic": "claudetube.providers.anthropic",
    "google": "claudetube.providers.google",
    "deepgram": "claudetube.providers.deepgram",
    "assemblyai": "claudetube.providers.assemblyai",
    "voyage": "claudetube.providers.voyage",
    "ollama": "claudetube.providers.ollama",
    "claude-code": "claudetube.providers.claude_code",
}

# Aliases for convenience
PROVIDER_ALIASES = {
    "whisper": "whisper-local",
    "gpt-4o": "openai",
    "gpt4": "openai",
    "claude": "anthropic",
    "gemini": "google",
    "gemini-2.0-flash": "google",
}

# Cached provider instances
_cache: dict[str, Provider] = {}


def _resolve_name(name: str) -> str:
    """Resolve provider aliases to canonical names."""
    return PROVIDER_ALIASES.get(name.lower(), name.lower())


def get_provider(name: str, **kwargs) -> Provider:
    """Get a provider instance by name.

    Args:
        name: Provider name or alias (e.g., "openai", "gpt-4o", "whisper-local")
        **kwargs: Provider-specific configuration

    Returns:
        Provider instance (cached for reuse)

    Raises:
        ValueError: If provider name is unknown
        ImportError: If provider module fails to import
    """
    canonical = _resolve_name(name)

    # Check cache (only if no kwargs - kwargs might change config)
    cache_key = canonical if not kwargs else None
    if cache_key and cache_key in _cache:
        return _cache[cache_key]

    # Find module
    if canonical not in PROVIDER_MODULES:
        available = ", ".join(sorted(PROVIDER_MODULES.keys()))
        raise ValueError(f"Unknown provider '{name}'. Available: {available}")

    # Import and instantiate
    module_path = PROVIDER_MODULES[canonical]
    try:
        module = import_module(module_path)
    except ImportError as e:
        raise ImportError(
            f"Failed to import provider '{canonical}': {e}. "
            f"You may need to install optional dependencies."
        ) from e

    # Find provider class (convention: {Name}Provider)
    class_name = "".join(word.title() for word in canonical.split("-")) + "Provider"
    if not hasattr(module, class_name):
        # Fallback: look for any Provider subclass
        from claudetube.providers.base import Provider as BaseProvider
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and issubclass(attr, BaseProvider) and attr is not BaseProvider:
                class_name = attr_name
                break
        else:
            raise ImportError(f"No Provider class found in {module_path}")

    provider_class = getattr(module, class_name)
    instance = provider_class(**kwargs)

    # Cache if no custom kwargs
    if cache_key:
        _cache[cache_key] = instance

    logger.debug(f"Loaded provider: {canonical} ({class_name})")
    return instance


def list_available() -> list[str]:
    """List providers that are configured and ready to use.

    Returns:
        List of provider names that have is_available() == True
    """
    available = []
    for name in PROVIDER_MODULES:
        try:
            provider = get_provider(name)
            if provider.is_available():
                available.append(name)
        except (ImportError, Exception) as e:
            logger.debug(f"Provider '{name}' not available: {e}")
    return available


def list_all() -> list[str]:
    """List all known provider names."""
    return sorted(PROVIDER_MODULES.keys())


def clear_cache() -> None:
    """Clear the provider instance cache."""
    _cache.clear()


def get_provider_info(name: str) -> dict:
    """Get provider info without fully initializing the provider.

    Returns basic info from PROVIDER_INFO if available.
    """
    from claudetube.providers.capabilities import PROVIDER_INFO
    canonical = _resolve_name(name)
    if canonical in PROVIDER_INFO:
        info = PROVIDER_INFO[canonical]
        return {
            "name": info.name,
            "capabilities": [c.name for c in info.capabilities],
            "supports_structured_output": info.supports_structured_output,
        }
    return {"name": canonical, "capabilities": [], "supports_structured_output": False}
```

### Gotchas / Things to Keep in Mind

- Lazy import is critical - don't load OpenAI SDK if user doesn't need it
- Cache by canonical name, not alias
- Don't cache if user passes custom kwargs (they might want different config)
- `list_available()` is expensive (imports all providers) - use sparingly
- Provider class naming convention: `{Name}Provider` (e.g., `OpenaiProvider`, `ClaudeCodeProvider`)

### Success Criteria

- [ ] `get_provider("openai")` returns provider instance
- [ ] Aliases work: `get_provider("gpt-4o")` returns OpenAI provider
- [ ] Caching works: same instance returned for same name
- [ ] `list_available()` only returns configured providers
- [ ] Unknown provider raises clear error
- [ ] Import errors have helpful messages about dependencies

---

## EPIC-1-T6: Create Configuration Loader

**Priority:** P0
**Estimate:** M (Medium)
**Blocked by:** EPIC-1-T4, EPIC-1-T5
**Blocks:** EPIC-6

### Requirements

1. Extend existing `config/loader.py` to support `providers:` section
2. Support `${ENV_VAR}` interpolation for API keys
3. Support provider-specific settings
4. Maintain backward compatibility with existing config

### Technical Details

```python
# src/claudetube/providers/config.py

from __future__ import annotations

import os
import re
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ENV_VAR_PATTERN = re.compile(r'\$\{([^}]+)\}')


@dataclass
class ProviderConfig:
    """Configuration for a single provider."""
    api_key: str | None = None
    model: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProvidersConfig:
    """Complete providers configuration."""
    # Provider-specific configs
    openai: ProviderConfig = field(default_factory=ProviderConfig)
    anthropic: ProviderConfig = field(default_factory=ProviderConfig)
    google: ProviderConfig = field(default_factory=ProviderConfig)
    deepgram: ProviderConfig = field(default_factory=ProviderConfig)
    assemblyai: ProviderConfig = field(default_factory=ProviderConfig)
    voyage: ProviderConfig = field(default_factory=ProviderConfig)

    # Local provider configs
    whisper_local_model: str = "small"
    ollama_model: str = "llava:13b"

    # Preferences (which provider to use for each capability)
    transcription_provider: str = "whisper-local"
    vision_provider: str = "claude-code"
    video_provider: str | None = None  # Only Gemini supports this
    reasoning_provider: str = "claude-code"
    embedding_provider: str = "voyage"

    # Fallback chains
    transcription_fallbacks: list[str] = field(default_factory=lambda: ["whisper-local"])
    vision_fallbacks: list[str] = field(default_factory=lambda: ["claude-code"])
    reasoning_fallbacks: list[str] = field(default_factory=lambda: ["claude-code"])


def _interpolate_env_vars(value: Any) -> Any:
    """Replace ${ENV_VAR} patterns with environment variable values."""
    if isinstance(value, str):
        def replace(match):
            var_name = match.group(1)
            env_value = os.environ.get(var_name)
            if env_value is None:
                logger.warning(f"Environment variable {var_name} not set")
                return ""
            return env_value
        return ENV_VAR_PATTERN.sub(replace, value)
    elif isinstance(value, dict):
        return {k: _interpolate_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_interpolate_env_vars(v) for v in value]
    return value


def _load_provider_config(data: dict) -> ProviderConfig:
    """Load a single provider's config from dict."""
    data = _interpolate_env_vars(data)
    return ProviderConfig(
        api_key=data.get("api_key"),
        model=data.get("model"),
        extra={k: v for k, v in data.items() if k not in ("api_key", "model")},
    )


def load_providers_config(config_dict: dict | None = None) -> ProvidersConfig:
    """Load providers configuration from dict or defaults.

    Args:
        config_dict: Optional config dict (from YAML). If None, uses defaults.

    Returns:
        ProvidersConfig with all settings resolved.
    """
    if config_dict is None:
        config_dict = {}

    providers_section = config_dict.get("providers", {})

    config = ProvidersConfig()

    # Load provider-specific configs
    if "openai" in providers_section:
        config.openai = _load_provider_config(providers_section["openai"])
    if "anthropic" in providers_section:
        config.anthropic = _load_provider_config(providers_section["anthropic"])
    if "google" in providers_section:
        config.google = _load_provider_config(providers_section["google"])
    if "deepgram" in providers_section:
        config.deepgram = _load_provider_config(providers_section["deepgram"])
    if "assemblyai" in providers_section:
        config.assemblyai = _load_provider_config(providers_section["assemblyai"])
    if "voyage" in providers_section:
        config.voyage = _load_provider_config(providers_section["voyage"])

    # Load local configs
    local = providers_section.get("local", {})
    if "whisper_model" in local:
        config.whisper_local_model = local["whisper_model"]
    if "ollama_model" in local:
        config.ollama_model = local["ollama_model"]

    # Load preferences
    prefs = providers_section.get("preferences", {})
    if "transcription" in prefs:
        config.transcription_provider = prefs["transcription"]
    if "vision" in prefs:
        config.vision_provider = prefs["vision"]
    if "video" in prefs:
        config.video_provider = prefs["video"]
    if "reasoning" in prefs:
        config.reasoning_provider = prefs["reasoning"]
    if "embedding" in prefs:
        config.embedding_provider = prefs["embedding"]

    # Load fallbacks
    fallbacks = providers_section.get("fallbacks", {})
    if "transcription" in fallbacks:
        config.transcription_fallbacks = fallbacks["transcription"]
    if "vision" in fallbacks:
        config.vision_fallbacks = fallbacks["vision"]
    if "reasoning" in fallbacks:
        config.reasoning_fallbacks = fallbacks["reasoning"]

    return config


# Singleton for global config
_config: ProvidersConfig | None = None


def get_providers_config() -> ProvidersConfig:
    """Get the global providers configuration.

    Loads from config file on first call, then returns cached.
    """
    global _config
    if _config is None:
        from claudetube.config.loader import get_config, _load_yaml_config

        # Try to load from existing config system
        try:
            base_config = get_config()
            config_path = base_config.cache_dir.parent / "config.yaml"
            if config_path.exists():
                yaml_config = _load_yaml_config(config_path)
                _config = load_providers_config(yaml_config)
            else:
                _config = ProvidersConfig()
        except Exception:
            _config = ProvidersConfig()

    return _config


def clear_config_cache() -> None:
    """Clear cached config (for testing or config reload)."""
    global _config
    _config = None
```

### Gotchas / Things to Keep in Mind

- Environment variable interpolation MUST happen at load time, not at config definition
- API keys should NEVER be logged
- Default preferences point to Claude Code (always available)
- Keep backward compat with existing CLAUDETUBE_* env vars
- Config reload should be possible (for testing)

### Success Criteria

- [ ] `${OPENAI_API_KEY}` in YAML gets resolved from environment
- [ ] Missing env vars produce warnings, not crashes
- [ ] Default config works without any YAML file
- [ ] Provider preferences are respected
- [ ] Fallback chains are configurable
- [ ] Unit tests for env var interpolation

---

# EPIC 2: Core Providers

**Summary:** Implement the essential providers: whisper-local, claude-code, openai, anthropic, google.

**Blocked by:** EPIC 1 (Provider Foundation)
**Blocks:** EPIC 3 (Operations Refactoring)

---

## EPIC-2-T1: Implement WhisperLocalProvider

**Priority:** P0
**Estimate:** M (Medium)
**Blocked by:** EPIC-1-T2, EPIC-1-T3
**Blocks:** EPIC-3-T1

### Requirements

1. Wrap existing `tools/whisper.py` implementation
2. Implement `Transcriber` protocol
3. Return `TranscriptionResult` with proper segments
4. Support model size configuration

### Technical Details

```python
# src/claudetube/providers/whisper_local/__init__.py
from .client import WhisperLocalProvider

# src/claudetube/providers/whisper_local/client.py

from pathlib import Path

from claudetube.providers.base import Provider, Transcriber
from claudetube.providers.capabilities import Capability, ProviderInfo
from claudetube.providers.types import TranscriptionResult, TranscriptionSegment


class WhisperLocalProvider(Provider, Transcriber):
    """Local Whisper transcription using faster-whisper."""

    def __init__(self, model_size: str = "small"):
        self._model_size = model_size
        self._tool = None  # Lazy load

    @property
    def info(self) -> ProviderInfo:
        return ProviderInfo(
            name="whisper-local",
            capabilities=frozenset({Capability.TRANSCRIBE}),
            supports_diarization=False,
            cost_per_minute_audio=0,
        )

    def is_available(self) -> bool:
        """Check if faster-whisper is installed."""
        try:
            import faster_whisper
            return True
        except ImportError:
            return False

    @property
    def _whisper_tool(self):
        """Lazy-load the WhisperTool."""
        if self._tool is None:
            from claudetube.tools.whisper import WhisperTool
            self._tool = WhisperTool(model_size=self._model_size)
        return self._tool

    async def transcribe(
        self,
        audio: Path,
        language: str | None = None,
        **kwargs,
    ) -> TranscriptionResult:
        """Transcribe audio using local Whisper model."""
        # Run sync tool in thread pool
        import asyncio
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self._whisper_tool.transcribe(audio, language=language or "en")
        )

        # Convert to TranscriptionResult
        segments = self._parse_srt_to_segments(result["srt"])

        return TranscriptionResult(
            text=result["txt"],
            segments=segments,
            language=language or "en",
            provider="whisper-local",
        )

    def _parse_srt_to_segments(self, srt_content: str) -> list[TranscriptionSegment]:
        """Parse SRT format to TranscriptionSegment list."""
        import re
        segments = []
        # SRT format: index\nstart --> end\ntext\n\n
        pattern = r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.+?)(?=\n\n|\Z)'
        for match in re.finditer(pattern, srt_content, re.DOTALL):
            start = self._srt_time_to_seconds(match.group(2))
            end = self._srt_time_to_seconds(match.group(3))
            text = match.group(4).strip()
            segments.append(TranscriptionSegment(start=start, end=end, text=text))
        return segments

    @staticmethod
    def _srt_time_to_seconds(time_str: str) -> float:
        """Convert SRT timestamp to seconds."""
        h, m, rest = time_str.split(":")
        s, ms = rest.split(",")
        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000
```

### Gotchas / Things to Keep in Mind

- Existing `WhisperTool` is synchronous - wrap in `run_in_executor`
- SRT parsing needs to handle edge cases (empty lines, special chars)
- Model loading is slow - lazy load and cache
- Default model should match existing behavior ("small" for quality, "tiny" for speed)

### Success Criteria

- [ ] `WhisperLocalProvider()` instantiates without importing faster-whisper
- [ ] `is_available()` returns False if faster-whisper not installed
- [ ] `transcribe()` returns valid `TranscriptionResult`
- [ ] SRT output matches existing `WhisperTool` behavior
- [ ] Async interface works correctly
- [ ] Integration test with real audio file

---

## EPIC-2-T2: Implement ClaudeCodeProvider

**Priority:** P0
**Estimate:** S (Small)
**Blocked by:** EPIC-1-T2
**Blocks:** EPIC-3 (used as fallback)

### Requirements

1. Implement `VisionAnalyzer` and `Reasoner` protocols
2. Always available when running in Claude Code context
3. Returns formatted prompts for host AI to process
4. Structured output support (host Claude supports it)

### Technical Details

```python
# src/claudetube/providers/claude_code/__init__.py
from .client import ClaudeCodeProvider

# src/claudetube/providers/claude_code/client.py

import os
from pathlib import Path

from claudetube.providers.base import Provider, VisionAnalyzer, Reasoner
from claudetube.providers.capabilities import Capability, ProviderInfo


class ClaudeCodeProvider(Provider, VisionAnalyzer, Reasoner):
    """Provider that uses the host Claude instance in Claude Code.

    This provider doesn't make API calls - instead, it formats content
    for the host AI to process in the conversation context.

    Always available when running inside Claude Code, requires no API key.
    """

    @property
    def info(self) -> ProviderInfo:
        return ProviderInfo(
            name="claude-code",
            capabilities=frozenset({Capability.VISION, Capability.REASON}),
            supports_structured_output=True,
            cost_per_1m_input_tokens=0,  # Included with Claude Code
        )

    def is_available(self) -> bool:
        """Check if running inside Claude Code.

        Detection methods:
        1. MCP_SERVER env var (set by claudetube MCP server)
        2. CLAUDE_CODE env var (if Claude Code sets this)
        3. Check if stdin is connected to MCP transport
        """
        # Primary: explicit env var
        if os.environ.get("MCP_SERVER") == "1":
            return True
        if os.environ.get("CLAUDE_CODE") == "1":
            return True

        # Fallback: assume available (user can configure differently)
        # This makes it the safe default
        return True

    async def analyze_images(
        self,
        images: list[Path],
        prompt: str,
        schema: type | None = None,
        **kwargs,
    ) -> str | dict:
        """Format images for host AI to analyze.

        Returns a formatted string with image references that Claude Code
        will render and analyze in the conversation.
        """
        image_refs = []
        for img in images:
            if img.exists():
                image_refs.append(f"[Image: {img}]")
            else:
                image_refs.append(f"[Image not found: {img}]")

        content = "\n".join(image_refs)

        if schema:
            schema_json = schema.model_json_schema() if hasattr(schema, 'model_json_schema') else str(schema)
            content += f"\n\n{prompt}\n\nRespond with JSON matching this schema:\n```json\n{schema_json}\n```"
        else:
            content += f"\n\n{prompt}"

        return content

    async def reason(
        self,
        messages: list[dict],
        schema: type | None = None,
        **kwargs,
    ) -> str | dict:
        """Format messages for host AI to process.

        Returns a formatted prompt that the host Claude will respond to.
        """
        formatted_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                formatted_parts.append(f"[System]: {content}")
            elif role == "assistant":
                formatted_parts.append(f"[Previous response]: {content}")
            else:
                formatted_parts.append(content)

        content = "\n\n".join(formatted_parts)

        if schema:
            schema_json = schema.model_json_schema() if hasattr(schema, 'model_json_schema') else str(schema)
            content += f"\n\nRespond with JSON matching this schema:\n```json\n{schema_json}\n```"

        return content
```

### Gotchas / Things to Keep in Mind

- This provider is UNIQUE - it doesn't make API calls
- It formats content for the host AI to process
- Image paths must be absolute and exist
- `is_available()` should default to True (safe fallback)
- Structured output relies on host Claude following the schema request

### Success Criteria

- [ ] `is_available()` returns True by default
- [ ] `analyze_images()` returns formatted string with image refs
- [ ] `reason()` formats messages correctly
- [ ] Schema requests are formatted clearly
- [ ] Works without any API keys or external dependencies

---

## EPIC-2-T3: Implement OpenAIProvider

**Priority:** P0
**Estimate:** L (Large)
**Blocked by:** EPIC-1-T2, EPIC-1-T3
**Blocks:** EPIC-3

### Requirements

1. Implement `Transcriber`, `VisionAnalyzer`, `Reasoner` protocols
2. Handle Whisper API 25MB file limit with auto-chunking
3. Support structured output via `response_format`
4. Support multiple models (whisper-1, gpt-4o, gpt-4o-mini)

### Technical Details

```python
# src/claudetube/providers/openai/__init__.py
from .client import OpenAIProvider

# src/claudetube/providers/openai/client.py

import base64
import os
from pathlib import Path

from claudetube.providers.base import Provider, Transcriber, VisionAnalyzer, Reasoner
from claudetube.providers.capabilities import Capability, ProviderInfo
from claudetube.providers.types import TranscriptionResult, TranscriptionSegment


class OpenAIProvider(Provider, Transcriber, VisionAnalyzer, Reasoner):
    """OpenAI provider for transcription, vision, and reasoning."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o",
        whisper_model: str = "whisper-1",
    ):
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._model = model
        self._whisper_model = whisper_model
        self._client = None

    @property
    def info(self) -> ProviderInfo:
        return ProviderInfo(
            name="openai",
            capabilities=frozenset({Capability.TRANSCRIBE, Capability.VISION, Capability.REASON}),
            supports_structured_output=True,
            max_audio_size_mb=25,
            max_audio_duration_sec=1500,
            max_images_per_request=10,
            supports_translation=True,
            cost_per_minute_audio=0.006,
            cost_per_1m_input_tokens=2.50,
        )

    def is_available(self) -> bool:
        return self._api_key is not None

    @property
    def client(self):
        """Lazy-load OpenAI client."""
        if self._client is None:
            try:
                import openai
                self._client = openai.AsyncOpenAI(api_key=self._api_key)
            except ImportError:
                raise ImportError("openai package required. Install with: pip install openai")
        return self._client

    async def transcribe(
        self,
        audio: Path,
        language: str | None = None,
        **kwargs,
    ) -> TranscriptionResult:
        """Transcribe audio using Whisper API."""
        from .chunker import chunk_audio_if_needed

        # Handle large files
        chunks = await chunk_audio_if_needed(audio, max_size_mb=25)
        all_segments = []
        full_text_parts = []

        for chunk in chunks:
            with open(chunk.path, "rb") as f:
                response = await self.client.audio.transcriptions.create(
                    file=f,
                    model=self._whisper_model,
                    response_format="verbose_json",
                    language=language,
                )

            # Parse response
            for seg in response.segments or []:
                all_segments.append(TranscriptionSegment(
                    start=seg.start + chunk.offset,
                    end=seg.end + chunk.offset,
                    text=seg.text.strip(),
                ))
            full_text_parts.append(response.text)

        return TranscriptionResult(
            text=" ".join(full_text_parts),
            segments=all_segments,
            language=language or response.language,
            duration=response.duration if hasattr(response, 'duration') else None,
            provider="openai",
        )

    async def analyze_images(
        self,
        images: list[Path],
        prompt: str,
        schema: type | None = None,
        **kwargs,
    ) -> str | dict:
        """Analyze images using GPT-4o vision."""
        content = []

        for img in images[:10]:  # Max 10 images
            b64 = self._encode_image(img)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })

        content.append({"type": "text", "text": prompt})

        kwargs_call = {
            "model": self._model,
            "messages": [{"role": "user", "content": content}],
        }

        if schema:
            kwargs_call["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": schema.model_json_schema()
                }
            }

        response = await self.client.chat.completions.create(**kwargs_call)
        result = response.choices[0].message.content

        if schema:
            return schema.model_validate_json(result)
        return result

    async def reason(
        self,
        messages: list[dict],
        schema: type | None = None,
        **kwargs,
    ) -> str | dict:
        """Generate response using chat completion."""
        kwargs_call = {
            "model": self._model,
            "messages": messages,
        }

        if schema:
            kwargs_call["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": schema.model_json_schema()
                }
            }

        response = await self.client.chat.completions.create(**kwargs_call)
        result = response.choices[0].message.content

        if schema:
            return schema.model_validate_json(result)
        return result

    def _encode_image(self, path: Path) -> str:
        """Encode image to base64."""
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
```

```python
# src/claudetube/providers/openai/chunker.py

import asyncio
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AudioChunk:
    path: Path
    offset: float  # Start time in seconds


async def chunk_audio_if_needed(
    audio_path: Path,
    max_size_mb: float = 25,
) -> list[AudioChunk]:
    """Split audio file if it exceeds size limit.

    Args:
        audio_path: Path to audio file
        max_size_mb: Maximum file size in MB

    Returns:
        List of AudioChunk with paths and time offsets
    """
    file_size_mb = audio_path.stat().st_size / (1024 * 1024)

    if file_size_mb <= max_size_mb:
        return [AudioChunk(path=audio_path, offset=0.0)]

    # Need to chunk - use ffmpeg
    from claudetube.tools.ffmpeg import FFmpegTool

    ffmpeg = FFmpegTool()
    chunks = []
    chunk_duration = 600  # 10 minutes per chunk (should be well under 25MB)

    # Get total duration
    duration = await _get_audio_duration(audio_path)

    chunk_dir = audio_path.parent / "chunks"
    chunk_dir.mkdir(exist_ok=True)

    offset = 0.0
    chunk_num = 0

    while offset < duration:
        chunk_path = chunk_dir / f"chunk_{chunk_num:03d}.mp3"

        # Extract chunk using ffmpeg
        cmd = [
            "ffmpeg", "-y",
            "-i", str(audio_path),
            "-ss", str(offset),
            "-t", str(chunk_duration),
            "-acodec", "libmp3lame",
            "-ab", "64k",
            str(chunk_path)
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await process.wait()

        if chunk_path.exists():
            chunks.append(AudioChunk(path=chunk_path, offset=offset))

        offset += chunk_duration
        chunk_num += 1

    return chunks


async def _get_audio_duration(audio_path: Path) -> float:
    """Get audio duration in seconds using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(audio_path)
    ]

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    stdout, _ = await process.communicate()

    return float(stdout.decode().strip())
```

### Gotchas / Things to Keep in Mind

- Whisper API has 25MB limit - chunking is required for long audio
- Chunk offsets must be added to segment timestamps
- Use `AsyncOpenAI` client for async operations
- `response_format` with `json_schema` requires specific format
- Image encoding should handle different formats (JPEG, PNG)
- Rate limits may apply - consider adding retry logic

### Success Criteria

- [ ] `is_available()` returns False without API key
- [ ] `transcribe()` works for files under 25MB
- [ ] `transcribe()` auto-chunks large files correctly
- [ ] Chunk timestamps are properly offset
- [ ] `analyze_images()` encodes and sends images
- [ ] Structured output works with Pydantic models
- [ ] Integration tests with real API (mock for unit tests)

---

## EPIC-2-T4: Implement AnthropicProvider

**Priority:** P0
**Estimate:** M (Medium)
**Blocked by:** EPIC-1-T2, EPIC-1-T3
**Blocks:** EPIC-3

### Requirements

1. Implement `VisionAnalyzer` and `Reasoner` protocols
2. Support structured output via tool_choice
3. Support multiple models (claude-3-haiku, claude-sonnet, etc.)
4. Migrate existing `_generate_visual_claude()` logic

### Technical Details

```python
# src/claudetube/providers/anthropic/__init__.py
from .client import AnthropicProvider

# src/claudetube/providers/anthropic/client.py

import base64
import os
from pathlib import Path

from claudetube.providers.base import Provider, VisionAnalyzer, Reasoner
from claudetube.providers.capabilities import Capability, ProviderInfo


class AnthropicProvider(Provider, VisionAnalyzer, Reasoner):
    """Anthropic Claude provider for vision and reasoning."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
    ):
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._model = model
        self._client = None

    @property
    def info(self) -> ProviderInfo:
        return ProviderInfo(
            name="anthropic",
            capabilities=frozenset({Capability.VISION, Capability.REASON}),
            supports_structured_output=True,
            max_images_per_request=20,
            max_context_tokens=200_000,
            cost_per_1m_input_tokens=3.00,
        )

    def is_available(self) -> bool:
        return self._api_key is not None

    @property
    def client(self):
        """Lazy-load Anthropic client."""
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.AsyncAnthropic(api_key=self._api_key)
            except ImportError:
                raise ImportError("anthropic package required. Install with: pip install anthropic")
        return self._client

    async def analyze_images(
        self,
        images: list[Path],
        prompt: str,
        schema: type | None = None,
        **kwargs,
    ) -> str | dict:
        """Analyze images using Claude vision."""
        content = []

        for img in images[:20]:  # Max 20 images
            media_type = self._get_media_type(img)
            b64 = self._encode_image(img)
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": b64,
                }
            })

        content.append({"type": "text", "text": prompt})

        if schema:
            # Use tool_choice for structured output
            return await self._call_with_tool(content, schema)
        else:
            response = await self.client.messages.create(
                model=self._model,
                max_tokens=4096,
                messages=[{"role": "user", "content": content}],
            )
            return response.content[0].text

    async def reason(
        self,
        messages: list[dict],
        schema: type | None = None,
        **kwargs,
    ) -> str | dict:
        """Generate response using Claude messages API."""
        if schema:
            # Convert to content format for tool call
            content = []
            for msg in messages:
                if msg["role"] == "user":
                    content.append({"type": "text", "text": msg["content"]})
            return await self._call_with_tool(content, schema)

        response = await self.client.messages.create(
            model=self._model,
            max_tokens=4096,
            messages=messages,
        )
        return response.content[0].text

    async def _call_with_tool(self, content: list, schema: type) -> dict:
        """Use tool_choice to force structured output."""
        tool_name = "structured_response"
        tool_schema = schema.model_json_schema()

        response = await self.client.messages.create(
            model=self._model,
            max_tokens=4096,
            messages=[{"role": "user", "content": content}],
            tools=[{
                "name": tool_name,
                "description": "Return structured response",
                "input_schema": tool_schema,
            }],
            tool_choice={"type": "tool", "name": tool_name},
        )

        # Extract tool input
        for block in response.content:
            if block.type == "tool_use" and block.name == tool_name:
                return schema.model_validate(block.input)

        raise ValueError("No tool response found")

    def _encode_image(self, path: Path) -> str:
        """Encode image to base64."""
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _get_media_type(self, path: Path) -> str:
        """Get media type from file extension."""
        ext = path.suffix.lower()
        return {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }.get(ext, "image/jpeg")
```

### Gotchas / Things to Keep in Mind

- Anthropic uses `tool_choice` for structured output, not `response_format`
- Image format includes `media_type` field
- Max 20 images per request
- Model names are different format (claude-sonnet-4-20250514)
- Tool response is in `block.input`, not `block.content`

### Success Criteria

- [ ] `is_available()` returns False without API key
- [ ] `analyze_images()` handles multiple images
- [ ] Structured output via tool_choice works
- [ ] Media types detected correctly
- [ ] Migration from existing `_generate_visual_claude()` is clean
- [ ] Integration tests with real API

---

## EPIC-2-T5: Implement GoogleProvider (Gemini)

**Priority:** P0
**Estimate:** L (Large)
**Blocked by:** EPIC-1-T2, EPIC-1-T3
**Blocks:** EPIC-3

### Requirements

1. Implement `VisionAnalyzer`, `VideoAnalyzer`, `Reasoner` protocols
2. Support native video upload via File API
3. Support structured output via `response_schema`
4. Handle large file uploads (up to 2GB)

### Technical Details

```python
# src/claudetube/providers/google/__init__.py
from .client import GoogleProvider

# src/claudetube/providers/google/client.py

import os
from pathlib import Path

from claudetube.providers.base import Provider, VisionAnalyzer, VideoAnalyzer, Reasoner
from claudetube.providers.capabilities import Capability, ProviderInfo


class GoogleProvider(Provider, VisionAnalyzer, VideoAnalyzer, Reasoner):
    """Google Gemini provider with native video support."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-2.0-flash",
    ):
        self._api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self._model = model
        self._genai = None

    @property
    def info(self) -> ProviderInfo:
        return ProviderInfo(
            name="google",
            capabilities=frozenset({Capability.VISION, Capability.VIDEO, Capability.REASON}),
            supports_structured_output=True,
            max_video_duration_sec=7200,  # 2 hours
            max_video_size_mb=2000,
            max_context_tokens=2_000_000,
            video_tokens_per_second=300,
            cost_per_1m_input_tokens=0.10,
        )

    def is_available(self) -> bool:
        return self._api_key is not None

    @property
    def genai(self):
        """Lazy-load Google GenAI client."""
        if self._genai is None:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self._api_key)
                self._genai = genai
            except ImportError:
                raise ImportError(
                    "google-generativeai package required. "
                    "Install with: pip install google-generativeai"
                )
        return self._genai

    async def analyze_images(
        self,
        images: list[Path],
        prompt: str,
        schema: type | None = None,
        **kwargs,
    ) -> str | dict:
        """Analyze images using Gemini vision."""
        from PIL import Image

        content = []
        for img in images:
            pil_img = Image.open(img)
            content.append(pil_img)
        content.append(prompt)

        model = self._get_model(schema)
        response = await self._generate_async(model, content)

        if schema:
            return schema.model_validate_json(response.text)
        return response.text

    async def analyze_video(
        self,
        video: Path,
        prompt: str,
        schema: type | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        **kwargs,
    ) -> str | dict:
        """Analyze video using Gemini native video understanding."""
        # Upload video to File API
        video_file = await self._upload_video(video)

        content = [video_file, prompt]

        if start_time is not None or end_time is not None:
            time_spec = ""
            if start_time is not None:
                time_spec += f"Starting from {self._format_time(start_time)}. "
            if end_time is not None:
                time_spec += f"Until {self._format_time(end_time)}. "
            content = [video_file, time_spec + prompt]

        model = self._get_model(schema)
        response = await self._generate_async(model, content)

        if schema:
            return schema.model_validate_json(response.text)
        return response.text

    async def reason(
        self,
        messages: list[dict],
        schema: type | None = None,
        **kwargs,
    ) -> str | dict:
        """Generate response using Gemini chat."""
        # Convert messages to Gemini format
        contents = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            contents.append({"role": role, "parts": [msg["content"]]})

        model = self._get_model(schema)
        chat = model.start_chat(history=contents[:-1])
        response = await self._send_message_async(chat, contents[-1]["parts"][0])

        if schema:
            return schema.model_validate_json(response.text)
        return response.text

    def _get_model(self, schema: type | None = None):
        """Get Gemini model with optional structured output config."""
        if schema:
            config = self.genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=schema,
            )
            return self.genai.GenerativeModel(self._model, generation_config=config)
        return self.genai.GenerativeModel(self._model)

    async def _upload_video(self, video_path: Path):
        """Upload video to Gemini File API."""
        import asyncio

        # File upload is sync in current SDK
        loop = asyncio.get_event_loop()
        video_file = await loop.run_in_executor(
            None,
            lambda: self.genai.upload_file(str(video_path))
        )

        # Wait for processing
        while video_file.state.name == "PROCESSING":
            await asyncio.sleep(2)
            video_file = await loop.run_in_executor(
                None,
                lambda: self.genai.get_file(video_file.name)
            )

        if video_file.state.name != "ACTIVE":
            raise RuntimeError(f"Video processing failed: {video_file.state.name}")

        return video_file

    async def _generate_async(self, model, content):
        """Run synchronous generate in executor."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: model.generate_content(content)
        )

    async def _send_message_async(self, chat, message):
        """Run synchronous send_message in executor."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: chat.send_message(message)
        )

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds as MM:SS."""
        m, s = divmod(int(seconds), 60)
        return f"{m:02d}:{s:02d}"
```

### Gotchas / Things to Keep in Mind

- Gemini SDK is mostly synchronous - wrap in `run_in_executor`
- Video upload requires waiting for ACTIVE state
- Large videos can take time to process on File API
- Gemini uses `response_schema` with Pydantic models directly
- Token cost is ~300 tokens/second of video
- Video file references must be cleaned up after use (or they persist)

### Success Criteria

- [ ] `is_available()` returns False without API key
- [ ] `analyze_images()` works with PIL images
- [ ] `analyze_video()` uploads and processes video
- [ ] Video processing waits for ACTIVE state
- [ ] Structured output works with Pydantic models
- [ ] Time-based video queries work (start_time, end_time)
- [ ] Integration test with real video file

---

# EPIC 3: Operations Layer Refactoring

**Summary:** Refactor operations to accept providers via dependency injection.

**Blocked by:** EPIC 2 (Core Providers)
**Blocks:** EPIC 4 (Analysis Layer)

---

## EPIC-3-T1: Refactor TranscribeOperation

**Priority:** P0
**Estimate:** M (Medium)
**Blocked by:** EPIC-2-T1, EPIC-2-T3
**Blocks:** None

### Requirements

1. Convert `transcribe_video()` function to `TranscribeOperation` class
2. Accept `Transcriber` via constructor
3. Maintain backward compatibility with existing function signature
4. Update MCP tool to use new operation

### Technical Details

```python
# src/claudetube/operations/transcribe.py

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from claudetube.cache.manager import CacheManager
from claudetube.config.loader import get_cache_dir
from claudetube.providers.base import Transcriber
from claudetube.providers.types import TranscriptionResult

if TYPE_CHECKING:
    pass


class TranscribeOperation:
    """Transcribe video audio using configurable provider."""

    def __init__(self, transcriber: Transcriber):
        self.transcriber = transcriber

    async def execute(
        self,
        video_id: str,
        audio_path: Path,
        language: str | None = None,
        cache_dir: Path | None = None,
    ) -> dict:
        """Execute transcription and save results.

        Args:
            video_id: Video identifier
            audio_path: Path to audio file
            language: Optional language code
            cache_dir: Optional cache directory

        Returns:
            Dict with success status, paths, and metadata
        """
        cache = CacheManager(cache_dir or get_cache_dir())
        srt_path, txt_path = cache.get_transcript_paths(video_id)

        # Run transcription
        result = await self.transcriber.transcribe(audio_path, language=language)

        # Save results
        srt_path.write_text(result.to_srt())
        txt_path.write_text(result.text)

        # Update state
        state = cache.get_state(video_id)
        if state:
            state.transcript_complete = True
            state.transcript_source = result.provider
            cache.save_state(video_id, state)

        return {
            "success": True,
            "video_id": video_id,
            "transcript_srt": str(srt_path),
            "transcript_txt": str(txt_path),
            "source": result.provider,
            "segments": len(result.segments),
            "duration": result.duration,
        }


# Backward-compatible function wrapper
async def transcribe_video(
    video_id_or_url: str,
    whisper_model: str = "small",
    force: bool = False,
    output_base: Path | None = None,
    transcriber: Transcriber | None = None,
) -> dict:
    """Transcribe a video's audio.

    This is a backward-compatible wrapper around TranscribeOperation.
    """
    from claudetube.parsing.utils import extract_video_id
    from claudetube.providers import get_provider

    cache = CacheManager(output_base or get_cache_dir())
    video_id = extract_video_id(video_id_or_url)

    # Check cache
    srt_path, txt_path = cache.get_transcript_paths(video_id)
    if not force and srt_path.exists() and txt_path.exists():
        return {
            "success": True,
            "video_id": video_id,
            "transcript_srt": str(srt_path),
            "transcript_txt": str(txt_path),
            "source": "cached",
        }

    # Get transcriber
    if transcriber is None:
        transcriber = get_provider("whisper-local", model_size=whisper_model)

    # Ensure audio exists
    audio_path = cache.get_audio_path(video_id)
    if not audio_path.exists():
        # Download audio...
        from claudetube.operations.download import download_audio
        state = cache.get_state(video_id)
        url = state.url if state else video_id_or_url
        download_audio(url, audio_path)

    # Execute operation
    op = TranscribeOperation(transcriber)
    return await op.execute(video_id, audio_path, cache_dir=output_base)
```

### Gotchas / Things to Keep in Mind

- Must maintain backward compatibility for existing callers
- Cache checking happens BEFORE creating operation
- Audio download is separate from transcription
- State updates should happen in operation, not caller

### Success Criteria

- [ ] `TranscribeOperation` class works with any `Transcriber`
- [ ] Backward-compatible `transcribe_video()` function works
- [ ] Cache is checked before transcription
- [ ] Results saved in correct format
- [ ] State is updated correctly
- [ ] MCP tool continues to work

---

## EPIC-3-T2: Refactor VisualTranscriptOperation

**Priority:** P0
**Estimate:** M (Medium)
**Blocked by:** EPIC-2-T2, EPIC-2-T4
**Blocks:** EPIC-3-T4

### Requirements

1. Extract visual transcript generation to `VisualTranscriptOperation`
2. Accept `VisionAnalyzer` via constructor
3. Support structured output for consistent results
4. Migrate from existing `_generate_visual_claude()` implementation

### Technical Details

See Operations Layer Refactoring section in main PRD for code example.

### Success Criteria

- [ ] `VisualTranscriptOperation` accepts any `VisionAnalyzer`
- [ ] Structured output produces consistent `VisualDescription`
- [ ] Backward compatibility maintained
- [ ] Scene-level caching works correctly
- [ ] MCP tool continues to work

---

## EPIC-3-T3: Refactor PersonTrackingOperation

**Priority:** P1
**Estimate:** M (Medium)
**Blocked by:** EPIC-2-T5 (Google for video), EPIC-3-T2
**Blocks:** None

### Requirements

1. Accept `VisionAnalyzer` and optional `VideoAnalyzer`
2. Use Gemini video for cross-scene tracking when available
3. Fall back to frame-by-frame analysis
4. Maintain face_recognition fallback

### Success Criteria

- [ ] Gemini video tracks people across scenes in one call
- [ ] Falls back to vision when video unavailable
- [ ] Falls back to face_recognition when vision unavailable
- [ ] Results match existing format

---

## EPIC-3-T4: Refactor EntityExtractionOperation

**Priority:** P0
**Estimate:** L (Large)
**Blocked by:** EPIC-2 (all core providers), EPIC-3-T2
**Blocks:** None

### Requirements

1. Accept `VisionAnalyzer`, `VideoAnalyzer`, `Reasoner`
2. Use best available provider for each entity type
3. Generate `visual.json` from entities (entities-first architecture)
4. Support structured output for all extraction

### Success Criteria

- [ ] Visual entities extracted via vision/video
- [ ] Semantic concepts extracted via reasoner
- [ ] `visual.json` generated from entities
- [ ] Backward compatibility with existing cache format

---

## EPIC-3-T5: Create OperationFactory

**Priority:** P0
**Estimate:** S (Small)
**Blocked by:** EPIC-3-T1 through EPIC-3-T4
**Blocks:** EPIC-6

### Requirements

1. Create factory that constructs operations with configured providers
2. Use router to select appropriate providers
3. Single entry point for MCP tools

### Technical Details

See Operations Layer Refactoring section in main PRD for code example.

### Success Criteria

- [ ] `OperationFactory` creates all operations
- [ ] Uses router for provider selection
- [ ] MCP tools use factory
- [ ] Easy to test with mock providers

---

# EPIC 4: Analysis Layer Migration

**Summary:** Migrate analysis modules to use provider architecture.

**Blocked by:** EPIC 3 (Operations Refactoring)
**Blocks:** EPIC 6 (Polish)

---

## EPIC-4-T1: Migrate Embeddings to Provider

**Priority:** P0
**Estimate:** M (Medium)
**Blocked by:** EPIC-1 (Foundation)
**Blocks:** EPIC-4-T2

### Requirements

1. Create `VoyageProvider` implementing `Embedder` protocol
2. Create local embeddings provider as fallback
3. Refactor `analysis/embeddings.py` to use providers
4. Maintain existing dispatch pattern compatibility

### Success Criteria

- [ ] `VoyageProvider` implements `Embedder`
- [ ] Local fallback works without API key
- [ ] Existing `embed_scene()` API unchanged
- [ ] Performance matches existing implementation

---

## EPIC-4-T2: Enhance Search with Providers

**Priority:** P1
**Estimate:** M (Medium)
**Blocked by:** EPIC-4-T1
**Blocks:** None

### Requirements

1. Add optional `Reasoner` for query expansion
2. Use `Embedder` protocol for embedding queries
3. Maintain text-first search strategy

### Success Criteria

- [ ] Query expansion with LLM improves results
- [ ] Falls back gracefully without LLM
- [ ] Text search remains fast

---

## EPIC-4-T3: Enhance OCR with Vision Provider

**Priority:** P2
**Estimate:** M (Medium)
**Blocked by:** EPIC-2 (Vision providers)
**Blocks:** None

### Requirements

1. Add optional `VisionAnalyzer` for OCR
2. Keep EasyOCR as fallback
3. Vision OCR better for code, handwriting

### Success Criteria

- [ ] Vision OCR optional enhancement
- [ ] EasyOCR always available as fallback
- [ ] Detects when vision OCR is better

---

# EPIC 5: Specialist Providers

**Summary:** Implement specialized providers for advanced features.

**Blocked by:** EPIC 2 (Core Providers)
**Blocks:** Nothing (parallel with EPIC 4)

---

## EPIC-5-T1: Implement DeepgramProvider

**Priority:** P1
**Estimate:** M (Medium)
**Blocked by:** EPIC-1
**Blocks:** None

### Requirements

1. Implement `Transcriber` protocol
2. Support speaker diarization
3. Support streaming (future)

### Success Criteria

- [ ] Transcription works
- [ ] Diarization populates speaker field
- [ ] Faster than local Whisper

---

## EPIC-5-T2: Implement AssemblyAIProvider

**Priority:** P2
**Estimate:** M (Medium)
**Blocked by:** EPIC-1
**Blocks:** None

### Requirements

1. Implement `Transcriber` protocol
2. Support auto-chapters
3. Support sentiment analysis (bonus)

### Success Criteria

- [ ] Transcription works
- [ ] Auto-chapters extracted

---

## EPIC-5-T3: Implement OllamaProvider

**Priority:** P1
**Estimate:** M (Medium)
**Blocked by:** EPIC-1
**Blocks:** None

### Requirements

1. Implement `VisionAnalyzer` and `Reasoner` protocols
2. Support LLaVA and Moondream models
3. Fully offline operation

### Success Criteria

- [ ] Works without internet
- [ ] Handles single image (LLaVA limitation)
- [ ] Reasonable performance on CPU

---

# EPIC 6: Router, Config & Polish

**Summary:** Implement smart routing, finalize config, and polish.

**Blocked by:** EPIC 3, EPIC 4
**Blocks:** Release

---

## EPIC-6-T1: Implement ProviderRouter

**Priority:** P0
**Estimate:** L (Large)
**Blocked by:** EPIC-3-T5
**Blocks:** EPIC-6-T3

### Requirements

1. Smart provider selection based on capabilities
2. Fallback chains with error handling
3. Claude Code as ultimate fallback for vision/reasoning
4. Capability-based routing for optimal provider

### Technical Details

See Router section in main PRD.

### Success Criteria

- [ ] Selects best provider for capability
- [ ] Falls back on 4xx/5xx errors
- [ ] Claude Code always catches vision/reasoning
- [ ] Clear logging of provider selection

---

## EPIC-6-T2: Finalize Configuration System

**Priority:** P0
**Estimate:** M (Medium)
**Blocked by:** EPIC-1-T6
**Blocks:** EPIC-6-T3

### Requirements

1. Complete YAML schema for providers
2. Validation with helpful error messages
3. Documentation for all config options
4. Migration guide from env vars

### Success Criteria

- [ ] All options documented
- [ ] Invalid config produces clear errors
- [ ] Env var fallbacks work
- [ ] Example configs provided

---

## EPIC-6-T3: Update MCP Tools

**Priority:** P0
**Estimate:** M (Medium)
**Blocked by:** EPIC-6-T1, EPIC-6-T2
**Blocks:** Release

### Requirements

1. Update all MCP tools to use OperationFactory
2. Add `list_providers` tool
3. Add provider override parameters where appropriate
4. Update tool documentation

### Success Criteria

- [ ] All tools use factory
- [ ] `list_providers` shows available providers
- [ ] Provider overrides work
- [ ] Tool docs updated

---

## EPIC-6-T4: Testing & Documentation

**Priority:** P0
**Estimate:** L (Large)
**Blocked by:** All previous tickets
**Blocks:** Release

### Requirements

1. Unit tests for all providers (with mocks)
2. Integration tests for core flows
3. Update CLAUDE.md with provider info
4. Update documentation/guides/configuration.md
5. Add examples for common configurations

### Success Criteria

- [ ] >80% test coverage on providers
- [ ] Integration tests pass
- [ ] Documentation complete
- [ ] Example configs work

---

# Dependency Graph Summary

```
EPIC-1-T1 (Structure)
    │
    ├──► EPIC-1-T2 (Protocols) ──┬──► EPIC-2-T1 (Whisper) ──► EPIC-3-T1 (Transcribe Op)
    │                            │
    ├──► EPIC-1-T3 (Types) ──────┼──► EPIC-2-T2 (Claude Code) ──► EPIC-3-T2 (Visual Op)
    │                            │
    ├──► EPIC-1-T4 (Capabilities)┼──► EPIC-2-T3 (OpenAI) ──────► EPIC-3-T4 (Entity Op)
    │         │                  │
    │         ▼                  ├──► EPIC-2-T4 (Anthropic) ──► EPIC-3-T3 (Person Op)
    │    EPIC-1-T5 (Registry)    │
    │         │                  └──► EPIC-2-T5 (Google) ──────► EPIC-3-T5 (Factory)
    │         ▼                                                        │
    └──► EPIC-1-T6 (Config) ──────────────────────────────────────────┘
                                                                       │
         EPIC-4-T1 (Embeddings) ◄──────────────────────────────────────┤
              │                                                        │
         EPIC-4-T2 (Search)                                            │
                                                                       │
         EPIC-5-T1 (Deepgram) ◄────────────────────────────────────────┤
         EPIC-5-T2 (AssemblyAI)                                        │
         EPIC-5-T3 (Ollama)                                            │
                                                                       │
                                                                       ▼
                                                               EPIC-6-T1 (Router)
                                                                       │
                                                               EPIC-6-T2 (Config Final)
                                                                       │
                                                               EPIC-6-T3 (MCP Tools)
                                                                       │
                                                               EPIC-6-T4 (Testing)
                                                                       │
                                                                       ▼
                                                                   RELEASE
```

---

# Effort Estimation Summary

| Epic | Tickets | Total Estimate |
|------|---------|----------------|
| EPIC 1: Foundation | 6 | ~2 weeks |
| EPIC 2: Core Providers | 5 | ~3 weeks |
| EPIC 3: Operations | 5 | ~2 weeks |
| EPIC 4: Analysis | 3 | ~1.5 weeks |
| EPIC 5: Specialists | 3 | ~1.5 weeks |
| EPIC 6: Polish | 4 | ~2 weeks |
| **Total** | **26** | **~12 weeks** |

**Critical Path:** EPIC 1 → EPIC 2 → EPIC 3 → EPIC 6

EPIC 4 and EPIC 5 can be parallelized with EPIC 3.
