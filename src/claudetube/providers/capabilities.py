"""
claudetube.providers.capabilities - Capability definitions and limits.

This module defines what capabilities providers can offer and their
operational limits (rate limits, token limits, supported modalities).

Classes:
    Capability: Enum of provider capabilities (TRANSCRIBE, VISION, etc.).
    ProviderInfo: Immutable metadata about a provider's capabilities and limits.

Example:
    >>> from claudetube.providers.capabilities import Capability, ProviderInfo
    >>> info = ProviderInfo(
    ...     name="openai",
    ...     capabilities=frozenset({Capability.TRANSCRIBE, Capability.VISION}),
    ... )
    >>> info.can(Capability.TRANSCRIBE)
    True
    >>> info.can(Capability.VIDEO)
    False
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class Capability(Enum):
    """AI capabilities that providers can offer.

    Each capability represents a specific type of AI task that a provider
    can perform. Providers may support one or more capabilities.

    Attributes:
        TRANSCRIBE: Audio to text conversion with timestamps.
        VISION: Image analysis and understanding.
        VIDEO: Native video analysis without frame extraction (Gemini).
        REASON: Text generation and chat completion.
        EMBED: Content to embedding vector conversion.

    Example:
        >>> from claudetube.providers.capabilities import Capability
        >>> caps = frozenset({Capability.TRANSCRIBE, Capability.VISION})
        >>> Capability.TRANSCRIBE in caps
        True
    """

    TRANSCRIBE = auto()  # Audio -> text
    VISION = auto()  # Image -> text
    VIDEO = auto()  # Native video -> text (Gemini)
    REASON = auto()  # Text -> text (chat/completion)
    EMBED = auto()  # Content -> vector


@dataclass(frozen=True)
class ProviderInfo:
    """Immutable provider metadata and capabilities.

    This dataclass contains all metadata about a provider including its
    capabilities, feature flags, operational limits, and cost estimates.
    The frozen=True ensures immutability for safe sharing across threads.

    Attributes:
        name: Provider identifier (e.g., "openai", "anthropic").
        capabilities: Frozenset of Capability enums this provider supports.
        supports_structured_output: Whether provider can return JSON matching a schema.
        supports_streaming: Whether provider supports streaming responses.
        max_audio_size_mb: Maximum audio file size in MB (None = unlimited).
        max_audio_duration_sec: Maximum audio duration in seconds (None = unlimited).
        supports_diarization: Whether transcription includes speaker identification.
        supports_translation: Whether transcription can translate to English.
        max_images_per_request: Maximum images in one request (None = unlimited).
        max_image_size_mb: Maximum image file size in MB (None = unlimited).
        max_video_duration_sec: Maximum video duration in seconds (None = unlimited).
        max_video_size_mb: Maximum video file size in MB (None = unlimited).
        video_tokens_per_second: Estimated tokens per second of video content.
        max_context_tokens: Maximum context window size (None = unlimited).
        cost_per_1m_input_tokens: Cost in USD per million input tokens (None = unknown).
        cost_per_1m_output_tokens: Cost in USD per million output tokens (None = unknown).
        cost_per_minute_audio: Cost in USD per minute of audio (None = unknown).

    Example:
        >>> info = ProviderInfo(
        ...     name="openai",
        ...     capabilities=frozenset({Capability.TRANSCRIBE, Capability.VISION}),
        ...     supports_structured_output=True,
        ...     max_audio_size_mb=25,
        ... )
        >>> info.can(Capability.TRANSCRIBE)
        True
        >>> info.can_all(Capability.TRANSCRIBE, Capability.VISION)
        True
        >>> info.can(Capability.VIDEO)
        False
    """

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
        """Check if provider has a specific capability.

        Args:
            capability: The capability to check for.

        Returns:
            True if the provider supports this capability.

        Example:
            >>> info.can(Capability.TRANSCRIBE)
            True
        """
        return capability in self.capabilities

    def can_all(self, *capabilities: Capability) -> bool:
        """Check if provider has ALL specified capabilities.

        Args:
            *capabilities: One or more capabilities to check for.

        Returns:
            True if the provider supports all specified capabilities.

        Example:
            >>> info.can_all(Capability.VISION, Capability.REASON)
            True
        """
        return all(c in self.capabilities for c in capabilities)

    def can_any(self, *capabilities: Capability) -> bool:
        """Check if provider has ANY of specified capabilities.

        Args:
            *capabilities: One or more capabilities to check for.

        Returns:
            True if the provider supports at least one of the capabilities.

        Example:
            >>> info.can_any(Capability.VIDEO, Capability.VISION)
            True
        """
        return any(c in self.capabilities for c in capabilities)


# Pre-defined provider info (can be overridden by config)
# These represent the default capabilities and limits for each provider.

PROVIDER_INFO: dict[str, ProviderInfo] = {
    "whisper-local": ProviderInfo(
        name="whisper-local",
        capabilities=frozenset({Capability.TRANSCRIBE}),
        supports_diarization=False,
        supports_translation=True,  # faster-whisper supports translation
        cost_per_minute_audio=0,  # Free (local)
    ),
    "openai": ProviderInfo(
        name="openai",
        capabilities=frozenset({Capability.TRANSCRIBE, Capability.VISION, Capability.REASON}),
        supports_structured_output=True,
        supports_streaming=True,
        max_audio_size_mb=25,
        max_audio_duration_sec=1500,  # ~25 minutes per request
        max_images_per_request=10,
        supports_translation=True,
        max_context_tokens=128_000,  # GPT-4o
        cost_per_minute_audio=0.006,
        cost_per_1m_input_tokens=2.50,
        cost_per_1m_output_tokens=10.00,
    ),
    "anthropic": ProviderInfo(
        name="anthropic",
        capabilities=frozenset({Capability.VISION, Capability.REASON}),
        supports_structured_output=True,
        supports_streaming=True,
        max_images_per_request=20,
        max_image_size_mb=5,  # 5MB per image
        max_context_tokens=200_000,
        cost_per_1m_input_tokens=3.00,
        cost_per_1m_output_tokens=15.00,
    ),
    "google": ProviderInfo(
        name="google",
        capabilities=frozenset({Capability.VISION, Capability.VIDEO, Capability.REASON}),
        supports_structured_output=True,
        supports_streaming=True,
        max_video_duration_sec=7200,  # 2 hours
        max_video_size_mb=2000,  # 2GB
        video_tokens_per_second=300,  # ~300 tokens per second of video
        max_context_tokens=2_000_000,  # Gemini 2.0 Flash
        cost_per_1m_input_tokens=0.10,  # Gemini 2.0 Flash
        cost_per_1m_output_tokens=0.40,
    ),
    "deepgram": ProviderInfo(
        name="deepgram",
        capabilities=frozenset({Capability.TRANSCRIBE}),
        supports_diarization=True,
        supports_streaming=True,
        supports_translation=False,  # Deepgram transcribes in original language
        cost_per_minute_audio=0.0043,
    ),
    "assemblyai": ProviderInfo(
        name="assemblyai",
        capabilities=frozenset({Capability.TRANSCRIBE}),
        supports_diarization=True,
        supports_streaming=True,
        supports_translation=False,
        cost_per_minute_audio=0.006,  # Standard model
    ),
    "claude-code": ProviderInfo(
        name="claude-code",
        capabilities=frozenset({Capability.VISION, Capability.REASON}),
        supports_structured_output=True,
        supports_streaming=False,  # Operates through MCP, not streaming
        max_images_per_request=20,  # Same as Anthropic
        cost_per_1m_input_tokens=0,  # Included with Claude Code subscription
        cost_per_1m_output_tokens=0,
    ),
    "ollama": ProviderInfo(
        name="ollama",
        capabilities=frozenset({Capability.VISION, Capability.REASON}),
        supports_structured_output=False,  # Limited JSON mode support
        supports_streaming=True,
        max_images_per_request=1,  # LLaVA only handles single image
        cost_per_1m_input_tokens=0,  # Local, free
        cost_per_1m_output_tokens=0,
    ),
    "voyage": ProviderInfo(
        name="voyage",
        capabilities=frozenset({Capability.EMBED}),
        supports_structured_output=False,
        cost_per_1m_input_tokens=0.06,  # voyage-3 pricing
    ),
}


__all__ = [
    # Capability definitions
    "Capability",
    # Provider info
    "ProviderInfo",
    # Pre-defined info
    "PROVIDER_INFO",
]
