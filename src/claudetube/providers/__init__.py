"""
claudetube.providers - AI provider abstraction layer.

This package provides a unified interface for multiple AI providers
(OpenAI, Anthropic, Google, local models) for video analysis tasks.

Public API:
    get_provider(name: str) -> Provider
        Get a configured provider instance by name.

    list_available() -> list[str]
        List all available (configured) provider names.

Base Classes:
    Provider: Abstract base class for all AI providers.

Protocols:
    Transcriber: Protocol for audio transcription providers.
    VisionAnalyzer: Protocol for image analysis providers.
    VideoAnalyzer: Protocol for native video analysis (Gemini).
    Reasoner: Protocol for text reasoning/chat completion.
    Embedder: Protocol for embedding generation.

Example:
    >>> from claudetube.providers import get_provider, list_available
    >>> from claudetube.providers import Transcriber, VisionAnalyzer
    >>> providers = list_available()
    >>> provider = get_provider("openai")
    >>> if isinstance(provider, Transcriber):
    ...     result = await provider.transcribe(audio_path)
"""

from __future__ import annotations

from typing import Any

# Import base classes and protocols for public API
from claudetube.providers.base import (
    Embedder,
    Provider,
    Reasoner,
    Transcriber,
    VideoAnalyzer,
    VisionAnalyzer,
)

# Import capability types
from claudetube.providers.capabilities import (
    PROVIDER_INFO,
    Capability,
    ProviderInfo,
)

# Import result types
from claudetube.providers.types import (
    EntityExtractionResult,
    SemanticConcept,
    TranscriptionResult,
    TranscriptionSegment,
    VisualDescription,
    VisualEntity,
    get_entity_extraction_result_model,
    get_semantic_concept_model,
    get_visual_description_model,
    get_visual_entity_model,
)

# Public API - implementations added in future tickets
__all__ = [
    # Provider access
    "get_provider",
    "list_available",
    # Base classes
    "Provider",
    # Protocols
    "Transcriber",
    "VisionAnalyzer",
    "VideoAnalyzer",
    "Reasoner",
    "Embedder",
    # Capability types
    "Capability",
    "ProviderInfo",
    "PROVIDER_INFO",
    # Result types
    "TranscriptionSegment",
    "TranscriptionResult",
    # Pydantic model accessors
    "get_visual_entity_model",
    "get_semantic_concept_model",
    "get_entity_extraction_result_model",
    "get_visual_description_model",
    # Lazy-loading model wrappers
    "VisualEntity",
    "SemanticConcept",
    "EntityExtractionResult",
    "VisualDescription",
]


def get_provider(name: str, **kwargs) -> Provider | Any:
    """
    Get a configured provider instance by name.

    Args:
        name: Provider name (e.g., "openai", "anthropic", "gemini", "whisper-local")
        **kwargs: Provider-specific configuration options.

    Returns:
        Configured provider instance.

    Raises:
        ValueError: If provider name is unknown.
        ImportError: If provider module fails to import.

    Example:
        >>> provider = get_provider("whisper-local", model_size="small")
        >>> result = await provider.transcribe(audio_path)
    """
    from claudetube.providers.registry import get_provider as _get_provider

    return _get_provider(name, **kwargs)


def list_available() -> list[str]:
    """
    List all available (configured) provider names.

    Returns:
        List of provider names that are configured and ready to use.
        Only includes providers where is_available() returns True.

    Example:
        >>> available = list_available()
        >>> print(available)  # ["whisper-local", "claude-code"]
    """
    from claudetube.providers.registry import list_available as _list_available

    return _list_available()
